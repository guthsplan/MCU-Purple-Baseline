from __future__ import annotations

import base64
import json
import logging
import math
import time

from uuid import uuid4
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TextPart
from a2a.utils import new_agent_text_message

from src.server.session_manager import SessionManager
from src.protocol.models import InitPayload, ObservationPayload
from src.agent.registry import build_agent
from src.agent.base import AgentState

logger = logging.getLogger("purple.executor")
logger.setLevel(logging.INFO)


class Executor(AgentExecutor):
    """
    Purple agent executor for MCU benchmark.

    Responsibilities:
      - Parse A2A Message(TextPart JSON) into Init/Obs payloads
      - Maintain session metadata via SessionManager (context_id keyed)
      - Maintain per-context policy recurrent state (AgentState) for Rocket-1
      - Always respond via TaskUpdater.complete() with a JSON string:
          - ack: {"type":"ack","success":...,"message":...}
          - action: {"type":"action","buttons":[...],"camera":[...]}
    """

    def __init__(
        self, 
        sessions: SessionManager, 
        agent_name: str,
        *,
        action_buttons_threshold: float = 0.5,
        state_ttl_seconds: Optional[int] = 60 * 60,
        decode_expect_rgb: bool = True,) -> None:

        self.sessions = sessions
        self.agent_name = agent_name
        self._buttons_threshold = float(action_buttons_threshold)
        self._state_ttl_seconds = state_ttl_seconds
        # Currently all agents expect RGB images.
        # This flag is reserved for future extensions.  
        self._decode_expect_rgb = bool(decode_expect_rgb)
        self.agent = build_agent(agent_name, device=None)
        # per-context recurrent state
        self.agent_states: dict[str, AgentState] = {}
        self._agent_state_touched_at: dict[str, float] = {}

    # ---------------- A2A helpers ----------------

    def _get_task_id(self, context: RequestContext) -> Optional[str]:
        """
        Extract task ID from RequestContext.
        """
        task = getattr(context, "current_task", None) or getattr(context, "task", None)
        if task is not None:
            tid = getattr(task, "id", None)
            if isinstance(tid, str) and tid:
                return tid
            
        # fallback for platform compatibility
        for k in ("task_id", "current_task_id"):
            v = getattr(context, k, None)
            if isinstance(v, str) and v:
                return v

        return None

    def _get_message_and_context_id(self, context: RequestContext) -> Tuple[Optional[Message], str]:
        """
        Extract Message and context_id from RequestContext. 
        If context_id is missing, generate a new random one.
        """
        msg = getattr(context, "message", None)
        
        ctx_id = None
        if msg is not None:
            ctx_id = getattr(msg, "context_id", None) or getattr(msg, "contextId", None)

        context_id = ctx_id if isinstance(ctx_id, str) and ctx_id else uuid4().hex
        return msg, context_id

    def _extract_text(self, msg: Message) -> Optional[str]:
        """
        Extract TextPart payload from message.parts
        """
        parts = getattr(msg, "parts", None)

        if isinstance(parts, list):
            for part in parts:

                root = getattr(part, "root", None)

                if isinstance(root, TextPart):
                    text = getattr(root, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text
                    
                if isinstance(part, TextPart):
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text
                    
                if isinstance(root, dict) and isinstance(root.get("text"), str) and root["text"].strip():
                    return root["text"]
                
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"]
                
                text_attr = getattr(part, "text", None)
                if isinstance(text_attr, str) and text_attr.strip():
                    return text_attr

        for attr in ("text", "content", "body"):
            v = getattr(msg, attr, None)
            if isinstance(v, str) and v.strip():
                return v

        return None

    def _decode_obs(self, obs_base64: str) -> np.ndarray:
        """
        Docstring for _decode_obs
        
        :param self: Description
        :param obs_base64: Description
        :type obs_base64: str
        :return: Description
        :rtype: Any
        """ 
        # strip data URI prefix if present
        if obs_base64.startswith("data:"):
            obs_base64 = obs_base64.split("base64,", 1)[-1]

        # base64 decode
        try:
            img_bytes = base64.b64decode(obs_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 image payload: {e}") from e
        # numpy frombuffer
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # OpenCV decode (BGR)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed (invalid or corrupted image bytes)")

        # BGR -> RGB
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # enforce shape & dtype
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Decoded image must be HxWx3, got shape={img.shape}")

        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)

        return img
    

    
    async def _complete_json(self, updater: TaskUpdater, payload_obj: Dict[str, Any]) -> None:
        """
        Always complete task with a JSON-string response so evaluator/ToolProvider can json.loads().
        """
        text = json.dumps(payload_obj, ensure_ascii=False)
        await updater.complete(new_agent_text_message(text))

    def _touch_agent_state(self, context_id: str) -> None:
        self._agent_state_touched_at[context_id] = time.time()


    def _gc_agent_states(self) -> None:
        """
        GC stale agent_states to avoid memory leak in long-running servers.
        Mirrors SessionManager TTL concept, but independently maintained.
        """
        if self._state_ttl_seconds is None:
            return
        now = time.time()
        dead = [
            cid for cid, ts in self._agent_state_touched_at.items()
            if (now - ts) > self._state_ttl_seconds
        ]
        for cid in dead:
            self._agent_state_touched_at.pop(cid, None)
            self.agent_states.pop(cid, None)


    def _normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate action dict to:
        """
        if not isinstance(action, dict):
            raise ValueError(f"action must be dict, got {type(action)}")

        buttons = action.get("buttons", [])
        camera = action.get("camera", [])

        # ---- buttons ----
        if hasattr(buttons, "detach"):  # torch tensor
            buttons = buttons.detach().cpu().reshape(-1).tolist()
        elif isinstance(buttons, np.ndarray):
            buttons = buttons.reshape(-1).tolist()

        if not isinstance(buttons, list):
            raise ValueError(f"buttons must be a list-like, got {type(buttons)}")

        buttons_int: list[int] = []
        for i, b in enumerate(buttons):
            # bool
            if isinstance(b, (bool, np.bool_)):
                buttons_int.append(1 if bool(b) else 0)
                continue

            # int
            if isinstance(b, (int, np.integer)):
                buttons_int.append(1 if int(b) != 0 else 0)
                continue

            # float-like
            if isinstance(b, (float, np.floating)):
                bf = float(b)
                if not math.isfinite(bf):
                    buttons_int.append(0)
                    continue

                # probability-style in [0,1]
                if 0.0 <= bf <= 1.0:
                    buttons_int.append(1 if bf >= self._buttons_threshold else 0)
                else:
                    # logits / real-valued
                    buttons_int.append(1 if bf > 0.0 else 0)
                continue

            # fallback: attempt numeric cast
            try:
                bf = float(b)
                if not math.isfinite(bf):
                    buttons_int.append(0)
                elif 0.0 <= bf <= 1.0:
                    buttons_int.append(1 if bf >= self._buttons_threshold else 0)
                else:
                    buttons_int.append(1 if bf > 0.0 else 0)
            except Exception as e:
                raise ValueError(f"buttons[{i}] has unsupported type {type(b)}: {e}") from e

        if len(buttons_int) != 20:
            raise ValueError(f"buttons must have length 20, got {len(buttons_int)}")

        # ---- camera ----
        if hasattr(camera, "detach"):  # torch tensor
            camera = camera.detach().cpu().reshape(-1).tolist()
        elif isinstance(camera, np.ndarray):
            camera = camera.reshape(-1).tolist()

        if not isinstance(camera, list):
            raise ValueError(f"camera must be a list-like, got {type(camera)}")
        if len(camera) != 2:
            raise ValueError(f"camera must have length 2, got {len(camera)}")

        try:
            camera_f = [float(camera[0]), float(camera[1])]
        except Exception as e:
            raise ValueError(f"camera entries must be float-castable: {e}") from e

        # sanitize non-finite values
        for j in range(2):
            if not math.isfinite(camera_f[j]):
                camera_f[j] = 0.0

        return {"buttons": buttons_int, "camera": camera_f}

    

    # ---------------- A2A entrypoints ----------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute Purple agent logic for the given RequestContext.
        """
        self._gc_agent_states()

        task_id = self._get_task_id(context)
        msg, context_id = self._get_message_and_context_id(context)

        # ignore if no task_id
        if not task_id:
            logger.warning(
                    "RequestContext has no task id; ignoring. "
                    "type(context)=%s has_message=%s context_id=%s",
                    type(context),
                    msg is not None,
                    context_id,
                )
            return

        updater = TaskUpdater(event_queue, task_id, context_id)

        try:
            if msg is None:
                await self._complete_json(
                    updater,
                    {"type": "ack", "success": False, "message": "No message in request context"},
                )
                return

            payload_text = self._extract_text(msg)
            if payload_text is None:
                await self._complete_json(
                    updater,
                    {"type": "ack", "success": False, "message": "Missing TextPart payload"},
                )
                return

            try:
                payload_obj = json.loads(payload_text)
            except Exception as e:
                await self._complete_json(
                    updater,
                    {"type": "ack", "success": False, "message": f"Payload is not valid JSON: {e}"},
                )
                return

            payload_type = payload_obj.get("type", None)

            # ---------------- init ----------------
            if payload_type == "init":
                init = InitPayload.model_validate(payload_obj)

                self.sessions.start_new_task(context_id=context_id, task_text=init.text)

                self.agent.reset()
                state = self.agent.initial_state(init.text)

                self.agent_states[context_id] = state
                self._touch_agent_state(context_id)
                            
                await self._complete_json(
                    updater,
                    {"type": "ack", "success": True, "message": f"Initialization success with task: {init.text}"},
                )
                return

            # ---------------- obs ----------------
            if payload_type == "obs":
                obs = ObservationPayload.model_validate(payload_obj)
                
                _ = self.sessions.on_observation(context_id, obs.step)

                image = self._decode_obs(obs.obs)
                obs_dict = {"image": image}

                state = self.agent_states.get(context_id)
                if state is None:
                    raise RuntimeError(f"Missing agent state for context_id={context_id}")
                    
                action, new_state = self.agent.act(
                    obs=obs_dict, 
                    state=state, 
                    deterministic=True
                )

                self.agent_states[context_id] = new_state
                self._touch_agent_state(context_id)

                action = self._normalize_action(action)

                
                await self._complete_json(
                    updater,
                    {
                        "type": "action",
                        "buttons": action["buttons"],
                        "camera": action["camera"],
                    },
                )
                return

            # ---------------- unknown ----------------
            await self._complete_json(
                updater,
                {"type": "ack", "success": False, "message": f"Unknown payload type: {payload_type}"},
            )

        except Exception as e:
            """Catch-all for unexpected errors."""
            try:
                await self._complete_json(
                    updater,
                    {"type": "ack", "success": False, "message": f"Unhandled server error: {e}"},
                )
            except Exception:
                await updater.failed(new_agent_text_message(f"Fatal error: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Purple agent does not support cancellation
        return
