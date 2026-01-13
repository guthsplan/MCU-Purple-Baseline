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
        decode_expect_rgb: bool = True,
        debug: bool = False,
        device: Optional[str] = None,
        ) -> None:

        # Initialize Purple agent instance
        self.sessions = sessions
        self.agent_name = agent_name
        self._state_ttl_seconds = state_ttl_seconds 
        self._debug = bool(debug)
        self._device = device

        # Per-context agent instance
        self.agents: dict[str, Any] = {}

        # Per-context recurrent state
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
        Decode base64-encoded image observation to RGB numpy array.
        """ 
        # strip data URI prefix if present
        if obs_base64.startswith("data:"):
            obs_base64 = obs_base64.split("base64,", 1)[-1]

        # base64 decode
        try:
            img_bytes = base64.b64decode(obs_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 image payload: {e}") from e
        
        # convert to numpy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        # OpenCV decode (BGR)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed (invalid or corrupted image bytes)")
        
        # convert to RGB for downstream agents
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

            agent = self.agents.pop(cid, None)
            
            try:
                if agent is not None and hasattr(agent, "close"):
                    agent.close()
            except Exception:
                logger.debug("Ignoring agent.close() error for cid=%s", cid, exc_info=True)


    def _get_or_create_agent(self, context_id: str) -> Any:
        """
        Get or create per-context agent instance.
        """
        agent = self.agents.get(context_id)
        if agent is not None:
            return agent

        agent = build_agent(self.agent_name, device=self._device)
        # If agent exposes reset, do it on creation; otherwise ignore.
        try:
            if hasattr(agent, "reset"):
                agent.reset()
        except Exception:
            logger.debug("Ignoring agent.reset() error during creation", exc_info=True)

        self.agents[context_id] = agent
        return agent
    
    def _build_obs_dict(self, obs: ObservationPayload, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Build observation dict for agent.act() from ObservationPayload and decoded image.
        """
        obs_dict: Dict[str, Any] = {
            "image": image_rgb,
            "step": getattr(obs, "step", None),
        }

        # Add optional fields if present
        for k in ("inventory", "status", "task", "reward", "done", "info"):
            v = getattr(obs, k, None)
            if v is not None:
                obs_dict[k] = v

        return obs_dict

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

                agent = self._get_or_create_agent(context_id)

                # Create per-context initial state
                state = agent.initial_state(init.text)

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

                image_rgb = self._decode_obs(obs.obs)
                obs_dict = self._build_obs_dict(obs, image_rgb)

                agent = self._get_or_create_agent(context_id)

                state = self.agent_states.get(context_id)
                if state is None:
                    raise RuntimeError(f"Missing agent state for context_id={context_id}. Did you receive init?")

                action, new_state = agent.act(obs=obs_dict, state=state, deterministic=True)

                self.agent_states[context_id] = new_state
                self._touch_agent_state(context_id)

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
            logger.exception(
                "Unhandled error in Executor.execute (context_id=%s, task_id=%s)",
                context_id,
                task_id,
            )

            fallback_action = {
                "type": "action",
                "buttons": [0] * 20,
                "camera": [0.0, 0.0],
            }

            try:
                await self._complete_json(updater, fallback_action)
            except Exception:
                await updater.failed(new_agent_text_message("Fatal error"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Purple agent does not support cancellation
        return
