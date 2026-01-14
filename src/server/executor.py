from __future__ import annotations

import base64
import json
import logging
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
    """

    def __init__(
        self,
        sessions: SessionManager,
        agent_name: str,
        *,
        state_ttl_seconds: Optional[int] = 60 * 60,
        debug: bool = False,
        device: Optional[str] = None,
    ) -> None:

        self.sessions = sessions
        self.agent_name = agent_name
        self._state_ttl_seconds = state_ttl_seconds
        self._debug = bool(debug)
        self._device = device

        self.agents: dict[str, Any] = {}
        self.agent_states: dict[str, AgentState] = {}
        self._agent_state_touched_at: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_task_id(self, context: RequestContext) -> Optional[str]:
        task = getattr(context, "current_task", None) or getattr(context, "task", None)
        if task is not None:
            tid = getattr(task, "id", None)
            if isinstance(tid, str) and tid:
                return tid
        return None

    def _get_message_and_context_id(self, context: RequestContext) -> Tuple[Optional[Message], str]:
        msg = getattr(context, "message", None)
        ctx_id = getattr(msg, "context_id", None) if msg else None
        return msg, ctx_id if isinstance(ctx_id, str) and ctx_id else uuid4().hex

    def _extract_text(self, msg: Message) -> Optional[str]:
        parts = getattr(msg, "parts", None)
        if isinstance(parts, list):
            for part in parts:
                root = getattr(part, "root", None)
                if isinstance(root, TextPart) and isinstance(root.text, str):
                    return root.text
        return None

    def _decode_obs(self, obs_base64: str) -> np.ndarray:
        if obs_base64.startswith("data:"):
            obs_base64 = obs_base64.split("base64,", 1)[-1]

        img_bytes = base64.b64decode(obs_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {img.shape}")
        return img

    async def _complete_json(self, updater: TaskUpdater, payload: Dict[str, Any]) -> None:
        await updater.complete(new_agent_text_message(json.dumps(payload)))

    def _touch_agent_state(self, context_id: str) -> None:
        self._agent_state_touched_at[context_id] = time.time()

    def _gc_agent_states(self) -> None:
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
            self.agents.pop(cid, None)

    def _get_or_create_agent(self, context_id: str) -> Any:
        agent = self.agents.get(context_id)
        if agent is None:
            agent = build_agent(self.agent_name, device=self._device)
            if hasattr(agent, "reset"):
                agent.reset()
            self.agents[context_id] = agent
        return agent

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._gc_agent_states()

        task_id = self._get_task_id(context)
        msg, context_id = self._get_message_and_context_id(context)

        if not task_id:
            return

        updater = TaskUpdater(event_queue, task_id, context_id)

        try:
            payload_text = self._extract_text(msg)
            payload = json.loads(payload_text)
            payload_type = payload.get("type")

            # ---------------- init ----------------
            if payload_type == "init":
                init = InitPayload.model_validate(payload)
                self.sessions.start_new_task(context_id=context_id, task_text=init.text)

                agent = self._get_or_create_agent(context_id)
                state = agent.initial_state(init.text)
                self.agent_states[context_id] = state
                self._touch_agent_state(context_id)

                await self._complete_json(
                    updater,
                    {"type": "ack", "success": True, "message": "Initialization success"},
                )
                return

            # ---------------- obs ----------------
            if payload_type == "obs":
                obs = ObservationPayload.model_validate(payload)

                image_rgb = self._decode_obs(obs.obs)
                obs_dict = {"image": image_rgb, "step": obs.step}

                agent = self._get_or_create_agent(context_id)
                state = self.agent_states.get(context_id)
                if state is None:
                    raise RuntimeError("Missing agent state (init not received)")

                action, new_state = agent.act(obs=obs_dict, state=state, deterministic=True)
                self.agent_states[context_id] = new_state
                self._touch_agent_state(context_id)

                # 🔒 FINAL, CONTRACT-SAFE ACTION
                await self._complete_json(
                    updater,
                    {
                        "type": "action",
                        "action_type": "agent",
                        "buttons": action["buttons"],   # len=20
                        "camera": action["camera"],     # len=2
                    },
                )
                return

            await self._complete_json(
                updater,
                {"type": "ack", "success": False, "message": "Unknown payload type"},
            )

        except Exception:
            logger.exception("Executor failure, returning noop action")

            await self._complete_json(
                updater,
                {
                    "type": "action",
                    "action_type": "agent",
                    "buttons": [0] * 20,
                    "camera": [0.0, 0.0],
                },
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return
