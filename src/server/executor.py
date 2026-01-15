from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel

import cv2
import numpy as np

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TextPart
from a2a.utils import new_agent_text_message  # fallback 용도로만 사용

from src.server.session_manager import SessionManager
from src.protocol.models import InitPayload, ObservationPayload, AckPayload, ActionPayload
from src.agent.registry import build_agent
from src.agent.base import AgentState

logger = logging.getLogger("purple.executor")
logger.setLevel(logging.DEBUG)

def noop_action() -> Dict[str, Any]:
    return {"buttons": [0], "camera": [60]}

def _noop_action_payload() -> Dict[str, Any]:
    return ActionPayload(   
        action_type="agent",
        buttons=noop_action()["buttons"],
        camera=noop_action()["camera"],
    )


class Executor(AgentExecutor):
    """
    MCU Purple Policy Executor (A2A-compatible).

    Contract:
      - Input : message/send with JSON in TextPart (init/obs)
      - Output: (A2A) emits exactly ONE terminal TaskStatusUpdateEvent via TaskUpdater.complete()
              and also returns Message for compatibility.
      - No streaming, no multi-event lifecycle.
      - Never returns None.
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

        # context_id -> agent/state/action
        self.agents: dict[str, Any] = {}
        self.agent_states: dict[str, AgentState] = {}
        self._agent_state_touched_at: dict[str, float] = {}
        self._last_actions: dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_text(self, msg: Optional[Message]) -> Optional[str]:
        if msg is None:
            return None

        parts = getattr(msg, "parts", None)
        if not isinstance(parts, list):
            return None

        for part in parts:
            # Case 1) Part(root=TextPart(...))
            root = getattr(part, "root", None)
            if isinstance(root, TextPart) and isinstance(root.text, str):
                return root.text

            # Case 2) part itself is TextPart
            if isinstance(part, TextPart) and isinstance(part.text, str):
                return part.text

            # Case 3) dict-like
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    return text

            # Case 4) generic attribute 'text'
            text = getattr(part, "text", None)
            if isinstance(text, str):
                return text

        return None

    def _decode_obs(self, obs_base64: str) -> np.ndarray:
        if obs_base64.startswith("data:"):
            obs_base64 = obs_base64.split("base64,", 1)[-1]

        img_bytes = base64.b64decode(obs_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed")

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {img.shape}")
        return img

    def _gc_agent_states(self) -> None:
        if self._state_ttl_seconds is None:
            return
        now = time.time()
        dead = [
            cid
            for cid, ts in self._agent_state_touched_at.items()
            if (now - ts) > self._state_ttl_seconds
        ]
        for cid in dead:
            self._agent_state_touched_at.pop(cid, None)
            self.agent_states.pop(cid, None)
            self.agents.pop(cid, None)
            self._last_actions.pop(cid, None)

    def _touch(self, context_id: str) -> None:
        self._agent_state_touched_at[context_id] = time.time()

    def _get_or_create_agent(self, context_id: str) -> Any:
        agent = self.agents.get(context_id)
        if agent is None:
            logger.info(
                "[Purple] build_agent name=%s device=%s",
                self.agent_name,
                self._device,
            )
            agent = build_agent(self.agent_name, device=self._device)
            if hasattr(agent, "reset"):
                agent.reset()
            self.agents[context_id] = agent
        return agent

    def _make_agent_message(self, *, task_id: str, context_id: str, payload_obj: Dict[str, Any] | BaseModel) -> Message:
        """
        Make agent Message with given payload object.
        """
        if isinstance(payload_obj, BaseModel):
            text = payload_obj.model_dump_json()
        else:
            text = json.dumps(payload_obj)
            
        return Message(
            role=Role.agent,
            task_id=task_id,
            context_id=context_id,
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=text))],
        )

    async def _finalize(self, *, event_queue, task_id: str, context_id: str, message: Message) -> Message:
        """
        Emit exactly one terminal completion event if event_queue is provided.
        """
        if event_queue is not None:
            updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
            )
            await updater.complete(message=message)
        return message

    # ------------------------------------------------------------------
    # Main entry (IMPORTANT)
    # ------------------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue=None) -> Message:
        """
        - MUST return Message
        - For A2A runtime: also emits exactly one terminal completion event (if event_queue provided)
        """
        self._gc_agent_states()

        msg = getattr(context, "message", None)
        payload_text = self._extract_text(msg)

     
        context_id = (getattr(msg, "context_id", None) or getattr(context, "context_id", None) or "default")
        task_id = (getattr(msg, "task_id", None) or getattr(context, "task_id", None) or context_id)

        try:
            if not payload_text:
                noop_msg = self._make_agent_message(
                    task_id=task_id,
                    context_id=context_id,
                    payload_obj=_noop_action_payload(),
                )
                return await self._finalize(
                    event_queue=event_queue, task_id=task_id, context_id=context_id, message=noop_msg
                )

            payload = json.loads(payload_text)
            payload_type = payload.get("type")

            # ---------------- init ----------------
            if payload_type == "init":
                init = InitPayload.model_validate(payload)

                self.sessions.start_new_task(
                    context_id=context_id,
                    task_text=init.text,
                )

                agent = self._get_or_create_agent(context_id)
                state = agent.initial_state(init.text)

                self.agent_states[context_id] = state
                self._last_actions[context_id] = noop_action()
                self._touch(context_id)

                ack_payload = AckPayload(success=True, message="Initialization successful.")
                ack_msg = self._make_agent_message(
                    task_id=task_id,
                    context_id=context_id,
                    payload_obj=ack_payload
                )
                return await self._finalize(
                    event_queue=event_queue, task_id=task_id, context_id=context_id, message=ack_msg
                )

            # ---------------- obs ----------------
            if payload_type == "obs":
                # Observation 
                obs = ObservationPayload.model_validate(payload)

                # Decode observation
                image_rgb = self._decode_obs(obs.obs)
                obs_dict = {"image": image_rgb, "step": obs.step}

                # Session 
                self.sessions.on_observation(context_id, obs.step)

                # Agent + State
                agent = self._get_or_create_agent(context_id)
                state = self.agent_states.get(context_id)

                if state is None:
                    noop_msg = self._make_agent_message(
                        task_id=task_id,
                        context_id=context_id,
                        payload_obj=_noop_action_payload(),
                    )
                    return await self._finalize(
                        event_queue=event_queue, task_id=task_id, context_id=context_id, message=noop_msg
                    )
                # Act
                action, new_state = agent.act(
                    obs=obs_dict,
                    state=state,
                    deterministic=True,
                )
                # Update state
                self.agent_states[context_id] = new_state
                self._touch(context_id)

                # Store last action
                self._last_actions[context_id] = action 

                # Build action message
                action_payload = ActionPayload(
                    action_type="agent",
                    buttons=action["buttons"],
                    camera=action["camera"],
                )
                
                # Emit action message
                action_msg = self._make_agent_message(
                    task_id=task_id,
                    context_id=context_id,
                    payload_obj=action_payload
                )

                # Finalize
                return await self._finalize(
                    event_queue=event_queue, task_id=task_id, context_id=context_id, message=action_msg
                )

            # ---------------- unknown ----------------
            noop_msg = self._make_agent_message(
                task_id=task_id,
                context_id=context_id,
                payload_obj=_noop_action_payload(),
            )
            return await self._finalize(
                event_queue=event_queue, task_id=task_id, context_id=context_id, message=noop_msg
            )

        except Exception:
            logger.exception("[EXEC] fatal error")

            
            noop_msg = self._make_agent_message(
                task_id=task_id,
                context_id=context_id,
                payload_obj=_noop_action_payload(),
            )
            try:
                return await self._finalize(
                    event_queue=event_queue, task_id=task_id, context_id=context_id, message=noop_msg
                )
            except Exception:
                return new_agent_text_message(json.dumps(_noop_action_payload()))

    async def cancel(self, context: RequestContext, event_queue=None) -> None:
        # MCU: ignore cancel
        return