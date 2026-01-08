from __future__ import annotations

import json
import base64
from uuid import uuid4

import cv2
import numpy as np

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TextPart
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

from src.server.session_manager import SessionManager
from src.protocol.models import (
    InitPayload,
    ObservationPayload,
    AckPayload,
    ActionPayload,
)

from src.agent.noop import NoOpAgent
from src.agent.rocket1.agent import Rocket1Agent
from src.agent.base import AgentState


class Executor(AgentExecutor):
    """
    Purple AgentExecutor

    - init / obs 메시지 처리
    - NoOp 또는 Rocket-1 policy 호출
    - ActionPayload 반환
    """

    def __init__(self, sessions: SessionManager, agent_name: str = "noop"):
        self.sessions = sessions

        if agent_name == "noop":
            self.agent = NoOpAgent()
            self.mode = "noop"

        elif agent_name == "rocket1":
            self.agent = Rocket1Agent()
            self.mode = "rocket1"
            # context_id -> AgentState (Rocket recurrent memory)
            self.agent_states: dict[str, AgentState] = {}

        else:
            raise ValueError(f"Unknown agent name: {agent_name}")

    # ---------------- main entry ----------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg: Message | None = context.message
        if msg is None:
            raise ServerError("No message in request context")

        context_id = msg.context_id or uuid4().hex

        # 1) extract TextPart
        payload_text = self._extract_text(msg)
        if payload_text is None:
            await self._fail(
                updater=TaskUpdater(event_queue, context.current_task.id, context_id),
                reason="Missing TextPart payload",
            )
            return

        # 2) parse JSON
        try:
            payload_obj = json.loads(payload_text)
        except Exception:
            await self._respond(
                event_queue,
                context_id,
                AckPayload(success=False, message="Payload is not valid JSON"),
            )
            return

        payload_type = payload_obj.get("type")

        # ---------------- init ----------------
        if payload_type == "init":
            try:
                init = InitPayload.model_validate(payload_obj)
            except Exception as e:
                await self._respond(
                    event_queue,
                    context_id,
                    AckPayload(success=False, message=f"Invalid init payload: {e}"),
                )
                return

            self.sessions.start_new_task(
                context_id=context_id,
                task_text=init.text,
            )

            # Rocket-1: reset recurrent state
            if self.mode == "rocket1":
                self.agent.reset()
                self.agent_states[context_id] = AgentState(memory=None, first=True)

            await self._respond(
                event_queue,
                context_id,
                AckPayload(
                    success=True,
                    message=f"Initialization success with task: {init.text}",
                ),
            )
            return

        # ---------------- obs ----------------
        if payload_type == "obs":
            try:
                obs = ObservationPayload.model_validate(payload_obj)
            except Exception as e:
                await self._respond(
                    event_queue,
                    context_id,
                    AckPayload(success=False, message=f"Invalid obs payload: {e}"),
                )
                return

            session = self.sessions.on_observation(context_id, obs.step)

            # ---------- NoOp ----------
            if self.mode == "noop":
                action_dict = self.agent.act(
                    obs_base64=obs.obs,
                    session=session,
                )

            # ---------- Rocket-1 ----------
            else:
                image = self._decode_obs(obs.obs)

                obs_dict = {
                    "image": image,
                    # segment 없음 → preprocess에서 zero-mask 자동 처리
                }

                state = self.agent_states.get(context_id)
                if state is None:
                    state = AgentState(memory=None, first=True)

                action_dict, new_state = self.agent.act(
                    obs=obs_dict,
                    state=state,
                    deterministic=True,
                )

                self.agent_states[context_id] = new_state

            await self._respond(
                event_queue,
                context_id,
                ActionPayload(
                    buttons=action_dict["buttons"],
                    camera=action_dict["camera"],
                ),
            )
            return

        # ---------------- unknown ----------------
        await self._respond(
            event_queue,
            context_id,
            AckPayload(success=False, message=f"Unknown payload type: {payload_type}"),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Purple agent does not support cancellation
        return

    # ---------------- helpers ----------------

    def _extract_text(self, msg: Message) -> str | None:
        for part in msg.parts:
            if isinstance(part.root, TextPart):
                return part.root.text
        return None

    def _decode_obs(self, obs_base64: str) -> np.ndarray:
        """base64 jpeg/png -> RGB numpy image"""
        img_bytes = base64.b64decode(obs_base64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    async def _respond(self, event_queue: EventQueue, context_id: str, payload) -> None:
        msg = Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text=payload.model_dump_json()))],
            message_id=uuid4().hex,
            context_id=context_id,
        )
        await event_queue.enqueue_event(msg)

    async def _fail(self, updater: TaskUpdater, reason: str) -> None:
        await updater.failed(new_agent_text_message(reason))
