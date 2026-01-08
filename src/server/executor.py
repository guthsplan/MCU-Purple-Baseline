from __future__ import annotations

import json
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message,
    Part,
    Role,
    TextPart,
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
#from a2a.types.errors import InvalidRequestError

from src.server.session_manager import SessionManager
from src.protocol.models import (
    InitPayload,
    ObservationPayload,
    AckPayload,
    ActionPayload,
)

from src.agent.noop import NoOpAgent
# from agent.rocket1.agent import Rocket1Agent   # 나중에 연결


class Executor(AgentExecutor):
    """
    Purple AgentExecutor

    역할:
    - A2A Message 수신
    - TextPart에서 JSON payload 추출
    - init / obs 분기
    - policy(agent) 호출
    - ActionPayload 또는 AckPayload 생성
    """

    def __init__(self, sessions: SessionManager, agent_name: str = "noop"):
        self.sessions = sessions

        if agent_name == "noop":
            self.agent = NoOpAgent()
        # elif agent_name == "rocket1":
        #     self.agent = Rocket1Agent()
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg: Message | None = context.message
        if msg is None:
            raise ServerError(
                "No message in request context"
            )

        context_id = msg.context_id or uuid4().hex

        # 1) TextPart → payload text
        payload_text = self._extract_text(msg)
        if payload_text is None:
            await self._fail(
                updater=TaskUpdater(event_queue, context.current_task.id, context_id),
                reason="Missing TextPart payload",
            )
            return

        # 2) JSON parse
        try:
            payload_obj = json.loads(payload_text)
        except Exception:
            response = AckPayload(
                success=False,
                message="Payload is not valid JSON",
            )
            await self._respond(event_queue, context_id, response)
            return

        payload_type = payload_obj.get("type")

        # 3) init
        if payload_type == "init":
            try:
                init = InitPayload.model_validate(payload_obj)
            except Exception as e:
                response = AckPayload(
                    success=False,
                    message=f"Invalid init payload: {e}",
                )
                await self._respond(event_queue, context_id, response)
                return

            self.sessions.start_new_task(
                context_id=context_id,
                task_text=init.text,
            )

            response = AckPayload(
                success=True,
                message=f"Initialization success with task: {init.text}",
            )
            await self._respond(event_queue, context_id, response)
            return

        # 4) obs
        if payload_type == "obs":
            try:
                obs = ObservationPayload.model_validate(payload_obj)
            except Exception as e:
                response = AckPayload(
                    success=False,
                    message=f"Invalid obs payload: {e}",
                )
                await self._respond(event_queue, context_id, response)
                return

            session = self.sessions.on_observation(context_id, obs.step)

            action_dict = self.agent.act(
                obs_base64=obs.obs,
                session=session,
            )

            response = ActionPayload(
                buttons=action_dict["buttons"],
                camera=action_dict["camera"],
            )
            await self._respond(event_queue, context_id, response)
            return

        # 5) unknown
        response = AckPayload(
            success=False,
            message=f"Unknown payload type: {payload_type}",
        )
        await self._respond(event_queue, context_id, response)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(
            error=InvalidRequestError(message="Cancel not supported by purple agent")
        )

    # ---------------- helpers ----------------

    def _extract_text(self, msg: Message) -> str | None:
        for part in msg.parts:
            if isinstance(part.root, TextPart):
                return part.root.text
        return None

    async def _respond(self, event_queue: EventQueue, context_id: str, payload) -> None:
        text = payload.model_dump_json()

        msg = Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        await event_queue.enqueue_event(msg)

    async def _fail(self, updater: TaskUpdater, reason: str) -> None:
        await updater.failed(
            new_agent_text_message(reason)
        )
