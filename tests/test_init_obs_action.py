import json
from uuid import uuid4

import httpx
import pytest

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

from pydantic import BaseModel, Field
from typing import Literal


class InitPayload(BaseModel):
    type: Literal["init"] = "init"
    text: str = Field(...)


class ObservationPayload(BaseModel):
    type: Literal["obs"] = "obs"
    step: int = Field(..., ge=0)
    obs: str = Field(...)


class AckPayload(BaseModel):
    type: Literal["ack"] = "ack"
    success: bool = False
    message: str = ""


class ActionPayload(BaseModel):
    type: Literal["action"] = "action"
    buttons: list[int]
    camera: list[float]


async def send_text(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        cfg = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        client = ClientFactory(cfg).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]
        return events


def _extract_last_text(events) -> str:
    last = events[-1]
    # Depending on streaming/non-streaming, could be Message() or (task, update)
    if hasattr(last, "parts"):  # Message
        return last.parts[0].root.text  # type: ignore
    if isinstance(last, tuple):
        task, update = last
        # completed task usually contains status.message
        if task.status and task.status.message and task.status.message.parts:
            return task.status.message.parts[0].root.text
    return ""


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_init_then_obs(agent_url, streaming):
    # init
    init = InitPayload(text="build a house").model_dump_json()
    events = await send_text(init, agent_url, streaming=streaming)
    out = _extract_last_text(events)
    ack = AckPayload.model_validate_json(out)
    assert ack.type == "ack"
    assert ack.success is True

    # obs (fake base64; purple will reject decode => 그래서 여기선 decode 가능한 dummy가 필요)
    # 테스트는 protocol 경로만 확인하면 되니, 최소한 decode 통과용으로 "빈 jpeg"를 넣는다.
    # 1x1 black jpeg base64 (static)
    tiny_jpeg_b64 = (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
        "2wCEAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
        "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAgP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCkA//Z"
    )
    obs = ObservationPayload(step=0, obs=tiny_jpeg_b64).model_dump_json()
    events2 = await send_text(obs, agent_url, streaming=streaming)
    out2 = _extract_last_text(events2)
    action = ActionPayload.model_validate_json(out2)
    assert action.type == "action"
    assert isinstance(action.buttons, list) and len(action.buttons) >= 1
    assert isinstance(action.camera, list) and len(action.camera) >= 1
