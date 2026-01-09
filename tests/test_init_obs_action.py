from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

from pydantic import BaseModel, Field
from typing import Any, Literal


class InitPayload(BaseModel):
    type: Literal["init"] = "init"
    text: str = Field(...)


class ObservationPayload(BaseModel):
    type: Literal["obs"] = "obs"
    step: int = Field(..., ge=0)
    obs: str = Field(..., min_length=1, description="base64-encoded jpeg/png (no data: prefix)")


class AckPayload(BaseModel):
    type: Literal["ack"] = "ack"
    success: bool = False
    message: str = ""


class ActionPayload(BaseModel):
    type: Literal["action"] = "action"
    buttons: list[int] = Field(..., description="MineRL/VPT 20 buttons (0/1)")
    camera: list[float] = Field(..., description="(dx, dy)")

    @model_validator(mode="after")
    def _validate_shapes_and_values(self):
        if len(self.buttons) != 20:
            raise ValueError(f"buttons must have length 20, got {len(self.buttons)}")
        if any(b not in (0, 1) for b in self.buttons):
            bad = [(i, b) for i, b in enumerate(self.buttons) if b not in (0, 1)]
            raise ValueError(f"buttons must be 0/1 only; bad entries: {bad[:5]}")
        if len(self.camera) != 2:
            raise ValueError(f"camera must have length 2, got {len(self.camera)}")
        # coercion
        self.camera = [float(self.camera[0]), float(self.camera[1])]
        return self

TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
    "2wCEAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
    "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAgP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCkA//Z"
)

def _find_last_text(events: Iterable[Any]) -> str:
    """
    Extract the last text message from a list of events.
    """
    ev_list = list(events)
    for ev in reversed(ev_list):
        # Message case
        if hasattr(ev, "parts"):
            try:
                parts = getattr(ev, "parts")
                if parts and hasattr(parts[0], "root") and hasattr(parts[0].root, "text"):
                    return parts[0].root.text  # type: ignore[attr-defined]
            except Exception:
                pass

        # Tuple case
        if isinstance(ev, tuple) and len(ev) == 2:
            task, _update = ev
            try:
                status = getattr(task, "status", None)
                message = getattr(status, "message", None) if status else None
                parts = getattr(message, "parts", None) if message else None
                if parts and hasattr(parts[0], "root") and hasattr(parts[0].root, "text"):
                    return parts[0].root.text  # type: ignore[attr-defined]
            except Exception:
                pass

    # If nothing extracted, return empty string
    return ""

 
async def send_text(
    text: str,
    url: str,
    *,
    context_id: Optional[str] = None,
    streaming: bool = False,
    timeout_s: float = 20.0,
):
    """
    A2A card resolve -> client -> send_message -> collect all events.
    """
    async with httpx.AsyncClient(timeout=timeout_s) as httpx_client:
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

def _pretty_json(s: str) -> str:
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return s
    
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
async def test_init_then_obs_same_context(agent_url, streaming):
    """
    Test init followed by obs in the same context_id.
    """
    cid = uuid4().hex

    # init
    init = InitPayload(text="build a house").model_dump_json()
    events = await send_text(init, agent_url, context_id=cid, streaming=streaming)
    out = _find_last_text(events)

    assert out, f"No response text extracted.\nEvents={events!r}"
    try:
        ack = AckPayload.model_validate_json(out)
    except Exception as e:
        raise AssertionError(
            "Init response is not a valid AckPayload JSON.\n"
            f"Extracted text:\n{_pretty_json(out)}\n"
            f"Error: {e}"
        ) from e

    assert ack.type == "ack", f"Unexpected ack.type={ack.type}. Full:\n{ack.model_dump()}"
    assert ack.success is True, f"Init ack.success must be True. Full:\n{ack.model_dump()}"

    # obs 
    obs = ObservationPayload(step=0, obs=TINY_JPEG_B64).model_dump_json()
    events2 = await send_text(obs, agent_url, context_id=cid, streaming=streaming)
    out2 = _find_last_text(events2)

    assert out2, f"No response text extracted for obs.\nEvents={events2!r}"
    try:
        action = ActionPayload.model_validate_json(out2)
    except Exception as e:
        raise AssertionError(
            "Obs response is not a valid ActionPayload JSON.\n"
            f"Extracted text:\n{_pretty_json(out2)}\n"
            f"Error: {e}"
        ) from e

    assert action.type == "action", f"Unexpected action.type={action.type}. Full:\n{action.model_dump()}"

    # These are already enforced by ActionPayload validator, but keep explicit asserts for readability.
    assert len(action.buttons) == 20
    assert len(action.camera) == 2
    assert all(b in (0, 1) for b in action.buttons)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_obs_without_init_should_not_crash(agent_url, streaming):
    """
    Test sending obs without prior init.
    """
    cid = uuid4().hex
    obs = ObservationPayload(step=0, obs=TINY_JPEG_B64).model_dump_json()
    events = await send_text(obs, agent_url, context_id=cid, streaming=streaming)
    out = _find_last_text(events)

    assert out, f"No response text extracted.\nEvents={events!r}"

    try:
        obj = json.loads(out)
    except Exception as e:
        raise AssertionError(
            "Obs(without init) response is not valid JSON.\n"
            f"Extracted text:\n{out}\n"
            f"Error: {e}"
        ) from e

    assert isinstance(obj, dict), f"Response JSON must be an object, got: {type(obj)}"
    assert "type" in obj, f"Response JSON must contain 'type'. Got:\n{_pretty_json(out)}"
    assert obj["type"] in ("ack", "action"), f"Unexpected type={obj['type']}. Got:\n{_pretty_json(out)}"