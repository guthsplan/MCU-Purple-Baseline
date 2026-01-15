from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

from pydantic import BaseModel, model_validator, Field

from typing import Any, Literal, Iterable, Optional, Dict


class InitPayload(BaseModel):
    """Initial task description sent to purple agent."""
    type: Literal["init"] = "init"
    text: str = Field(..., description="Task description")

class ObservationPayload(BaseModel):
    """Observation sent to purple agent at each step."""
    type: Literal["obs"] = "obs"
    step: int = Field(..., ge=0, description="Current step number")
    obs: str = Field(..., description="Base64 encoded image")


class AckPayload(BaseModel):
    type: Literal["ack"] = "ack"
    success: bool = False
    message: str = ""


class ActionPayload(BaseModel):
    """Action response from purple agent.
    
    Supports three formats:
    1. Compact agent format: {"type": "action", "action_type": "agent", "buttons": [123], "camera": [60]}
    2. Expanded agent format: {"type": "action", "action_type": "agent", "buttons": [0,0,0,1,...], "camera": [0.0, 90.0]}
    3. Env format: {"type": "action", "action_type": "env", "action": {"forward": 0, "back": 0, ..., "camera": [...]}}
    """
    type: Literal["action"] = "action"
    action_type: Literal["agent", "env"] = "agent"
    # agent action type fields (formats 1 & 2)
    buttons: Optional[list] = Field(None, description="Button action (agent action space)")
    camera: Optional[list] = Field(None, description="Camera movements (agent action space)")
    # env action type field (format 3)
    action: Optional[Dict[str, Any]] = Field(None, description="Detailed action dict (env action space)")
    
    @model_validator(mode='after')
    def validate_format(self):
        """Validate action format based on action_type."""
        
        if self.action_type == "agent":
            # Validate agent action format
            if self.buttons is None:
                raise ValueError("buttons field is required for action_type='agent'")
            if self.camera is None:
                raise ValueError("camera field is required for action_type='agent'")
            
            if not isinstance(self.buttons, list):
                raise ValueError("buttons field must be a list")
            if len(self.buttons) != 1 and len(self.buttons) != 20:
                raise ValueError(f"buttons must have length 1 or 20, got {len(self.buttons)}")
            
            if not isinstance(self.camera, (list, tuple)):
                raise ValueError("camera field must be a list or tuple")
            if len(self.camera) != 1 and len(self.camera) != 2:
                raise ValueError(f"camera must have length 1 or 2, got {len(self.camera)}")
                
        elif self.action_type == "env":
            # Validate env action format
            if self.action is None:
                raise ValueError("action field is required for action_type='env'")
            
            if not isinstance(self.action, dict):
                raise ValueError("action field must be a dictionary")
            
            # Validate required keys for env action
            required_keys = {
                'forward', 'back', 'left', 'right', 
                'jump', 'sneak', 'sprint',
                'attack', 'use', 'drop', 'inventory',
                'camera'
            }
            required_keys.update({f'hotbar.{i}' for i in range(1, 10)})
            
            missing_keys = required_keys - set(self.action.keys())
            if missing_keys:
                # Auto-fill missing keys with defaults
                for key in missing_keys:
                    if key == 'camera':
                        self.action['camera'] = [0.0, 0.0]
                    else:
                        self.action[key] = 0
            
            # Validate camera format
            camera = self.action.get('camera')
            if camera is not None:
                if not isinstance(camera, (list, tuple)):
                    raise ValueError(f"camera must be a list or tuple, got {type(camera)}")
                if len(camera) != 2:
                    raise ValueError(f"camera must have length 2, got {len(camera)}")
        
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
    timeout_s: float = 120.0,
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
    assert action.action_type == "agent", f"Unexpected action_type={action.action_type}. Full:\n{action.model_dump()}"

    # Validate action format (support both compact and expanded formats)
    assert action.buttons is not None, "buttons field is required"
    assert action.camera is not None, "camera field is required"
    assert len(action.buttons) in (1, 20), f"buttons length must be 1 or 20, got {len(action.buttons)}"
    assert len(action.camera) in (1, 2), f"camera length must be 1 or 2, got {len(action.camera)}"
    
    # If expanded format (20 buttons), validate 0/1 values
    if len(action.buttons) == 20:
        assert all(b in (0, 1) for b in action.buttons), "buttons must be 0/1 in expanded format"


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