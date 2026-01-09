from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, HttpUrl


# Payloads for HTTP communication with Purple agent server.
class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]

# Init payloads for WebSocket communication with Purple agent server.
class InitPayload(BaseModel):
    type: Literal["init"] = "init"
    text: str = Field(..., description="Task description")

# Observation payloads for WebSocket communication with Purple agent server.
class ObservationPayload(BaseModel):
    type: Literal["obs"] = "obs"
    step: int = Field(..., ge=0, description="Current step number")
    obs: str = Field(..., description="Base64 encoded image")

# Ack payloads for WebSocket communication with Purple agent server.
class AckPayload(BaseModel):
    type: Literal["ack"] = "ack"
    success: bool = False
    message: str = ""

# Action payloads for WebSocket communication with Purple agent server.
class ActionPayload(BaseModel):
    type: Literal["action"] = "action"
    buttons: list[int] = Field(..., description="Button states")
    camera: list[float] = Field(..., description="Camera movements")
