from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, HttpUrl


# (참고) 이건 green이 받는 EvalRequest와 동일한 형태를 유지해도 좋다.
# Purple은 직접 EvalRequest를 받진 않지만, 테스트/문서에서 공유되면 편하다.
class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class InitPayload(BaseModel):
    type: Literal["init"] = "init"
    text: str = Field(..., description="Task description")


class ObservationPayload(BaseModel):
    type: Literal["obs"] = "obs"
    step: int = Field(..., ge=0, description="Current step number")
    obs: str = Field(..., description="Base64 encoded image")


class AckPayload(BaseModel):
    type: Literal["ack"] = "ack"
    success: bool = False
    message: str = ""


class ActionPayload(BaseModel):
    type: Literal["action"] = "action"
    # MineStudio/MineRL 관례상 20개 버튼이 가장 흔함
    buttons: list[int] = Field(..., description="Button states")
    # yaw, pitch 2축
    camera: list[float] = Field(..., description="Camera movements")
