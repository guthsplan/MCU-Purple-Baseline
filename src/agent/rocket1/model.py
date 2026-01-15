from __future__ import annotations
from typing import List
from pydantic import BaseModel, model_validator


class ActionPayload(BaseModel):
    buttons: List[int]
    camera: List[float]

    @model_validator(mode="after")
    def _validate(self):
        if len(self.buttons) != 20:
            raise ValueError("buttons must have length 20")
        if any(b not in (0, 1) for b in self.buttons):
            raise ValueError("buttons must be 0/1")
        if len(self.camera) != 2:
            raise ValueError("camera must have length 2")
        self.camera = [float(self.camera[0]), float(self.camera[1])]
        return self
