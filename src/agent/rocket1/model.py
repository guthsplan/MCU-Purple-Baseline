from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, model_validator


class ObsPayload(BaseModel):
    """Observation payload that Purple receives from Green.
    MUST contain:
      image: HxWx3 uint8 array-like
    """
    image: Any = Field(..., description="RGB image, typically HxWx3 uint8 array-like")
    segment: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional segmentation dict with keys obj_mask, obj_id",
    )

    # Some pipelines send additional fields; keep permissive
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_extra(cls, data: Any):
        if not isinstance(data, dict):
            return data
        known = {"image", "segment"}
        extra = {k: v for k, v in data.items() if k not in known}
        data = dict(data)
        data["extra"] = extra
        return data


class ActionPayload(BaseModel):
    """Action payload Purple returns to Green.
    MUST be:
      buttons: length 20, int {0,1}
      camera: length 2, float
    """
    buttons: List[int]
    camera: List[float]

    @model_validator(mode="after")
    def _validate_shapes(self):
        """
        Validate action shapes and types.
        """ 
        if len(self.buttons) != 20:
            raise ValueError("buttons must have length 20 (MineRL/VPT standard).")
        if any((b not in (0, 1)) for b in self.buttons):
            raise ValueError("buttons entries must be 0 or 1.")
        if len(self.camera) != 2:
            raise ValueError("camera must have length 2: [dx, dy].")
        # float coercion already done by pydantic, but enforce here
        self.camera = [float(self.camera[0]), float(self.camera[1])]
        return self


class StepResult(BaseModel):
    """ Result of a single step from Purple agent."""
    action: ActionPayload
    info: Dict[str, Any] = Field(default_factory=dict)


class SamplingMode(str):
    """Sampling mode for action selection."""
    pass


SamplingModeLiteral = Literal["deterministic", "stochastic"]
