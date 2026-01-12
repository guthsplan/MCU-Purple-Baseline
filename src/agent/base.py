from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Button order as expected by MineRL/VPT standard
BUTTONS_ORDER: Tuple[str, ...] = (
    "attack",
    "back",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "use",
    "drop",
    "inventory",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
)

NUM_BUTTONS = len(BUTTONS_ORDER)


@dataclass
class AgentState:
    """Recurrent memory/state cache for a single context_id session."""
    memory: Optional[List[torch.Tensor]] = None
    first: bool = False  # 'first' flag for recurrent policies


class BaseAgent(ABC):
    """Purple-side agent contract.

    MUST output:
      {
        "buttons": List[int]  # len==20, 0/1
        "camera":  List[float]  # len==2 (dx, dy)
      }
    """

    def __init__(self, device: Optional[str] = None):
        self._device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    # Validate action dict
    def validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize and validate action dict structure and types.
        if "buttons" not in action or "camera" not in action:
            raise ValueError("Action must contain keys: 'buttons', 'camera'.")
    
        buttons = action["buttons"]
        camera = action["camera"]

        if not isinstance(buttons, (list, tuple)) or len(buttons) != NUM_BUTTONS:
            raise ValueError(f"'buttons' must be a list/tuple of length {NUM_BUTTONS}.")

        # normalize buttons to int 0/1
        norm_buttons: List[int] = []
        for i, b in enumerate(buttons):
            if isinstance(b, (bool, np.bool_)):
                b = int(b)
            if isinstance(b, (int, np.integer)):
                norm_buttons.append(1 if int(b) != 0 else 0)
            else:
                raise ValueError(f"buttons[{i}] must be int/bool, got {type(b)}")

        if not isinstance(camera, (list, tuple)) or len(camera) != 2:
            raise ValueError("'camera' must be a list/tuple of length 2: [dx, dy].")

        # normalize camera to float
        dx, dy = camera
        dx = float(dx)
        dy = float(dy)

        return {"buttons": norm_buttons, "camera": [dx, dy]}

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (used when a new context_id session starts)."""
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], AgentState]:
        """Compute action for the given observation and state."""
        raise NotImplementedError
