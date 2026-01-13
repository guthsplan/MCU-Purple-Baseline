from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math 
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Button order as expected by MineRL/VPT standard
BUTTONS_ORDER: Tuple[str, ...] = (
    "attack","back","forward","jump","left","right","sneak","sprint","use",
    "drop","inventory",
    "hotbar.1","hotbar.2","hotbar.3","hotbar.4",
    "hotbar.5","hotbar.6","hotbar.7","hotbar.8","hotbar.9",
)

NUM_BUTTONS = len(BUTTONS_ORDER)

@dataclass
class AgentState:
    memory: Optional[Any] = None
    first: bool = False


class BaseAgent(ABC):
    """
    Baseline Purple Agent Contract.

    act() MUST:
      - never raise
      - always return a valid MineRL action dict
    """

    def __init__(self, device: Optional[str] = None):
        self._device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self) -> None:
        pass

    def initial_state(self, task_text: Optional[str] = None) -> AgentState:
        return AgentState(memory=None, first=True)

    # ---------- NOOP ----------

    def noop_action(self) -> Dict[str, Any]:
        return {
            "buttons": [0] * NUM_BUTTONS,
            "camera": [0.0, 0.0],
        }

    # ---------- SAFE VALIDATION ----------

    def validate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate action dict.

        Ensures:
          - buttons: List[int] of length 20 with values {0,1}
          - camera: List[float] of length 2 with finite values
        """
        if not isinstance(action, dict):
            raise ValueError(f"action must be dict, got {type(action)}")

        if "buttons" not in action or "camera" not in action:
            raise ValueError("action must contain 'buttons' and 'camera'")

        buttons = action["buttons"]
        camera = action["camera"]

        # ---- buttons ----
        if hasattr(buttons, "detach"):  # torch tensor
            buttons = buttons.detach().cpu().reshape(-1).tolist()
        elif isinstance(buttons, np.ndarray):
            buttons = buttons.reshape(-1).tolist()

        if not isinstance(buttons, list) or len(buttons) != NUM_BUTTONS:
            raise ValueError(f"'buttons' must be list of length {NUM_BUTTONS}")

        norm_buttons: List[int] = []
        for i, b in enumerate(buttons):
            if isinstance(b, (bool, np.bool_)):
                norm_buttons.append(1 if bool(b) else 0)
            elif isinstance(b, (int, np.integer)):
                norm_buttons.append(1 if int(b) != 0 else 0)
            elif isinstance(b, (float, np.floating)):
                bf = float(b)
                if not math.isfinite(bf):
                    norm_buttons.append(0)
                elif 0.0 <= bf <= 1.0:
                    norm_buttons.append(1 if bf >= 0.5 else 0)
                else:
                    norm_buttons.append(1 if bf > 0.0 else 0)
            else:
                raise ValueError(f"buttons[{i}] has unsupported type {type(b)}")

        # ---- camera ----
        if hasattr(camera, "detach"):
            camera = camera.detach().cpu().reshape(-1).tolist()
        elif isinstance(camera, np.ndarray):
            camera = camera.reshape(-1).tolist()

        if not isinstance(camera, list) or len(camera) != 2:
            raise ValueError("'camera' must be list of length 2")

        cam: List[float] = []
        for j, v in enumerate(camera):
            try:
                fv = float(v)
                cam.append(fv if math.isfinite(fv) else 0.0)
            except Exception:
                cam.append(0.0)

        return {
            "buttons": norm_buttons,
            "camera": cam,
        }

    # ---------- PUBLIC ENTRY ----------

    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], AgentState]:
        """
        FINAL public entrypoint called by Executor.

        This method:
          1) calls agent-specific _act_impl()
          2) validates & normalizes action
          3) guarantees MineRL contract
          4) catches ALL exceptions and falls back to noop
        """
        try:
            action, new_state = self._act_impl(
                obs=obs,
                state=state,
                deterministic=deterministic,
            )

            action = self.validate_action(action)
            return action, new_state

        except Exception as e:
            logger.exception("Agent.act() failed, falling back to noop: %s", e)

            # Absolute safety fallback
            fallback_action = {
                "buttons": [0] * NUM_BUTTONS,
                "camera": [0.0, 0.0],
            }
            fallback_state = AgentState(memory=None, first=False)

            return fallback_action, fallback_state

    # ---------- IMPLEMENTED BY POLICY ----------

    @abstractmethod
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], AgentState]:
        """
        Agent-specific policy implementation.

        MUST:
          - return (action_dict, new_state)
          - may raise exceptions (will be caught by act())
        """
        raise NotImplementedError

