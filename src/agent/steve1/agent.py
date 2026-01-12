from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import torch

from minestudio.models.steve_one import SteveOnePolicy

from src.agent.base import BaseAgent
from .model import Steve1State
from .preprocess import build_steve1_obs

logger = logging.getLogger("purple.steve1")


class Steve1Agent(BaseAgent):
    """
    STEVE-1 wrapper for Purple baseline.

    Mirrors long-term script usage:
      condition = model.prepare_condition(...)
      state_in = model.initial_state(condition, 1)
      action, state_in = model.get_steve_action(condition, obs, state_in)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_id: str = "CraftJarvis/MineStudio_STEVE-1.official",
        cond_scale: float = 4.0,
    ) -> None:
        super().__init__(device=device)

        self.hf_id = hf_id
        self.cond_scale = cond_scale

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading STEVE-1 model: %s", hf_id)
        self.model = SteveOnePolicy.from_pretrained(hf_id).to(self.device)
        self.model.eval()

    def reset(self) -> None:
        # STEVE-1 has no global reset; per-context reset handled via state init
        return

    def initial_state(self, task_text: str) -> Steve1State:
        """
        Initialize condition + recurrent state for a new context.
        """
        condition = self.model.prepare_condition(
            {
                "cond_scale": self.cond_scale,
                "text": task_text,
            }
        )
        state_in = self.model.initial_state(condition, batch_size=1)
        return Steve1State(condition=condition, state_in=state_in, first=True)

    @torch.inference_mode()
    def act(
        self,
        *,
        obs: Dict[str, Any],
        state: Steve1State,
        deterministic: bool = True,
        input_shape: str = "*",
    ) -> Tuple[Dict[str, Any], Steve1State]:
        """
        Compute action using STEVE-1 policy.
        """
        if state.condition is None or state.state_in is None:
            raise RuntimeError("Steve1State is not initialized (missing condition/state_in)")

        steve_obs = build_steve1_obs(obs)

        action, new_state_in = self.model.get_steve_action(
            state.condition,
            steve_obs,
            state.state_in,
            input_shape=input_shape,
        )

        new_state = Steve1State(
            condition=state.condition,
            state_in=new_state_in,
            first=False,
        )
        return action, new_state
