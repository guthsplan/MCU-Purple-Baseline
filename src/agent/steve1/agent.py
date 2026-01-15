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

    This implementation intentionally mirrors the *local* long-term script usage:
        action, state_out = model.get_steve_action(condition, obs, state_in, input_shape='*')
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_id: str = "CraftJarvis/MineStudio_STEVE-1.official",
        cond_scale: float = 4.0,
    ) -> None:
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(device=device)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hf_id = hf_id
        self.cond_scale = cond_scale


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
        state_in = self.model.initial_state(
            batch_size=1,
            condition=condition,
        )
        
        return Steve1State(
            condition=condition,
            state_in=state_in,
            first=True,
        )

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: Steve1State,
        deterministic: bool = True, # unused
    ) -> Tuple[Dict[str, Any], Steve1State]:

        steve_obs = build_steve1_obs(obs)

        input_dict = {
            "image": steve_obs["image"],
            "condition": state.condition,
        }

        action, new_state_in = self.model.get_action(
            input=input_dict,
            state_in=state.state_in,
            input_shape="*",
            deterministic=deterministic,
        )

        action["buttons"] = action["buttons"].cpu().numpy().tolist()
        action["camera"] = action["camera"].cpu().numpy().tolist()
        
        new_state = Steve1State(
            condition=state.condition,
            state_in=new_state_in,
            first=False,
        )
        return action, new_state
