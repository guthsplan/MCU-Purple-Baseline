# src/agent/rocket1/agent.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import torch

from minestudio.models.rocket_one.body import RocketPolicy

from src.agent.base import BaseAgent
from .preprocess import build_rocket_input
from .model import RocketState

logger = logging.getLogger("purple.rocket1")


class Rocket1Agent(BaseAgent):
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(device=device)

        # Load Rocket-1.12w_EMA model
        logger.info("Loading Rocket-1 model: CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
        self.model = RocketPolicy.from_pretrained(
            "CraftJarvis/MineStudio_ROCKET-1.12w_EMA"
        ).to(self.device)
        # Set model to eval mode
        self.model.eval()
        logger.info("Rocket1Agent loaded on device=%s", self.device)

    def initial_state(self, task_text: Optional[str] = None) -> RocketState:
        """
        Initialize state for a new task.
        
        task_text: Optional task description (ignored for Rocket-1, included for BaseAgent compatibility)
        """
        state_in = self.model.initial_state(batch_size=1)
        return RocketState(memory=state_in, first=True)

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: RocketState,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], RocketState]:
        
        input_dict = build_rocket_input(obs, self.device)
        
        action, new_state_in = self.model.get_action(
            input=input_dict,
            state_in=state.memory,
            deterministic=deterministic,
        )
        
        new_state = RocketState(memory=new_state_in, first=False)
        action["buttons"] = action["buttons"].cpu().numpy().tolist()
        action["camera"] = action["camera"].cpu().numpy().tolist()
        return action, new_state
    
