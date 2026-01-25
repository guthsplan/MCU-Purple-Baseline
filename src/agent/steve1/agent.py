from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import base64
import logging
from dataclasses import dataclass

import numpy as np
import torch

from minestudio.models.steve_one import SteveOnePolicy

from src.agent.base import BaseAgent
from src.protocol.models import (
    InitPayload,
    ObservationPayload,
    AckPayload,
    ActionPayload,
)
from .model import Steve1State
from .preprocess import build_steve1_input 

logger = logging.getLogger("purple.steve1")


class Steve1Agent(BaseAgent):
    """
    Docstring for Steve1Agent
    
    Responsibilities:
      - load model once
      - create initial (condition + recurrent state) from task text
      - step inference: (obs image) + (condition/state) -> action tokens + new state
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
        # parameters
        self.hf_id = hf_id
        self.cond_scale = cond_scale
        # load model
        logger.info("Loading STEVE-1 model: %s", hf_id)
        self.model = SteveOnePolicy.from_pretrained(hf_id).to(self.device)
        self.model = self.model.float()  
        self.model.eval()

    def reset(self) -> None:
        return

    def initial_state(self, task_text: str) -> Steve1State:
    
        condition = self.model.prepare_condition(
            {"cond_scale": float(self.cond_scale), "text": task_text}
        )
        embeds = condition.get("mineclip_embeds")
        if not isinstance(embeds, torch.Tensor):
            raise RuntimeError("mineclip_embeds missing or not a tensor")

        batch_size = embeds.shape[0]

        state_in = self.model.initial_state(
            batch_size=batch_size,
            condition=condition,
        )

        return Steve1State(condition=condition, state_in=state_in, first=True)

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: Steve1State,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], Steve1State]:
        """
        Step inference.

        Contract (our team decision):
          - return token format ONLY:
              buttons: [int]  (len=1)
              camera:  [int]  (len=1)
        """
        # preprocess obs
        input_dict = build_steve1_input(obs, state.condition, self.device)
        
        # step inference
        action, new_state_in = self.model.get_action(
            input=input_dict,
            state_in=state.state_in,
            deterministic=deterministic,
        )
        
        new_state = Steve1State(condition=state.condition, state_in=new_state_in, first=False)
        action["buttons"] = action["buttons"].cpu().numpy().tolist()
        action["camera"] = action["camera"].cpu().numpy().tolist()
        return action, new_state

