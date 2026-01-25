# src/agent/vpt/agent.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import logging
import numpy as np
import torch

from minestudio.models.vpt import VPTPolicy

from src.agent.base import BaseAgent
from .model import VPTState
from .preprocess import build_vpt_input
from src.action.action_space import build_vpt_action_space

logger = logging.getLogger("purple.vpt")


class VPTAgent(BaseAgent):
    """
    VPT wrapper for Purple baseline.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_id_rl: str = "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
        hf_id_fallback: str = "CraftJarvis/MineStudio_VPT.foundation_2x",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(device=device)

        self.hf_id_rl = hf_id_rl
        self.hf_id_fallback = hf_id_fallback

        self.action_space = build_vpt_action_space()

        self.model = self._load_model().to(self.device)
        self.model.eval()

        logger.info("VPTAgent loaded on device=%s", self.device)

    def _load_model(self) -> VPTPolicy:
        try:
            logger.info("Loading VPT RL model: %s", self.hf_id_rl)
            return VPTPolicy.from_pretrained(
                self.hf_id_rl,
                action_space=self.action_space,
            )
        except Exception as e:
            logger.warning("Failed to load RL VPT model (%s): %s", self.hf_id_rl, e)
            logger.info("Loading VPT fallback model: %s", self.hf_id_fallback)
            return VPTPolicy.from_pretrained(
                self.hf_id_fallback,
                action_space=self.action_space,
            )

    def reset(self) -> None:
        return

    def initial_state(self, task_text: Optional[str] = None) -> VPTState:
        state_in = self.model.initial_state(batch_size=1)
        return VPTState(memory=state_in, first=True)

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: VPTState,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], VPTState]:

        input_dict = build_vpt_input(obs, self.device) 

        action, new_state_in = self.model.get_action(
            input=input_dict,
            state_in=state.memory,
            deterministic=deterministic
        )

        new_state = VPTState(memory=new_state_in, first=False)
        action["buttons"] = action["buttons"].cpu().numpy().tolist()
        action["camera"] = action["camera"].cpu().numpy().tolist()
        return action, new_state
