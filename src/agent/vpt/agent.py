from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import logging

import numpy as np
import torch

from minestudio.models.vpt import VPTPolicy

from src.agent.base import BaseAgent  # 너희 프로젝트의 BaseAgent가 있다고 가정
from .model import VPTState
from .preprocess import build_vpt_obs

logger = logging.getLogger("purple.vpt")


class VPTAgent(BaseAgent):
    """
    VPT wrapper for Purple baseline.

    This implementation intentionally mirrors the *local* long-term script usage:
        action, memory = model.get_action(obs, memory, input_shape='*')

    We adapt Purple's obs dict (containing RGB image) into the obs format expected by VPTPolicy.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_id_rl: str = "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
        hf_id_fallback: str = "CraftJarvis/MineStudio_VPT.foundation_2x",
    ) -> None:
        super().__init__(device=device)

        self.hf_id_rl = hf_id_rl
        self.hf_id_fallback = hf_id_fallback

        # Resolve device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model with fallback
        self.model = self._load_model().to(self.device)
        self.model.eval()

        logger.info("VPTAgent loaded on device=%s", self.device)

    def _load_model(self) -> VPTPolicy:
        try:
            logger.info("Loading VPT RL model: %s", self.hf_id_rl)
            return VPTPolicy.from_pretrained(self.hf_id_rl)
        except Exception as e:
            logger.warning("Failed to load RL VPT model (%s): %s", self.hf_id_rl, e)
            logger.info("Loading VPT fallback model: %s", self.hf_id_fallback)
            return VPTPolicy.from_pretrained(self.hf_id_fallback)

    def reset(self) -> None:
        """
        VPT has no explicit reset required beyond clearing recurrent memory.
        Executor should create a fresh VPTState(memory=None, first=True) per context_id.
        """
        return

    def initial_state(self, task_text: Optional[str] = None) -> VPTState:
        return VPTState(memory=None, first=True)


    @torch.inference_mode()
    def act(
        self,
        *,
        obs: Dict[str, Any],
        state: Optional[VPTState] = None,
        deterministic: bool = True,
        input_shape: str = "*",
    ) -> Tuple[Dict[str, Any], VPTState]:
        """
        Compute action using MineStudio VPTPolicy.get_action.

        Args:
          obs: dict with at least {"image": HxWx3 uint8 RGB}
          state: VPTState holding recurrent memory (may be None)
          deterministic: unused by get_action; kept for API consistency
          input_shape: pass-through to MineStudio get_action (long-term script uses '*')

        Returns:
          action_dict: expected to contain {"buttons": (20,), "camera": (2,)} (tensor/np/list OK)
          new_state: updated VPTState(memory=new_memory)
        """
        if state is None:
            state = self.initial_state()

        vpt_obs = build_vpt_obs(obs)

        # MineStudio expects obs possibly on CPU; it handles conversion internally.
        # Keep memory as-is; do not cast unless required.
        action, new_memory = self.model.get_action(vpt_obs, state.memory, input_shape=input_shape)

        new_state = VPTState(memory=new_memory, first=False)
        return action, new_state
