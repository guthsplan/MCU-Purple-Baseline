from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import logging
import torch

from minestudio.models.rocket_one.body import RocketPolicy

from src.agent.base import BaseAgent, AgentState
from .preprocess import build_rocket_input, decode_rocket_action

logger = logging.getLogger("purple.rocket1")


class Rocket1Agent(BaseAgent):
    """
    Rocket-1 (MineStudio) wrapper for Purple baseline.

    Design:
      - B=1, T=1 only
      - Zero segmentation mask (online MCU baseline)
      - Per-context recurrent memory via AgentState
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_id: str = "CraftJarvis/MineStudio_ROCKET-1.12w_EMA",
        deterministic_default: bool = True,
    ):
        super().__init__(device=device)

        self.hf_id = hf_id
        self.deterministic_default = deterministic_default

        logger.info("Loading Rocket-1 model: %s", hf_id)
        self.model = RocketPolicy.from_pretrained(hf_id).to(self.device)
        self.model.eval()

    def reset(self) -> None:
        # no global reset; per-context state only
        return

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], AgentState]:

        if deterministic is None:
            deterministic = self.deterministic_default

        # 1) preprocess obs -> Rocket input
        rocket_in = build_rocket_input(obs, device=self.device)

        # 2) forward
        latents, new_memory = self.model(
            input=rocket_in,
            memory=state.memory,
        )

        # 3) sample action
        if hasattr(self.model, "sample_action"):
            action_tokens = self.model.sample_action(
                latents["pi_logits"],
                deterministic=deterministic,
            )
        else:
            action_tokens = self.model.pi_head.sample(
                latents["pi_logits"],
                deterministic=deterministic,
            )

        # 4) decode to MCU-safe action
        action = decode_rocket_action(action_tokens)

        new_state = AgentState(
            memory=new_memory,
            first=False,
        )

        return action, new_state
