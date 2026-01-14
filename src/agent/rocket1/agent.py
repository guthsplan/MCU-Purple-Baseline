from __future__ import annotations

from typing import Any, Dict, Tuple
import torch
import logging

from src.agent.base import BaseAgent, AgentState
from src.agent.rocket1.preprocess import build_rocket_input, decode_rocket_action
from src.agent.rocket1.model import ActionPayload  # pydantic validator
from minestudio.models.rocket_one.body import RocketPolicy as MineStudioRocketPolicy

logger = logging.getLogger("purple.rocket1")

# Rocket-1 wrapper for Purple agent.
class Rocket1Agent(BaseAgent):

    def __init__(
        self,
        device: str | None = None,
        hf_id: str = "CraftJarvis/MineStudio_ROCKET-1.12w_EMA",
        deterministic_default: bool = True,
        debug: bool = False,
    ):
        super().__init__(device=device)

        # Load parameters
        self.hf_id = hf_id
        self.deterministic_default = deterministic_default
        self.debug = debug

        # Load RocketPolicy model
        self.model = MineStudioRocketPolicy.from_pretrained(self.hf_id).to(self.device)
        self.model.eval()

        # Logging
        logger.info(
            "Rocket1Agent loaded: hf_id=%s device=%s deterministic_default=%s",
            hf_id,
            self.device,
            deterministic_default,
        )

    def reset(self) -> None:
        # State is handled externally (per context_id). Nothing global to reset.
        return

    # Forward pass through the model
    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], AgentState]:
        logger.debug("[Rocket1] step start, first=%s", state.first)
        if deterministic is None:
            deterministic = self.deterministic_default

        rocket_in = build_rocket_input(obs, device=self.device)
        rocket_in.first[:] = torch.tensor([[bool(state.first)]], device=self.device)
        rocket_in.input_dict["first"] = rocket_in.first

        latents, new_memory = self.model(
            input=rocket_in.input_dict,
            memory=state.memory,
        )

        if hasattr(self.model, "sample_action"):
            action_t = self.model.sample_action(
                latents["pi_logits"], deterministic=deterministic
            )
        else:
            action_t = self.model.pi_head.sample(
                latents["pi_logits"], deterministic=deterministic
            )

        action = decode_rocket_action(action_t)

        new_state = AgentState(
            memory=new_memory,
            first=False,
        )

        logger.debug(
            "[Rocket1] action buttons_sum=%d camera=%s",
            sum(action["buttons"]),
            action["camera"],
        )
        
        return action, new_state
