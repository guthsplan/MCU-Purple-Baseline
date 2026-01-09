from __future__ import annotations

from typing import Any, Dict, Tuple
import torch

from src.agent.base import BaseAgent, AgentState
from src.agent.rocket1.preprocess import build_rocket_input, decode_rocket_action
from src.agent.rocket1.model import ActionPayload  # pydantic validator
from minestudio.models.rocket_one.body import RocketPolicy as MineStudioRocketPolicy

# Rocket-1 wrapper for Purple agent.
class Rocket1Agent(BaseAgent):

    def __init__(
        self,
        device: str | None = None,
        hf_id: str = "CraftJarvis/MineStudio_ROCKET-1.12w_EMA",
        deterministic_default: bool = True,
    ):
        super().__init__(device=device)
        # Load parameters
        self.hf_id = hf_id
        self.deterministic_default = deterministic_default
        # Load RocketPolicy model
        self.model = MineStudioRocketPolicy.from_pretrained(self.hf_id).to(self.device)
        self.model.eval()

    def reset(self) -> None:
        # State is handled externally (per context_id). Nothing global to reset.
        return

    # Forward pass through the model
    @torch.inference_mode()
    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool | None = None,
    ) -> Tuple[Dict[str, Any], AgentState]:
        deterministic = self.deterministic_default if deterministic is None else deterministic

        # Preprocess observation to RocketInput
        rocket_in = build_rocket_input(obs, device=self.device)

        # Override first flag from per-session state
        rocket_in.first[:] = torch.tensor([[bool(state.first)]], device=self.device, dtype=torch.bool)

        # Forward: returns (latents, memory)
        latents, new_memory = self.model(input=rocket_in.input_dict, memory=state.memory)

        # Sample action
        if hasattr(self.model, "sample_action"):
            action_t = self.model.sample_action(latents["pi_logits"], deterministic=deterministic)
        else:
            # fallback to pi_head.sample()
            if not hasattr(self.model, "pi_head") or not hasattr(self.model.pi_head, "sample"):
                raise RuntimeError("RocketPolicy cannot sample action (no sample_action / pi_head.sample).")
            action_t = self.model.pi_head.sample(latents["pi_logits"], deterministic=deterministic)

        # Decode Rocket action to MineRL action dict
        action = decode_rocket_action(action_t)

        # Validate + normalize (BaseAgent.validate_action should ensure lengths, types)
        action = self.validate_action(action)

        # Hard validate schema (will raise if wrong)
        _ = ActionPayload(**action)

        # Update state
        new_state = AgentState(memory=new_memory, first=False)
        return action, new_state
