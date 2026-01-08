# rocket1/agent.py
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from agent.base import BaseAgent, AgentState
from model import ActionPayload
from preprocess import build_rocket_input, decode_rocket_action

# MineStudio RocketPolicy loader (당신이 제공한 minestudio 코드 기준)
from minestudio.models.rocket_one.body import RocketPolicy as MineStudioRocketPolicy


class Rocket1Agent(BaseAgent):
    """Rocket-1 wrapper for Purple agent.

    - Loads MineStudio RocketPolicy from HF by default
    - Maintains recurrent memory per context session via AgentState
    - Outputs MineRL-standard {buttons[20], camera[2]}
    """

    def __init__(
        self,
        device: str | None = None,
        hf_id: str = "CraftJarvis/MineStudio_ROCKET-1.12w_EMA",
        deterministic_default: bool = True,
    ):
        super().__init__(device=device)
        self.hf_id = hf_id
        self.deterministic_default = deterministic_default

        # Load model
        self.model = MineStudioRocketPolicy.from_pretrained(self.hf_id).to(self.device)
        self.model.eval()

    def reset(self) -> None:
        # model 자체에 global cache를 두지 않는다는 전제.
        # session별 memory는 AgentState가 들고 있으므로 여기서는 no-op 가능.
        return

    @torch.inference_mode()
    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool | None = None,
    ) -> Tuple[Dict[str, Any], AgentState]:
        deterministic = self.deterministic_default if deterministic is None else deterministic

        rocket_in = build_rocket_input(obs, device=self.device)

        # First-flag: session 첫 스텝이면 True로 1-step만 주고 이후 False
        first = torch.tensor([[state.first]], device=self.device, dtype=torch.bool)
        rocket_in.first[:] = first  # (1,1)

        # Forward: MineStudio RocketPolicy returns (latents, memory)
        latents, new_memory = self.model(input=rocket_in.input_dict, memory=state.memory)

        # MineStudio MinePolicy 기반: sample_action 제공되는 경우가 많다.
        # 하지만 버전 차이가 있을 수 있으니, 가능한 경로를 모두 지원한다.
        if hasattr(self.model, "sample_action"):
            # MinePolicy 표준: pi_logits -> sample_action
            action_t = self.model.sample_action(
                latents["pi_logits"],
                deterministic=deterministic,
            )
        else:
            # fallback: pi_head가 있으면 직접 샘플링
            if not hasattr(self.model, "pi_head"):
                raise RuntimeError("RocketPolicy has no sample_action and no pi_head; cannot sample actions.")
            # pi_head.sample(logits, deterministic=...)
            pi_head = self.model.pi_head
            if not hasattr(pi_head, "sample"):
                raise RuntimeError("pi_head has no .sample(); cannot sample actions.")
            action_t = pi_head.sample(latents["pi_logits"], deterministic=deterministic)

        # action_t -> {buttons:[20], camera:[2]}
        action = decode_rocket_action(action_t)

        # Validate + coerce to canonical format
        action = self.validate_action(action)

        # Update state: after first step, first=False
        new_state = AgentState(memory=new_memory, first=False)

        # Optional: pydantic validation (hard fail if shape wrong)
        _ = ActionPayload(**action)

        return action, new_state
