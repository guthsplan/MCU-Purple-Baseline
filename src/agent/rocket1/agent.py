# src/agent/rocket1/agent.py
from typing import Dict, Any, Tuple
import torch

from minestudio.models.rocket_one.body import RocketPolicy

from .preprocess import build_rocket_input
from .model import RocketState


class Rocket1Agent:
    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = RocketPolicy.from_pretrained(
            "CraftJarvis/MineStudio_ROCKET-1.12w_EMA"
        ).to(self.device)
        self.model.eval()

    def initial_state(self) -> RocketState:
        return RocketState(memory=None, first=True)

    @torch.inference_mode()
    def act(
        self,
        obs: Dict[str, Any],
        state: RocketState,
        deterministic: bool = True,  # executor에서 넘김
    ) -> Tuple[Dict[str, Any], RocketState]:
        """
        Must return:
          action = {
              "buttons": List[int] (len=20),
              "camera":  List[float] (len=2),
          }
        """

        # obs to Rocket input
        rocket_input = build_rocket_input(obs, self.device)

        # forward
        latents, new_memory = self.model(
            input=rocket_input,
            memory=state.memory,
        )

        pi_logits = latents["pi_logits"]

        # logits to action
        # buttons
        btn = pi_logits["buttons"].detach().cpu().view(-1)
        buttons = (btn > 0).long().tolist()
        buttons = (buttons + [0] * 20)[:20]

        # camera
        cam = pi_logits["camera"].detach().cpu().view(-1)[:2]
        cam = torch.clamp(cam, -1.0, 1.0)
        camera = [float(cam[0]), float(cam[1])]

        action = {
            "buttons": buttons,
            "camera": camera,
        }

        # 4) update state
        new_state = RocketState(
            memory=new_memory,
            first=False,
        )

        return action, new_state
