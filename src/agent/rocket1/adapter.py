from typing import Dict, Any
import torch

def rocket_logits_to_action(
    pi_logits: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    RocketPolicy pi_logits -> single-step action
    Green / Simulator compatible
    """

    # buttons
    btn = pi_logits["buttons"].view(-1)
    buttons = (btn > 0).long().tolist()
    buttons = (buttons + [0]*20)[:20]

    # camera
    cam = pi_logits["camera"].view(-1)[:2]
    cam = torch.clamp(cam, -1.0, 1.0)
    camera = [float(cam[0]), float(cam[1])]

    return {
        "buttons": buttons,
        "camera": camera,
    }
