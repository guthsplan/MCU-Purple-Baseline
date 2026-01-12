from __future__ import annotations
from typing import Dict, Any
from src.agent.base import BUTTONS_ORDER

def intent_to_action(intent: str, camera: Dict[str, float]) -> Dict[str, Any]:
    buttons = {k: 0 for k in BUTTONS_ORDER}

    if intent == "move_forward":
        buttons["forward"] = 1
    elif intent == "turn_left":
        buttons["left"] = 1
    elif intent == "turn_right":
        buttons["right"] = 1
    elif intent == "attack":
        buttons["attack"] = 1
    elif intent == "use":
        buttons["use"] = 1
    # fallback: no-op

    camera_dx = float(camera.get("yaw", 0.0))
    camera_dy = float(camera.get("pitch", 0.0))

    return {
        "buttons": [buttons[k] for k in BUTTONS_ORDER],
        "camera": [camera_dx, camera_dy],
    }
