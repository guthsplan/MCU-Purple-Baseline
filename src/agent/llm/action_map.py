from __future__ import annotations
from typing import Dict, Any, Iterable


# intent vocabulary 
MOVE_INTENTS = {
    "move_forward": ["forward"],
    "move_back": ["back"],
    "strafe_left": ["left"],
    "strafe_right": ["right"],
}

LOOK_INTENTS = {
    "turn_left": {"yaw": -5.0},
    "turn_right": {"yaw": 5.0},
    "look_up": {"pitch": -5.0},
    "look_down": {"pitch": 5.0},
}

ACTION_INTENTS = {
    "attack": ["attack"],
    "use": ["use"],
    "jump": ["jump"],
    "sneak": ["sneak"],
    "sprint": ["sprint"],
}

BUTTONS_ORDER: List[str] = [
    # movement
    "forward",
    "back",
    "left",
    "right",

    # camera-independent movement
    "jump",
    "sneak",
    "sprint",

    # interaction
    "attack",
    "use",
    "drop",

    # inventory / hotbar
    "inventory",
    "swapHands",

    # hotbar slots
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
]

def _init_buttons() -> Dict[str, int]:
    return {k: 0 for k in BUTTONS_ORDER}


def _apply_buttons(buttons: Dict[str, int], keys: Iterable[str]) -> None:
    for k in keys:
        if k in buttons:
            buttons[k] = 1


def intent_to_action(intent: str, camera: Dict[str, float]) -> Dict[str, Any]:
    """
    Translate LLM intent + camera hint into MineRL action.

    intent: str (single high-level intent)
    camera: dict (may contain yaw/pitch overrides)
    """
    buttons = _init_buttons()

    # movement
    if intent in MOVE_INTENTS:
        _apply_buttons(buttons, MOVE_INTENTS[intent])

    # action 
    elif intent in ACTION_INTENTS:
        _apply_buttons(buttons, ACTION_INTENTS[intent])

    # look intents
    yaw = 0.0
    pitch = 0.0
    if intent in LOOK_INTENTS:
        yaw += LOOK_INTENTS[intent].get("yaw", 0.0)
        pitch += LOOK_INTENTS[intent].get("pitch", 0.0)

    # camera override from LLM 
    if isinstance(camera, dict):
        yaw += float(camera.get("yaw", 0.0))
        pitch += float(camera.get("pitch", 0.0))

    # clamp camera 
    yaw = max(min(yaw, 10.0), -10.0)
    pitch = max(min(pitch, 10.0), -10.0)

    return {
        "buttons": [buttons[k] for k in BUTTONS_ORDER],
        "camera": [yaw, pitch],
    }
