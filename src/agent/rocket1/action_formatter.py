# src/agent/rocket1/action_formatter.py
"""
Format Rocket-1 agent actions for Green agent compatibility.
Handles conversion from raw agent output to Green-compatible ActionPayload.
"""
from __future__ import annotations

from typing import Any, Dict, List
import logging

logger = logging.getLogger("purple.rocket1.formatter")


def format_rocket_action(
    buttons: List[int],
    camera: List[float],
) -> Dict[str, Any]:
    """
    Format rocket-1 agent output to Green agent ActionPayload format.
    
    Rocket-1 outputs:
      - buttons: list of 20 binary values (0/1)
      - camera: list of 2 floats (pitch, yaw) in [-1, 1]
    
    Green agent ActionPayload expects:
      - action_type: "agent"
      - buttons: list of 20 ints (0/1) OR list of 1 int
      - camera: list of 2 floats OR list of 1 float
    
    Args:
        buttons: Button action list (20 elements)
        camera: Camera movement list (2 elements)
        
    Returns:
        Formatted action dict compatible with ActionPayload validation
    """
    
    # Validate inputs
    if not isinstance(buttons, list):
        logger.warning(f"buttons is not a list: {type(buttons)}, converting...")
        buttons = list(buttons)
    
    if not isinstance(camera, list):
        logger.warning(f"camera is not a list: {type(camera)}, converting...")
        camera = list(camera)
    
    # Ensure buttons has 20 elements
    if len(buttons) < 20:
        logger.warning(f"buttons has {len(buttons)} elements, padding to 20")
        buttons = buttons + [0] * (20 - len(buttons))
    elif len(buttons) > 20:
        logger.warning(f"buttons has {len(buttons)} elements, truncating to 20")
        buttons = buttons[:20]
    
    # Ensure all buttons are 0 or 1
    buttons = [1 if b else 0 for b in buttons]
    
    # Ensure camera has 2 elements
    if len(camera) < 2:
        logger.warning(f"camera has {len(camera)} elements, padding with 0.0")
        camera = camera + [0.0] * (2 - len(camera))
    elif len(camera) > 2:
        logger.warning(f"camera has {len(camera)} elements, truncating to 2")
        camera = camera[:2]
    
    # Ensure camera values are floats in [-1, 1]
    camera = [float(c) for c in camera]
    camera = [max(-1.0, min(1.0, c)) for c in camera]
    
    # ActionPayload format (agent action type)
    action_payload = {
        "type": "action",
        "action_type": "agent",
        "buttons": buttons,
        "camera": camera,
    }
    
    logger.debug(f"Formatted action: buttons len={len(buttons)}, camera={camera}")
    
    return action_payload
