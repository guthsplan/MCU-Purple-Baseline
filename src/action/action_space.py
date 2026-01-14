# src/action/action_space.py
from __future__ import annotations

import gymnasium as gym
import numpy as np


def build_vpt_action_space() -> gym.spaces.Dict:
    """
    VPT action space.
    """
    return gym.spaces.Dict(
        {
            "camera": gym.spaces.MultiDiscrete([121]),
            "buttons": gym.spaces.MultiDiscrete([8641]),
        }
    )
