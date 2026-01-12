from __future__ import annotations

from typing import Any, Dict
import numpy as np


def build_vpt_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Purple-style obs dict to a MineStudio VPT-compatible obs dict.

    Input expectation:
      obs["image"] is HxWx3 uint8 RGB

    Output:
      provide keys commonly used by MineStudio/MineRL wrappers:
        - "image"
        - "pov"
    """
    if not isinstance(obs, dict):
        raise TypeError(f"obs must be dict, got {type(obs)}")

    if "image" not in obs:
        raise KeyError("obs must contain 'image' key")

    image = obs["image"]
    if not isinstance(image, np.ndarray):
        raise TypeError(f"obs['image'] must be np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"obs['image'] must be HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)

    # Many MineStudio policies use 'pov' naming for the RGB frame.
    return {
        "image": image,
        "pov": image,
    }
