from __future__ import annotations

from typing import Any, Dict
import numpy as np


def build_steve1_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Purple-style obs dict to STEVE-1 compatible obs.

    Input:
      obs["image"]: HxWx3 uint8 RGB
    """
    if "image" not in obs:
        raise KeyError("obs must contain 'image'")

    image = obs["image"]
    if not isinstance(image, np.ndarray):
        raise TypeError("obs['image'] must be np.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)

    return {
        "image": image,
        "pov": image,
    }
