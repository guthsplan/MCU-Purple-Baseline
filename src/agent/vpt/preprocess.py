# src/agent/vpt/preprocess.py
from __future__ import annotations
from typing import Any, Dict

import numpy as np
import torch


def build_vpt_input(obs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if "image" not in obs:
        raise KeyError("obs must contain 'image'")

    image = obs["image"]
    if not isinstance(image, np.ndarray):
        raise TypeError("obs['image'] must be np.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    
    image = torch.tensor(image, dtype=torch.uint8, device=device)
    if image.ndim == 3:
        image = image[None, None, ...]
    elif image.ndim == 4:
        image = image[None, ...]
    
    return {"image": image}
