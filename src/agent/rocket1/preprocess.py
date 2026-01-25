# src/agent/rocket1/preprocess.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import cv2
import torch
import logging

logger = logging.getLogger("purple.rocket1.preprocess")
ROCKET_IMAGE_SIZE = 224

def build_rocket_input(obs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Convert Purple-style obs dict to Rocket-1 compatible obs.
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
    image = cv2.resize(image, (ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image /= 255.0
    
    if image.ndim == 3:
        image = image[None, None, ...]
    elif image.ndim == 4:
        image = image[None, ...]
    
    obj_mask = torch.zeros((1, 1, ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE), dtype=torch.float32, device=device)
    obj_id = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    rocket_input = {
        "image": image,
        "segment": {
            "obj_mask": obj_mask,
            "obj_id": obj_id,
        },
    }
    
    return rocket_input