# src/agent/vpt/preprocess.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np


def build_vpt_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    image = obs["image"] # observation in dict format
    
    # Convert list to numpy array if needed
    if isinstance(image, list):
        image = np.asarray(image)

    # Ensure the image is of type uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    return {"image": image}
