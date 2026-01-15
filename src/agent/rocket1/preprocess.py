# src/agent/rocket1/preprocess.py
from typing import Dict, Any
import numpy as np
import torch

ROCKET_IMAGE_SIZE = 224


def build_rocket_input(obs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Simulator obs -> RocketPolicy input
    Zero-mask baseline (no segmentation in simulator)
    """

    img = obs["image"]  # HWC uint8

    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    # (T,H,W,3) -> (H,W,3), T=1
    if img.ndim == 4:
        img = img[0]

    import cv2
    img = cv2.resize(img, (ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE))
    img = np.ascontiguousarray(img)

    image_t = torch.from_numpy(img).to(device)
    image_t = image_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W,3)

    # zero segmentation
    obj_mask = torch.zeros(
        (1, 1, ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    obj_id = torch.full((1, 1), -1, dtype=torch.long, device=device)

    return {
        "image": image_t,
        "segment": {
            "obj_mask": obj_mask,
            "obj_id": obj_id,
        },
    }
