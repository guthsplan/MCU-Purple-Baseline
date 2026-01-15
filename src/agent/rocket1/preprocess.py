from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch

ROCKET_IMAGE_SIZE = 224


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def build_rocket_input(
    obs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Build RocketPolicy-compatible input (B=1, T=1).
    """

    if "image" not in obs:
        raise KeyError("obs must contain 'image'")

    img = obs["image"]

    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)

    if img.ndim == 4:
        img = img[0]

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {img.shape}")

    img = _ensure_uint8(img)
    img = _resize(img, ROCKET_IMAGE_SIZE)

    # image: (1,1,H,W,3)
    image_t = torch.from_numpy(img).to(device)
    image_t = image_t.unsqueeze(0).unsqueeze(0)

    # zero segmentation
    obj_mask = torch.zeros(
        (1, 1, ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    obj_id = torch.zeros((1, 1), dtype=torch.long, device=device)

    return {
        "image": image_t,
        "segment": {
            "obj_mask": obj_mask,
            "obj_id": obj_id,
        },
    }


def decode_rocket_action(action_tokens: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Decode Rocket action to MCU-safe action.
    """

    noop = {
        "buttons": [0] * 20,
        "camera": [0.0, 0.0],
    }

    try:
        buttons = action_tokens.get("buttons")
        camera = action_tokens.get("camera")

        if buttons is None or camera is None:
            return noop

        buttons = buttons.detach().cpu().reshape(-1).tolist()
        camera = camera.detach().cpu().reshape(-1).tolist()

        if len(buttons) != 20 or len(camera) != 2:
            return noop

        buttons = [1 if b > 0 else 0 for b in buttons]
        camera = [float(camera[0]), float(camera[1])]

        return {"buttons": buttons, "camera": camera}

    except Exception:
        return noop
