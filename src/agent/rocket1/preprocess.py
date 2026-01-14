from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Rocket-1 (MineStudio RocketPolicy ViT backbone) expects 224x224
ROCKET_IMAGE_SIZE: int = 224

@dataclass
class RocketInput:
    """Canonical input for MineStudio RocketPolicy.forward(input, memory)."""
    input_dict: Dict[str, Any]       # contains 'image' and 'segment'
    first: torch.Tensor              # (B, T) bool
    b: int
    t: int

def _to_numpy(x: Any) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    # list-like
    return np.asarray(x)

def _ensure_hwc3(img: np.ndarray) -> np.ndarray:
    """Ensure image is HxWx3."""
    if img.ndim == 2:
        # grayscale -> 3ch
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={img.shape}")
    return img

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Convert to uint8 safely."""
    if img.dtype == np.uint8:
        return img
    # float in [0,1] or [0,255]
    if np.issubdtype(img.dtype, np.floating):
        vmax = float(np.nanmax(img)) if img.size else 0.0
        if vmax <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img
    # int types
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize HxWxC (or HxW) to size x size. Always returns same dtype as input."""
    h, w = img.shape[:2]
    if h == size and w == size:
        return img

    # OpenCV path (preferred)
    try:
        import cv2
        interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
        return cv2.resize(img, (size, size), interpolation=interp)
    except Exception:
        # Torch fallback
        x = torch.from_numpy(img)
        if x.ndim == 2:
            x = x[None, None].float()  # 1,1,H,W
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="nearest")
            out = x[0, 0].to(torch.uint8).cpu().numpy()
            return out
        else:
            x = x.permute(2, 0, 1)[None].float()  # 1,C,H,W
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            out = x[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            return out


def build_rocket_input(
    obs: Dict[str, Any],
    device: torch.device,
) -> RocketInput:
    """Convert single-step obs into RocketPolicy expected format (B=1, T=1) with 224x224 enforced."""
    if "image" not in obs:
        raise ValueError("obs must contain key 'image'")
    # Process image
    img = _to_numpy(obs["image"])
    img = _ensure_hwc3(img)
    img = _ensure_uint8(img)

    # HARD GUARANTEE: 224x224
    img = _resize_to_square(img, ROCKET_IMAGE_SIZE)
    img = _ensure_hwc3(img)  # re-check

    # Verify
    h, w, _ = img.shape
    if (h, w) != (ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE):
        raise RuntimeError(f"Resize failed: got {(h, w)} expected {(ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE)}")

    # Segment (optional). If not provided, safe zeros.
    seg = obs.get("segment") or {}
    obj_mask = seg.get("obj_mask", None)
    obj_id = seg.get("obj_id", -1)

    if obj_mask is None:
        obj_mask_np = np.zeros((ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE), dtype=np.uint8)
    else:
        obj_mask_np = _to_numpy(obj_mask)
        if obj_mask_np.ndim == 3 and obj_mask_np.shape[-1] == 1:
            obj_mask_np = obj_mask_np[:, :, 0]
        if obj_mask_np.ndim != 2:
            raise ValueError(f"obj_mask must be HxW, got shape={obj_mask_np.shape}")
        obj_mask_np = _ensure_uint8(obj_mask_np)
        obj_mask_np = _resize_to_square(obj_mask_np, ROCKET_IMAGE_SIZE)
        if obj_mask_np.shape != (ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE):
            raise RuntimeError(f"obj_mask resize failed: got {obj_mask_np.shape}")

    # Tensors (B=1, T=1)
    # image: (1,1,H,W,3) uint8
    image_t = torch.from_numpy(img).to(device=device)
    image_t = image_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W,3)

    # obj_mask: (1,1,H,W)
    obj_mask_t = torch.from_numpy(obj_mask_np).to(device=device)
    obj_mask_t = obj_mask_t.unsqueeze(0).unsqueeze(0)

    # obj_id: (1,1)
    obj_id_t = torch.tensor([[int(obj_id)]], device=device, dtype=torch.long)

    input_dict: Dict[str, Any] = {
        "image": image_t,
        "segment": {
            "obj_mask": obj_mask_t,
            "obj_id": obj_id_t,
        },
    }

    # first: (1,1) bool (will be overwritten by AgentState.first each step)
    first = torch.tensor([[False]], device=device, dtype=torch.bool)

    return RocketInput(input_dict=input_dict, first=first, b=1, t=1)


def decode_rocket_action(rocket_action: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Decode MineStudio Rocket action to MineRL buttons(20)/camera(2).

    IMPORTANT:
    - NEVER raise here.
    - Always return a valid action dict.
    """

    # Safe noop fallback
    noop = {
        "buttons": [0] * 20,
        "camera": [0.0, 0.0],
    }

    try:
        if "buttons" not in rocket_action or "camera" not in rocket_action:
            return noop

        buttons_t = rocket_action["buttons"].detach().cpu().reshape(-1)
        camera_t = rocket_action["camera"].detach().cpu().reshape(-1)

        # Shape guard
        if buttons_t.numel() != 20 or camera_t.numel() != 2:
            return noop

        buttons = [1 if float(x) > 0 else 0 for x in buttons_t]
        camera = [float(camera_t[0]), float(camera_t[1])]

        return {"buttons": buttons, "camera": camera}

    except Exception:
        # Absolute last safety net
        logger.error("[Rocket1 decode fallback] %s", e)
        return noop


