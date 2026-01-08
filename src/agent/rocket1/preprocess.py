# preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from einops import rearrange


@dataclass
class RocketInput:
    """Canonical input for MineStudio RocketPolicy.forward(input, memory)."""
    input_dict: Dict[str, Any]       # contains 'image' and 'segment'
    first: torch.Tensor              # (B, T) bool
    b: int
    t: int


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # list-like
    return np.asarray(x)


def _ensure_uint8_image(img: np.ndarray) -> np.ndarray:
    # Expect HxWx3, uint8
    if img.dtype != np.uint8:
        # if normalized float, rescale cautiously
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255)
        img = img.astype(np.uint8)
    return img


def build_rocket_input(
    obs: Dict[str, Any],
    device: torch.device,
    timesteps: int = 1,
) -> RocketInput:
    """Convert a single-step obs into RocketPolicy expected batch/time format.

    MineStudio RocketPolicy.forward expects:
      input['image']: (b, t, h, w, c) uint8/float tensor
      input['segment']['obj_mask']: (b, t, h, w)
      input['segment']['obj_id']: (b, t) long

    We produce b=1,t=1 for inference.
    """
    img = _ensure_uint8_image(_to_numpy(obs["image"]))  # H,W,3
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image shape HxWx3, got {img.shape}")

    h, w, _ = img.shape

    seg = obs.get("segment") or {}
    obj_mask = seg.get("obj_mask", None)
    obj_id = seg.get("obj_id", -1)

    if obj_mask is None:
        # RocketPolicy in MineStudio concatenates obj_mask as 4th channel;
        # if not available, provide zeros (safe default).
        obj_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        obj_mask = _to_numpy(obj_mask)
        if obj_mask.shape != (h, w):
            raise ValueError(f"obj_mask must be shape {(h,w)}, got {obj_mask.shape}")
        if obj_mask.dtype != np.uint8:
            # allow 0/1 float mask
            if obj_mask.max() <= 1.0:
                obj_mask = (obj_mask * 255.0).clip(0, 255)
            obj_mask = obj_mask.astype(np.uint8)

    # Build tensors with (b=1,t=1, ...)
    image_t = torch.from_numpy(img).to(device)                     # (H,W,3) uint8
    image_t = image_t.unsqueeze(0).unsqueeze(0)                    # (1,1,H,W,3)

    obj_mask_t = torch.from_numpy(obj_mask).to(device)             # (H,W)
    obj_mask_t = obj_mask_t.unsqueeze(0).unsqueeze(0)              # (1,1,H,W)

    obj_id_t = torch.tensor([[int(obj_id)]], device=device, dtype=torch.long)  # (1,1)

    input_dict = {
        "image": image_t,
        "segment": {
            "obj_mask": obj_mask_t,
            "obj_id": obj_id_t,
        },
    }

    first = torch.tensor([[False]], device=device, dtype=torch.bool)  # (1,1)

    return RocketInput(input_dict=input_dict, first=first, b=1, t=1)


def decode_rocket_action(
    rocket_action: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Decode MineStudio action head sample output into Purple->Green format.

    Expect rocket_action like:
      {
        "buttons": Tensor[b,t,20] or Tensor[b,t,?] (already correct),
        "camera": Tensor[b,t,2] or Tensor[b,t,1] depends on head; in MineStudio it is [dx,dy]
      }

    Output:
      {"buttons": List[int] len 20, "camera": [dx, dy]}
    """
    if "buttons" not in rocket_action or "camera" not in rocket_action:
        raise ValueError("rocket_action must contain keys 'buttons' and 'camera'.")

    buttons_t = rocket_action["buttons"]
    camera_t = rocket_action["camera"]

    # squeeze batch/time
    # Common shape: (1,1,20) and (1,1,2)
    buttons = buttons_t.detach().cpu()
    camera = camera_t.detach().cpu()

    # Handle possible extra dims (e.g. (1,1,20,1))
    buttons = buttons.reshape(-1)
    if buttons.numel() != 20:
        raise ValueError(f"Decoded buttons must have 20 elements, got {buttons.numel()}")

    # camera
    camera = camera.reshape(-1)
    if camera.numel() != 2:
        raise ValueError(f"Decoded camera must have 2 elements, got {camera.numel()}")

    buttons_list = [int(x != 0) for x in buttons.tolist()]
    dx, dy = float(camera[0].item()), float(camera[1].item())

    return {"buttons": buttons_list, "camera": [dx, dy]}
