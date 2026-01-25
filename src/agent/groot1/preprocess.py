# src/agent/groot1/preprocess.py
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import cv2
import torch
import logging

logger = logging.getLogger("purple.groot1.preprocess")

# Groot1 model expects 224x224 images (same as reference video resolution)
GROOT_IMAGE_SIZE = 224


def build_groot1_input(
    obs: Dict[str, Any], 
    device: torch.device,
    ref_video_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert Purple-style obs dict to Groot1 compatible input.
    
    Groot1 expects:
    - image: (b, t, h, w, c) tensor with uint8 [0, 255], resized to 224x224
    - ref_video_path: optional path to reference video for conditioning
    
    The model's backbone (ViT/EfficientNet) requires 224x224 input size.
    The model's transforms will handle normalization internally.
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
    
    # Resize to 224x224 (required by backbone model)
    image = cv2.resize(
        image, 
        (GROOT_IMAGE_SIZE, GROOT_IMAGE_SIZE), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # Convert to torch tensor with uint8 dtype
    # Groot1 model will handle normalization via transforms
    image = torch.tensor(image, dtype=torch.uint8, device=device)
    
    # Add batch and time dimensions: (H, W, C) -> (1, 1, H, W, C)
    if image.ndim == 3:
        image = image[None, None, ...]
    elif image.ndim == 4:
        image = image[None, ...]
    
    groot_input = {"image": image}
    
    # Add reference video path if provided (for inference conditioning)
    if ref_video_path is not None:
        groot_input["ref_video_path"] = ref_video_path
    
    logger.debug(f"[BUILD_GROOT_INPUT] image shape: {image.shape}, dtype: {image.dtype}")
    
    return groot_input
