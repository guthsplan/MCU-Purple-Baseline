# src/agent/rocket1/preprocess.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch
import logging

logger = logging.getLogger("purple.rocket1.preprocess")

ROCKET_IMAGE_SIZE = 224


def build_rocket_input(obs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Purple obs -> RocketPolicy input.
    
    CRITICAL: This function MUST validate types at every step.
    Includes extensive logging to diagnose shape attribute errors.
    """
    
    from .input_validator import validate_rocket_input
    
    logger.info(f"[BUILD_ROCKET_INPUT] Starting")
    logger.info(f"[BUILD_ROCKET_INPUT] obs type: {type(obs)}, isinstance dict: {isinstance(obs, dict)}")
    logger.info(f"[BUILD_ROCKET_INPUT] obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
    
    if not isinstance(obs, dict):
        error_msg = f"[BUILD_ROCKET_INPUT] obs must be dict, got {type(obs)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    logger.info(f"[BUILD_ROCKET_INPUT] obs['image'] type: {type(obs['image'])}")
    
    # 1. Extract and validate image
    if "image" not in obs:
        raise KeyError("obs must contain 'image'")
    
    img = obs["image"]
    
    # CRITICAL: Log exact type before any operations
    logger.info(f"[BUILD_ROCKET_INPUT] img type={type(img)}, repr={repr(type(img))}")
    logger.info(f"[BUILD_ROCKET_INPUT] is list={isinstance(img, list)}, is ndarray={isinstance(img, np.ndarray)}")
    
    # Additional debug for list items
    if isinstance(img, list):
        logger.warning(f"[BUILD_ROCKET_INPUT] img IS A LIST with {len(img)} items")
        if len(img) > 0:
            logger.warning(f"[BUILD_ROCKET_INPUT]   First item type: {type(img[0])}")
            if isinstance(img[0], np.ndarray):
                logger.warning(f"[BUILD_ROCKET_INPUT]   First item shape: {img[0].shape}")
    
    # Convert to numpy array if needed
    if isinstance(img, list):
        logger.warning(f"[BUILD_ROCKET_INPUT] img is list, converting to ndarray")
        img = np.asarray(img, dtype=np.uint8)
        logger.info(f"[BUILD_ROCKET_INPUT] After list->ndarray: type={type(img)}, shape={img.shape}")
    elif isinstance(img, np.ndarray):
        logger.info(f"[BUILD_ROCKET_INPUT] img is ndarray, shape={img.shape}, dtype={img.dtype}")
        if img.dtype != np.uint8:
            logger.info(f"[BUILD_ROCKET_INPUT] Converting dtype {img.dtype} -> uint8")
            img = img.astype(np.uint8, copy=False)
    else:
        error_msg = f"[BUILD_ROCKET_INPUT] img type {type(img)} is not list or ndarray"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Double-check type after conversion
    if not isinstance(img, np.ndarray):
        raise TypeError(f"[BUILD_ROCKET_INPUT] After conversion, img is {type(img)}, not ndarray!")
    
    logger.info(f"[BUILD_ROCKET_INPUT] After type checking: shape={img.shape}, dtype={img.dtype}, ndim={img.ndim}")
    
    # 2. Handle temporal dimension: extract first frame if needed
    if img.ndim == 4:
        logger.info(f"[BUILD_ROCKET_INPUT] img.ndim==4, taking first frame from {img.shape}")
        if img.shape[0] == 1:
            img = img[0]
        else:
            logger.warning(f"[BUILD_ROCKET_INPUT] Multiple frames, taking frame 0")
            img = img[0]
    
    # Validate HxWx3 format
    logger.info(f"[BUILD_ROCKET_INPUT] Before spatial validation: ndim={img.ndim}, shape={img.shape}")
    if img.ndim != 3:
        raise ValueError(f"Expected ndim=3, got {img.ndim}")
    if img.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {img.shape[2]}")
    
    logger.info(f"[BUILD_ROCKET_INPUT] Spatial validation passed")
    
    # 3. Resize to model input size
    try:
        import cv2
        logger.info(f"[BUILD_ROCKET_INPUT] Resizing {img.shape} -> ({ROCKET_IMAGE_SIZE}, {ROCKET_IMAGE_SIZE})")
        img_resized = cv2.resize(img, (ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        logger.info(f"[BUILD_ROCKET_INPUT] Resize successful: {img_resized.shape}")
    except Exception as e:
        logger.error(f"[BUILD_ROCKET_INPUT] Resize failed: {e}, img type={type(img)}, shape={img.shape}")
        raise
    
    img_resized = np.ascontiguousarray(img_resized)
    logger.info(f"[BUILD_ROCKET_INPUT] After ascontiguousarray: {img_resized.shape}, {img_resized.dtype}")
    
    # 4. Ensure device is a torch.device
    if isinstance(device, str):
        device = torch.device(device)
    
    logger.info(f"[BUILD_ROCKET_INPUT] Target device: {device}")
    
    # Convert to torch tensor
    try:
        logger.info(f"[BUILD_ROCKET_INPUT] Converting numpy to torch...")
        image_t = torch.from_numpy(img_resized)
        logger.info(f"[BUILD_ROCKET_INPUT] After from_numpy: dtype={image_t.dtype}, shape={image_t.shape}")
        
        image_t = image_t.to(device=device, dtype=torch.float32)
        logger.info(f"[BUILD_ROCKET_INPUT] After .to(): dtype={image_t.dtype}, device={image_t.device}, shape={image_t.shape}")
    except Exception as e:
        logger.error(f"[BUILD_ROCKET_INPUT] Torch conversion failed: {e}")
        raise
    
    image_t = image_t / 255.0
    image_t = image_t.unsqueeze(0).unsqueeze(0)
    
    logger.info(f"[BUILD_ROCKET_INPUT] After unsqueeze: shape={image_t.shape}, dtype={image_t.dtype}")
    logger.info(f"[BUILD_ROCKET_INPUT] image_t is Tensor: {isinstance(image_t, torch.Tensor)}")
    
    # 5. Create segmentation masks
    obj_mask = torch.zeros(
        (1, 1, ROCKET_IMAGE_SIZE, ROCKET_IMAGE_SIZE),
        dtype=torch.float32,
        device=device,
    )
    
    obj_id = torch.full(
        (1, 1),
        -1,
        dtype=torch.long,
        device=device,
    )
    
    logger.info(f"[BUILD_ROCKET_INPUT] obj_mask type: {type(obj_mask)}, shape={obj_mask.shape}")
    logger.info(f"[BUILD_ROCKET_INPUT] obj_id type: {type(obj_id)}, shape={obj_id.shape}")
    
    rocket_input = {
        "image": image_t,
        "segment": {
            "obj_mask": obj_mask,
            "obj_id": obj_id,
        },
    }
    
    logger.info(f"[BUILD_ROCKET_INPUT] rocket_input created, running validation...")
    validate_rocket_input(rocket_input, stage="preprocess-output")
    
    logger.info(f"[BUILD_ROCKET_INPUT] SUCCESS")
    return rocket_input
