# src/agent/rocket1/input_validator.py
"""
Validation and diagnostic utilities for Rocket-1 input pipeline.
Ensures all data structures match RocketPolicy expectations exactly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
import logging

logger = logging.getLogger("purple.rocket1.validator")


def validate_rocket_input(rocket_input: Dict[str, Any], stage: str = "pre-model") -> bool:
    """
    Comprehensive validation of Rocket-1 input dictionary.
    
    Args:
        rocket_input: The input dictionary to validate
        stage: Context string for error messages
        
    Returns:
        True if all validations pass
        
    Raises:
        TypeError: If any component is not the expected type
        ValueError: If any component has unexpected shape or values
    """
    
    # Validate image
    if "image" not in rocket_input:
        raise KeyError(f"[{stage}] Missing 'image' key in input")
    
    image = rocket_input["image"]
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"[{stage}] image must be torch.Tensor, got {type(image).__name__}"
        )
    
    if image.ndim != 5:
        raise ValueError(
            f"[{stage}] image must be 5D (B, T, H, W, C), got ndim={image.ndim}, shape={image.shape}"
        )
    
    b, t, h, w, c = image.shape
    if c != 3:
        raise ValueError(
            f"[{stage}] image channels must be 3 (RGB), got {c}"
        )
    
    if image.dtype != torch.float32:
        raise TypeError(
            f"[{stage}] image dtype must be float32, got {image.dtype}"
        )
    
    if image.min() < 0 or image.max() > 1.0:
        raise ValueError(
            f"[{stage}] image values must be in [0, 1], got range [{image.min():.3f}, {image.max():.3f}]"
        )
    
    logger.debug(f"[{stage}] image: shape={image.shape}, dtype={image.dtype}, device={image.device}")
    
    # Validate segment/segmentation
    seg_key = "segment" if "segment" in rocket_input else "segmentation"
    if seg_key not in rocket_input:
        raise KeyError(f"[{stage}] Missing '{seg_key}' key in input")
    
    segment = rocket_input[seg_key]
    if not isinstance(segment, dict):
        raise TypeError(
            f"[{stage}] {seg_key} must be dict, got {type(segment).__name__}"
        )
    
    # Validate obj_mask
    if "obj_mask" not in segment:
        raise KeyError(f"[{stage}] Missing 'obj_mask' in {seg_key}")
    
    obj_mask = segment["obj_mask"]
    if not isinstance(obj_mask, torch.Tensor):
        raise TypeError(
            f"[{stage}] obj_mask must be torch.Tensor, got {type(obj_mask).__name__}"
        )
    
    if obj_mask.ndim != 4:
        raise ValueError(
            f"[{stage}] obj_mask must be 4D (B, T, H, W), got ndim={obj_mask.ndim}, shape={obj_mask.shape}"
        )
    
    if obj_mask.shape != (b, t, h, w):
        raise ValueError(
            f"[{stage}] obj_mask shape {obj_mask.shape} doesn't match image spatial dims (B={b}, T={t}, H={h}, W={w})"
        )
    
    if obj_mask.dtype != torch.float32:
        raise TypeError(
            f"[{stage}] obj_mask dtype must be float32, got {obj_mask.dtype}"
        )
    
    logger.debug(f"[{stage}] obj_mask: shape={obj_mask.shape}, dtype={obj_mask.dtype}, device={obj_mask.device}")
    
    # Validate obj_id
    if "obj_id" not in segment:
        raise KeyError(f"[{stage}] Missing 'obj_id' in {seg_key}")
    
    obj_id = segment["obj_id"]
    if not isinstance(obj_id, torch.Tensor):
        raise TypeError(
            f"[{stage}] obj_id must be torch.Tensor, got {type(obj_id).__name__}"
        )
    
    if obj_id.ndim != 2:
        raise ValueError(
            f"[{stage}] obj_id must be 2D (B, T), got ndim={obj_id.ndim}, shape={obj_id.shape}"
        )
    
    if obj_id.shape != (b, t):
        raise ValueError(
            f"[{stage}] obj_id shape {obj_id.shape} doesn't match batch and time dims (B={b}, T={t})"
        )
    
    if obj_id.dtype != torch.long:
        raise TypeError(
            f"[{stage}] obj_id dtype must be long, got {obj_id.dtype}"
        )
    
    # obj_id is used as: self.interaction(obj_id + 1), where interaction has 10 embeddings
    # So valid range after +1 is [0, 9], meaning obj_id should be in [-1, 8]
    min_val = obj_id.min().item()
    max_val = obj_id.max().item()
    if min_val < -1 or max_val > 8:
        logger.warning(
            f"[{stage}] obj_id range [{min_val}, {max_val}] may be out of embedding bounds [-1, 8]"
        )
    
    logger.debug(f"[{stage}] obj_id: shape={obj_id.shape}, dtype={obj_id.dtype}, device={obj_id.device}, range=[{min_val}, {max_val}]")
    
    # Verify all tensors are on same device
    devices = {
        "image": image.device,
        "obj_mask": obj_mask.device,
        "obj_id": obj_id.device,
    }
    if len(set(str(d) for d in devices.values())) > 1:
        logger.warning(f"[{stage}] Tensors on different devices: {devices}")
    
    logger.debug(f"[{stage}] All validations passed")
    return True


def validate_model_output(latents: Any, memory: Any, stage: str = "post-model") -> bool:
    """
    Validate RocketPolicy output format.
    
    Args:
        latents: The latents dict from model forward
        memory: The memory list from model forward
        stage: Context string for error messages
        
    Returns:
        True if validation passes
        
    Raises:
        TypeError/ValueError if output format is incorrect
    """
    
    if not isinstance(latents, dict):
        raise TypeError(
            f"[{stage}] latents must be dict, got {type(latents).__name__}"
        )
    
    if "pi_logits" not in latents:
        raise KeyError(
            f"[{stage}] Missing 'pi_logits' in latents dict. Keys: {list(latents.keys())}"
        )
    
    pi_logits = latents["pi_logits"]
    
    # pi_logits should be a dict with 'buttons' and 'camera' keys
    if not isinstance(pi_logits, dict):
        raise TypeError(
            f"[{stage}] pi_logits must be dict, got {type(pi_logits).__name__}"
        )
    
    if "buttons" not in pi_logits or "camera" not in pi_logits:
        raise KeyError(
            f"[{stage}] pi_logits missing required keys. Got: {list(pi_logits.keys())}"
        )
    
    buttons_logits = pi_logits["buttons"]
    camera_logits = pi_logits["camera"]
    
    if not isinstance(buttons_logits, torch.Tensor):
        raise TypeError(
            f"[{stage}] buttons logits must be torch.Tensor, got {type(buttons_logits).__name__}"
        )
    
    if not isinstance(camera_logits, torch.Tensor):
        raise TypeError(
            f"[{stage}] camera logits must be torch.Tensor, got {type(camera_logits).__name__}"
        )
    
    logger.debug(f"[{stage}] buttons_logits: shape={buttons_logits.shape}, dtype={buttons_logits.dtype}")
    logger.debug(f"[{stage}] camera_logits: shape={camera_logits.shape}, dtype={camera_logits.dtype}")
    
    # Memory should be a list of tensors
    if memory is not None and not isinstance(memory, list):
        raise TypeError(
            f"[{stage}] memory must be None or list of tensors, got {type(memory).__name__}"
        )
    
    if isinstance(memory, list):
        for i, mem_tensor in enumerate(memory):
            if not isinstance(mem_tensor, torch.Tensor):
                raise TypeError(
                    f"[{stage}] memory[{i}] must be torch.Tensor, got {type(mem_tensor).__name__}"
                )
    
    logger.debug(f"[{stage}] All output validations passed")
    return True
