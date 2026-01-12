from __future__ import annotations
from typing import Dict, Any
import base64
import cv2
import numpy as np


def preprocess_llm_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Purple canonical obs to LLM-friendly format.

    Input:
      obs["image"]: HxWx3 uint8 RGB numpy array

    Output:
      {
        "image_b64": str,
        "width": int,
        "height": int
      }
    """
    if "image" not in obs:
        raise KeyError("obs must contain 'image' key")

    image = obs["image"]

    if not isinstance(image, np.ndarray):
        raise TypeError(f"obs['image'] must be np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"obs['image'] must be HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)

    _, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )

    image_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "image_b64": image_b64,
        "width": image.shape[1],
        "height": image.shape[0],
    }
