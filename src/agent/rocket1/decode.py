from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _squeeze_last(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    a = np.squeeze(a)
    if a.ndim == 2:
        a = a[-1]
    if a.ndim == 0:
        a = np.asarray([a])
    return a.reshape(-1)

def decode_rocket1_agent_action(
    raw_action: Dict[str, Any],
) -> Dict[str, Any]:
    """
    ActionPayload(agent) validator-compliant decode.
    """
    noop = {"buttons": [0], "camera": [0.0]}

    if not isinstance(raw_action, dict):
        return noop

    btn_key = next((k for k in ("buttons", "button", "btn") if k in raw_action), None)
    cam_key = next((k for k in ("camera", "cam", "mouse", "look") if k in raw_action), None)

    if btn_key is not None:
        b = _squeeze_last(_to_numpy(raw_action[btn_key]))
        if b.size >= 20:
            b = (b[:20] > 0).astype(int)
            buttons = b.tolist()
        else:
            buttons = [int(b[0] != 0)] if b.size > 0 else noop["buttons"]
    else:
        buttons = noop["buttons"]

    if cam_key is not None:
        c = _squeeze_last(_to_numpy(raw_action[cam_key]))
        camera = [float(c[0]), float(c[1])] if c.size >= 2 else [float(c[0])]
    else:
        camera = noop["camera"]

    return {"buttons": buttons, "camera": camera}
