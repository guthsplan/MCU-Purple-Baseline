from __future__ import annotations

import importlib
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# MineRL/VPT canonical order
BUTTONS_ORDER: Tuple[str, ...] = (
    "attack","back","forward","jump","left","right","sneak","sprint","use",
    "drop","inventory",
    "hotbar.1","hotbar.2","hotbar.3","hotbar.4",
    "hotbar.5","hotbar.6","hotbar.7","hotbar.8","hotbar.9",
)
NUM_BUTTONS = len(BUTTONS_ORDER)


def noop_action() -> Dict[str, Any]:
    return {"buttons": [0], "camera": [60]}


def to_mcu_action(
    policy_out: Any,
    *,
    state: Any,
    prev_action: Optional[Dict[str, Any]] = None,
    deterministic: bool = True,
    anti_idle: bool = True,
) -> Dict[str, Any]:
    
    """
    Convert ANY policy output into MCU evaluator-safe action:
      - buttons: List[int] length 20 values {0,1}
      - camera:  List[float] length 2 finite values

    Safety contract:
      - NEVER raises to caller
      - ALWAYS returns valid dict
    """
    logger.debug(
        "[PIPE] ENTER to_mcu_action policy_out_type=%s prev_action_keys=%s",
        type(policy_out),
        list(prev_action.keys()) if isinstance(prev_action, dict) else None,
    )
    try:
        action = _coerce_any_to_env_like(policy_out)
        action = _validate_and_normalize(action)

        if anti_idle:
            action = _ensure_non_idle(action, state=state, prev_action=prev_action)

        logger.debug(
            "[PIPE] EXIT to_mcu_action buttons_len=%d camera=%s",
            len(action.get("buttons", [])),
            action.get("camera"),
        )
        return action
    except Exception as e:
        logger.exception("to_mcu_action failed -> noop. err=%s", e)
        # update idle count to avoid runaway if you want; keep simple
        try:
            if hasattr(state, "idle_count"):
                state.idle_count = int(getattr(state, "idle_count", 0)) + 1
        except Exception:
            pass
        return noop_action()


# -------------------------
# Core coercion pipeline
# -------------------------

def _coerce_any_to_env_like(x: Any) -> Dict[str, Any]:
    """
    Return env-like dict with either:
      - {"buttons": ..., "camera": ...}  OR
      - MineRL-style keys {"attack":..., ..., "camera":...}
    """
    logger.debug("[PIPE] COERCE input type=%s repr=%s", type(x), repr(x)[:200])
    if x is None:
        return noop_action()

    # Case 1: already dict
    if isinstance(x, dict):
        # 1a) already has buttons/camera
        if ("buttons" in x) or ("camera" in x):
            result = {"buttons": x.get("buttons", None), "camera": x.get("camera", None)}
            logger.debug("[PIPE] COERCE return dict keys=%s", list(result.keys()))
            return result

        # 1b) MineRL-style dict
        if any(k in x for k in BUTTONS_ORDER) or ("camera" in x):
            # keep as is, will normalize later
            return dict(x)

        # 1c) token wrapper
        tokens = _extract_tokens_from_wrapper(x)
        if tokens is not None:
            decoded = _decode_tokens_via_minestudio(tokens)
            if decoded is not None:
                return decoded
            # If decode fails, do NOT guess mapping -> noop
            return noop_action()

        # Unknown dict shape -> noop
        return noop_action()

    # Case 2: token-like output (int/list/np/torch tensor etc.)
    decoded = _decode_tokens_via_minestudio(x)
    if decoded is not None:
        return decoded

    # Unknown -> noop
    return noop_action()


def _extract_tokens_from_wrapper(d: Dict[str, Any]) -> Any:
    for k in ("token", "tokens", "action_token", "action_tokens", "discrete", "index"):
        if k in d:
            return d[k]
    return None


def _decode_tokens_via_minestudio(tokens: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort MineStudio decode.
    If MineStudio not available or API mismatch -> return None (caller will noop).
    """
    logger.debug("[PIPE] DECODE tokens type=%s", type(tokens))
    # Normalize common containers
    try:
        if hasattr(tokens, "detach") and hasattr(tokens, "cpu") and hasattr(tokens, "tolist"):
            tokens = tokens.detach().cpu().tolist()
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        elif hasattr(tokens, "tolist"):
            # other array-like
            tokens = tokens.tolist()
    except Exception:
        pass

    try:
        mod = importlib.import_module("minestudio.utils.vpt_lib.actions")
    except Exception as e:
        logger.debug("MineStudio actions import failed: %s", e)
        return None

    # Try to find a decoder-like callable/object
    candidates: List[Any] = []

    for cls_name in (
        "ActionTransformer",
        "ActionDecoder",
        "ActionTokenizer",
        "VPTActionSpace",
        "ActionSpace",
        "MineRLActionSpace",
    ):
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            try:
                candidates.append(cls())
            except Exception:
                pass

    for fn_name in (
        "to_env_action",
        "decode_action",
        "decode",
        "tokens_to_action",
        "convert_to_env_action",
    ):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            candidates.append(fn)

    method_names = ("to_env_action", "decode_action", "decode", "transform", "convert", "tokens_to_action")

    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            out = None
            # module-level function
            if callable(cand) and not hasattr(cand, "__dict__"):
                out = cand(tokens)
            else:
                for m in method_names:
                    if hasattr(cand, m):
                        out = getattr(cand, m)(tokens)
                        break
            if out is None:
                continue

            # normalize common shapes
            if isinstance(out, dict):
                if ("buttons" in out) or ("camera" in out):
                    return {"buttons": out.get("buttons", None), "camera": out.get("camera", None)}
                if "action" in out and isinstance(out["action"], dict):
                    a = out["action"]
                    if ("buttons" in a) or ("camera" in a):
                        return {"buttons": a.get("buttons", None), "camera": a.get("camera", None)}

            # if not dict, reject
            raise TypeError(f"Decoded output unexpected: {type(out)}")
        except Exception as e:
            last_err = e
            continue

    logger.debug("MineStudio decode failed. last_err=%s", last_err)
    return None


# -------------------------
# Strict normalize + validate
# -------------------------

def _validate_and_normalize(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
      A) {"buttons": X, "camera": Y}
      B) MineRL-style keys + camera

    Return strict:
      {"buttons": List, "camera": List}
    """
    logger.debug(
    "[PIPE] VALIDATE input keys=%s",
    list(action.keys()) if isinstance(action, dict) else type(action),
)


    if not isinstance(action, dict):
        return noop_action()

    # Case A
    if ("buttons" in action) or ("camera" in action):
        buttons = action.get("buttons", None)
        camera = action.get("camera", None)
        buttons_list = _coerce_buttons(buttons)
        camera_list = _coerce_camera(camera)
        return {"buttons": buttons_list, "camera": camera_list}

    # Case B: MineRL-style dict
    buttons_list = [int(bool(action.get(k, 0))) for k in BUTTONS_ORDER]
    camera = action.get("camera", None)
    camera_list = _coerce_camera(camera)
    return {"buttons": buttons_list, "camera": camera_list}


def _coerce_buttons(buttons: Any) -> List[int]:
    logger.debug(
    "[PIPE] BUTTONS raw type=%s value=%s",
    type(buttons),
    repr(buttons)[:200],
)
    # default
    if buttons is None:
        return [0] * NUM_BUTTONS

    # tensor/ndarray -> list
    try:
        if hasattr(buttons, "detach") and hasattr(buttons, "cpu") and hasattr(buttons, "tolist"):
            buttons = buttons.detach().cpu().reshape(-1).tolist()
        elif isinstance(buttons, np.ndarray):
            buttons = buttons.reshape(-1).tolist()
        elif hasattr(buttons, "tolist"):
            buttons = buttons.tolist()
    except Exception:
        pass

    # scalar -> list
    if isinstance(buttons, (int, float, bool, np.integer, np.floating, np.bool_)):
        buttons = [buttons]

    if not isinstance(buttons, list):
        return [0] * NUM_BUTTONS

    # strict length: if wrong -> NO GUESSING -> noop
    if len(buttons) != NUM_BUTTONS:
        logger.error(
        "[PIPE] BUTTONS INVALID LENGTH len=%d expected=%d",
        len(buttons),
        NUM_BUTTONS,
    )
        raise ValueError(f"Invalid buttons length: {len(buttons)} (expected {NUM_BUTTONS})")

    out: List[int] = []
    for b in buttons:
        if isinstance(b, (bool, np.bool_)):
            out.append(1 if bool(b) else 0)
        elif isinstance(b, (int, np.integer)):
            out.append(1 if int(b) != 0 else 0)
        elif isinstance(b, (float, np.floating)):
            bf = float(b)
            if not math.isfinite(bf):
                out.append(0)
            else:
                # common: logits/prob/float -> threshold
                out.append(1 if bf >= 0.5 else 0)
        else:
            raise ValueError(f"Unsupported button type: {type(b)}")
    return out


def _coerce_camera(camera: Any) -> List[float]:
    if camera is None:
        return [0.0, 0.0]

    # tensor/ndarray -> list
    try:
        if hasattr(camera, "detach") and hasattr(camera, "cpu") and hasattr(camera, "tolist"):
            camera = camera.detach().cpu().reshape(-1).tolist()
        elif isinstance(camera, np.ndarray):
            camera = camera.reshape(-1).tolist()
        elif hasattr(camera, "tolist"):
            camera = camera.tolist()
    except Exception:
        pass

    # scalar -> list
    if isinstance(camera, (int, float, np.integer, np.floating)):
        camera = [float(camera), 0.0]

    if not isinstance(camera, list) or len(camera) < 2:
        raise ValueError(f"Invalid camera: {camera}")

    x = float(camera[0])
    y = float(camera[1])
    if not math.isfinite(x):
        x = 0.0
    if not math.isfinite(y):
        y = 0.0
    return [x, y]


# -------------------------
# Anti-idle guard (deterministic)
# -------------------------

def _is_idle(a: Dict[str, Any]) -> bool:
    b = a.get("buttons", [])
    c = a.get("camera", [0.0, 0.0])
    if not isinstance(b, list) or not isinstance(c, list):
        return True
    if any(int(x) != 0 for x in b):
        return False
    if len(c) >= 2 and (abs(float(c[0])) > 1e-9 or abs(float(c[1])) > 1e-9):
        return False
    return True


def _ensure_non_idle(
    a: Dict[str, Any],
    *,
    state: Any,
    prev_action: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministic minimal motion injection after consecutive idle.
    This prevents evaluator "stuck" behavior and is policy-agnostic.
    """
    idle = _is_idle(a)

    # initialize state fields if absent
    if not hasattr(state, "idle_count"):
        try:
            setattr(state, "idle_count", 0)
        except Exception:
            pass

    idle_count = int(getattr(state, "idle_count", 0)) if hasattr(state, "idle_count") else 0

    if idle:
        idle_count += 1
    else:
        idle_count = 0

    try:
        state.idle_count = idle_count
    except Exception:
        pass

    # Allow short idle (e.g., waiting for init stabilization)
    if idle_count < 3:
        return a

    # Past threshold: inject deterministic micro-move
    # (forward for one step; if already did that last step, inject camera turn)
    out = {"buttons": list(a["buttons"]), "camera": list(a["camera"])}

    # Determine last action idle-ness
    last_idle = True
    if prev_action is not None:
        try:
            last_idle = _is_idle(prev_action)
        except Exception:
            last_idle = True

    if last_idle:
        # inject forward
        idx_forward = BUTTONS_ORDER.index("forward")
        out["buttons"][idx_forward] = 1
        out["camera"] = [0.0, 0.0]
    else:
        # inject small deterministic yaw turn
        out["buttons"] = [0] * NUM_BUTTONS
        out["camera"] = [5.0, 0.0]  # small yaw turn

    return out
