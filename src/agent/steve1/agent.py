from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import base64
import logging
from dataclasses import dataclass

import numpy as np
import torch

from minestudio.models.steve_one import SteveOnePolicy

from src.agent.base import BaseAgent
from src.protocol.models import (
    InitPayload,
    ObservationPayload,
    AckPayload,
    ActionPayload,
)
from .model import Steve1State
from .preprocess import build_steve1_obs 

logger = logging.getLogger("purple.steve1")


# -------------------------
# Context store
# -------------------------
@dataclass
class _Context:
    task_text: str
    state: Steve1State


# -------------------------
# STEVE-1 Policy wrapper 
# -------------------------
class Steve1Agent(BaseAgent):
    """
    Docstring for Steve1Agent
    
    Responsibilities:
      - load model once
      - create initial (condition + recurrent state) from task text
      - step inference: (obs image) + (condition/state) -> action tokens + new state
    """
    def __init__(
        self,
        device: Optional[str] = None,
        hf_id: str = "CraftJarvis/MineStudio_STEVE-1.official",
        cond_scale: float = 4.0,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(device=device)
        # parameters
        self.hf_id = hf_id
        self.cond_scale = cond_scale
        # load model
        logger.info("Loading STEVE-1 model: %s", hf_id)
        self.model = SteveOnePolicy.from_pretrained(hf_id).to(self.device)
        self.model = self.model.float()  
        self.model.eval()

    def reset(self) -> None:
        return

    def initial_state(self, task_text: str) -> Steve1State:
    
        condition = self.model.prepare_condition(
            {"cond_scale": float(self.cond_scale), "text": task_text}
        )

        embed = condition.get("mineclip_embeds")
        if not isinstance(embed, torch.Tensor):
            raise RuntimeError("mineclip_embeds missing or not a torch.Tensor")

        if embed.ndim == 4:
            # (cfg, B, T, C)
            embed = embed[0]
            # 이제 (B, T, C) → (B, C)
            embed = embed[:, 0, :]
        elif embed.ndim == 3:
            # (B, T, C) → (B, C)
            embed = embed[:, 0, :]
        elif embed.ndim == 2:
            # already (B, C)
            pass
        else:
            raise RuntimeError(f"Unexpected mineclip_embeds shape: {tuple(embed.shape)}")

        if embed.ndim != 2:
            raise RuntimeError(f"mineclip_embeds MUST be 2D (B,C), got {tuple(embed.shape)}")

        condition["mineclip_embeds"] = embed

        batch_size = embed.shape[0]

        state_in = self.model.initial_state(
            batch_size=batch_size,
            condition=condition,
        )

        state_in = self.model.initial_state(
            batch_size=1,
            condition=condition,
        )

        return Steve1State(condition=condition, state_in=state_in, first=True)

    @torch.inference_mode()
    def act(
        self,
        obs: Dict[str, Any],
        state: Steve1State,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], Steve1State]:
        """
        Public entrypoint: calls internal impl.
        """
        return self._act_impl(obs=obs, state=state, deterministic=deterministic)

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: Steve1State,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], Steve1State]:
        """
        Step inference.

        Contract (our team decision):
          - return token format ONLY:
              buttons: [int]  (len=1)
              camera:  [int]  (len=1)
        """
        # preprocess obs
        steve_obs = build_steve1_obs(obs)
        image_bt = steve_obs["image"]  # HWC uint8 numpy

        # dtype 
        if image_bt.dtype == np.float64:
            image_bt = image_bt.astype(np.float32, copy=False)
        elif image_bt.dtype != np.uint8 and image_bt.dtype != np.float32:
            image_bt = image_bt.astype(np.uint8, copy=False)

        # input dict expected by SteveOnePolicy.get_action
        input_dict = {
            "image": image_bt,              # numpy HWC uint8
            "condition": state.condition,  # Dict[str, Any]
        }

        # step inference
        action, new_state_in = self.model.get_action(
            input=input_dict,
            state_in=state.state_in,
            deterministic=deterministic,
        )
        
        # postprocess action tokens 
        buttons = action.get("buttons")
        camera = action.get("camera")

        if buttons is None or camera is None:
            raise KeyError("Action must contain 'buttons' and 'camera'")

        # tensors -> numpy
        if isinstance(buttons, torch.Tensor):
            buttons = buttons.detach().cpu().numpy()
        if isinstance(camera, torch.Tensor):
            camera = camera.detach().cpu().numpy()

        # batch dim elimination
        if getattr(buttons, "ndim", 0) > 1 and buttons.shape[0] > 1:
            buttons = buttons[0]
        if getattr(camera, "ndim", 0) > 1 and camera.shape[0] > 1:
            camera = camera[0]

        # convert to int tokens
        buttons_token = _as_int_token(buttons)
        camera_token = _as_int_token(camera)

        action_dict = {"buttons": [buttons_token], "camera": [camera_token]}

        new_state = Steve1State(condition=state.condition, state_in=new_state_in, first=False)
        return action_dict, new_state


# -------------------------
# Protocol handler 
# -------------------------
class Steve1ProtocolHandler:
    """
    Handle init/obs messages for STEVE-1 agent protocol.    
    """

    def __init__(self, agent: Steve1Agent) -> None:
        self.agent = agent
        self._contexts: Dict[str, _Context] = {}

    def handle_init(self, context_id: str, payload: InitPayload) -> AckPayload:
        if not context_id:
            raise ValueError("context_id is required")
        task_text = payload.text
        state = self.agent.initial_state(task_text=task_text)
        self._contexts[context_id] = _Context(task_text=task_text, state=state)
        return AckPayload(success=True, message="initialized")

    def handle_obs(self, context_id: str, payload: ObservationPayload) -> ActionPayload:
        if not context_id:
            raise ValueError("context_id is required")
        ctx = self._contexts.get(context_id)
        if ctx is None:
            raise RuntimeError(f"Received obs before init for context_id={context_id}")

        image = _decode_base64_rgb(payload.obs)

        obs_dict = {"image": image, "step": payload.step}

        action_dict, new_state = self.agent.act(obs=obs_dict, state=ctx.state, deterministic=True)
        ctx.state = new_state

        return ActionPayload(
            action_type="agent",
            buttons=action_dict["buttons"],
            camera=action_dict["camera"],
        )

    def reset_context(self, context_id: str) -> None:
        self._contexts.pop(context_id, None)
    
    def has_context(self, context_id: str) -> bool:
        return context_id in self._contexts

# -------------------------
# Helpers
# -------------------------

def _as_int_token(x: Any) -> int:
    """
    Convert a model output into a single integer token.
    Accepts: int/float scalar, numpy scalar, list/tuple/np.ndarray with at least 1 element.
    """
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty token container")
        return int(x[0])

    if isinstance(x, np.ndarray):
        if x.size == 0:
            raise ValueError("Empty token ndarray")
        return int(x.reshape(-1)[0])

    # numpy scalar
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return int(x)

    # python scalar
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)

    raise TypeError(f"Cannot convert token from type={type(x)} value={x}")

def _decode_base64_rgb(b64_str: str) -> np.ndarray:
    """
    Decode base64-encoded RGB image to uint8 numpy array (HWC).
    """
    from PIL import Image
    import io

    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return arr
