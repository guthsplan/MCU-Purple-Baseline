# src/agent/rocket1/agent.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import torch

from minestudio.models.rocket_one.body import RocketPolicy

from src.agent.base import BaseAgent
from .preprocess import build_rocket_input
from .model import RocketState

logger = logging.getLogger("purple.rocket1")


class Rocket1Agent(BaseAgent):
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(device=device)

        # Load Rocket-1.12w_EMA model
        logger.info("Loading Rocket-1 model: CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
        self.model = RocketPolicy.from_pretrained(
            "CraftJarvis/MineStudio_ROCKET-1.12w_EMA"
        ).to(self.device)
        # Set model to eval mode
        self.model.eval()
        logger.info("Rocket1Agent loaded on device=%s", self.device)

    def initial_state(self, task_text: Optional[str] = None) -> RocketState:
        """
        Initialize state for a new task.
        
        task_text: Optional task description (ignored for Rocket-1, included for BaseAgent compatibility)
        """
        return RocketState(memory=None, first=True)

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: RocketState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], RocketState]:
        """
        Core action generation logic.
        """
        try:
            logger.info(f"[ROCKET1_ACT] Starting _act_impl")
            logger.info(f"[ROCKET1_ACT] obs type: {type(obs)}")
            logger.info(f"[ROCKET1_ACT] obs keys: {list(obs.keys())}")
            if "image" in obs:
                logger.info(f"[ROCKET1_ACT] obs['image'] type: {type(obs['image'])}, isinstance(list)={isinstance(obs['image'], list)}")
            logger.info(f"[ROCKET1_ACT] state type: {type(state)}, state.first={state.first}, memory is None={state.memory is None}")
            
            from .input_validator import validate_rocket_input, validate_model_output
            
            # Preprocess observation to model input format
            logger.info(f"[ROCKET1_ACT] Calling build_rocket_input...")
            rocket_input = build_rocket_input(obs, self.device)
            logger.info(f"[ROCKET1_ACT] build_rocket_input returned successfully")
            logger.info(f"[ROCKET1_ACT] rocket_input keys: {list(rocket_input.keys())}")
            logger.info(f"[ROCKET1_ACT] rocket_input['image'] type: {type(rocket_input['image'])}, shape: {rocket_input['image'].shape if hasattr(rocket_input['image'], 'shape') else 'N/A'}")
            logger.info(f"[ROCKET1_ACT] rocket_input['segment'] type: {type(rocket_input['segment'])}")
            if isinstance(rocket_input['segment'], dict):
                logger.info(f"[ROCKET1_ACT] rocket_input['segment'] keys: {list(rocket_input['segment'].keys())}")
                for key in rocket_input['segment']:
                    val = rocket_input['segment'][key]
                    logger.info(f"[ROCKET1_ACT]   segment['{key}'] type: {type(val)}, isinstance Tensor: {isinstance(val, torch.Tensor)}")
                    if isinstance(val, torch.Tensor):
                        logger.info(f"[ROCKET1_ACT]   segment['{key}'] shape: {val.shape}, dtype: {val.dtype}")
            
            # Validate input before passing to model
            logger.info(f"[ROCKET1_ACT] Validating rocket_input...")
            validate_rocket_input(rocket_input, stage="agent-act-pre-model")
            logger.info(f"[ROCKET1_ACT] Input validation passed")
            
            # Forward pass through model
            logger.info(f"[ROCKET1_ACT] Calling model forward...")
            latents, new_memory = self.model(
                input=rocket_input,
                memory=state.memory,
            )
            logger.info(f"[ROCKET1_ACT] Model forward returned successfully")
            
            # Validate output from model
            logger.info(f"[ROCKET1_ACT] Validating model output...")
            validate_model_output(latents, new_memory, stage="agent-act-post-model")
            logger.info(f"[ROCKET1_ACT] Output validation passed")
            
            # Extract policy logits from output
            pi_logits = latents["pi_logits"]
            logger.info(f"[ROCKET1_ACT] pi_logits keys: {list(pi_logits.keys())}")
            
            # Decode logits to action
            logger.info(f"[ROCKET1_ACT] Decoding actions...")
            btn = pi_logits["buttons"].detach().cpu().view(-1)
            buttons = (btn > 0).long().tolist()
            buttons = (buttons + [0] * 20)[:20]
            
            cam = pi_logits["camera"].detach().cpu().view(-1)[:2]
            cam = torch.clamp(cam, -1.0, 1.0)
            camera = [float(cam[0]), float(cam[1])]
            
            action = {
                "buttons": buttons,
                "camera": camera,
            }
            logger.info(f"[ROCKET1_ACT] Action decoded: buttons len={len(buttons)}, camera={camera}")
            
            # Update state with new recurrent memory
            new_state = RocketState(
                memory=new_memory,
                first=False,
            )
            
            logger.info(f"[ROCKET1_ACT] SUCCESS - returning action and new_state")
            return action, new_state
            
        except Exception as e:
            logger.error(f"[ROCKET1_ACT] EXCEPTION: {type(e).__name__}: {str(e)}", exc_info=True)
            # Fallback to safe action on any error
            action = {
                "buttons": [0] * 20,
                "camera": [0.0, 0.0],
            }
            new_state = RocketState(memory=state.memory, first=False)
            logger.info(f"[ROCKET1_ACT] Returning fallback safe action due to exception")
            return action, new_state
