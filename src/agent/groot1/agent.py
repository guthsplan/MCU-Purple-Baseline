from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import torch

from minestudio.models.groot_one.body import GrootPolicy

from src.agent.base import BaseAgent
from .preprocess import build_groot1_input
from .model import Groot1State

logger = logging.getLogger("purple.groot1")


class Groot1Agent(BaseAgent):
    """
    Groot1 agent wrapper for Purple baseline.
    
    Groot1 is a video-conditioned policy that can use reference videos
    for zero-shot task execution.
    """
    
    def __init__(
        self, 
        device: Optional[str] = None,
        ref_video_path: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(device=device)

        # Store reference video path for conditioning
        self.ref_video_path = ref_video_path

        # Load Groot-1 model from HuggingFace
        logger.info("Loading Groot-1 model: CraftJarvis/MineStudio_GROOT.18w_EMA")
        self.model = GrootPolicy.from_pretrained(
            "CraftJarvis/MineStudio_GROOT.18w_EMA"
        ).to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        logger.info("Groot1Agent loaded on device=%s", self.device)
        
        if self.ref_video_path:
            logger.info("Using reference video: %s", self.ref_video_path)

    def reset(self) -> None:
        """Reset agent state (no-op for Groot1)."""
        return

    def initial_state(self, task_text: Optional[str] = None) -> Groot1State:
        """
        Initialize state for a new task.
        
        Args:
            task_text: Optional task description (ignored for Groot1, 
                      conditioning is done via ref_video_path)
        
        Returns:
            Groot1State with initial memory
        """
        # Pass batch_size=1 to maintain batch dimension
        state_in = self.model.initial_state(batch_size=1)
        return Groot1State(memory=state_in, first=True)

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: Groot1State,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], Groot1State]:
        """
        Generate action from observation using Groot1 policy.
        
        Args:
            obs: Observation dict containing 'image' key
            state: Current Groot1State
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action dict, new state)
        """
        # Preprocess observation
        input_dict = build_groot1_input(
            obs, 
            self.device,
            ref_video_path=self.ref_video_path
        )
        
        # Get action from model
        # Groot1's get_action internally calls forward and samples actions
        action, new_state_in = self.model.get_action(
            input=input_dict,
            state_in=state.memory,
            deterministic=deterministic,
        )
        
        # Update state
        new_state = Groot1State(memory=new_state_in, first=False)
        
        # Convert tensors to lists for compatibility
        action["buttons"] = action["buttons"].cpu().numpy().tolist()
        action["camera"] = action["camera"].cpu().numpy().tolist()
        
        return action, new_state
