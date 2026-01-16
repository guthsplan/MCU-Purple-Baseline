from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch


@dataclass
class Steve1State:
    """
    STEVE-1 per-context state.

    condition: output of model.prepare_condition(...)
    state_in: recurrent state returned by model.initial_state / get_steve_action
    """
    condition: Dict[str, Any]
    state_in: List[torch.Tensor]
    first: bool = True
