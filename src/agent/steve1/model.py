from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Steve1State:
    """
    STEVE-1 per-context state.

    condition: output of model.prepare_condition(...)
    state_in: recurrent state returned by model.initial_state / get_steve_action
    """
    condition: Optional[Any]
    state_in: Optional[Any]
    first: bool = True
