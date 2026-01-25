# src/agent/groot1/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Groot1State:
    """
    Groot1 recurrent memory wrapper.
    
    Groot1Policy.get_action(input, memory, ...) returns (action, new_memory).
    memory is a list of tensors representing the decoder's recurrent state.
    """
    memory: Optional[Any] = None
    first: bool = True
