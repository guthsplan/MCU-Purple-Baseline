from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VPTState:
    """
    VPT recurrent memory wrapper.

    MineStudio VPTPolicy.get_action(obs, memory, ...) returns (action, new_memory).
    memory can be None or a model-specific object. We store it verbatim.
    """
    memory: Optional[Any] = None
    first: bool = True
