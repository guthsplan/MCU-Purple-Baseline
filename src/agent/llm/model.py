from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class LLMState:
    """
    Lightweight state for LLM-based agent.
    Stores conversation or step-level metadata only.
    """
    memory: Optional[List[Dict[str, Any]]] = None
    step: int = 0
    first: bool = True
