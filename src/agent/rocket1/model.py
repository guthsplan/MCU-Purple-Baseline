from dataclasses import dataclass
from typing import Any, Optional

from src.agent.base import AgentState


@dataclass
class RocketState(AgentState):
    """
    Rocket-1 specific state that extends AgentState.
    
    memory: Recurrent state from the Rocket-1 model
    first: Indicates if this is the first step in the episode
    """
    memory: Optional[Any] = None
    first: bool = True

