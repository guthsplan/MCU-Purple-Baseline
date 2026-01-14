# src/agent/noop.py
from __future__ import annotations

from typing import Any, Dict, Tuple
from src.agent.base import BaseAgent, AgentState


class NoOpAgent(BaseAgent):
    """
    No-op baseline agent.

    Returns None policy_out.
    ActionPipeline is responsible for producing MCU-safe noop.
    """

    def reset(self) -> None:
        return

    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Any, AgentState]:
        # ❗ action을 만들지 않는다
        # ❗ pipeline이 noop을 만든다
        return None, state
