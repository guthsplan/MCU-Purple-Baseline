from __future__ import annotations

from typing import Any, Dict, Tuple

from src.agent.base import BaseAgent, AgentState, NUM_BUTTONS


class NoOpAgent(BaseAgent):
    """
    No-op baseline agent.

    Always returns zero actions:
      - buttons: all zeros
      - camera: [0.0, 0.0]

    This agent is intended only as a sanity-check / debug baseline.
    """

    def reset(self) -> None:
        """
        No internal state to reset.
        """
        return

    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], AgentState]:

        action = {
            "buttons": [0] * NUM_BUTTONS,
            "camera": [0.0, 0.0],
        }
        return action, AgentState(memory=None, first=False)