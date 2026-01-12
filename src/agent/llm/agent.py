from __future__ import annotations
import json
from typing import Any, Dict, Tuple, Optional

from src.agent.base import BaseAgent, AgentState
from .model import LLMState
from .prompt import build_prompt
from .client import OpenAIClient
from .action_map import intent_to_action

class LLMAgent(BaseAgent):
    """
    OpenAI GPT-based Purple baseline agent.
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        self.client = OpenAIClient()

    def reset(self) -> None:
        return

    def initial_state(self) -> LLMState:
        return LLMState(memory=[], step=0, first=True)

    def act(
        self,
        *,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], AgentState]:

        if state is None or not isinstance(state, LLMState):
            state = self.initial_state()

        messages = build_prompt(
            task_text="",
            obs=obs,
        )

        raw = self.client.chat(messages)

        try:
            parsed = json.loads(raw)
            intent = parsed.get("intent", "")
            camera = parsed.get("camera", {})
        except Exception:
            intent, camera = "", {}

        action = intent_to_action(intent, camera)
        action = self.validate_action(action)

        new_state = LLMState(
            memory=state.memory,
            step=state.step + 1,
            first=False,
        )

        return action, new_state
