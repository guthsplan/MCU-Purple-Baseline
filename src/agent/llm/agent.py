from __future__ import annotations
import json
from typing import Any, Dict, Tuple, Optional

from src.agent.base import BaseAgent
from .model import LLMState
from .prompt import build_prompt
from .client import OpenAIClient
from .action_map import intent_to_action
from .preprocess import preprocess_llm_obs


class LLMAgent(BaseAgent):
    """
    OpenAI GPT-based Purple baseline agent (experimental).
    """

    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        self.client = OpenAIClient()

    def reset(self) -> None:
        return

    def initial_state(self, task_text: Optional[str] = None) -> LLMState:
        return LLMState(
            memory=[],
            step=0,
            first=True,
            task_text=task_text,
        )

    def act(
        self,
        *,
        obs: Dict[str, Any],
        state: LLMState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], LLMState]:
        """
        LLM policy step.
        """

        # 1. preprocess observation
        llm_obs = preprocess_llm_obs(obs)

        # 2. build prompt using task_text from state
        messages = build_prompt(
            task_text=state.task_text,
            llm_obs=llm_obs,
            step=state.step,
            memory=state.memory,
        )

        # 3. query LLM
        raw = self.client.chat(messages)

        # 4. parse response
        try:
            parsed = json.loads(raw)
            intent = parsed.get("intent", "")
            camera = parsed.get("camera", {})
        except Exception:
            intent, camera = "", {}

        # 5. map intent → MineRL action
        action = intent_to_action(intent, camera)
        action = self.validate_action(action)

        # 6. update state
        new_state = LLMState(
            memory=state.memory + [parsed],
            step=state.step + 1,
            first=False,
            task_text=state.task_text,
        )

        return action, new_state
