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

    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: LLMState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], LLMState]:

        llm_obs = preprocess_llm_obs(obs)

        messages = build_prompt(
            task_text=state.task_text,
            llm_obs=llm_obs,
            step=state.step,
            memory=state.memory,
        )

        raw = self.client.chat(messages)

        try:
            raw = self.client.chat(messages)
            parsed = json.loads(raw)
            intent = parsed["intent"]
            camera = parsed.get("camera", {})
            action = intent_to_action(intent, camera)
        except Exception:
            # Fallback: no-op action
            action = {
                "buttons": [0] * 20,
                "camera": [0.0, 0.0],
            }

        new_state = LLMState(
            memory=state.memory + [parsed],
            step=state.step + 1,
            first=False,
            task_text=state.task_text,
        )
        return action, new_state