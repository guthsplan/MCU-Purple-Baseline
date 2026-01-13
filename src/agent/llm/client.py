from __future__ import annotations

import os
import time
from typing import List, Dict

from openai import OpenAI


class OpenAIClient:
    """
    Thin wrapper around OpenAI ChatCompletion API.

    Responsibility:
    - Send prompt messages to LLM
    - Retry on transient failures
    - Return RAW string content only

    Does NOT:
    - Parse JSON
    - Interpret intent
    - Handle actions or state
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 150,
        timeout: int = 20,
        retries: int = 3,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages: List[Dict]) -> str:
        """
        Send chat messages to OpenAI and return raw text response.

        Returns:
            str: model output (NOT parsed)
        """
        last_err: Exception | None = None

        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )

                content = resp.choices[0].message.content
                if not isinstance(content, str):
                    raise RuntimeError("LLM returned non-string content")

                return self._sanitize(content)

            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    break

        raise RuntimeError(f"OpenAI chat failed after {self.retries} retries") from last_err

    @staticmethod
    def _sanitize(text: str) -> str:
        """
        Remove common formatting wrappers (e.g. ```json ... ```).
        """
        text = text.strip()

        if text.startswith("```"):
            # Remove ```json or ``` wrappers
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        return text
