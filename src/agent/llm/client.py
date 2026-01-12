from __future__ import annotations
import os
import time
from openai import OpenAI

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict]) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
