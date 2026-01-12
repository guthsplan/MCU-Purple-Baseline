from __future__ import annotations
from typing import Dict, Any


def build_prompt(task_text: str, llm_obs: Dict[str, Any]) -> list[dict]:
    system = {
        "role": "system",
        "content": (
            "You are a Minecraft agent. "
            "Decide the next high-level intent based on the task and observation. "
            "Always respond in JSON with keys: intent, camera."
        ),
    }

    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Task: {task_text}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{llm_obs['image_b64']}"
                },
            },
        ],
    }

    return [system, user]
