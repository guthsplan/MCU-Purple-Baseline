from __future__ import annotations
from typing import Dict, Any


INTENT_SPEC = """
You must choose exactly ONE intent from the following list.

Movement intents:
- move_forward
- move_back
- strafe_left
- strafe_right

Look intents:
- turn_left
- turn_right
- look_up
- look_down

Action intents:
- attack
- use
- jump
- sneak
- sprint

Rules:
- Output JSON only
- Do not explain your reasoning
- intent must be one of the listed strings
- If uncertain, output: {"intent": "", "camera": {}}
- camera is optional
- camera may contain:
  - yaw: float (left negative, right positive)
  - pitch: float (up negative, down positive)
"""

def build_prompt(task_text: str, llm_obs: Dict[str, Any]) -> list[dict]:
    system = {
        "role": "system",
        "content": (
            "You are a Minecraft control policy.\n"
            "Your job is to select the NEXT action.\n\n"
            + INTENT_SPEC
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
