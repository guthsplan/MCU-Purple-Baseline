from typing import Any
import httpx


def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = frozenset(
        [
            "name",
            "description",
            "url",
            "version",
            "capabilities",
            "defaultInputModes",
            "defaultOutputModes",
            "skills",
        ]
    )

    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    if "url" in card_data and not (
        card_data["url"].startswith("http://") or card_data["url"].startswith("https://")
    ):
        errors.append("Field 'url' must be an absolute URL starting with http:// or https://.")

    if "capabilities" in card_data and not isinstance(card_data["capabilities"], dict):
        errors.append("Field 'capabilities' must be an object.")

    for field in ["defaultInputModes", "defaultOutputModes"]:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    if "skills" in card_data:
        if not isinstance(card_data["skills"], list):
            errors.append("Field 'skills' must be an array of AgentSkill objects.")
        elif not card_data["skills"]:
            errors.append("Field 'skills' array is empty. Agent must have at least one skill.")

    return errors


def test_agent_card(agent_url):
    resp = httpx.get(f"{agent_url}/.well-known/agent-card.json")
    assert resp.status_code == 200
    errors = validate_agent_card(resp.json())
    assert not errors, "\n".join(errors)
