from __future__ import annotations

from typing import Protocol

from client.api import ActionRequest, JsonObject
from npc.dummy_bot.policy import DummyBotPolicy


class LlmActionProvider(Protocol):
    def choose_action(self, prompt: JsonObject) -> JsonObject:
        """Return structured action JSON plus a concise thinking explanation."""


class DeterministicLlmProvider:
    """Dependency-free provider used for tests and local dry runs."""

    def __init__(self) -> None:
        self._fallback = DummyBotPolicy()

    def choose_action(self, prompt: JsonObject) -> JsonObject:
        request = ActionRequest(
            request_id=str(prompt.get("request_id", "")),
            prompt=_dict(prompt.get("prompt")),
            snapshot=_dict(prompt.get("snapshot")),
        )
        action = self._fallback.choose_action(request)
        thinking = _thinking_for(action, request)
        return {
            **action,
            "thinking": thinking,
            "memory_updates": {
                "skills": [_skill_for(action, request)],
            },
        }


def _thinking_for(action: JsonObject, request: ActionRequest) -> str:
    action_type = action.get("type")
    kind = request.prompt.get("kind")
    if action_type == "pass":
        return "Passing because the current prompt allows passing and preserving cards is safest."
    if action_type == "play_cards":
        return "Playing the lowest valid card set available for this prompt to keep stronger cards for later tricks."
    if action_type == "submit_tribute":
        return "Submitting the highest eligible tribute card because tribute rules require it."
    if action_type == "return_tribute":
        return "Returning a low eligible card to minimize the cost of the tribute exchange."
    return f"Using fallback behavior for prompt kind {kind}."


def _skill_for(action: JsonObject, request: ActionRequest) -> str:
    action_type = action.get("type")
    if action_type == "pass":
        return "Preserve stronger combinations when responding to another player's trick."
    if action_type == "play_cards":
        return "Lead cheaply when forced unless memory indicates an aggressive style is needed."
    if action_type == "submit_tribute":
        return "Tribute selection must exclude red-heart level wild cards."
    if action_type == "return_tribute":
        return "Return low cards during tribute to protect flexible combinations."
    return "Fallback to legal conservative play when uncertain."


def _dict(value: object) -> JsonObject:
    return value if isinstance(value, dict) else {}
