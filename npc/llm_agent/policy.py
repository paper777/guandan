from __future__ import annotations

from pathlib import Path
from typing import Any

from client.api import ActionRequest, JsonObject
from npc.dummy_bot.policy import DummyBotPolicy
from npc.llm_agent.config import LlmAgentConfig
from npc.llm_agent.provider import DeterministicLlmProvider, LlmActionProvider
from npc.llm_agent.storage import JsonActionLog, JsonMemoryStore


class LlmAgentPolicy:
    """Broker-compatible LLM NPC policy with isolated filesystem memory."""

    def __init__(self, config: LlmAgentConfig | None = None, provider: LlmActionProvider | None = None) -> None:
        self.config = config or LlmAgentConfig()
        self.provider = provider or DeterministicLlmProvider()
        self._fallback = DummyBotPolicy()
        self._stores_by_seat: dict[str, tuple[JsonMemoryStore, JsonActionLog]] = {}

    def choose_action(self, request: ActionRequest) -> JsonObject:
        seat = _seat_from_request(request)
        memory_store, action_log = self._stores_for(seat)
        memory = memory_store.load()
        provider_prompt = self._build_provider_prompt(request, memory, action_log)

        provider_action = self._request_provider(provider_prompt)
        action = self._validated_action(provider_action, request)
        fallback_used = False
        if action is None:
            fallback_used = True
            action = self._fallback.choose_action(request)
            provider_action = {
                **action,
                "thinking": "Provider output was invalid for the current prompt, so conservative fallback play was used.",
            }

        thinking = str(provider_action.get("thinking") or "Selected a legal conservative action.")
        action_with_thinking = {**action, "thinking": thinking}
        action_log.append(
            {
                "kind": "decision",
                "request_id": request.request_id,
                "table_id": _snapshot_value(request, "table_id"),
                "seat": seat,
                "event_seq": _public_value(request, "event_seq"),
                "phase": _public_value(request, "phase"),
                "current_level": request.prompt.get("current_level") or _public_value(request, "current_level"),
                "legal_action": request.prompt.get("kind"),
                "snapshot": _safe_snapshot(request.snapshot),
                "selected_action": action,
                "thinking": thinking,
                "fallback_used": fallback_used,
            }
        )
        self._update_memory(memory_store, memory, provider_action, request=request)
        return action_with_thinking

    def observe_action(self, observation: JsonObject) -> None:
        """Record an action submitted by any broker-controlled player."""

        observer_seat = str(observation.get("observer_seat") or self.config.seat or "")
        memory_store, action_log = self._stores_for(observer_seat or None)
        memory = memory_store.load()
        action_log.append(
            {
                "kind": "observed_action",
                "table_id": observation.get("table_id"),
                "observer_seat": observer_seat or None,
                "actor_seat": observation.get("actor_seat"),
                "action": observation.get("action"),
                "response_events": observation.get("events", []),
                "event_seq": observation.get("event_seq"),
            }
        )
        self._update_memory(memory_store, memory, {}, events=_events_from_observation(observation))

    @property
    def storage_paths(self) -> dict[str, tuple[Path, Path]]:
        return {
            seat: (memory_store.path, action_log.path)
            for seat, (memory_store, action_log) in self._stores_by_seat.items()
        }

    def _stores_for(self, seat: str | None) -> tuple[JsonMemoryStore, JsonActionLog]:
        key = seat or self.config.seat or self.config.namespace_for(None)
        stores = self._stores_by_seat.get(key)
        if stores is not None:
            return stores
        display_name = self.config.display_name_for(seat)
        memory_store = JsonMemoryStore(
            self.config.resolved_memory_path(seat),
            player_name=display_name,
            seat=seat or self.config.seat,
        )
        action_log = JsonActionLog(self.config.resolved_action_log_path(seat))
        stores = (memory_store, action_log)
        self._stores_by_seat[key] = stores
        return stores

    def _build_provider_prompt(self, request: ActionRequest, memory: JsonObject, action_log: JsonActionLog) -> JsonObject:
        return {
            "request_id": request.request_id,
            "prompt": request.prompt,
            "snapshot": request.snapshot,
            "memory": memory,
            "recent_actions": action_log.recent(self.config.max_recent_actions),
            "model": {
                "provider": self.config.provider_name,
                "name": self.config.model_name,
                "temperature": self.config.temperature,
            },
        }

    def _request_provider(self, prompt: JsonObject) -> JsonObject:
        try:
            action = self.provider.choose_action(prompt)
        except Exception as exc:
            return {"type": "error", "message": str(exc)}
        return action if isinstance(action, dict) else {"type": "error", "message": "provider returned non-object"}

    def _validated_action(self, action: JsonObject, request: ActionRequest) -> JsonObject | None:
        action_type = action.get("type")
        prompt_kind = request.prompt.get("kind")
        hand = tuple(str(card_id) for card_id in request.snapshot.get("hand", []))

        if action_type == "pass":
            return {"type": "pass"} if prompt_kind == "play_or_pass" else None
        if action_type == "play_cards":
            if prompt_kind not in {"lead", "play_or_pass"}:
                return None
            card_ids = tuple(str(card_id) for card_id in action.get("card_ids", []))
            if not card_ids or len(set(card_ids)) != len(card_ids) or not set(card_ids).issubset(set(hand)):
                return None
            normalized: JsonObject = {"type": "play_cards", "card_ids": list(card_ids)}
            declared_type = action.get("declared_type")
            if declared_type is not None:
                normalized["declared_type"] = str(declared_type)
            return normalized
        if action_type == "submit_tribute":
            if prompt_kind != "tribute":
                return None
            card_id = str(action.get("card_id", ""))
            return {"type": "submit_tribute", "card_id": card_id} if card_id in hand else None
        if action_type == "return_tribute":
            if prompt_kind != "return_tribute":
                return None
            card_id = str(action.get("card_id", ""))
            return {"type": "return_tribute", "card_id": card_id} if card_id in hand else None
        return None

    def _update_memory(
        self,
        memory_store: JsonMemoryStore,
        memory: JsonObject,
        provider_action: JsonObject,
        *,
        request: ActionRequest | None = None,
        events: list[JsonObject] | None = None,
    ) -> None:
        updates = provider_action.get("memory_updates")
        if isinstance(updates, dict):
            play_style = updates.get("play_style")
            if isinstance(play_style, str) and play_style.strip():
                memory["play_style"] = play_style.strip()
            skills = updates.get("skills")
            if isinstance(skills, list):
                memory["skills"] = _merge_skills(memory.get("skills"), skills)
        if request is not None:
            memory["player_name"] = self.config.display_name_for(_seat_from_request(request))
        _apply_score_events(memory, events or [])
        memory_store.save(memory)


def _merge_skills(existing: object, incoming: list[object], *, limit: int = 30) -> list[str]:
    merged: list[str] = []
    for item in existing if isinstance(existing, list) else []:
        text = str(item).strip()
        if text and text not in merged:
            merged.append(text)
    for item in incoming:
        text = str(item).strip()
        if text and text not in merged:
            merged.append(text)
    return merged[-limit:]


def _apply_score_events(memory: JsonObject, events: list[JsonObject]) -> None:
    score = memory.get("score")
    if not isinstance(score, dict):
        score = {}
    for event in events:
        if event.get("type") == "DealEnded":
            score["deals_played"] = int(score.get("deals_played", 0)) + 1
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            finish_order = payload.get("finish_order", [])
            if isinstance(finish_order, list):
                score["last_finish_order"] = finish_order
            seat = memory.get("seat")
            winning_team = payload.get("winning_team")
            if isinstance(seat, str) and _team_for_seat(seat) == winning_team:
                score["wins"] = int(score.get("wins", 0)) + 1
        elif event.get("type") == "LevelAdvanced":
            levels = score.get("level_by_team")
            if not isinstance(levels, dict):
                levels = {}
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            team = payload.get("team")
            next_level = payload.get("next_level")
            if isinstance(team, str) and isinstance(next_level, str):
                levels[team] = next_level
            score["level_by_team"] = levels
    memory["score"] = score


def _team_for_seat(seat: str) -> str:
    return "EW" if seat in {"E", "W"} else "SN"


def _seat_from_request(request: ActionRequest) -> str | None:
    seat = request.snapshot.get("seat")
    return str(seat) if seat is not None else None


def _snapshot_value(request: ActionRequest, key: str) -> object:
    return request.snapshot.get(key)


def _public_value(request: ActionRequest, key: str) -> object:
    public = request.snapshot.get("public")
    return public.get(key) if isinstance(public, dict) else None


def _safe_snapshot(snapshot: JsonObject) -> JsonObject:
    public = snapshot.get("public")
    return {
        "table_id": snapshot.get("table_id"),
        "seat": snapshot.get("seat"),
        "hand": list(snapshot.get("hand", [])),
        "public": public if isinstance(public, dict) else {},
    }


def _events_from_observation(observation: JsonObject) -> list[JsonObject]:
    events = observation.get("events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]
