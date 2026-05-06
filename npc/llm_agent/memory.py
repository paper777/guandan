from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol

from client.api import JsonObject
from npc.llm_agent.prompts import (
    MEMORY_RULE_CONTEXT,
    MEMORY_PLAYER_ANALYSIS_PROMPT,
    MEMORY_TECHNIQUE_COMPACTION_PROMPT,
    MEMORY_TECHNIQUE_SUMMARY_PROMPT,
)
from npc.llm_agent.storage import TECHNIQUE_CATEGORIES


class MemoryModelProvider(Protocol):
    def complete_memory(
        self,
        *,
        system_prompt: str,
        context: JsonObject,
        max_output_tokens: int | None = None,
    ) -> JsonObject:
        """Return strict JSON for a memory sub-agent task."""


class MemoryAgent:
    """Best-effort memory sub-agent for finished deals."""

    def __init__(
        self,
        provider: object,
        *,
        compaction_char_limit: int = 16000,
        max_output_tokens: int = 1200,
    ) -> None:
        self.provider = provider
        self.compaction_char_limit = compaction_char_limit
        self.max_output_tokens = max_output_tokens

    def process_deal(
        self,
        memory: JsonObject,
        *,
        recent_actions: list[JsonObject],
        events: list[JsonObject],
        players_by_seat: JsonObject,
        observer_name: str,
    ) -> None:
        deal_event = _deal_ended_event(events)
        if deal_event is None:
            return
        deal_seq = _event_seq(deal_event)
        if deal_seq is not None and memory.get("last_memory_deal_seq") == deal_seq:
            return

        self._summarize_deal(memory, recent_actions, events, players_by_seat, observer_name, deal_seq)
        self._analyze_players(memory, recent_actions, events, players_by_seat, observer_name)
        self._compact_if_needed(memory, recent_actions, events, players_by_seat, observer_name)
        if deal_seq is not None:
            memory["last_memory_deal_seq"] = deal_seq

    def _summarize_deal(
        self,
        memory: JsonObject,
        recent_actions: list[JsonObject],
        events: list[JsonObject],
        players_by_seat: JsonObject,
        observer_name: str,
        deal_seq: int | None,
    ) -> None:
        response = self._complete(
            MEMORY_TECHNIQUE_SUMMARY_PROMPT,
            {
                "rule_context": MEMORY_RULE_CONTEXT,
                "observer_name": observer_name,
                "players_by_seat": players_by_seat,
                "deal_events": events,
                "recent_actions": recent_actions,
                "existing_techniques": memory.get("techniques", {}),
            },
        )
        entry = _level1_entry(response, deal_seq)
        if not entry:
            return
        techniques = _techniques(memory)
        level1 = techniques["level1"]
        if isinstance(level1, list):
            level1.append(entry)

    def _analyze_players(
        self,
        memory: JsonObject,
        recent_actions: list[JsonObject],
        events: list[JsonObject],
        players_by_seat: JsonObject,
        observer_name: str,
    ) -> None:
        response = self._complete(
            MEMORY_PLAYER_ANALYSIS_PROMPT,
            {
                "rule_context": MEMORY_RULE_CONTEXT,
                "observer_name": observer_name,
                "players_by_seat": players_by_seat,
                "deal_events": events,
                "recent_actions": recent_actions,
                "existing_player_profiles": memory.get("player_profiles", {}),
            },
        )
        players = response.get("players") if isinstance(response, dict) else None
        if not isinstance(players, dict):
            return
        profiles = memory.get("player_profiles")
        if not isinstance(profiles, dict):
            profiles = {}
            memory["player_profiles"] = profiles
        for raw_name, raw_profile in players.items():
            name = str(raw_name).strip()
            if not name or not isinstance(raw_profile, dict):
                continue
            profile: JsonObject = dict(raw_profile)
            profile["updated_at"] = _utc_now()
            profiles[name] = profile

    def _compact_if_needed(
        self,
        memory: JsonObject,
        recent_actions: list[JsonObject],
        events: list[JsonObject],
        players_by_seat: JsonObject,
        observer_name: str,
    ) -> None:
        techniques = _techniques(memory)
        level1 = techniques.get("level1")
        if not isinstance(level1, list) or not level1:
            return
        size = len(json.dumps(level1, ensure_ascii=False, sort_keys=True))
        if size <= self.compaction_char_limit:
            return
        response = self._complete(
            MEMORY_TECHNIQUE_COMPACTION_PROMPT,
            {
                "rule_context": MEMORY_RULE_CONTEXT,
                "observer_name": observer_name,
                "players_by_seat": players_by_seat,
                "deal_events": events,
                "recent_actions": recent_actions,
                "level1": level1,
                "existing_level2": techniques.get("level2", {}),
            },
        )
        compacted = _level2_from_response(response)
        if not any(compacted.values()):
            return
        level2 = techniques.get("level2")
        if not isinstance(level2, dict):
            level2 = {}
            techniques["level2"] = level2
        for category in TECHNIQUE_CATEGORIES:
            level2[category] = _merge_texts(level2.get(category), compacted[category])
        techniques["level1"] = []

    def _complete(self, system_prompt: str, context: JsonObject) -> JsonObject:
        complete_memory = getattr(self.provider, "complete_memory", None)
        if not callable(complete_memory):
            return {}
        try:
            response = complete_memory(
                system_prompt=system_prompt,
                context=context,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception:
            return {}
        return response if isinstance(response, dict) else {}


def append_technique_updates(memory: JsonObject, updates: object) -> None:
    """Accept legacy/action-model technique notes as level-1 memory."""

    if not isinstance(updates, list):
        return
    level1 = _techniques(memory)["level1"]
    if not isinstance(level1, list):
        return
    for item in updates:
        text = str(item).strip()
        if text:
            level1.append({"summary": text, "techniques": [text], "source": "action_memory_update"})


def _techniques(memory: JsonObject) -> JsonObject:
    techniques = memory.get("techniques")
    if not isinstance(techniques, dict):
        techniques = {}
        memory["techniques"] = techniques
    level1 = techniques.get("level1")
    if not isinstance(level1, list):
        techniques["level1"] = []
    level2 = techniques.get("level2")
    if not isinstance(level2, dict):
        level2 = {}
        techniques["level2"] = level2
    for category in TECHNIQUE_CATEGORIES:
        if not isinstance(level2.get(category), list):
            level2[category] = []
    return techniques


def _level1_entry(response: JsonObject, deal_seq: int | None) -> JsonObject:
    summary = str(response.get("summary", "")).strip()
    techniques = _text_list(response.get("techniques"))
    if not summary and not techniques:
        return {}
    entry: JsonObject = {"created_at": _utc_now(), "source": "memory_agent"}
    if deal_seq is not None:
        entry["deal_seq"] = deal_seq
    if summary:
        entry["summary"] = summary
    if techniques:
        entry["techniques"] = techniques
    return entry


def _level2_from_response(response: JsonObject) -> dict[str, list[str]]:
    return {category: _text_list(response.get(category)) for category in TECHNIQUE_CATEGORIES}


def _merge_texts(existing: object, incoming: list[str], *, limit: int = 80) -> list[str]:
    merged = _text_list(existing)
    for item in incoming:
        if item not in merged:
            merged.append(item)
    return merged[-limit:]


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _deal_ended_event(events: list[JsonObject]) -> JsonObject | None:
    for event in events:
        if event.get("type") == "DealEnded":
            return event
    return None


def _event_seq(event: JsonObject) -> int | None:
    seq = event.get("seq")
    return seq if isinstance(seq, int) else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
