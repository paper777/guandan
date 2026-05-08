from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from client.types import ActionRequest, JsonObject
from db.player.types import Player
from common.log import deadline_fields, deadline_remaining_ms, elapsed_ms, trace_event
from npc.dummy_bot.player import DummyBotPlayer
from npc.llm_agent.actions import validate_action
from npc.llm_agent.config import LlmAgentConfig
from npc.llm_agent.context import (
    AgentRequestContext,
    model_snapshot,
    public_value,
    safe_snapshot,
    seat_from_request,
    snapshot_value,
    team_for_seat,
)
from npc.llm_agent.memory import MemoryAgent, append_technique_updates
from npc.llm_agent.personality import personality_context
from npc.llm_agent.prompts import SYSTEM_PROMPT
from npc.llm_agent.provider import LlmActionProvider, provider_from_config
from npc.llm_agent.strategy import build_strategy_context
from npc.llm_agent.storage import JsonActionLog, JsonMemoryStore


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Prepared inputs shared by model prompting, fallback, and logging."""

    request: ActionRequest
    request_context: AgentRequestContext
    memory_store: JsonMemoryStore
    action_log: JsonActionLog
    memory: JsonObject
    provider_prompt: JsonObject


class LlmAgentPlayer(Player):
    """Broker-compatible LLM NPC player with isolated filesystem memory."""

    def __init__(
        self,
        config: LlmAgentConfig | None = None,
        provider: LlmActionProvider | None = None,
        memory_agent: MemoryAgent | None = None,
    ) -> None:
        self.config = config or LlmAgentConfig()
        self.provider = provider or provider_from_config(self.config)
        self.memory_agent = memory_agent or MemoryAgent(
            self.provider,
            compaction_char_limit=self.config.memory_compaction_char_limit,
            max_output_tokens=self.config.memory_max_output_tokens,
        )
        self._fallback = DummyBotPlayer()
        self._stores_by_namespace: dict[str, tuple[JsonMemoryStore, JsonActionLog]] = {}

    def choose_action(self, request: ActionRequest) -> JsonObject:
        started = time.perf_counter()
        public = request.snapshot.get("public")
        public_snapshot = public if isinstance(public, dict) else {}
        deadline_epoch_ms = public_snapshot.get("action_deadline_epoch_ms")
        trace_event(
            "llm_player.decision_started",
            request_id=request.request_id,
            table_id=snapshot_value(request, "table_id"),
            seat=seat_from_request(request),
            player_name=self.config.display_name_for(seat_from_request(request)),
            provider=self.config.provider_name,
            model=self.config.model_name,
            legal_action=request.prompt.get("kind"),
            event_seq=public_value(request, "event_seq"),
            **deadline_fields(deadline_epoch_ms),
        )
        context = self._prepare_decision(request)
        trace_event(
            "llm_player.prompt_prepared",
            request_id=request.request_id,
            table_id=snapshot_value(request, "table_id"),
            seat=context.request_context.seat,
            duration_ms=elapsed_ms(started),
            deadline_remaining_ms=deadline_remaining_ms(deadline_epoch_ms),
            recent_action_count=len(context.provider_prompt.get("recent_actions", [])),
            hand_count=len(context.provider_prompt.get("snapshot", {}).get("hand", []))
            if isinstance(context.provider_prompt.get("snapshot"), dict)
            else None,
        )
        provider_action = self._request_provider(context.provider_prompt)
        action, fallback_used = self._select_action(provider_action, context)
        trace_event(
            "llm_player.action_selected",
            request_id=request.request_id,
            table_id=snapshot_value(request, "table_id"),
            seat=context.request_context.seat,
            duration_ms=elapsed_ms(started),
            deadline_remaining_ms=deadline_remaining_ms(deadline_epoch_ms),
            action=action,
            fallback_used=fallback_used,
            provider_result_type=provider_action.get("type"),
            provider_error=provider_action.get("message") if provider_action.get("type") == "error" else None,
        )

        self._record_decision(context, action, provider_action, fallback_used)
        self._update_memory(
            context.memory_store,
            context.memory,
            provider_action,
            request=request,
            action_log=context.action_log,
        )
        trace_event(
            "llm_player.decision_completed",
            request_id=request.request_id,
            table_id=snapshot_value(request, "table_id"),
            seat=context.request_context.seat,
            duration_ms=elapsed_ms(started),
            deadline_remaining_ms=deadline_remaining_ms(deadline_epoch_ms),
            fallback_used=fallback_used,
        )
        return action

    def _prepare_decision(self, request: ActionRequest) -> DecisionContext:
        request_context = AgentRequestContext.from_request(request)
        memory_store, action_log = self._stores_for(request_context.seat)
        memory = memory_store.load()
        personality = personality_context(self.config.personality)
        strategy_context = build_strategy_context(request_context)
        provider_prompt = self._build_provider_prompt(
            request,
            memory,
            action_log,
            table_context=request_context.table_context,
            strategy_context=strategy_context,
            personality=personality,
        )
        return DecisionContext(
            request=request,
            request_context=request_context,
            memory_store=memory_store,
            action_log=action_log,
            memory=memory,
            provider_prompt=provider_prompt,
        )

    def _select_action(self, provider_action: JsonObject, context: DecisionContext) -> tuple[JsonObject, bool]:
        model_action = validate_action(provider_action, context.request_context)
        if model_action is not None:
            return model_action, False

        fallback_action = self._fallback.choose_action(context.request)
        normalized_fallback = validate_action(fallback_action, context.request_context)
        if normalized_fallback is not None:
            fallback_action = normalized_fallback
        return fallback_action, True

    def _record_decision(
        self,
        context: DecisionContext,
        action: JsonObject,
        provider_action: JsonObject,
        fallback_used: bool,
    ) -> None:
        request = context.request
        seat = context.request_context.seat
        entry = {
            "kind": "decision",
            "request_id": request.request_id,
            "table_id": snapshot_value(request, "table_id"),
            "deal_id": public_value(request, "deal_id"),
            "seat": seat,
            "event_seq": public_value(request, "event_seq"),
            "phase": public_value(request, "phase"),
            "current_level": request.prompt.get("current_level") or public_value(request, "current_level"),
            "legal_action": request.prompt.get("kind"),
            "snapshot": safe_snapshot(request.snapshot),
            "selected_action": action,
            "fallback_used": fallback_used,
        }
        players_by_seat = _players_by_seat_from_request(request)
        if players_by_seat:
            entry["players_by_seat"] = players_by_seat
        llm_output = _llm_output_for_log(provider_action)
        if llm_output:
            entry["llm_output"] = llm_output
        if fallback_used:
            entry["fallback_reason"] = "provider output was invalid for the current prompt"
        context.action_log.append(entry)

    def observe_action(self, observation: JsonObject) -> None:
        """Record an action submitted by any broker-controlled player."""

        observer_seat = str(observation.get("observer_seat") or self.config.seat or "")
        memory_store, action_log = self._stores_for(observer_seat or None)
        memory = memory_store.load()
        if observer_seat:
            memory["seat"] = observer_seat
        observer_name = str(observation.get("observer_name") or self.config.display_name_for(observer_seat or None))
        action_log.append(
            {
                "kind": "observed_action",
                "table_id": observation.get("table_id"),
                "observer_seat": observer_seat or None,
                "observer_name": observer_name,
                "actor_seat": observation.get("actor_seat"),
                "actor_name": observation.get("actor_name"),
                "deal_id": observation.get("deal_id"),
                "players_by_seat": _players_by_seat_from_observation(observation),
                "action": observation.get("action"),
                "response_events": observation.get("events", []),
                "event_seq": observation.get("event_seq"),
            }
        )
        self._update_memory(
            memory_store,
            memory,
            {},
            events=_events_from_observation(observation),
            action_log=action_log,
            players_by_seat=_players_by_seat_from_observation(observation),
            observer_name=observer_name,
            deal_id=observation.get("deal_id"),
        )

    @property
    def storage_paths(self) -> dict[str, tuple[Path, Path]]:
        return {
            namespace: (memory_store.path, action_log.path)
            for namespace, (memory_store, action_log) in self._stores_by_namespace.items()
        }

    def _stores_for(self, seat: str | None) -> tuple[JsonMemoryStore, JsonActionLog]:
        key = self.config.namespace_for(seat)
        stores = self._stores_by_namespace.get(key)
        if stores is not None:
            return stores
        display_name = self.config.display_name_for(seat)
        memory_store = JsonMemoryStore(
            self.config.resolved_memory_path(seat),
            player_name=display_name,
            seat=seat or self.config.seat,
        )
        action_log = JsonActionLog(
            self.config.resolved_action_log_path(seat),
            max_entries=self.config.max_action_log_entries,
        )
        stores = (memory_store, action_log)
        self._stores_by_namespace[key] = stores
        return stores

    def _build_provider_prompt(
        self,
        request: ActionRequest,
        memory: JsonObject,
        action_log: JsonActionLog,
        *,
        table_context: JsonObject,
        strategy_context: JsonObject,
        personality: JsonObject,
    ) -> JsonObject:
        players_by_seat = _players_by_seat_from_request(request)
        user_name = _user_name(table_context, players_by_seat, memory)
        return {
            "request_id": request.request_id,
            "snapshot": model_snapshot(request.snapshot),
            "techniques": _memory_techniques(memory),
            "table_context": table_context,
            "strategy_context": strategy_context,
            "personality": personality,
            "players_by_seat": players_by_seat,
            "recent_actions": _recent_current_deal_actions(
                action_log,
                self.config.max_recent_actions,
                deal_id=public_value(request, "deal_id"),
            ),
            "system_prompt": SYSTEM_PROMPT,
            "player_profiles": _other_player_profiles(memory, table_context, players_by_seat, user_name),
            "model": {
                "provider": self.config.provider_name,
                "name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            },
        }

    def _request_provider(self, prompt: JsonObject) -> JsonObject:
        started = time.perf_counter()
        request_id = str(prompt.get("request_id") or "")
        table_context = prompt.get("table_context") if isinstance(prompt.get("table_context"), dict) else {}
        deadline_epoch_ms = table_context.get("action_deadline_epoch_ms")
        trace_event(
            "llm_player.provider_started",
            request_id=request_id,
            table_id=table_context.get("table_id"),
            seat=table_context.get("seat"),
            provider=self.config.provider_name,
            model=self.config.model_name,
            **deadline_fields(deadline_epoch_ms),
        )
        try:
            action = self.provider.choose_action(prompt)
        except Exception as exc:
            trace_event(
                "llm_player.provider_failed",
                request_id=request_id,
                table_id=table_context.get("table_id"),
                seat=table_context.get("seat"),
                provider=self.config.provider_name,
                model=self.config.model_name,
                duration_ms=elapsed_ms(started),
                deadline_remaining_ms=deadline_remaining_ms(deadline_epoch_ms),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            return {"type": "error", "message": str(exc)}
        trace_event(
            "llm_player.provider_completed",
            request_id=request_id,
            table_id=table_context.get("table_id"),
            seat=table_context.get("seat"),
            provider=self.config.provider_name,
            model=self.config.model_name,
            duration_ms=elapsed_ms(started),
            deadline_remaining_ms=deadline_remaining_ms(deadline_epoch_ms),
            response_type=action.get("type") if isinstance(action, dict) else type(action).__name__,
        )
        return action if isinstance(action, dict) else {"type": "error", "message": "provider returned non-object"}

    def _update_memory(
        self,
        memory_store: JsonMemoryStore,
        memory: JsonObject,
        provider_action: JsonObject,
        *,
        request: ActionRequest | None = None,
        events: list[JsonObject] | None = None,
        action_log: JsonActionLog | None = None,
        players_by_seat: JsonObject | None = None,
        observer_name: str | None = None,
        deal_id: object = None,
    ) -> None:
        updates = provider_action.get("memory_updates")
        if isinstance(updates, dict):
            play_style = updates.get("play_style")
            if isinstance(play_style, str) and play_style.strip():
                memory["play_style"] = play_style.strip()
            techniques = updates.get("techniques")
            if not isinstance(techniques, list):
                techniques = updates.get("skills")
            append_technique_updates(memory, techniques)
        if request is not None:
            seat = seat_from_request(request)
            memory["player_name"] = self.config.display_name_for(seat)
            memory["seat"] = seat
            players_by_seat = _players_by_seat_from_request(request)
            observer_name = self.config.display_name_for(seat)
        _apply_score_events(memory, events or [])
        if events and action_log is not None:
            self.memory_agent.process_deal(
                memory,
                recent_actions=_recent_current_deal_entries(
                    action_log,
                    self.config.memory_recent_deal_scan_limit,
                    deal_id=deal_id,
                ),
                events=events,
                players_by_seat=players_by_seat or {},
                observer_name=observer_name or str(memory.get("player_name") or self.config.display_name_for(None)),
            )
        memory_store.save(memory)


def _llm_output_for_log(provider_action: JsonObject) -> JsonObject:
    output: JsonObject = {}
    thinking = provider_action.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        output["thinking"] = thinking.strip()
    role = provider_action.get("role")
    if isinstance(role, str) and role.strip():
        output["role"] = role.strip()
    candidates = provider_action.get("candidates")
    if isinstance(candidates, list):
        output["candidates"] = candidates
    recommended_action = provider_action.get("recommended_action")
    if isinstance(recommended_action, dict):
        output["recommended_action"] = recommended_action
    message = provider_action.get("message")
    if provider_action.get("type") == "error" and isinstance(message, str) and message.strip():
        output["provider_error"] = message.strip()
    return output


def _memory_techniques(memory: JsonObject) -> JsonObject:
    techniques = memory.get("techniques")
    return techniques if isinstance(techniques, dict) else {}


def _other_player_profiles(
    memory: JsonObject,
    table_context: JsonObject,
    players_by_seat: JsonObject,
    user_name: str | None,
) -> JsonObject:
    profiles = memory.get("player_profiles")
    if not isinstance(profiles, dict):
        return {}
    seat = str(table_context.get("seat") or "")
    current_other_names = {
        str(name)
        for raw_seat, name in players_by_seat.items()
        if str(raw_seat) != seat and str(name).strip()
    }
    if current_other_names:
        return {
            name: profiles[name]
            for name in current_other_names
            if name in profiles and isinstance(profiles[name], dict)
        }
    return {
        str(name): profile
        for name, profile in profiles.items()
        if str(name) != (user_name or "") and isinstance(profile, dict)
    }


def _user_name(table_context: JsonObject, players_by_seat: JsonObject, memory: JsonObject) -> str | None:
    seat = table_context.get("seat")
    if seat is not None:
        name = players_by_seat.get(str(seat))
        if name is not None:
            return str(name)
    name = memory.get("player_name")
    return str(name) if name is not None else None


def _recent_current_deal_actions(action_log: JsonActionLog, limit: int, *, deal_id: object = None) -> list[JsonObject]:
    return [_action_summary(entry) for entry in _recent_current_deal_entries(action_log, limit, deal_id=deal_id)]


def _recent_current_deal_entries(action_log: JsonActionLog, limit: int, *, deal_id: object = None) -> list[JsonObject]:
    if limit <= 0:
        return []
    entries = action_log.load()
    if deal_id is not None:
        current_deal_entries = [entry for entry in entries if entry.get("deal_id") == deal_id]
    else:
        start = 0
        for index, entry in enumerate(entries):
            if _has_deal_boundary(entry):
                start = index + 1
        current_deal_entries = entries[start:]
    return current_deal_entries[-limit:]


def _action_summary(entry: JsonObject) -> JsonObject:
    action = entry.get("action") if entry.get("kind") == "observed_action" else entry.get("selected_action")
    actor_seat = entry.get("actor_seat") or entry.get("seat")
    actor_name = entry.get("actor_name")
    if actor_name is None:
        players_by_seat = entry.get("players_by_seat")
        if isinstance(players_by_seat, dict) and actor_seat is not None:
            actor_name = players_by_seat.get(str(actor_seat))
    summary: JsonObject = {
        "actor_seat": actor_seat,
        "action": action if isinstance(action, dict) else None,
    }
    if actor_name is not None:
        summary["actor_name"] = actor_name
    legal_action = entry.get("legal_action")
    if legal_action is not None:
        summary["legal_action"] = legal_action
    fallback_used = entry.get("fallback_used")
    if fallback_used is not None:
        summary["fallback_used"] = bool(fallback_used)
    return {key: value for key, value in summary.items() if value is not None}


def _has_deal_boundary(entry: JsonObject) -> bool:
    events = entry.get("response_events")
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, dict) and event.get("type") in {"DealStarted", "DealEnded", "MatchEnded"}
        for event in events
    )


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
            if isinstance(seat, str) and team_for_seat(seat) == winning_team:
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


def _events_from_observation(observation: JsonObject) -> list[JsonObject]:
    events = observation.get("events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _players_by_seat_from_request(request: ActionRequest) -> JsonObject:
    snapshot_players = request.snapshot.get("players_by_seat")
    if isinstance(snapshot_players, dict):
        return _string_dict(snapshot_players)
    public = request.snapshot.get("public")
    if isinstance(public, dict) and isinstance(public.get("players_by_seat"), dict):
        return _string_dict(public["players_by_seat"])
    return {}


def _players_by_seat_from_observation(observation: JsonObject) -> JsonObject:
    players = observation.get("players_by_seat")
    return _string_dict(players) if isinstance(players, dict) else {}


def _string_dict(value: dict[object, object]) -> JsonObject:
    return {str(key): str(item) for key, item in value.items() if str(key).strip() and str(item).strip()}


LlmAgentPolicy = LlmAgentPlayer
