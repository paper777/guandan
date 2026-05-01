from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from client.api import ActionRequest, JsonObject
from npc.common.player import Player
from npc.dummy_bot.policy import DummyBotPolicy
from npc.llm_agent.actions import validate_action
from npc.llm_agent.advisor import ActionAdvisor
from npc.llm_agent.config import LlmAgentConfig
from npc.llm_agent.context import (
    AgentRequestContext,
    public_value,
    safe_snapshot,
    seat_from_request,
    snapshot_value,
    team_for_seat,
)
from npc.llm_agent.personality import personality_context
from npc.llm_agent.prompts import SYSTEM_PROMPT
from npc.llm_agent.provider import LlmActionProvider, provider_from_config
from npc.llm_agent.skills import LLM_AGENT_SKILLS
from npc.llm_agent.strategy import build_strategy_context
from npc.llm_agent.storage import JsonActionLog, JsonMemoryStore


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Prepared inputs shared by model prompting, advisor fallback, and logging."""

    request: ActionRequest
    request_context: AgentRequestContext
    memory_store: JsonMemoryStore
    action_log: JsonActionLog
    memory: JsonObject
    provider_prompt: JsonObject
    advisor_action: JsonObject


class LlmAgentPlayer(Player):
    """Broker-compatible LLM NPC player with isolated filesystem memory."""

    def __init__(self, config: LlmAgentConfig | None = None, provider: LlmActionProvider | None = None) -> None:
        self.config = config or LlmAgentConfig()
        self.provider = provider or provider_from_config(self.config)
        self._advisor = ActionAdvisor()
        self._fallback = DummyBotPolicy()
        self._stores_by_seat: dict[str, tuple[JsonMemoryStore, JsonActionLog]] = {}

    def choose_action(self, request: ActionRequest) -> JsonObject:
        context = self._prepare_decision(request)
        provider_action = self._request_provider(context.provider_prompt)
        action, provider_action, fallback_used = self._select_action(provider_action, context)

        thinking = str(provider_action.get("thinking") or "Selected a legal conservative action.")
        action_with_thinking = {**action, "thinking": thinking}
        self._record_decision(context, action, thinking, fallback_used)
        self._update_memory(context.memory_store, context.memory, provider_action, request=request)
        return action_with_thinking

    def _prepare_decision(self, request: ActionRequest) -> DecisionContext:
        request_context = AgentRequestContext.from_request(request)
        memory_store, action_log = self._stores_for(request_context.seat)
        memory = memory_store.load()
        personality = personality_context(self.config.personality)
        strategy_context = build_strategy_context(request_context)
        strategy_context["personality"] = personality
        advisor_advice = self._advisor.build_advice(request_context, strategy_context)
        provider_prompt = self._build_provider_prompt(
            request,
            memory,
            action_log,
            table_context=request_context.table_context,
            strategy_context=strategy_context,
            card_player=advisor_advice.to_json(),
            personality=personality,
        )
        return DecisionContext(
            request=request,
            request_context=request_context,
            memory_store=memory_store,
            action_log=action_log,
            memory=memory,
            provider_prompt=provider_prompt,
            advisor_action=advisor_advice.recommended_action,
        )

    def _select_action(self, provider_action: JsonObject, context: DecisionContext) -> tuple[JsonObject, JsonObject, bool]:
        model_action = validate_action(provider_action, context.request_context)
        if model_action is not None:
            return model_action, provider_action, False

        fallback_action = validate_action(context.advisor_action, context.request_context)
        fallback_source = "advisor"
        if fallback_action is None:
            fallback_action = self._fallback.choose_action(context.request)
            fallback_source = "dummy bot"
        fallback_provider_action = {
            **fallback_action,
            "thinking": f"Provider output was invalid for the current prompt, so {fallback_source} fallback was used.",
        }
        return fallback_action, fallback_provider_action, True

    def _record_decision(
        self,
        context: DecisionContext,
        action: JsonObject,
        thinking: str,
        fallback_used: bool,
    ) -> None:
        request = context.request
        seat = context.request_context.seat
        context.action_log.append(
            {
                "kind": "decision",
                "request_id": request.request_id,
                "table_id": snapshot_value(request, "table_id"),
                "seat": seat,
                "event_seq": public_value(request, "event_seq"),
                "phase": public_value(request, "phase"),
                "current_level": request.prompt.get("current_level") or public_value(request, "current_level"),
                "legal_action": request.prompt.get("kind"),
                "snapshot": safe_snapshot(request.snapshot),
                "selected_action": action,
                "thinking": thinking,
                "fallback_used": fallback_used,
            }
        )

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

    def _build_provider_prompt(
        self,
        request: ActionRequest,
        memory: JsonObject,
        action_log: JsonActionLog,
        *,
        table_context: JsonObject,
        strategy_context: JsonObject,
        card_player: JsonObject,
        personality: JsonObject,
    ) -> JsonObject:
        return {
            "request_id": request.request_id,
            "prompt": request.prompt,
            "snapshot": request.snapshot,
            "table_context": table_context,
            "strategy_context": strategy_context,
            "personality": personality,
            "card_player": card_player,
            "skills": [dict(skill) for skill in LLM_AGENT_SKILLS],
            "memory": memory,
            "recent_actions": action_log.recent(self.config.max_recent_actions),
            "system_prompt": SYSTEM_PROMPT,
            "model": {
                "provider": self.config.provider_name,
                "name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            },
        }

    def _request_provider(self, prompt: JsonObject) -> JsonObject:
        try:
            action = self.provider.choose_action(prompt)
        except Exception as exc:
            return {"type": "error", "message": str(exc)}
        return action if isinstance(action, dict) else {"type": "error", "message": "provider returned non-object"}

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
            memory["player_name"] = self.config.display_name_for(seat_from_request(request))
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


LlmAgentPolicy = LlmAgentPlayer
