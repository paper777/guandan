from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


class Player(ABC):
    """Broker-compatible player implementation for dummy bots and LLM agents."""

    @abstractmethod
    def choose_action(self, request: Any) -> JsonObject:
        """Return a command-like action for the broker or HTTP agent server."""

    def observe_action(self, observation: JsonObject) -> None:
        """Observe an action submitted by another broker-controlled player."""


@dataclass(frozen=True, slots=True)
class PlayerStatistics:
    deal_count: int = 0
    deal_wins: int = 0
    deal_win_rate: float = 0.0
    score: int = 0
    match_count: int = 0
    match_wins: int = 0
    match_win_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class LlmConfig:
    provider_name: str | None = None
    model_name: str | None = None
    api_base_url: str | None = None
    timeout_seconds: float | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    memory_compaction_char_limit: int | None = None
    memory_recent_deal_scan_limit: int | None = None
    memory_max_output_tokens: int | None = None
    codex_binary: str | None = None
    codex_working_dir: str | Path | None = None


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    display_name: str
    kind: str
    profile_key: str = ""
    preferred_seat: str = ""
    personality: str = "balanced"
    llm_config: LlmConfig = field(default_factory=LlmConfig)
    statistics: PlayerStatistics = field(default_factory=PlayerStatistics)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def provider_name(self) -> str | None:
        return self.llm_config.provider_name

    @property
    def model_name(self) -> str | None:
        return self.llm_config.model_name

    @property
    def codex_binary(self) -> str | None:
        return self.llm_config.codex_binary

    @property
    def score(self) -> int:
        return self.statistics.score
