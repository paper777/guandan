from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LlmAgentConfig:
    """Runtime configuration for one isolated LLM agent player."""

    player_name: str | None = None
    player_id: str | None = None
    seat: str | None = None
    personality: str = "balanced"
    storage_dir: str | Path = Path("../../data")
    memory_path: str | Path | None = None
    action_log_path: str | Path | None = None
    audit_log_path: str | Path | None = None
    provider_name: str = "deterministic"
    model_name: str = "deterministic-guandan-v1"
    api_key: str | None = None
    api_base_url: str | None = None
    codex_binary: str = "codex"
    codex_working_dir: str | Path | None = None
    timeout_seconds: float = 3.0
    temperature: float = 0.2
    max_output_tokens: int = 800
    max_recent_actions: int = 20
    max_action_log_entries: int = 1000
    memory_compaction_char_limit: int = 16000
    memory_recent_deal_scan_limit: int = 200
    memory_max_output_tokens: int = 1200

    def namespace_for(self, seat: str | None = None) -> str:
        raw = self.player_name or self.player_id or self.seat or seat or "llm-agent"
        return _safe_path_part(raw)

    def display_name_for(self, seat: str | None = None) -> str:
        return self.player_name or f"LLM {self.seat or seat or 'Agent'}"

    def resolved_memory_path(self, seat: str | None = None) -> Path:
        if self.memory_path is not None:
            return Path(self.memory_path)
        return Path(self.storage_dir) / self.namespace_for(seat) / "memory.json"

    def resolved_action_log_path(self, seat: str | None = None) -> Path:
        if self.action_log_path is not None:
            return Path(self.action_log_path)
        return Path(self.storage_dir) / self.namespace_for(seat) / "actions.json"

    def resolved_audit_log_path(self) -> Path:
        if self.audit_log_path is not None:
            return Path(self.audit_log_path)
        return Path(self.storage_dir) / "llm_completions.jsonl"


def _safe_path_part(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    cleaned = cleaned.strip("-_")
    return cleaned or "llm-agent"
