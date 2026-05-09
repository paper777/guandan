from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from common.log import DEFAULT_LOG_DIR


@dataclass(frozen=True, slots=True)
class ModelSettings:
    provider_name: str | None = None
    api_base_url: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    timeout_seconds: float | None = None
    max_output_tokens: int | None = None
    model_reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedModelSettings:
    role: str
    provider_name: str
    api_base_url: str | None
    model_name: str
    temperature: float
    timeout_seconds: float
    max_output_tokens: int
    model_reasoning_effort: str | None = None


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
    api_key: str | None = None
    codex_binary: str = "codex"
    codex_working_dir: str | Path | None = None
    play_fast: ModelSettings = field(default_factory=ModelSettings)
    play_pro: ModelSettings | None = None
    memory_model: ModelSettings | None = None
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
        return DEFAULT_LOG_DIR / "llm_completions.jsonl"

    def resolved_model(self, role: str) -> ResolvedModelSettings:
        base = ResolvedModelSettings(
            role="fast",
            provider_name=self.play_fast.provider_name or "deterministic",
            api_base_url=self.play_fast.api_base_url,
            model_name=self.play_fast.model_name or "deterministic-guandan-v1",
            temperature=self.play_fast.temperature if self.play_fast.temperature is not None else 0.2,
            timeout_seconds=self.play_fast.timeout_seconds if self.play_fast.timeout_seconds is not None else 40.0,
            max_output_tokens=self.play_fast.max_output_tokens if self.play_fast.max_output_tokens is not None else 800,
            model_reasoning_effort=None,
        )
        fast = _resolve_model("fast", base, self.play_fast)
        pro = _resolve_model("pro", base, self.play_pro)
        if role == "fast":
            return fast
        if role == "pro":
            return pro
        if role == "memory":
            memory_base = ResolvedModelSettings(
                role="memory",
                provider_name=pro.provider_name,
                api_base_url=pro.api_base_url,
                model_name=pro.model_name,
                temperature=pro.temperature,
                timeout_seconds=pro.timeout_seconds,
                max_output_tokens=self.memory_max_output_tokens,
            )
            return _resolve_model("memory", memory_base, self.memory_model)
        raise ValueError(f"unsupported model role: {role}")


def _resolve_model(
    role: str,
    base: ResolvedModelSettings,
    override: ModelSettings | None,
) -> ResolvedModelSettings:
    if override is None:
        return ResolvedModelSettings(
            role=role,
            provider_name=base.provider_name,
            api_base_url=base.api_base_url,
            model_name=base.model_name,
            temperature=base.temperature,
            timeout_seconds=base.timeout_seconds,
            max_output_tokens=base.max_output_tokens,
            model_reasoning_effort=base.model_reasoning_effort,
        )
    return ResolvedModelSettings(
        role=role,
        provider_name=override.provider_name if override.provider_name is not None else base.provider_name,
        api_base_url=override.api_base_url if override.api_base_url is not None else base.api_base_url,
        model_name=override.model_name if override.model_name is not None else base.model_name,
        temperature=override.temperature if override.temperature is not None else base.temperature,
        timeout_seconds=override.timeout_seconds if override.timeout_seconds is not None else base.timeout_seconds,
        max_output_tokens=(
            override.max_output_tokens if override.max_output_tokens is not None else base.max_output_tokens
        ),
        model_reasoning_effort=(
            override.model_reasoning_effort
            if override.model_reasoning_effort is not None
            else base.model_reasoning_effort
        ),
    )


def _safe_path_part(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    cleaned = cleaned.strip("-_")
    return cleaned or "llm-agent"
