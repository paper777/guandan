from __future__ import annotations

from pathlib import Path

from db.player.types import Player, PlayerProfile
from npc.dummy_bot.player import DummyBotPlayer
from npc.llm_agent import LlmAgentConfig, LlmAgentPlayer, ModelSettings


NPC_LINEUPS = ("mixed", "dummy", "llm")


def player_for_profile(profile: PlayerProfile, lineup: str, storage_dir: str | Path = Path("data")) -> Player:
    if lineup not in NPC_LINEUPS:
        raise ValueError(f"unsupported NPC lineup: {lineup}")
    kind = profile.kind if lineup == "mixed" else lineup
    if kind == "dummy":
        return DummyBotPlayer()
    llm = profile.llm_config
    provider_name = llm.play_fast.provider_name or "deterministic"
    base_model = llm.play_fast.model_name or _default_model_for_provider(provider_name)
    base_settings = ModelSettings(
        provider_name=provider_name,
        api_base_url=llm.play_fast.api_base_url,
        model_name=base_model,
        timeout_seconds=llm.play_fast.timeout_seconds if llm.play_fast.timeout_seconds is not None else 40.0,
        temperature=llm.play_fast.temperature if llm.play_fast.temperature is not None else 0.2,
        max_output_tokens=llm.play_fast.max_output_tokens or 800,
        model_reasoning_effort=llm.play_fast.model_reasoning_effort,
    )
    memory_max_output_tokens = llm.memory_max_output_tokens if llm.memory_max_output_tokens is not None else 1200
    return LlmAgentPlayer(
        LlmAgentConfig(
            player_name=profile.display_name,
            seat=profile.preferred_seat,
            personality=profile.personality,
            storage_dir=storage_dir,
            play_fast=_merge_model_settings(base_settings, _model_settings(llm.play_fast)),
            play_pro=_merge_model_settings(base_settings, _model_settings(llm.play_pro)),
            memory_model=_model_settings(llm.memory_model),
            memory_compaction_char_limit=(
                llm.memory_compaction_char_limit if llm.memory_compaction_char_limit is not None else 16000
            ),
            memory_recent_deal_scan_limit=(
                llm.memory_recent_deal_scan_limit if llm.memory_recent_deal_scan_limit is not None else 200
            ),
            memory_max_output_tokens=memory_max_output_tokens,
            codex_binary=llm.codex_binary or "codex",
            codex_working_dir=llm.codex_working_dir,
        )
    )


def _model_settings(config: object) -> ModelSettings | None:
    provider_name = getattr(config, "provider_name", None)
    api_base_url = getattr(config, "api_base_url", None)
    model_name = getattr(config, "model_name", None)
    timeout_seconds = getattr(config, "timeout_seconds", None)
    temperature = getattr(config, "temperature", None)
    max_output_tokens = getattr(config, "max_output_tokens", None)
    model_reasoning_effort = getattr(config, "model_reasoning_effort", None)
    if (
        provider_name is None
        and api_base_url is None
        and model_name is None
        and timeout_seconds is None
        and temperature is None
        and max_output_tokens is None
        and model_reasoning_effort is None
    ):
        return None
    return ModelSettings(
        provider_name=provider_name,
        api_base_url=api_base_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        model_reasoning_effort=model_reasoning_effort,
    )


def _merge_model_settings(base: ModelSettings, override: ModelSettings | None) -> ModelSettings:
    if override is None:
        return base
    return ModelSettings(
        provider_name=override.provider_name if override.provider_name is not None else base.provider_name,
        api_base_url=override.api_base_url if override.api_base_url is not None else base.api_base_url,
        model_name=override.model_name if override.model_name is not None else base.model_name,
        timeout_seconds=override.timeout_seconds if override.timeout_seconds is not None else base.timeout_seconds,
        temperature=override.temperature if override.temperature is not None else base.temperature,
        max_output_tokens=override.max_output_tokens if override.max_output_tokens is not None else base.max_output_tokens,
        model_reasoning_effort=(
            override.model_reasoning_effort
            if override.model_reasoning_effort is not None
            else base.model_reasoning_effort
        ),
    )


def _default_model_for_provider(provider_name: str) -> str:
    provider_name = provider_name.lower()
    if provider_name in {"codex-cli", "codex_signed_in", "codex-signed-in"}:
        return "gpt-5.2"
    if provider_name in {"glm", "bigmodel", "zhipu", "zhipuai"}:
        return "glm-5.1"
    return "deterministic-guandan-v1"
