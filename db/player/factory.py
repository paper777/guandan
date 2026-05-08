from __future__ import annotations

from pathlib import Path

from db.player.types import Player, PlayerProfile
from npc.dummy_bot.player import DummyBotPlayer
from npc.llm_agent import LlmAgentConfig, LlmAgentPlayer


NPC_LINEUPS = ("mixed", "dummy", "llm")


def player_for_profile(profile: PlayerProfile, lineup: str, storage_dir: str | Path = Path("data")) -> Player:
    if lineup not in NPC_LINEUPS:
        raise ValueError(f"unsupported NPC lineup: {lineup}")
    kind = profile.kind if lineup == "mixed" else lineup
    if kind == "dummy":
        return DummyBotPlayer()
    llm = profile.llm_config
    provider_name = llm.provider_name or "deterministic"
    return LlmAgentPlayer(
        LlmAgentConfig(
            player_name=profile.display_name,
            seat=profile.preferred_seat,
            personality=profile.personality,
            storage_dir=storage_dir,
            provider_name=provider_name,
            model_name=llm.model_name or _default_model_for_provider(provider_name),
            api_base_url=llm.api_base_url,
            timeout_seconds=llm.timeout_seconds or _default_timeout_for_provider(provider_name),
            temperature=llm.temperature if llm.temperature is not None else 0.2,
            max_output_tokens=llm.max_output_tokens or 800,
            memory_compaction_char_limit=(
                llm.memory_compaction_char_limit if llm.memory_compaction_char_limit is not None else 16000
            ),
            memory_recent_deal_scan_limit=(
                llm.memory_recent_deal_scan_limit if llm.memory_recent_deal_scan_limit is not None else 200
            ),
            memory_max_output_tokens=llm.memory_max_output_tokens if llm.memory_max_output_tokens is not None else 1200,
            codex_binary=llm.codex_binary or "codex",
            codex_working_dir=llm.codex_working_dir,
        )
    )


def _default_model_for_provider(provider_name: str) -> str:
    if provider_name in {"codex-cli", "codex_signed_in", "codex-signed-in"}:
        return "gpt-5.2"
    return "deterministic-guandan-v1"


def _default_timeout_for_provider(provider_name: str) -> float:
    if provider_name in {"codex-cli", "codex_signed_in", "codex-signed-in"}:
        return 120.0
    return 3.0
