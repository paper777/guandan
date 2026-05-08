from db.player.store import (
    DEFAULT_PLAYER_PROFILES,
    PLAYER_DATABASE_PATH,
    PlayerDatabase,
    assigned_profile,
    deal_score_delta,
    load_player_database,
    load_player_profiles,
    profile_assignments,
    record_profile_result,
    team_for_seat,
)
from db.player.types import LlmConfig, Player, PlayerProfile, PlayerStatistics

_FACTORY_EXPORTS = {"NPC_LINEUPS", "player_for_profile"}

__all__ = [
    "DEFAULT_PLAYER_PROFILES",
    "NPC_LINEUPS",
    "PLAYER_DATABASE_PATH",
    "LlmConfig",
    "Player",
    "PlayerDatabase",
    "PlayerProfile",
    "PlayerStatistics",
    "assigned_profile",
    "deal_score_delta",
    "load_player_database",
    "load_player_profiles",
    "player_for_profile",
    "profile_assignments",
    "record_profile_result",
    "team_for_seat",
]


def __getattr__(name: str) -> object:
    if name in _FACTORY_EXPORTS:
        from importlib import import_module

        return getattr(import_module("db.player.factory"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
