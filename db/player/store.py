from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random, SystemRandom

from db.player.types import LlmConfig, LlmModelConfig, PlayerProfile, PlayerStatistics
from server.domain.cards import STANDARD_RANKS


SEATS = ("E", "S", "W", "N")
PLAYER_DATABASE_PATH = Path("data")
PLAYER_INDEX_FILE = "players.json"
PLAYER_STORAGE_FILES = ("actions.json", "llm_config.json", "memory.json", "profile.json", "statistics.json")
DEFAULT_PLAYER_PROFILES = (
    PlayerProfile("Ming", "dummy", profile_key="Ming", preferred_seat="E", personality="balanced"),
    PlayerProfile("Jade", "llm", profile_key="Jade", preferred_seat="S", personality="aggressive"),
    PlayerProfile("River", "llm", profile_key="River", preferred_seat="W", personality="balanced"),
    PlayerProfile("Atlas", "llm", profile_key="Atlas", preferred_seat="N", personality="defensive"),
)


@dataclass(slots=True)
class PlayerDatabase:
    path: Path
    profiles: list[PlayerProfile]

    def load_by_name(self, display_name: str) -> PlayerProfile:
        profile = next((profile for profile in self.profiles if profile.display_name == display_name), None)
        if profile is None:
            raise KeyError(display_name)
        return profile

    def profile_for_seat(self, seat: str) -> PlayerProfile | None:
        return next((profile for profile in self.profiles if profile.preferred_seat == seat), None)

    def profile_for_key(self, profile_key: str) -> PlayerProfile | None:
        return next((profile for profile in self.profiles if profile.profile_key == profile_key), None)

    def new_or_update(self, updated: PlayerProfile) -> None:
        key = updated.profile_key or updated.display_name
        for index, profile in enumerate(self.profiles):
            if (profile.profile_key or profile.display_name) == key:
                self.profiles[index] = updated
                return
        self.profiles.append(updated)

    def replace_profile(self, updated: PlayerProfile) -> None:
        self.new_or_update(updated)

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        index_players: list[str] = []
        for profile in self.profiles:
            directory = _player_dir_name(profile)
            player_dir = self.path / directory
            player_dir.mkdir(parents=True, exist_ok=True)
            _write_json(player_dir / "profile.json", profile_identity_to_json(profile))
            _write_json(player_dir / "llm_config.json", llm_config_to_json(profile.llm_config))
            _write_json(player_dir / "statistics.json", statistics_to_json(profile.statistics))
            _ensure_json_file(player_dir / "actions.json", [])
            _ensure_json_file(player_dir / "memory.json", {})
            index_players.append(directory)
        _write_json(self.path / PLAYER_INDEX_FILE, {"players": index_players})


def load_player_profiles(path: str | Path = PLAYER_DATABASE_PATH) -> tuple[PlayerProfile, ...]:
    return tuple(load_player_database(path).profiles)


def load_player_database(path: str | Path = PLAYER_DATABASE_PATH) -> PlayerDatabase:
    config_path = Path(path).expanduser()
    storage_path = _storage_path(config_path)
    index_path = config_path if config_path.suffix == ".json" else storage_path / PLAYER_INDEX_FILE
    if index_path.exists():
        return PlayerDatabase(storage_path, _load_player_index(storage_path, index_path))
    return PlayerDatabase(storage_path, list(DEFAULT_PLAYER_PROFILES))


def _load_player_index(storage_path: Path, index_path: Path) -> list[PlayerProfile]:
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid player index JSON: {index_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"player index must be a JSON object: {index_path}")
    players = raw.get("players")
    if not isinstance(players, list):
        raise ValueError(f"player index must contain a players list: {index_path}")

    items: list[dict[str, object]] = []
    seen_directories: set[str] = set()
    for entry in players:
        directory = _index_entry(entry)
        if directory in seen_directories:
            raise ValueError(f"duplicate player directory: {directory}")
        seen_directories.add(directory)
        player_dir = storage_path / directory
        profile = _read_json_object(player_dir / "profile.json")
        if "seat" in profile:
            raise ValueError(f"player profile must not contain seat: {player_dir / 'profile.json'}")
        llm_config_path = player_dir / "llm_config.json"
        if llm_config_path.exists():
            profile.update(_read_json_object(llm_config_path))
        statistics_path = player_dir / "statistics.json"
        if statistics_path.exists():
            profile.update(_read_json_object(statistics_path))
        items.append(profile)

    return _profiles_from_items(items)


def _profiles_from_items(items: list[object]) -> list[PlayerProfile]:
    profiles: list[PlayerProfile] = []
    seen_keys: set[str] = set()
    seen_seats: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("player database entries must be objects")
        fallback = DEFAULT_PLAYER_PROFILES[index % len(DEFAULT_PLAYER_PROFILES)]
        raw_seat = _optional_str(item.get("seat"))
        preferred_seat = raw_seat.upper() if raw_seat is not None else fallback.preferred_seat
        if preferred_seat and preferred_seat not in SEATS:
            raise ValueError(f"invalid player seat: {preferred_seat}")
        if raw_seat is not None:
            if preferred_seat in seen_seats:
                raise ValueError(f"duplicate player seat: {preferred_seat}")
            seen_seats.add(preferred_seat)

        display_name = str(item.get("display_name") or item.get("name") or fallback.display_name)
        profile_key = str(item.get("id") or item.get("profile_id") or display_name).strip()
        if not profile_key:
            raise ValueError("player profile id/display_name cannot be empty")
        if profile_key in seen_keys:
            raise ValueError(f"duplicate player profile: {profile_key}")
        seen_keys.add(profile_key)

        kind = str(item.get("kind") or fallback.kind).lower()
        if kind not in {"dummy", "llm"}:
            raise ValueError(f"invalid player kind for {display_name}: {kind}")
        profiles.append(
            PlayerProfile(
                display_name=display_name,
                kind=kind,
                profile_key=profile_key,
                preferred_seat=preferred_seat,
                personality=str(item.get("personality") or fallback.personality),
                llm_config=LlmConfig(
                    memory_compaction_char_limit=_memory_int(
                        item, "compaction_char_limit", "memory_compaction_char_limit"
                    ),
                    memory_recent_deal_scan_limit=_memory_int(
                        item, "recent_deal_scan_limit", "memory_recent_deal_scan_limit"
                    ),
                    memory_max_output_tokens=_memory_int(item, "max_output_tokens", "memory_max_output_tokens"),
                    play_fast=_play_model_config(item, "fast", include_flat=True),
                    play_pro=_play_model_config(item, "pro"),
                    memory_model=_memory_model_config(item),
                    codex_binary=_optional_str(item.get("codex_binary")),
                    codex_working_dir=_optional_str(item.get("codex_working_dir")),
                ),
                statistics=PlayerStatistics(
                    deal_count=_stat_int(item.get("deal_count")),
                    deal_wins=_stat_int(item.get("deal_wins")),
                    deal_win_rate=_stat_float(item.get("deal_win_rate")),
                    score=_stat_int(item.get("score")),
                    match_count=_stat_int(item.get("match_count")),
                    match_wins=_stat_int(item.get("match_wins")),
                    match_win_rate=_stat_float(item.get("match_win_rate")),
                ),
                extra={str(key): value for key, value in item.items() if key not in _PROFILE_KEYS},
            )
        )
    return profiles


def profile_assignments(
    profiles: list[PlayerProfile],
    seats: tuple[str, ...],
    *,
    shuffle_seed: object = None,
    exclude_profile_keys: set[str] | frozenset[str] | None = None,
) -> list[tuple[PlayerProfile, str]]:
    selected = _profiles_for_selected_seats(profiles, seats, exclude_profile_keys=exclude_profile_keys)
    available_seats = list(seats)
    rng = SystemRandom() if shuffle_seed is None else Random(shuffle_seed)
    rng.shuffle(available_seats)
    return list(zip(selected, available_seats))


def assigned_profile(profile: PlayerProfile, seat: str) -> PlayerProfile:
    return replace(profile, preferred_seat=seat)


def record_profile_result(
    profile: PlayerProfile,
    *,
    kind: str,
    won: bool,
    score_delta: int = 0,
) -> PlayerProfile:
    stats = profile.statistics
    if kind == "deal":
        count = stats.deal_count + 1
        wins = stats.deal_wins + (1 if won else 0)
        return replace(
            profile,
            statistics=replace(
                stats,
                deal_count=count,
                deal_wins=wins,
                deal_win_rate=wins / count,
                score=stats.score + score_delta,
            ),
        )
    if kind == "match":
        count = stats.match_count + 1
        wins = stats.match_wins + (1 if won else 0)
        return replace(
            profile,
            statistics=replace(stats, match_count=count, match_wins=wins, match_win_rate=wins / count),
        )
    raise ValueError(f"unsupported player result kind: {kind}")


def deal_score_delta(winning_team: str, snapshot: dict[str, object] | None, payload: object) -> int:
    losing_team = "SN" if winning_team == "EW" else "EW"
    opponent_level = _level_for_team(snapshot, losing_team)
    if opponent_level is None and isinstance(payload, dict):
        raw_levels = payload.get("level_by_team")
        if isinstance(raw_levels, dict):
            opponent_level = _level_value(raw_levels, losing_team)
        if opponent_level is None:
            opponent_level = _optional_str(payload.get("opponent_level"))
    return _ace_gap_score(opponent_level or "2")


def team_for_seat(seat: str) -> str:
    return "EW" if seat in {"E", "W"} else "SN"


def profile_to_json(profile: PlayerProfile) -> dict[str, object]:
    payload = profile_identity_to_json(profile)
    payload.update(llm_config_to_json(profile.llm_config))
    payload.update(statistics_to_json(profile.statistics))
    return payload


def profile_identity_to_json(profile: PlayerProfile) -> dict[str, object]:
    payload: dict[str, object] = dict(profile.extra)
    if profile.profile_key != profile.display_name:
        payload["id"] = profile.profile_key
    payload["display_name"] = profile.display_name
    payload["kind"] = profile.kind
    payload["personality"] = profile.personality
    return payload


def llm_config_to_json(config: LlmConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in (
        "codex_binary",
        "codex_working_dir",
    ):
        value = getattr(config, key)
        if value is not None:
            payload[key] = str(value) if isinstance(value, Path) else value
    play_payload: dict[str, object] = {}
    fast = _model_config_to_json(config.play_fast)
    pro = _model_config_to_json(config.play_pro)
    if fast:
        play_payload["fast"] = fast
    if pro:
        play_payload["pro"] = pro
    if play_payload:
        payload["play"] = play_payload

    memory_model = _model_config_to_json(config.memory_model)
    memory_payload: dict[str, object] = {}
    if memory_model:
        memory_payload.update(memory_model)
    if config.memory_compaction_char_limit is not None:
        memory_payload["compaction_char_limit"] = config.memory_compaction_char_limit
    if config.memory_recent_deal_scan_limit is not None:
        memory_payload["recent_deal_scan_limit"] = config.memory_recent_deal_scan_limit
    if config.memory_max_output_tokens is not None and "max_output_tokens" not in memory_payload:
        memory_payload["max_output_tokens"] = config.memory_max_output_tokens
    if memory_payload:
        payload["memory"] = memory_payload
    return payload


def _play_model_config(item: dict[str, object], role: str, *, include_flat: bool = False) -> LlmModelConfig:
    play = item.get("play")
    role_config = play.get(role) if isinstance(play, dict) else None
    nested = _model_config_from_object(role_config)
    if not include_flat:
        return nested
    legacy = LlmModelConfig(
        provider_name=_optional_str(item.get("provider_name")),
        api_base_url=_optional_str(item.get("api_base_url")),
        model_name=_optional_str(item.get("model_name")),
        timeout_seconds=_optional_float(item.get("timeout_seconds")),
        temperature=_optional_float(item.get("temperature")),
        max_output_tokens=_optional_int(item.get("max_output_tokens")),
    )
    return _merged_model_config(legacy, nested)


def _memory_model_config(item: dict[str, object]) -> LlmModelConfig:
    memory = item.get("memory")
    nested = _model_config_from_object(memory)
    legacy = LlmModelConfig(
        model_name=_optional_str(item.get("memory_model_name")),
        timeout_seconds=_optional_float(item.get("memory_timeout_seconds")),
        temperature=_optional_float(item.get("memory_temperature")),
        max_output_tokens=_optional_int(item.get("memory_max_output_tokens")),
        model_reasoning_effort=_optional_str(item.get("memory_model_reasoning_effort")),
    )
    return _merged_model_config(legacy, nested)


def _memory_int(item: dict[str, object], nested_key: str, legacy_key: str) -> int | None:
    memory = item.get("memory")
    if isinstance(memory, dict) and memory.get(nested_key) is not None:
        return _optional_int(memory.get(nested_key))
    return _optional_int(item.get(legacy_key))


def _model_config_from_object(value: object) -> LlmModelConfig:
    if not isinstance(value, dict):
        return LlmModelConfig()
    return LlmModelConfig(
        provider_name=_optional_str(value.get("provider_name")),
        api_base_url=_optional_str(value.get("api_base_url")),
        model_name=_optional_str(value.get("model_name")),
        timeout_seconds=_optional_float(value.get("timeout_seconds")),
        temperature=_optional_float(value.get("temperature")),
        max_output_tokens=_optional_int(value.get("max_output_tokens")),
        model_reasoning_effort=_optional_str(value.get("model_reasoning_effort")),
    )


def _merged_model_config(base: LlmModelConfig, override: LlmModelConfig) -> LlmModelConfig:
    return LlmModelConfig(
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


def _model_config_to_json(config: LlmModelConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in (
        "provider_name",
        "api_base_url",
        "model_name",
        "timeout_seconds",
        "temperature",
        "max_output_tokens",
        "model_reasoning_effort",
    ):
        value = getattr(config, key)
        if value is not None:
            payload[key] = value
    return payload


def statistics_to_json(statistics: PlayerStatistics) -> dict[str, object]:
    payload: dict[str, object] = {}
    payload["deal_count"] = statistics.deal_count
    payload["deal_wins"] = statistics.deal_wins
    payload["deal_win_rate"] = statistics.deal_win_rate
    payload["score"] = statistics.score
    payload["match_count"] = statistics.match_count
    payload["match_wins"] = statistics.match_wins
    payload["match_win_rate"] = statistics.match_win_rate
    return payload


def _profiles_for_selected_seats(
    profiles: list[PlayerProfile],
    seats: tuple[str, ...],
    *,
    exclude_profile_keys: set[str] | frozenset[str] | None = None,
) -> list[PlayerProfile]:
    excluded = set(exclude_profile_keys or ())
    available_profiles = _with_fallback_profiles(
        [profile for profile in profiles if profile.profile_key not in excluded],
        excluded,
        required_count=len(seats),
    )
    selected = set(seats)
    seat_matched = [profile for profile in available_profiles if profile.preferred_seat in selected]
    if len(seat_matched) >= len(seats):
        return seat_matched[: len(seats)]
    matched_keys = {profile.profile_key for profile in seat_matched}
    remaining = [profile for profile in available_profiles if profile.profile_key not in matched_keys]
    return [*seat_matched, *remaining][: len(seats)]


def _with_fallback_profiles(
    profiles: list[PlayerProfile],
    excluded: set[str],
    *,
    required_count: int,
) -> list[PlayerProfile]:
    if len(profiles) >= required_count:
        return profiles
    seen = {profile.profile_key for profile in profiles}
    filled = list(profiles)
    for fallback in DEFAULT_PLAYER_PROFILES:
        if fallback.profile_key in seen or fallback.profile_key in excluded:
            continue
        filled.append(fallback)
        seen.add(fallback.profile_key)
    return filled


def _level_for_team(snapshot: dict[str, object] | None, team: str) -> str | None:
    if not isinstance(snapshot, dict):
        return None
    raw_levels = snapshot.get("level_by_team")
    if not isinstance(raw_levels, dict):
        return None
    return _level_value(raw_levels, team)


def _level_value(levels: dict[object, object], team: str) -> str | None:
    for raw_team, raw_level in levels.items():
        if str(raw_team) == team:
            return str(raw_level)
    return None


def _ace_gap_score(level: str) -> int:
    rank_order = [rank.value for rank in reversed(STANDARD_RANKS)]
    try:
        return rank_order.index(level) + 1
    except ValueError:
        return len(rank_order)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stat_int(value: object) -> int:
    return 0 if value is None else int(value)


def _stat_float(value: object) -> float:
    return 0.0 if value is None else float(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _storage_path(path: Path) -> Path:
    return path.parent if path.suffix == ".json" else path


def _index_entry(entry: object) -> str:
    if isinstance(entry, str):
        directory = entry.strip()
    elif isinstance(entry, dict):
        if "seat" in entry:
            raise ValueError("player index entries must not contain seat")
        directory = str(entry.get("directory") or entry.get("dir") or entry.get("path") or "").strip()
    else:
        raise ValueError("player index entries must be strings or objects")
    if not directory:
        raise ValueError("player index entry directory cannot be empty")
    return directory


def _player_dir_name(profile: PlayerProfile) -> str:
    raw = (profile.display_name or profile.profile_key or "player").strip()
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in raw)
    cleaned = cleaned.strip(".-")
    return cleaned or "player"


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid player storage JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"player storage file must be a JSON object: {path}")
    return dict(raw)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_json_file(path: Path, default_payload: object) -> None:
    if path.exists():
        return
    _write_json(path, default_payload)


_PROFILE_KEYS = {
    "seat",
    "id",
    "profile_id",
    "display_name",
    "name",
    "kind",
    "personality",
    "provider_name",
    "model_name",
    "api_base_url",
    "timeout_seconds",
    "temperature",
    "max_output_tokens",
    "play",
    "memory",
    "memory_compaction_char_limit",
    "memory_recent_deal_scan_limit",
    "memory_max_output_tokens",
    "memory_model_name",
    "memory_timeout_seconds",
    "memory_temperature",
    "memory_model_reasoning_effort",
    "codex_binary",
    "codex_working_dir",
    "deal_count",
    "deal_wins",
    "deal_win_rate",
    "score",
    "match_count",
    "match_wins",
    "match_win_rate",
}
