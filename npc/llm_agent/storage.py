from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

TECHNIQUE_CATEGORIES = (
    "team_coordination",
    "bomb_usage",
    "offensive_card_formation",
    "defensive_card_formation",
    "combo_removal",
    "others",
)


class JsonMemoryStore:
    def __init__(self, path: Path, *, player_name: str, seat: str | None = None) -> None:
        self.path = path
        self.player_name = player_name
        self.seat = seat

    def load(self) -> JsonObject:
        raw = _load_json(self.path)
        if not isinstance(raw, dict):
            return self.default_profile()
        profile = self.default_profile()
        profile.update(raw)
        profile.pop("score", None)
        legacy_skills = profile.pop("skills", None)
        profile["techniques"] = _normalize_techniques(profile.get("techniques"), legacy_skills)
        if not isinstance(profile.get("player_profiles"), dict):
            profile["player_profiles"] = {}
        return profile

    def save(self, profile: JsonObject) -> None:
        profile = dict(profile)
        profile["updated_at"] = _utc_now()
        _write_json_atomic(self.path, profile)

    def default_profile(self) -> JsonObject:
        return {
            "player_name": self.player_name,
            "seat": self.seat,
            "play_style": "balanced",
            "techniques": {
                "level1": [],
                "level2": {category: [] for category in TECHNIQUE_CATEGORIES},
            },
            "player_profiles": {},
            "updated_at": _utc_now(),
        }


class JsonActionLog:
    def __init__(self, path: Path, *, max_entries: int | None = None) -> None:
        self.path = path
        self.max_entries = max_entries

    def append(self, entry: JsonObject) -> None:
        entries = self.load()
        entries.append({**entry, "recorded_at": _utc_now()})
        if self.max_entries is not None and self.max_entries > 0:
            entries = entries[-self.max_entries :]
        _write_json_atomic(self.path, entries)

    def load(self) -> list[JsonObject]:
        raw = _load_json(self.path)
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def recent(self, limit: int) -> list[JsonObject]:
        if limit <= 0:
            return []
        return self.load()[-limit:]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _normalize_techniques(value: object, legacy_skills: object = None) -> JsonObject:
    if isinstance(value, dict):
        level1 = _normalize_level1(value.get("level1"))
        level2 = _normalize_level2(value.get("level2"))
    else:
        level1 = _normalize_level1(value)
        level2 = _normalize_level2(None)
    legacy_entries = _normalize_level1(legacy_skills)
    if legacy_entries:
        level1 = [*legacy_entries, *level1]
    return {"level1": level1, "level2": level2}


def _normalize_level1(value: object) -> list[JsonObject]:
    entries: list[JsonObject] = []
    if not isinstance(value, list):
        return entries
    for item in value:
        if isinstance(item, dict):
            summary = str(item.get("summary", "")).strip()
            techniques = _string_list(item.get("techniques"))
            entry = {key: item[key] for key in ("deal_seq", "created_at", "source") if key in item}
            if summary:
                entry["summary"] = summary
            if techniques:
                entry["techniques"] = techniques
            if entry.get("summary") or entry.get("techniques"):
                entries.append(entry)
            continue
        text = str(item).strip()
        if text:
            entries.append({"summary": text, "techniques": [text], "source": "legacy"})
    return entries


def _normalize_level2(value: object) -> JsonObject:
    source = value if isinstance(value, dict) else {}
    return {category: _string_list(source.get(category)) for category in TECHNIQUE_CATEGORIES}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
