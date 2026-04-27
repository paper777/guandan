from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


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
        if not isinstance(profile.get("score"), dict):
            profile["score"] = {}
        if not isinstance(profile.get("skills"), list):
            profile["skills"] = []
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
            "score": {"deals_played": 0, "wins": 0, "last_finish_order": []},
            "skills": [],
            "updated_at": _utc_now(),
        }


class JsonActionLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, entry: JsonObject) -> None:
        entries = self.load()
        entries.append({**entry, "recorded_at": _utc_now()})
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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
