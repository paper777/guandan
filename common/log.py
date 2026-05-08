from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_TRACE_LOG_PATH = Path("data/guandan_trace.jsonl")
TRACE_LOG_PATH_ENV = "GUANDAN_TRACE_LOG_PATH"
TRACE_LOG_ENABLED_ENV = "GUANDAN_TRACE_LOG_ENABLED"
REDACTED = "<redacted>"

_PRIVATE_KEYS = {
    "api_key",
    "authorization",
    "card_id",
    "card_ids",
    "controller_id",
    "eligible_card_ids",
    "hand",
    "hands",
    "player_id",
    "rejected_card_ids",
    "shared_secret",
}


JsonObject = dict[str, Any]


def trace_log_enabled() -> bool:
    value = os.environ.get(TRACE_LOG_ENABLED_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def trace_log_path() -> Path:
    raw = os.environ.get(TRACE_LOG_PATH_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_TRACE_LOG_PATH


def trace_event(event: str, **fields: object) -> None:
    """Append a best-effort structured trace event.

    Tracing must never change game behavior. File write errors are intentionally
    swallowed because this path is used inside server and NPC action timing.
    """

    if not trace_log_enabled():
        return
    entry: JsonObject = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "event": event,
    }
    entry.update(redact_trace_payload(fields))
    try:
        write_trace_entry(entry)
    except OSError:
        return


def write_trace_entry(entry: JsonObject, path: Path | None = None) -> None:
    target = path or trace_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def redact_trace_payload(value: object) -> object:
    if isinstance(value, dict):
        redacted: JsonObject = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key.lower() in _PRIVATE_KEYS:
                redacted[text_key] = REDACTED
            else:
                redacted[text_key] = redact_trace_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_trace_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_trace_payload(item) for item in value]
    return value


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def deadline_remaining_ms(deadline_epoch_ms: object, *, now_epoch_ms: int | None = None) -> int | None:
    if not isinstance(deadline_epoch_ms, int):
        return None
    now = now_epoch_ms if now_epoch_ms is not None else int(time.time() * 1000)
    return deadline_epoch_ms - now


def deadline_fields(deadline_epoch_ms: object) -> JsonObject:
    if not isinstance(deadline_epoch_ms, int):
        return {}
    return {
        "deadline_epoch_ms": deadline_epoch_ms,
        "deadline_remaining_ms": deadline_remaining_ms(deadline_epoch_ms),
    }

