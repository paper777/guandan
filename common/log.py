from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl


DEFAULT_LOG_DIR = Path("data/log")
DEFAULT_TRACE_LOG_PATH = DEFAULT_LOG_DIR / "guandan_trace.jsonl"
TRACE_LOG_PATH_ENV = "GUANDAN_TRACE_LOG_PATH"
TRACE_LOG_ENABLED_ENV = "GUANDAN_TRACE_LOG_ENABLED"
DEFAULT_AUDIT_LOG_PATH = DEFAULT_LOG_DIR / "server_audit.jsonl"
AUDIT_LOG_PATH_ENV = "GUANDAN_AUDIT_LOG_PATH"
AUDIT_LOG_ENABLED_ENV = "GUANDAN_AUDIT_LOG_ENABLED"
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
    log_event("trace", event, **fields)


def debug_event(event: str, **fields: object) -> None:
    log_event("debug", event, **fields)


def error_event(event: str, **fields: object) -> None:
    log_event("error", event, **fields)


def log_event(level: str, event: str, **fields: object) -> None:
    """Append a best-effort structured trace event.

    Tracing must never change game behavior. File write errors are intentionally
    swallowed because this path is used inside server and NPC action timing.
    """

    if not trace_log_enabled():
        return
    entry: JsonObject = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "level": level,
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


def audit_log_enabled() -> bool:
    value = os.environ.get(AUDIT_LOG_ENABLED_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def audit_log_path() -> Path:
    raw = os.environ.get(AUDIT_LOG_PATH_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_AUDIT_LOG_PATH


def make_audit_entry(
    *,
    method: str,
    path: str,
    query: str,
    status: int,
    started_at: float,
    request_body: object,
    response_body: object,
    client: object = None,
) -> JsonObject:
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "duration_ms": round((time.monotonic() - started_at) * 1000, 3),
        "client": client_host(client),
        "request": {
            "method": method,
            "path": path,
            "query": redact_audit_payload(query_dict(query)),
            "body": redact_audit_payload(request_body),
        },
        "response": {
            "status": status,
            "body": redact_audit_payload(response_body),
        },
    }


def write_audit_entry(entry: JsonObject, path: Path | None = None) -> None:
    target = path or audit_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def parse_json_body(raw: bytes) -> object:
    if not raw:
        return {}
    try:
        return json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"_raw_body_bytes": len(raw), "_parse_error": "non-json"}


def redact_audit_payload(value: object) -> object:
    return redact_trace_payload(value)


def client_host(client: object) -> str | None:
    if isinstance(client, tuple) and client:
        return str(client[0])
    host = getattr(client, "host", None)
    return str(host) if host is not None else None


def query_dict(query: str) -> JsonObject:
    values: JsonObject = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key in values:
            current = values[key]
            if isinstance(current, list):
                current.append(value)
            else:
                values[key] = [current, value]
        else:
            values[key] = value
    return values
