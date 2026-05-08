from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl


DEFAULT_AUDIT_LOG_PATH = Path("data/server_audit.jsonl")
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
) -> dict[str, object]:
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "duration_ms": round((time.monotonic() - started_at) * 1000, 3),
        "client": _client_host(client),
        "request": {
            "method": method,
            "path": path,
            "query": redact_audit_payload(_query_dict(query)),
            "body": redact_audit_payload(request_body),
        },
        "response": {
            "status": status,
            "body": redact_audit_payload(response_body),
        },
    }


def write_audit_entry(entry: dict[str, object], path: Path | None = None) -> None:
    target = path or audit_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def parse_json_body(raw: bytes) -> object:
    if not raw:
        return {}
    try:
        return json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"_raw_body_bytes": len(raw), "_parse_error": "non-json"}


def redact_audit_payload(value: object) -> object:
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key.lower() in _PRIVATE_KEYS:
                redacted[text_key] = REDACTED
            else:
                redacted[text_key] = redact_audit_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_audit_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_audit_payload(item) for item in value]
    return value


def _client_host(client: object) -> str | None:
    if isinstance(client, tuple) and client:
        return str(client[0])
    host = getattr(client, "host", None)
    return str(host) if host is not None else None


def _query_dict(query: str) -> dict[str, object]:
    values: dict[str, object] = {}
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
