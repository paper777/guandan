from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from guandan.services.snapshots import public_snapshot
from guandan.services.table_actor import TableActor


TABLES: dict[str, TableActor] = {}


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    """Minimal ASGI app until the FastAPI layer is built."""
    if scope["type"] != "http":
        raise RuntimeError(f"unsupported scope type: {scope['type']}")
    method = scope.get("method", "GET")
    path = scope.get("path", "")
    if path == "/health":
        await _json(send, 200, {"ok": True, "service": "guandan-server"})
        return
    if path == "/version":
        await _json(send, 200, {"version": "0.1.0"})
        return
    if path == "/tables" and method == "POST":
        table_id = f"table-{uuid.uuid4().hex[:12]}"
        TABLES[table_id] = TableActor(table_id=table_id)
        await _json(send, 201, {"table_id": table_id})
        return
    if path == "/tables" and method == "GET":
        await _json(send, 200, {"tables": list(TABLES)})
        return
    if path.startswith("/tables/") and method == "GET":
        table_id = path.rsplit("/", 1)[-1]
        actor = TABLES.get(table_id)
        if actor is None:
            await _json(send, 404, {"error": "table not found"})
            return
        await _json(send, 200, public_snapshot(actor.state))
        return
    await _json(send, 404, {"error": "not found"})


async def _json(send: Any, status: int, payload: Any) -> None:
    body = json.dumps(_to_jsonable(payload), sort_keys=True).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_to_jsonable(key)): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    return value
