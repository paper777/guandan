from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any
from urllib.parse import parse_qs

from guandan.domain.commands import JoinTable, Pass, PlayCards, Ready, StartMatch
from guandan.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from guandan.domain.seats import Seat
from guandan.services.table_config import TableConfig, TimeoutFallback
from guandan.services.table_actor import TableActor


TABLES: dict[str, TableActor] = {}


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    """Minimal ASGI app until the FastAPI layer is built."""
    if scope["type"] == "websocket":
        await _websocket(scope, receive, send)
        return
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
        body = await _read_json(receive)
        if body is None:
            await _json(send, 400, {"error": "invalid JSON body"})
            return
        try:
            config = _table_config_from_body(body)
        except ValueError as exc:
            await _json(send, 400, {"error": str(exc)})
            return
        table_id = f"table-{uuid.uuid4().hex[:12]}"
        TABLES[table_id] = TableActor(table_id=table_id, config=config)
        await _json(
            send,
            201,
            {
                "table_id": table_id,
                "action_timeout_seconds": config.action_timeout_seconds,
                "timeout_fallback": config.timeout_fallback.value,
            },
        )
        return
    if path == "/tables" and method == "GET":
        await _json(send, 200, {"tables": list(TABLES)})
        return
    if path.startswith("/tables/") and method == "GET":
        seat_request = _seat_snapshot_action(path)
        if seat_request is not None:
            actor, seat = seat_request
            if actor is None:
                await _json(send, 404, {"error": "table not found"})
                return
            query = parse_qs(scope.get("query_string", b"").decode())
            controller_id = query.get("controller_id", [""])[0]
            try:
                snapshot = actor.seat_snapshot(seat, controller_id)
            except PermissionError as exc:
                await _json(send, 400, {"error": str(exc)})
                return
            await _json(send, 200, snapshot)
            return
        table_id = path.rsplit("/", 1)[-1]
        actor = TABLES.get(table_id)
        if actor is None:
            await _json(send, 404, {"error": "table not found"})
            return
        await _json(send, 200, actor.public_snapshot())
        return
    if path.startswith("/tables/") and method == "POST":
        actor, action = _table_action(path)
        if actor is None:
            await _json(send, 404, {"error": "table not found"})
            return
        body = await _read_json(receive)
        if body is None:
            await _json(send, 400, {"error": "invalid JSON body"})
            return
        try:
            status, payload = await _handle_table_action(actor, action, body)
        except (KeyError, ValueError) as exc:
            await _json(send, 400, {"error": str(exc)})
            return
        await _json(send, status, payload)
        return
    await _json(send, 404, {"error": "not found"})


async def _websocket(scope: dict[str, Any], receive: Any, send: Any) -> None:
    actor = _websocket_table(scope.get("path", ""))
    message = await receive()
    if message["type"] != "websocket.connect":
        return
    if actor is None:
        await send({"type": "websocket.close", "code": 1008})
        return
    await send({"type": "websocket.accept"})
    await _send_ws_json(send, {"type": "snapshot", "payload": actor.public_snapshot()})
    while True:
        message = await receive()
        if message["type"] == "websocket.disconnect":
            return
        if message["type"] != "websocket.receive":
            continue
        raw = message.get("text")
        if raw is None and message.get("bytes") is not None:
            raw = message["bytes"].decode()
        try:
            payload = json.loads(raw or "{}")
            if not isinstance(payload, dict):
                raise ValueError("websocket message must be a JSON object")
            status, response = await _handle_websocket_message(actor, payload)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            status, response = 400, {"error": str(exc)}
        await _send_ws_json(send, {"type": "response", "status": status, "payload": response})


async def _send_ws_json(send: Any, payload: Any) -> None:
    await send({"type": "websocket.send", "text": json.dumps(_to_jsonable(payload), sort_keys=True)})


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


async def _read_json(receive: Any) -> dict[str, Any] | None:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    raw = b"".join(chunks)
    if not raw:
        return {}
    try:
        value = json.loads(raw.decode())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _table_config_from_body(body: dict[str, Any]) -> TableConfig:
    return TableConfig(
        action_timeout_seconds=int(body.get("action_timeout_seconds", 45)),
        timeout_fallback=TimeoutFallback(str(body.get("timeout_fallback", TimeoutFallback.AUTO_PASS.value))),
    )


def _table_action(path: str) -> tuple[TableActor | None, str]:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 3 or parts[0] != "tables":
        return None, ""
    actor = TABLES.get(parts[1])
    return actor, parts[2]


def _seat_snapshot_action(path: str) -> tuple[TableActor | None, Seat] | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 5 or parts[0] != "tables" or parts[2] != "seats" or parts[4] != "snapshot":
        return None
    return TABLES.get(parts[1]), Seat(parts[3])


def _websocket_table(path: str) -> TableActor | None:
    parts = [part for part in path.split("/") if part]
    if len(parts) != 3 or parts[0] != "ws" or parts[1] != "tables":
        return None
    return TABLES.get(parts[2])


async def _handle_websocket_message(actor: TableActor, message: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    message_type = str(message.get("type", ""))
    if message_type == "snapshot":
        return 200, {"snapshot": actor.public_snapshot(), "event_seq": actor.state.event_seq}
    body = dict(message.get("payload", {}))
    if "request_id" in message and "request_id" not in body:
        body["request_id"] = message["request_id"]
    action = {
        "ready": "ready",
        "start": "start",
        "play_cards": "play",
        "pass": "pass",
    }.get(message_type)
    if action is None:
        raise ValueError(f"unsupported websocket message type: {message_type}")
    return await _handle_table_action(actor, action, body)


async def _handle_table_action(actor: TableActor, action: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if action in {"join-human", "join-local-bot", "join-agent"}:
        return await _join(actor, action, body)
    if action == "ready":
        return await _dispatch(actor, Ready(controller_id=str(body["controller_id"]), seat=Seat(body["seat"])), body)
    if action == "start":
        return await _dispatch(actor, StartMatch(seed=body.get("seed")), body)
    if action == "play":
        return await _dispatch(
            actor,
            PlayCards(
                controller_id=str(body["controller_id"]),
                seat=Seat(body["seat"]),
                card_ids=tuple(body["card_ids"]),
                declared_type=body.get("declared_type"),
            ),
            body,
        )
    if action == "pass":
        return await _dispatch(actor, Pass(controller_id=str(body["controller_id"]), seat=Seat(body["seat"])), body)
    raise ValueError(f"unsupported table action: {action}")


async def _join(actor: TableActor, action: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    seat = Seat(body["seat"])
    player_kind = {
        "join-human": PlayerKind.HUMAN,
        "join-local-bot": PlayerKind.BOT,
        "join-agent": PlayerKind.AGENT,
    }[action]
    controller_kind = {
        "join-human": ControllerKind.HUMAN_WS,
        "join-local-bot": ControllerKind.LOCAL_BOT,
        "join-agent": ControllerKind.EXTERNAL_AGENT,
    }[action]
    player_id = str(body.get("player_id", f"player-{uuid.uuid4().hex[:12]}"))
    controller_id = str(body.get("controller_id", f"controller-{uuid.uuid4().hex[:12]}"))
    display_name = str(body.get("display_name", player_id))
    capabilities = {
        ControllerCapability.PLAY,
        ControllerCapability.OBSERVE_PUBLIC,
        ControllerCapability.OBSERVE_PRIVATE,
    }
    if action == "join-local-bot":
        capabilities.add(ControllerCapability.AUTO_READY)
    player = PlayerRef(id=player_id, display_name=display_name, kind=player_kind)
    controller = ControllerRef(
        id=controller_id,
        kind=controller_kind,
        seat=seat,
        player_id=player_id,
        capabilities=frozenset(capabilities),
    )
    return await _dispatch(
        actor,
        JoinTable(player, controller, seat),
        body,
        extra={"player_id": player_id, "controller_id": controller_id},
    )


async def _dispatch(
    actor: TableActor,
    command: Any,
    body: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    result = await actor.dispatch_async(
        command,
        controller_id=body.get("controller_id"),
        request_id=body.get("request_id"),
    )
    if result.rejection is not None:
        return 400, {"rejection": result.rejection, "event_seq": actor.state.event_seq}
    payload: dict[str, Any] = {
        "events": list(result.events),
        "event_seq": actor.state.event_seq,
        "snapshot": actor.public_snapshot(),
        "replayed": result.replayed,
    }
    if extra:
        payload.update(extra)
    return 200, payload


try:
    from guandan.api.http import create_app as _create_fastapi_app
except ModuleNotFoundError as exc:
    if exc.name not in {"fastapi", "pydantic"}:
        raise
else:
    app = _create_fastapi_app(TABLES)
