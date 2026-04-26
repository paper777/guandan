from __future__ import annotations

from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from guandan.api.schemas import (
    CommandResponse,
    EventSchema,
    PassRequest,
    PlayCardsRequest,
    PublicTableSnapshotSchema,
    ReadyRequest,
    RejectionSchema,
    StartMatchRequest,
    WebSocketClientMessage,
    WebSocketServerMessage,
)
from guandan.domain.commands import Pass, PlayCards, Ready, StartMatch
from guandan.services.table_actor import TableActor


def register_websocket_routes(app: FastAPI, tables: dict[str, TableActor]) -> None:
    @app.websocket("/ws/tables/{table_id}")
    async def table_websocket(websocket: WebSocket, table_id: str) -> None:
        actor = tables.get(table_id)
        if actor is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await _send(
            websocket,
            WebSocketServerMessage(
                type="snapshot",
                payload={"snapshot": PublicTableSnapshotSchema.from_snapshot(actor.public_snapshot())},
            ),
        )
        while True:
            try:
                raw = await websocket.receive_json()
            except WebSocketDisconnect:
                return
            try:
                message = WebSocketClientMessage.model_validate(raw)
                status, payload = await _handle_message(actor, message)
            except (KeyError, ValueError, ValidationError) as exc:
                status = 400
                payload = {"error": str(exc)}
            await _send(websocket, WebSocketServerMessage(type="response", status=status, payload=payload))


async def _handle_message(actor: TableActor, message: WebSocketClientMessage) -> tuple[int, dict[str, Any]]:
    if message.type == "snapshot":
        return 200, {
            "snapshot": PublicTableSnapshotSchema.from_snapshot(actor.public_snapshot()),
            "event_seq": actor.state.event_seq,
        }
    payload = dict(message.payload)
    if message.request_id is not None and "request_id" not in payload:
        payload["request_id"] = message.request_id
    if message.type == "ready":
        request = ReadyRequest.model_validate(payload)
        result = await actor.dispatch_async(
            Ready(request.controller_id, request.seat),
            controller_id=request.controller_id,
            request_id=request.request_id,
        )
    elif message.type == "start":
        request = StartMatchRequest.model_validate(payload)
        result = await actor.dispatch_async(
            StartMatch(request.seed),
            controller_id=request.controller_id,
            request_id=request.request_id,
        )
    elif message.type == "play_cards":
        request = PlayCardsRequest.model_validate(payload)
        result = await actor.dispatch_async(
            PlayCards(
                request.controller_id,
                request.seat,
                request.card_ids,
                declared_type=request.declared_type,
                wild_assignments=request.wild_assignments,
            ),
            controller_id=request.controller_id,
            request_id=request.request_id,
        )
    elif message.type == "pass":
        request = PassRequest.model_validate(payload)
        result = await actor.dispatch_async(
            Pass(request.controller_id, request.seat),
            controller_id=request.controller_id,
            request_id=request.request_id,
        )
    else:
        raise ValueError(f"unsupported websocket message type: {message.type}")

    if result.rejection is not None:
        return 400, {
            "rejection": RejectionSchema.from_rejection(result.rejection),
            "event_seq": actor.state.event_seq,
        }
    response = CommandResponse(
        events=[EventSchema.from_event(event) for event in result.events],
        event_seq=actor.state.event_seq,
        snapshot=PublicTableSnapshotSchema.from_snapshot(actor.public_snapshot()),
        replayed=result.replayed,
    )
    return 200, response.model_dump(mode="json")


async def _send(websocket: WebSocket, message: WebSocketServerMessage) -> None:
    await websocket.send_json(message.model_dump(mode="json"))
