from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from server.api.schemas import (
    AgentJoinRequest,
    CommandResponse,
    ErrorResponse,
    EventSchema,
    HealthResponse,
    JoinTableRequest,
    PassRequest,
    PlayCardsRequest,
    PublicTableSnapshotSchema,
    ReadyRequest,
    ReturnTributeRequest,
    RejectionResponse,
    RejectionSchema,
    SeatSnapshotSchema,
    StartMatchRequest,
    SubmitTributeRequest,
    TableCreateRequest,
    TableCreateResponse,
    TableListResponse,
    VersionResponse,
)
from server.api.websocket import register_websocket_routes
from server.domain.commands import JoinTable, Pass, PlayCards, Ready, ReturnTribute, StartMatch, SubmitTribute
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.seats import Seat
from server.services.table_config import TableConfig, TimeoutFallback
from server.services.table_actor import ActorResult, TableActor


def create_app(tables: dict[str, TableActor] | None = None) -> FastAPI:
    table_registry: dict[str, TableActor] = {} if tables is None else tables
    app = FastAPI(title="Guandan Server", version="0.1.0")
    router = create_router(table_registry)
    app.include_router(router)
    register_websocket_routes(app, table_registry)
    return app


def create_router(tables: dict[str, TableActor]) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(ok=True, service="guandan-server")

    @router.get("/version", response_model=VersionResponse)
    async def version() -> VersionResponse:
        return VersionResponse(version="0.1.0")

    @router.post("/tables", response_model=TableCreateResponse, status_code=201)
    async def create_table(request: TableCreateRequest | None = None) -> TableCreateResponse:
        table_id = f"table-{uuid.uuid4().hex[:12]}"
        table_request = request or TableCreateRequest()
        config = TableConfig(
            action_timeout_seconds=table_request.action_timeout_seconds,
            timeout_fallback=TimeoutFallback(table_request.timeout_fallback),
        )
        tables[table_id] = TableActor(table_id=table_id, config=config)
        return TableCreateResponse(
            table_id=table_id,
            action_timeout_seconds=config.action_timeout_seconds,
            timeout_fallback=config.timeout_fallback,
        )

    @router.get("/tables", response_model=TableListResponse)
    async def list_tables() -> TableListResponse:
        return TableListResponse(tables=list(tables))

    @router.get(
        "/tables/{table_id}",
        response_model=PublicTableSnapshotSchema,
        responses={404: {"model": ErrorResponse}},
    )
    async def get_table(table_id: str) -> PublicTableSnapshotSchema:
        actor = _actor_or_404(tables, table_id)
        return PublicTableSnapshotSchema.from_snapshot(actor.public_snapshot())

    @router.get(
        "/tables/{table_id}/seats/{seat}/snapshot",
        response_model=SeatSnapshotSchema,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    async def get_seat_snapshot(table_id: str, seat: Seat, controller_id: str) -> SeatSnapshotSchema:
        actor = _actor_or_404(tables, table_id)
        try:
            snapshot = actor.seat_snapshot(seat, controller_id)
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SeatSnapshotSchema.from_snapshot(snapshot)

    @router.post(
        "/tables/{table_id}/join-human",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def join_human(table_id: str, request: JoinTableRequest) -> CommandResponse | JSONResponse:
        return await _join(tables, table_id, request, PlayerKind.HUMAN, ControllerKind.HUMAN_WS)

    @router.post(
        "/tables/{table_id}/join-local-bot",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def join_local_bot(table_id: str, request: JoinTableRequest) -> CommandResponse | JSONResponse:
        return await _join(tables, table_id, request, PlayerKind.BOT, ControllerKind.LOCAL_BOT)

    @router.post(
        "/tables/{table_id}/join-agent",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def join_agent(table_id: str, request: AgentJoinRequest) -> CommandResponse | JSONResponse:
        return await _join(tables, table_id, request, PlayerKind.AGENT, ControllerKind.EXTERNAL_AGENT)

    @router.post(
        "/tables/{table_id}/ready",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def ready(table_id: str, request: ReadyRequest) -> CommandResponse | JSONResponse:
        actor = _actor_or_404(tables, table_id)
        return await _command_response(actor, Ready(request.controller_id, request.seat), request)

    @router.post(
        "/tables/{table_id}/start",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def start(table_id: str, request: StartMatchRequest) -> CommandResponse | JSONResponse:
        actor = _actor_or_404(tables, table_id)
        return await _command_response(actor, StartMatch(), request)

    @router.post(
        "/tables/{table_id}/play",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def play(table_id: str, request: PlayCardsRequest) -> CommandResponse | JSONResponse:
        actor = _actor_or_404(tables, table_id)
        return await _command_response(
            actor,
            PlayCards(
                request.controller_id,
                request.seat,
                request.card_ids,
                declared_type=request.declared_type,
                wild_assignments=request.wild_assignments,
            ),
            request,
        )

    @router.post(
        "/tables/{table_id}/pass",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def pass_turn(table_id: str, request: PassRequest) -> CommandResponse | JSONResponse:
        actor = _actor_or_404(tables, table_id)
        return await _command_response(actor, Pass(request.controller_id, request.seat), request)

    @router.post(
        "/tables/{table_id}/tribute",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def submit_tribute(table_id: str, request: SubmitTributeRequest) -> CommandResponse | JSONResponse:
        actor = _actor_or_404(tables, table_id)
        return await _command_response(
            actor,
            SubmitTribute(request.controller_id, request.seat, request.card_id),
            request,
        )

    @router.post(
        "/tables/{table_id}/return-tribute",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def return_tribute(table_id: str, request: ReturnTributeRequest) -> CommandResponse | JSONResponse:
        actor = _actor_or_404(tables, table_id)
        return await _command_response(
            actor,
            ReturnTribute(request.controller_id, request.seat, request.card_id),
            request,
        )

    return router


def _actor_or_404(tables: dict[str, TableActor], table_id: str) -> TableActor:
    actor = tables.get(table_id)
    if actor is None:
        raise HTTPException(status_code=404, detail="table not found")
    return actor


async def _join(
    tables: dict[str, TableActor],
    table_id: str,
    request: JoinTableRequest,
    player_kind: PlayerKind,
    controller_kind: ControllerKind,
) -> CommandResponse | JSONResponse:
    actor = _actor_or_404(tables, table_id)
    player_id = request.player_id or f"player-{uuid.uuid4().hex[:12]}"
    controller_id = request.controller_id or f"controller-{uuid.uuid4().hex[:12]}"
    capabilities = {
        ControllerCapability.PLAY,
        ControllerCapability.OBSERVE_PUBLIC,
        ControllerCapability.OBSERVE_PRIVATE,
    }
    if controller_kind == ControllerKind.LOCAL_BOT:
        capabilities.add(ControllerCapability.AUTO_READY)
    player = PlayerRef(
        id=player_id,
        display_name=request.display_name or player_id,
        kind=player_kind,
    )
    controller = ControllerRef(
        id=controller_id,
        kind=controller_kind,
        seat=request.seat,
        player_id=player_id,
        capabilities=frozenset(capabilities),
    )
    return await _command_response(
        actor,
        JoinTable(player, controller, request.seat),
        request,
        extra={"player_id": player_id, "controller_id": controller_id},
    )


async def _command_response(
    actor: TableActor,
    command: Any,
    request: Any,
    *,
    extra: dict[str, Any] | None = None,
) -> CommandResponse | JSONResponse:
    result = await actor.dispatch_async(
        command,
        controller_id=getattr(request, "controller_id", None),
        request_id=getattr(request, "request_id", None),
    )
    if result.rejection is not None:
        rejection = RejectionResponse(
            rejection=RejectionSchema.from_rejection(result.rejection),
            event_seq=actor.state.event_seq,
        )
        return JSONResponse(status_code=400, content=rejection.model_dump(mode="json"))
    payload = _response_from_result(actor, result, extra=extra)
    return payload


def _response_from_result(
    actor: TableActor,
    result: ActorResult,
    *,
    extra: dict[str, Any] | None = None,
) -> CommandResponse:
    response = CommandResponse(
        events=[EventSchema.from_event(event) for event in result.events],
        event_seq=actor.state.event_seq,
        snapshot=PublicTableSnapshotSchema.from_snapshot(actor.public_snapshot()),
        replayed=result.replayed,
    )
    if extra:
        for key, value in extra.items():
            setattr(response, key, value)
    return response
