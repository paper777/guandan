from __future__ import annotations

import time
import uuid
from typing import Any

from client.types import ActionRequest
from fastapi import Request
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

from common.log import (
    audit_log_enabled,
    client_host,
    debug_event,
    error_event,
    make_audit_entry,
    parse_json_body,
    query_dict,
    trace_event,
    trace_log_enabled,
    write_audit_entry,
)
from server.api.schemas import (
    AgentJoinRequest,
    BotActionRequest,
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
from server.api.web_ui import register_web_ui
from server.api.websocket import register_websocket_routes
from server.domain.commands import JoinTable, Pass, PlayCards, Ready, ReturnTribute, StartMatch, SubmitTribute
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.seats import Seat
from server.services.table_config import TableConfig, TimeoutFallback
from server.services.table_actor import ActorResult, TableActor
from server.services.public_events import public_events
from server.services.snapshots import SeatSnapshot


_RL_AGENT_PLAYER: Any | None = None


def create_app(tables: dict[str, TableActor] | None = None) -> FastAPI:
    table_registry: dict[str, TableActor] = {} if tables is None else tables
    app = FastAPI(title="Guandan Server", version="0.1.0")
    _register_audit_middleware(app)
    router = create_router(table_registry)
    app.include_router(router)
    register_websocket_routes(app, table_registry)
    register_web_ui(app)
    return app


def _register_audit_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def audit_http_request(request: Request, call_next: Any) -> Response:
        audit_enabled = audit_log_enabled()
        if not audit_enabled and not trace_log_enabled():
            return await call_next(request)

        started_at = time.monotonic()
        request_body = await request.body()
        parsed_request_body = parse_json_body(request_body)
        request_log = {
            "method": request.method,
            "path": request.url.path,
            "query": query_dict(request.url.query),
        }
        trace_event(
            "server.http.request_received",
            client=client_host(request.client),
            request=request_log,
        )
        replayed_request = Request(request.scope, _receive_once(request_body))
        response_body = b""
        status = 500
        try:
            response = await call_next(replayed_request)
            status = response.status_code
            async for chunk in response.body_iterator:
                response_body += chunk
            log_response = {
                "status": status,
            }
            log_fields = {
                "client": client_host(request.client),
                "duration_ms": _duration_ms(started_at),
                "request": request_log,
                "response": log_response,
            }
            if status >= 500:
                error_event("server.http.response_sent", **log_fields)
            else:
                debug_event("server.http.response_sent", **log_fields)
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
                background=response.background,
            )
        except Exception as exc:
            error_event(
                "server.http.request_failed",
                client=client_host(request.client),
                duration_ms=_duration_ms(started_at),
                request=request_log,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            if audit_enabled:
                write_audit_entry(
                    make_audit_entry(
                        method=request.method,
                        path=request.url.path,
                        query=request.url.query,
                        status=status,
                        started_at=started_at,
                        request_body=parsed_request_body,
                        response_body=parse_json_body(response_body),
                        client=request.client,
                    )
                )


def _duration_ms(started_at: float) -> float:
    return round((time.monotonic() - started_at) * 1000, 3)


def _receive_once(body: bytes) -> Any:
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


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
        "/tables/{table_id}/bot-action",
        response_model=CommandResponse,
        responses={400: {"model": RejectionResponse}, 404: {"model": ErrorResponse}},
    )
    async def bot_action(table_id: str, request: BotActionRequest) -> CommandResponse | JSONResponse:
        actor = _actor_or_404(tables, table_id)
        controller = actor.state.controllers.get(request.seat)
        if controller is None or controller.id != request.controller_id:
            raise HTTPException(status_code=400, detail="controller is not attached to that seat")
        if controller.kind not in {ControllerKind.LOCAL_BOT, ControllerKind.EXTERNAL_AGENT}:
            raise HTTPException(status_code=400, detail="bot action requires a bot or agent controller")
        try:
            snapshot = actor.seat_snapshot(request.seat, request.controller_id)
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if snapshot.legal_action is None:
            raise HTTPException(status_code=400, detail="seat is not waiting for an action")
        selected_action = _rl_agent_player().choose_action(_agent_request_from_snapshot(snapshot, request.request_id))
        try:
            command = _command_from_agent_action(selected_action, request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"RL agent selected an invalid action: {exc}") from exc
        return await _command_response(
            actor,
            command,
            request,
            extra={"selected_action": selected_action},
        )

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


def _rl_agent_player() -> Any:
    global _RL_AGENT_PLAYER
    if _RL_AGENT_PLAYER is None:
        from npc.rl_agent import RlAgentPlayer

        _RL_AGENT_PLAYER = RlAgentPlayer()
    return _RL_AGENT_PLAYER


def _agent_request_from_snapshot(snapshot: SeatSnapshot, request_id: str | None) -> ActionRequest:
    public = PublicTableSnapshotSchema.from_snapshot(snapshot.public).model_dump(mode="json")
    prompt = {
        "kind": snapshot.legal_action,
        "current_level": public.get("current_level", "2"),
        "current_trick": public.get("current_trick"),
        "eligible_card_ids": list(snapshot.eligible_card_ids),
        "tribute_from": snapshot.tribute_from.value if snapshot.tribute_from else None,
        "tribute_to": snapshot.tribute_to.value if snapshot.tribute_to else None,
        "return_rank_at_most_ten": snapshot.return_rank_at_most_ten,
    }
    players_by_seat = {
        seat: player.get("display_name", seat)
        for seat, player in dict(public.get("seats", {})).items()
        if isinstance(player, dict)
    }
    return ActionRequest(
        request_id or f"web-bot-{snapshot.seat.value}-{snapshot.public.event_seq}",
        prompt,
        {
            "table_id": snapshot.public.table_id,
            "seat": snapshot.seat.value,
            "hand": list(snapshot.hand),
            "players_by_seat": players_by_seat,
            "public": public,
        },
    )


def _command_from_agent_action(action: dict[str, Any], request: BotActionRequest) -> Any:
    action_type = action.get("type")
    if action_type == "pass":
        return Pass(request.controller_id, request.seat)
    if action_type == "play_cards":
        card_ids = tuple(str(card_id) for card_id in action.get("card_ids", []))
        if not card_ids:
            raise ValueError("play_cards requires card_ids")
        declared_type = action.get("declared_type")
        return PlayCards(
            request.controller_id,
            request.seat,
            card_ids,
            declared_type=str(declared_type) if declared_type is not None else None,
        )
    if action_type == "submit_tribute":
        return SubmitTribute(request.controller_id, request.seat, str(action["card_id"]))
    if action_type == "return_tribute":
        return ReturnTribute(request.controller_id, request.seat, str(action["card_id"]))
    raise ValueError(f"unsupported action type: {action_type}")


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
        events=[EventSchema.from_event(event) for event in public_events(result.events)],
        event_seq=actor.state.event_seq,
        snapshot=PublicTableSnapshotSchema.from_snapshot(actor.public_snapshot()),
        replayed=result.replayed,
    )
    if extra:
        for key, value in extra.items():
            setattr(response, key, value)
    return response
