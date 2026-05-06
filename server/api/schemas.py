from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from server.domain.events import CommandRejected, Event
from server.domain.seats import Seat
from server.domain.state import MatchPhase
from server.services.snapshots import PublicTableSnapshot, SeatSnapshot


class TableCreateResponse(BaseModel):
    table_id: str
    action_timeout_seconds: int = 45
    timeout_fallback: Literal["auto_pass"] = "auto_pass"


class TableCreateRequest(BaseModel):
    action_timeout_seconds: int = Field(default=45, ge=5, le=300)
    timeout_fallback: Literal["auto_pass"] = "auto_pass"


class TableListResponse(BaseModel):
    tables: list[str]


class ErrorResponse(BaseModel):
    error: str


class PublicPlayerSchema(BaseModel):
    player_id: str
    display_name: str
    kind: str
    controlled: bool


class PublicTableSnapshotSchema(BaseModel):
    table_id: str
    phase: MatchPhase
    seats: dict[Seat, PublicPlayerSchema]
    hand_counts: dict[Seat, int]
    current_turn: Seat | None
    finish_order: tuple[Seat, ...]
    event_seq: int
    current_level: str = "2"
    action_deadline_epoch_ms: int | None = None
    action_timeout_seconds: int = 45
    acting_seat: Seat | None = None
    current_trick: dict[str, Any] | None = None

    @classmethod
    def from_snapshot(cls, snapshot: PublicTableSnapshot) -> PublicTableSnapshotSchema:
        return cls.model_validate(snapshot, from_attributes=True)


class SeatSnapshotSchema(BaseModel):
    public: PublicTableSnapshotSchema
    seat: Seat
    hand: tuple[str, ...]
    legal_action: str | None

    @classmethod
    def from_snapshot(cls, snapshot: SeatSnapshot) -> SeatSnapshotSchema:
        return cls(
            public=PublicTableSnapshotSchema.from_snapshot(snapshot.public),
            seat=snapshot.seat,
            hand=snapshot.hand,
            legal_action=snapshot.legal_action,
        )


class JoinTableRequest(BaseModel):
    seat: Seat
    player_id: str | None = None
    controller_id: str | None = None
    display_name: str | None = None


class AgentJoinRequest(JoinTableRequest):
    agent_url: str | None = None
    timeout_ms: int = Field(default=3000, ge=100)
    timeout_fallback: Literal["auto_pass", "simple_bot_takeover", "forfeit"] = "auto_pass"
    shared_secret: str | None = None


class ControllerCommandRequest(BaseModel):
    controller_id: str
    seat: Seat
    request_id: str | None = None


class ReadyRequest(ControllerCommandRequest):
    pass


class StartMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    controller_id: str | None = None


class PlayCardsRequest(ControllerCommandRequest):
    card_ids: tuple[str, ...]
    declared_type: str | None = None
    wild_assignments: tuple[dict[str, str], ...] = ()


class PassRequest(ControllerCommandRequest):
    pass


class SubmitTributeRequest(ControllerCommandRequest):
    card_id: str


class ReturnTributeRequest(ControllerCommandRequest):
    card_id: str


class EventSchema(BaseModel):
    seq: int
    type: str
    payload: dict[str, Any]

    @classmethod
    def from_event(cls, event: Event) -> EventSchema:
        return cls.model_validate(event, from_attributes=True)


class RejectionSchema(BaseModel):
    code: str
    message: str

    @classmethod
    def from_rejection(cls, rejection: CommandRejected) -> RejectionSchema:
        return cls(code=rejection.code.value, message=rejection.message)


class CommandResponse(BaseModel):
    events: list[EventSchema] = Field(default_factory=list)
    event_seq: int
    snapshot: PublicTableSnapshotSchema | None = None
    replayed: bool = False
    player_id: str | None = None
    controller_id: str | None = None


class RejectionResponse(BaseModel):
    rejection: RejectionSchema
    event_seq: int


class WebSocketClientMessage(BaseModel):
    type: Literal["snapshot", "ready", "start", "play_cards", "pass", "submit_tribute", "return_tribute"]
    request_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WebSocketServerMessage(BaseModel):
    type: Literal["snapshot", "response", "error"]
    status: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool
    service: str


class VersionResponse(BaseModel):
    version: str


class ReplayResponse(BaseModel):
    match_id: str
    events: list[EventSchema]
