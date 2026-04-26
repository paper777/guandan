from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from guandan.domain.seats import Seat


class RejectCode(StrEnum):
    NOT_YOUR_TURN = "NOT_YOUR_TURN"
    CONTROLLER_NOT_ATTACHED = "CONTROLLER_NOT_ATTACHED"
    INSUFFICIENT_CONTROLLER_CAPABILITY = "INSUFFICIENT_CONTROLLER_CAPABILITY"
    INVALID_PHASE = "INVALID_PHASE"
    SEAT_OCCUPIED = "SEAT_OCCUPIED"
    TABLE_NOT_FULL = "TABLE_NOT_FULL"
    NOT_ALL_READY = "NOT_ALL_READY"
    CARD_NOT_OWNED = "CARD_NOT_OWNED"
    INVALID_HAND_TYPE = "INVALID_HAND_TYPE"
    AMBIGUOUS_WILD_CARD_DECLARATION = "AMBIGUOUS_WILD_CARD_DECLARATION"
    DOES_NOT_BEAT_CURRENT_HAND = "DOES_NOT_BEAT_CURRENT_HAND"
    CANNOT_PASS_WHEN_LEADING = "CANNOT_PASS_WHEN_LEADING"
    INVALID_TRIBUTE_CARD = "INVALID_TRIBUTE_CARD"
    INVALID_RETURN_CARD = "INVALID_RETURN_CARD"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_PROTOCOL_ERROR = "AGENT_PROTOCOL_ERROR"
    ACTION_TIMEOUT = "ACTION_TIMEOUT"
    STALE_ACTION_PROMPT = "STALE_ACTION_PROMPT"
    INVALID_TIMEOUT_CONFIG = "INVALID_TIMEOUT_CONFIG"


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    type: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CommandRejected:
    code: RejectCode
    message: str


@dataclass(frozen=True, slots=True)
class ReducerResult:
    state: Any
    events: tuple[Event, ...] = ()
    rejection: CommandRejected | None = None


def event_payload_seat(seat: Seat) -> str:
    return seat.value
