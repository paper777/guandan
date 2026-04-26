from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from guandan.domain.controllers import ControllerKind, ControllerRef, PlayerRef
from guandan.domain.seats import Seat


@dataclass(frozen=True, slots=True)
class JoinTable:
    player: PlayerRef
    controller: ControllerRef
    requested_seat: Seat


@dataclass(frozen=True, slots=True)
class AttachController:
    controller: ControllerRef


@dataclass(frozen=True, slots=True)
class DetachController:
    controller_id: str
    seat: Seat


@dataclass(frozen=True, slots=True)
class LeaveTable:
    player_id: str


@dataclass(frozen=True, slots=True)
class Ready:
    controller_id: str
    seat: Seat


@dataclass(frozen=True, slots=True)
class StartMatch:
    seed: str | int | bytes | None = None


@dataclass(frozen=True, slots=True)
class SubmitTribute:
    controller_id: str
    seat: Seat
    card_id: str


@dataclass(frozen=True, slots=True)
class ReturnTribute:
    controller_id: str
    seat: Seat
    card_id: str


@dataclass(frozen=True, slots=True)
class PlayCards:
    controller_id: str
    seat: Seat
    card_ids: tuple[str, ...]
    declared_type: str | None = None
    wild_assignments: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Pass:
    controller_id: str
    seat: Seat


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    controller_id: str
    seat: Seat | None = None


@dataclass(frozen=True, slots=True)
class RegisterControllerTemplate:
    kind: ControllerKind
    label: str


Command = (
    JoinTable
    | AttachController
    | DetachController
    | LeaveTable
    | Ready
    | StartMatch
    | SubmitTribute
    | ReturnTribute
    | PlayCards
    | Pass
    | RequestSnapshot
    | RegisterControllerTemplate
)

ActionType = Literal["play_cards", "pass", "ready"]
