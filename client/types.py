from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from db.player.types import Player


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ActionRequest:
    request_id: str
    prompt: JsonObject
    snapshot: JsonObject

    @classmethod
    def from_payload(cls, payload: JsonObject) -> "ActionRequest":
        return cls(
            request_id=str(payload.get("request_id", "")),
            prompt=_dict(payload.get("prompt")),
            snapshot=_dict(payload.get("snapshot")),
        )


@dataclass(frozen=True, slots=True)
class Deal:
    deal_id: int | str
    level: str
    level_team: str
    phase: str


@dataclass(slots=True)
class Match:
    match_id: int | str
    level_by_team: dict[str, str]
    current_deal: Deal | None = None
    history_deals: tuple[Deal, ...] = ()
    complete: bool = False


@dataclass(frozen=True, slots=True)
class TableTransition:
    kind: str
    message: str
    match_id: int | str | None = None
    deal_id: int | str | None = None


class SeatRole(StrEnum):
    PLAYER = "player"
    GOSSIPER = "gossiper"
    WITNESS = "witness"


@dataclass(slots=True)
class SeatMember:
    role: SeatRole
    display_name: str
    policy: Player | None = None
    controller_id: str = ""
    profile_key: str = ""
    is_human: bool = False


@dataclass(slots=True)
class SeatMembers:
    seat: str
    player: SeatMember | None = None
    gossiper: SeatMember | None = None
    witnesses: list[SeatMember] = field(default_factory=list)

    def trigger_order(self) -> tuple[SeatMember, ...]:
        ordered: list[SeatMember] = []
        if self.player is not None:
            ordered.append(self.player)
        if self.gossiper is not None:
            ordered.append(self.gossiper)
        ordered.extend(self.witnesses)
        return tuple(ordered)


@dataclass(slots=True)
class Table:
    table_id: str
    current_match: Match | None = None
    history_matches: list[Match] = field(default_factory=list)
    seats: dict[str, SeatMembers] = field(default_factory=dict)

    def members_for(self, seat: str) -> SeatMembers:
        if seat not in self.seats:
            self.seats[seat] = SeatMembers(seat)
        return self.seats[seat]

    def rotate_seat_members(self, seat_map: dict[str, str]) -> None:
        rotated: dict[str, SeatMembers] = {}
        for old_seat, members in self.seats.items():
            new_seat = seat_map.get(old_seat, old_seat)
            members.seat = new_seat
            rotated[new_seat] = members
        self.seats = rotated

    def record_snapshot(self, snapshot: JsonObject) -> tuple[TableTransition, ...]:
        phase = str(snapshot.get("phase") or "")
        if phase in {"", "WAITING_FOR_PLAYERS", "READY_CHECK", "ABORTED"}:
            return ()

        changes: list[TableTransition] = []
        match = self._current_or_new_match(snapshot, changes)
        if phase != "MATCH_COMPLETE":
            self._record_deal_snapshot(match, snapshot, phase, changes)
        else:
            self._complete_current_match(match, changes)
        return tuple(changes)

    def _current_or_new_match(self, snapshot: JsonObject, changes: list[TableTransition]) -> Match:
        match_id = snapshot.get("match_id") or f"{self.table_id}-match-{len(self.history_matches) + 1}"
        if self.current_match is None:
            self.current_match = Match(
                match_id=match_id,
                level_by_team=_string_dict(snapshot.get("level_by_team")),
            )
            changes.append(
                TableTransition(
                    kind="match_started",
                    message=f"Match {len(self.history_matches) + 1} started.",
                    match_id=match_id,
                )
            )
        else:
            self.current_match.level_by_team = _string_dict(snapshot.get("level_by_team"))
        return self.current_match

    def _record_deal_snapshot(
        self,
        match: Match,
        snapshot: JsonObject,
        phase: str,
        changes: list[TableTransition],
    ) -> None:
        deal_id = snapshot.get("deal_id", 0)
        level_by_team = _string_dict(snapshot.get("level_by_team"))
        level_team = _level_team(snapshot)
        level = level_by_team.get(level_team, str(snapshot.get("current_level") or "2"))
        current = match.current_deal
        if current is None or current.deal_id != deal_id:
            if current is not None:
                match.history_deals = (*match.history_deals, current)
            match.current_deal = Deal(deal_id=deal_id, level=level, level_team=level_team, phase=phase)
            changes.append(
                TableTransition(
                    kind="deal_started",
                    message=f"Deal {deal_id} started for {level_team} level {level}.",
                    match_id=match.match_id,
                    deal_id=deal_id,
                )
            )
            return
        if current.phase != phase:
            match.current_deal = Deal(deal_id=current.deal_id, level=level, level_team=level_team, phase=phase)
            if phase == "DEAL_COMPLETE":
                changes.append(
                    TableTransition(
                        kind="deal_complete",
                        message=f"Deal {deal_id} complete.",
                        match_id=match.match_id,
                        deal_id=deal_id,
                    )
                )

    def _complete_current_match(self, match: Match, changes: list[TableTransition]) -> None:
        if match.current_deal is not None:
            match.history_deals = (*match.history_deals, match.current_deal)
            match.current_deal = None
        match.complete = True
        self.history_matches.append(match)
        self.current_match = None
        changes.append(
            TableTransition(
                kind="match_complete",
                message=f"Match {len(self.history_matches)} complete.",
                match_id=match.match_id,
            )
        )


def _dict(value: Any) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _level_team(snapshot: JsonObject) -> str:
    raw = snapshot.get("level_team")
    if isinstance(raw, str) and raw:
        return raw
    acting_seat = snapshot.get("acting_seat") or snapshot.get("current_turn")
    return "EW" if str(acting_seat) in {"E", "W"} else "SN"
