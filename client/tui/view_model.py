from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from client.tui.render import format_card_id, format_friend_mark, sort_card_ids
from client.types import JsonObject


SEAT_LAYOUT = ("N", "W", "E", "S")
PLAY_ACTIONS = {"lead", "play_or_pass"}


@dataclass(frozen=True, slots=True)
class CardView:
    index: int
    card_id: str
    label: str
    selected: bool = False
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class SeatView:
    seat: str
    display_name: str
    hand_count: int
    metadata: str = ""
    is_viewer: bool = False
    is_partner: bool = False
    is_acting: bool = False
    is_finished: bool = False


@dataclass(frozen=True, slots=True)
class TrickView:
    seat: str
    hand_type: str
    cards: str
    primary_rank: str = ""

    @property
    def summary(self) -> str:
        if not self.seat:
            return "No active trick"
        rank = f" rank {self.primary_rank}" if self.primary_rank else ""
        return f"{self.seat} {self.hand_type}{rank}: {self.cards}"


@dataclass(frozen=True, slots=True)
class ActionRow:
    seq: str
    seat: str
    action: str
    cards: str = ""
    detail: str = ""
    tone: str = "normal"


@dataclass(frozen=True, slots=True)
class ActionAvailability:
    can_play: bool = False
    can_pass: bool = False
    can_submit_tribute: bool = False
    can_return_tribute: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TableView:
    table_id: str
    phase: str
    turn: str
    acting_seat: str
    deal_id: str
    level_summary: str
    timer: str
    seats: tuple[SeatView, ...]
    trick: TrickView
    finish_order: tuple[str, ...]


def card_views(
    hand: object,
    *,
    selected_ids: Iterable[str] = (),
    eligible_card_ids: Iterable[str] = (),
) -> tuple[CardView, ...]:
    selected = set(selected_ids)
    eligible = set(eligible_card_ids)
    has_eligibility_filter = bool(eligible)
    views: list[CardView] = []
    for index, card_id in enumerate(sort_card_ids(hand), start=1):
        views.append(
            CardView(
                index=index,
                card_id=card_id,
                label=format_card_id(card_id),
                selected=card_id in selected,
                eligible=(not has_eligibility_filter or card_id in eligible),
            )
        )
    return tuple(views)


def action_availability(seat_snapshot: JsonObject | None, selected_ids: Iterable[str]) -> ActionAvailability:
    if not seat_snapshot:
        return ActionAvailability(reason="Waiting for a private seat snapshot.")
    selected = tuple(selected_ids)
    legal_action = seat_snapshot.get("legal_action")
    if legal_action in PLAY_ACTIONS:
        return ActionAvailability(
            can_play=bool(selected),
            can_pass=legal_action == "play_or_pass",
            reason="" if selected else "Select one or more cards.",
        )
    if legal_action == "tribute":
        return ActionAvailability(
            can_submit_tribute=len(selected) == 1,
            reason="" if len(selected) == 1 else "Select exactly one tribute card.",
        )
    if legal_action == "return_tribute":
        return ActionAvailability(
            can_return_tribute=len(selected) == 1,
            reason="" if len(selected) == 1 else "Select exactly one return card.",
        )
    return ActionAvailability(reason="Waiting for your turn.")


def selected_card_ids_in_hand_order(hand: object, selected_ids: Iterable[str]) -> tuple[str, ...]:
    selected = set(selected_ids)
    return tuple(card_id for card_id in sort_card_ids(hand) if card_id in selected)


def table_view(
    snapshot: JsonObject,
    *,
    viewer_seat: object = None,
    npc_metadata: dict[str, str] | None = None,
) -> TableView:
    return TableView(
        table_id=str(snapshot.get("table_id") or "-"),
        phase=str(snapshot.get("phase") or "-"),
        turn=str(snapshot.get("current_turn") or "-"),
        acting_seat=str(snapshot.get("acting_seat") or snapshot.get("current_turn") or "-"),
        deal_id=str(snapshot.get("deal_id", 0)),
        level_summary=_level_summary(snapshot, viewer_seat),
        timer=_timer_text(snapshot),
        seats=seat_views(snapshot, viewer_seat=viewer_seat, npc_metadata=npc_metadata),
        trick=trick_view(snapshot.get("current_trick")),
        finish_order=tuple(str(seat) for seat in snapshot.get("finish_order") or ()),
    )


def seat_views(
    snapshot: JsonObject,
    *,
    viewer_seat: object = None,
    npc_metadata: dict[str, str] | None = None,
) -> tuple[SeatView, ...]:
    seats = snapshot.get("seats")
    hand_counts = snapshot.get("hand_counts")
    acting = str(snapshot.get("acting_seat") or snapshot.get("current_turn") or "")
    finished = {str(seat) for seat in snapshot.get("finish_order") or ()}
    metadata = npc_metadata or {}
    views: list[SeatView] = []
    for seat in SEAT_LAYOUT:
        player = seats.get(seat) if isinstance(seats, dict) else None
        name = player.get("display_name", "-") if isinstance(player, dict) else "-"
        count = hand_counts.get(seat, 0) if isinstance(hand_counts, dict) else 0
        views.append(
            SeatView(
                seat=seat,
                display_name=str(name or "-"),
                hand_count=int(count) if isinstance(count, int) else 0,
                metadata=metadata.get(seat, ""),
                is_viewer=str(viewer_seat) == seat,
                is_partner=bool(format_friend_mark(seat, viewer_seat)),
                is_acting=acting == seat,
                is_finished=seat in finished,
            )
        )
    return tuple(views)


def trick_view(value: object) -> TrickView:
    if not isinstance(value, dict):
        return TrickView("", "", "")
    return TrickView(
        seat=str(value.get("last_play_seat") or ""),
        hand_type=str(value.get("hand_type") or "played"),
        cards=card_labels(value.get("card_ids", ())),
        primary_rank=str(value.get("primary_rank") or ""),
    )


def action_rows_from_response(response: JsonObject) -> tuple[ActionRow, ...]:
    events = response.get("events")
    if not isinstance(events, list):
        return ()
    current_trick = _response_current_trick(response)
    return tuple(
        row
        for event in events
        if isinstance(event, dict) and event.get("type") != "ActionPrompted"
        for row in (action_row_from_event(event, current_trick=current_trick),)
    )


def action_row_from_event(event: JsonObject, *, current_trick: JsonObject | None = None) -> ActionRow:
    payload = event.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    event_type = str(event.get("type") or "")
    seq = str(event.get("seq") or "-")
    seat = str(payload.get("seat") or payload.get("giver") or payload.get("receiver") or "")
    if event_type == "CardsPlayed":
        return ActionRow(
            seq,
            seat,
            f"played {payload.get('hand_type') or 'cards'}",
            card_labels(payload.get("card_ids", ())),
            tone="play",
        )
    if event_type == "PlayerPassed":
        last_play = trick_view(current_trick).summary if current_trick else ""
        return ActionRow(seq, seat, "passed", detail=last_play, tone="pass")
    if event_type == "ActionTimedOut":
        return ActionRow(seq, str(payload.get("seat") or ""), "timed out", str(payload.get("kind") or ""), tone="warn")
    if event_type == "TimeoutFallbackApplied":
        detail = f"{payload.get('fallback')} submitted {payload.get('command_type')}"
        return ActionRow(seq, str(payload.get("seat") or ""), "timeout fallback", detail=detail, tone="warn")
    if event_type == "TrickEnded":
        return ActionRow(seq, str(payload.get("last_play_seat") or ""), "trick ended", detail=f"next {payload.get('next_leader')}", tone="system")
    if event_type == "PlayerFinished":
        return ActionRow(seq, seat, "finished", detail=f"position {payload.get('position')}", tone="system")
    if event_type == "DealEnded":
        finish = " ".join(str(item) for item in payload.get("finish_order", ()))
        detail = f"{payload.get('winning_team')} advanced {payload.get('advance_count')} ({finish})"
        return ActionRow(seq, "", "deal ended", detail=detail, tone="system")
    if event_type == "LevelAdvanced":
        detail = f"{payload.get('team')} {payload.get('previous_level')} -> {payload.get('next_level')}"
        return ActionRow(seq, "", "level advanced", detail=detail, tone="system")
    if event_type == "MatchEnded":
        return ActionRow(seq, "", "match ended", detail=f"winner {payload.get('winning_team')}", tone="system")
    if event_type == "TributeRequired":
        return ActionRow(seq, "", "tribute required", detail=str(payload.get("obligations", ())), tone="system")
    if event_type == "TributeResisted":
        return ActionRow(seq, "", "tribute resisted", detail=f"leader {payload.get('leader')}", tone="system")
    if event_type == "TributePaid":
        detail = f"{payload.get('giver')} -> {payload.get('receiver')}"
        return ActionRow(seq, str(payload.get("giver") or ""), "paid tribute", card_labels([payload.get("card_id")]), detail, "play")
    if event_type == "TributeReturned":
        detail = f"{payload.get('receiver')} -> {payload.get('giver')}"
        return ActionRow(seq, str(payload.get("receiver") or ""), "returned tribute", card_labels([payload.get("card_id")]), detail, "play")
    if event_type == "TributeComplete":
        return ActionRow(seq, "", "tribute complete", detail=f"leader {payload.get('leader')}", tone="system")
    if event_type == "TenCardReport":
        return ActionRow(seq, seat, "card report", detail=f"{payload.get('remaining_count')} remaining", tone="warn")
    return ActionRow(seq, seat, event_type or "event", detail=str(payload), tone="system")


def system_action_row(message: str) -> ActionRow:
    return ActionRow("-", "", "system", detail=message, tone="system")


def card_labels(card_ids: object) -> str:
    if not isinstance(card_ids, (list, tuple)):
        return str(card_ids)
    return ", ".join(format_card_id(str(card_id)) for card_id in card_ids if card_id)


def _response_current_trick(response: JsonObject) -> JsonObject | None:
    snapshot = response.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    trick = snapshot.get("current_trick")
    return trick if isinstance(trick, dict) else None


def _level_summary(snapshot: JsonObject, viewer_seat: object = None) -> str:
    levels = snapshot.get("level_by_team")
    current_level = str(snapshot.get("current_level", "2"))
    if not isinstance(levels, dict) or not levels:
        return f"Level {current_level}"
    normalized = {str(team): str(level) for team, level in levels.items()}
    viewer_team = _team_for_seat(viewer_seat)
    if viewer_team is None:
        return f"EW {normalized.get('EW', current_level)} / SN {normalized.get('SN', current_level)}"
    opponent_team = "SN" if viewer_team == "EW" else "EW"
    return f"Us {normalized.get(viewer_team, current_level)} / Them {normalized.get(opponent_team, current_level)}"


def _team_for_seat(seat: object) -> str | None:
    if str(seat) in {"E", "W"}:
        return "EW"
    if str(seat) in {"S", "N"}:
        return "SN"
    return None


def _timer_text(snapshot: JsonObject) -> str:
    deadline = snapshot.get("action_deadline_epoch_ms")
    if not isinstance(deadline, int):
        return ""
    remaining = max(0, int((deadline - int(time.time() * 1000) + 999) / 1000))
    return f"{remaining}s"
