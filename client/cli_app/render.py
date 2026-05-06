from __future__ import annotations

import time
from io import StringIO
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from client.api import GuandanClientError, JsonObject
from server.domain.seats import SEATS


SUIT_EMOJI = {
    "S": "♠️ ",
    "H": "♥️ ",
    "D": "♦️ ",
    "C": "♣️ ",
}
SUIT_INPUT_ALIASES = {
    "S": "S",
    "SPADE": "S",
    "SPADES": "S",
    "♠": "S",
    "♠️": "S",
    "H": "H",
    "HEART": "H",
    "HEARTS": "H",
    "♥": "H",
    "♥️": "H",
    "D": "D",
    "DIAMOND": "D",
    "DIAMONDS": "D",
    "♦": "D",
    "♦️": "D",
    "C": "C",
    "CLUB": "C",
    "CLUBS": "C",
    "♣": "C",
    "♣️": "C",
}
RANK_SORT_ORDER = {
    rank: index
    for index, rank in enumerate(("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "SJ", "BJ"))
}
SUIT_SORT_ORDER = {suit: index for index, suit in enumerate(("S", "H", "D", "C"))}
HIDDEN_EVENT_TYPES = {"ActionPrompted"}
PARTNERS = {"E": "W", "W": "E", "S": "N", "N": "S"}


def format_public_snapshot(
    snapshot: JsonObject,
    *,
    viewer_seat: object = None,
    npc_metadata: dict[str, str] | None = None,
) -> str:
    lines = _public_snapshot_lines(snapshot, viewer_seat=viewer_seat, npc_metadata=npc_metadata)
    return _render_panel(lines, title="Table")


def format_seat_snapshot(snapshot: JsonObject, *, npc_metadata: dict[str, str] | None = None) -> str:
    public = snapshot.get("public")
    public_snapshot = public if isinstance(public, dict) else {}
    lines = _public_snapshot_lines(public_snapshot, viewer_seat=snapshot.get("seat"), npc_metadata=npc_metadata)
    lines.append(f"Action: {snapshot.get('legal_action') or '-'}")
    eligible_cards = snapshot.get("eligible_card_ids")
    if isinstance(eligible_cards, (list, tuple)) and eligible_cards:
        lines.append("Eligible cards: " + format_hand(eligible_cards))
    lines.append("Hand: " + format_hand(snapshot.get("hand", ())))
    return _render_panel(lines, title="Your Seat")


def _public_snapshot_lines(
    snapshot: JsonObject,
    *,
    viewer_seat: object = None,
    npc_metadata: dict[str, str] | None = None,
) -> list[str]:
    seats = snapshot.get("seats", {})
    hand_counts = snapshot.get("hand_counts", {})
    current_turn = snapshot.get("current_turn") or "-"
    acting = snapshot.get("acting_seat") or current_turn
    header = f"{snapshot.get('phase', '-')} | Turn {current_turn}"
    if viewer_seat is not None:
        header = f"{snapshot.get('phase', '-')} | Seat {viewer_seat} | Turn {current_turn}"
    if acting != current_turn:
        header += f" | Acting {acting}"
    header += f" | Level {snapshot.get('current_level', '2')}"

    lines = [header]
    timer = format_timer(snapshot)
    if timer is not None:
        lines.append(f"Timer: {timer}")
    metadata = npc_metadata or {}
    lines.extend(_player_lines(seats, hand_counts, metadata, viewer_seat))
    trick = format_trick(snapshot.get("current_trick"))
    if trick:
        lines.append(f"Trick: {trick}")
    finish_order = snapshot.get("finish_order") or ()
    if finish_order:
        lines.append("Finish: " + " ".join(str(seat) for seat in finish_order))
    return lines


def _render_panel(lines: Iterable[str], *, title: str) -> str:
    content = "\n".join(lines)
    console = Console(width=130, color_system=None, force_terminal=False, record=True, file=StringIO())
    console.print(Panel(Text(content), title=title, expand=False))
    return console.export_text(styles=False)


def _player_lines(
    seats: object,
    hand_counts: object,
    npc_metadata: dict[str, str],
    viewer_seat: object,
) -> list[str]:
    east = _directional_seat_lines("E", seats, hand_counts, npc_metadata, viewer_seat)
    south = _directional_seat_lines("S", seats, hand_counts, npc_metadata, viewer_seat)
    west = _directional_seat_lines("W", seats, hand_counts, npc_metadata, viewer_seat)
    north = _directional_seat_lines("N", seats, hand_counts, npc_metadata, viewer_seat)
    middle_gap = max(8, 78 - len(west[0]) - len(east[0]))
    west_meta = west[1] if len(west) > 1 else ""
    east_meta = east[1] if len(east) > 1 else ""
    meta_gap = max(8, 78 - len(west_meta) - len(east_meta))
    lines = [north[0].center(78).rstrip()]
    if len(north) > 1:
        lines.append(north[1].center(78).rstrip())
    lines.append(f"{west[0]}{' ' * middle_gap}{east[0]}")
    if west_meta or east_meta:
        lines.append(f"{west_meta}{' ' * meta_gap}{east_meta}".rstrip())
    lines.append(south[0].center(78).rstrip())
    if len(south) > 1:
        lines.append(south[1].center(78).rstrip())
    return lines


def _directional_seat_lines(
    seat: str,
    seats: object,
    hand_counts: object,
    npc_metadata: dict[str, str] | None = None,
    viewer_seat: object = None,
) -> list[str]:
    player = seats.get(seat) if isinstance(seats, dict) else None
    name = player.get("display_name", "-") if isinstance(player, dict) else "-"
    count = hand_counts.get(seat, 0) if isinstance(hand_counts, dict) else 0
    suffixes: list[str] = []
    if format_friend_mark(seat, viewer_seat):
        suffixes.append("(F)")
    if str(viewer_seat) == seat:
        suffixes.append("(You)")
    suffix = " " + " ".join(suffixes) if suffixes else ""
    metadata = (npc_metadata or {}).get(seat)
    lines = [f"{seat} {name} {count}{suffix}"]
    if metadata:
        lines.append(f"  {metadata}")
    return lines


def format_directional_seat_summary(
    seat: str,
    seats: object,
    hand_counts: object,
    npc_metadata: dict[str, str] | None = None,
    viewer_seat: object = None,
) -> str:
    return " ".join(_directional_seat_lines(seat, seats, hand_counts, npc_metadata, viewer_seat))


def format_seat_summary(
    seat: str,
    seats: object,
    hand_counts: object,
    npc_metadata: dict[str, str] | None = None,
    viewer_seat: object = None,
) -> str:
    player = seats.get(seat) if isinstance(seats, dict) else None
    name = player.get("display_name", "-") if isinstance(player, dict) else "-"
    mark = format_friend_mark(seat, viewer_seat)
    count = hand_counts.get(seat, 0) if isinstance(hand_counts, dict) else 0
    metadata = (npc_metadata or {}).get(seat)
    suffix = f" [{metadata}]" if metadata else ""
    return f"{seat}{mark} {name} {count}{suffix}"


def format_friend_mark(seat: str, viewer_seat: object) -> str:
    return "(F)" if PARTNERS.get(str(viewer_seat)) == seat else ""


def format_npc_metadata(policy: object) -> str:
    config = getattr(policy, "config", None)
    if config is None:
        return ""
    provider = str(getattr(config, "provider_name", "") or "").strip()
    model = str(getattr(config, "model_name", "") or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    return provider or model


def format_command_response(response: JsonObject) -> str:
    events = response.get("events", [])
    visible_events = [event for event in events if isinstance(event, dict) and event.get("type") not in HIDDEN_EVENT_TYPES]
    if not visible_events:
        return "No events.\n"
    current_trick = _response_current_trick(response)
    return "\n".join(format_event(event, current_trick=current_trick) for event in visible_events) + "\n"


def format_event(event: JsonObject, *, current_trick: JsonObject | None = None) -> str:
    payload = event.get("payload", {})
    event_type = event.get("type")
    seq = event.get("seq")
    seat = payload.get("seat") if isinstance(payload, dict) else None
    if event_type == "CardsPlayed":
        cards = format_card_list(payload.get("card_ids", ()))
        return f"{seq}: {seat} played {payload.get('hand_type')} {cards}"
    if event_type == "PlayerPassed":
        trick = format_trick(current_trick)
        if trick:
            return f"{seq}: {seat} passed; last play {trick}"
        return f"{seq}: {seat} passed"
    if event_type == "ActionTimedOut":
        return f"{seq}: {payload.get('seat')} timed out on {payload.get('kind')}"
    if event_type == "TimeoutFallbackApplied":
        return f"{seq}: timeout fallback {payload.get('fallback')} submitted {payload.get('command_type')} for {payload.get('seat')}"
    if event_type == "TrickEnded":
        return f"{seq}: trick ended; next leader {payload.get('next_leader')}"
    if event_type == "DealEnded":
        finish = " ".join(str(item) for item in payload.get("finish_order", ()))
        return f"{seq}: deal ended; {payload.get('winning_team')} advanced {payload.get('advance_count')} ({finish})"
    if event_type == "LevelAdvanced":
        return f"{seq}: {payload.get('team')} level {payload.get('previous_level')} -> {payload.get('next_level')}"
    if event_type == "MatchEnded":
        return f"{seq}: match ended; winner {payload.get('winning_team')}"
    if event_type == "TributeRequired":
        obligations = payload.get("obligations", ())
        return f"{seq}: tribute required {obligations}"
    if event_type == "TributeResisted":
        return f"{seq}: tribute resisted; leader {payload.get('leader')}"
    if event_type == "TributePaid":
        return f"{seq}: {payload.get('giver')} paid tribute to {payload.get('receiver')} {format_card_id(str(payload.get('card_id')))}"
    if event_type == "TributeReturned":
        return f"{seq}: {payload.get('receiver')} returned tribute to {payload.get('giver')} {format_card_id(str(payload.get('card_id')))}"
    if event_type == "TributeComplete":
        return f"{seq}: tribute complete; leader {payload.get('leader')}"
    if event_type == "PlayerFinished":
        return f"{seq}: {seat} finished position {payload.get('position')}"
    if event_type == "TenCardReport":
        return f"{seq}: {seat} has {payload.get('remaining_count')} cards"
    return f"{seq}: {event_type} {payload}"


def _response_current_trick(response: JsonObject) -> JsonObject | None:
    snapshot = response.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    trick = snapshot.get("current_trick")
    return trick if isinstance(trick, dict) else None


def format_timeout_fallback(before: JsonObject, after: JsonObject, *, kind: str | None = None) -> str:
    seat = before.get("acting_seat") or before.get("current_turn") or "-"
    prompt_kind = kind or _infer_prompt_kind(before)
    if prompt_kind == "play_or_pass":
        if _trick_was_passed_out(before, after):
            next_leader = after.get("acting_seat") or after.get("current_turn") or "-"
            return f"{seat} timed out; server fallback passed and ended the trick. {next_leader} leads next."
        return f"{seat} timed out; server fallback passed."
    if prompt_kind == "lead":
        trick = after.get("current_trick")
        if isinstance(trick, dict) and trick.get("last_play_seat") == seat:
            hand_type = trick.get("hand_type") or "cards"
            return f"{seat} timed out; server fallback played {hand_type} {format_card_list(trick.get('card_ids', ()))}."
        return f"{seat} timed out; server fallback played a legal lead."
    if prompt_kind == "tribute":
        return f"{seat} timed out; server fallback submitted tribute."
    if prompt_kind == "return_tribute":
        return f"{seat} timed out; server fallback returned tribute."
    return f"{seat} timed out; server fallback applied."


def _trick_was_passed_out(before: JsonObject, after: JsonObject) -> bool:
    before_trick = before.get("current_trick")
    after_trick = after.get("current_trick")
    return isinstance(before_trick, dict) and after_trick is None


def _infer_prompt_kind(snapshot: JsonObject) -> str | None:
    phase = snapshot.get("phase")
    if phase == "TRIBUTE":
        return "tribute"
    if phase == "PLAYING":
        return "play_or_pass" if isinstance(snapshot.get("current_trick"), dict) else "lead"
    return None


def help_text() -> str:
    return "\n".join(
        (
            "Commands:",
            "  play <card-label-or-id> [<card-label-or-id>...]",
            "    examples: play S3, play H10 C10, play small",
            "  pass",
            "  tribute <card-label-or-id>",
            "  return <card-label-or-id>",
            "  hand",
            "  table",
            "  help",
            "  quit",
        )
    ) + "\n"


def format_client_error(error: GuandanClientError) -> str:
    status = f"HTTP {error.status}: " if error.status is not None else ""
    return f"Error: {status}{error}"


def client_error_code(error: GuandanClientError) -> str | None:
    rejection = error.payload.get("rejection")
    if not isinstance(rejection, dict):
        return None
    code = rejection.get("code")
    return code if isinstance(code, str) else None


def format_timer(snapshot: JsonObject) -> str | None:
    deadline = snapshot.get("action_deadline_epoch_ms")
    if not isinstance(deadline, int):
        return None
    remaining = max(0, int((deadline - int(time.time() * 1000) + 999) / 1000))
    return f"{remaining}s remaining"


def format_hand(card_ids: object) -> str:
    if not isinstance(card_ids, (list, tuple)):
        return ""
    return "  ".join(format_card_id(card_id) for card_id in sort_card_ids(card_ids))


def format_card_list(card_ids: object) -> str:
    if not isinstance(card_ids, (list, tuple)):
        return str(card_ids)
    return "[" + ", ".join(format_card_id(str(card_id)) for card_id in card_ids) + "]"


def format_trick(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    last_play_seat = value.get("last_play_seat", "-")
    hand_type = value.get("hand_type", "-")
    cards = format_card_list(value.get("card_ids", ()))
    primary = value.get("primary_rank")
    rank_suffix = f", rank {primary}" if primary else ""
    return f"{last_play_seat} {hand_type} {cards}{rank_suffix}"


def format_card_id(card_id: str) -> str:
    parts = card_id.split("-")
    deck_label = ""
    if len(parts) == 2 and parts[0].startswith("D"):
        joker = {"SJ": "🃏 Small Joker", "BJ": "🃏 Big Joker"}.get(parts[1], parts[1])
        return f"{deck_label}{joker}"
    if len(parts) == 3 and parts[0].startswith("D"):
        suit = SUIT_EMOJI.get(parts[1], parts[1])
        return f"{deck_label}{suit}{parts[2]}"
    return card_id


def sort_card_ids(card_ids: object) -> tuple[str, ...]:
    if not isinstance(card_ids, (list, tuple)):
        return ()
    return tuple(sorted((str(card_id) for card_id in card_ids), key=card_sort_key))


def card_sort_key(card_id: str) -> tuple[int, int, int, str]:
    parts = card_id.split("-")
    if len(parts) == 3 and parts[0].startswith("D"):
        return (RANK_SORT_ORDER.get(parts[2], 99), SUIT_SORT_ORDER.get(parts[1], 99), _deck_sort_value(parts[0]), card_id)
    if len(parts) == 2 and parts[0].startswith("D"):
        return (RANK_SORT_ORDER.get(parts[1], 99), 99, _deck_sort_value(parts[0]), card_id)
    return (99, 99, 99, card_id)


def _deck_sort_value(deck: str) -> int:
    value = deck.removeprefix("D")
    return int(value) if value.isdecimal() else 99
