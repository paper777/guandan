from __future__ import annotations

import select
import sys
import time

from client.api import GuandanClientError, GuandanHttpClient, JsonObject
from client.cli_app.render import RANK_SORT_ORDER, SUIT_INPUT_ALIASES, format_card_id, sort_card_ids
from client.cli_app.session import CliSession
from client.cli_app.types import InputFn


def read_command(input_fn: InputFn, prompt: str, deadline_epoch_ms: int | None) -> str | None:
    if input_fn is input and deadline_epoch_ms is not None:
        return read_stdin_with_deadline(prompt, deadline_epoch_ms)
    return input_fn(prompt)


def read_stdin_with_deadline(prompt: str, deadline_epoch_ms: int) -> str | None:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    timeout = max(0.0, (deadline_epoch_ms - int(time.time() * 1000)) / 1000)
    try:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        return input("")
    if not readable:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return None
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def submit_human_command(
    client: GuandanHttpClient,
    session: CliSession,
    command: str,
    seat_snapshot: JsonObject | None = None,
) -> JsonObject:
    parts = command.split()
    action = parts[0]
    if action == "play" and len(parts) > 1:
        return client.play_cards(
            session.table_id,
            session.human_seat,
            session.human_controller_id,
            resolve_card_inputs(parts[1:], seat_snapshot),
        )
    if action == "pass" and len(parts) == 1:
        return client.pass_turn(session.table_id, session.human_seat, session.human_controller_id)
    if action == "tribute" and len(parts) == 2:
        card_id = resolve_card_inputs(parts[1:], seat_snapshot)[0]
        return client.submit_tribute(session.table_id, session.human_seat, session.human_controller_id, card_id)
    if action in {"return", "return_tribute"} and len(parts) == 2:
        card_id = resolve_card_inputs(parts[1:], seat_snapshot)[0]
        return client.return_tribute(session.table_id, session.human_seat, session.human_controller_id, card_id)
    raise GuandanClientError(None, f"unsupported command: {command}")


def command_card_ids(command: str, seat_snapshot: JsonObject | None = None) -> tuple[str, ...]:
    parts = command.split()
    if not parts:
        return ()
    action = parts[0]
    if action == "play" and len(parts) > 1:
        return resolve_card_inputs(parts[1:], seat_snapshot)
    if action == "tribute" and len(parts) == 2:
        return resolve_card_inputs(parts[1:], seat_snapshot)
    if action in {"return", "return_tribute"} and len(parts) == 2:
        return resolve_card_inputs(parts[1:], seat_snapshot)
    return ()


def resolve_card_inputs(tokens: list[str], snapshot: JsonObject | None) -> tuple[str, ...]:
    hand = sort_card_ids(snapshot.get("hand", ())) if snapshot is not None else ()
    resolved: list[str] = []
    used: set[str] = set()
    for token in tokens:
        if token.isdecimal():
            index = int(token) - 1
            if index < 0 or index >= len(hand):
                raise GuandanClientError(None, f"hand index out of range: {token}")
            card_id = str(hand[index])
            if card_id in used:
                raise GuandanClientError(None, f"card already selected: {format_card_id(card_id)}")
            resolved.append(card_id)
            used.add(card_id)
        else:
            card_id = resolve_card_label(token, hand, used)
            resolved.append(card_id)
            used.add(card_id)
    return tuple(resolved)


def resolve_card_label(token: str, hand: tuple[str, ...], used: set[str]) -> str:
    if token in hand:
        if token in used:
            raise GuandanClientError(None, f"card already selected: {format_card_id(token)}")
        return token
    if not hand:
        return token
    label = normalized_card_label(token)
    if label is None:
        return token
    for card_id in hand:
        if card_id in used:
            continue
        if normalized_card_label(card_id) == label:
            return card_id
    raise GuandanClientError(None, f"card not in hand: {token}")


def normalized_card_label(value: str) -> tuple[str | None, str] | None:
    raw = value.strip().upper().replace("\ufe0f", "").replace("🃏", "")
    parts = raw.replace("_", "-").replace(":", "-").split("-")
    if len(parts) == 2 and parts[0].startswith("D"):
        rank = normalize_rank(parts[1])
        return (None, rank) if rank in {"SJ", "BJ"} else None
    if len(parts) == 3 and parts[0].startswith("D"):
        suit = SUIT_INPUT_ALIASES.get(parts[1])
        rank = normalize_rank(parts[2])
        return (suit, rank) if suit is not None and rank in RANK_SORT_ORDER else None
    if len(parts) == 2:
        suit = SUIT_INPUT_ALIASES.get(parts[0])
        rank = normalize_rank(parts[1])
        if suit is not None and rank in RANK_SORT_ORDER:
            return (suit, rank)

    token = raw.replace("-", "").replace("_", "").replace(":", "").replace(" ", "")
    if not token:
        return None
    if token in {"SJ", "SMALLJOKER", "SMALL"}:
        return (None, "SJ")
    if token in {"BJ", "BIGJOKER", "BIG"}:
        return (None, "BJ")
    for suit_alias in sorted(SUIT_INPUT_ALIASES, key=len, reverse=True):
        if token.startswith(suit_alias):
            suit = SUIT_INPUT_ALIASES[suit_alias]
            rank = normalize_rank(token[len(suit_alias) :])
            return (suit, rank) if rank in RANK_SORT_ORDER else None
    return None


def normalize_rank(value: str) -> str:
    rank = value.upper()
    if rank == "T":
        return "10"
    return rank
