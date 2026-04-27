from __future__ import annotations

from enum import StrEnum


class Seat(StrEnum):
    EAST = "E"
    SOUTH = "S"
    WEST = "W"
    NORTH = "N"


class Team(StrEnum):
    EAST_WEST = "EW"
    SOUTH_NORTH = "SN"


COUNTER_CLOCKWISE_ORDER: tuple[Seat, ...] = (Seat.EAST, Seat.NORTH, Seat.WEST, Seat.SOUTH)
SEATS: tuple[Seat, ...] = (Seat.EAST, Seat.SOUTH, Seat.WEST, Seat.NORTH)


def team_for_seat(seat: Seat) -> Team:
    if seat in {Seat.EAST, Seat.WEST}:
        return Team.EAST_WEST
    return Team.SOUTH_NORTH


def partner_for_seat(seat: Seat) -> Seat:
    return {
        Seat.EAST: Seat.WEST,
        Seat.WEST: Seat.EAST,
        Seat.SOUTH: Seat.NORTH,
        Seat.NORTH: Seat.SOUTH,
    }[seat]


def next_seat(seat: Seat, active_seats: set[Seat] | frozenset[Seat] | None = None) -> Seat:
    active = set(SEATS) if active_seats is None else set(active_seats)
    order = COUNTER_CLOCKWISE_ORDER
    start = order.index(seat)
    for offset in range(1, len(order) + 1):
        candidate = order[(start + offset) % len(order)]
        if candidate in active:
            return candidate
    raise ValueError("no active seat is available")
