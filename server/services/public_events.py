from __future__ import annotations

from server.domain.events import Event


def public_event(event: Event) -> Event:
    if event.type != "CardsDealt":
        return event
    hands = event.payload.get("hands")
    if not isinstance(hands, dict):
        return Event(seq=event.seq, type=event.type, payload={})
    return Event(
        seq=event.seq,
        type=event.type,
        payload={
            "hand_counts": {
                str(seat): len(cards)
                for seat, cards in hands.items()
                if isinstance(cards, (list, tuple))
            }
        },
    )


def public_events(events: tuple[Event, ...]) -> tuple[Event, ...]:
    return tuple(public_event(event) for event in events)
