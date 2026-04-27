from __future__ import annotations

from dataclasses import replace

from server.domain.cards import Rank, resolve_cards
from server.domain.controllers import ControllerCapability, ControllerKind, ControllerRef, PlayerKind, PlayerRef
from server.domain.events import Event
from server.domain.hand_types import parse_hand
from server.domain.seats import SEATS, Seat, Team, next_seat
from server.domain.state import (
    DealResult,
    DealState,
    MatchPhase,
    MatchState,
    ScoreState,
    TributeObligation,
    TributeState,
    TrickState,
)


def rebuild_state_from_events(table_id: str, events: tuple[Event, ...]) -> MatchState:
    state = MatchState(table_id=table_id)
    pending_leader: Seat | None = None
    pending_deal_result: dict[str, object] | None = None

    for event in events:
        if event.type == "PlayerSeated":
            state = _apply_player_seated(state, event)
        elif event.type == "ControllerAttached":
            state = _apply_controller_attached(state, event)
        elif event.type == "ControllerDetached":
            seat = Seat(event.payload["seat"])
            controllers = dict(state.controllers)
            controllers.pop(seat, None)
            ready_seats = frozenset(item for item in state.ready_seats if item != seat)
            state = replace(state, controllers=controllers, ready_seats=ready_seats)
        elif event.type == "PlayerReady":
            state = replace(state, ready_seats=frozenset({*state.ready_seats, Seat(event.payload["seat"])}))
        elif event.type == "MatchStarted":
            pass
        elif event.type == "DealStarted":
            pending_leader = Seat(event.payload["leader"])
        elif event.type == "CardsDealt":
            if pending_leader is None:
                raise ValueError("CardsDealt event appeared before DealStarted")
            state = _apply_cards_dealt(state, event, pending_leader)
            pending_leader = None
        elif event.type == "TributeRequired":
            state = _apply_tribute_required(state, event)
        elif event.type == "TributeResisted":
            state = _apply_tribute_resisted(state, event)
        elif event.type == "TributePaid":
            state = _apply_tribute_paid(state, event)
        elif event.type == "TributeReturned":
            state = _apply_tribute_returned(state, event)
        elif event.type == "TributeComplete":
            state = _apply_tribute_complete(state, event)
        elif event.type == "CardsPlayed":
            state = _apply_cards_played(state, event)
        elif event.type == "TenCardReport":
            state = _apply_ten_card_report(state, event)
        elif event.type == "PlayerFinished":
            state = _apply_player_finished(state, event)
        elif event.type == "PlayerPassed":
            state = _apply_player_passed(state, event)
        elif event.type == "TrickEnded":
            state = _apply_trick_ended(state, event)
        elif event.type == "DealEnded":
            pending_deal_result = dict(event.payload)
            state = _apply_deal_ended(state, event)
        elif event.type == "LevelAdvanced":
            state = _apply_level_advanced(state, event, pending_deal_result, match_complete=False)
        elif event.type == "MatchEnded":
            if state.last_deal_result is not None:
                state = replace(state, last_deal_result=replace(state.last_deal_result, match_complete=True))
            state = replace(state, phase=MatchPhase.MATCH_COMPLETE)
        elif event.type in {"ActionPrompted", "ActionTimedOut", "TimeoutFallbackApplied"}:
            pass
        else:
            raise ValueError(f"unsupported event type during replay: {event.type}")
        state = replace(state, event_seq=event.seq)

    return state


def _apply_player_seated(state: MatchState, event: Event) -> MatchState:
    seat = Seat(event.payload["seat"])
    player_payload = event.payload.get("player", {})
    controller_payload = event.payload.get("controller", {})
    player = PlayerRef(
        id=player_payload.get("id", event.payload.get("player_id", "")),
        display_name=player_payload.get("display_name", player_payload.get("id", "")),
        kind=PlayerKind(player_payload.get("kind", PlayerKind.HUMAN.value)),
    )
    seats = dict(state.seats)
    seats[seat] = player
    controllers = dict(state.controllers)
    if controller_payload:
        controllers[seat] = _controller_from_payload(controller_payload)
    phase = MatchPhase.READY_CHECK if all(item in seats and item in controllers for item in SEATS) else state.phase
    return replace(state, seats=seats, controllers=controllers, phase=phase)


def _apply_controller_attached(state: MatchState, event: Event) -> MatchState:
    controller = _controller_from_payload(event.payload["controller"])
    controllers = dict(state.controllers)
    controllers[controller.seat] = controller
    return replace(state, controllers=controllers)


def _apply_cards_dealt(state: MatchState, event: Event, leader: Seat) -> MatchState:
    hands = {Seat(seat): tuple(card_ids) for seat, card_ids in event.payload["hands"].items()}
    deal = DealState(
        hands=hands,
        active_seats=frozenset(SEATS),
        finish_order=(),
        leader=leader,
        turn=leader,
        current_trick=TrickState(lead_seat=leader),
    )
    return replace(state, phase=MatchPhase.PLAYING, deal=deal)


def _apply_tribute_required(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("TributeRequired event appeared before CardsDealt")
    obligations = tuple(
        TributeObligation(giver=Seat(item["giver"]), receiver=Seat(item["receiver"]))
        for item in event.payload["obligations"]
    )
    leader_after = Seat(event.payload["leader_after"])
    tribute = TributeState(obligations=obligations, leader_after=leader_after)
    deal = replace(state.deal, tribute=tribute)
    return replace(state, phase=MatchPhase.TRIBUTE, deal=deal)


def _apply_tribute_resisted(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("TributeResisted event appeared before CardsDealt")
    leader = Seat(event.payload["leader"])
    deal = replace(state.deal, leader=leader, turn=leader, current_trick=TrickState(lead_seat=leader), tribute=None)
    return replace(state, phase=MatchPhase.PLAYING, deal=deal)


def _apply_tribute_paid(state: MatchState, event: Event) -> MatchState:
    if state.deal is None or state.deal.tribute is None:
        raise ValueError("TributePaid event appeared outside tribute")
    giver = Seat(event.payload["giver"])
    receiver = Seat(event.payload["receiver"])
    card_id = event.payload["card_id"]
    hands = _move_card(state.deal.hands, giver, receiver, card_id)
    obligations = tuple(
        replace(item, tribute_card_id=card_id) if item.giver == giver and item.receiver == receiver else item
        for item in state.deal.tribute.obligations
    )
    deal = replace(state.deal, hands=hands, tribute=replace(state.deal.tribute, obligations=obligations))
    return replace(state, deal=deal)


def _apply_tribute_returned(state: MatchState, event: Event) -> MatchState:
    if state.deal is None or state.deal.tribute is None:
        raise ValueError("TributeReturned event appeared outside tribute")
    giver = Seat(event.payload["giver"])
    receiver = Seat(event.payload["receiver"])
    card_id = event.payload["card_id"]
    hands = _move_card(state.deal.hands, receiver, giver, card_id)
    obligations = tuple(
        replace(item, return_card_id=card_id) if item.giver == giver and item.receiver == receiver else item
        for item in state.deal.tribute.obligations
    )
    deal = replace(state.deal, hands=hands, tribute=replace(state.deal.tribute, obligations=obligations))
    return replace(state, deal=deal)


def _apply_tribute_complete(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("TributeComplete event appeared before CardsDealt")
    leader = Seat(event.payload["leader"])
    deal = replace(state.deal, tribute=None, leader=leader, turn=leader, current_trick=TrickState(lead_seat=leader))
    return replace(state, phase=MatchPhase.PLAYING, deal=deal)


def _apply_cards_played(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("CardsPlayed event appeared before CardsDealt")
    seat = Seat(event.payload["seat"])
    card_ids = tuple(event.payload["card_ids"])
    hand = list(state.deal.hand_for(seat))
    for card_id in card_ids:
        hand.remove(card_id)
    hands = dict(state.deal.hands)
    hands[seat] = tuple(hand)
    active = set(state.deal.active_seats)
    if event.payload["remaining_count"] == 0:
        active.discard(seat)
    played_hand = parse_hand(resolve_cards(card_ids), event.payload["hand_type"], level=state.current_level)
    trick = replace(state.deal.current_trick, last_play=played_hand, last_play_seat=seat, pass_count=0)
    turn = next_seat(seat, active) if active else seat
    deal = replace(state.deal, hands=hands, active_seats=frozenset(active), turn=turn, current_trick=trick)
    return replace(state, deal=deal)


def _apply_ten_card_report(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("TenCardReport event appeared before CardsDealt")
    seat = Seat(event.payload["seat"])
    return replace(state, deal=replace(state.deal, report_10_done=frozenset({*state.deal.report_10_done, seat})))


def _apply_player_finished(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("PlayerFinished event appeared before CardsDealt")
    seat = Seat(event.payload["seat"])
    active = frozenset(item for item in state.deal.active_seats if item != seat)
    finish_order = state.deal.finish_order if seat in state.deal.finish_order else (*state.deal.finish_order, seat)
    return replace(state, deal=replace(state.deal, active_seats=active, finish_order=finish_order))


def _apply_player_passed(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("PlayerPassed event appeared before CardsDealt")
    seat = Seat(event.payload["seat"])
    trick = replace(state.deal.current_trick, pass_count=state.deal.current_trick.pass_count + 1)
    turn = next_seat(seat, state.deal.active_seats)
    return replace(state, deal=replace(state.deal, turn=turn, current_trick=trick))


def _apply_trick_ended(state: MatchState, event: Event) -> MatchState:
    if state.deal is None:
        raise ValueError("TrickEnded event appeared before CardsDealt")
    leader = Seat(event.payload["next_leader"])
    deal = replace(state.deal, leader=leader, turn=leader, current_trick=TrickState(lead_seat=leader))
    return replace(state, deal=deal)


def _apply_deal_ended(state: MatchState, event: Event) -> MatchState:
    finish_order = tuple(Seat(seat) for seat in event.payload["finish_order"])
    deal = replace(state.deal, finish_order=finish_order) if state.deal is not None else None
    return replace(state, phase=MatchPhase.DEAL_COMPLETE, deal=deal)


def _apply_level_advanced(
    state: MatchState,
    event: Event,
    deal_payload: dict[str, object] | None,
    *,
    match_complete: bool,
) -> MatchState:
    team = Team(event.payload["team"])
    previous_level = Rank(event.payload["previous_level"])
    next_level_value = Rank(event.payload["next_level"])
    level_by_team = dict(state.scores.level_by_team)
    level_by_team[team] = next_level_value
    scores = ScoreState(level_by_team=level_by_team)
    last_deal_result = state.last_deal_result
    if deal_payload is not None:
        last_deal_result = DealResult(
            finish_order=tuple(Seat(seat) for seat in deal_payload["finish_order"]),
            winning_team=Team(deal_payload["winning_team"]),
            advance_count=int(deal_payload["advance_count"]),
            previous_level=previous_level,
            next_level=next_level_value,
            match_complete=match_complete,
        )
    return replace(state, current_level=next_level_value, scores=scores, last_deal_result=last_deal_result)


def _controller_from_payload(payload: dict[str, object]) -> ControllerRef:
    return ControllerRef(
        id=str(payload["id"]),
        kind=ControllerKind(payload["kind"]),
        seat=Seat(payload["seat"]),
        player_id=str(payload["player_id"]),
        capabilities=frozenset(ControllerCapability(item) for item in payload.get("capabilities", ())),
    )


def _move_card(hands: dict[Seat, tuple[str, ...]], source: Seat, target: Seat, card_id: str) -> dict[Seat, tuple[str, ...]]:
    next_hands = {seat: list(hand) for seat, hand in hands.items()}
    next_hands[source].remove(card_id)
    next_hands[target].append(card_id)
    return {seat: tuple(hand) for seat, hand in next_hands.items()}
