from __future__ import annotations

from dataclasses import replace

from server.domain.cards import CARD_BY_ID, LEVEL_RANKS, Rank, deal_cards, is_red_heart_level_card, resolve_cards
from server.domain.commands import (
    AttachController,
    Command,
    DetachController,
    JoinTable,
    Pass,
    PlayCards,
    Ready,
    ReturnTribute,
    StartMatch,
    SubmitTribute,
)
from server.domain.comparator import RankContext, can_beat
from server.domain.controllers import ControllerCapability
from server.domain.events import CommandRejected, Event, ReducerResult, RejectCode
from server.domain.hand_types import AmbiguousHandError, parse_hand
from server.domain.seats import SEATS, Seat, next_seat, partner_for_seat, team_for_seat
from server.domain.state import DealResult, DealState, MatchPhase, MatchState, TributeObligation, TributeState, TrickState


def reduce_command(state: MatchState, command: Command) -> ReducerResult:
    if isinstance(command, JoinTable):
        return _join_table(state, command)
    if isinstance(command, AttachController):
        return _attach_controller(state, command)
    if isinstance(command, DetachController):
        return _detach_controller(state, command)
    if isinstance(command, Ready):
        return _ready(state, command)
    if isinstance(command, StartMatch):
        return _start_match(state, command)
    if isinstance(command, SubmitTribute):
        return _submit_tribute(state, command)
    if isinstance(command, ReturnTribute):
        return _return_tribute(state, command)
    if isinstance(command, PlayCards):
        return _play_cards(state, command)
    if isinstance(command, Pass):
        return _pass(state, command)
    return _reject(state, RejectCode.INVALID_PHASE, f"unsupported command: {type(command).__name__}")


def _join_table(state: MatchState, command: JoinTable) -> ReducerResult:
    seat = command.requested_seat
    if seat in state.seats:
        return _reject(state, RejectCode.SEAT_OCCUPIED, f"seat {seat.value} is already occupied")
    seats = dict(state.seats)
    controllers = dict(state.controllers)
    seats[seat] = command.player
    controllers[seat] = command.controller
    phase = MatchPhase.READY_CHECK if len(seats) == len(SEATS) else state.phase
    next_state = state.bump_seq()
    next_state = replace(next_state, seats=seats, controllers=controllers, phase=phase)
    return ReducerResult(
        state=next_state,
        events=(
            _event(
                next_state,
                "PlayerSeated",
                {
                    "seat": seat.value,
                    "player": {
                        "id": command.player.id,
                        "display_name": command.player.display_name,
                        "kind": command.player.kind.value,
                    },
                    "controller": _controller_payload(command.controller),
                },
            ),
        ),
    )


def _attach_controller(state: MatchState, command: AttachController) -> ReducerResult:
    seat = command.controller.seat
    if seat not in state.seats:
        return _reject(state, RejectCode.CONTROLLER_NOT_ATTACHED, f"seat {seat.value} has no player")
    controllers = dict(state.controllers)
    controllers[seat] = command.controller
    next_state = replace(state.bump_seq(), controllers=controllers)
    return ReducerResult(
        state=next_state,
        events=(
            _event(
                next_state,
                "ControllerAttached",
                {"seat": seat.value, "controller": _controller_payload(command.controller)},
            ),
        ),
    )


def _detach_controller(state: MatchState, command: DetachController) -> ReducerResult:
    controller = state.controllers.get(command.seat)
    if controller is None or controller.id != command.controller_id:
        return _reject(state, RejectCode.CONTROLLER_NOT_ATTACHED, "controller is not attached to that seat")
    controllers = dict(state.controllers)
    del controllers[command.seat]
    ready_seats = frozenset(seat for seat in state.ready_seats if seat != command.seat)
    next_state = replace(state.bump_seq(), controllers=controllers, ready_seats=ready_seats)
    return ReducerResult(
        state=next_state,
        events=(_event(next_state, "ControllerDetached", {"seat": command.seat.value, "controller_id": command.controller_id}),),
    )


def _ready(state: MatchState, command: Ready) -> ReducerResult:
    rejection = _require_controller(state, command.controller_id, command.seat, ControllerCapability.PLAY)
    if rejection:
        return _reject(state, rejection.code, rejection.message)
    ready = frozenset({*state.ready_seats, command.seat})
    next_state = replace(state.bump_seq(), ready_seats=ready)
    return ReducerResult(state=next_state, events=(_event(next_state, "PlayerReady", {"seat": command.seat.value}),))


def _start_match(state: MatchState, command: StartMatch) -> ReducerResult:
    if state.phase == MatchPhase.DEAL_COMPLETE and state.last_deal_result is not None:
        return _start_next_deal(state, command)
    if not state.is_full:
        return _reject(state, RejectCode.TABLE_NOT_FULL, "all four seats must be occupied and controlled")
    if set(state.ready_seats) != set(SEATS):
        return _reject(state, RejectCode.NOT_ALL_READY, "all seats must be ready")
    hands = deal_cards(command.seed, player_count=4)
    hands_by_seat = {seat: hands[index] for index, seat in enumerate(SEATS)}
    leader = Seat.EAST
    deal = DealState(
        hands=hands_by_seat,
        active_seats=frozenset(SEATS),
        finish_order=(),
        leader=leader,
        turn=leader,
        current_trick=TrickState(lead_seat=leader),
    )
    next_state = replace(state.bump_seq(3), phase=MatchPhase.PLAYING, deal=deal)
    return ReducerResult(
        state=next_state,
        events=(
            Event(seq=next_state.event_seq - 2, type="MatchStarted", payload={"table_id": state.table_id}),
            Event(seq=next_state.event_seq - 1, type="DealStarted", payload={"leader": leader.value}),
            Event(seq=next_state.event_seq, type="CardsDealt", payload=_hands_payload(hands_by_seat)),
        ),
    )


def _start_next_deal(state: MatchState, command: StartMatch) -> ReducerResult:
    assert state.last_deal_result is not None
    hands = deal_cards(command.seed, player_count=4)
    hands_by_seat = {seat: hands[index] for index, seat in enumerate(SEATS)}
    tribute = _build_tribute_state(state.last_deal_result, hands_by_seat, state.current_level)
    leader = tribute.leader_after
    deal = DealState(
        hands=hands_by_seat,
        active_seats=frozenset(SEATS),
        finish_order=(),
        leader=leader,
        turn=leader,
        current_trick=TrickState(lead_seat=leader),
        tribute=None if tribute.resisted else tribute,
    )
    if tribute.resisted:
        next_state = replace(state.bump_seq(3), phase=MatchPhase.PLAYING, deal=deal)
        return ReducerResult(
            state=next_state,
            events=(
                Event(seq=next_state.event_seq - 2, type="DealStarted", payload={"leader": leader.value}),
                Event(seq=next_state.event_seq - 1, type="CardsDealt", payload=_hands_payload(hands_by_seat)),
                Event(seq=next_state.event_seq, type="TributeResisted", payload={"leader": leader.value}),
            ),
        )
    next_state = replace(state.bump_seq(3), phase=MatchPhase.TRIBUTE, deal=deal)
    return ReducerResult(
        state=next_state,
        events=(
            Event(seq=next_state.event_seq - 2, type="DealStarted", payload={"leader": leader.value}),
            Event(seq=next_state.event_seq - 1, type="CardsDealt", payload=_hands_payload(hands_by_seat)),
            Event(
                seq=next_state.event_seq,
                type="TributeRequired",
                payload={
                    "obligations": [
                        {"giver": item.giver.value, "receiver": item.receiver.value} for item in tribute.obligations
                    ],
                    "leader_after": tribute.leader_after.value,
                },
            ),
        ),
    )


def _submit_tribute(state: MatchState, command: SubmitTribute) -> ReducerResult:
    if state.phase != MatchPhase.TRIBUTE or state.deal is None or state.deal.tribute is None:
        return _reject(state, RejectCode.INVALID_PHASE, "match is not waiting for tribute")
    rejection = _require_controller(state, command.controller_id, command.seat, ControllerCapability.PLAY)
    if rejection:
        return _reject(state, rejection.code, rejection.message)
    tribute = state.deal.tribute
    index = _find_obligation_index(tribute, giver=command.seat, needs_tribute=True)
    if index is None:
        return _reject(state, RejectCode.INVALID_TRIBUTE_CARD, "seat has no pending tribute")
    if command.card_id not in state.deal.hand_for(command.seat):
        return _reject(state, RejectCode.CARD_NOT_OWNED, "tribute card is not in giver hand")
    if command.card_id not in _highest_eligible_tribute_cards(state.deal.hand_for(command.seat), state.current_level):
        return _reject(state, RejectCode.INVALID_TRIBUTE_CARD, "tribute must be the highest eligible card")

    obligation = tribute.obligations[index]
    hands = _move_card(state.deal.hands, command.seat, obligation.receiver, command.card_id)
    obligations = list(tribute.obligations)
    obligations[index] = replace(obligation, tribute_card_id=command.card_id)
    next_tribute = replace(tribute, obligations=tuple(obligations))
    deal = replace(state.deal, hands=hands, tribute=next_tribute)
    next_state = replace(state.bump_seq(), deal=deal)
    return ReducerResult(
        state=next_state,
        events=(
            _event(
                next_state,
                "TributePaid",
                {"giver": command.seat.value, "receiver": obligation.receiver.value, "card_id": command.card_id},
            ),
        ),
    )


def _return_tribute(state: MatchState, command: ReturnTribute) -> ReducerResult:
    if state.phase != MatchPhase.TRIBUTE or state.deal is None or state.deal.tribute is None:
        return _reject(state, RejectCode.INVALID_PHASE, "match is not waiting for tribute return")
    rejection = _require_controller(state, command.controller_id, command.seat, ControllerCapability.PLAY)
    if rejection:
        return _reject(state, rejection.code, rejection.message)
    tribute = state.deal.tribute
    index = _find_obligation_index(tribute, receiver=command.seat, needs_return=True)
    if index is None:
        return _reject(state, RejectCode.INVALID_RETURN_CARD, "seat has no pending return")
    obligation = tribute.obligations[index]
    if command.card_id not in state.deal.hand_for(command.seat):
        return _reject(state, RejectCode.CARD_NOT_OWNED, "return card is not in receiver hand")
    if team_for_seat(obligation.giver) == team_for_seat(obligation.receiver) and not _rank_at_most_ten(command.card_id):
        return _reject(state, RejectCode.INVALID_RETURN_CARD, "return card to partner must be 10 or lower")

    hands = _move_card(state.deal.hands, command.seat, obligation.giver, command.card_id)
    obligations = list(tribute.obligations)
    obligations[index] = replace(obligation, return_card_id=command.card_id)
    next_tribute = replace(tribute, obligations=tuple(obligations))
    complete = next_tribute.complete
    deal = replace(
        state.deal,
        hands=hands,
        tribute=None if complete else next_tribute,
        leader=tribute.leader_after if complete else state.deal.leader,
        turn=tribute.leader_after if complete else state.deal.turn,
        current_trick=TrickState(lead_seat=tribute.leader_after) if complete else state.deal.current_trick,
    )
    event_specs: list[tuple[str, dict[str, object]]] = [
        (
            "TributeReturned",
            {"giver": obligation.giver.value, "receiver": command.seat.value, "card_id": command.card_id},
        )
    ]
    if complete:
        event_specs.append(("TributeComplete", {"leader": tribute.leader_after.value}))
    next_state = replace(state.bump_seq(len(event_specs)), phase=MatchPhase.PLAYING if complete else state.phase, deal=deal)
    events = _events_from_specs(state.event_seq, event_specs)
    return ReducerResult(state=next_state, events=tuple(events))


def _play_cards(state: MatchState, command: PlayCards) -> ReducerResult:
    if state.phase != MatchPhase.PLAYING or state.deal is None:
        return _reject(state, RejectCode.INVALID_PHASE, "match is not playing")
    rejection = _require_controller(state, command.controller_id, command.seat, ControllerCapability.PLAY)
    if rejection:
        return _reject(state, rejection.code, rejection.message)
    if state.deal.turn != command.seat:
        return _reject(state, RejectCode.NOT_YOUR_TURN, "it is not that seat's turn")
    hand = list(state.deal.hand_for(command.seat))
    if len(set(command.card_ids)) != len(command.card_ids) or not set(command.card_ids).issubset(set(hand)):
        return _reject(state, RejectCode.CARD_NOT_OWNED, "one or more cards are not in the seat's hand")
    try:
        played_hand = parse_hand(resolve_cards(command.card_ids), command.declared_type, level=state.current_level)
    except AmbiguousHandError as exc:
        return _reject(state, RejectCode.AMBIGUOUS_WILD_CARD_DECLARATION, str(exc))
    except ValueError as exc:
        return _reject(state, RejectCode.INVALID_HAND_TYPE, str(exc))
    if not can_beat(played_hand, state.deal.current_trick.last_play, state.current_level):
        return _reject(state, RejectCode.DOES_NOT_BEAT_CURRENT_HAND, "played hand does not beat current trick hand")
    for card_id in command.card_ids:
        hand.remove(card_id)
    hands = dict(state.deal.hands)
    hands[command.seat] = tuple(hand)
    active = set(state.deal.active_seats)
    finish_order = list(state.deal.finish_order)
    finished = not hand
    if finished:
        active.remove(command.seat)
        finish_order.append(command.seat)
    completion = _completed_finish_order(tuple(finish_order), frozenset(active))
    next_turn = next_seat(command.seat, active) if active else command.seat
    reported_10 = len(hand) <= 10 and command.seat not in state.deal.report_10_done
    report_10_done = (
        frozenset({*state.deal.report_10_done, command.seat}) if reported_10 else state.deal.report_10_done
    )
    trick = replace(
        state.deal.current_trick,
        last_play=played_hand,
        last_play_seat=command.seat,
        pass_count=0,
    )
    deal = replace(
        state.deal,
        hands=hands,
        active_seats=frozenset(active),
        finish_order=completion if completion is not None else tuple(finish_order),
        turn=next_turn,
        current_trick=trick,
        report_10_done=report_10_done,
    )
    event_specs: list[tuple[str, dict[str, object]]] = [
        (
            "CardsPlayed",
            {
                "seat": command.seat.value,
                "card_ids": list(command.card_ids),
                "hand_type": played_hand.type.value,
                "remaining_count": len(hand),
            },
        )
    ]
    if reported_10:
        event_specs.append(("TenCardReport", {"seat": command.seat.value, "remaining_count": len(hand)}))
    if finished:
        event_specs.append(("PlayerFinished", {"seat": command.seat.value, "position": len(finish_order)}))

    next_state = replace(state.bump_seq(len(event_specs)), deal=deal)
    events = list(_events_from_specs(state.event_seq, event_specs))
    if completion is not None:
        completed = _complete_deal(next_state, completion)
        next_state = completed.state
        events.extend(completed.events)
    return ReducerResult(state=next_state, events=tuple(events))


def _completed_finish_order(finish_order: tuple[Seat, ...], active: frozenset[Seat]) -> tuple[Seat, ...] | None:
    if len(finish_order) >= 2 and team_for_seat(finish_order[0]) == team_for_seat(finish_order[1]):
        remaining = tuple(seat for seat in SEATS if seat not in finish_order)
        return finish_order + remaining
    if len(active) <= 1 and finish_order:
        return finish_order + tuple(active)
    return None


def _complete_deal(state: MatchState, finish_order: tuple[Seat, ...]) -> ReducerResult:
    winner = finish_order[0]
    winning_team = team_for_seat(winner)
    partner = partner_for_seat(winner)
    partner_position = finish_order.index(partner) + 1
    advance_count = {2: 3, 3: 2}.get(partner_position, 1)
    previous_level = state.scores.level_by_team[winning_team]
    next_level, match_complete = _advance_level(previous_level, advance_count, partner_position)
    level_by_team = dict(state.scores.level_by_team)
    level_by_team[winning_team] = next_level
    scores = replace(state.scores, level_by_team=level_by_team)
    result = DealResult(
        finish_order=finish_order,
        winning_team=winning_team,
        advance_count=advance_count,
        previous_level=previous_level,
        next_level=next_level,
        match_complete=match_complete,
    )
    next_phase = MatchPhase.MATCH_COMPLETE if match_complete else MatchPhase.DEAL_COMPLETE
    next_seq_state = state.bump_seq(2 + int(match_complete))
    next_state = replace(
        next_seq_state,
        phase=next_phase,
        deal=replace(state.deal, finish_order=finish_order) if state.deal else None,
        current_level=next_level,
        scores=scores,
        last_deal_result=result,
    )
    events = [
        Event(
            seq=state.event_seq + 1,
            type="DealEnded",
            payload={
                "finish_order": [seat.value for seat in finish_order],
                "winning_team": winning_team.value,
                "advance_count": advance_count,
            },
        ),
        Event(
            seq=state.event_seq + 2,
            type="LevelAdvanced",
            payload={
                "team": winning_team.value,
                "previous_level": previous_level.value,
                "next_level": next_level.value,
            },
        ),
    ]
    if match_complete:
        events.append(
            Event(
                seq=state.event_seq + 3,
                type="MatchEnded",
                payload={"winning_team": winning_team.value},
            )
        )
    return ReducerResult(state=next_state, events=tuple(events))


def _advance_level(level: Rank, advance_count: int, partner_position: int) -> tuple[Rank, bool]:
    index = LEVEL_RANKS.index(level)
    ace_index = LEVEL_RANKS.index(Rank.ACE)
    if level == Rank.ACE and partner_position != 4:
        return Rank.ACE, True
    next_index = min(index + advance_count, ace_index)
    return LEVEL_RANKS[next_index], False


def _build_tribute_state(result: DealResult, hands: dict[Seat, tuple[str, ...]], level: Rank) -> TributeState:
    finish = result.finish_order
    winner = finish[0]
    if team_for_seat(finish[0]) == team_for_seat(finish[1]):
        losing = (finish[2], finish[3])
        resisted = _double_tribute_resisted(losing, hands)
        obligations = (
            TributeObligation(giver=finish[3], receiver=finish[0]),
            TributeObligation(giver=finish[2], receiver=finish[1]),
        )
        return TributeState(obligations=obligations, leader_after=winner, resisted=resisted)
    giver = finish[-1]
    resisted = _big_joker_count(hands[giver]) >= 2
    return TributeState(
        obligations=(TributeObligation(giver=giver, receiver=winner),),
        leader_after=winner if resisted else giver,
        resisted=resisted,
    )


def _double_tribute_resisted(losing: tuple[Seat, Seat], hands: dict[Seat, tuple[str, ...]]) -> bool:
    counts = [_big_joker_count(hands[seat]) for seat in losing]
    return any(count >= 2 for count in counts) or all(count >= 1 for count in counts)


def _big_joker_count(card_ids: tuple[str, ...]) -> int:
    return sum(1 for card_id in card_ids if CARD_BY_ID[card_id].rank == Rank.BIG_JOKER)


def _highest_eligible_tribute_cards(card_ids: tuple[str, ...], level: Rank) -> frozenset[str]:
    ctx = RankContext(level)
    eligible = [CARD_BY_ID[card_id] for card_id in card_ids if not is_red_heart_level_card(CARD_BY_ID[card_id], level)]
    if not eligible:
        return frozenset()
    max_value = max(ctx.rank_value(card.rank) for card in eligible)
    return frozenset(card.id for card in eligible if ctx.rank_value(card.rank) == max_value)


def _rank_at_most_ten(card_id: str) -> bool:
    rank = CARD_BY_ID[card_id].rank
    low = {
        Rank.TWO,
        Rank.THREE,
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.EIGHT,
        Rank.NINE,
        Rank.TEN,
    }
    return rank in low


def _find_obligation_index(
    tribute: TributeState,
    *,
    giver: Seat | None = None,
    receiver: Seat | None = None,
    needs_tribute: bool = False,
    needs_return: bool = False,
) -> int | None:
    for index, obligation in enumerate(tribute.obligations):
        if giver is not None and obligation.giver != giver:
            continue
        if receiver is not None and obligation.receiver != receiver:
            continue
        if needs_tribute and obligation.tribute_card_id is not None:
            continue
        if needs_return and (obligation.tribute_card_id is None or obligation.return_card_id is not None):
            continue
        return index
    return None


def _move_card(hands: dict[Seat, tuple[str, ...]], source: Seat, target: Seat, card_id: str) -> dict[Seat, tuple[str, ...]]:
    next_hands = {seat: list(hand) for seat, hand in hands.items()}
    next_hands[source].remove(card_id)
    next_hands[target].append(card_id)
    return {seat: tuple(hand) for seat, hand in next_hands.items()}


def _pass(state: MatchState, command: Pass) -> ReducerResult:
    if state.phase != MatchPhase.PLAYING or state.deal is None:
        return _reject(state, RejectCode.INVALID_PHASE, "match is not playing")
    rejection = _require_controller(state, command.controller_id, command.seat, ControllerCapability.PLAY)
    if rejection:
        return _reject(state, rejection.code, rejection.message)
    if state.deal.turn != command.seat:
        return _reject(state, RejectCode.NOT_YOUR_TURN, "it is not that seat's turn")
    if state.deal.current_trick.last_play is None or state.deal.current_trick.last_play_seat is None:
        return _reject(state, RejectCode.CANNOT_PASS_WHEN_LEADING, "leading seat cannot pass")
    active = state.deal.active_seats
    pass_count = state.deal.current_trick.pass_count + 1
    event_specs: list[tuple[str, dict[str, object]]] = [("PlayerPassed", {"seat": command.seat.value})]
    required_passes = len(active) - 1 if state.deal.current_trick.last_play_seat in active else len(active)
    if pass_count >= max(required_passes, 1):
        leader = state.deal.current_trick.last_play_seat
        assert leader is not None
        next_leader = _leader_after_trick(leader, active)
        trick = TrickState(lead_seat=next_leader)
        deal = replace(state.deal, leader=next_leader, turn=next_leader, current_trick=trick)
        event_specs.append(
            (
                "TrickEnded",
                {"last_play_seat": leader.value, "next_leader": next_leader.value},
            )
        )
    else:
        trick = replace(state.deal.current_trick, pass_count=pass_count)
        deal = replace(state.deal, turn=next_seat(command.seat, active), current_trick=trick)
    next_state = replace(state.bump_seq(len(event_specs)), deal=deal)
    events = _events_from_specs(state.event_seq, event_specs)
    return ReducerResult(state=next_state, events=events)


def _require_controller(
    state: MatchState,
    controller_id: str,
    seat: Seat,
    capability: ControllerCapability,
) -> CommandRejected | None:
    controller = state.controllers.get(seat)
    if controller is None or controller.id != controller_id:
        return CommandRejected(RejectCode.CONTROLLER_NOT_ATTACHED, "controller is not attached to that seat")
    if not controller.can(capability):
        return CommandRejected(RejectCode.INSUFFICIENT_CONTROLLER_CAPABILITY, f"controller lacks {capability.value}")
    return None


def _leader_after_trick(last_play_seat: Seat, active: frozenset[Seat]) -> Seat:
    if last_play_seat in active:
        return last_play_seat
    partner = partner_for_seat(last_play_seat)
    if partner in active:
        return partner
    return next_seat(last_play_seat, active)


def _reject(state: MatchState, code: RejectCode, message: str) -> ReducerResult:
    return ReducerResult(state=state, rejection=CommandRejected(code, message))


def _event(state: MatchState, event_type: str, payload: dict[str, object]) -> Event:
    return Event(seq=state.event_seq, type=event_type, payload=payload)


def _events_from_specs(start_seq: int, specs: list[tuple[str, dict[str, object]]]) -> tuple[Event, ...]:
    return tuple(
        Event(seq=start_seq + index, type=event_type, payload=payload)
        for index, (event_type, payload) in enumerate(specs, start=1)
    )


def _hands_payload(hands_by_seat: dict[Seat, tuple[str, ...]]) -> dict[str, object]:
    return {"hands": {seat.value: list(hand) for seat, hand in hands_by_seat.items()}}


def _controller_payload(controller: object) -> dict[str, object]:
    return {
        "id": controller.id,
        "kind": controller.kind.value,
        "seat": controller.seat.value,
        "player_id": controller.player_id,
        "capabilities": sorted(capability.value for capability in controller.capabilities),
    }
