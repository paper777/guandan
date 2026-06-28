from __future__ import annotations

import time

from client.types import ActionRequest, JsonObject
from db.player.types import Player
from npc.dummy_bot.player import DummyBotPlayer
from npc.rl_agent.model_loader import RlAgentConfig, RlModelLoader
from server.domain.cards import Rank
from server.domain.legal_actions import ActionCandidate, legal_actions_for_snapshot
from server.domain.seats import SEATS, Seat, Team
from server.domain.state import MatchPhase
from server.services.table_config import DEFAULT_ACTION_TIMEOUT_SECONDS
from server.services.snapshots import PublicPlayer, PublicTableSnapshot, SeatSnapshot
from training.heuristic import HeuristicPolicy


class RlAgentPlayer(Player):
    """Broker-compatible local RL player with deterministic fallback."""

    def __init__(
        self,
        config: RlAgentConfig | None = None,
        *,
        model_loader: RlModelLoader | None = None,
        fallback_policy: HeuristicPolicy | None = None,
    ) -> None:
        self.config = config or RlAgentConfig()
        self.model_loader = model_loader or RlModelLoader(self.config)
        self.fallback_policy = fallback_policy or HeuristicPolicy()
        self._protocol_fallback = DummyBotPlayer()

    def choose_action(self, request: ActionRequest) -> JsonObject:
        snapshot = seat_snapshot_from_request(request)
        actions = legal_actions_for_snapshot(snapshot)
        if not actions:
            return self._protocol_fallback.choose_action(request)
        if not _deadline_is_too_close(snapshot, self.config):
            try:
                selected = self.model_loader.choose_action(snapshot, actions)
            except Exception:
                selected = None
            if selected is not None:
                return selected.to_payload()
        return self._fallback_action(request, snapshot, actions)

    def _fallback_action(
        self,
        request: ActionRequest,
        snapshot: SeatSnapshot,
        actions: tuple[ActionCandidate, ...],
    ) -> JsonObject:
        try:
            return self.fallback_policy.choose_action(snapshot, actions).to_payload()
        except Exception:
            return self._protocol_fallback.choose_action(request)


def seat_snapshot_from_request(request: ActionRequest) -> SeatSnapshot:
    snapshot_payload = _dict(request.snapshot)
    prompt = _dict(request.prompt)
    public_payload = _dict(snapshot_payload.get("public"))
    seat = _seat(snapshot_payload.get("seat") or prompt.get("seat"))
    legal_action = _optional_str(prompt.get("kind") or snapshot_payload.get("legal_action"))
    public = _public_snapshot(public_payload, snapshot_payload, prompt, seat, legal_action)
    return SeatSnapshot(
        public=public,
        seat=seat,
        hand=tuple(str(card_id) for card_id in _list(snapshot_payload.get("hand"))),
        legal_action=legal_action,
        eligible_card_ids=tuple(
            str(card_id)
            for card_id in _list(prompt.get("eligible_card_ids") or snapshot_payload.get("eligible_card_ids"))
        ),
        tribute_from=_optional_seat(prompt.get("tribute_from") or snapshot_payload.get("tribute_from")),
        tribute_to=_optional_seat(prompt.get("tribute_to") or snapshot_payload.get("tribute_to")),
        return_rank_at_most_ten=bool(
            prompt.get("return_rank_at_most_ten", snapshot_payload.get("return_rank_at_most_ten", False))
        ),
    )


def _public_snapshot(
    public_payload: JsonObject,
    snapshot_payload: JsonObject,
    prompt: JsonObject,
    seat: Seat,
    legal_action: str | None,
) -> PublicTableSnapshot:
    current_level = _rank(prompt.get("current_level") or public_payload.get("current_level"))
    return PublicTableSnapshot(
        table_id=str(public_payload.get("table_id") or snapshot_payload.get("table_id") or ""),
        deal_id=int(public_payload.get("deal_id") or snapshot_payload.get("deal_id") or 0),
        phase=_phase(public_payload.get("phase"), legal_action),
        seats=_public_players(public_payload.get("seats")),
        hand_counts=_hand_counts(public_payload.get("hand_counts"), seat, snapshot_payload.get("hand")),
        current_turn=_optional_seat(public_payload.get("current_turn")),
        finish_order=tuple(
            seat
            for seat in (_optional_seat(item) for item in _list(public_payload.get("finish_order")))
            if seat is not None
        ),
        event_seq=int(public_payload.get("event_seq") or 0),
        current_level=current_level,
        level_by_team=_level_by_team(public_payload.get("level_by_team"), current_level),
        action_deadline_epoch_ms=_optional_int(public_payload.get("action_deadline_epoch_ms")),
        action_timeout_seconds=int(public_payload.get("action_timeout_seconds") or DEFAULT_ACTION_TIMEOUT_SECONDS),
        acting_seat=_optional_seat(public_payload.get("acting_seat")),
        current_trick=_dict(prompt.get("current_trick") or public_payload.get("current_trick")) or None,
        played_card_counts=_played_card_counts(public_payload.get("played_card_counts")),
    )


def _deadline_is_too_close(snapshot: SeatSnapshot, config: RlAgentConfig) -> bool:
    deadline = snapshot.public.action_deadline_epoch_ms
    if deadline is None or config.min_model_deadline_ms <= 0:
        return False
    remaining_ms = deadline - int(time.time() * 1000)
    return remaining_ms <= config.min_model_deadline_ms


def _dict(value: object) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _seat(value: object) -> Seat:
    seat = _optional_seat(value)
    if seat is None:
        raise ValueError("RL action request is missing seat")
    return seat


def _optional_seat(value: object) -> Seat | None:
    if value is None:
        return None
    try:
        return Seat(str(value))
    except ValueError:
        return None


def _rank(value: object) -> Rank:
    if value is not None:
        try:
            return Rank(str(value))
        except ValueError:
            pass
    return Rank.TWO


def _phase(value: object, legal_action: str | None) -> MatchPhase:
    if value is not None:
        try:
            return MatchPhase(str(value))
        except ValueError:
            pass
    if legal_action in {"tribute", "return_tribute"}:
        return MatchPhase.TRIBUTE
    if legal_action in {"lead", "play_or_pass"}:
        return MatchPhase.PLAYING
    return MatchPhase.WAITING_FOR_PLAYERS


def _public_players(value: object) -> dict[Seat, PublicPlayer]:
    if not isinstance(value, dict):
        return {}
    players: dict[Seat, PublicPlayer] = {}
    for raw_seat, raw_player in value.items():
        seat = _optional_seat(raw_seat)
        if seat is None:
            continue
        player = _dict(raw_player)
        players[seat] = PublicPlayer(
            player_id=str(player.get("player_id") or player.get("id") or ""),
            display_name=str(player.get("display_name") or player.get("name") or seat.value),
            kind=str(player.get("kind") or ""),
            controlled=bool(player.get("controlled", True)),
        )
    return players


def _hand_counts(value: object, own_seat: Seat, own_hand: object) -> dict[Seat, int]:
    counts = {seat: 0 for seat in SEATS}
    if isinstance(value, dict):
        for raw_seat, raw_count in value.items():
            seat = _optional_seat(raw_seat)
            if seat is not None:
                counts[seat] = int(raw_count or 0)
    counts[own_seat] = counts.get(own_seat) or len(_list(own_hand))
    return counts


def _played_card_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for raw_face, raw_count in value.items():
        face = str(raw_face)
        if face:
            counts[face] = int(raw_count or 0)
    return counts


def _level_by_team(value: object, current_level: Rank) -> dict[Team, Rank]:
    levels = {Team.EAST_WEST: current_level, Team.SOUTH_NORTH: current_level}
    if not isinstance(value, dict):
        return levels
    for team in Team:
        raw = value.get(team) if team in value else value.get(team.value)
        if raw is not None:
            levels[team] = _rank(raw)
    return levels


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
