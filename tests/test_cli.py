from __future__ import annotations

import getpass
import io
import json
import tempfile
import unittest
import asyncio
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit
from unittest.mock import patch

from client.http_client import GuandanClientError, GuandanHttpClient
from client.cli import (
    Session,
    drive_bot_turns,
    format_card_id,
    format_command_response,
    format_friend_mark,
    format_hand,
    format_timeout_fallback,
    format_npc_metadata,
    format_public_snapshot,
    format_seat_snapshot,
    resolve_card_inputs,
    run_cli,
)
from client.session import prepare_default_table
from server.app.main import TABLES, app


class FakeClient:
    def __init__(self) -> None:
        self.table_id = "table-1"
        self.seats: dict[str, dict[str, str]] = {}
        self.phase = "WAITING_FOR_PLAYERS"
        self.current_turn: str | None = None
        self.current_trick = None
        self.action_deadline_epoch_ms = None
        self.event_seq = 0
        self.calls: list[tuple] = []
        self.hands = {
            "E": ["D1-H-4", "D1-C-3", "D2-S-3"],
            "S": ["D1-H-4"],
            "W": ["D1-C-4"],
            "N": ["D1-D-4"],
        }

    def create_table(self):
        self.calls.append(("create_table",))
        return {"table_id": self.table_id}

    def table_snapshot(self, table_id):
        self.calls.append(("table_snapshot", table_id))
        return self._snapshot()

    def join_human(self, table_id, seat, *, player_id=None, controller_id=None, display_name=None):
        self.calls.append(("join_human", table_id, seat, player_id, controller_id, display_name))
        self.seats[seat] = {"display_name": display_name or player_id, "kind": "human"}
        return {"controller_id": controller_id, "snapshot": self._snapshot()}

    def join_local_bot(self, table_id, seat, *, player_id=None, controller_id=None, display_name=None):
        self.calls.append(("join_local_bot", table_id, seat, player_id, controller_id, display_name))
        self.seats[seat] = {"display_name": display_name or player_id, "kind": "bot"}
        return {"controller_id": controller_id, "snapshot": self._snapshot()}

    def join_agent(self, table_id, seat, display_name):
        self.calls.append(("join_agent", table_id, seat, display_name))
        self.seats[seat] = {"display_name": display_name, "kind": "agent"}
        return {"player_id": f"agent-{seat}", "controller_id": f"agent-controller-{seat}"}

    def ready(self, table_id, seat, controller_id):
        self.calls.append(("ready", table_id, seat, controller_id))
        return {"events": [], "snapshot": self._snapshot()}

    def start(self, table_id):
        self.calls.append(("start", table_id))
        self.phase = "PLAYING"
        self.current_turn = "E"
        return {"events": [{"seq": 1, "type": "MatchStarted", "payload": {"table_id": table_id}}], "snapshot": self._snapshot()}

    def seat_snapshot(self, table_id, seat, controller_id):
        self.calls.append(("seat_snapshot", table_id, seat, controller_id))
        legal_action = None
        if seat == self.current_turn:
            legal_action = "lead" if seat == "E" else "play_or_pass"
        return {
            "public": self._snapshot(),
            "seat": seat,
            "hand": list(self.hands[seat]),
            "legal_action": legal_action,
        }

    def play_cards(self, table_id, seat, controller_id, card_ids, *, declared_type=None):
        self.calls.append(("play_cards", table_id, seat, controller_id, card_ids, declared_type))
        for card_id in card_ids:
            self.hands[seat].remove(card_id)
        self.current_trick = {"last_play_seat": seat, "hand_type": "single", "card_ids": list(card_ids)}
        self.current_turn = {"E": "S", "S": "W", "W": "N", "N": "E"}[seat]
        self.event_seq += 1
        return {
            "events": [{"seq": self.event_seq, "type": "CardsPlayed", "payload": {"seat": seat, "hand_type": "single", "card_ids": list(card_ids)}}],
            "snapshot": self._snapshot(),
        }

    def pass_turn(self, table_id, seat, controller_id):
        self.calls.append(("pass_turn", table_id, seat, controller_id))
        self.current_turn = {"E": "S", "S": "W", "W": "N", "N": "E"}[seat]
        self.event_seq += 1
        return {
            "events": [{"seq": self.event_seq, "type": "PlayerPassed", "payload": {"seat": seat}}],
            "snapshot": self._snapshot(),
        }

    def submit_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("submit_tribute", table_id, seat, controller_id, card_id))
        self.event_seq += 1
        return {
            "events": [
                {
                    "seq": self.event_seq,
                    "type": "TributePaid",
                    "payload": {"giver": seat, "receiver": "S", "card_id": card_id},
                }
            ],
            "snapshot": self._snapshot(),
        }

    def return_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("return_tribute", table_id, seat, controller_id, card_id))
        self.event_seq += 1
        return {
            "events": [
                {
                    "seq": self.event_seq,
                    "type": "TributeReturned",
                    "payload": {"giver": "S", "receiver": seat, "card_id": card_id},
                }
            ],
            "snapshot": self._snapshot(),
        }

    def _snapshot(self):
        return {
            "table_id": self.table_id,
            "phase": self.phase,
            "seats": self.seats,
            "hand_counts": {seat: len(hand) for seat, hand in self.hands.items()},
            "current_turn": self.current_turn,
            "acting_seat": self.current_turn,
            "current_level": "2",
            "level_by_team": {"EW": "2", "SN": "3"},
            "current_trick": self.current_trick,
            "action_deadline_epoch_ms": self.action_deadline_epoch_ms,
            "finish_order": [],
            "event_seq": self.event_seq,
        }


def _write_player_storage(root: Path, profiles: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    directories: list[str] = []
    profile_keys = {"id", "profile_id", "display_name", "name", "kind", "personality"}
    llm_keys = {
        "provider_name",
        "model_name",
        "api_base_url",
        "timeout_seconds",
        "temperature",
        "max_output_tokens",
        "memory_compaction_char_limit",
        "memory_recent_deal_scan_limit",
        "memory_max_output_tokens",
        "codex_binary",
        "codex_working_dir",
    }
    stat_keys = {
        "deal_count",
        "deal_wins",
        "deal_win_rate",
        "score",
        "match_count",
        "match_wins",
        "match_win_rate",
    }
    for profile in profiles:
        display_name = str(profile.get("display_name") or profile.get("name") or "player")
        directory = str(profile.get("directory") or display_name.replace(" ", "-"))
        directories.append(directory)
        player_dir = root / directory
        player_dir.mkdir(parents=True, exist_ok=True)
        profile_payload = {key: value for key, value in profile.items() if key in profile_keys}
        llm_payload = {key: value for key, value in profile.items() if key in llm_keys}
        stat_payload = {key: value for key, value in profile.items() if key in stat_keys}
        (player_dir / "profile.json").write_text(json.dumps(profile_payload), encoding="utf-8")
        (player_dir / "llm_config.json").write_text(json.dumps(llm_payload), encoding="utf-8")
        (player_dir / "statistics.json").write_text(json.dumps(stat_payload), encoding="utf-8")
        (player_dir / "actions.json").write_text("[]", encoding="utf-8")
        (player_dir / "memory.json").write_text("{}", encoding="utf-8")
    (root / "players.json").write_text(json.dumps({"players": directories}), encoding="utf-8")
    return root


def _play_args(**overrides):
    values = {
        "table_id": None,
        "player_id": None,
        "controller_id": None,
        "display_name": None,
        "player_mode": "human",
        "npc_lineup": "mixed",
        "npc_player_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _asgi_transport(method, path, body, query):
    status, payload = asyncio.run(_call_app(method, path, body, query))
    if status >= 400:
        raise GuandanClientError(status, _error_message(payload), payload)
    return payload


async def _call_app(method, path, body, query):
    messages = []
    request_body = json.dumps(body or {}).encode()
    parsed = urlsplit(path)
    query_string = urlencode(query or {}).encode()

    async def receive():
        return {"type": "http.request", "body": request_body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": query_string,
            "headers": [(b"content-type", b"application/json")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    raw_body = next(message["body"] for message in messages if message["type"] == "http.response.body")
    return status, json.loads(raw_body.decode()) if raw_body else {}


def _error_message(payload):
    rejection = payload.get("rejection")
    if isinstance(rejection, dict):
        return f"{rejection.get('code', 'rejected')}: {rejection.get('message', '')}".rstrip()
    return str(payload.get("detail") or payload.get("error") or "Guandan server request failed")


class CommandLineClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.choose_seat = patch("client.session._choose_available_seat", return_value="E")
        self.choose_seat.start()
        self.addCleanup(self.choose_seat.stop)

    def test_play_parser_rejects_removed_seat_argument(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_cli(["--seat", "S"], input_fn=lambda prompt: "quit", client=FakeClient())

    def test_default_play_creates_human_and_three_mixed_broker_agents(self) -> None:
        client = FakeClient()

        result = run_cli([], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Table table-1 | You are E", result.output)
        self.assertIn("PLAYING | Seat E | Turn E", result.output)
        login_name = getpass.getuser()
        self.assertIn(f"E {login_name} 3 (You)", result.output)
        self.assertIn("Deal 0 | Level Card 2 | Opp Level Card 3", result.output)
        self.assertIn("Jade 1", result.output)
        self.assertIn("River 1", result.output)
        self.assertIn("Atlas 1", result.output)
        self.assertIn("codex-cli/", result.output)
        self.assertIn("Hand: ♠️ 3  ♣️ 3  ♥️ 4", result.output)
        self.assertIn(("join_human", "table-1", "E", "human-E", "human-controller-E", login_name), client.calls)
        join_agent_calls = [call for call in client.calls if call[0] == "join_agent"]
        self.assertEqual({call[2] for call in join_agent_calls}, {"S", "W", "N"})
        self.assertEqual({call[3] for call in join_agent_calls}, {"Jade", "River", "Atlas"})
        self.assertNotIn(("join_local_bot", "table-1", "S", "bot-S", "bot-controller-S", "Bot S"), client.calls)
        self.assertIn(("start", "table-1"), client.calls)

    def test_llm_player_mode_joins_selected_seat_as_agent_and_watches_private_hand(self) -> None:
        class LlmWatchClient(FakeClient):
            def start(self, table_id):
                response = super().start(table_id)
                if self.calls.count(("start", table_id)) > 1:
                    self.current_turn = None
                    response["snapshot"] = self._snapshot()
                return response

            def play_cards(self, table_id, seat, controller_id, card_ids, *, declared_type=None):
                response = super().play_cards(table_id, seat, controller_id, card_ids, declared_type=declared_type)
                self.phase = "MATCH_COMPLETE"
                self.current_turn = None
                response["snapshot"] = self._snapshot()
                return response

        client = LlmWatchClient()

        def unexpected_input(prompt):
            raise AssertionError("LLM watch mode should not prompt for input")

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(Path(tmp), [{"display_name": "Pilot", "kind": "llm"}])
            result = run_cli(
                [
                    "--player-mode",
                    "llm",
                    "--npc-lineup",
                    "dummy",
                    "--display-name",
                    "Pilot",
                    "--npc-player-config",
                    str(path),
                ],
                input_fn=unexpected_input,
                client=client,
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Table table-1 | Watching E", result.output)
        self.assertIn("Hand: ♠️ 3  ♣️ 3  ♥️ 4", result.output)
        self.assertIn(("join_agent", "table-1", "E", "Pilot"), client.calls)
        self.assertNotIn(("join_human", "table-1", "E", "human-E", "human-controller-E", "Pilot"), client.calls)
        self.assertIn(("seat_snapshot", "table-1", "E", "agent-controller-E"), client.calls)
        self.assertIn(("play_cards", "table-1", "E", "agent-controller-E", ("D1-C-3",), None), client.calls)

    def test_player_mode_sets_human_as_llm_witness(self) -> None:
        class MatchCompleteClient(FakeClient):
            def start(self, table_id):
                self.calls.append(("start", table_id))
                if self.calls.count(("start", table_id)) == 1:
                    self.phase = "MATCH_COMPLETE"
                    events = [{"seq": 1, "type": "MatchEnded", "payload": {"winning_team": "EW"}}]
                else:
                    self.phase = "PLAYING"
                    events = [{"seq": 2, "type": "MatchStarted", "payload": {"table_id": table_id}}]
                self.current_turn = None
                return {"events": events, "snapshot": self._snapshot()}

        client = MatchCompleteClient()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(Path(tmp), [{"display_name": "Pilot", "kind": "llm"}])

            result = run_cli(
                ["--player-mode", "llm", "--display-name", "Pilot", "--npc-player-config", str(path)],
                input_fn=lambda prompt: "unused",
                client=client,
            )

            session, _ = prepare_default_table(
                MatchCompleteClient(),
                _play_args(player_mode="llm", display_name="Pilot", npc_player_config=str(path)),
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Table table-1 | Watching E", result.output)
        self.assertTrue(any(member.is_human for member in session.table.members_for("E").witnesses))

    def test_llm_gossiper_can_advise_human_player(self) -> None:
        client = FakeClient()
        commands = iter(["quit"])

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(
                Path(tmp),
                [
                    {
                        "display_name": "Advisor",
                        "kind": "llm",
                        "provider_name": "deterministic",
                        "model_name": "advisor-model",
                    }
                ],
            )

            result = run_cli(
                ["--gossiper-mode", "llm", "--npc-player-config", str(path)],
                input_fn=lambda prompt: next(commands),
                client=client,
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Advice from Advisor Gossiper:", result.output)
        self.assertIn(("join_human", "table-1", "E", "human-E", "human-controller-E", getpass.getuser()), client.calls)

    def test_llm_player_mode_uses_selected_seat_config_metadata(self) -> None:
        class MatchCompleteClient(FakeClient):
            def start(self, table_id):
                self.calls.append(("start", table_id))
                if self.calls.count(("start", table_id)) == 1:
                    self.phase = "MATCH_COMPLETE"
                    events = [{"seq": 1, "type": "MatchEnded", "payload": {"winning_team": "EW"}}]
                else:
                    self.phase = "PLAYING"
                    events = [{"seq": 2, "type": "MatchStarted", "payload": {"table_id": table_id}}]
                self.current_turn = None
                return {"events": events, "snapshot": self._snapshot()}

        client = MatchCompleteClient()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(
                Path(tmp),
                [
                    {
                        "display_name": "Configured East",
                        "kind": "llm",
                        "provider_name": "deterministic",
                        "model_name": "configured-model",
                    }
                ],
            )

            result = run_cli(
                ["--player-mode", "llm", "--npc-player-config", str(path), "--display-name", "Pilot"],
                input_fn=lambda prompt: "unused",
                client=client,
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("join_agent", "table-1", "E", "Pilot"), client.calls)
        self.assertIn("deterministic/configured-model", result.output)

    def test_llm_player_mode_does_not_duplicate_watched_seatless_profile(self) -> None:
        class MatchCompleteClient(FakeClient):
            def start(self, table_id):
                self.calls.append(("start", table_id))
                if self.calls.count(("start", table_id)) == 1:
                    self.phase = "MATCH_COMPLETE"
                    events = [{"seq": 1, "type": "MatchEnded", "payload": {"winning_team": "EW"}}]
                else:
                    self.phase = "PLAYING"
                    events = [{"seq": 2, "type": "MatchStarted", "payload": {"table_id": table_id}}]
                self.current_turn = None
                return {"events": events, "snapshot": self._snapshot()}

        client = MatchCompleteClient()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(
                Path(tmp),
                [
                    {"display_name": "Pilot", "kind": "llm"},
                    {"display_name": "Jade", "kind": "dummy"},
                    {"display_name": "River", "kind": "dummy"},
                    {"display_name": "Atlas", "kind": "dummy"},
                ],
            )

            with patch("client.session._choose_available_seat", return_value="S"):
                result = run_cli(
                    [
                        "--player-mode",
                        "llm",
                        "--display-name",
                        "Pilot",
                        "--npc-player-config",
                        str(path),
                    ],
                    input_fn=lambda prompt: "unused",
                    client=client,
                )

        self.assertEqual(result.exit_code, 0)
        join_agent_calls = [call for call in client.calls if call[0] == "join_agent"]
        self.assertEqual({call[2] for call in join_agent_calls}, {"E", "S", "W", "N"})
        first_match_joins = join_agent_calls[:4]
        self.assertEqual([call[3] for call in first_match_joins].count("Pilot"), 1)
        self.assertEqual({call[3] for call in first_match_joins}, {"Pilot", "Jade", "River", "Atlas"})

    def test_llm_player_mode_with_single_profile_fills_real_server_table(self) -> None:
        TABLES.clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(Path(tmp), [{"display_name": "Pilot", "kind": "llm"}])
            args = _play_args(player_mode="llm", display_name="Pilot", npc_player_config=str(path))
            client = GuandanHttpClient(transport=_asgi_transport)

            session, snapshot = prepare_default_table(client, args)

        self.assertEqual(snapshot["phase"], "PLAYING")
        self.assertEqual(len(snapshot["seats"]), 4)
        self.assertEqual(session.human_seat, "E")
        self.assertEqual({player["display_name"] for player in snapshot["seats"].values()}, {"Pilot", "Jade", "River", "Atlas"})

    def test_human_play_readable_card_label_then_drives_bot_passes(self) -> None:
        client = FakeClient()
        commands = iter(["play C3", "quit"])

        result = run_cli(["--npc-lineup", "dummy"], input_fn=lambda prompt: next(commands), client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("play_cards", "table-1", "E", "human-controller-E", ("D1-C-3",), None), client.calls)
        self.assertIn(("pass_turn", "table-1", "S", "agent-controller-S"), client.calls)
        self.assertIn(("pass_turn", "table-1", "W", "agent-controller-W"), client.calls)
        self.assertIn(("pass_turn", "table-1", "N", "agent-controller-N"), client.calls)
        self.assertIn("1: E played single [♣️ 3]", result.output)
        self.assertIn("2: S passed; last play E single [♣️ 3]", result.output)
        self.assertIn("3: W passed; last play E single [♣️ 3]", result.output)
        self.assertIn("4: N passed; last play E single [♣️ 3]", result.output)

    def test_server_rejection_shows_resolved_command_cards(self) -> None:
        class RejectPlayClient(FakeClient):
            def play_cards(self, table_id, seat, controller_id, card_ids, *, declared_type=None):
                self.calls.append(("play_cards", table_id, seat, controller_id, card_ids, declared_type))
                raise GuandanClientError(
                    400,
                    "INVALID_HAND_TYPE: invalid hand",
                    {"rejection": {"code": "INVALID_HAND_TYPE", "message": "invalid hand"}},
                )

        client = RejectPlayClient()
        commands = iter(["play C3 S3", "quit"])

        result = run_cli(["--npc-lineup", "dummy"], input_fn=lambda prompt: next(commands), client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Error: HTTP 400: INVALID_HAND_TYPE: invalid hand", result.output)
        self.assertIn("Rejected cards: [♣️ 3, ♠️ 3]", result.output)

    def test_llm_rejection_shows_submitted_action_cards(self) -> None:
        class RejectLlmPlayClient(FakeClient):
            def play_cards(self, table_id, seat, controller_id, card_ids, *, declared_type=None):
                self.calls.append(("play_cards", table_id, seat, controller_id, card_ids, declared_type))
                raise GuandanClientError(
                    400,
                    "INVALID_HAND_TYPE: invalid hand",
                    {"rejection": {"code": "INVALID_HAND_TYPE", "message": "invalid hand"}},
                )

        client = RejectLlmPlayClient()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(Path(tmp), [{"display_name": "Pilot", "kind": "llm"}])

            result = run_cli(
                [
                    "--player-mode",
                    "llm",
                    "--display-name",
                    "Pilot",
                    "--npc-player-config",
                    str(path),
                ],
                input_fn=lambda prompt: "unused",
                client=client,
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Error: HTTP 400: INVALID_HAND_TYPE: invalid hand", result.output)
        self.assertIn("Rejected cards: [♣️ 3]", result.output)

    def test_human_timeout_refreshes_before_processing_input_and_prints_bot_actions(self) -> None:
        client = FakeClient()
        commands = iter(["", "quit"])
        timed_out = False

        def input_after_timeout(prompt):
            nonlocal timed_out
            if not timed_out and client.current_turn == "E":
                timed_out = True
                client.current_turn = "S"
                client.event_seq += 1
            return next(commands)

        result = run_cli(["--npc-lineup", "dummy"], input_fn=input_after_timeout, client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("pass_turn", "table-1", "S", "agent-controller-S"), client.calls)
        self.assertIn(("pass_turn", "table-1", "W", "agent-controller-W"), client.calls)
        self.assertIn(("pass_turn", "table-1", "N", "agent-controller-N"), client.calls)
        self.assertIn("2: S passed", result.output)
        self.assertIn("3: W passed", result.output)
        self.assertIn("4: N passed", result.output)

    def test_human_input_deadline_refreshes_without_command(self) -> None:
        client = FakeClient()
        timed_out = False

        def input_timeout(prompt):
            nonlocal timed_out
            if not timed_out and client.current_turn == "E":
                timed_out = True
                client.current_turn = "S"
                client.event_seq += 1
                return None
            return "quit"

        result = run_cli(["--npc-lineup", "dummy"], input_fn=input_timeout, client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("pass_turn", "table-1", "S", "agent-controller-S"), client.calls)
        self.assertIn(("pass_turn", "table-1", "W", "agent-controller-W"), client.calls)
        self.assertIn(("pass_turn", "table-1", "N", "agent-controller-N"), client.calls)
        self.assertIn("2: S passed", result.output)
        self.assertIn("3: W passed", result.output)
        self.assertIn("4: N passed", result.output)

    def test_human_timeout_prints_server_fallback_action(self) -> None:
        class TimeoutPassClient(FakeClient):
            def seat_snapshot(self, table_id, seat, controller_id):
                snapshot = super().seat_snapshot(table_id, seat, controller_id)
                snapshot["legal_action"] = "play_or_pass"
                snapshot["public"]["current_trick"] = {
                    "last_play_seat": "N",
                    "hand_type": "single",
                    "card_ids": ["D1-S-3"],
                }
                snapshot["public"]["action_deadline_epoch_ms"] = 1
                return snapshot

        client = TimeoutPassClient()
        timed_out = False

        def input_timeout(prompt):
            nonlocal timed_out
            if not timed_out:
                timed_out = True
                client.current_turn = "S"
                client.current_trick = {"last_play_seat": "N", "hand_type": "single", "card_ids": ["D1-S-3"]}
                client.event_seq += 1
                return None
            return "quit"

        result = run_cli(["--npc-lineup", "dummy"], input_fn=input_timeout, client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("E timed out; server fallback passed.", result.output)

    def test_timeout_fallback_shows_auto_played_lead_card_from_snapshot(self) -> None:
        output = format_timeout_fallback(
            {
                "table_id": "table-1",
                "phase": "PLAYING",
                "current_turn": "E",
                "acting_seat": "E",
                "current_trick": None,
                "event_seq": 4,
            },
            {
                "table_id": "table-1",
                "phase": "PLAYING",
                "current_turn": "S",
                "acting_seat": "S",
                "current_trick": {
                    "last_play_seat": "E",
                    "hand_type": "single",
                    "card_ids": ["D1-C-3"],
                },
                "event_seq": 6,
            },
            kind="lead",
        )

        self.assertEqual(output, "E timed out; server fallback played single [♣️ 3].")

    def test_timeout_fallback_shows_passed_out_trick_from_snapshot(self) -> None:
        output = format_timeout_fallback(
            {
                "table_id": "table-1",
                "phase": "PLAYING",
                "current_turn": "E",
                "acting_seat": "E",
                "current_trick": {
                    "last_play_seat": "N",
                    "hand_type": "single",
                    "card_ids": ["D1-S-3"],
                },
                "event_seq": 4,
            },
            {
                "table_id": "table-1",
                "phase": "PLAYING",
                "current_turn": "N",
                "acting_seat": "N",
                "current_trick": None,
                "event_seq": 6,
            },
            kind="play_or_pass",
        )

        self.assertEqual(output, "E timed out; server fallback passed and ended the trick. N leads next.")

    def test_deal_complete_starts_next_deal_then_restarts_after_match_complete(self) -> None:
        class DealCompleteClient(FakeClient):
            def start(self, table_id):
                self.calls.append(("start", table_id))
                self.event_seq += 1
                start_count = self.calls.count(("start", table_id))
                if start_count == 1:
                    self.phase = "DEAL_COMPLETE"
                    self.current_turn = None
                    events = [
                        {
                            "seq": self.event_seq,
                            "type": "LevelAdvanced",
                            "payload": {"team": "EW", "previous_level": "2", "next_level": "3"},
                        }
                    ]
                elif start_count == 2:
                    self.phase = "MATCH_COMPLETE"
                    self.current_turn = None
                    events = [
                        {
                            "seq": self.event_seq,
                            "type": "MatchEnded",
                            "payload": {"winning_team": "EW"},
                        }
                    ]
                else:
                    self.phase = "PLAYING"
                    self.current_turn = None
                    events = [
                        {
                            "seq": self.event_seq,
                            "type": "MatchStarted",
                            "payload": {"table_id": table_id},
                        }
                    ]
                return {"events": events, "snapshot": self._snapshot()}

        client = DealCompleteClient()

        result = run_cli([], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(client.calls.count(("start", "table-1")), 3)
        self.assertIn("match ended; winner EW", result.output)
        self.assertIn("Seat roles rotated for next match:", result.output)
        self.assertIn("Match 2 started.", result.output)

    def test_human_tribute_command_submits_resolved_card(self) -> None:
        class TributeClient(FakeClient):
            def start(self, table_id):
                self.calls.append(("start", table_id))
                self.phase = "TRIBUTE"
                self.current_turn = "E"
                return {"events": [], "snapshot": self._snapshot()}

            def seat_snapshot(self, table_id, seat, controller_id):
                snapshot = super().seat_snapshot(table_id, seat, controller_id)
                snapshot["legal_action"] = "tribute"
                return snapshot

        client = TributeClient()
        commands = iter(["tribute C3", "quit"])

        result = run_cli(["--npc-lineup", "dummy"], input_fn=lambda prompt: next(commands), client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("submit_tribute", "table-1", "E", "human-controller-E", "D1-C-3"), client.calls)
        self.assertIn("paid tribute", result.output)

    def test_human_return_command_submits_resolved_card(self) -> None:
        class ReturnClient(FakeClient):
            def start(self, table_id):
                self.calls.append(("start", table_id))
                self.phase = "TRIBUTE"
                self.current_turn = "E"
                return {"events": [], "snapshot": self._snapshot()}

            def seat_snapshot(self, table_id, seat, controller_id):
                snapshot = super().seat_snapshot(table_id, seat, controller_id)
                snapshot["legal_action"] = "return_tribute"
                return snapshot

        client = ReturnClient()
        commands = iter(["return C3", "quit"])

        result = run_cli(["--npc-lineup", "dummy"], input_fn=lambda prompt: next(commands), client=client)

        self.assertEqual(result.exit_code, 0)
        self.assertIn(("return_tribute", "table-1", "E", "human-controller-E", "D1-C-3"), client.calls)
        self.assertIn("returned tribute", result.output)

    def test_cli_dummy_lineup_uses_named_dummy_players(self) -> None:
        client = FakeClient()

        result = run_cli(["--npc-lineup", "dummy"], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        join_agent_calls = [call for call in client.calls if call[0] == "join_agent"]
        self.assertEqual({call[2] for call in join_agent_calls}, {"S", "W", "N"})
        self.assertEqual({call[3] for call in join_agent_calls}, {"Jade", "River", "Atlas"})

    def test_cli_llm_lineup_uses_named_llm_players(self) -> None:
        client = FakeClient()

        result = run_cli(["--npc-lineup", "llm"], input_fn=lambda prompt: "quit", client=client)

        self.assertEqual(result.exit_code, 0)
        join_agent_calls = [call for call in client.calls if call[0] == "join_agent"]
        self.assertEqual({call[2] for call in join_agent_calls}, {"S", "W", "N"})
        self.assertEqual({call[3] for call in join_agent_calls}, {"Jade", "River", "Atlas"})

    def test_cli_uses_custom_npc_player_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_player_storage(
                Path(tmp),
                [
                    {"display_name": "South Config", "kind": "dummy"},
                    {"display_name": "West Config", "kind": "dummy"},
                    {"display_name": "North Config", "kind": "dummy"},
                ],
            )
            client = FakeClient()

            result = run_cli(
                ["--npc-player-config", str(path), "--npc-lineup", "mixed"],
                input_fn=lambda prompt: "quit",
                client=client,
        )

        self.assertEqual(result.exit_code, 0)
        join_agent_calls = [call for call in client.calls if call[0] == "join_agent"]
        self.assertEqual({call[2] for call in join_agent_calls}, {"S", "W", "N"})
        self.assertEqual({call[3] for call in join_agent_calls}, {"South Config", "West Config", "North Config"})

    def test_card_formatter_hides_first_deck_and_uses_suit_emoji(self) -> None:
        self.assertEqual(format_card_id("D1-S-3"), "♠️ 3")
        self.assertEqual(format_card_id("D2-H-10"), "♥️ 10")
        self.assertEqual(format_card_id("D1-SJ"), "🃏 Small Joker")

    def test_hand_formatter_sorts_by_number_then_suit(self) -> None:
        self.assertEqual(
            format_hand(["D1-C-3", "D1-H-2", "D1-S-3", "D2-D-2", "D1-BJ", "D1-SJ"]),
            "♥️ 2  ♦️ 2  ♠️ 3  ♣️ 3  🃏 Small Joker  🃏 Big Joker",
        )

    def test_numeric_card_input_uses_sorted_hand_order(self) -> None:
        self.assertEqual(
            resolve_card_inputs(["1", "3"], {"hand": ["D1-C-3", "D1-H-2", "D1-S-3"]}),
            ("D1-H-2", "D1-C-3"),
        )

    def test_readable_card_input_resolves_against_hand(self) -> None:
        self.assertEqual(
            resolve_card_inputs(["S3", "♥2", "SJ"], {"hand": ["D1-SJ", "D1-H-2", "D2-S-3"]}),
            ("D2-S-3", "D1-H-2", "D1-SJ"),
        )

    def test_delimited_spade_jack_does_not_resolve_as_small_joker(self) -> None:
        self.assertEqual(
            resolve_card_inputs(["S-J", "SJ"], {"hand": ["D1-S-J", "D1-SJ"]}),
            ("D1-S-J", "D1-SJ"),
        )

    def test_repeated_readable_card_input_uses_distinct_physical_cards(self) -> None:
        self.assertEqual(
            resolve_card_inputs(["S3", "S3"], {"hand": ["D2-S-3", "D1-S-3"]}),
            ("D1-S-3", "D2-S-3"),
        )

    def test_bot_turn_race_refreshes_instead_of_printing_not_your_turn(self) -> None:
        class RaceBroker:
            seats = {"S": object()}

            def poll_once_results(self, seat):
                raise GuandanClientError(400, "NOT_YOUR_TURN", {"rejection": {"code": "NOT_YOUR_TURN"}})

        client = FakeClient()
        client.phase = "PLAYING"
        client.current_turn = "S"
        output = []

        snapshot = drive_bot_turns(
            client,
            Session("table-1", "E", "human-controller-E", RaceBroker(), {}),
            client._snapshot(),
            output.append,
            4,
        )

        self.assertEqual(snapshot["current_turn"], "S")
        self.assertEqual(output, [])

    def test_format_npc_metadata_shows_llm_provider_and_model(self) -> None:
        class Policy:
            config = type(
                "Config",
                (),
                {"provider_name": "codex-cli", "model_name": "gpt-5.2"},
            )()

        self.assertEqual(format_npc_metadata(Policy()), "codex-cli/gpt-5.2")

    def test_public_snapshot_shows_timer_when_deadline_is_present(self) -> None:
        output = format_public_snapshot(
            {
                "table_id": "table-1",
                "phase": "PLAYING",
                "event_seq": 1,
                "current_turn": "E",
                "action_deadline_epoch_ms": 9_999_999_999_999,
                "seats": {},
                "hand_counts": {},
                "finish_order": [],
            }
        )

        self.assertIn("Timer:", output)
        self.assertIn("PLAYING | Turn E", output)

    def test_seat_snapshot_merges_header_and_seats(self) -> None:
        output = format_seat_snapshot(
            {
                "public": {
                    "table_id": "table-1",
                    "phase": "PLAYING",
                    "event_seq": 1,
                    "current_turn": "E",
                    "seats": {"E": {"display_name": "East"}, "S": {"display_name": "South"}},
                    "hand_counts": {"E": 3, "S": 4, "W": 0, "N": 0},
                    "finish_order": [],
                },
                "seat": "E",
                "legal_action": "lead",
                "eligible_card_ids": ["D1-S-3"],
                "hand": ["D1-H-4", "D1-C-3", "D2-S-3"],
            }
        )

        self.assertIn("PLAYING | Seat E | Turn E", output)
        self.assertIn("E East 3 (You)", output)
        self.assertIn("S South 4", output)
        self.assertIn("W - 0 (F)", output)
        self.assertIn("N - 0", output)
        self.assertIn("Eligible cards: ♠️ 3", output)
        self.assertNotIn("Your seat:", output)

    def test_friend_mark_identifies_partner_for_viewer_seat(self) -> None:
        self.assertEqual(format_friend_mark("W", "E"), "(F)")
        self.assertEqual(format_friend_mark("S", "E"), "")

    def test_command_response_omits_action_prompted_events(self) -> None:
        output = format_command_response(
            {
                "events": [
                    {
                        "seq": 1,
                        "type": "CardsPlayed",
                        "payload": {"seat": "E", "hand_type": "single", "card_ids": ["D1-S-3"]},
                    },
                    {
                        "seq": 2,
                        "type": "ActionPrompted",
                        "payload": {"seat": "S", "kind": "play_or_pass"},
                    },
                ]
            }
        )

        self.assertEqual(output, "1: E played single [♠️ 3]\n")
        self.assertNotIn("ActionPrompted", output)

    def test_pass_response_repeats_current_played_cards(self) -> None:
        output = format_command_response(
            {
                "events": [{"seq": 2, "type": "PlayerPassed", "payload": {"seat": "S"}}],
                "snapshot": {
                    "current_trick": {
                        "last_play_seat": "E",
                        "hand_type": "single",
                        "card_ids": ["D1-C-3"],
                    }
                },
            }
        )

        self.assertEqual(output, "2: S passed; last play E single [♣️ 3]\n")


if __name__ == "__main__":
    unittest.main()
