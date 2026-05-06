from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from client.api import ActionRequest
from npc.broker.broker import NpcBroker
from npc.common.player import Player
from npc.llm_agent import LlmAgentConfig, LlmAgentPlayer, LlmAgentPolicy
from npc.llm_agent.prompts import SYSTEM_PROMPT


class InvalidProvider:
    def choose_action(self, prompt):
        return {
            "type": "play_cards",
            "card_ids": ["D1-S-A"],
            "thinking": "This intentionally chooses a card that is not in hand.",
        }


class StaticProvider:
    def __init__(self, action):
        self.action = action
        self.prompts = []

    def choose_action(self, prompt):
        self.prompts.append(prompt)
        return dict(self.action)


class StaticMemoryAgent:
    def __init__(self):
        self.calls = []

    def process_deal(self, memory, *, recent_actions, events, players_by_seat, observer_name):
        self.calls.append(
            {
                "recent_actions": recent_actions,
                "events": events,
                "players_by_seat": players_by_seat,
                "observer_name": observer_name,
            }
        )
        memory["techniques"]["level1"].append(
            {"summary": "Partner delivery worked.", "techniques": ["Lead low to transfer tempo."]}
        )
        memory["player_profiles"][players_by_seat["S"]] = {
            "latest_seat": "S",
            "personality": "aggressive",
            "playing_style": "sprints when close",
        }


class FakeClient:
    def __init__(self):
        self.calls = []

    def seat_snapshot(self, table_id, seat, controller_id):
        self.calls.append(("seat_snapshot", table_id, seat, controller_id))
        return {
            "public": {
                "phase": "PLAYING",
                "current_turn": seat,
                "event_seq": 4,
                "current_level": "2",
                "hand_counts": {"E": 27, "S": 1, "W": 1, "N": 27},
            },
            "seat": seat,
            "hand": ["D1-S-3"],
            "legal_action": "lead",
        }

    def play_cards(self, table_id, seat, controller_id, card_ids):
        self.calls.append(("play_cards", table_id, seat, controller_id, card_ids))
        return {
            "event_seq": 5,
            "events": [
                {
                    "seq": 5,
                    "type": "CardsPlayed",
                    "payload": {
                        "seat": seat,
                        "card_ids": list(card_ids),
                        "hand_type": "single",
                        "remaining_count": 0,
                    },
                }
            ],
        }

    def pass_turn(self, table_id, seat, controller_id):
        self.calls.append(("pass_turn", table_id, seat, controller_id))
        return {"event_seq": 5, "events": [{"seq": 5, "type": "PlayerPassed", "payload": {"seat": seat}}]}

    def submit_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("submit_tribute", table_id, seat, controller_id, card_id))
        return {"event_seq": 5, "events": []}

    def return_tribute(self, table_id, seat, controller_id, card_id):
        self.calls.append(("return_tribute", table_id, seat, controller_id, card_id))
        return {"event_seq": 5, "events": []}


class LlmAgentPolicyTests(unittest.TestCase):
    def test_policy_alias_points_to_player_class(self) -> None:
        self.assertIs(LlmAgentPolicy, LlmAgentPlayer)
        self.assertIsInstance(LlmAgentPolicy(), Player)

    def test_provider_prompt_includes_central_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = StaticProvider(
                {
                    "type": "play_cards",
                    "card_ids": ["D1-S-3"],
                    "thinking": "Lead low.",
                    "role": "primary_attacker",
                    "candidates": [{"type": "play_cards", "card_ids": ["D1-S-3"]}],
                    "recommended_action": {"type": "play_cards", "card_ids": ["D1-S-3"]},
                }
            )
            policy = LlmAgentPlayer(
                LlmAgentConfig(storage_dir=tmp, seat="S", personality="aggressive"),
                provider=provider,
            )

            action = policy.choose_action(_lead_request("S"))

            self.assertEqual(action, {"type": "play_cards", "card_ids": ["D1-S-3"]})
            self.assertEqual(provider.prompts[-1]["system_prompt"], SYSTEM_PROMPT)
            self.assertIn("Tribute giver", provider.prompts[-1]["system_prompt"])
            self.assertEqual(provider.prompts[-1]["table_context"]["partner"], "N")
            self.assertEqual(provider.prompts[-1]["table_context"]["opponents"], ["E", "W"])
            self.assertNotIn("role_estimate", provider.prompts[-1]["strategy_context"])
            self.assertEqual(provider.prompts[-1]["strategy_context"]["hand_features"]["card_count"], 1)
            self.assertEqual(provider.prompts[-1]["personality"]["type"], "aggressive")
            self.assertEqual(provider.prompts[-1]["personality"]["risk_tolerance"], "high")
            self.assertNotIn("card_player", provider.prompts[-1])
            self.assertNotIn("skills", provider.prompts[-1])
            self.assertIn("Bombs are tempo tools", provider.prompts[-1]["system_prompt"])
            self.assertIn("Rank strength is level-dependent", provider.prompts[-1]["system_prompt"])
            self.assertIn("Legal hand shapes are", provider.prompts[-1]["system_prompt"])
            self.assertIn("Bomb hierarchy", provider.prompts[-1]["system_prompt"])
            self.assertIn("A trick ends after three consecutive passes", provider.prompts[-1]["system_prompt"])
            self.assertIn("During tribute, submit the highest eligible card", provider.prompts[-1]["system_prompt"])
            self.assertIn('inferred "role"', provider.prompts[-1]["system_prompt"])
            decision = _read_json(Path(tmp) / "S" / "actions.json")[0]
            self.assertEqual(decision["llm_output"]["thinking"], "Lead low.")
            self.assertEqual(decision["llm_output"]["role"], "primary_attacker")
            self.assertEqual(
                decision["llm_output"]["recommended_action"],
                {"type": "play_cards", "card_ids": ["D1-S-3"]},
            )

    def test_provider_prompt_omits_eligible_card_ids_and_slim_public_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = StaticProvider({"type": "return_tribute", "card_id": "D1-S-3"})
            policy = LlmAgentPlayer(LlmAgentConfig(storage_dir=tmp, seat="S"), provider=provider)

            policy.choose_action(
                ActionRequest(
                    "r-tribute",
                    {
                        "kind": "return_tribute",
                        "current_level": "2",
                        "eligible_card_ids": ["D1-S-3"],
                        "return_rank_at_most_ten": True,
                    },
                    {
                        "table_id": "table-1",
                        "seat": "S",
                        "hand": ["D1-S-3", "D1-H-A"],
                        "eligible_card_ids": ["D1-S-3"],
                        "public": {
                            "phase": "TRIBUTE",
                            "event_seq": 8,
                            "current_level": "2",
                            "current_turn": "S",
                            "acting_seat": "S",
                            "hand_counts": {"E": 27, "S": 2, "W": 27, "N": 2},
                            "finish_order": [],
                            "current_trick": None,
                            "seats": {"S": {"display_name": "South"}},
                            "action_deadline_epoch_ms": 123,
                            "action_timeout_seconds": 45,
                        },
                    },
                )
            )

            prompt = provider.prompts[-1]
            self.assertNotIn("prompt", prompt)
            self.assertNotIn("eligible_card_ids", prompt["table_context"])
            self.assertNotIn("eligible_card_ids", prompt["snapshot"])
            self.assertEqual(prompt["snapshot"], {"hand": ["D1-S-3", "D1-H-A"]})
            self.assertEqual(prompt["table_context"]["phase"], "TRIBUTE")
            self.assertEqual(prompt["table_context"]["event_seq"], 8)
            self.assertEqual(prompt["table_context"]["hand_counts"]["S"], 2)
            self.assertEqual(prompt["table_context"]["prompt_kind"], "return_tribute")
            self.assertEqual(prompt["table_context"]["return_rank_at_most_ten"], True)

    def test_play_or_pass_prompt_omits_deterministic_card_player_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = StaticProvider({"type": "play_cards", "card_ids": ["D1-S-4"], "thinking": "Beat cheaply."})
            policy = LlmAgentPlayer(LlmAgentConfig(storage_dir=tmp, seat="S"), provider=provider)

            action = policy.choose_action(
                ActionRequest(
                    "r-1",
                    {
                        "kind": "play_or_pass",
                        "current_level": "2",
                        "current_trick": {"card_ids": ["D1-S-3"], "hand_type": "single", "last_play_seat": "E"},
                    },
                    {
                        "seat": "S",
                        "hand": ["D1-S-4", "D1-S-A"],
                        "public": {
                            "current_level": "2",
                            "current_turn": "S",
                            "current_trick": {"card_ids": ["D1-S-3"], "hand_type": "single", "last_play_seat": "E"},
                        },
                    },
                )
            )

            self.assertEqual(action["type"], "play_cards")
            self.assertEqual(action["card_ids"], ["D1-S-4"])
            self.assertNotIn("card_player", provider.prompts[-1])
            self.assertEqual(
                provider.prompts[-1]["table_context"]["current_trick"],
                {"card_ids": ["D1-S-3"], "hand_type": "single", "last_play_seat": "E"},
            )

    def test_provider_prompt_splits_memory_into_techniques_and_other_player_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            memory_path.write_text(
                json.dumps(
                    {
                        "player_name": "Jade",
                        "seat": "S",
                        "play_style": "aggressive",
                        "score": {"deals_played": 3},
                        "techniques": {
                            "level1": [{"summary": "Transfer tempo.", "techniques": ["Lead low to partner."]}],
                            "level2": {"team_coordination": ["Protect partner tempo."]},
                        },
                        "player_profiles": {
                            "Jade": {"latest_seat": "S", "playing_style": "self profile"},
                            "Ming": {"latest_seat": "E", "playing_style": "fast"},
                            "River": {"latest_seat": "W", "playing_style": "patient"},
                            "Atlas": {"latest_seat": "N", "playing_style": "defensive"},
                            "Old Player": {"latest_seat": "E", "playing_style": "stale"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            provider = StaticProvider({"type": "play_cards", "card_ids": ["D1-S-3"]})
            policy = LlmAgentPlayer(
                LlmAgentConfig(memory_path=memory_path, action_log_path=Path(tmp) / "actions.json", seat="S"),
                provider=provider,
            )

            policy.choose_action(
                ActionRequest(
                    "r-1",
                    {"kind": "lead", "current_level": "2"},
                    {
                        "seat": "S",
                        "hand": ["D1-S-3"],
                        "players_by_seat": {"E": "Ming", "S": "Jade", "W": "River", "N": "Atlas"},
                        "public": {"current_level": "2", "current_turn": "S"},
                    },
                )
            )

            prompt = provider.prompts[-1]

        self.assertNotIn("memory", prompt)
        self.assertNotIn("user_profile", prompt)
        self.assertEqual(prompt["techniques"]["level1"][0]["summary"], "Transfer tempo.")
        self.assertEqual(set(prompt["player_profiles"]), {"Ming", "River", "Atlas"})
        self.assertNotIn("Jade", prompt["player_profiles"])
        self.assertNotIn("Old Player", prompt["player_profiles"])

    def test_explicit_player_storage_paths_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "custom-memory.json"
            action_path = Path(tmp) / "custom-actions.json"
            policy = LlmAgentPolicy(
                LlmAgentConfig(
                    player_name="South Agent",
                    memory_path=memory_path,
                    action_log_path=action_path,
                )
            )

            action = policy.choose_action(_lead_request("S"))

            self.assertEqual(action["type"], "play_cards")
            self.assertTrue(memory_path.exists())
            self.assertTrue(action_path.exists())
            self.assertEqual(_read_json(memory_path)["player_name"], "South Agent")

    def test_default_storage_paths_are_isolated_by_player_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policies = {
                name: LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, player_name=name, seat=seat))
                for name, seat in (("South Agent", "S"), ("West Agent", "W"), ("North Agent", "N"))
            }

            for seat, policy in zip(("S", "W", "N"), policies.values(), strict=True):
                policy.choose_action(_lead_request(seat))

            paths = [next(iter(policy.storage_paths.values())) for policy in policies.values()]
            self.assertEqual(len(set(paths)), 3)
            for name, seat, (memory_path, action_path) in zip(
                ("South-Agent", "West-Agent", "North-Agent"),
                ("S", "W", "N"),
                paths,
                strict=True,
            ):
                self.assertEqual(memory_path, Path(tmp) / name / "memory.json")
                self.assertEqual(action_path, Path(tmp) / name / "actions.json")
                self.assertEqual(_read_json(memory_path)["seat"], seat)

    def test_player_name_storage_survives_seat_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, player_name="Jade", seat="S"))

            policy.choose_action(_lead_request("S"))
            policy.choose_action(_lead_request("W"))

            self.assertEqual(len(policy.storage_paths), 1)
            actions = _read_json(Path(tmp) / "Jade" / "actions.json")
            self.assertEqual([entry["seat"] for entry in actions], ["S", "W"])
            self.assertEqual(_read_json(Path(tmp) / "Jade" / "memory.json")["seat"], "W")

    def test_action_log_is_capped_by_configured_entry_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = LlmAgentPolicy(
                LlmAgentConfig(storage_dir=tmp, seat="S", max_action_log_entries=2),
                provider=StaticProvider({"type": "play_cards", "card_ids": ["D1-S-3"]}),
            )

            policy.choose_action(_lead_request("S", request_id="r-1"))
            policy.choose_action(_lead_request("S", request_id="r-2"))
            policy.choose_action(_lead_request("S", request_id="r-3"))

            actions = _read_json(Path(tmp) / "S" / "actions.json")

        self.assertEqual([entry["request_id"] for entry in actions], ["r-2", "r-3"])

    def test_invalid_provider_action_falls_back_and_logs_llm_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            action_path = Path(tmp) / "actions.json"
            policy = LlmAgentPolicy(
                LlmAgentConfig(action_log_path=action_path, memory_path=Path(tmp) / "memory.json"),
                provider=InvalidProvider(),
            )

            action = policy.choose_action(_lead_request("S"))

            self.assertEqual(action["type"], "play_cards")
            self.assertEqual(action["card_ids"], ["D1-S-3"])
            self.assertNotIn("thinking", action)
            entries = _read_json(action_path)
            self.assertTrue(entries[0]["fallback_used"])
            self.assertEqual(entries[0]["fallback_reason"], "provider output was invalid for the current prompt")
            self.assertEqual(
                entries[0]["llm_output"]["thinking"],
                "This intentionally chooses a card that is not in hand.",
            )

    def test_provider_prompt_uses_only_that_player_recent_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            south_provider = StaticProvider(
                {
                    "type": "play_cards",
                    "card_ids": ["D1-S-3"],
                    "thinking": "Lead one low card.",
                    "memory_updates": {"skills": ["South skill"]},
                }
            )
            west_provider = StaticProvider(
                {
                    "type": "play_cards",
                    "card_ids": ["D1-S-3"],
                    "thinking": "Lead one low card.",
                    "memory_updates": {"skills": ["West skill"]},
                }
            )
            south = LlmAgentPolicy(
                LlmAgentConfig(storage_dir=tmp, player_name="South Agent", seat="S"),
                provider=south_provider,
            )
            west = LlmAgentPolicy(
                LlmAgentConfig(storage_dir=tmp, player_name="West Agent", seat="W"),
                provider=west_provider,
            )

            south.choose_action(_lead_request("S"))
            west.choose_action(_lead_request("W"))
            south.choose_action(_lead_request("S"))

            self.assertEqual(south_provider.prompts[-1]["recent_actions"][0]["actor_seat"], "S")
            south_memory = json.dumps(_read_json(Path(tmp) / "South-Agent" / "memory.json"))
            west_memory = json.dumps(_read_json(Path(tmp) / "West-Agent" / "memory.json"))
            self.assertNotIn("West skill", south_memory)
            self.assertNotIn("South skill", west_memory)

    def test_provider_prompt_recent_actions_are_limited_to_current_deal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            action_path = Path(tmp) / "actions.json"
            action_path.write_text(
                json.dumps(
                    [
                        {
                            "kind": "decision",
                            "deal_id": 1,
                            "seat": "S",
                            "event_seq": 4,
                            "selected_action": {"type": "play_cards", "card_ids": ["D1-S-4"]},
                        },
                        {
                            "kind": "observed_action",
                            "deal_id": 1,
                            "actor_seat": "S",
                            "event_seq": 5,
                            "action": {"type": "play_cards", "card_ids": ["D1-S-5"]},
                            "response_events": [
                                {"seq": 5, "type": "CardsPlayed", "payload": {"seat": "S"}},
                            ],
                        },
                        {
                            "kind": "observed_action",
                            "deal_id": 1,
                            "actor_seat": "N",
                            "event_seq": 10,
                            "action": {"type": "play_cards", "card_ids": ["D1-S-6"]},
                            "response_events": [
                                {"seq": 10, "type": "DealEnded", "payload": {"winning_team": "SN"}},
                            ],
                        },
                        {
                            "kind": "observed_action",
                            "deal_id": 2,
                            "actor_seat": "W",
                            "actor_name": "River",
                            "event_seq": 15,
                            "action": {"type": "play_cards", "card_ids": ["D1-S-7"]},
                            "response_events": [
                                {"seq": 15, "type": "CardsPlayed", "payload": {"seat": "W"}},
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            provider = StaticProvider({"type": "play_cards", "card_ids": ["D1-S-3"]})
            policy = LlmAgentPolicy(
                LlmAgentConfig(action_log_path=action_path, memory_path=Path(tmp) / "memory.json"),
                provider=provider,
            )

            policy.choose_action(_lead_request("S", deal_id=2))

            recent_actions = provider.prompts[-1]["recent_actions"]

        self.assertEqual(len(recent_actions), 1)
        self.assertEqual(
            recent_actions[0],
            {
                "actor_seat": "W",
                "actor_name": "River",
                "action": {"type": "play_cards", "card_ids": ["D1-S-7"]},
            },
        )
        self.assertNotIn("response_events", recent_actions[0])

    def test_provider_prompt_recent_actions_empty_after_deal_end_until_current_deal_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            action_path = Path(tmp) / "actions.json"
            action_path.write_text(
                json.dumps(
                    [
                        {"kind": "decision", "deal_id": 1, "seat": "S", "event_seq": 4},
                        {
                            "kind": "observed_action",
                            "deal_id": 1,
                            "actor_seat": "N",
                            "event_seq": 10,
                            "response_events": [
                                {"seq": 10, "type": "DealEnded", "payload": {"winning_team": "SN"}},
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            provider = StaticProvider({"type": "play_cards", "card_ids": ["D1-S-3"]})
            policy = LlmAgentPolicy(
                LlmAgentConfig(action_log_path=action_path, memory_path=Path(tmp) / "memory.json"),
                provider=provider,
            )

            policy.choose_action(_lead_request("S", deal_id=2))

            recent_actions = provider.prompts[-1]["recent_actions"]

        self.assertEqual(recent_actions, [])

    def test_deal_end_updates_techniques_and_name_keyed_player_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_agent = StaticMemoryAgent()
            policy = LlmAgentPolicy(
                LlmAgentConfig(storage_dir=tmp, player_name="Jade", seat="S"),
                memory_agent=memory_agent,
            )
            action_path = Path(tmp) / "Jade" / "actions.json"
            action_path.parent.mkdir(parents=True, exist_ok=True)
            action_path.write_text(
                json.dumps(
                    [
                        {
                            "kind": "observed_action",
                            "deal_id": 1,
                            "actor_seat": "E",
                            "action": {"type": "play_cards", "card_ids": ["D1-S-A"]},
                        },
                        {
                            "kind": "observed_action",
                            "deal_id": 2,
                            "actor_seat": "W",
                            "action": {"type": "pass"},
                        },
                    ]
                ),
                encoding="utf-8",
            )

            policy.observe_action(
                {
                    "table_id": "table-1",
                    "deal_id": 2,
                    "observer_seat": "S",
                    "observer_name": "Jade",
                    "actor_seat": "S",
                    "actor_name": "Jade",
                    "players_by_seat": {"E": "Ming", "S": "Jade", "W": "River", "N": "Atlas"},
                    "action": {"type": "play_cards", "card_ids": ["D1-S-3"]},
                    "event_seq": 12,
                    "events": [
                        {"seq": 10, "type": "CardsPlayed", "payload": {"seat": "S", "card_ids": ["D1-S-3"]}},
                        {
                            "seq": 11,
                            "type": "DealEnded",
                            "payload": {"finish_order": ["S", "N", "E", "W"], "winning_team": "SN"},
                        },
                        {
                            "seq": 12,
                            "type": "LevelAdvanced",
                            "payload": {"team": "SN", "next_level": "5"},
                        },
                    ],
                }
            )

            memory = _read_json(Path(tmp) / "Jade" / "memory.json")
            self.assertEqual(memory_agent.calls[0]["observer_name"], "Jade")
            self.assertEqual(memory_agent.calls[0]["players_by_seat"]["S"], "Jade")
            self.assertEqual([entry["deal_id"] for entry in memory_agent.calls[0]["recent_actions"]], [2, 2])
            self.assertEqual(memory_agent.calls[0]["recent_actions"][0]["actor_seat"], "W")
            self.assertEqual(memory_agent.calls[0]["recent_actions"][1]["actor_seat"], "S")
            self.assertEqual(memory["techniques"]["level1"][0]["summary"], "Partner delivery worked.")
            self.assertIn("Jade", memory["player_profiles"])
            self.assertEqual(memory["player_profiles"]["Jade"]["latest_seat"], "S")
            self.assertEqual(memory["score"]["deals_played"], 1)
            self.assertEqual(memory["score"]["wins"], 1)

    def test_broker_notifies_all_llm_agents_after_each_submitted_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broker = NpcBroker(FakeClient(), "table-1")
            south = LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, player_name="South Agent", seat="S"))
            west = LlmAgentPolicy(LlmAgentConfig(storage_dir=tmp, player_name="West Agent", seat="W"))
            broker.add_seat("S", south, "South Agent").controller_id = "c-S"
            broker.add_seat("W", west, "West Agent").controller_id = "c-W"

            actions = broker.poll_once("S")

            self.assertEqual(actions[0]["type"], "play_cards")
            west_log = _read_json(Path(tmp) / "West-Agent" / "actions.json")
            observed = [entry for entry in west_log if entry["kind"] == "observed_action"]
            self.assertEqual(observed[0]["actor_seat"], "S")
            self.assertEqual(observed[0]["response_events"][0]["type"], "CardsPlayed")

    def test_action_log_does_not_store_opponent_private_hands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            action_path = Path(tmp) / "actions.json"
            policy = LlmAgentPolicy(
                LlmAgentConfig(memory_path=Path(tmp) / "memory.json", action_log_path=action_path)
            )

            policy.choose_action(_lead_request("S"))

            entry = _read_json(action_path)[0]
            self.assertEqual(entry["snapshot"]["hand"], ["D1-S-3"])
            self.assertNotIn("hands", json.dumps(entry))


def _lead_request(seat: str, *, deal_id: int | None = None, request_id: str | None = None) -> ActionRequest:
    public = {
        "phase": "PLAYING",
        "event_seq": 4,
        "current_level": "2",
        "current_turn": seat,
        "hand_counts": {"E": 27, "S": 1, "W": 1, "N": 27},
    }
    if deal_id is not None:
        public["deal_id"] = deal_id
    return ActionRequest(
        request_id=request_id or f"r-{seat}",
        prompt={"kind": "lead", "current_level": "2"},
        snapshot={
            "table_id": "table-1",
            "seat": seat,
            "hand": ["D1-S-3"],
            "public": public,
        },
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
