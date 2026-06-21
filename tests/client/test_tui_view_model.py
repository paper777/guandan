from __future__ import annotations

import unittest

from client.tui.view_model import (
    action_availability,
    action_rows_from_response,
    card_views,
    selected_card_ids_in_hand_order,
    table_view,
    trick_view,
)


class TuiViewModelTests(unittest.TestCase):
    def test_card_views_track_selection_and_tribute_eligibility(self) -> None:
        views = card_views(
            ["D1-H-4", "D1-C-3", "D2-S-3"],
            selected_ids=["D1-C-3"],
            eligible_card_ids=["D1-C-3", "D1-H-4"],
        )

        self.assertEqual([view.card_id for view in views], ["D2-S-3", "D1-C-3", "D1-H-4"])
        self.assertEqual([view.index for view in views], [1, 2, 3])
        self.assertTrue(views[1].selected)
        self.assertTrue(views[1].eligible)
        self.assertFalse(views[0].eligible)

    def test_action_availability_matches_prompt_type(self) -> None:
        self.assertTrue(action_availability({"legal_action": "play_or_pass"}, ["D1-S-3"]).can_play)
        self.assertTrue(action_availability({"legal_action": "play_or_pass"}, []).can_pass)
        self.assertFalse(action_availability({"legal_action": "tribute"}, []).can_submit_tribute)
        self.assertTrue(action_availability({"legal_action": "tribute"}, ["D1-S-A"]).can_submit_tribute)
        self.assertTrue(action_availability({"legal_action": "return_tribute"}, ["D1-S-9"]).can_return_tribute)

    def test_selected_card_ids_follow_visible_hand_order(self) -> None:
        ordered = selected_card_ids_in_hand_order(
            ["D1-H-4", "D1-C-3", "D2-S-3"],
            {"D1-H-4", "D2-S-3"},
        )

        self.assertEqual(ordered, ("D2-S-3", "D1-H-4"))

    def test_action_rows_split_event_details_into_columns(self) -> None:
        rows = action_rows_from_response(
            {
                "events": [
                    {
                        "seq": 10,
                        "type": "CardsPlayed",
                        "payload": {"seat": "E", "hand_type": "single", "card_ids": ["D1-S-3"]},
                    },
                    {"seq": 11, "type": "PlayerPassed", "payload": {"seat": "S"}},
                    {"seq": 12, "type": "ActionPrompted", "payload": {"seat": "W", "kind": "play_or_pass"}},
                ],
                "snapshot": {
                    "current_trick": {
                        "last_play_seat": "E",
                        "hand_type": "single",
                        "card_ids": ["D1-S-3"],
                    }
                },
            }
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].seq, "10")
        self.assertEqual(rows[0].seat, "E")
        self.assertEqual(rows[0].action, "played single")
        self.assertEqual(rows[0].cards, "♠️ 3")
        self.assertEqual(rows[1].seat, "S")
        self.assertEqual(rows[1].action, "passed")
        self.assertIn("E single", rows[1].detail)

    def test_table_view_marks_acting_viewer_and_partner(self) -> None:
        view = table_view(
            {
                "table_id": "table-1",
                "phase": "PLAYING",
                "current_turn": "E",
                "acting_seat": "E",
                "deal_id": 2,
                "current_level": "5",
                "level_by_team": {"EW": "5", "SN": "7"},
                "seats": {"E": {"display_name": "East"}, "W": {"display_name": "West"}},
                "hand_counts": {"E": 3, "W": 4},
                "finish_order": ["S"],
            },
            viewer_seat="E",
            npc_metadata={"W": "dummy"},
        )

        seats = {seat.seat: seat for seat in view.seats}
        self.assertEqual(view.level_summary, "Us 5 / Them 7")
        self.assertTrue(seats["E"].is_viewer)
        self.assertTrue(seats["E"].is_acting)
        self.assertTrue(seats["W"].is_partner)
        self.assertEqual(seats["W"].metadata, "dummy")
        self.assertTrue(seats["S"].is_finished)

    def test_trick_view_formats_empty_and_active_tricks(self) -> None:
        self.assertEqual(trick_view(None).summary, "No active trick")
        self.assertEqual(
            trick_view({"last_play_seat": "N", "hand_type": "pair", "primary_rank": "9", "card_ids": ["D1-C-9"]}).summary,
            "N pair rank 9: ♣️ 9",
        )


if __name__ == "__main__":
    unittest.main()
