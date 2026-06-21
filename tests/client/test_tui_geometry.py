from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from client.tui.textual_app import GuandanTextualApp, MIN_UI_WIDTH
from tests.test_cli import FakeClient


class FullHandClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        suits = ["S", "H", "D", "C"]
        ranks = ["3", "4", "5", "6", "7", "8", "9", "10", "J"]
        self.hands["E"] = [f"D1-{suit}-{rank}" for rank in ranks for suit in suits]


class TuiGeometryTests(unittest.TestCase):
    def test_full_hand_actions_and_feed_fit_in_eighty_by_forty_viewport(self) -> None:
        asyncio.run(self._run_geometry_check())

    async def _run_geometry_check(self) -> None:
        client = FullHandClient()
        app = GuandanTextualApp(args=_args(), client=client)
        with patch("client.session._choose_available_seat", return_value="E"):
            async with app.run_test(size=(MIN_UI_WIDTH, 40)) as pilot:
                await pilot.pause(0.8)
                regions = [app.query_one(f"#card-{index}").region for index in range(36)]
                regions.extend(
                    app.query_one(f"#{button_id}").region
                    for button_id in (
                        "play-action",
                        "pass-action",
                        "tribute-action",
                        "return-action",
                        "clear-action",
                        "refresh-action",
                    )
                )
                regions.append(app.query_one("#feed").region)
                self.assertTrue(all(region.x >= 0 and region.x + region.width <= MIN_UI_WIDTH for region in regions))
                self.assertTrue(all(region.y >= 0 and region.y + region.height <= 40 for region in regions))
                board_text = " ".join(
                    str(app.query_one(f"#{seat_id}").content)
                    for seat_id in ("seat-e", "seat-s", "seat-w", "seat-n")
                )
                self.assertIn("Tester", board_text)
                self.assertIn("Jade", board_text)
                self.assertIn("River", board_text)
                self.assertIn("Atlas", board_text)
                self.assertIn("36 cards", str(app.query_one("#seat-e").content))
                await pilot.click("#card-1")
                await pilot.pause(0.1)
                self.assertEqual(app.selected_ids, {"D1-H-3"})
                await pilot.click("#play-action")
                await pilot.pause(0.8)
                self.assertIn(
                    ("play_cards", "table-1", "E", "human-controller-E", ("D1-H-3",), None),
                    client.calls,
                )
                app.exit()


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        base_url="http://127.0.0.1:8000",
        table_id=None,
        player_id=None,
        controller_id=None,
        display_name="Tester",
        player_mode="human",
        gossiper_mode="none",
        max_bot_actions=8,
        npc_lineup="dummy",
        npc_player_config=None,
    )


if __name__ == "__main__":
    unittest.main()
