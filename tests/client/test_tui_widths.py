from __future__ import annotations

import math
import unittest

from client.tui.textual_app import (
    ACTION_BUTTON_WIDTH,
    ACTION_GRID_COLUMNS,
    CARD_BUTTON_WIDTH,
    HAND_GRID_COLUMNS,
    MAX_HAND_CARDS,
    MIN_UI_WIDTH,
    action_columns_for_width,
    feed_table_width,
    hand_columns_for_width,
)


class TuiWidthTests(unittest.TestCase):
    def test_eighty_column_terminal_has_wrapping_card_grid_budget(self) -> None:
        columns = hand_columns_for_width(MIN_UI_WIDTH)

        self.assertEqual(columns, HAND_GRID_COLUMNS)
        self.assertEqual(math.ceil(MAX_HAND_CARDS / columns), 5)
        self.assertLessEqual(HAND_GRID_COLUMNS * CARD_BUTTON_WIDTH, MIN_UI_WIDTH)

    def test_eighty_column_terminal_fits_all_action_buttons_on_one_row(self) -> None:
        self.assertEqual(action_columns_for_width(MIN_UI_WIDTH), ACTION_GRID_COLUMNS)
        self.assertGreaterEqual(ACTION_GRID_COLUMNS, 6)
        self.assertLessEqual(ACTION_GRID_COLUMNS * ACTION_BUTTON_WIDTH, MIN_UI_WIDTH)

    def test_feed_columns_leave_room_for_table_chrome(self) -> None:
        self.assertLessEqual(feed_table_width(), MIN_UI_WIDTH - 8)


if __name__ == "__main__":
    unittest.main()
