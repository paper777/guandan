from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from client.app import _should_run_textual


class AppUiSelectionTests(unittest.TestCase):
    def test_explicit_textual_ui_runs_even_with_injected_input(self) -> None:
        self.assertTrue(_should_run_textual(SimpleNamespace(ui="textual"), lambda prompt: "quit"))

    def test_auto_ui_stays_classic_for_injected_input(self) -> None:
        self.assertFalse(_should_run_textual(SimpleNamespace(ui="auto"), lambda prompt: "quit"))

    def test_auto_ui_uses_textual_for_real_tty(self) -> None:
        with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
            self.assertTrue(_should_run_textual(SimpleNamespace(ui="auto"), None))


if __name__ == "__main__":
    unittest.main()
