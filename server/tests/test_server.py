from __future__ import annotations

import unittest

from guandan.server import DEFAULT_APP, parse_args


class ServerEntrypointTests(unittest.TestCase):
    def test_parse_default_server_options(self) -> None:
        options = parse_args([])

        self.assertEqual(options.app, DEFAULT_APP)
        self.assertEqual(options.host, "127.0.0.1")
        self.assertEqual(options.port, 8000)
        self.assertFalse(options.reload)
        self.assertEqual(options.log_level, "info")

    def test_parse_custom_server_options(self) -> None:
        options = parse_args(["--host", "0.0.0.0", "--port", "9000", "--reload", "--log-level", "debug"])

        self.assertEqual(options.host, "0.0.0.0")
        self.assertEqual(options.port, 9000)
        self.assertTrue(options.reload)
        self.assertEqual(options.log_level, "debug")


if __name__ == "__main__":
    unittest.main()
