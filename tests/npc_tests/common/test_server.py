from __future__ import annotations

import unittest

from client.types import ActionRequest


class ActionRequestTests(unittest.TestCase):
    def test_builds_from_payload_with_dict_defaults(self) -> None:
        request = ActionRequest.from_payload({"request_id": "r-1", "prompt": {"kind": "lead"}})

        self.assertEqual(request.request_id, "r-1")
        self.assertEqual(request.prompt, {"kind": "lead"})
        self.assertEqual(request.snapshot, {})


if __name__ == "__main__":
    unittest.main()
