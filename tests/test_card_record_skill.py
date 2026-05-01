from __future__ import annotations

import unittest
from pathlib import Path


class CardRecordSkillTests(unittest.TestCase):
    def test_skill_has_frontmatter_and_card_recorder_guidance(self) -> None:
        text = Path("skills/card-record/SKILL.md").read_text(encoding="utf-8")

        self.assertTrue(text.startswith("---\nname: card-record\n"))
        self.assertIn("description:", text)
        self.assertIn("tools.card_recorder.CardRecorder", text)
        self.assertIn("Never infer hidden cards", text)


if __name__ == "__main__":
    unittest.main()
