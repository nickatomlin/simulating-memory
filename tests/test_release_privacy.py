import json
from pathlib import Path
import unittest

from bench.core.scoring import ROOT
from scripts.release_privacy import ABSOLUTE_TIME_FIELDS, PRIVATE_FIELDS, clean_record, field_name


class PrivacyTests(unittest.TestCase):
    def test_nested_identifiers_and_browser_details_are_removed(self):
        private_id = "a" * 24
        record = {"participant_id": private_id, "completed_at": "2026-01-01",
                  "duration_ms": 1234, "request": {"remote_addr": "127.0.0.1", "user_agent": "browser"},
                  "payload": {"reading": {"startedAtMs": 123456, "readingDurationMs": 987},
                              "answers": [{"correct": True}]}}
        clean = clean_record(record, {private_id: "participant-application-0001"})
        self.assertEqual(clean, {"participant_id": "participant-application-0001", "duration_ms": 1234,
                                 "payload": {"reading": {"readingDurationMs": 987}, "answers": [{"correct": True}]}})

    def test_all_released_human_records_have_pseudonymous_paths_and_fields(self):
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(field_name(key), PRIVATE_FIELDS | ABSOLUTE_TIME_FIELDS)
                    if key == "participant_id":
                        self.assertRegex(item, r"^participant-(?:application-)?\d{4}$")
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        for folder in ["runs/human", "runs/human-2026-07-27", "application/approved_runs"]:
            for path in (ROOT / folder).rglob("*.json"):
                with self.subTest(file=str(path.relative_to(ROOT))):
                    record = json.loads(path.read_text())
                    self.assertEqual(path.name, record["run_id"] + ".json")
                    self.assertIsInstance(record["completion_order"], int)
                    check(record)
        self.assertFalse((ROOT / "runs/human/feedback").exists())


if __name__ == "__main__":
    unittest.main()
