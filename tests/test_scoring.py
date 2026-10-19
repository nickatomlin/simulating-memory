from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from bench.core import scoring as s
from src.score import discover_models, score_models, write_reports


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def model(self, task, rows, compactor=False):
        folder = self.root / "model/tasks"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{'wm_' if compactor else ''}{task}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        return folder.parent

    def human(self, task, records):
        folder = self.root / "humans" / s.HUMAN_TASK_DIR[task]
        folder.mkdir(parents=True, exist_ok=True)
        for i, record in enumerate(records):
            (folder / f"run-{i}.json").write_text(json.dumps(record))
        return folder.parent

    def test_exact_wasserstein(self):
        self.assertEqual(s.wasserstein_1d([0, 1, 2], [3]), 2.)
        self.assertEqual(s.wasserstein_1d([1, 1], [1]), 0.)
        self.assertAlmostEqual(s.wasserstein_1d([0, 0, 1], [1]), 2/3)
        self.assertTrue(np.isnan(s.wasserstein_1d([], [1])))
        with self.assertRaises(ValueError):
            s.wasserstein_1d([np.nan], [1])

    def test_wasserstein_against_scipy(self):
        try:
            from scipy.stats import wasserstein_distance
        except ImportError:
            self.skipTest("Optional scipy test dependency is not installed")
        rng = np.random.default_rng(42)
        for n, m in [(1, 100), (17, 23), (100, 50)]:
            x, y = rng.integers(-5, 6, n), rng.integers(-5, 6, m)
            self.assertAlmostEqual(s.wasserstein_1d(x, y), wasserstein_distance(x, y), places=14)

    def test_digit_span_two_attempts_stop_and_ignore_later_success(self):
        rows = [{"participant_id": 1, "span_length": span, "sequence_index": seq,
                 "metrics": {"exact": exact}}
                for span, attempts in [(2, [1, 0]), (3, [0, 1]), (4, [0, 0]), (5, [1, 1])]
                for seq, exact in enumerate(attempts, 1)]
        self.assertEqual(s.digit_span_scores(rows), [3.])
        # A third, successful sequence does not rescue the failed span or
        # create another participant in the historical layout.
        rows.append({"participant_id": 1, "span_length": 4, "sequence_index": 3, "metrics": {"exact": 1}})
        with self.assertWarns(s.ScoringWarning):
            self.assertEqual(s.digit_span_scores(rows), [3.])

    def test_digit_span_does_not_jump_missing_lengths(self):
        rows = [{"participant_id": 1, "span_length": n, "sequence_index": 1, "metrics": {"exact": 1}}
                for n in [2, 4]]
        self.assertEqual(s.digit_span_scores(rows), [2.])

    def test_human_summary_and_completed_latest_selection(self):
        root = self.human("digit_span_forward", [
            {"participant_id": "a", "status": "completed", "completed_at": "2026-01-01T00:00:00Z", "summary": {"bestSpan": 6}},
            {"participant_id": "a", "status": "completed", "completed_at": "2026-02-01T00:00:00Z", "summary": {"bestSpan": 1},
             "payload": {"trials": [{"length": 2, "correct": False}]}},
            {"participant_id": "b", "status": "abandoned", "summary": {"bestSpan": 20}},
        ])
        np.testing.assert_allclose(s.human_scores("digit_span_forward", root), [.05])
        np.testing.assert_allclose(sorted(s.human_scores("digit_span_forward", root, selection="paper")), [.05, .3])

    def test_nback_aggregates_blocks_per_session(self):
        model = self.model("nback", [
            {"condition_id": "C1", "participant_id": 1, "repeat_index": 1,
             "acc_over_answered": a, "acc_over_14": 0.} for a in [1., .5, 0.]])
        np.testing.assert_allclose(s.llm_scores("nback", model, "C1"), [.5])
        root = self.human("nback", [{"status": "completed", "participant_id": "a",
                                   "summary": {"correctResponses": 21, "totalResponses": 42},
                                   "payload": {"trials": [{"phase": "practice", "correct": False}]}}])
        np.testing.assert_allclose(s.human_scores("nback", root), [.5])

    def test_word_recognition_counts_trials_reached_not_correct(self):
        model = self.model("word_recognition", [{"condition_id": "C1", "repeat_index": 1,
                                                "metrics": {"score": 7}, "per_trial": [{}] * 10}])
        root = self.human("word_recognition", [{"status": "completed", "participant_id": "a",
                                               "summary": {"correctResponses": 7, "trialsCompleted": 10}}])
        np.testing.assert_allclose(s.llm_scores("word_recognition", model, "C1"), [.1])
        np.testing.assert_allclose(s.human_scores("word_recognition", root), [.1])
        np.testing.assert_allclose(s.llm_scores("word_recognition", model, "C1", word_recognition_scale=200), [.05])

    def test_variable_mapping_best_attempt_per_participant(self):
        rows = [{"condition_id": "C1", "participant_id": pid, "metrics": {"score": score}}
                for pid, score in [(1, 2), (1, 8), (1, 4), (2, 3), (2, 6)]]
        model = self.model("variable_mapping", rows)
        np.testing.assert_allclose(s.llm_scores("variable_mapping", model, "C1"), [.8, .6])
        root = self.human("variable_mapping", [
            {"status": "completed", "participant_id": "a", "summary": {"bestScore": score},
             "payload": {"questions": [{"correct": True}]}} for score in [2, 8, 4]])
        np.testing.assert_allclose(s.human_scores("variable_mapping", root), [.8])

    def test_craft_aggregates_three_trials_not_individual_rows(self):
        rows = [{"condition_id": "C2", "repeat_index": pid, "task_id": f"C200{i}",
                 "metrics": {"correct": score, "total": 5}}
                for pid, scores in [(1, [5, 4, 3]), (2, [2, 3, 4])]
                for i, score in enumerate(scores, 1)]
        model = self.model("craft_task", rows, compactor=True)
        np.testing.assert_allclose(s.llm_scores("craft_task", model, "compactor"), [.8, .6])

    def test_missing_human_recall_is_not_zero_or_older_replacement(self):
        root = self.human("semantic_story_recall", [
            {"status": "completed", "participant_id": "a", "completed_at": "2026-01-01", "summary": {"embeddingSimilarity": .8}},
            {"status": "completed", "participant_id": "a", "completed_at": "2026-02-01", "summary": {"embeddingSimilarity": None}},
            {"status": "completed", "participant_id": "b", "payload": {"evaluation": {"embeddingSimilarity": .6}}},
        ])
        with self.assertWarns(s.ScoringWarning):
            np.testing.assert_allclose(s.human_scores("semantic_story_recall", root), [.6])

    def test_cli_reports_are_machine_readable_and_discover_parent(self):
        model = self.model("word_recognition", [{"condition_id": "C1", "repeat_index": 1, "per_trial": [{}] * 10}])
        root = self.human("word_recognition", [{"status": "completed", "participant_id": "a", "summary": {"trialsCompleted": 10}}])
        entries = discover_models(explicit=[model.parent, model / "tasks"])
        self.assertEqual(len(entries), 1)
        report = score_models(entries, ["word_recognition"], ["C1"], human_root=root, human_selection="latest")
        self.assertEqual(report["rows"][0]["humanlikeness"], 1.)
        paths = write_reports(report, self.root / "out/scores.txt")
        self.assertTrue(all(p.is_file() for p in paths))
        self.assertEqual(json.loads(paths[2].read_text())["rows"][0]["model_n"], 1)

    def test_default_discovery_does_not_treat_prompting_pilot_as_compactor(self):
        model = self.model("craft_task", [])
        self.model("craft_task", [], compactor=True)
        entries = discover_models(prompting_root=model.parent, compactor_root=self.root / "absent")
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0][2])
        self.assertEqual(len(discover_models(explicit=[model])), 2)


class ReleasedDataTests(unittest.TestCase):
    def test_expanded_cohort_counts_and_means(self):
        expected = dict(zip(s.TASKS, [6.89, 5.13, .8771428571428571, 35.15, 6.71, 6.95, 7.76,
                                      .6062, 11.1, 12.53]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", s.ScoringWarning)
            for task in s.TASKS:
                with self.subTest(task=task):
                    values = s.human_scores(task)
                    self.assertEqual(len(values), 99 if task == "semantic_story_recall" else 100)
                    self.assertAlmostEqual(float(np.mean(values)) * s.TASK_DENOM[task], expected[task],
                                           places=3 if task == "semantic_story_recall" else 10)

    def test_craft_current_filter_is_explicit(self):
        self.assertEqual(len(s.human_scores("craft_task", craft_version="current")), 98)

    def test_imported_files_match_manifest_and_no_private_fields(self):
        import hashlib
        from scripts.import_paper_data import PRIVATE_KEYS
        manifest = json.loads((s.ROOT / "runs/data_update_2026-07-27.json").read_text())
        def check(value):
            if isinstance(value, dict):
                self.assertFalse({k.lower() for k in value} & PRIVATE_KEYS)
                for k, v in value.items():
                    if k == "participant_id":
                        self.assertRegex(v, r"^participant-\d{4}$")
                    check(v)
            elif isinstance(value, list):
                for v in value:
                    check(v)
        for item in manifest["files"]:
            path = s.ROOT / item["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            if "human-2026-07-27" in item["path"]:
                check(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
