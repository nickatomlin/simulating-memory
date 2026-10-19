import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import run


class WrapperTests(unittest.TestCase):
    def invoke(self, *args):
        stdout = io.StringIO()
        with patch("sys.argv", ["run.py", "--model", "qwen/qwen3-8b", "--out-dir", "runs/test", *args]), contextlib.redirect_stdout(stdout):
            run.main()
        return stdout.getvalue()

    def test_dry_run_forwards_repeat_and_selected_compactor(self):
        with patch.object(run.subprocess, "call") as call, patch.object(run, "_write_yaml") as write:
            text = self.invoke("--tasks", "digit_span_forward,craft_task", "--repeat", "7", "--include-compactor", "--dry-run")
        self.assertIn("--repeat 7", text)
        self.assertIn("--task wm_digit_span_forward", text)
        self.assertIn("--task wm_craft_task", text)
        self.assertNotIn("--task wm_word_recognition", text)
        call.assert_not_called()
        write.assert_not_called()

    def test_compactor_only_and_thinking_output_path(self):
        text = self.invoke("--tasks", "craft_task", "--compactor-only", "--qwen-thinking", "false", "--dry-run")
        command = text.split("Command:\n")[1].splitlines()[0]
        self.assertIn("--task wm_craft_task", command)
        self.assertNotIn("--task craft_task", command)
        self.assertIn("qwen_qwen3-8b_false", text)

    def test_existing_run_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "test/tasks").mkdir(parents=True)
            with patch("sys.argv", ["run.py", "--model", "test", "--out-dir", tmp]), contextlib.redirect_stderr(io.StringIO()), patch.object(run.subprocess, "call") as call:
                with self.assertRaises(SystemExit) as error:
                    run.main()
            self.assertEqual(error.exception.code, 2)
            call.assert_not_called()

    def test_invalid_repeat_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.invoke("--repeat", "0", "--dry-run")


class BenchmarkRoutingTests(unittest.TestCase):
    def test_repeat_means_participants_and_two_attempts_for_all_digit_agents(self):
        os.environ.setdefault("MPLCONFIGDIR", str(run.ROOT / ".mplcache"))
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            from bench import cli
            from typer.testing import CliRunner
            import yaml
        except ImportError as exc:
            self.skipTest(f"Install .[experiments] for runner routing checks: {exc}")
        names = ["digit_span_forward", "digit_span_reverse", "wm_digit_span_forward",
                 "wm_digit_span_reverse", "sum_digit_span_forward", "sum_digit_span_reverse"]
        with tempfile.TemporaryDirectory() as tmp:
            config = run.build_config(model="offline-test", backend="openai", base_url=None,
                                      out_dir=Path(tmp) / "results", repeat=7, temperature=0., max_tokens=10,
                                      max_parallel_participants=1, max_parallel_tasks=1)
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            stubs = {name: Mock(return_value={"task": name, "conditions": []}) for name in names}
            client = Mock(request_count=0)
            # The actual CLI dispatch and local runner execute; only model creation
            # and task implementations are replaced. No network path is reachable.
            with patch.object(cli, "OpenAIChatLLM", return_value=client), patch.dict(cli.TASKS, stubs):
                args = ["run", "--config", str(path)]
                for name in names:
                    args += ["--task", name]
                result = CliRunner().invoke(cli.app, args)
            self.assertEqual(result.exit_code, 0, result.output)
            for name, stub in stubs.items():
                with self.subTest(task=name):
                    stub.assert_called_once()
                    self.assertEqual(stub.call_args.kwargs["n_participants"], 7)
                    self.assertEqual(stub.call_args.kwargs["sequences_per_span"], 2)
                    self.assertEqual(stub.call_args.kwargs["max_parallel_participants"], 1)
            client.generate.assert_not_called()
            client.generate_with_tools.assert_not_called()


if __name__ == "__main__":
    unittest.main()
