"""Run a user-supplied OpenAI-compatible chat model on the 10 memory tasks
and write per-participant outputs in the same JSONL format that ``score.py``
consumes.

This is a thin convenience wrapper over ``bench.cli``. We assemble a YAML
config from the same defaults that produced the released ``runs/`` data,
then hand off to ``python -m bench.cli run``. The result lands at
``<out_dir>/<model_slug>/`` and can be scored with:

    python src/score.py --model-dir <out_dir>/<model_slug>

Examples
--------

OpenAI directly:

    OPENAI_API_KEY=sk-... python src/run.py \\
        --model gpt-4o \\
        --out-dir runs/my-run

OpenRouter (covers Anthropic, Llama, Qwen, ...):

    OPENROUTER_API_KEY=sk-or-... python src/run.py \\
        --model meta-llama/llama-3-8b-instruct \\
        --base-url https://openrouter.ai/api/v1 \\
        --out-dir runs/my-run

Subset of tasks, smaller participant count, include the compactor agent:

    python src/run.py --model gpt-4o-mini --out-dir runs/quick \\
        --tasks digit_span_forward,word_recognition \\
        --repeat 5 --include-compactor

Notes
-----
- ``--repeat`` overrides the number of simulated participants per task
  (the released runs used 50 per prompting task; we default to that).
- ``--include-compactor`` additionally runs the ``wm_<task>`` agent
  variants, which is the 4-slot key-value memory module ("Compactor"
  in the paper).
- ``--qwen-thinking {true,false}`` is forwarded to the underlying CLI.
- Costs scale with --repeat and number of tasks; start small.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Tasks that mirror what the released runs/ folder contains
PROMPTING_TASKS = [
    "digit_span_forward",
    "digit_span_reverse",
    "nback",
    "word_recognition",
    "variable_mapping",
    "factual_qa",
    "narrative_qa",
    "semantic_story_recall",
    "map_task",
    "craft_task",
]

COMPACTOR_TASKS = [f"wm_{t}" for t in PROMPTING_TASKS]


def _default_task_config() -> dict:
    """Defaults that match the released runs/ data (stimuli_seed=42, etc.)."""
    return {
        "digit_span_forward": {
            "min_span": 2, "max_span": 20, "sequences_per_span": 2,
            "stimuli_seed": 42, "prompt_mode": "zero_shot",
        },
        "digit_span_reverse": {
            "min_span": 2, "max_span": 20, "sequences_per_span": 2,
            "stimuli_seed": 42, "prompt_mode": "zero_shot",
        },
        "nback": {
            "n_repeat": 5, "n_repeats_per_participant": 1,
            "stimuli_seed": 123, "temperature": 0,
        },
        "word_recognition": {
            "words_json_path": "data/words.json",
            "n_repeat": 5, "max_trials_per_game": 100, "stimuli_seed": 42,
        },
        "variable_mapping": {
            "names_json_path": "data/names.json",
            "city_json_path": "data/city.json",
            "n_repeat": 5, "n_runs_per_participant": 3, "stimuli_seed": 42,
        },
        "factual_qa": {
            "data_json_path": "data/wikipedia_10docs_questions.json",
            "n_participants": 5, "n_docs": None, "stimuli_seed": 42,
        },
        "narrative_qa": {
            "data_json_path": "data/narrative_QA.json",
            "n_participants": 5, "n_stories": None, "stimuli_seed": 42,
        },
        "semantic_story_recall": {
            "n_participants": 5, "stimuli_seed": 42,
            "use_judge": True, "recall_max_tokens": 16384,
        },
        "map_task": {
            "data_json_path": "data/maps.json",
            "n_participants": 5, "n_maps": None, "stimuli_seed": 42,
        },
        "craft_task": {
            "data_json_path": "data/craft_task.json",
            "n_participants": 5, "n_tasks": None, "stimuli_seed": 42,
        },
        # WM (Compactor) variants
        "wm_digit_span_forward": {
            "min_span": 2, "max_span": 20, "sequences_per_span": 2,
            "stimuli_seed": 42, "temperature": 0.0,
        },
        "wm_digit_span_reverse": {
            "min_span": 2, "max_span": 20, "sequences_per_span": 2,
            "stimuli_seed": 42, "temperature": 0.0,
        },
        "wm_nback": {
            "n_repeat": 5, "n_repeats_per_participant": 1,
            "stimuli_seed": 123, "temperature": 0.0,
        },
        "wm_word_recognition": {
            "words_json_path": "data/words.json",
            "n_repeat": 5, "max_trials_per_game": 100,
            "stimuli_seed": 42, "temperature": 0.0,
        },
        "wm_variable_mapping": {
            "names_json_path": "data/names.json",
            "city_json_path": "data/city.json",
            "n_repeat": 5, "stimuli_seed": 42, "temperature": 0.0,
        },
        "wm_factual_qa": {
            "data_json_path": "data/wikipedia_10docs_questions.json",
            "n_participants": 5, "n_docs": None,
            "stimuli_seed": 42, "temperature": 0.0,
        },
        "wm_narrative_qa": {
            "data_json_path": "data/narrative_QA.json",
            "n_participants": 5, "n_stories": None,
            "stimuli_seed": 42, "temperature": 0.0,
        },
        "wm_semantic_story_recall": {
            "repeats_per_stimulus": 5, "temperature": 0.0,
        },
        "wm_map_task": {
            "data_json_path": "data/maps.json",
            "n_participants": 5, "n_maps": None,
            "stimuli_seed": 42, "temperature": 0.0,
        },
        "wm_craft_task": {
            "data_json_path": "data/craft_task.json",
            "n_participants": 5, "n_tasks": None,
            "stimuli_seed": 42, "temperature": 0.0,
        },
    }


def _write_yaml(cfg: dict) -> Path:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        print(
            "PyYAML is not installed but is required for src/run.py "
            "(it writes a YAML config consumed by bench.cli). "
            "Install with: pip install pyyaml",
            file=sys.stderr,
        )
        raise SystemExit(2)
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


def build_config(*,
                 model: str,
                 backend: str,
                 base_url: str | None,
                 out_dir: Path,
                 repeat: int,
                 temperature: float,
                 max_tokens: int) -> dict:
    return {
        "model": {
            "backend": backend,
            "name": model,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1.0,
        },
        "run": {
            "out_dir": str(out_dir),
            "seed": 42,
            "max_parallel_participants": 10,
            "max_parallel_tasks": 5,
            "repeat": int(repeat),
        },
        "task_config": _default_task_config(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Model id (e.g. gpt-4o, anthropic/claude-opus-4-1, "
                             "meta-llama/llama-3-8b-instruct).")
    parser.add_argument("--backend", default="openai", choices=["openai", "anthropic"],
                        help="Chat-completions backend (default: openai-compatible). "
                             "Use 'anthropic' for direct Anthropic API.")
    parser.add_argument("--base-url", default=None,
                        help="Override base URL (e.g. https://openrouter.ai/api/v1).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Parent output directory; results land under "
                             "<out_dir>/<model_slug>/.")
    parser.add_argument("--tasks", default=None,
                        help="Comma-separated subset of tasks. Default: all 10 "
                             "prompting tasks (plus all wm_ ones if --include-compactor).")
    parser.add_argument("--include-compactor", action="store_true",
                        help="Also run the wm_<task> variants (4-slot key-value "
                             "memory agent, 'Compactor' in the paper).")
    parser.add_argument("--repeat", type=int, default=50,
                        help="Override n_participants per task (default: 50).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--qwen-thinking", choices=["true", "false"], default=None,
        help="Forwarded to bench.cli: toggles OpenRouter Qwen3-8B "
             "extra_body.enable_thinking.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the assembled config and the bench.cli command, then exit.",
    )
    args = parser.parse_args()

    if args.tasks:
        chosen = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        chosen = list(PROMPTING_TASKS)
        if args.include_compactor:
            chosen += list(COMPACTOR_TASKS)

    unknown = [t for t in chosen if t not in PROMPTING_TASKS + COMPACTOR_TASKS]
    if unknown:
        raise SystemExit(f"Unknown task name(s): {unknown}")

    cfg = build_config(
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        out_dir=args.out_dir,
        repeat=args.repeat,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    yaml_path = _write_yaml(cfg)
    cmd = [
        sys.executable, "-m", "bench.cli", "run",
        "--config", str(yaml_path),
    ]
    for t in chosen:
        cmd += ["--task", t]
    if args.qwen_thinking is not None:
        cmd += ["--qwen-thinking", args.qwen_thinking]

    if args.dry_run:
        print("Config:")
        print(json.dumps(cfg, indent=2))
        print()
        print("Command:")
        print(" ".join(cmd))
        print()
        print(f"YAML written to: {yaml_path}")
        return

    env = os.environ.copy()
    # Make sure the repo root is on PYTHONPATH so 'bench' is importable
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Launching: {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
    if rc != 0:
        raise SystemExit(rc)

    print()
    print("Done. Score with:")
    # bench appends a model slug under out_dir; mirror that derivation here.
    slug = args.model.replace("/", "_").replace("\\", "_")
    final_dir = args.out_dir / slug
    print(f"  python src/score.py --model-dir {final_dir}")


if __name__ == "__main__":
    main()
