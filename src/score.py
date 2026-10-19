"""Score saved runs against human reference data without making model API calls.

Examples:
    python src/score.py
    python src/score.py --model-dir runs/my-experiment --tasks digit_span_forward craft_task
    python src/score.py --human-cohort original --conditions C1 --out results/original.txt

Writes a text table, a CSV, and a JSON report with settings, sample counts and
warnings. See bench/core/scoring.py for reusable score-extraction functions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from bench.core.scoring import (
    CONDITION_DISPLAY, HUMAN_ROOTS, PROMPT_CONDITIONS, TASK_DENOM, TASK_DISPLAY,
    TASKS, ScoringWarning, human_scores, humanlikeness, llm_scores, wasserstein_1d,
)

PROMPTING_ROOT = ROOT / "runs/prompting"
COMPACTOR_ROOT = ROOT / "runs/compactor"


def discover_models(*, prompting_root=PROMPTING_ROOT, compactor_root=COMPACTOR_ROOT,
                    explicit=None):
    """Accept a model folder, its tasks folder, or a parent containing model folders."""
    roots = explicit or [prompting_root, compactor_root]
    found = []
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            if explicit:
                raise ValueError(f"Model directory does not exist: {root}")
            continue
        if root.name == "tasks":
            folders = [root]
        elif (root / "tasks").is_dir():
            folders = [root / "tasks"]
        else:
            folders = sorted(root.rglob("tasks"))
        for folder in folders:
            model_dir = folder.parent
            for compactor in (False, True):
                # The published prompting folder contains an auxiliary WM pilot;
                # default scoring must not count that as another Compactor run.
                if not explicit and compactor != (root == Path(compactor_root).resolve()):
                    continue
                key = (model_dir, compactor)
                prefix = "wm_" if compactor else ""
                if key not in seen and any((folder / f"{prefix}{t}.jsonl").is_file() for t in TASKS):
                    seen.add(key)
                    found.append((model_dir.name, model_dir, compactor))
    return found


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def score_models(entries, tasks, conditions, *, human_root, human_selection,
                 word_recognition_scale=100., craft_version="all", min_span=2,
                 max_span=20, sequences_per_span=2, include_samples=False):
    """Build a JSON-serializable report; all computation is offline."""
    options = dict(word_recognition_scale=word_recognition_scale, min_span=min_span,
                   max_span=max_span, sequences_per_span=sequences_per_span)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ScoringWarning)
        humans = {t: human_scores(t, human_root, selection=human_selection,
                                 word_recognition_scale=word_recognition_scale,
                                 craft_version=craft_version) for t in tasks}
        rows = []
        for name, model_dir, compactor in entries:
            selected = ["compactor"] if compactor else PROMPT_CONDITIONS
            for condition in selected:
                if condition not in conditions:
                    continue
                for task in tasks:
                    model = llm_scores(task, model_dir, condition, **options)
                    human = humans[task]
                    valid = bool(model.size and human.size)
                    row = dict(model=name, model_dir=str(model_dir), task=task,
                               condition=condition, condition_name=CONDITION_DISPLAY[condition],
                               human_n=int(human.size), model_n=int(model.size),
                               human_mean=_mean(human), mean_score=_mean(model),
                               wasserstein=wasserstein_1d(human, model) if valid else None,
                               humanlikeness=humanlikeness(human, model) if valid else None,
                               status="ok" if valid else "missing_model" if not model.size else "missing_human")
                    if include_samples:
                        row["human_scores"] = human.tolist()
                        row["model_scores"] = model.tolist()
                    rows.append(row)
    return {
        "schema_version": 1,
        "scoring": {
            "aggregation": "paper_repository", "distance": "exact_empirical_wasserstein_1",
            "human_root": str(Path(human_root).resolve()), "human_selection": human_selection,
            "craft_version": craft_version, "tasks": tasks, "conditions": conditions,
            "normalizers": {t: word_recognition_scale if t == "word_recognition" else TASK_DENOM[t] for t in tasks},
            "digit_span": {"min_span": min_span, "max_span": max_span,
                           "sequences_per_span": sequences_per_span, "legacy_layout": "first_sequences_per_participant"},
            "word_recognition": "trials_reached_including_errors",
            "missing_human_scores": "excluded", "missing_model_recall_similarity": "zero_as_in_paper",
        },
        "warnings": list(dict.fromkeys(str(w.message) for w in caught)),
        "rows": rows,
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.6f}"


def build_table(report):
    lines = ["Memory benchmark scores", "", "Mean score is normalized by the recorded task scale.",
             "Humanlikeness = 1 - exact Wasserstein distance between normalized distributions.",
             f"Human reference: {report['scoring']['human_root']} ({report['scoring']['human_selection']})", "",
             f"{'model':<38} {'condition':<10} {'task':<23} {'human_n':>7} {'model_n':>7} {'mean':>10} {'similarity':>10}"]
    for r in report["rows"]:
        lines.append(f"{r['model']:<38} {r['condition_name']:<10} {r['task']:<23} "
                     f"{r['human_n']:>7} {r['model_n']:>7} {_fmt(r['mean_score']):>10} {_fmt(r['humanlikeness']):>10}")
    if report["warnings"]:
        lines += ["", "Scoring notes:"] + [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


def write_reports(report, out):
    out = Path(out)
    text_path = out if out.suffix in (".txt", ".md") else out.with_suffix(".txt")
    paths = [text_path, out.with_suffix(".csv"), out.with_suffix(".json")]
    out.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(build_table(report), encoding="utf-8")
    fields = ["model", "model_dir", "task", "condition", "condition_name", "human_n", "model_n",
              "human_mean", "mean_score", "wasserstein", "humanlikeness", "status"]
    with paths[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    paths[2].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths


def _choices(values, allowed, parser, label):
    selected = list(dict.fromkeys(v.strip() for value in values for v in value.split(",") if v.strip()))
    unknown = set(selected) - set(allowed)
    if not selected or unknown:
        parser.error(f"Invalid {label}: {', '.join(sorted(unknown)) or '(empty)'}. Choose from {', '.join(allowed)}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, action="append", help="Model folder or experiment parent; repeatable.")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="Task names, separated by spaces or commas.")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITION_DISPLAY), help="C1 C2 C3 compactor; spaces or commas.")
    parser.add_argument("--human-cohort", choices=list(HUMAN_ROOTS), default="expanded")
    parser.add_argument("--human-root", type=Path, help="Override the selected human data folder.")
    parser.add_argument("--human-selection", choices=["latest", "paper"], help="Defaults to latest for expanded, paper for original.")
    parser.add_argument("--word-recognition-scale", type=float, choices=[100., 200.], default=100.,
                        help="100 matches Figure 3; some other paper analyses use 200.")
    parser.add_argument("--craft-version", choices=["all", "current"], default="all",
                        help="all matches paper cohorts; current excludes human sessions using old crafting systems.")
    parser.add_argument("--digit-span-min", type=int, default=2)
    parser.add_argument("--digit-span-max", type=int, default=20)
    parser.add_argument("--digit-span-sequences-per-span", type=int, default=2)
    parser.add_argument("--include-samples", action="store_true", help="Include score arrays in JSON.")
    parser.add_argument("--out", type=Path, default=ROOT / "results/scores.txt", help="Output name; writes .txt, .csv and .json.")
    args = parser.parse_args()
    tasks = _choices(args.tasks, TASKS, parser, "tasks")
    conditions = _choices(args.conditions, CONDITION_DISPLAY, parser, "conditions")
    if args.digit_span_min < 1 or args.digit_span_max < args.digit_span_min or args.digit_span_sequences_per_span < 1:
        parser.error("Invalid digit-span bounds or sequences per span")
    try:
        entries = discover_models(explicit=args.model_dir)
        if not entries:
            parser.error("No model task files found. Pass --model-dir <model-or-experiment-folder>.")
        report = score_models(entries, tasks, conditions,
                              human_root=args.human_root or HUMAN_ROOTS[args.human_cohort],
                              human_selection=args.human_selection or ("latest" if args.human_cohort == "expanded" else "paper"),
                              word_recognition_scale=args.word_recognition_scale, craft_version=args.craft_version,
                              min_span=args.digit_span_min, max_span=args.digit_span_max,
                              sequences_per_span=args.digit_span_sequences_per_span, include_samples=args.include_samples)
        if not any(r["status"] == "ok" for r in report["rows"]):
            parser.error("No scoreable model/human pairs for the selected tasks and conditions.")
        paths = write_reports(report, args.out)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    n = sum(r["status"] == "ok" for r in report["rows"])
    print(f"Scored {n} model/task/condition comparisons offline.")
    for path in paths:
        print(f"Wrote {path}")
    if report["warnings"]:
        print(f"{len(report['warnings'])} scoring notes are recorded in the text and JSON reports.")


if __name__ == "__main__":
    main()
