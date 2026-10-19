"""Offline score extraction shared by the public CLI and downstream experiments.

Aggregation follows LLM_memory_bench/scripts/compare_wasserstein_llm_human.py.
The expanded human cohort follows analysis/reproduce_paper_figures.py. No model
SDK, network access, embedding model, or plotting dependency is required.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    "digit_span_forward", "digit_span_reverse", "nback", "word_recognition",
    "variable_mapping", "factual_qa", "narrative_qa", "semantic_story_recall",
    "map_task", "craft_task",
]
TASK_DISPLAY = dict(zip(TASKS, [
    "Digit Span", "Reverse Digit Span", "N-Back", "Word Recognition",
    "Variable Mapping", "Factual QA", "Narrative QA", "Narrative Free Recall",
    "Map Task", "Craft Task",
]))
HUMAN_TASK_DIR = dict(zip(TASKS, [
    "working-memory-digit-span", "working-memory-reverse-digit-span",
    "working-memory-nback", "working-memory-word-recognition",
    "working-memory-variable-mapping", "factual-qa", "narrative-qa",
    "semantic-memory-story-recall", "procedure-memory-map-task",
    "procedure-memory-craft-task",
]))
# Figure 3 uses 100 for recognition; some other paper analyses use 200.
# This is a normalization scale, not a guarantee that every score is <= 1.
TASK_DENOM = dict(zip(TASKS, [20., 20., 1., 100., 10., 10., 10., 1., 15., 15.]))
PROMPT_CONDITIONS = ["C1", "C2", "C3"]
CONDITION_DISPLAY = {"C1": "TaskPr", "C2": "HumPr", "C3": "MemPr", "compactor": "Compactor"}
HUMAN_ROOTS = {
    "expanded": ROOT / "runs/human-2026-07-27",
    "original": ROOT / "runs/human",
}
BEST_FIELDS = {
    "variable_mapping": "bestScore", "factual_qa": "correctAnswers",
    "narrative_qa": "correctAnswers", "map_task": "totalCorrect", "craft_task": "totalCorrect",
}


class ScoringWarning(UserWarning):
    """A saved run has a limitation that affects interpretation of its scores."""


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _normalizer(task, word_recognition_scale):
    if word_recognition_scale <= 0:
        raise ValueError("word_recognition_scale must be positive")
    return float(word_recognition_scale) if task == "word_recognition" else TASK_DENOM[task]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected an object")
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                rows.append(row)
    return rows


def _human_value(task: str, record: dict):
    summary = record.get("summary") or {}
    if task in ("digit_span_forward", "digit_span_reverse"):
        return _number(summary.get("bestSpan"))
    if task == "nback":
        correct, total = _number(summary.get("correctResponses")), _number(summary.get("totalResponses"))
        if correct is not None and total is not None and total > 0:
            return correct / total
        pct = _number(summary.get("accuracyPercent"))
        return pct / 100 if pct is not None else None
    if task == "word_recognition":
        return _number(summary.get("trialsCompleted"))
    if task == "semantic_story_recall":
        value = ((record.get("payload") or {}).get("evaluation") or {}).get("embeddingSimilarity")
        return _number(value if value is not None else summary.get("embeddingSimilarity"))
    return _number(summary.get(BEST_FIELDS[task]))


def _completion_time(record, path):
    if record.get("completion_order") is not None:
        return float(record["completion_order"]), path.name
    for field in ("completed_at", "updated_at", "created_at", "started_at"):
        value = record.get(field)
        if value:
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).timestamp(), path.name
            except ValueError:
                pass
    return 0., path.name


def _current_craft(record):
    items = json.loads((ROOT / "data/craft_task.json").read_text())["items"]
    expected = {tuple(q["prompt"] for q in item["questions"]) for item in items}
    trials = (record.get("payload") or {}).get("trials") or []
    observed = [tuple(q.get("prompt") for q in t.get("responses", [])) for t in trials]
    return len(observed) == len(expected) and set(observed) == expected


def human_scores(task: str, human_root: Path | None = None, *, selection="latest",
                 word_recognition_scale=100., craft_version="all") -> np.ndarray:
    """Normalized human samples; missing scores are excluded, never filled with zero.

    ``latest`` selects the last completed session per participant for single-session
    tasks; ``paper`` retains all completed sessions for those tasks. Both retain the
    paper's best-over-runs rule for variable mapping, QA, map, and craft.
    """
    if selection not in ("latest", "paper") or craft_version not in ("all", "current"):
        raise ValueError("Unknown human selection or craft version")
    root = Path(human_root) if human_root is not None else HUMAN_ROOTS["expanded"]
    folder = root / HUMAN_TASK_DIR[task]
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing human task directory: {folder}")
    by_pid = defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        if task == "craft_task" and craft_version == "current" and not _current_craft(record):
            continue
        # A missing participant ID must not merge unrelated sessions.
        pid = record.get("participant_id") or path.name
        by_pid[str(pid)].append((path, record))
    selected = []
    for group in by_pid.values():
        if task in BEST_FIELDS:
            valid = [(p, r) for p, r in group if _human_value(task, r) is not None]
            if valid:
                selected.append(max(valid, key=lambda pr: _human_value(task, pr[1])))
        elif selection == "latest":
            selected.append(max(group, key=lambda pr: _completion_time(pr[1], pr[0])))
        else:
            selected.extend(group)
    if task == "craft_task" and craft_version == "all":
        unmatched = sum(not _current_craft(r) for _, r in selected)
        if unmatched:
            warnings.warn(f"Craft: {unmatched}/{len(selected)} selected human sessions use other task "
                          "versions. Use --craft-version current for matched current stimuli.", ScoringWarning)
    values = [_human_value(task, r) for _, r in selected]
    missing = sum(v is None for v in values)
    if missing:
        warnings.warn(f"{task}: excluded {missing} human sessions with missing scores.", ScoringWarning)
    return np.asarray([v for v in values if v is not None], dtype=float) / _normalizer(task, word_recognition_scale)


def participant_key(row):
    for field in ("participant_id", "repeat_index"):
        if row.get(field) is not None:
            return str(row[field])
    match = re.search(r":p(\d+):", str(row.get("id") or ""))
    return match[1] if match else None


def digit_span_scores(rows: list[dict], *, min_span=2, max_span=20, sequences_per_span=2) -> list[float]:
    """Raw best spans, with two attempts and stopping at the first double failure.

    Historical files with one pid and many sequences use the paper's first K
    sequences per length. We report that limitation rather than inventing pids.
    """
    if min_span < 1 or max_span < min_span or sequences_per_span < 1:
        raise ValueError("Invalid digit-span bounds or attempts per span")
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[str(row.get("participant_id", 0))][int(row["span_length"])].append(row)
    extra = any(len(group) > sequences_per_span for spans in groups.values() for group in spans.values())
    if extra:
        warnings.warn(f"Digit span: more than {sequences_per_span} sequences per length; using the first "
                      f"{sequences_per_span} per participant as in the paper analysis. "
                      f"This produces {len(groups)} participant score(s), not one per sequence.", ScoringWarning)
    values = []
    for spans in groups.values():
        best = 0
        for span in range(min_span, max_span + 1):
            attempts = sorted(spans.get(span, []), key=lambda r: int(r.get("sequence_index", 0)))[:sequences_per_span]
            if not attempts:
                break
            if any(float((r.get("metrics") or {}).get("exact", 0)) >= 1 - 1e-9 for r in attempts):
                best = span
            else:
                break
        values.append(float(best))
    return values


def resolve_jsonl(task, model_dir, *, compactor):
    # Use the recorded original file, matching the paper analysis; do not silently
    # substitute the optional reparsed variable_mapping output.
    path = Path(model_dir) / "tasks" / f"{'wm_' if compactor else ''}{task}.jsonl"
    return path if path.is_file() else None


def llm_scores(task: str, model_dir: Path, condition: str, *, word_recognition_scale=100.,
               min_span=2, max_span=20, sequences_per_span=2) -> np.ndarray:
    """Normalized samples, aggregated at the same unit as the human reference."""
    if condition not in CONDITION_DISPLAY:
        raise ValueError(f"Unknown condition: {condition}")
    path = resolve_jsonl(task, model_dir, compactor=condition == "compactor")
    if path is None:
        return np.asarray([], dtype=float)
    cond_id = "C2" if condition == "compactor" else condition
    rows = [r for r in read_jsonl(path) if (r.get("condition_id") or r.get("condition")) == cond_id]
    values = []
    if task in ("digit_span_forward", "digit_span_reverse"):
        values = digit_span_scores(rows, min_span=min_span, max_span=max_span,
                                   sequences_per_span=sequences_per_span)
    elif task == "nback":
        groups = defaultdict(list)
        for r in rows:
            if r.get("participant_id") is None or r.get("repeat_index") is None:
                raise ValueError(f"{path}: n-back needs participant_id and repeat_index")
            value = r.get("acc_over_answered")
            if value is None:
                value = r.get("acc_over_14")
            value = _number(value)
            if value is not None:
                groups[(str(r["participant_id"]), str(r["repeat_index"]))].append(value)
        values = [float(np.mean(v)) for v in groups.values()]
    elif task == "word_recognition":
        for r in rows:
            if "per_trial" not in r:
                raise ValueError(f"{path}: word recognition requires per_trial to count trials reached")
            if r.get("repeat_index") is not None:
                values.append(float(len(r["per_trial"] or [])))
    elif task == "variable_mapping":
        groups = defaultdict(list)
        for r in rows:
            value = _number((r.get("metrics") or {}).get("score"))
            if value is not None:
                groups[participant_key(r) or "0"].append(value)
        values = [max(v) for v in groups.values()]
    elif task == "semantic_story_recall":
        # Preserve the paper's model-side missing-value rule explicitly.
        vals = [_number((r.get("metrics") or {}).get("embeddingSimilarity")) for r in rows]
        if any(v is None for v in vals):
            warnings.warn(f"{path}: missing model recall similarities scored as zero, matching the paper.", ScoringWarning)
        values = [v if v is not None else 0. for v in vals]
    else:
        groups = defaultdict(list)
        for r in rows:
            value = _number((r.get("metrics") or {}).get("correct"))
            if value is not None:
                groups[participant_key(r) or "0"].append(value)
        values = [sum(v) for v in groups.values()]
    result = np.asarray(values, dtype=float) / _normalizer(task, word_recognition_scale)
    if result.size and np.any(result > 1 + 1e-9):
        warnings.warn(f"{path}: scores exceed the chosen normalization scale; check trial counts "
                      "and --word-recognition-scale. Scores are not clipped.", ScoringWarning)
    return result


def wasserstein_1d(x, y) -> float:
    """Exact empirical 1-Wasserstein distance, including unequal sample sizes."""
    x, y = np.sort(np.asarray(x, dtype=float).ravel()), np.sort(np.asarray(y, dtype=float).ravel())
    if not x.size or not y.size:
        return float("nan")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Wasserstein inputs must be finite")
    grid = np.unique(np.concatenate([x, y]))
    if grid.size < 2:
        return 0.
    gaps = np.abs(np.searchsorted(x, grid[:-1], side="right") / x.size
                  - np.searchsorted(y, grid[:-1], side="right") / y.size)
    return float(np.sum(gaps * np.diff(grid)))


def humanlikeness(human, model) -> float:
    return 1. - wasserstein_1d(human, model)
