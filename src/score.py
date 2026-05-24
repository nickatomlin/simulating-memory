"""Score every model under ``runs/`` against the human data and write a .txt
summary of per-task accuracy + humanlikeness per prompting condition.

Usage
-----

    # Score everything under runs/prompting + runs/compactor (default):
    python src/score.py

    # Score one specific model directory (e.g. a new run produced by run.py):
    python src/score.py --model-dir runs/my-model

    # Override output path:
    python src/score.py --out tables/my-table.txt

Layout expected per model directory (matches what ``bench`` produces and what
``run.py`` writes):

    <model_dir>/
        tasks/
            digit_span_forward.jsonl          # TaskPr / HumPr / MemPr rows
            digit_span_reverse.jsonl
            nback.jsonl
            word_recognition.jsonl
            variable_mapping.jsonl
            factual_qa.jsonl
            narrative_qa.jsonl
            semantic_story_recall.jsonl
            map_task.jsonl
            craft_task.jsonl
            wm_digit_span_forward.jsonl       # Compactor rows (optional)
            wm_digit_span_reverse.jsonl
            ...

Rows can carry the prompting conditions ``C1`` (TaskPr), ``C2`` (HumPr),
``C3`` (MemPr); compactor rows under ``wm_<task>.jsonl`` use ``C2``.

The output table has one row per (task, condition) and columns
``mean_score`` (per-participant normalized score averaged over the model)
and ``humanlikeness`` (= 1 - W_1 between the model's score distribution
and the human distribution loaded from ``runs/human/``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
HUMAN_ROOT = RUNS / "human"
PROMPTING_ROOT = RUNS / "prompting"
COMPACTOR_ROOT = RUNS / "compactor"

TASKS = [
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

TASK_DISPLAY = {
    "digit_span_forward": "Digit Span",
    "digit_span_reverse": "Reverse Digit Span",
    "nback": "N-Back",
    "word_recognition": "Word Recognition",
    "variable_mapping": "Variable Mapping",
    "factual_qa": "Factual QA",
    "narrative_qa": "Narrative QA",
    "semantic_story_recall": "Narrative Free Recall",
    "map_task": "Map Task",
    "craft_task": "Craft Task",
}

HUMAN_TASK_DIR = {
    "digit_span_forward": "working-memory-digit-span",
    "digit_span_reverse": "working-memory-reverse-digit-span",
    "nback": "working-memory-nback",
    "word_recognition": "working-memory-word-recognition",
    "variable_mapping": "working-memory-variable-mapping",
    "factual_qa": "factual-qa",
    "narrative_qa": "narrative-qa",
    "semantic_story_recall": "semantic-memory-story-recall",
    "map_task": "procedure-memory-map-task",
    "craft_task": "procedure-memory-craft-task",
}

# Per-task denominator so that 1.0 = perfect.
TASK_DENOM = {
    "digit_span_forward": 20.0,
    "digit_span_reverse": 20.0,
    "nback": 1.0,
    "word_recognition": 100.0,
    "variable_mapping": 10.0,
    "factual_qa": 10.0,
    "narrative_qa": 10.0,
    "semantic_story_recall": 1.0,
    "map_task": 15.0,
    "craft_task": 15.0,
}

PROMPT_CONDITIONS = ["C1", "C2", "C3"]
CONDITION_DISPLAY = {
    "C1": "TaskPr",
    "C2": "HumPr",
    "C3": "MemPr",
    "compactor": "Compactor",
}


# ---------- Per-participant scoring ------------------------------------------


def _best_span_from_trials(trials: list[dict[str, Any]]) -> int:
    by_span: dict[int, list[bool]] = defaultdict(list)
    for t in trials:
        length = t.get("length")
        if length is None:
            continue
        by_span[int(length)].append(bool(t.get("correct", False)))
    best = 0
    for span in sorted(by_span.keys()):
        if any(by_span[span]):
            best = span
        else:
            break
    return best


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _score_human_record(task: str, rec: dict[str, Any]) -> float | None:
    summary = rec.get("summary") or {}
    payload = rec.get("payload") or {}
    denom = TASK_DENOM[task]

    if task in ("digit_span_forward", "digit_span_reverse"):
        trials = payload.get("trials") or []
        if trials:
            return _best_span_from_trials(trials) / denom
        best = summary.get("bestSpan")
        return None if best is None else float(best) / denom

    if task == "nback":
        trials = payload.get("trials") or []
        scored = [t for t in trials if t.get("phase") != "practice"]
        if scored:
            correct = sum(1 for t in scored if t.get("correct"))
            return correct / len(scored)
        pct = summary.get("accuracyPercent")
        return None if pct is None else float(pct) / 100.0

    if task == "word_recognition":
        correct = summary.get("correctResponses")
        return None if correct is None else float(correct) / denom

    if task == "variable_mapping":
        questions = payload.get("questions") or []
        if questions:
            correct = sum(1 for q in questions if q.get("correct"))
            return min(correct, int(denom)) / denom
        best = summary.get("bestScore")
        return None if best is None else float(best) / denom

    if task in ("factual_qa", "narrative_qa"):
        total = summary.get("totalQuestions") or 10
        correct = summary.get("correctAnswers")
        return None if correct is None else float(correct) / float(total)

    if task == "semantic_story_recall":
        # Paper text says BLEU but the web app's BLEU is ~0 across participants
        # (heavy smoothing); embeddingSimilarity is the metric that lines up
        # with the published Figure 2 visually.
        sim = summary.get("embeddingSimilarity")
        return None if sim is None else float(sim)

    if task in ("map_task", "craft_task"):
        total = summary.get("totalQuestions") or 0
        correct = summary.get("totalCorrect") or 0
        return None if total <= 0 else float(correct) / float(total)

    return None


def human_scores(task: str) -> np.ndarray:
    folder = HUMAN_ROOT / HUMAN_TASK_DIR[task]
    out: list[float] = []
    if not folder.is_dir():
        return np.asarray(out, dtype=np.float64)
    for path in sorted(folder.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                rec = json.load(f)
        except json.JSONDecodeError:
            continue
        s = _score_human_record(task, rec)
        if s is not None:
            out.append(float(s))
    return np.asarray(out, dtype=np.float64)


def _score_llm_row(task: str, row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics") or {}
    denom = TASK_DENOM[task]

    if task in ("digit_span_forward", "digit_span_reverse"):
        # Aggregated upstream via _digit_span_participant_scores.
        ex = metrics.get("exact")
        return None if ex is None else float(ex)

    if task == "nback":
        acc = metrics.get("accuracy")
        if acc is not None:
            return float(acc)
        acc = row.get("acc_over_14")
        if acc is None:
            acc = row.get("acc_over_answered")
        return None if acc is None else float(acc)

    if task == "word_recognition":
        score = metrics.get("score")
        return None if score is None else float(score) / denom

    if task == "variable_mapping":
        score = metrics.get("score")
        return None if score is None else float(score) / denom

    if task in ("factual_qa", "narrative_qa"):
        acc = metrics.get("accuracy")
        if acc is None:
            correct = metrics.get("correct")
            total = metrics.get("total")
            if correct is not None and total:
                acc = float(correct) / float(total)
        return None if acc is None else float(acc)

    if task == "semantic_story_recall":
        sim = metrics.get("embeddingSimilarity")
        return None if sim is None else float(sim)

    if task in ("map_task", "craft_task"):
        acc = metrics.get("accuracy")
        if acc is not None:
            return float(acc)
        correct = metrics.get("correct") or metrics.get("totalCorrect")
        total = metrics.get("total") or metrics.get("totalQuestions")
        if correct is not None and total:
            return float(correct) / float(total)
        return None

    return None


def _digit_span_participant_scores(rows: Iterable[dict[str, Any]], condition: str) -> list[float]:
    # Group rows by (participant_id, sequence_index_bucket). The prompting
    # protocol runs N participants x 2 sequences per span; the compactor
    # protocol typically runs 1 participant x M sequences per span. To keep
    # both protocols comparable we group on (pid, seq_idx) when a single
    # participant carries many sequences, so each seq becomes one observation.
    rows = list(rows)
    rows = [r for r in rows
            if (r.get("condition_id") or r.get("condition")) == condition]
    pid_set = {int(r.get("participant_id", 0)) for r in rows}
    seq_set = {int(r.get("sequence_index", 1)) for r in rows}
    # If there's only one participant but many sequences, split on sequence_index.
    split_by_seq = len(pid_set) <= 1 and len(seq_set) > 2

    buckets: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pid = int(r.get("participant_id", 0))
        seq = int(r.get("sequence_index", 1)) if split_by_seq else 0
        span = r.get("span_length")
        ex = (r.get("metrics") or {}).get("exact")
        if span is None or ex is None:
            continue
        buckets[(pid, seq)][int(span)].append(float(ex))
    out: list[float] = []
    for by_span in buckets.values():
        best = 0
        for s in sorted(by_span):
            if any(v == 1.0 for v in by_span[s]):
                best = s
            else:
                break
        out.append(best / 20.0)
    return out


def _resolve_jsonl(task: str, model_dir: Path, *, compactor: bool) -> Path | None:
    fname = f"wm_{task}.jsonl" if compactor else f"{task}.jsonl"
    p = model_dir / "tasks" / fname
    if not compactor and task == "variable_mapping":
        rp = model_dir / "tasks" / "variable_mapping.reparsed.jsonl"
        if rp.is_file():
            p = rp
    return p if p.is_file() else None


def llm_scores(task: str, model_dir: Path, condition: str) -> np.ndarray:
    """Per-participant scores from a model directory.

    ``condition`` is one of: ``C1``, ``C2``, ``C3`` (prompting) or
    ``compactor``. For the compactor case we read ``wm_<task>.jsonl``
    where the synthetic condition id is ``C2``.
    """
    compactor = condition == "compactor"
    path = _resolve_jsonl(task, model_dir, compactor=compactor)
    if path is None:
        return np.asarray([], dtype=np.float64)
    cond_id = "C2" if compactor else condition
    rows = _read_jsonl(path)
    if task in ("digit_span_forward", "digit_span_reverse"):
        return np.asarray(
            _digit_span_participant_scores(rows, cond_id), dtype=np.float64
        )
    out = []
    for r in rows:
        row_cond = r.get("condition_id") or r.get("condition")
        if row_cond != cond_id:
            continue
        s = _score_llm_row(task, r)
        if s is not None:
            out.append(float(s))
    return np.asarray(out, dtype=np.float64)


# ---------- Statistics -------------------------------------------------------


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x, y]))
    if grid.size == 1:
        return float(abs(np.mean(x) - np.mean(y)))
    edges = np.concatenate([[grid[0]], (grid[:-1] + grid[1:]) / 2.0, [grid[-1]]])
    total = 0.0
    for i in range(len(edges) - 1):
        mid = 0.5 * (edges[i] + edges[i + 1])
        fx = np.searchsorted(x, mid, side="right") / x.size
        fy = np.searchsorted(y, mid, side="right") / y.size
        total += abs(fx - fy) * (edges[i + 1] - edges[i])
    return float(total)


def humanlikeness(human: np.ndarray, model: np.ndarray) -> float:
    w = wasserstein_1d(human, model)
    return float("nan") if np.isnan(w) else 1.0 - w


# ---------- Model directory discovery ----------------------------------------


def discover_models(*, prompting_root: Path = PROMPTING_ROOT,
                    compactor_root: Path = COMPACTOR_ROOT,
                    explicit: list[Path] | None = None) -> list[tuple[str, Path, bool]]:
    """Return [(display_name, model_dir, is_compactor)] entries.

    If ``explicit`` is provided, treat each as a self-contained model dir
    that may carry both prompting jsonls and ``wm_*`` (compactor) jsonls.
    """
    found: list[tuple[str, Path, bool]] = []
    if explicit:
        for d in explicit:
            d = Path(d)
            if not d.is_dir():
                continue
            has_prompt = any((d / "tasks").glob("*.jsonl")) if (d / "tasks").is_dir() else False
            if has_prompt:
                found.append((d.name, d, False))
            has_compactor = (
                (d / "tasks").is_dir()
                and any((d / "tasks").glob("wm_*.jsonl"))
            )
            if has_compactor:
                found.append((d.name, d, True))
        return found

    if prompting_root.is_dir():
        for d in sorted(prompting_root.iterdir()):
            if d.is_dir() and (d / "tasks").is_dir():
                found.append((d.name, d, False))
    if compactor_root.is_dir():
        for d in sorted(compactor_root.iterdir()):
            if d.is_dir() and (d / "tasks").is_dir():
                found.append((d.name, d, True))
    return found


# ---------- Reporting --------------------------------------------------------


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "  n/a "
    return f"{v: .3f}"


def build_table(model_entries: list[tuple[str, Path, bool]],
                human_dists: dict[str, np.ndarray]) -> str:
    """Render a fixed-width table with per-(task, condition) rows."""
    out: list[str] = []
    out.append("# Per-model task scores and humanlikeness")
    out.append("")
    out.append("Each block reports, for one model and one prompting condition, "
               "per-task mean normalized score (1.0 = perfect) and humanlikeness "
               "(= 1 - W_1 on per-participant score distributions vs. humans).")
    out.append("")

    # First, per-task human means for reference
    out.append("## Humans (reference)")
    out.append("")
    out.append(f"{'task':<26}  {'n':>5}  {'mean_score':>10}")
    out.append("-" * 46)
    for t in TASKS:
        h = human_dists[t]
        out.append(f"{TASK_DISPLAY[t]:<26}  {h.size:>5}  {_fmt(np.mean(h) if h.size else float('nan')):>10}")
    out.append("")

    for display_name, model_dir, is_compactor in model_entries:
        suffix = " [compactor]" if is_compactor else ""
        out.append(f"## {display_name}{suffix}")
        out.append(f"_source: {model_dir}_")
        out.append("")

        conditions = ["compactor"] if is_compactor else PROMPT_CONDITIONS
        for cond in conditions:
            out.append(f"### {CONDITION_DISPLAY[cond]} ({cond})")
            out.append("")
            out.append(
                f"{'task':<26}  {'n':>5}  {'mean_score':>10}  {'humanlikeness':>13}"
            )
            out.append("-" * 64)
            for t in TASKS:
                m = llm_scores(t, model_dir, cond)
                h = human_dists[t]
                mean_s = float(np.mean(m)) if m.size else float("nan")
                hl = humanlikeness(h, m) if m.size else float("nan")
                out.append(
                    f"{TASK_DISPLAY[t]:<26}  {m.size:>5}  {_fmt(mean_s):>10}  {_fmt(hl):>13}"
                )
            out.append("")
        out.append("")

    return "\n".join(out) + "\n"


# ---------- CLI --------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help="Path to a model run directory (repeatable). If omitted, scores "
             "every model found under runs/prompting and runs/compactor.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "src" / "scores.txt",
        help="Output .txt path (default: src/scores.txt).",
    )
    args = parser.parse_args()

    explicit = [Path(p) for p in args.model_dir] if args.model_dir else None
    entries = discover_models(explicit=explicit)
    if not entries:
        raise SystemExit("No model directories found. Pass --model-dir <path>.")

    human_dists = {t: human_scores(t) for t in TASKS}

    text = build_table(entries, human_dists)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
