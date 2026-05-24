#!/usr/bin/env python3
"""Align LLM vs human on which *cell* has higher mean accuracy (reading QA).

Each **trial** compares two cells: **A** = (article ``doc1``, text level ``l1``) vs **B** = (``doc2``,
``l2``), with **``l1 != l2``** (two distinct level types). Articles are drawn **independently with
replacement** (``doc1`` and ``doc2`` may be the same biography or not). Defaults to **many random
trials** (``--n-trials``); use ``--enumerate-doc-level-pairs`` for the older design (one article,
all ``C(4,2)`` level pairs, usually ``len(docs)*6`` trials).

C1, C2, C3 are three prompt conditions; **compactor** is separate when merged in.

**Human:** if either cell lacks human accuracy draws, **skip the trial**. Mean-accuracy ties
between the two cells are broken at random (reproducible).

**Per LLM condition:** missing data for a cell skips that condition on this trial; ties between
cells broken at random. Agreement = model's winning side (A/B) matches human's.

**Summary:** per-condition hit rates plus ``accuracy_C1`` … ``accuracy_compactor``.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.compare_wasserstein_reading_qa import (  # noqa: E402
    COMPACTOR_CONDITION_ID,
    DEFAULT_COMPACTOR_DIR_TO_OUT_MODEL,
    DEFAULT_COMPACTOR_JSONL,
    HUMAN_LEVEL_MAP,
    LEVELS,
    _load_model_rows,
    _merge_compactor_into_model_rows,
)

# Three reading-QA prompt conditions (see ``reading_qa/prompting.py``); compactor is separate.
PROMPT_CONDITIONS: tuple[str, ...] = ("C1", "C2", "C3")


def load_human_doc_level_accuracies(human_dir: Path) -> dict[str, dict[str, list[float]]]:
    """doc_id -> level -> accuracies from completed human runs."""
    out: dict[str, dict[str, list[float]]] = {}
    for p in sorted(human_dir.glob("run-*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") != "completed":
            continue
        doc_id = str(data.get("story_id") or "").strip()
        if not doc_id:
            continue
        raw_variant = str(data.get("story_variant") or "").strip()
        level = HUMAN_LEVEL_MAP.get(raw_variant)
        if level is None:
            continue
        summary = data.get("summary") or {}
        acc = summary.get("accuracy")
        if acc is None:
            continue
        try:
            fv = float(acc)
        except (TypeError, ValueError):
            continue
        out.setdefault(doc_id, {}).setdefault(level, []).append(fv)
    return out


def load_llm_doc_level_condition(
    rows: list[dict[str, Any]],
    conditions: list[str],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """condition_id -> doc_id -> level -> list of accuracy (one entry per repeat)."""
    out: dict[str, dict[str, dict[str, list[float]]]] = {c: {} for c in conditions}
    for r in rows:
        cond = r.get("condition_id")
        if cond not in out:
            continue
        doc_id = str(r.get("doc_id") or "").strip()
        level = r.get("level")
        if not doc_id or level not in LEVELS:
            continue
        acc = (r.get("metrics") or {}).get("accuracy")
        if acc is None:
            continue
        try:
            fv = float(acc)
        except (TypeError, ValueError):
            continue
        out[cond].setdefault(doc_id, {}).setdefault(str(level), []).append(fv)
    return out


def _two_cell_preference(
    stats_a: dict[str, list[float]],
    level_a: str,
    stats_b: dict[str, list[float]],
    level_b: str,
    rng: np.random.Generator,
) -> str | None:
    """Return ``\"A\"`` if cell (stats_a, level_a) has higher mean accuracy than B, else ``\"B\"``."""
    a = stats_a.get(level_a) or []
    b = stats_b.get(level_b) or []
    if not a or not b:
        return None
    m1 = float(np.mean(a))
    m2 = float(np.mean(b))
    if m1 > m2:
        return "A"
    if m2 > m1:
        return "B"
    return "A" if int(rng.integers(0, 2)) == 0 else "B"


def _strict_two_cells_ok(
    llm: dict[str, dict[str, dict[str, list[float]]]],
    eval_conditions: list[str],
    doc1: str,
    level1: str,
    doc2: str,
    level2: str,
) -> bool:
    for c in eval_conditions:
        s1 = llm.get(c, {}).get(doc1) or {}
        s2 = llm.get(c, {}).get(doc2) or {}
        if not (s1.get(level1) and s2.get(level2)):
            return False
    return True


def sample_trials_random_two_docs(
    docs: list[str],
    rng: np.random.Generator,
    n_trials: int,
    *,
    llm: dict[str, dict[str, dict[str, list[float]]]] | None = None,
    eval_conditions: list[str] | None = None,
    strict_llm: bool = False,
) -> tuple[list[tuple[str, str, str, str]], int]:
    """Sample ``n_trials`` tuples ``(doc1, l1, doc2, l2)`` with ``l1 != l2``.

    Returns (trials, n_attempts). When ``strict_llm``, rejection-sampling requires ``llm`` and
    ``eval_conditions``; otherwise they are ignored.
    """
    if not docs or n_trials <= 0:
        return [], 0
    if strict_llm and (llm is None or eval_conditions is None):
        raise ValueError("strict_llm requires llm and eval_conditions")
    doc_idx = np.arange(len(docs))
    trials: list[tuple[str, str, str, str]] = []
    attempts = 0
    max_attempts = n_trials * 500 if strict_llm else n_trials
    while len(trials) < n_trials and attempts < max_attempts:
        attempts += 1
        i1 = int(rng.choice(doc_idx))
        i2 = int(rng.choice(doc_idx))
        d1, d2 = docs[i1], docs[i2]
        li, lj = rng.choice(len(LEVELS), size=2, replace=False)
        l1, l2 = LEVELS[int(li)], LEVELS[int(lj)]
        if strict_llm and not _strict_two_cells_ok(llm, eval_conditions, d1, l1, d2, l2):
            continue
        trials.append((d1, l1, d2, l2))
    return trials, attempts


def enumerate_trials_single_doc_level_pairs(
    docs: list[str],
    llm: dict[str, dict[str, dict[str, list[float]]]],
    eval_conditions: list[str],
    *,
    strict_llm: bool,
) -> list[tuple[str, str, str, str]]:
    pairs = list(combinations(LEVELS, 2))
    trials = []
    for d in docs:
        for a, b in pairs:
            if strict_llm and not _strict_two_cells_ok(llm, eval_conditions, d, a, d, b):
                continue
            trials.append((d, a, d, b))
    return trials


def _intersect_docs(
    human: dict[str, dict[str, list[float]]],
    llm: dict[str, dict[str, dict[str, list[float]]]],
    base_conditions: list[str],
) -> list[str]:
    docs_h = set(human.keys())
    docs_m: set[str] = set()
    for c in base_conditions:
        docs_m |= set(llm.get(c, {}).keys())
    return sorted(docs_h & docs_m)


def _rng_for_model(global_seed: int, model: str) -> np.random.Generator:
    salt = zlib.adler32(model.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([int(global_seed) & 0xFFFFFFFF, salt]))


def run_one_model(
    *,
    model: str,
    llm: dict[str, dict[str, dict[str, list[float]]]],
    human: dict[str, dict[str, list[float]]],
    trials: list[tuple[str, str, str, str]],
    eval_conditions: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not trials:
        return {
            "model": model,
            "n_docs_in_pool": 0,
            "n_trials_drawn": 0,
            "n_trials_human_defined": 0,
            "error": "no trials (no overlapping doc_ids between human and C1–C3 LLM rows?)",
        }

    hits = {c: 0 for c in eval_conditions}
    denom = {c: 0 for c in eval_conditions}
    n_h = 0

    for doc1, l1, doc2, l2 in trials:
        h_a = human.get(doc1) or {}
        h_b = human.get(doc2) or {}
        h_side = _two_cell_preference(h_a, l1, h_b, l2, rng)
        if h_side is None:
            continue
        n_h += 1
        for c in eval_conditions:
            la = llm.get(c, {}).get(doc1) or {}
            lb = llm.get(c, {}).get(doc2) or {}
            m_side = _two_cell_preference(la, l1, lb, l2, rng)
            if m_side is None:
                continue
            denom[c] += 1
            if m_side == h_side:
                hits[c] += 1

    acc = {c: (float(hits[c]) / float(denom[c])) if denom[c] else None for c in eval_conditions}

    n_docs = len({t[0] for t in trials} | {t[2] for t in trials})

    return {
        "model": model,
        "n_docs_in_pool": n_docs,
        "n_trials_drawn": len(trials),
        "n_trials_human_defined": n_h,
        "hits": hits,
        "denom": denom,
        "accuracy_by_condition": acc,
        "accuracy_C1": acc.get("C1"),
        "accuracy_C2": acc.get("C2"),
        "accuracy_C3": acc.get("C3"),
        "accuracy_compactor": acc.get(COMPACTOR_CONDITION_ID) if COMPACTOR_CONDITION_ID in eval_conditions else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare two (article, level) cells: which has higher mean accuracy vs human and per "
            "prompt condition (C1–C3; optional compactor). Default: random two articles (with replacement) "
            "and two distinct levels; many trials via --n-trials."
        )
    )
    ap.add_argument("--llm-out-root", type=Path, default=REPO_ROOT / "application" / "out")
    ap.add_argument("--human-dir", type=Path, default=REPO_ROOT / "application" / "approved_runs")
    ap.add_argument("--task-name", default="application_reading_qa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated model dir names under llm-out-root (default: all that have task jsonl).",
    )
    ap.add_argument(
        "--compactor-root",
        type=Path,
        default=REPO_ROOT / "application" / "compactor",
    )
    ap.add_argument(
        "--no-compactor",
        action="store_true",
        help="Do not merge compactor JSONL; evaluate C1–C3 only.",
    )
    ap.add_argument("--compactor-jsonl", default=DEFAULT_COMPACTOR_JSONL)
    ap.add_argument(
        "--compactor-model-map",
        type=Path,
        default=None,
        help="Optional JSON object: compactor folder name -> application/out model dir name.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write results here (default: application/comparisons/level_pair_alignment_<utc>.json).",
    )
    ap.add_argument(
        "--n-trials",
        type=int,
        default=5000,
        help="Random design: number of trials to draw (default: 5000). Ignored with --enumerate-doc-level-pairs.",
    )
    ap.add_argument(
        "--enumerate-doc-level-pairs",
        action="store_true",
        help=(
            "Legacy: one article per trial, all unordered level pairs per article "
            "(~ len(docs)*6 trials), then shuffle."
        ),
    )
    ap.add_argument(
        "--strict-llm-coverage",
        action="store_true",
        help=(
            "Random design: resample until each trial has LLM data for both cells under every eval "
            "condition (may use many attempts; capped). Enumerate design: skip trials that fail this."
        ),
    )
    args = ap.parse_args()

    if args.n_trials <= 0 and not args.enumerate_doc_level_pairs:
        raise SystemExit("--n-trials must be > 0 unless using --enumerate-doc-level-pairs")

    human = load_human_doc_level_accuracies(args.human_dir)
    model_rows = _load_model_rows(args.llm_out_root, task_name=args.task_name)

    compactor_counts: dict[str, int] = {}
    if not args.no_compactor:
        extra_map: dict[str, Any] = {}
        if args.compactor_model_map is not None and args.compactor_model_map.is_file():
            extra_map = json.loads(args.compactor_model_map.read_text(encoding="utf-8"))
            if not isinstance(extra_map, dict):
                raise ValueError("--compactor-model-map must contain a JSON object")
        dir_map: dict[str, str] = {
            str(k): str(v) for k, v in {**DEFAULT_COMPACTOR_DIR_TO_OUT_MODEL, **extra_map}.items()
        }
        compactor_counts = _merge_compactor_into_model_rows(
            model_rows,
            args.compactor_root,
            jsonl_name=args.compactor_jsonl,
            synthetic_condition=COMPACTOR_CONDITION_ID,
            compactor_dir_to_out_model=dir_map,
        )

    want = [m.strip() for m in args.models.split(",") if m.strip()]
    models = sorted(model_rows.keys())
    if want:
        models = [m for m in models if m in want]
        missing = set(want) - set(models)
        if missing:
            raise SystemExit(f"Unknown or missing models (no {args.task_name}.jsonl): {sorted(missing)}")

    eval_conditions = list(PROMPT_CONDITIONS) + ([] if args.no_compactor else [COMPACTOR_CONDITION_ID])

    rows_out: list[dict[str, Any]] = []
    for model in models:
        rows = model_rows[model]
        llm = load_llm_doc_level_condition(rows, eval_conditions)
        docs = _intersect_docs(human, llm, list(PROMPT_CONDITIONS))
        # Fresh RNG each model so non-strict random trials match across models; strict mode differs per llm.
        trial_list_rng = np.random.default_rng(
            np.random.SeedSequence([int(args.seed) & 0xFFFFFFFF, 0x46524941])
        )
        if args.enumerate_doc_level_pairs:
            trials = enumerate_trials_single_doc_level_pairs(
                docs, llm, eval_conditions, strict_llm=args.strict_llm_coverage
            )
            shuffle_rng = np.random.default_rng(
                np.random.SeedSequence([int(args.seed) & 0xFFFFFFFF, 0x5348_FFFF])
            )
            shuffle_rng.shuffle(trials)
            n_attempts: int | None = None
        else:
            trials, n_attempts = sample_trials_random_two_docs(
                docs,
                trial_list_rng,
                args.n_trials,
                llm=llm,
                eval_conditions=eval_conditions,
                strict_llm=args.strict_llm_coverage,
            )
        row: dict[str, Any] = run_one_model(
            model=model,
            llm=llm,
            human=human,
            trials=trials,
            eval_conditions=eval_conditions,
            rng=_rng_for_model(args.seed, model),
        )
        if not args.enumerate_doc_level_pairs:
            row["n_trial_generation_attempts"] = int(n_attempts)
            if len(trials) < args.n_trials:
                row["trial_generation_underfilled"] = True
        rows_out.append(row)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_json or (REPO_ROOT / "application" / "comparisons" / f"level_pair_alignment_{stamp}.json")

    summary = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "trial_design": (
                "enumerate_single_doc_level_pairs"
                if args.enumerate_doc_level_pairs
                else "random_two_docs_two_distinct_levels"
            ),
            "n_trials_requested": None if args.enumerate_doc_level_pairs else int(args.n_trials),
            "strict_llm_coverage": bool(args.strict_llm_coverage),
            "levels": LEVELS,
            "prompt_conditions": list(PROMPT_CONDITIONS),
            "eval_conditions": eval_conditions,
            "human_dir": str(args.human_dir),
            "llm_out_root": str(args.llm_out_root),
            "task_name": args.task_name,
            "compactor": {
                "skipped": bool(args.no_compactor),
                "root": str(args.compactor_root),
                "rows_merged_by_model": compactor_counts,
            },
        },
        "models": rows_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    def _fmt_acc(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "na"

    for r in rows_out:
        if "accuracy_by_condition" in r:
            c1, c2, c3 = r.get("accuracy_C1"), r.get("accuracy_C2"), r.get("accuracy_C3")
            comp = r.get("accuracy_compactor")
            print(
                f"  {r['model']}: human_trials={r['n_trials_human_defined']}/"
                f"{r.get('n_trials_drawn', 0)} "
                f"C1={_fmt_acc(c1)} C2={_fmt_acc(c2)} C3={_fmt_acc(c3)} compactor={_fmt_acc(comp)}"
            )


if __name__ == "__main__":
    main()
