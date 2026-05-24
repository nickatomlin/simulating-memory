#!/usr/bin/env python3
"""Per-article correlation: human pooled score vs C1/C2/C3/compactor (one point = one doc).

For each ``doc_id``, the **score** is the mean over text levels (biography, distractor, …) of
that cohort's mean accuracy — only levels with at least one observation count. Human uses
``approved_runs``; LLM uses ``application/out`` jsonl (+ optional compactor merge).

Plots a 2×2 scatter grid (human vs each condition) and a bar chart of Pearson *r*.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.compare_wasserstein_reading_qa import (  # noqa: E402
    COMPACTOR_CONDITION_ID,
    DEFAULT_COMPACTOR_DIR_TO_OUT_MODEL,
    DEFAULT_COMPACTOR_JSONL,
    LEVELS,
    SIMILARITY_LINE_CONDITION_LABELS,
    _load_model_rows,
    _merge_compactor_into_model_rows,
)
from application.level_pair_preference_alignment import (  # noqa: E402
    PROMPT_CONDITIONS,
    load_human_doc_level_accuracies,
    load_llm_doc_level_condition,
)

CONDITION_ORDER = list(PROMPT_CONDITIONS) + [COMPACTOR_CONDITION_ID]


def _doc_level_mean_scores(
    doc_level_lists: dict[str, dict[str, list[float]]],
) -> dict[str, float]:
    """doc_id -> mean over levels of mean(accuracy list); levels with no data omitted."""
    out: dict[str, float] = {}
    for doc, by_lvl in doc_level_lists.items():
        vals: list[float] = []
        for lv in LEVELS:
            xs = by_lvl.get(lv) or []
            if xs:
                vals.append(float(np.mean(xs)))
        if vals:
            out[doc] = float(np.mean(vals))
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    d = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if d <= 0.0:
        return float("nan")
    return float(np.sum(x * y) / d)


def _condition_axis_label(cond: str) -> str:
    if cond == COMPACTOR_CONDITION_ID:
        return "Compactor"
    return SIMILARITY_LINE_CONDITION_LABELS.get(cond, cond)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-out-root", type=Path, default=REPO_ROOT / "application" / "out")
    ap.add_argument("--human-dir", type=Path, default=REPO_ROOT / "application" / "approved_runs")
    ap.add_argument("--task-name", default="application_reading_qa")
    ap.add_argument(
        "--model",
        required=True,
        help="Model directory name under llm-out-root (e.g. openai_gpt-5.4).",
    )
    ap.add_argument("--compactor-root", type=Path, default=REPO_ROOT / "application" / "compactor")
    ap.add_argument("--no-compactor", action="store_true")
    ap.add_argument("--compactor-jsonl", default=DEFAULT_COMPACTOR_JSONL)
    ap.add_argument("--compactor-model-map", type=Path, default=None)
    ap.add_argument(
        "--out-figure",
        type=Path,
        default=None,
        help="Combined figure path (.pdf or .png). Default: application/comparisons/...",
    )
    ap.add_argument("--out-table", type=Path, default=None, help="Optional JSON with per-doc scores and r.")
    args = ap.parse_args()

    human_nested = load_human_doc_level_accuracies(args.human_dir)
    human_by_doc = _doc_level_mean_scores(human_nested)

    model_rows = _load_model_rows(args.llm_out_root, task_name=args.task_name)
    if args.model not in model_rows:
        raise SystemExit(f"Model {args.model!r} not found under {args.llm_out_root} (missing {args.task_name}.jsonl?)")

    eval_conditions = list(PROMPT_CONDITIONS) + ([] if args.no_compactor else [COMPACTOR_CONDITION_ID])
    rows = model_rows[args.model]
    if not args.no_compactor:
        extra: dict[str, Any] = {}
        if args.compactor_model_map is not None and args.compactor_model_map.is_file():
            extra = json.loads(args.compactor_model_map.read_text(encoding="utf-8"))
            if not isinstance(extra, dict):
                raise ValueError("--compactor-model-map must be a JSON object")
        dir_map = {str(k): str(v) for k, v in {**DEFAULT_COMPACTOR_DIR_TO_OUT_MODEL, **extra}.items()}
        _merge_compactor_into_model_rows(
            {args.model: rows},
            args.compactor_root,
            jsonl_name=args.compactor_jsonl,
            synthetic_condition=COMPACTOR_CONDITION_ID,
            compactor_dir_to_out_model=dir_map,
        )

    llm_nested = load_llm_doc_level_condition(rows, eval_conditions)
    cond_doc_score: dict[str, dict[str, float]] = {
        c: _doc_level_mean_scores(llm_nested.get(c, {})) for c in eval_conditions
    }

    docs_with_any_llm = set().union(*[set(cond_doc_score[c].keys()) for c in eval_conditions])
    docs_common = sorted(set(human_by_doc.keys()) & docs_with_any_llm)
    table_rows: list[dict[str, Any]] = []
    for d in docs_common:
        rec: dict[str, Any] = {"doc_id": d, "human": human_by_doc[d]}
        for c in eval_conditions:
            rec[c] = cond_doc_score[c].get(d)
        table_rows.append(rec)

    # Pairwise Pearson (only docs where both human and condition defined)
    r_by_cond: dict[str, float] = {}
    for c in eval_conditions:
        xs, ys = [], []
        for rec in table_rows:
            hx, cx = rec["human"], rec.get(c)
            if cx is not None:
                xs.append(hx)
                ys.append(cx)
        r_by_cond[c] = _pearson_r(np.array(xs), np.array(ys))

    # --- figures ---
    fig_scatter, axes = plt.subplots(2, 2, figsize=(7.5, 6.5), constrained_layout=True)
    ax_flat = axes.ravel()
    for ax, cond in zip(ax_flat, CONDITION_ORDER):
        if cond not in eval_conditions:
            ax.set_visible(False)
            continue
        xs, ys = [], []
        for rec in table_rows:
            cx = rec.get(cond)
            if cx is None:
                continue
            xs.append(rec["human"])
            ys.append(cx)
        ax.scatter(xs, ys, s=36, alpha=0.85, edgecolors="k", linewidths=0.4)
        lo = min(min(xs, default=0), min(ys, default=0))
        hi = max(max(xs, default=1), max(ys, default=1))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.45)
        r = r_by_cond.get(cond, float("nan"))
        ax.set_title(f"{_condition_axis_label(cond)}  (r = {r:.3f})" if np.isfinite(r) else f"{_condition_axis_label(cond)}  (r = n/a)")
        ax.set_xlabel("Human mean accuracy (per doc)")
        ax.set_ylabel(f"{_condition_axis_label(cond)} mean accuracy (per doc)")
        ax.set_xlim(max(0, lo - pad), min(1, hi + pad) + 1e-6)
        ax.set_ylim(max(0, lo - pad), min(1, hi + pad) + 1e-6)

    fig_bar, axb = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    present = [c for c in CONDITION_ORDER if c in eval_conditions]
    rs = [r_by_cond[c] for c in present]
    labs = [_condition_axis_label(c) for c in present]
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"][: len(present)]
    xpos = np.arange(len(present))
    heights = [r if np.isfinite(r) else 0.0 for r in rs]
    axb.bar(xpos, heights, color=colors, edgecolor="k", linewidth=0.5)
    for i, r in enumerate(rs):
        if not np.isfinite(r):
            axb.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=9)
    axb.set_xticks(xpos)
    axb.set_xticklabels(labs, rotation=12, ha="right")
    axb.axhline(0.0, color="k", lw=0.6)
    axb.set_ylabel("Pearson r vs human (per doc)")
    axb.set_title(args.model)

    default_dir = REPO_ROOT / "application" / "comparisons" / f"doc_corr_{args.model}"
    default_dir.mkdir(parents=True, exist_ok=True)
    out_fig = args.out_figure or (default_dir / "doc_human_condition_correlation.pdf")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    suf = out_fig.suffix.lower().lstrip(".") or "pdf"
    if suf == "pdf":
        with PdfPages(out_fig) as pdf:
            pdf.savefig(fig_scatter)
            pdf.savefig(fig_bar)
    else:
        fig_scatter.savefig(out_fig, format=suf)
        fig_bar.savefig(out_fig.with_name(out_fig.stem + "_r_bar" + out_fig.suffix), format=suf)

    if args.out_table:
        payload = {
            "model": args.model,
            "n_docs": len(table_rows),
            "pearson_r_vs_human": {k: (float(v) if np.isfinite(v) else None) for k, v in r_by_cond.items()},
            "docs": table_rows,
        }
        args.out_table.parent.mkdir(parents=True, exist_ok=True)
        args.out_table.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plt.close(fig_scatter)
    plt.close(fig_bar)

    print(f"Figure: {out_fig}" + (" (2 pages: scatters, r bar)" if suf == "pdf" else ""))
    for c in present:
        print(f"  r(human, {c}) = {r_by_cond[c]:.4f}" if np.isfinite(r_by_cond[c]) else f"  r(human, {c}) = n/a")


if __name__ == "__main__":
    main()
