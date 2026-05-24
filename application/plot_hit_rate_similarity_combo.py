#!/usr/bin/env python3
"""One figure: left = condition hit-rate bars (TaskPr … Compactor); right = reading-QA similarity lines.

Right panel matches ``similarity_lines_by_model_x_condition`` (accuracy, mean similarity over levels).
Data for lines is loaded from ``compare_wasserstein_reading_qa`` ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.compare_wasserstein_reading_qa import (
    SIMILARITY_LINE_CONDITION_LABELS,
    SIMILARITY_LINE_MODEL_DISPLAY_NAMES,
    SIMILARITY_LINE_PLOT_RC,
    _draw_similarity_lines_on_ax,
    _mean_similarity_and_ci_bar,
    similarity_line_fig_legend,
)

BAR_LABELS = ["TaskPr", "HumPr", "MemPr", "Compactor"]
BAR_LEFT_FILL = "#C9A8E8"
BAR_COLORS = {k: BAR_LEFT_FILL for k in BAR_LABELS}

# Right-panel line order: ``application/out`` model directory names (matches legend display names).
COMBO_MODEL_ORDER: tuple[str, ...] = (
    "openai_gpt-5.4",
    "anthropic_claude-opus-4.6",
    "meta-llama_llama-3-8b-instruct",
    "meta-llama_llama-3.3-70b-instruct",
    "qwen_qwen3-8b_thinking_false",
    "qwen_qwen3-8b_thinking_true",
    "qwen_qwen3-30b-a3b-instruct-2507",
    "qwen_qwen3-30b-a3b-thinking-2507",
    "qwen_qwen3-next-80b-a3b-instruct",
)


def _sort_models_for_combo(found: set[str]) -> list[str]:
    """Stable preferred order, then any extra models (e.g. new runs) alphabetically."""
    primary = [k for k in COMBO_MODEL_ORDER if k in found]
    rest = sorted(found.difference(COMBO_MODEL_ORDER))
    return primary + rest


# Match ``SIMILARITY_LINE_PLOT_RC`` / right panel axis typography.
COMBO_AXIS_LABEL_FS = int(SIMILARITY_LINE_PLOT_RC["axes.labelsize"])
COMBO_AXIS_TICK_FS = int(SIMILARITY_LINE_PLOT_RC["xtick.labelsize"])


def _print_right_panel_similarity_table(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
    metric: str,
    summary_path: Path,
) -> None:
    """Print mean similarity (and CI bounds) used by the right subplot — same as ``_draw_similarity_lines_on_ax``."""
    cond_headers = [SIMILARITY_LINE_CONDITION_LABELS.get(c, c) for c in conditions]
    print(f"\nRight-panel data (from {summary_path})")
    print(
        f"metric={metric!r}: mean similarity across article levels per condition; "
        "CI bounds are averaged per-level intervals (same as plot error bars).\n"
    )
    model_w = max(
        28,
        max((len(SIMILARITY_LINE_MODEL_DISPLAY_NAMES.get(m, m)) for m in models), default=0) + 2,
    )
    col_w = 26
    hdr = f"{'model':<{model_w}}" + "".join(f"{h:>{col_w}}" for h in cond_headers)
    print(hdr)
    print("-" * len(hdr))
    for model in models:
        disp = SIMILARITY_LINE_MODEL_DISPLAY_NAMES.get(model, model)
        parts: list[str] = [f"{disp:<{model_w}}"]
        for cond in conditions:
            y, lo, hi = _mean_similarity_and_ci_bar(rows, model=model, condition=cond, metric=metric)
            if np.isfinite(y):
                cell = f"{y:.4f}"
                if np.isfinite(lo) and np.isfinite(hi):
                    cell = f"{y:.4f}[{lo:.4f},{hi:.4f}]"
            else:
                cell = "nan"
            parts.append(f"{cell:>{col_w}}")
        print("".join(parts))
    print()


def _load_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected list at key 'rows'")
    meta = data.get("meta") or {}
    return meta, rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combined hit-rate bar plot + similarity-by-condition line plot (reading QA)."
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "application" / "comparisons" / "results" / "summary.json",
    )
    ap.add_argument("--c1", type=float, default=0.4672)
    ap.add_argument("--c2", type=float, default=0.4648)
    ap.add_argument("--c3", type=float, default=0.5660)
    ap.add_argument("--compactor", type=float, default=0.6472)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output base path: writes PNG and sibling PDF. "
            "Default: …/hit_rate_similarity_combo.png + …/hit_rate_similarity_combo.pdf. "
            "If you pass a .pdf path, that file is PDF and PNG uses the same stem."
        ),
    )
    ap.add_argument(
        "--no-print-right-data",
        action="store_true",
        help="Skip printing the right-panel similarity table to stdout.",
    )
    args = ap.parse_args()

    summary_path = args.summary_json.resolve()
    meta, rows = _load_summary(summary_path)
    conditions = list(meta.get("conditions") or [])
    if not conditions:
        conditions = sorted({str(r.get("condition")) for r in rows if r.get("condition")})
    models = _sort_models_for_combo({str(r["model"]) for r in rows if r.get("model")})

    if not args.no_print_right_data:
        _print_right_panel_similarity_table(
            rows,
            models=models,
            conditions=conditions,
            metric="accuracy",
            summary_path=summary_path,
        )

    values = [args.c1, args.c2, args.c3, args.compactor]
    x = range(len(BAR_LABELS))
    colors = [BAR_COLORS[k] for k in BAR_LABELS]

    default_dir = ROOT / "application" / "comparisons" / "results" / "figures"
    if args.out is None:
        png_out = default_dir / "hit_rate_similarity_combo.png"
        pdf_out = default_dir / "hit_rate_similarity_combo.pdf"
    else:
        p = Path(args.out)
        if p.suffix.lower() == ".pdf":
            pdf_out = p
            png_out = p.with_suffix(".png")
        else:
            png_out = p
            pdf_out = p.with_suffix(".pdf")

    with plt.rc_context(SIMILARITY_LINE_PLOT_RC):
        fig, (ax_bar, ax_line) = plt.subplots(
            1,
            2,
            figsize=(13.2, 4.05),
            gridspec_kw={"width_ratios": [1.05, 1.4], "wspace": 0.28},
        )

        ax_bar.bar(x, values, color=colors, edgecolor="black", linewidth=0.85)
        ax_bar.set_xticks(list(x))
        ax_bar.set_xticklabels(BAR_LABELS, fontsize=COMBO_AXIS_TICK_FS)
        ax_bar.set_ylabel("Pairwise Ranking Accuracy", fontsize=COMBO_AXIS_LABEL_FS)
        ax_bar.set_ylim(0.0, 1.0)
        ax_bar.tick_params(axis="both", which="major", labelsize=COMBO_AXIS_TICK_FS)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.grid(False)

        cmap = plt.get_cmap("tab10")
        legend_handles: list[Any] = []
        _draw_similarity_lines_on_ax(
            ax_line,
            rows,
            models=models,
            conditions=conditions,
            metric="accuracy",
            condition_axis_labels=SIMILARITY_LINE_CONDITION_LABELS,
            cmap=cmap,
            axis_label_fs=COMBO_AXIS_LABEL_FS,
            axis_tick_fs=COMBO_AXIS_TICK_FS,
            legend_handles_out=legend_handles,
            y_axis_label="Humanlikeness",
        )
        legend_labels = [SIMILARITY_LINE_MODEL_DISPLAY_NAMES.get(m, m) for m in models]

        # Exactly two rows: ncol = ceil(n/2) (e.g. 6 models → 3 columns × 2 rows).
        legend_ncol = max(1, math.ceil(len(models) / 2)) if models else 1
        similarity_line_fig_legend(
            fig,
            legend_handles,
            legend_labels,
            bbox_to_anchor=(0.52, 0.02),
            ncol=legend_ncol,
            labelspacing=0.45,
            columnspacing=1.35,
        )
        fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.145, wspace=0.30)

        png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_out, dpi=180, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(pdf_out, format="pdf", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)

    print(f"Saved {png_out}")
    print(f"Saved {pdf_out}")


if __name__ == "__main__":
    main()
