#!/usr/bin/env python3
"""Grouped bar plot: compactor mean accuracy by reading level, x = model (compactor subfolder)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.compare_wasserstein_reading_qa import (  # noqa: E402
    DEFAULT_COMPACTOR_DIR_TO_OUT_MODEL,
    SIMILARITY_LINE_MODEL_DISPLAY_NAMES,
)

LEVELS: tuple[str, ...] = ("biography", "distractor", "reading_level", "redundant")
LEVEL_COLORS: dict[str, str] = {
    "biography": "#C9A8E8",
    "distractor": "#ff7f0e",
    "reading_level": "#2ca02c",
    "redundant": "#d62728",
}

DEFAULT_COMPACTOR_ROOT = REPO_ROOT / "application" / "compactor"
DEFAULT_JSONL = "wm_application_reading_qa.jsonl"


def _model_xlabel(comp_dir: str) -> str:
    out_key = DEFAULT_COMPACTOR_DIR_TO_OUT_MODEL.get(comp_dir, comp_dir)
    return SIMILARITY_LINE_MODEL_DISPLAY_NAMES.get(out_key, comp_dir.replace("_", " "))


def mean_accuracy_by_level(jsonl_path: Path) -> dict[str, float]:
    by_lv: dict[str, list[float]] = defaultdict(list)
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r: dict[str, Any] = json.loads(line)
            lv = str(r.get("level") or "")
            if lv not in LEVELS:
                continue
            acc = (r.get("metrics") or {}).get("accuracy")
            if acc is None:
                continue
            try:
                by_lv[lv].append(float(acc))
            except (TypeError, ValueError):
                continue
    return {lv: float(np.mean(by_lv[lv])) if by_lv[lv] else float("nan") for lv in LEVELS}


def discover_models(compactor_root: Path, jsonl_name: str) -> list[str]:
    dirs: list[str] = []
    for sub in sorted(compactor_root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / jsonl_name).is_file():
            dirs.append(sub.name)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compactor-root", type=Path, default=DEFAULT_COMPACTOR_ROOT)
    ap.add_argument("--jsonl-name", default=DEFAULT_JSONL)
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated compactor folder names (default: all that contain jsonl).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .pdf or .png. Default: application/comparisons/figures/...",
    )
    args = ap.parse_args()

    if args.models.strip():
        comp_dirs = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        comp_dirs = discover_models(args.compactor_root, args.jsonl_name)
    if not comp_dirs:
        raise SystemExit(f"No compactor runs under {args.compactor_root} with {args.jsonl_name}")

    stats: dict[str, dict[str, float]] = {}
    for name in comp_dirs:
        path = args.compactor_root / name / args.jsonl_name
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        stats[name] = mean_accuracy_by_level(path)

    x_labels = [_model_xlabel(d) for d in comp_dirs]
    x = np.arange(len(comp_dirs), dtype=np.float64)
    n_lv = len(LEVELS)
    width = min(0.8 / n_lv, 0.2)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(comp_dirs)), 4.8), constrained_layout=True)
    for i, lv in enumerate(LEVELS):
        offs = (i - (n_lv - 1) / 2.0) * width
        heights = [stats[d].get(lv, float("nan")) for d in comp_dirs]
        hplot = [0.0 if np.isnan(h) else float(h) for h in heights]
        ax.bar(
            x + offs,
            hplot,
            width * 0.92,
            label=lv.replace("_", " "),
            color=LEVEL_COLORS[lv],
            edgecolor="k",
            linewidth=0.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=22, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="#cccccc", lw=0.8, zorder=0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Reading level", ncol=2, frameon=True)
    ax.set_title(f"Compactor mean accuracy by reading level — {args.jsonl_name}")

    out = args.out or (
        REPO_ROOT / "application" / "comparisons" / "figures" / "compactor_accuracy_by_model_bar.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lower().lstrip(".") or "pdf"
    fig.savefig(out, format=fmt, dpi=180)
    plt.close(fig)

    print(f"Wrote {out}")
    print(f"Models ({len(comp_dirs)}): {', '.join(x_labels[:5])}{'…' if len(x_labels) > 5 else ''}")


if __name__ == "__main__":
    main()
