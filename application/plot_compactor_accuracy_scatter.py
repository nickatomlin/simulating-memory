#!/usr/bin/env python3
"""Scatter: x = document 1–10, y = accuracy, color = text level.

Supports compactor JSONL (default) or ``application/out`` reading QA rows filtered by ``--condition`` (e.g. C1).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPACTOR_ROOT = REPO_ROOT / "application" / "compactor"
DEFAULT_JSONL = "wm_application_reading_qa.jsonl"

LEVELS: tuple[str, ...] = ("biography", "distractor", "reading_level", "redundant")
LEVEL_COLORS: dict[str, str] = {
    "biography": "#1f77b4",
    "distractor": "#ff7f0e",
    "reading_level": "#2ca02c",
    "redundant": "#d62728",
}
# Slight horizontal shift so same-doc × multi-repeat points separate by level.
LEVEL_X_OFFSET: dict[str, float] = {
    "biography": -0.22,
    "distractor": -0.08,
    "reading_level": 0.08,
    "redundant": 0.22,
}


def doc_id_to_index(doc_id: str) -> int | None:
    s = str(doc_id).strip()
    m = re.match(r"^bio_0*(\d+)$", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)$", s)
    if m:
        return int(m.group(1))
    return None


def load_points(
    jsonl_path: Path,
    *,
    condition_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[float] = []
    ys: list[float] = []
    levels: list[str] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if condition_id is not None and str(r.get("condition_id") or "") != condition_id:
                continue
            doc_ix = doc_id_to_index(str(r.get("doc_id") or ""))
            if doc_ix is None:
                continue
            lv = str(r.get("level") or "")
            if lv not in LEVELS:
                continue
            acc = (r.get("metrics") or {}).get("accuracy")
            if acc is None:
                continue
            try:
                y = float(acc)
            except (TypeError, ValueError):
                continue
            x = float(doc_ix) + LEVEL_X_OFFSET[lv]
            xs.append(x)
            ys.append(y)
            levels.append(lv)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), levels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--condition",
        default=None,
        help="Prompt condition id to keep (e.g. C1). Uses application/out; requires --model unless --jsonl is set.",
    )
    ap.add_argument("--llm-out-root", type=Path, default=REPO_ROOT / "application" / "out")
    ap.add_argument(
        "--model",
        default=None,
        help="Model directory under llm-out-root (e.g. openai_gpt-5.4) when using --condition.",
    )
    ap.add_argument("--task-name", default="application_reading_qa", help="Task jsonl stem when using --condition.")
    ap.add_argument(
        "--compactor-subdir",
        default="gpt-5.4",
        help="Folder under application/compactor/ when not using --condition (default: gpt-5.4).",
    )
    ap.add_argument("--compactor-root", type=Path, default=DEFAULT_COMPACTOR_ROOT)
    ap.add_argument("--jsonl-name", default=DEFAULT_JSONL, help=f"Default: {DEFAULT_JSONL}")
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Exact file path (overrides compactor-root/subdir/jsonl-name).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image (.pdf or .png). Default: application/comparisons/figures/...",
    )
    args = ap.parse_args()

    cond_filter: str | None = args.condition.strip() if args.condition else None
    if args.jsonl is not None:
        jsonl_path = args.jsonl
    elif cond_filter:
        if not args.model:
            raise SystemExit("With --condition, pass --model (folder under llm-out-root) or --jsonl.")
        jsonl_path = args.llm_out_root / args.model / "tasks" / f"{args.task_name}.jsonl"
    else:
        jsonl_path = args.compactor_root / args.compactor_subdir / args.jsonl_name

    if not jsonl_path.is_file():
        raise SystemExit(f"Missing jsonl: {jsonl_path}")

    xs, ys, levels = load_points(jsonl_path, condition_id=cond_filter)
    if xs.size == 0:
        raise SystemExit("No valid points to plot (check doc_id, level, metrics.accuracy).")

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for lv in LEVELS:
        mask = np.array([l == lv for l in levels], dtype=bool)
        if not np.any(mask):
            continue
        ax.scatter(
            xs[mask],
            ys[mask],
            s=52,
            c=LEVEL_COLORS[lv],
            alpha=0.85,
            edgecolors="k",
            linewidths=0.35,
            label=lv.replace("_", " "),
            zorder=2,
        )

    ax.set_xticks(np.arange(1, 11))
    ax.set_xlim(0.35, 10.65)
    ax.set_xlabel("Document (1–10)")
    ax.set_ylabel("Accuracy")
    if cond_filter and args.model:
        title = f"{cond_filter}: {args.model} — {args.task_name}.jsonl"
        default_stem = f"condition_{cond_filter}_accuracy_scatter_{args.model.replace('/', '_')}"
    elif cond_filter:
        title = f"{cond_filter} — {jsonl_path.name}"
        default_stem = f"condition_{cond_filter}_accuracy_scatter"
    else:
        title = f"Compactor: {args.compactor_subdir} — {args.jsonl_name}"
        default_stem = f"compactor_accuracy_scatter_{args.compactor_subdir.replace('/', '_')}"

    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.legend(title="Reading level", ncol=2, frameon=True)

    out = args.out or (
        REPO_ROOT / "application" / "comparisons" / "figures" / f"{default_stem}.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lower().lstrip(".") or "pdf"
    fig.savefig(out, format=fmt, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
