from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from bench.core.plotting import save_fig


def _condition_order(summary: Dict[str, Any]) -> List[str]:
    conds = summary.get("conditions", [])
    ids = [str(c.get("condition", "")) for c in conds if c.get("condition")]
    return ids or ["C1", "C2", "C3", "C4"]


def _plot_accuracy_by_condition(summary: Dict[str, Any], out_dir: Path) -> None:
    cond_ids = _condition_order(summary)
    by_condition = summary.get("by_condition", {})
    means = [float((by_condition.get(cid, {}).get("accuracy", {}) or {}).get("mean") or 0.0) for cid in cond_ids]
    fig = plt.figure(figsize=(6, 4))
    plt.bar(cond_ids, means, edgecolor="black")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Condition")
    plt.ylabel("Mean accuracy")
    plt.title("Reading QA: accuracy by condition")
    save_fig(fig, out_dir / "accuracy_by_condition.png")


def _plot_difficulty_by_condition(summary: Dict[str, Any], out_dir: Path) -> None:
    cond_ids = _condition_order(summary)
    by_condition = summary.get("by_condition", {})
    means = [float((by_condition.get(cid, {}).get("difficulty", {}) or {}).get("mean") or 0.0) for cid in cond_ids]
    fig = plt.figure(figsize=(6, 4))
    plt.bar(cond_ids, means, edgecolor="black", color="coral")
    plt.ylim(1.0, 10.0)
    plt.xlabel("Condition")
    plt.ylabel("Mean difficulty (1-10)")
    plt.title("Reading QA: difficulty by condition")
    save_fig(fig, out_dir / "difficulty_by_condition.png")


def _plot_grouped_by_condition_level(
    summary: Dict[str, Any],
    out_dir: Path,
    metric_key: str,
    title: str,
    y_label: str,
    filename: str,
) -> None:
    matrix = summary.get("by_condition_level_c123", {})
    cond_ids = [c for c in ["C1", "C2", "C3"] if c in matrix]
    levels = ["biography", "distractor", "reading_level", "redundant"]
    if not cond_ids:
        return

    x = np.arange(len(levels))
    width = 0.24
    fig = plt.figure(figsize=(9, 4.8))
    for idx, cond_id in enumerate(cond_ids):
        vals = []
        for level in levels:
            val = (
                (matrix.get(cond_id, {}).get(level, {}).get(metric_key, {}) or {}).get("mean")
                if metric_key == "accuracy"
                else (matrix.get(cond_id, {}).get(level, {}).get("difficulty", {}) or {}).get("mean")
            )
            vals.append(float(val or 0.0))
        plt.bar(x + (idx - 1) * width, vals, width=width, label=cond_id, edgecolor="black")

    plt.xticks(x, levels)
    if metric_key == "accuracy":
        plt.ylim(0.0, 1.0)
    else:
        plt.ylim(1.0, 10.0)
    plt.xlabel("Article level")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    save_fig(fig, out_dir / filename)


def plot_from_summary(summary_json_path: Path, out_dir: Path) -> List[Path]:
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_accuracy_by_condition(summary, out_dir)
    _plot_difficulty_by_condition(summary, out_dir)
    _plot_grouped_by_condition_level(
        summary,
        out_dir,
        metric_key="accuracy",
        title="C1-C3 accuracy by article level",
        y_label="Mean accuracy",
        filename="c123_accuracy_by_level.png",
    )
    _plot_grouped_by_condition_level(
        summary,
        out_dir,
        metric_key="difficulty",
        title="C1-C3 difficulty by article level",
        y_label="Mean difficulty (1-10)",
        filename="c123_difficulty_by_level.png",
    )
    return sorted(out_dir.glob("*.png"))
