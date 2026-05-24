#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["biography", "distractor", "reading_level", "redundant"]
DEFAULT_CONDITIONS = ["C1", "C2", "C3"]
COMPACTOR_CONDITION_ID = "compactor"
DEFAULT_COMPACTOR_JSONL = "wm_application_reading_qa.jsonl"
# Subfolder under application/compactor/ -> model folder name under application/out/
DEFAULT_COMPACTOR_DIR_TO_OUT_MODEL: dict[str, str] = {
    "claude-opus-4-6": "anthropic_claude-opus-4.6",
    "gpt-5.4": "openai_gpt-5.4",
    "qwen_qwen3-8b": "qwen_qwen3-8b_thinking_true",
    "qwen_qwen3-8b_false": "qwen_qwen3-8b_thinking_false",
}
# X-axis labels for the similarity line figure only (reading-QA conditions).
SIMILARITY_LINE_CONDITION_LABELS: dict[str, str] = {
    "C1": "TaskPr",
    "C2": "HumPr",
    "C3": "MemPr",
    "compactor": "Compactor",
}
# Legend labels for ``application/out`` model directory names (similarity line plot).
SIMILARITY_LINE_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "anthropic_claude-opus-4.6": "Claude Opus 4.6",
    "meta-llama_llama-3-8b-instruct": "Llama 3 8B Instruct",
    "meta-llama_llama-3.3-70b-instruct": "Llama 3.3 70B Instruct",
    "openai_gpt-5.4": "GPT-5.4",
    "qwen_qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B-Instruct",
    "qwen_qwen3-30b-a3b-thinking-2507": "Qwen3-30B-A3B-Thinking",
    "qwen_qwen3-8b_thinking_false": "Qwen3-8B (Standard)",
    "qwen_qwen3-8b_thinking_true": "Qwen3-8B (Thinking)",
    "qwen_qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B-A3B-Instruct",
}
# Line colors for similarity plots (keys = ``SIMILARITY_LINE_MODEL_DISPLAY_NAMES`` values).
SIMILARITY_LINE_MODEL_DISPLAY_COLORS: dict[str, str] = {
    "GPT-5.4": "#CBE4B1",
    "Claude Opus 4.6": "#B8E5FA",
    "Human": "#BFBFBF",
    "Llama 3 8B Instruct": "#C9A8E8",
    "Llama 3.3 70B Instruct": "#7B3FA8",
    "Qwen3-30B-A3B-Instruct": "#E8151B",
    "Qwen3-30B-A3B-Thinking": "#F7C94E",
    "Qwen3-8B (Standard)": "#F4831F",
    "Qwen3-8B (Thinking)": "#F7B2C7",
    "Qwen3-Next-80B-A3B-Instruct": "#FCE98A",
}

# Serif/academic styling for similarity line figures (legend + axis typography).
SIMILARITY_LINE_PLOT_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
        "Bitstream Vera Serif",
        "Computer Modern Roman",
        "serif",
    ],
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}

SIMILARITY_LINE_ERRORBAR_ECOLOR = "#404040"

METRICS = ["accuracy", "difficulty"]
SPAN_BY_METRIC = {"accuracy": 1.0, "difficulty": 9.0}

HUMAN_LEVEL_MAP = {
    "biography_text": "biography",
    "distractor": "distractor",
    "reading_level": "reading_level",
    "redundancy": "redundant",
}


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())
    n, m = x.size, y.size
    if n == 0 or m == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x, y]))
    if grid.size == 1:
        return float(abs(np.mean(x) - np.mean(y)))
    edges = np.concatenate([[grid[0]], (grid[:-1] + grid[1:]) / 2.0, [grid[-1]]])
    total = 0.0
    for i in range(len(edges) - 1):
        mid = 0.5 * (edges[i] + edges[i + 1])
        fx = np.searchsorted(x, mid, side="right") / n
        fy = np.searchsorted(y, mid, side="right") / m
        total += abs(fx - fy) * (edges[i + 1] - edges[i])
    return float(total)


def _bootstrap_ci_from_values(values: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = max(0.0, min(100.0, 100.0 - float(ci)))
    lo = float(np.percentile(values, alpha / 2.0))
    hi = float(np.percentile(values, 100.0 - alpha / 2.0))
    return lo, hi


def bootstrap_w1_ci(
    h: np.ndarray,
    m: np.ndarray,
    *,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    point = wasserstein_1d(h, m)
    if h.size == 0 or m.size == 0 or np.isnan(point):
        return point, float("nan"), float("nan")
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        hb = h[rng.integers(0, h.size, size=h.size)]
        mb = m[rng.integers(0, m.size, size=m.size)]
        boots[i] = wasserstein_1d(hb, mb)
    lo, hi = _bootstrap_ci_from_values(boots, ci)
    return point, lo, hi


def _mean_sem(arr: np.ndarray) -> tuple[float, float]:
    """Sample mean and standard error of the mean (ddof=1). SEM=0 when n<=1."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(a))
    if a.size <= 1:
        return mean, 0.0
    sem = float(np.std(a, ddof=1) / np.sqrt(a.size))
    return mean, sem


def _normalize_w1(w_raw: float, w_lo: float, w_hi: float, span: float) -> tuple[float, float, float]:
    if np.isnan(w_raw):
        return float("nan"), float("nan"), float("nan")
    if span <= 1e-12:
        return 0.0, 0.0, 0.0
    return float(w_raw / span), float(w_lo / span), float(w_hi / span)


def _sim_from_w1norm(w_norm: float, w_lo: float, w_hi: float) -> tuple[float, float, float]:
    if np.isnan(w_norm):
        return float("nan"), float("nan"), float("nan")
    sim = 1.0 - float(w_norm)
    if np.isnan(w_lo) or np.isnan(w_hi):
        return sim, float("nan"), float("nan")
    sim_lo = 1.0 - float(w_hi)
    sim_hi = 1.0 - float(w_lo)
    return sim, sim_lo, sim_hi


def _json_safe(v: float) -> float | None:
    return None if np.isnan(v) else float(v)


def _load_model_rows(llm_out_root: Path, *, task_name: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for model_dir in sorted(llm_out_root.iterdir()):
        if not model_dir.is_dir():
            continue
        path = model_dir / "tasks" / f"{task_name}.jsonl"
        if not path.is_file():
            continue
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            out[model_dir.name] = rows
    return out


def _merge_compactor_into_model_rows(
    model_rows: dict[str, list[dict[str, Any]]],
    compactor_root: Path,
    *,
    jsonl_name: str,
    synthetic_condition: str,
    compactor_dir_to_out_model: dict[str, str],
) -> dict[str, int]:
    """Append compactor JSONL rows into ``model_rows`` with ``condition_id`` rewritten.

    Returns counts of rows merged per ``application/out`` model key (empty if nothing loaded).
    """
    counts: dict[str, int] = {}
    if not compactor_root.is_dir():
        return counts
    for sub in sorted(compactor_root.iterdir()):
        if not sub.is_dir():
            continue
        path = sub / jsonl_name
        if not path.is_file():
            continue
        comp_dir = sub.name
        target_model = compactor_dir_to_out_model.get(comp_dir, comp_dir)
        if target_model not in model_rows:
            continue
        new_rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec = dict(rec)
                rec["condition_id"] = synthetic_condition
                new_rows.append(rec)
        if not new_rows:
            continue
        model_rows[target_model].extend(new_rows)
        counts[target_model] = counts.get(target_model, 0) + len(new_rows)
    return counts


def _extract_model_values(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    level: str,
    metric: str,
) -> np.ndarray:
    vals: list[float] = []
    for r in rows:
        if r.get("condition_id") != condition:
            continue
        if r.get("level") != level:
            continue
        if metric == "accuracy":
            v = ((r.get("metrics") or {}).get("accuracy"))
        else:
            v = r.get("difficulty")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return np.asarray(vals, dtype=np.float64)


def _load_human_values(human_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    by_level_metric: dict[str, dict[str, list[float]]] = {
        lvl: {"accuracy": [], "difficulty": []} for lvl in LEVELS
    }
    for p in sorted(human_dir.glob("run-*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") != "completed":
            continue
        raw_variant = str(data.get("story_variant") or "").strip()
        level = HUMAN_LEVEL_MAP.get(raw_variant)
        if level is None:
            continue
        summary = data.get("summary") or {}
        acc = summary.get("accuracy")
        diff = summary.get("difficultyRating")
        if acc is not None:
            try:
                by_level_metric[level]["accuracy"].append(float(acc))
            except (TypeError, ValueError):
                pass
        if diff is not None:
            try:
                by_level_metric[level]["difficulty"].append(float(diff))
            except (TypeError, ValueError):
                pass
    return {
        lvl: {
            "accuracy": np.asarray(by_level_metric[lvl]["accuracy"], dtype=np.float64),
            "difficulty": np.asarray(by_level_metric[lvl]["difficulty"], dtype=np.float64),
        }
        for lvl in LEVELS
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Reading-QA model-human similarity")
    lines.append("")
    lines.append("| model | metric | condition | level | n_human | n_llm | w1_raw | w1_norm | similarity |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        wr = r["w1_raw"]
        wn = r["w1_norm"]
        sim = r["similarity"]
        lines.append(
            f"| `{r['model']}` | {r['metric']} | {r['condition']} | {r['level']} | "
            f"{r['n_human']} | {r['n_llm']} | "
            f"{'nan' if wr is None else f'{wr:.6f}'} | "
            f"{'nan' if wn is None else f'{wn:.6f}'} | "
            f"{'nan' if sim is None else f'{sim:.6f}'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_xmodel_group_level_subplot_by_condition(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    models: list[str],
    conditions: list[str],
    out_path: Path,
) -> None:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r["metric"] == metric:
            by_key[(r["model"], r["condition"], r["level"])] = r

    n_subplots = max(1, len(conditions))
    fig, axes = plt.subplots(
        nrows=n_subplots,
        ncols=1,
        figsize=(max(12, 1.3 * max(len(models), 1)), max(7.2, 2.8 * n_subplots)),
        sharex=True,
        sharey=True,
    )
    if n_subplots == 1:
        axes = [axes]
    x = np.arange(len(models))
    width = 0.18

    for ax, cond in zip(axes, conditions):
        for i, level in enumerate(LEVELS):
            ys: list[float] = []
            yerr_low: list[float] = []
            yerr_high: list[float] = []
            for model in models:
                rec = by_key.get((model, cond, level), {})
                sim = rec.get("similarity")
                lo = rec.get("similarity_ci_low")
                hi = rec.get("similarity_ci_high")
                if sim is None:
                    ys.append(np.nan)
                    yerr_low.append(np.nan)
                    yerr_high.append(np.nan)
                else:
                    ys.append(float(sim))
                    yerr_low.append(
                        max(0.0, float(sim - lo)) if lo is not None else np.nan
                    )
                    yerr_high.append(
                        max(0.0, float(hi - sim)) if hi is not None else np.nan
                    )
            yerr = np.ma.vstack(
                [
                    np.ma.masked_invalid(np.asarray(yerr_low, dtype=np.float64)),
                    np.ma.masked_invalid(np.asarray(yerr_high, dtype=np.float64)),
                ]
            )
            ax.bar(
                x + (i - (len(LEVELS) - 1) / 2.0) * width,
                ys,
                width=width,
                label=level,
                edgecolor="black",
                yerr=yerr,
                capsize=2,
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Similarity")
        ax.set_title(cond)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, rotation=20, ha="right")
    axes[0].legend(title="Level", ncol=min(4, len(LEVELS)))
    fig.suptitle(f"{metric.title()} similarity | x=model, grouped/color=level, subplots=condition")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_xmodel_group_condition_subplot_by_level(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    models: list[str],
    conditions: list[str],
    out_path: Path,
) -> None:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r["metric"] == metric:
            by_key[(r["model"], r["condition"], r["level"])] = r

    n_subplots = max(1, len(LEVELS))
    fig, axes = plt.subplots(
        nrows=n_subplots,
        ncols=1,
        figsize=(max(12, 1.3 * max(len(models), 1)), max(9.0, 2.5 * n_subplots)),
        sharex=True,
        sharey=True,
    )
    if n_subplots == 1:
        axes = [axes]
    x = np.arange(len(models))
    width = 0.22

    for ax, level in zip(axes, LEVELS):
        for i, cond in enumerate(conditions):
            ys: list[float] = []
            yerr_low: list[float] = []
            yerr_high: list[float] = []
            for model in models:
                rec = by_key.get((model, cond, level), {})
                sim = rec.get("similarity")
                lo = rec.get("similarity_ci_low")
                hi = rec.get("similarity_ci_high")
                if sim is None:
                    ys.append(np.nan)
                    yerr_low.append(np.nan)
                    yerr_high.append(np.nan)
                else:
                    ys.append(float(sim))
                    yerr_low.append(
                        max(0.0, float(sim - lo)) if lo is not None else np.nan
                    )
                    yerr_high.append(
                        max(0.0, float(hi - sim)) if hi is not None else np.nan
                    )
            yerr = np.ma.vstack(
                [
                    np.ma.masked_invalid(np.asarray(yerr_low, dtype=np.float64)),
                    np.ma.masked_invalid(np.asarray(yerr_high, dtype=np.float64)),
                ]
            )
            ax.bar(
                x + (i - (len(conditions) - 1) / 2.0) * width,
                ys,
                width=width,
                label=cond,
                edgecolor="black",
                yerr=yerr,
                capsize=2,
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Similarity")
        ax.set_title(level)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, rotation=20, ha="right")
    axes[0].legend(title="Condition", ncol=min(3, len(conditions)))
    fig.suptitle(f"{metric.title()} similarity | x=model, grouped/color=condition, subplots=level")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _mean_similarity_and_ci_bar(
    rows: list[dict[str, Any]],
    *,
    model: str,
    condition: str,
    metric: str,
) -> tuple[float, float, float]:
    """Mean point similarity across levels, and mean of per-level bootstrap CI bounds."""
    recs = [
        r
        for r in rows
        if r["model"] == model
        and r["condition"] == condition
        and r["metric"] == metric
        and r.get("similarity") is not None
    ]
    if not recs:
        return float("nan"), float("nan"), float("nan")
    y = float(np.mean([float(r["similarity"]) for r in recs]))
    lows = [float(r["similarity_ci_low"]) for r in recs if r.get("similarity_ci_low") is not None]
    highs = [float(r["similarity_ci_high"]) for r in recs if r.get("similarity_ci_high") is not None]
    if not lows or not highs or len(lows) != len(recs) or len(highs) != len(recs):
        lo_bar = float(np.nanmean(lows)) if lows else float("nan")
        hi_bar = float(np.nanmean(highs)) if highs else float("nan")
    else:
        lo_bar = float(np.mean(lows))
        hi_bar = float(np.mean(highs))
    return y, lo_bar, hi_bar


def _draw_similarity_lines_on_ax(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
    metric: str,
    condition_axis_labels: dict[str, str] | None = None,
    cmap: Any | None = None,
    axis_label_fs: int = 17,
    axis_tick_fs: int = 16,
    legend_handles_out: list[Any] | None = None,
    y_axis_label: str = "Similarity",
) -> tuple[float, float]:
    """Draw similarity-vs-condition lines for one ``metric`` on a single axes."""
    label_map = condition_axis_labels or {}
    if cmap is None:
        cmap = plt.get_cmap("tab10")
    n_colors = int(getattr(cmap, "N", 10))
    x = np.arange(len(conditions), dtype=np.float64)
    xtick_labels = [label_map.get(c, c) for c in conditions]
    ymax = 1.0
    ymin = 0.0
    for mi, model in enumerate(models):
        yvals: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        for cond in conditions:
            y, lo_bar, hi_bar = _mean_similarity_and_ci_bar(
                rows, model=model, condition=cond, metric=metric
            )
            yvals.append(y)
            if np.isfinite(y) and np.isfinite(lo_bar) and np.isfinite(hi_bar):
                err_lo.append(max(0.0, y - lo_bar))
                err_hi.append(max(0.0, hi_bar - y))
                ymin = min(ymin, lo_bar)
                ymax = max(ymax, hi_bar)
            else:
                err_lo.append(float("nan"))
                err_hi.append(float("nan"))
        disp = SIMILARITY_LINE_MODEL_DISPLAY_NAMES.get(model, model)
        color_hex = SIMILARITY_LINE_MODEL_DISPLAY_COLORS.get(disp)
        if color_hex is not None:
            color = color_hex
        else:
            color = cmap(mi % n_colors)
        yv = np.asarray(yvals, dtype=np.float64)
        yerr = np.vstack(
            [
                np.asarray(err_lo, dtype=np.float64),
                np.asarray(err_hi, dtype=np.float64),
            ]
        )
        yerr_m = np.ma.masked_invalid(yerr)
        eb = ax.errorbar(
            x,
            yv,
            yerr=yerr_m,
            fmt="-o",
            markersize=5,
            linewidth=1.8,
            color=color,
            ecolor=SIMILARITY_LINE_ERRORBAR_ECOLOR,
            elinewidth=1.05,
            capsize=2.8,
            capthick=1.05,
            label="_nolegend_",
        )
        if legend_handles_out is not None:
            legend_handles_out.append(eb)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(y_axis_label, fontsize=axis_label_fs)
    ax.tick_params(axis="both", labelsize=axis_tick_fs)
    pad = 0.03 + 0.02 * max(0.0, ymax - 1.0)
    ax.set_ylim(max(-0.02, ymin - pad), min(1.02, ymax + pad))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ymin, ymax


def similarity_line_fig_legend(
    fig: plt.Figure,
    handles: list[Any],
    labels: list[str],
    *,
    loc: str = "upper center",
    bbox_to_anchor: tuple[float, float],
    ncol: int = 3,
    fontsize: float | None = None,
    borderaxespad: float = 0.35,
    labelspacing: float | None = None,
    columnspacing: float | None = None,
) -> Any:
    """Shared legend styling for similarity line figures (matches academic reference)."""
    fs = fontsize if fontsize is not None else float(SIMILARITY_LINE_PLOT_RC["legend.fontsize"])
    legend_kw: dict[str, Any] = {
        "loc": loc,
        "bbox_to_anchor": bbox_to_anchor,
        "ncol": ncol,
        "frameon": True,
        "fancybox": True,
        "facecolor": "white",
        "edgecolor": "#CCCCCC",
        "framealpha": 1.0,
        "handlelength": 2.25,
        "handletextpad": 0.65,
        "borderaxespad": borderaxespad,
        "fontsize": fs,
    }
    if labelspacing is not None:
        legend_kw["labelspacing"] = labelspacing
    if columnspacing is not None:
        legend_kw["columnspacing"] = columnspacing
    leg = fig.legend(handles, labels, **legend_kw)
    leg.get_frame().set_linewidth(0.6)
    return leg


def _plot_similarity_lines_by_condition_per_model(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    conditions: list[str],
    out_path: Path,
    metrics: list[str] | None = None,
    condition_axis_labels: dict[str, str] | None = None,
) -> None:
    """Line plot: x = condition, y = mean similarity over article levels.

    One line per model (color-coded). Error bars: asymmetric interval from the mean
    of per-level bootstrap CI bounds (same rows as ``summary.json``), not a joint CI.
    """
    use_metrics = list(metrics) if metrics is not None else list(METRICS)
    if not use_metrics:
        return
    label_map = condition_axis_labels or {}
    n_m = len(use_metrics)
    n_cond = len(conditions)
    # Panel width ~ legend width (3 columns of model names).
    fig_w = max(6.5, 2.0 + 0.55 * float(n_cond))
    fig_h = 3.65
    axis_label_fs = 17
    axis_tick_fs = 16
    with plt.rc_context(SIMILARITY_LINE_PLOT_RC):
        fig, axes = plt.subplots(
            1,
            n_m,
            figsize=(fig_w * n_m, fig_h),
            squeeze=False,
        )
        axes_flat = axes.ravel()
        cmap = plt.get_cmap("tab10")
        legend_handles: list[Any] = []
        legend_labels = [SIMILARITY_LINE_MODEL_DISPLAY_NAMES.get(m, m) for m in models]

        for ax_idx, (ax, metric) in enumerate(zip(axes_flat, use_metrics)):
            _draw_similarity_lines_on_ax(
                ax,
                rows,
                models=models,
                conditions=conditions,
                metric=metric,
                condition_axis_labels=label_map,
                cmap=cmap,
                axis_label_fs=axis_label_fs,
                axis_tick_fs=axis_tick_fs,
                legend_handles_out=legend_handles if ax_idx == 0 else None,
            )

        # tight_layout leaves a bottom band for x-tick labels; place legend just under the axes.
        fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0), pad=0.4)
        similarity_line_fig_legend(
            fig,
            legend_handles,
            legend_labels,
            bbox_to_anchor=(0.5, 0.195),
            ncol=3,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)


def _pooled_llm_values(
    model_rows: dict[str, list[dict[str, Any]]],
    *,
    condition: str,
    level: str,
    metric: str,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for _model, rows in sorted(model_rows.items()):
        parts.append(_extract_model_values(rows, condition=condition, level=level, metric=metric))
    non_empty = [p for p in parts if p.size > 0]
    if not non_empty:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(non_empty)


def _plot_raw_human_vs_llm_xlevel_subplot_condition(
    model_rows: dict[str, list[dict[str, Any]]],
    human_values: dict[str, dict[str, np.ndarray]],
    *,
    conditions: list[str],
    out_path: Path,
    llm_label: str = "LLM (pooled, all models)",
) -> None:
    """One figure: rows = conditions (C1..), cols = accuracy | difficulty.

    Each subplot: x = article level, grouped bars = Human vs pooled LLM (all models).
    Error bars: mean ± SEM (sample std / sqrt(n), ddof=1).
    """
    n_rows = max(1, len(conditions))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(11, max(7.0, 2.6 * n_rows)),
        sharex="col",
    )
    if n_rows == 1:
        axes = np.asarray([axes])
    x = np.arange(len(LEVELS))
    width = 0.36
    colors = {"Human": "#7EB6D4", "LLM": "#E89C6C"}

    for row, cond in enumerate(conditions):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            h_means: list[float] = []
            h_sem: list[float] = []
            m_means: list[float] = []
            m_sem: list[float] = []
            for level in LEVELS:
                h = human_values[level][metric]
                m = _pooled_llm_values(model_rows, condition=cond, level=level, metric=metric)
                pt_h, sem_h = _mean_sem(h)
                pt_m, sem_m = _mean_sem(m)
                h_means.append(pt_h)
                h_sem.append(sem_h)
                m_means.append(pt_m)
                m_sem.append(sem_m)

            yerr_h = np.ma.masked_invalid(np.asarray(h_sem, dtype=np.float64))
            yerr_m = np.ma.masked_invalid(np.asarray(m_sem, dtype=np.float64))
            ax.bar(
                x - width / 2,
                h_means,
                width,
                label="Human",
                color=colors["Human"],
                edgecolor="black",
                yerr=yerr_h,
                capsize=2,
            )
            ax.bar(
                x + width / 2,
                m_means,
                width,
                label=llm_label,
                color=colors["LLM"],
                edgecolor="black",
                yerr=yerr_m,
                capsize=2,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(LEVELS, rotation=15, ha="right")
            ax.set_title(f"{cond} — {metric}")
            ax.grid(axis="y", alpha=0.25)
            if metric == "accuracy":
                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylim(0.5, 10.5)
                ax.set_ylabel("Difficulty (1–10)")

    fig.legend(
        handles=[
            Patch(facecolor=colors["Human"], edgecolor="black", label="Human"),
            Patch(facecolor=colors["LLM"], edgecolor="black", label=llm_label),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(f"Raw scores: Human vs {llm_label} | mean +/- SEM | x=level, grouped=Human/LLM, rows=condition")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare reading_qa model outputs vs human with Wasserstein similarity.")
    ap.add_argument("--llm-out-root", type=Path, default=ROOT / "application" / "out")
    ap.add_argument("--human-dir", type=Path, default=ROOT / "application" / "approved_runs")
    ap.add_argument("--task-name", default="application_reading_qa")
    ap.add_argument("--conditions", default="C1,C2,C3", help="Comma-separated list (default: C1,C2,C3)")
    ap.add_argument("--bootstrap-n", type=int, default=500)
    ap.add_argument("--bootstrap-ci", type=float, default=95.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--compactor-root",
        type=Path,
        default=ROOT / "application" / "compactor",
        help="Directory with per-model compactor runs (default: application/compactor).",
    )
    ap.add_argument(
        "--no-compactor",
        action="store_true",
        help="Do not load compactor JSONL or add the compactor condition.",
    )
    ap.add_argument(
        "--compactor-jsonl",
        default=DEFAULT_COMPACTOR_JSONL,
        help=f"Filename under each compactor model dir (default: {DEFAULT_COMPACTOR_JSONL}).",
    )
    ap.add_argument(
        "--compactor-model-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON object: compactor folder name -> application/out model dir name. "
            "Merged with built-in defaults for known renames."
        ),
    )
    args = ap.parse_args()

    if args.bootstrap_n <= 0:
        raise ValueError("--bootstrap-n must be > 0")
    if not (0.0 < args.bootstrap_ci < 100.0):
        raise ValueError("--bootstrap-ci must be in (0, 100)")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions:
        conditions = list(DEFAULT_CONDITIONS)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or (ROOT / "application" / "comparisons" / f"reading_qa_w1_{stamp}")
    rng = np.random.default_rng(42)

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
        if compactor_counts and not any(c.lower() == COMPACTOR_CONDITION_ID for c in conditions):
            conditions.append(COMPACTOR_CONDITION_ID)
    human_values = _load_human_values(args.human_dir)
    models = sorted(model_rows.keys())

    rows_out: list[dict[str, Any]] = []
    for model in models:
        rows = model_rows[model]
        for condition in conditions:
            for level in LEVELS:
                for metric in METRICS:
                    h = human_values[level][metric]
                    m = _extract_model_values(rows, condition=condition, level=level, metric=metric)
                    w_raw, w_lo, w_hi = bootstrap_w1_ci(
                        h,
                        m,
                        n_boot=args.bootstrap_n,
                        ci=args.bootstrap_ci,
                        rng=rng,
                    )
                    w_norm, w_norm_lo, w_norm_hi = _normalize_w1(
                        w_raw, w_lo, w_hi, SPAN_BY_METRIC[metric]
                    )
                    sim, sim_lo, sim_hi = _sim_from_w1norm(w_norm, w_norm_lo, w_norm_hi)
                    rows_out.append(
                        {
                            "model": model,
                            "condition": condition,
                            "level": level,
                            "metric": metric,
                            "n_human": int(h.size),
                            "n_llm": int(m.size),
                            "w1_raw": _json_safe(w_raw),
                            "w1_raw_ci_low": _json_safe(w_lo),
                            "w1_raw_ci_high": _json_safe(w_hi),
                            "w1_norm": _json_safe(w_norm),
                            "w1_norm_ci_low": _json_safe(w_norm_lo),
                            "w1_norm_ci_high": _json_safe(w_norm_hi),
                            "similarity": _json_safe(sim),
                            "similarity_ci_low": _json_safe(sim_lo),
                            "similarity_ci_high": _json_safe(sim_hi),
                        }
                    )

    summary = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "llm_out_root": str(args.llm_out_root),
            "human_dir": str(args.human_dir),
            "task_name": args.task_name,
            "conditions": conditions,
            "levels": LEVELS,
            "metrics": METRICS,
            "bootstrap_n": int(args.bootstrap_n),
            "bootstrap_ci": float(args.bootstrap_ci),
            "compactor": {
                "loaded": bool(compactor_counts),
                "skipped": bool(args.no_compactor),
                "root": str(args.compactor_root),
                "jsonl": args.compactor_jsonl,
                "rows_merged_by_model": compactor_counts,
            },
        },
        "rows": rows_out,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary_md, rows_out)

    figures_dir = out_dir / "figures"
    _plot_xmodel_group_level_subplot_by_condition(
        rows_out,
        metric="accuracy",
        models=models,
        conditions=conditions,
        out_path=figures_dir / "accuracy_xmodel_groupLevel_colorLevel_subplotCondition.pdf",
    )
    _plot_xmodel_group_condition_subplot_by_level(
        rows_out,
        metric="accuracy",
        models=models,
        conditions=conditions,
        out_path=figures_dir / "accuracy_xmodel_groupCondition_colorCondition_subplotLevel.pdf",
    )
    _plot_xmodel_group_level_subplot_by_condition(
        rows_out,
        metric="difficulty",
        models=models,
        conditions=conditions,
        out_path=figures_dir / "difficulty_xmodel_groupLevel_colorLevel_subplotCondition.pdf",
    )
    _plot_xmodel_group_condition_subplot_by_level(
        rows_out,
        metric="difficulty",
        models=models,
        conditions=conditions,
        out_path=figures_dir / "difficulty_xmodel_groupCondition_colorCondition_subplotLevel.pdf",
    )
    _plot_raw_human_vs_llm_xlevel_subplot_condition(
        model_rows,
        human_values,
        conditions=conditions,
        out_path=figures_dir / "raw_human_vs_llm_xlevel_subplotCondition_accuracy_difficulty.pdf",
        llm_label="LLM (pooled, all models)",
    )
    for model in models:
        _plot_raw_human_vs_llm_xlevel_subplot_condition(
            {model: model_rows[model]},
            human_values,
            conditions=conditions,
            out_path=figures_dir / f"raw_human_vs_{model}_xlevel_subplotCondition_accuracy_difficulty.pdf",
            llm_label=f"LLM ({model})",
        )

    _plot_similarity_lines_by_condition_per_model(
        rows_out,
        models=models,
        conditions=conditions,
        out_path=figures_dir / "similarity_lines_by_model_x_condition.pdf",
        metrics=["accuracy"],
        condition_axis_labels=SIMILARITY_LINE_CONDITION_LABELS,
    )

    print(f"Saved JSON: {summary_json}")
    print(f"Saved MD:   {summary_md}")
    print(f"Saved figs: {figures_dir}")


if __name__ == "__main__":
    main()
