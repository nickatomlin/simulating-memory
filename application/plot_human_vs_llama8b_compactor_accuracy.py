#!/usr/bin/env python3
"""Grouped bars: Human vs Llama 3 8B (compactor WM) reading-QA accuracy.

X-axis = model (Human, Llama 3 8B Compactor); within each model, bars are reading variants
(biography, reading_level, distractor, redundant), each with its own color.

Human data: ``application/approved_runs/run-*.json`` per variant.

Llama: ``by_level[..].accuracy`` from ``wm_application_reading_qa_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.compare_wasserstein_reading_qa import (  # noqa: E402
    LEVELS,
    SIMILARITY_LINE_PLOT_RC,
    _load_human_values,
)

DEFAULT_COMPACTOR_SUMMARY = (
    ROOT / "application" / "compactor" / "meta-llama_llama-3-8b-instruct" / "wm_application_reading_qa_summary.json"
)
# Publication order for x-axis (``LEVELS`` elsewhere uses biography, distractor, reading_level, redundant).
LEVEL_PLOT_ORDER: tuple[str, ...] = ("biography", "reading_level", "distractor", "redundant")
# Distinct fills per reading variant (paired across Human / Llama columns).
LEVEL_COLORS: dict[str, str] = {
    "biography": "#4477AA",
    "reading_level": "#EE6677",
    "distractor": "#228833",
    "redundant": "#CCBB44",
}

MODEL_LABELS: tuple[str, ...] = ("Human", "Llama 3 8B\n(Compactor)")


def _level_xtick(lv: str) -> str:
    return {
        "biography": "Biography",
        "reading_level": "Reading level",
        "distractor": "Distractor",
        "redundant": "Redundant",
    }.get(lv, lv.replace("_", " ").title())


def _bootstrap_ci_mean(arr: np.ndarray, rng: np.random.Generator, *, b: int = 6000) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        m = float(arr[0])
        return m, m
    boots = rng.choice(arr, size=(b, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human-dir", type=Path, default=ROOT / "application" / "approved_runs")
    ap.add_argument(
        "--compactor-summary",
        type=Path,
        default=DEFAULT_COMPACTOR_SUMMARY,
        help="wm_application_reading_qa_summary.json under application/compactor/<model>/",
    )
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap seed for human 95%% CI (per level, salted).")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG base path; sibling PDF saved too. Default: …/human_vs_llama8b_compactor_accuracy_by_level.png",
    )
    args = ap.parse_args()

    hv = _load_human_values(args.human_dir.resolve())
    summary_path = args.compactor_summary.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Missing compactor summary: {summary_path}")
    sj = json.loads(summary_path.read_text(encoding="utf-8"))
    by_lvl = sj.get("by_level")
    if not isinstance(by_lvl, dict):
        raise SystemExit(f"{summary_path}: expected object at key 'by_level'")

    means_h: list[float] = []
    err_low_h: list[float] = []
    err_high_h: list[float] = []
    n_h: list[int] = []

    means_m: list[float] = []
    err_sem_m: list[float] = []
    n_m: list[int] = []

    base_seed = int(args.seed) & 0xFFFFFFFF

    missing_human: list[str] = []

    if set(LEVEL_PLOT_ORDER) != set(LEVELS):
        raise RuntimeError(f"LEVEL_PLOT_ORDER must match LEVELS ({LEVELS}); got {LEVEL_PLOT_ORDER}")

    for i, lv in enumerate(LEVEL_PLOT_ORDER):
        rng_lv = np.random.default_rng(np.random.SeedSequence([base_seed, 0x4C766C + i]))

        raw_h = np.asarray(hv.get(lv, {}).get("accuracy", []), dtype=np.float64).ravel()
        raw_h = raw_h[~np.isnan(raw_h)]

        lvl_obj = by_lvl.get(lv)
        if not isinstance(lvl_obj, dict):
            raise SystemExit(f"{summary_path}: by_level[{lv!r}] missing or not an object")

        try:
            oa_llm = lvl_obj["accuracy"]
            m_ll = float(oa_llm["mean"])
            n_ll = int(oa_llm["n"])
            std_ll = float(oa_llm["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{summary_path}: by_level[{lv!r}] missing accuracy mean/std/n") from exc

        means_m.append(m_ll)
        n_m.append(n_ll)
        err_sem_m.append(1.96 * (std_ll / max(np.sqrt(max(n_ll, 1)), 1e-12)))

        if raw_h.size == 0:
            missing_human.append(lv)
            means_h.append(float("nan"))
            err_low_h.append(float("nan"))
            err_high_h.append(float("nan"))
            n_h.append(0)
            continue

        mean_h = float(np.mean(raw_h))
        lo95, hi95 = _bootstrap_ci_mean(raw_h, rng_lv)
        means_h.append(mean_h)
        err_low_h.append(mean_h - lo95 if np.isfinite(lo95) else float("nan"))
        err_high_h.append(hi95 - mean_h if np.isfinite(hi95) else float("nan"))
        n_h.append(int(raw_h.size))

    if missing_human and all(lv in missing_human for lv in LEVEL_PLOT_ORDER):
        raise SystemExit(f"No human accuracy rows under {args.human_dir} at any level")
    if missing_human:
        print(f"Warn: no human trials for levels: {', '.join(missing_human)}", file=sys.stderr)

    default_out = (
        ROOT
        / "application"
        / "comparisons"
        / "results"
        / "figures"
        / "human_vs_llama8b_compactor_accuracy_by_level.png"
    )
    p = Path(args.out) if args.out is not None else default_out
    if p.suffix.lower() == ".pdf":
        pdf_out = p
        png_out = p.with_suffix(".png")
    else:
        png_out = p
        pdf_out = p.with_suffix(".pdf")

    for i, lv in enumerate(LEVEL_PLOT_ORDER):
        disp = _level_xtick(lv)
        print(f"{disp}: Human mean={means_h[i]:.4f} (n={n_h[i]}, boot 95%); ", end="")
        print(f"Llama mean={means_m[i]:.4f} ±1.96·SEM (n_participants={n_m[i]})")

    mh = np.asarray(means_h, dtype=np.float64)
    mm = np.asarray(means_m, dtype=np.float64)

    x_models = np.arange(len(MODEL_LABELS), dtype=np.float64)
    n_cond = len(LEVEL_PLOT_ORDER)
    group_span = min(0.78, 0.88 / max(n_cond / 4.0, 1e-12))
    bar_w = group_span / n_cond

    midpoints = -(group_span / 2.0) + bar_w / 2.0 + np.arange(n_cond, dtype=np.float64) * bar_w

    with plt.rc_context(SIMILARITY_LINE_PLOT_RC):
        lab_fs = int(SIMILARITY_LINE_PLOT_RC["axes.labelsize"])
        tick_fs = int(SIMILARITY_LINE_PLOT_RC["ytick.labelsize"])
        xt_fs = lab_fs - 1

        fig, ax = plt.subplots(figsize=(6.95, 4.25))

        cap = 2.95
        ekw = {"linewidth": 0.9}

        for j, lv in enumerate(LEVEL_PLOT_ORDER):
            color = LEVEL_COLORS.get(lv, "#888888")
            disp = _level_xtick(lv)
            mh_i, mm_i = float(mh[j]), float(mm[j])
            elo, ehi = float(err_low_h[j]), float(err_high_h[j])
            ese = float(err_sem_m[j])
            xh = float(x_models[0] + midpoints[j])
            xl = float(x_models[1] + midpoints[j])

            if np.isfinite(mh_i) and np.isfinite(elo) and np.isfinite(ehi):
                ax.bar(
                    xh,
                    mh_i,
                    bar_w,
                    label=disp,
                    yerr=np.vstack([[elo], [ehi]]),
                    capsize=cap,
                    color=color,
                    edgecolor="black",
                    linewidth=0.65,
                    ecolor="#404040",
                    error_kw=ekw,
                )

            if np.isfinite(mm_i) and ese >= 0.0:
                ax.bar(
                    xl,
                    mm_i,
                    bar_w,
                    label="_nolegend_",
                    yerr=ese,
                    capsize=cap,
                    color=color,
                    edgecolor="black",
                    linewidth=0.65,
                    ecolor="#404040",
                    error_kw=ekw,
                )

        ax.set_xticks(x_models)
        ax.set_xticklabels(MODEL_LABELS, fontsize=xt_fs)
        ax.set_ylabel("Accuracy", fontsize=lab_fs)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(x_models.min() - 0.53, x_models.max() + 0.53)
        ax.tick_params(axis="y", labelsize=tick_fs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        handles, labels_txt = ax.get_legend_handles_labels()
        uniq: list[tuple[Any, str]] = []
        seen_lab: set[str] = set()
        for hi, lb in zip(handles, labels_txt):
            if lb in ("", "_nolegend_") or lb in seen_lab:
                continue
            seen_lab.add(lb)
            uniq.append((hi, lb))
        leg_obj = None
        if uniq:
            leg_obj = ax.legend(
                [t[0] for t in uniq],
                [t[1] for t in uniq],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.35),
                bbox_transform=ax.transAxes,
                ncol=4,
                columnspacing=1.35,
                frameon=True,
                fancybox=False,
                edgecolor="black",
                facecolor="white",
                fontsize=int(SIMILARITY_LINE_PLOT_RC["legend.fontsize"]) - 1,
            )
            leg_obj.get_frame().set_linewidth(0.75)
            leg_obj.set_clip_on(False)

        fig.subplots_adjust(bottom=0.44)
        png_out.parent.mkdir(parents=True, exist_ok=True)
        extra = [] if leg_obj is None else [leg_obj]
        fig.savefig(png_out, dpi=180, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=extra)
        fig.savefig(pdf_out, format="pdf", bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=extra)
        plt.close(fig)

    print(f"Saved {png_out}")
    print(f"Saved {pdf_out}")


if __name__ == "__main__":
    main()