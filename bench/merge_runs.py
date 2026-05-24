"""Combine raw data from multiple runs into a single merged run.

Concatenates per-task ``tasks/*.jsonl`` files and recomputes
``tasks/*_summary.json`` and the top-level ``summary.json``.
Figures are not regenerated.

Usage (CLI)::

    python -m bench.merge_runs <dest> <src1> <src2> [<src3> ...]

Each source/destination should point at the model directory inside a run,
e.g. ``runs/gpt-test/gpt-5.4``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .core.io import read_json, write_json, write_jsonl
from .tasks.craft_task import summarize_craft_rows
from .tasks.factual_qa import summarize_doc_rows
from .tasks.map_task import summarize_map_rows
from .tasks.narrative_qa import summarize_story_rows
from .tasks.nback import summarize_nback_condition
from .tasks.variable_mapping import summarize_variable_mapping_rows
from .tasks.word_recognition import summarize_game_rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _mean_optional(vals: List[Any]) -> Optional[float]:
    nums = [float(v) for v in vals if v is not None]
    return float(sum(nums) / len(nums)) if nums else None


def _row_cond(row: Dict[str, Any]) -> Optional[str]:
    return row.get("condition_id") or row.get("condition")


REMAPPABLE_KEYS = ("participant_id", "repeat_index", "run_index")


def _next_offset_delta(rows: List[Dict[str, Any]], key: str) -> int:
    """How much to bump the offset after consuming this source.

    Uses ``max + 1`` to ensure zero-indexed ids from the next source don't
    collide with this source's ids. Returns 0 if the key never appears.
    """
    vals = [int(r[key]) for r in rows if key in r and r[key] is not None]
    return (max(vals) + 1) if vals else 0


def _participant_count(rows: List[Dict[str, Any]]) -> int:
    """Best-effort count of distinct participants in a row set."""
    pids = {int(r["participant_id"]) for r in rows if "participant_id" in r}
    if pids:
        return len(pids)
    reps = {int(r["repeat_index"]) for r in rows if "repeat_index" in r}
    return len(reps) if reps else 1


def _remap_row(row: Dict[str, Any], offsets: Dict[str, int]) -> Dict[str, Any]:
    if not any(offsets.values()):
        return row
    r = dict(row)
    pid_old = int(r["participant_id"]) if "participant_id" in r else None
    for k, off in offsets.items():
        if off and k in r and r[k] is not None:
            r[k] = int(r[k]) + off
    if pid_old is not None and offsets.get("participant_id"):
        rid = r.get("id")
        if isinstance(rid, str):
            new = pid_old + offsets["participant_id"]
            r["id"] = rid.replace(f":p{pid_old}:", f":p{new}:")
    return r


def _remap_rows(
    rows: List[Dict[str, Any]], offsets: Dict[str, int]
) -> List[Dict[str, Any]]:
    return [_remap_row(r, offsets) for r in rows]


# ---------------------------------------------------------------------------
# Task-specific aggregators
# ---------------------------------------------------------------------------


def _digit_span_conditions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recompute digit-span style summary (forward + reverse).

    digit_span per condition = mean over participants of the highest span
    where any trial was exactly correct. Other metrics are simple means
    over all rows.
    """
    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cond_names: Dict[str, str] = {}
    for r in rows:
        cid = _row_cond(r)
        if cid is None:
            continue
        by_cond[cid].append(r)
        cond_names.setdefault(cid, r.get("condition_name", cid))

    cond_summaries: List[Dict[str, Any]] = []
    for cid in sorted(by_cond.keys()):
        crows = by_cond[cid]
        # group by participant_id
        by_pid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in crows:
            by_pid[int(r.get("participant_id", 1))].append(r)

        per_participant_span: List[float] = []
        per_participant_curve: List[Dict[Any, float]] = []
        for pid, prows in by_pid.items():
            by_span_exact: Dict[int, List[float]] = defaultdict(list)
            for r in prows:
                span = int(r["span_length"])
                ex = float(r["metrics"]["exact"])
                by_span_exact[span].append(ex)
            # digit_span = max contiguous span where any_correct from min span up
            spans_sorted = sorted(by_span_exact.keys())
            digit_span = 0
            for span in spans_sorted:
                if any(v == 1.0 for v in by_span_exact[span]):
                    digit_span = span
                else:
                    break
            per_participant_span.append(float(digit_span))
            per_participant_curve.append(
                {s: _mean(by_span_exact[s]) for s in spans_sorted}
            )

        # merge per-span across participants
        merged: Dict[Any, List[float]] = defaultdict(list)
        for d in per_participant_curve:
            for k, v in d.items():
                merged[k].append(float(v))
        merged_curve = {str(k): _mean(merged[k]) for k in sorted(merged.keys())}

        # row-level metric means
        metrics_keys = set()
        for r in crows:
            metrics_keys.update(r.get("metrics", {}).keys())
        agg_metrics: Dict[str, float] = {"digit_span": _mean(per_participant_span)}
        for k in metrics_keys:
            vals = [
                float(r["metrics"][k])
                for r in crows
                if k in r.get("metrics", {})
            ]
            agg_metrics[k] = _mean(vals)

        cond_summaries.append(
            {
                "condition": cid,
                "condition_name": cond_names[cid],
                "n": len(by_pid),
                "metrics": agg_metrics,
                "breakdown": {"exact_by_span": merged_curve},
            }
        )
    return cond_summaries


def _story_recall_conditions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cond_names: Dict[str, str] = {}
    for r in rows:
        cid = _row_cond(r)
        if cid is None:
            continue
        by_cond[cid].append(r)
        cond_names.setdefault(cid, r.get("condition_name", cid))

    out: List[Dict[str, Any]] = []
    for cid in sorted(by_cond.keys()):
        crows = by_cond[cid]
        cov = [float(r["metrics"].get("coverage", 1.0)) for r in crows]
        fai = [float(r["metrics"].get("faithfulness", 1.0)) for r in crows]
        hal = [float(r["metrics"].get("hallucination", 5.0)) for r in crows]
        util = [float(r["metrics"].get("slot_utilization", 0.0)) for r in crows]
        sims = [r["metrics"].get("embeddingSimilarity") for r in crows]
        bleus = [r["metrics"].get("bleuScore") for r in crows]
        out.append(
            {
                "condition": cid,
                "condition_name": cond_names[cid],
                "n": len(crows),
                "metrics": {
                    "embeddingSimilarity": _mean_optional(sims),
                    "bleuScore": _mean_optional(bleus),
                    "coverage": _mean(cov),
                    "faithfulness": _mean(fai),
                    "hallucination": _mean(hal),
                    "slot_utilization": _mean(util),
                },
            }
        )
    return out


def _add_slot_utilization(
    cond_summary: Dict[str, Any], rows: List[Dict[str, Any]]
) -> None:
    util_vals: List[float] = []
    for r in rows:
        # nback exposes slot_utilization at top level; others inside metrics
        if "slot_utilization" in r and not isinstance(r.get("metrics"), dict):
            util_vals.append(float(r["slot_utilization"]))
        elif isinstance(r.get("metrics"), dict) and "slot_utilization" in r["metrics"]:
            util_vals.append(float(r["metrics"]["slot_utilization"]))
        elif "slot_utilization" in r:
            util_vals.append(float(r["slot_utilization"]))
    if util_vals:
        cond_summary["metrics"]["slot_utilization"] = _mean(util_vals)


def _by_condition(
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cond_names: Dict[str, str] = {}
    for r in rows:
        cid = _row_cond(r)
        if cid is None:
            continue
        by_cond[cid].append(r)
        cond_names.setdefault(cid, r.get("condition_name", cid))
    return by_cond, cond_names


def _generic_conditions(
    rows: List[Dict[str, Any]],
    summarizer,
    *,
    extra_kwargs_fn=None,
) -> List[Dict[str, Any]]:
    by_cond, cond_names = _by_condition(rows)
    out: List[Dict[str, Any]] = []
    for cid in sorted(by_cond.keys()):
        crows = by_cond[cid]
        kwargs = extra_kwargs_fn(crows) if extra_kwargs_fn else {}
        cs = summarizer(crows, cid, cond_names[cid], **kwargs)
        _add_slot_utilization(cs, crows)
        out.append(cs)
    return out


def _nback_meta(rows: List[Dict[str, Any]], prev_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(prev_meta) if prev_meta else {}
    meta["total_records"] = len(rows)
    pids = {int(r.get("participant_id", 0)) for r in rows}
    if pids:
        meta["n_repeat"] = len(pids)
    return meta


def _variable_mapping_kwargs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_participants = _participant_count(rows)
    n_runs_per_participant = max(1, len(rows) // n_participants) if n_participants else len(rows)
    return {
        "n_participants": n_participants,
        "n_runs_per_participant": n_runs_per_participant,
    }


# Map task name -> aggregator. Both wm_ and non-wm variants share row schemas.
def _aggregate_task(
    task_name: str,
    rows: List[Dict[str, Any]],
    prior_summary: Dict[str, Any],
) -> Dict[str, Any]:
    base = task_name.removeprefix("wm_") if task_name.startswith("wm_") else task_name

    if base in {"digit_span_forward", "digit_span_reverse"}:
        conditions = _digit_span_conditions(rows)
    elif base == "semantic_story_recall":
        conditions = _story_recall_conditions(rows)
    elif base == "nback":
        conditions = _generic_conditions(rows, summarize_nback_condition)
    elif base == "word_recognition":
        conditions = _generic_conditions(rows, summarize_game_rows)
    elif base == "variable_mapping":
        conditions = _generic_conditions(
            rows,
            summarize_variable_mapping_rows,
            extra_kwargs_fn=_variable_mapping_kwargs,
        )
    elif base == "factual_qa":
        conditions = _generic_conditions(rows, summarize_doc_rows)
    elif base == "narrative_qa":
        conditions = _generic_conditions(rows, summarize_story_rows)
    elif base == "map_task":
        conditions = _generic_conditions(rows, summarize_map_rows)
    elif base == "craft_task":
        conditions = _generic_conditions(rows, summarize_craft_rows)
    else:
        # Fallback: mean of every numeric metric, grouped by condition
        by_cond, cond_names = _by_condition(rows)
        conditions = []
        for cid in sorted(by_cond.keys()):
            crows = by_cond[cid]
            keys = set()
            for r in crows:
                keys.update((r.get("metrics") or {}).keys())
            agg = {
                k: _mean(
                    [
                        float(r["metrics"][k])
                        for r in crows
                        if k in (r.get("metrics") or {})
                        and isinstance(r["metrics"][k], (int, float))
                    ]
                )
                for k in keys
            }
            conditions.append(
                {
                    "condition": cid,
                    "condition_name": cond_names[cid],
                    "n": len(crows),
                    "metrics": agg,
                }
            )

    # Build new summary, preserving task-level scalar fields from prior summary.
    new_summary: Dict[str, Any] = {"task": task_name}
    n_participants = _participant_count(rows)

    if "n_participants" in prior_summary:
        new_summary["n_participants"] = n_participants
    if "stimuli_seed_base" in prior_summary:
        new_summary["stimuli_seed_base"] = prior_summary["stimuli_seed_base"]
    if "participant_seeds" in prior_summary:
        # Best-effort: extend seeds with the reconstructed list from rows.
        seeds: List[int] = []
        seen: set = set()
        for r in rows:
            pid = int(r.get("participant_id", 0))
            seed = r.get("stimuli_seed")
            if pid not in seen and seed is not None:
                seeds.append(int(seed))
                seen.add(pid)
        new_summary["participant_seeds"] = seeds or prior_summary["participant_seeds"]
    if "repeats_per_stimulus" in prior_summary:
        new_summary["repeats_per_stimulus"] = prior_summary["repeats_per_stimulus"]
    if "n_runs_per_participant" in prior_summary:
        # Recompute from rows when possible.
        if n_participants:
            new_summary["n_runs_per_participant"] = max(1, len(rows) // n_participants)
        else:
            new_summary["n_runs_per_participant"] = prior_summary["n_runs_per_participant"]
    if "n_maps_per_participant" in prior_summary:
        new_summary["n_maps_per_participant"] = prior_summary["n_maps_per_participant"]
    if "n_tasks_per_participant" in prior_summary:
        new_summary["n_tasks_per_participant"] = prior_summary["n_tasks_per_participant"]
    if "meta" in prior_summary:
        new_summary["meta"] = _nback_meta(rows, prior_summary["meta"])

    new_summary["conditions"] = conditions
    return new_summary


# ---------------------------------------------------------------------------
# Top-level merge
# ---------------------------------------------------------------------------


def merge_runs(sources: List[Path], dest: Path) -> Dict[str, Any]:
    if len(sources) < 2:
        raise ValueError("Need at least two source runs to merge.")
    for s in sources:
        if not (s / "tasks").is_dir():
            raise FileNotFoundError(f"No tasks/ dir in source: {s}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tasks").mkdir(parents=True, exist_ok=True)

    # 1. discover task names from first source
    task_files = sorted((sources[0] / "tasks").glob("*.jsonl"))
    task_names = [p.stem for p in task_files]

    merged_task_summaries: Dict[str, Dict[str, Any]] = {}

    for task in task_names:
        # collect rows across sources, remapping per-source offsets so rows
        # from different sources stay distinguishable.
        merged_rows: List[Dict[str, Any]] = []
        offsets: Dict[str, int] = {k: 0 for k in REMAPPABLE_KEYS}
        for src in sources:
            jsonl_path = src / "tasks" / f"{task}.jsonl"
            if not jsonl_path.exists():
                print(f"  [warn] {src.name}: missing {task}.jsonl, skipping")
                continue
            src_rows = _read_jsonl(jsonl_path)
            merged_rows.extend(_remap_rows(src_rows, offsets))
            for k in REMAPPABLE_KEYS:
                offsets[k] += _next_offset_delta(src_rows, k)

        # use the first source's summary as a template for task-level fields
        prior_summary_path = sources[0] / "tasks" / f"{task}_summary.json"
        prior_summary = (
            read_json(prior_summary_path) if prior_summary_path.exists() else {}
        )

        new_summary = _aggregate_task(task, merged_rows, prior_summary)

        # write outputs
        write_jsonl(dest / "tasks" / f"{task}.jsonl", merged_rows)
        write_json(dest / "tasks" / f"{task}_summary.json", new_summary)
        merged_task_summaries[task] = new_summary
        print(f"  merged {task}: {len(merged_rows)} rows")

    # 2. top-level summary.json — preserve order from the first source's summary.
    top_summary_path = sources[0] / "summary.json"
    if top_summary_path.exists():
        first_summary = read_json(top_summary_path)
        ordered_tasks = [
            t["task"] for t in first_summary.get("tasks", []) if t.get("task")
        ]
    else:
        ordered_tasks = task_names
    seen = set(ordered_tasks)
    for t in task_names:
        if t not in seen:
            ordered_tasks.append(t)

    top: Dict[str, Any] = {
        "tasks": [
            merged_task_summaries[t]
            for t in ordered_tasks
            if t in merged_task_summaries
        ]
    }
    write_json(dest / "summary.json", top)

    # 3. preserve config_snapshot.json from first source, plus a merge record.
    cs_src = sources[0] / "config_snapshot.json"
    if cs_src.exists():
        shutil.copy2(cs_src, dest / "config_snapshot.json")
    write_json(
        dest / "merge_info.json",
        {
            "sources": [str(s) for s in sources],
            "task_names": task_names,
        },
    )

    return top


def main() -> None:
    p = argparse.ArgumentParser(
        description="Combine multiple run directories into one merged run."
    )
    p.add_argument("dest", type=Path, help="Destination model dir, e.g. runs/merged/gpt-5.4")
    p.add_argument("sources", type=Path, nargs="+", help="Source model dirs to merge")
    args = p.parse_args()

    if len(args.sources) < 2:
        p.error("Pass at least two source directories.")

    print(f"Merging {len(args.sources)} runs into {args.dest}")
    merge_runs(args.sources, args.dest)
    print("Done.")


if __name__ == "__main__":
    main()
