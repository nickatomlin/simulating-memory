from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List

from .io import write_json


def render_text_table(rows: List[Dict[str, Any]]) -> str:
    """Render a simple fixed-width table for terminal / .txt logs."""
    if not rows:
        return "(no rows)"

    # stable columns: task, n, then sorted metric keys
    metric_keys = sorted({k for r in rows for k in r.keys()} - {"task", "n"})
    cols = ["task", "n"] + metric_keys

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            # keep enough precision but avoid huge width
            return f"{v:.4f}"
        if v is None:
            return ""
        return str(v)

    # compute column widths
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(fmt(r.get(c))))

    def row_line(r: Dict[str, Any]) -> str:
        return " | ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols)

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    lines.extend(row_line(r) for r in rows)
    return "\n".join(lines)


def _normalize_metrics_for_task(task_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(metrics or {})

    if task_name in {"digit_span_forward", "digit_span_reverse"}:
        return {
            "digit_span": m.get("digit_span"),
            "accuracy": m.get("exact"),
        }

    if task_name == "nback":
        return {"accuracy": m.get("acc_over_14")}

    if task_name in {"map_task", "craft_task"}:
        m.pop("total_per_map", None)
        m.pop("total_per_task", None)
        m.pop("correct_mean", None)  # superseded by correct_total + total_questions
        return m

    if task_name == "semantic_story_recall":
        return {
            "embeddingSimilarity": m.get("embeddingSimilarity"),
            "bleuScore": m.get("bleuScore"),
        }

    return m


def build_per_task_rows(task_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build normalized rows for one task (n, condition first)."""
    task_name = str(task_summary.get("task", ""))
    conds = task_summary.get("conditions", []) or []
    rows: List[Dict[str, Any]] = []

    for c in conds:
        base = {
            "n": c.get("n"),
            "condition": c.get("condition"),
        }
        base.update(_normalize_metrics_for_task(task_name, c.get("metrics", {}) or {}))
        rows.append(base)

    return rows


def render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Render a markdown table; keeps n, condition first when present."""
    if not rows:
        return "_(no rows)_"

    metric_keys = sorted({k for r in rows for k in r.keys()} - {"n", "condition"})
    cols = ["n", "condition"] + metric_keys

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def write_metrics_markdown(task_summaries: List[Dict[str, Any]], path: Path) -> None:
    """Write one markdown file containing all task tables."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["# Benchmark Tables", ""]
    for s in task_summaries:
        task_name = str(s.get("task", "unknown_task"))
        rows = build_per_task_rows(s)
        lines.append(f"## {task_name}")
        lines.append("")
        lines.append(render_markdown_table(rows))
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def flatten_metrics(task_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-task summaries into table rows.

    If a task emits per-condition results (preferred), we create 1 row per condition.
    Otherwise we fall back to a single per-task row.
    """
    rows: List[Dict[str, Any]] = []
    for s in task_summaries:
        task = s.get("task")
        conds = s.get("conditions")
        if isinstance(conds, list) and conds:
            for c in conds:
                row = {
                    "task": f"{task}:{c.get('condition')}",
                    "n": c.get("n"),
                }
                metrics = c.get("metrics", {}) or {}
                for k, v in metrics.items():
                    row[k] = v
                rows.append(row)
        else:
            row = {"task": task, "n": s.get("n")}
            metrics = s.get("metrics", {}) or {}
            for k, v in metrics.items():
                row[k] = v
            rows.append(row)
    return rows

def write_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames=[]
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_metrics_html(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("<p>(no rows)</p>", encoding="utf-8")
        return
    # stable columns: task, n, then sorted metric keys
    metric_keys=sorted({k for r in rows for k in r.keys()} - {"task","n"})
    cols=["task","n"] + metric_keys

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    lines=[]
    lines.append("<table border='1' cellpadding='6' cellspacing='0'>")
    lines.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead>")
    lines.append("<tbody>")
    for r in rows:
        lines.append("<tr>" + "".join(f"<td>{fmt(r.get(c))}</td>" for c in cols) + "</tr>")
    lines.append("</tbody></table>")
    path.write_text("\n".join(lines), encoding="utf-8")
