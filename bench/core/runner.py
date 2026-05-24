from __future__ import annotations
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional
from .io import write_json


def run_tasks(
    task_fns: List[Callable[[], Dict[str, Any]]],
    out_dir: Path,
    *,
    max_parallel_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    if max_parallel_tasks is None or max_parallel_tasks <= 1:
        return _run_sequential(task_fns, out_dir)
    return _run_parallel(task_fns, out_dir, max_workers=max_parallel_tasks)


def _run_sequential(task_fns, out_dir):
    summaries = []
    total = len(task_fns)
    for i, fn in enumerate(task_fns, start=1):
        label = getattr(fn, "task_name", None) or getattr(fn, "__name__", "<task>")
        if i > 1:
            sys.stderr.write("\n")
            sys.stderr.flush()
        print(f"\n▶ [{i}/{total}] Start: {label}")
        summaries.append(fn())
        print(f"✔ [{i}/{total}] Done:  {label}")
        write_json(out_dir / "summary.json", {"tasks": summaries})
    return {"tasks": summaries}


def _run_parallel(task_fns, out_dir, *, max_workers):
    total = len(task_fns)
    workers = min(total, max_workers)
    # Map future → (original index, label) for ordered output
    summaries = [None] * total
    lock = threading.Lock()
    completed = [0]

    def _run_one(idx_fn):
        idx, fn = idx_fn
        label = getattr(fn, "task_name", None) or getattr(fn, "__name__", "<task>")
        print(f"▶ [parallel] Start: {label}")
        result = fn()
        with lock:
            summaries[idx] = result
            completed[0] += 1
            n = completed[0]
            print(f"✔ [{n}/{total}] Done:  {label}")
            # Incremental save (only completed slots)
            done = [s for s in summaries if s is not None]
            write_json(out_dir / "summary.json", {"tasks": done})
        return result

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_run_one, enumerate(task_fns)))

    return {"tasks": summaries}
