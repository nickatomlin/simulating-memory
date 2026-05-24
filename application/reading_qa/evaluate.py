from __future__ import annotations

import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from bench.core.io import write_json, write_jsonl
from bench.core.llm import LLM
from bench.core.parallel import map_participants, resolve_worker_count

from .data import LEVELS, Document, load_documents
from .prompting import CONDITIONS, build_prompt, parse_answers_and_difficulty


TASK_NAME = "application_reading_qa"


def _score(doc: Document, answers: Dict[int, str]) -> Dict[str, Any]:
    correct = 0
    total = len(doc.questions)
    for i, q in enumerate(doc.questions, start=1):
        if answers.get(i) == q.answer:
            correct += 1
    acc = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": acc}


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _difficulty_histogram(values: List[int]) -> Dict[str, int]:
    hist = {str(k): 0 for k in range(1, 11)}
    for val in values:
        if 1 <= val <= 10:
            hist[str(val)] += 1
    return hist


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc_values = [float(r["metrics"]["accuracy"]) for r in rows]
    diff_values = [int(r["difficulty"]) for r in rows if isinstance(r.get("difficulty"), int)]
    return {
        "n": len(rows),
        "accuracy": _summary_stats(acc_values),
        "difficulty": {
            **_summary_stats([float(v) for v in diff_values]),
            "valid_n": len(diff_values),
            "histogram_1_to_10": _difficulty_histogram(diff_values),
        },
    }


def _summarize_condition_level(rows: List[Dict[str, Any]], condition_ids: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cond_id in condition_ids:
        out[cond_id] = {}
        for level in LEVELS:
            cell_rows = [r for r in rows if r["condition_id"] == cond_id and r["level"] == level]
            out[cond_id][level] = _summarize_rows(cell_rows)
    return out


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    documents_dir: Path,
    n_repeat: int = 1,
    seed: int = 42,
    max_tokens: Optional[int] = None,
    max_parallel_repeats: Optional[int] = None,
) -> Dict[str, Any]:
    documents = load_documents(documents_dir)
    rng = random.Random(seed)
    repeats = max(1, int(n_repeat))
    condition_ids = ["C1", "C2", "C3", "C4"]
    job_count = repeats * len(condition_ids)
    workers = resolve_worker_count(job_count, max_parallel=max_parallel_repeats)
    # Pre-sample in the main thread so sampling remains reproducible.
    sampled_pairs: Dict[tuple[int, str], tuple[Document, str]] = {}
    for repeat_index in range(1, repeats + 1):
        for cond_id in condition_ids:
            sampled_pairs[(repeat_index, cond_id)] = (rng.choice(documents), rng.choice(LEVELS))

    jobs = [(repeat_index, cond_id) for repeat_index in range(1, repeats + 1) for cond_id in condition_ids]

    def _run_one_job(job: tuple[int, str]) -> Dict[str, Any]:
        repeat_index, cond_id = job
        doc, level = sampled_pairs[(repeat_index, cond_id)]
        reading_text = doc.levels[level]
        prompt = build_prompt(cond_id, reading_text, doc.questions)

        resp = llm.generate(
            prompt,
            temperature=float(model_cfg.get("temperature", 0.0)),
            max_tokens=int(max_tokens or model_cfg.get("max_tokens", 2048)),
            top_p=float(model_cfg.get("top_p", 1.0)),
            seed=model_cfg.get("seed"),
        )
        parsed = parse_answers_and_difficulty(resp.text)
        answers = parsed["answers"]
        difficulty = parsed["difficulty"]
        parse_errors = parsed["parse_errors"]
        metrics = _score(doc, answers)

        return {
            "id": f"{TASK_NAME}:{cond_id}:r{repeat_index}:{doc.doc_id}:{level}",
            "condition_id": cond_id,
            "condition_name": CONDITIONS[cond_id]["name"],
            "repeat_index": repeat_index,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "level": level,
            "prompt": prompt,
            "raw_response": resp.text,
            "parsed_answers": {str(k): v for k, v in answers.items()},
            "difficulty": difficulty,
            "parse_errors": parse_errors,
            "metrics": metrics,
        }

    rows = map_participants(
        jobs,
        _run_one_job,
        max_workers=workers,
    )

    by_level: Dict[str, Any] = {}
    for level in LEVELS:
        level_rows = [r for r in rows if r["level"] == level]
        by_level[level] = _summarize_rows(level_rows)

    condition_summaries: List[Dict[str, Any]] = []
    by_condition: Dict[str, Any] = {}
    for cond_id in condition_ids:
        cond_rows = [r for r in rows if r["condition_id"] == cond_id]
        cond_summary = {
            "condition": cond_id,
            "condition_name": CONDITIONS[cond_id]["name"],
            **_summarize_rows(cond_rows),
        }
        condition_summaries.append(cond_summary)
        by_condition[cond_id] = cond_summary

    by_condition_level = _summarize_condition_level(rows, condition_ids)
    by_condition_level_c123 = _summarize_condition_level(rows, ["C1", "C2", "C3"])

    summary = {
        "task": TASK_NAME,
        "n_repeat": repeats,
        "seed": int(seed),
        "parallel_workers": int(workers),
        "documents_dir": str(documents_dir),
        "conditions": condition_summaries,
        "overall": _summarize_rows(rows),
        "by_condition": by_condition,
        "by_level": by_level,
        "by_condition_level": by_condition_level,
        "by_condition_level_c123": by_condition_level_c123,
    }

    write_jsonl(out_dir / "tasks" / f"{TASK_NAME}.jsonl", rows)
    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
