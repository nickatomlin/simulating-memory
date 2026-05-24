from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.io import write_json, JsonlSink
from ..core.llm import LLM
from ..core.parallel import map_participants, resolve_worker_count
from .factual_qa import (
    FORMAT_RULES,
    HUMAN_PROMPT,
    TASK_DESC,
    _rng_for_item_draw,
    load_data,
    parse_answers,
    save_accuracy_by_condition_figure,
    score_doc,
    summarize_doc_rows,
)
from .wm_mcq_common import run_summarizer_mcq_trial, run_wm_mcq_trial
from .wm_prompt_parts import (
    CONDITIONS,
    SUMMARIZER_CONDITIONS,
    summarizer_mcq_recall_preamble,
    summarizer_system_prompt,
    wm_mcq_recall_preamble,
    wm_system_prompts,
)

TASK_NAME = "wm_factual_qa"
TASK_NAME_SUM = "sum_factual_qa"

RECALL_PREAMBLE = wm_mcq_recall_preamble(
    task_prompt=TASK_DESC,
    answer_focus="about the passage you read",
)

WM_SYSTEM_PROMPTS = wm_system_prompts(
    task_prompt=TASK_DESC,
    human_task_prompt=HUMAN_PROMPT,
)

SUM_RECALL_PREAMBLE = summarizer_mcq_recall_preamble(
    task_prompt=TASK_DESC,
    answer_focus="about the passage you read",
)

SUM_SYSTEM_PROMPT = summarizer_system_prompt(task_prompt=TASK_DESC)


def _format_questions(questions: List[Dict[str, Any]]) -> str:
    lines = []
    for i, q in enumerate(questions, start=1):
        choices = q.get("choices", {})
        opts = " ".join(f"{k}) {choices.get(k, '')}" for k in ["A", "B", "C", "D"])
        lines.append(f"Question {i}: {q.get('question', '')} {opts}")
    return "\n".join(lines)


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    data_json_path: str = "data/wikipedia_10docs_questions.json",
    n_participants: int = 1,
    n_docs: Optional[int] = None,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    all_docs = load_data(data_json_path)
    if not all_docs:
        raise ValueError(f"No documents in {data_json_path}.")

    rng = random.Random(stimuli_seed)
    pool = list(all_docs)
    if n_docs is not None:
        pool = rng.sample(pool, min(int(n_docs), len(pool)))

    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(
        n_participants, max_parallel=max_parallel_participants
    )
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in CONDITIONS:

        def _one_participant(pid: int, cond_id: str = cond_id) -> Dict[str, Any]:
            pick_rng = _rng_for_item_draw(stimuli_seed, pid, cond_id)
            doc = pick_rng.choice(pool)
            doc_id = doc.get("doc_id", 0)
            questions = doc.get("questions", [])

            title = doc.get("title", "Reading")
            excerpt = doc.get("excerpt_first_600_words_plus_sentence", "")
            encode_content = f"Title: {title}\n\n{excerpt}"
            questions_text = _format_questions(questions)

            result = run_wm_mcq_trial(
                llm=llm,
                condition_id=cond_id,
                temperature=temperature,
                debug=debug,
                encode_content=encode_content,
                questions_text=questions_text,
                recall_preamble=RECALL_PREAMBLE,
                format_rules=FORMAT_RULES,
                system_prompt_override=WM_SYSTEM_PROMPTS[cond_id],
            )

            answer_map = parse_answers(result["recall_raw"])
            scored = score_doc(questions, answer_map)
            scored["slot_utilization"] = result["slot_utilization"]

            if debug:
                print(f"  {TASK_NAME} | {cond_id} | p{pid} | doc={doc_id}")
                print(f"  answers: {answer_map}")
                print(f"  accuracy: {scored['accuracy']:.2f}")
                print()

            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{pid}:{doc_id}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "repeat_index": pid,
                "doc_id": doc_id,
                "title": title,
                "encoding_log": result["encoding_log"],
                "final_kv": result["final_kv"],
                "recall_raw": result["recall_raw"],
                "questions": questions,
                "metrics": scored,
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        cond_summary = summarize_doc_rows(rows, cond_id, CONDITIONS[cond_id]["name"])
        cond_summary["metrics"]["slot_utilization"] = (
            sum(r["metrics"]["slot_utilization"] for r in rows) / len(rows)
            if rows
            else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_accuracy_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "conditions": cond_summaries,
    }
    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary


def evaluate_summarizer(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    data_json_path: str = "data/wikipedia_10docs_questions.json",
    n_participants: int = 1,
    n_docs: Optional[int] = None,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    all_docs = load_data(data_json_path)
    if not all_docs:
        raise ValueError(f"No documents in {data_json_path}.")

    rng = random.Random(stimuli_seed)
    pool = list(all_docs)
    if n_docs is not None:
        pool = rng.sample(pool, min(int(n_docs), len(pool)))

    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(
        n_participants, max_parallel=max_parallel_participants
    )
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME_SUM}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in SUMMARIZER_CONDITIONS:

        def _one_participant(pid: int, cond_id: str = cond_id) -> Dict[str, Any]:
            pick_rng = _rng_for_item_draw(stimuli_seed, pid, cond_id)
            doc = pick_rng.choice(pool)
            doc_id = doc.get("doc_id", 0)
            questions = doc.get("questions", [])

            title = doc.get("title", "Reading")
            excerpt = doc.get("excerpt_first_600_words_plus_sentence", "")
            encode_content = f"Title: {title}\n\n{excerpt}"
            questions_text = _format_questions(questions)

            result = run_summarizer_mcq_trial(
                llm=llm,
                condition_id=cond_id,
                temperature=temperature,
                debug=debug,
                encode_content=encode_content,
                questions_text=questions_text,
                recall_preamble=SUM_RECALL_PREAMBLE,
                format_rules=FORMAT_RULES,
                system_prompt_override=summarizer_system_prompt(
                    TASK_DESC, condition_id=cond_id
                ),
            )

            answer_map = parse_answers(result["recall_raw"])
            scored = score_doc(questions, answer_map)
            scored["summary_length_words"] = result["summary_length_words"]

            if debug:
                print(f"  {TASK_NAME_SUM} | {cond_id} | p{pid} | doc={doc_id}")
                print(f"  answers: {answer_map}")
                print(f"  accuracy: {scored['accuracy']:.2f}")
                print()

            row = {
                "id": f"{TASK_NAME_SUM}:{cond_id}:p{pid}:{doc_id}",
                "condition_id": cond_id,
                "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
                "repeat_index": pid,
                "doc_id": doc_id,
                "title": title,
                "encoding_log": result["encoding_log"],
                "final_summary": result["final_summary"],
                "recall_raw": result["recall_raw"],
                "questions": questions,
                "metrics": scored,
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        cond_summary = summarize_doc_rows(
            rows, cond_id, SUMMARIZER_CONDITIONS[cond_id]["name"]
        )
        cond_summary["metrics"]["summary_length_words"] = (
            sum(r["metrics"]["summary_length_words"] for r in rows) / len(rows)
            if rows
            else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_accuracy_by_condition_figure(out_dir, TASK_NAME_SUM, cond_summaries)

    summary = {
        "task": TASK_NAME_SUM,
        "n_participants": n_participants,
        "conditions": cond_summaries,
    }
    write_json(out_dir / "tasks" / f"{TASK_NAME_SUM}_summary.json", summary)
    return summary
