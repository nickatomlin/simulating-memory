from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.io import write_json, JsonlSink
from ..core.llm import LLM
from ..core.parallel import map_participants, resolve_worker_count
from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
from ..core.working_memory import MAX_KEYS
from .variable_mapping import (
    HUMAN_PROMPT,
    N_RUNS_PER_PARTICIPANT,
    TASK_DESC,
    TURNS_PER_QUESTION,
    generate_one_run,
    load_json_list,
    parse_answers,
    save_score_by_condition_figure,
    score_run,
    summarize_variable_mapping_rows,
)
from .wm_prompt_parts import (
    CONDITIONS,
    SUMMARIZER_CONDITIONS,
    summarizer_system_prompt,
    wm_system_prompts,
)

TASK_NAME = "wm_variable_mapping"
TASK_NAME_SUM = "sum_variable_mapping"

WM_SYSTEM_PROMPTS = wm_system_prompts(
    task_prompt=TASK_DESC,
    human_task_prompt=HUMAN_PROMPT,
)

ENCODE_PROMPT = """\
New statements:
{statements}

Update your memory as needed."""

QUESTION_PROMPT = """\
Your working memory currently contains:
{wm_contents}

Original task instructions:
{task_prompt}

Based ONLY on the above contents, answer this question:
{question}

Output ONLY one line in this format:
Question {q_idx}: A
No extra text."""

SUM_QUESTION_PROMPT = """\
Original task instructions:
{task_prompt}

Based ONLY on your current summary, answer this question:
{question}

Output ONLY one line in this format:
Question {q_idx}: A
No extra text."""


def _format_questions(questions: List[Dict[str, Any]]) -> str:
    lines = []
    for q in questions:
        idx = q["question_index"]
        name = q["name"]
        options = q["options"]
        opts = " ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(options))
        lines.append(f"Question {idx}: Where does {name} live? {opts}")
    return "\n".join(lines)


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    names_json_path: str = "data/names.json",
    city_json_path: str = "data/city.json",
    n_repeat: int = 1,
    n_runs_per_participant: int = N_RUNS_PER_PARTICIPANT,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    names = load_json_list(names_json_path, "names")
    cities = load_json_list(city_json_path, "cities")
    n_participants = max(1, int(n_repeat))
    n_runs_per_participant = max(1, int(n_runs_per_participant))
    total_runs = n_participants * n_runs_per_participant

    rng = random.Random(stimuli_seed)
    runs = []
    for r in range(total_runs):
        run = generate_one_run(names, cities, rng)
        runs.append({"repeat_index": r + 1, **run})

    workers = resolve_worker_count(total_runs, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in CONDITIONS:

        def _one_run(i: int, cond_id: str = cond_id) -> Dict[str, Any]:
            run = runs[i]
            assignments = run["assignments"]
            questions = run["questions"]

            agent = WorkingMemoryAgent(
                llm=llm,
                condition_id=cond_id,
                temperature=temperature,
                debug=debug,
                system_prompt_override=WM_SYSTEM_PROMPTS[cond_id],
            )

            if debug:
                print(f"\n{'=' * 50}")
                print(f"{TASK_NAME} | {cond_id} | run={run['repeat_index']}")
                print(f"{'=' * 50}")

            # Incremental encoding: present TURNS_PER_QUESTION statements at a time,
            # then ask the question for that interval — matching the human task structure.
            answer_map: Dict[int, str] = {}
            step_logs: List[Dict] = []

            for q_idx, q in enumerate(questions):
                # Present the next TURNS_PER_QUESTION statements
                start = q_idx * TURNS_PER_QUESTION
                end = start + TURNS_PER_QUESTION
                statement_block = "\n".join(
                    a["statement"] for a in assignments[start:end]
                )
                encode_msg = ENCODE_PROMPT.format(statements=statement_block)
                agent.step(encode_msg, allow_tools=True)

                # Ask the question — model answers from KV store only
                q_text = _format_questions([q])
                question_msg = QUESTION_PROMPT.format(
                    wm_contents=agent.wm.to_recall_text(),
                    task_prompt=TASK_DESC.strip(),
                    question=q_text,
                    q_idx=q["question_index"],
                )
                answer_raw = agent.step(question_msg, allow_tools=False)
                parsed = parse_answers(answer_raw)
                answer_map.update(parsed)

                step_logs.append(
                    {
                        "q_idx": q["question_index"],
                        "statements": statement_block,
                        "kv_at_question": agent.wm.store.copy(),
                        "answer_raw": answer_raw,
                        "parsed": parsed,
                    }
                )

                if debug:
                    print(
                        f"  Q{q['question_index']}: {q['name']} | "
                        f"kv={agent.wm.store} | answer={parsed}"
                    )

            scored = score_run(questions, answer_map)
            scored["slot_utilization"] = len(agent.wm.store) / MAX_KEYS

            if debug:
                print(f"  final score: {scored['score']}")
                print()

            participant_id = i // n_runs_per_participant
            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{participant_id}:run{i % n_runs_per_participant}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "participant_id": participant_id,
                "run_index": i,
                "repeat_index": run["repeat_index"],
                "step_logs": step_logs,
                "final_kv": agent.wm.store,
                "assignments": assignments,
                "questions": questions,
                "parsed_answers": answer_map,
                "metrics": scored,
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(len(runs))),
            _one_run,
            max_workers=workers,
        )
        cond_summary = summarize_variable_mapping_rows(
            rows,
            cond_id,
            CONDITIONS[cond_id]["name"],
            n_participants=n_participants,
            n_runs_per_participant=n_runs_per_participant,
        )
        cond_summary["metrics"]["slot_utilization"] = (
            sum(r["metrics"]["slot_utilization"] for r in rows) / len(rows)
            if rows
            else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_score_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "n_runs_per_participant": n_runs_per_participant,
        "conditions": cond_summaries,
    }
    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary


def evaluate_summarizer(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    names_json_path: str = "data/names.json",
    city_json_path: str = "data/city.json",
    n_repeat: int = 1,
    n_runs_per_participant: int = N_RUNS_PER_PARTICIPANT,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    names = load_json_list(names_json_path, "names")
    cities = load_json_list(city_json_path, "cities")
    n_participants = max(1, int(n_repeat))
    n_runs_per_participant = max(1, int(n_runs_per_participant))
    total_runs = n_participants * n_runs_per_participant

    rng = random.Random(stimuli_seed)
    runs = []
    for r in range(total_runs):
        run = generate_one_run(names, cities, rng)
        runs.append({"repeat_index": r + 1, **run})

    workers = resolve_worker_count(total_runs, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME_SUM}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in SUMMARIZER_CONDITIONS:

        def _one_run(i: int, cond_id: str = cond_id) -> Dict[str, Any]:
            run = runs[i]
            assignments = run["assignments"]
            questions = run["questions"]

            agent = SummarizerAgent(
                llm=llm,
                condition_id=cond_id,
                temperature=temperature,
                debug=debug,
                system_prompt_override=summarizer_system_prompt(
                    TASK_DESC, condition_id=cond_id
                ),
            )

            if debug:
                print(f"\n{'=' * 50}")
                print(f"{TASK_NAME_SUM} | {cond_id} | run={run['repeat_index']}")
                print(f"{'=' * 50}")

            answer_map: Dict[int, str] = {}
            step_logs: List[Dict[str, Any]] = []

            for q_idx, q in enumerate(questions):
                start = q_idx * TURNS_PER_QUESTION
                end = start + TURNS_PER_QUESTION
                statement_block = "\n".join(
                    a["statement"] for a in assignments[start:end]
                )
                encode_msg = ENCODE_PROMPT.format(statements=statement_block)
                agent.step(encode_msg)

                q_text = _format_questions([q])
                question_msg = SUM_QUESTION_PROMPT.format(
                    task_prompt=TASK_DESC.strip(),
                    question=q_text,
                    q_idx=q["question_index"],
                )
                answer_raw = agent.step(question_msg)
                parsed = parse_answers(answer_raw)
                answer_map.update(parsed)

                step_logs.append(
                    {
                        "q_idx": q["question_index"],
                        "statements": statement_block,
                        "summary_at_question": agent.summary,
                        "answer_raw": answer_raw,
                        "parsed": parsed,
                    }
                )

                if debug:
                    print(
                        f"  Q{q['question_index']}: {q['name']} | "
                        f"summary={agent.summary!r} | answer={parsed}"
                    )

            scored = score_run(questions, answer_map)
            scored["summary_length_words"] = agent.summary_length_words()

            if debug:
                print(f"  final score: {scored['score']}")
                print()

            participant_id = i // n_runs_per_participant
            row = {
                "id": f"{TASK_NAME_SUM}:{cond_id}:p{participant_id}:run{i % n_runs_per_participant}",
                "condition_id": cond_id,
                "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
                "participant_id": participant_id,
                "run_index": i,
                "repeat_index": run["repeat_index"],
                "step_logs": step_logs,
                "agent_step_log": agent.get_step_log(),
                "final_summary": agent.summary,
                "assignments": assignments,
                "questions": questions,
                "parsed_answers": answer_map,
                "metrics": scored,
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(len(runs))),
            _one_run,
            max_workers=workers,
        )
        cond_summary = summarize_variable_mapping_rows(
            rows,
            cond_id,
            SUMMARIZER_CONDITIONS[cond_id]["name"],
            n_participants=n_participants,
            n_runs_per_participant=n_runs_per_participant,
        )
        cond_summary["metrics"]["summary_length_words"] = (
            sum(r["metrics"]["summary_length_words"] for r in rows) / len(rows)
            if rows
            else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_score_by_condition_figure(out_dir, TASK_NAME_SUM, cond_summaries)

    summary = {
        "task": TASK_NAME_SUM,
        "n_participants": n_participants,
        "n_runs_per_participant": n_runs_per_participant,
        "conditions": cond_summaries,
    }
    write_json(out_dir / "tasks" / f"{TASK_NAME_SUM}_summary.json", summary)
    return summary
