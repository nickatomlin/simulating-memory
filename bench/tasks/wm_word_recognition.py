from __future__ import annotations
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count
from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
from ..core.working_memory import MAX_KEYS
from .word_recognition import (
    FORMAT_RULES,
    HUMAN_PROMPT,
    TASK_DESC,
    load_words,
    generate_one_game,
    parse_responses,
    save_score_by_condition_figure,
    score_game,
    summarize_game_rows,
)
from .wm_prompt_parts import (
    CONDITIONS,
    SUMMARIZER_CONDITIONS,
    summarizer_recall_prompt,
    summarizer_system_prompt,
    wm_recall_prompt,
    wm_system_prompts,
)

TASK_NAME = "wm_word_recognition"
TASK_NAME_SUM = "sum_word_recognition"

WM_SYSTEM_PROMPTS = wm_system_prompts(
    task_prompt=TASK_DESC,
    human_task_prompt=HUMAN_PROMPT,
)

RECALL_PROMPT = wm_recall_prompt(
    task_prompt=TASK_DESC,
    wm_recall_instructions=(
        'classify each word below as "Old" (appeared earlier in the original list) or '
        '"New" (first appearance at that position in the list).\n\n{trials_text}'
    ),
    format_rules=FORMAT_RULES,
)

SUM_SYSTEM_PROMPT = summarizer_system_prompt(task_prompt=TASK_DESC)

SUM_RECALL_PROMPT = summarizer_recall_prompt(
    task_prompt=TASK_DESC,
    recall_instructions=(
        'classify each word below as "Old" (appeared earlier in the original list) or '
        '"New" (first appearance at that position in the list).\n\n{trials_text}'
    ),
    format_rules=FORMAT_RULES,
)


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    words_json_path: str = "data/words.json",
    n_repeat: int = 1,
    max_trials_per_game: int = 100,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    words = load_words(words_json_path)
    n_participants = max(1, int(n_repeat))
    rng = random.Random(stimuli_seed)

    # Generate one game per participant (shared across conditions)
    games = []
    for _ in range(n_participants):
        trials = generate_one_game(words, rng, max_trials_per_game)
        games.append(trials)

    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in CONDITIONS:
        def _one_participant(pid: int, cond_id: str = cond_id) -> Dict[str, Any]:
            trials = games[pid - 1]
            word_list_text = "\n".join(f"{t['trial_index']}: {t['word']}" for t in trials)
            agent = WorkingMemoryAgent(
                llm=llm,
                condition_id=cond_id,
                temperature=temperature,
                debug=debug,
                system_prompt_override=WM_SYSTEM_PROMPTS[cond_id],
            )

            if debug:
                print(f"\n{'='*50}")
                print(f"{TASK_NAME} | {cond_id} | p{pid} | {len(trials)} trials")
                print(f"{'='*50}")

            encoding_log = agent.encode(word_list_text)

            # Recall: classify each trial
            trials_text = "\n".join(f"trial {t['trial_index']}: {t['word']}" for t in trials)
            recall_prompt = RECALL_PROMPT.format(
                trials_text=trials_text,
                wm_contents="{wm_contents}",
            )
            recall_raw = agent.recall(recall_prompt=recall_prompt, max_tokens=1024)

            resp_map = parse_responses(recall_raw)
            scored = score_game(trials, resp_map)
            scored["slot_utilization"] = len(agent.wm.store) / MAX_KEYS

            if debug:
                print(f"  score: {scored['score']}")
                print()

            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{pid}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "repeat_index": pid,
                "encoding_log": encoding_log,
                "final_kv": agent.wm.store,
                "recall_raw": recall_raw,
                "gold_trials": trials,
                "metrics": {
                    "score": scored["score"],
                    "first_error_at": scored["first_error_at"],
                    "n_trials": scored["n_trials"],
                    "slot_utilization": scored["slot_utilization"],
                },
                "per_trial": scored["per_trial"],
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        cond_summary = summarize_game_rows(rows, cond_id, CONDITIONS[cond_id]["name"])
        cond_summary["metrics"]["slot_utilization"] = (
            sum(r["metrics"]["slot_utilization"] for r in rows) / len(rows) if rows else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_score_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {"task": TASK_NAME, "n_participants": n_participants, "conditions": cond_summaries}
    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary


def evaluate_summarizer(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    words_json_path: str = "data/words.json",
    n_repeat: int = 1,
    max_trials_per_game: int = 100,
    stimuli_seed: int = 42,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    words = load_words(words_json_path)
    n_participants = max(1, int(n_repeat))
    rng = random.Random(stimuli_seed)

    games = []
    for _ in range(n_participants):
        trials = generate_one_game(words, rng, max_trials_per_game)
        games.append(trials)

    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME_SUM}.jsonl"
    sink = JsonlSink(jsonl_path)

    for cond_id in SUMMARIZER_CONDITIONS:
        def _one_participant(pid: int, cond_id: str = cond_id) -> Dict[str, Any]:
            trials = games[pid - 1]
            word_list_text = "\n".join(f"{t['trial_index']}: {t['word']}" for t in trials)
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
                print(f"\n{'='*50}")
                print(f"{TASK_NAME_SUM} | {cond_id} | p{pid} | {len(trials)} trials")
                print(f"{'='*50}")

            encoding_log = agent.encode(word_list_text)

            trials_text = "\n".join(f"trial {t['trial_index']}: {t['word']}" for t in trials)
            recall_prompt = SUM_RECALL_PROMPT.format(
                trials_text=trials_text,
                summary="{summary}",
            )
            recall_raw = agent.recall(recall_prompt=recall_prompt, max_tokens=1024)

            resp_map = parse_responses(recall_raw)
            scored = score_game(trials, resp_map)
            scored["summary_length_words"] = agent.summary_length_words()

            if debug:
                print(f"  score: {scored['score']}")
                print()

            row = {
                "id": f"{TASK_NAME_SUM}:{cond_id}:p{pid}",
                "condition_id": cond_id,
                "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
                "repeat_index": pid,
                "encoding_log": encoding_log,
                "final_summary": agent.summary,
                "recall_raw": recall_raw,
                "gold_trials": trials,
                "metrics": {
                    "score": scored["score"],
                    "first_error_at": scored["first_error_at"],
                    "n_trials": scored["n_trials"],
                    "summary_length_words": scored["summary_length_words"],
                },
                "per_trial": scored["per_trial"],
            }
            sink.append(row)
            return row

        rows = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        cond_summary = summarize_game_rows(
            rows, cond_id, SUMMARIZER_CONDITIONS[cond_id]["name"]
        )
        cond_summary["metrics"]["summary_length_words"] = (
            sum(r["metrics"]["summary_length_words"] for r in rows) / len(rows) if rows else 0.0
        )
        cond_summaries.append(cond_summary)
        all_rows.extend(rows)

    save_score_by_condition_figure(out_dir, TASK_NAME_SUM, cond_summaries)

    summary = {"task": TASK_NAME_SUM, "n_participants": n_participants, "conditions": cond_summaries}
    write_json(out_dir / "tasks" / f"{TASK_NAME_SUM}_summary.json", summary)
    return summary
