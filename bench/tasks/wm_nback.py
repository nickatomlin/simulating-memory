from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.io import write_json, JsonlSink
from ..core.llm import LLM
from ..core.parallel import map_participants, resolve_worker_count
from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
from ..core.working_memory import MAX_KEYS
from .nback import (
    BLOCK_INTRO_MS,
    CONSONANTS,
    HUMAN_PROMPT_BY_N,
    ISI_MS,
    N_LEVELS,
    STIMULUS_MS,
    TARGET_RATIO,
    TASK_DESC_BY_N,
    TRIALS_PER_BLOCK,
    NBackBlock,
    generate_block,
    save_acc_by_n_figure,
    score_block,
    summarize_nback_condition,
)
from .wm_prompt_parts import CONDITIONS as WM_CONDITIONS
from .wm_prompt_parts import SUMMARIZER_CONDITIONS, summarizer_system_prompt
from .wm_prompt_parts import wm_system_prompt

TASK_NAME = "wm_nback"
TASK_NAME_SUM = "sum_nback"

CONDITIONS = WM_CONDITIONS


def wm_system_prompt_for(condition_id: str, n: int) -> str:
    return wm_system_prompt(
        condition_id,
        task_prompt=TASK_DESC_BY_N[n],
        human_task_prompt=HUMAN_PROMPT_BY_N[n],
    )


WM_SYSTEM_PROMPTS = {
    cond_id: wm_system_prompt_for(cond_id, 1) for cond_id in CONDITIONS
}


def _parse_classification(text: str) -> Optional[str]:
    """Extract same / different / no response from agent text output."""
    t = re.sub(r"\s+", " ", text.strip()).lower()
    if "no response" in t or "no-response" in t:
        return "No response"
    if "non-target" in t or "non target" in t or "nontarget" in t or "different" in t:
        return "Different"
    if "same" in t or "target" in t:
        return "Same"
    return None


def run_nback_block(
    llm: LLM,
    block: NBackBlock,
    condition_id: str,
    temperature: float,
    debug: bool,
) -> Dict[str, Any]:
    """Run one n-back block turn-by-turn using the WM agent's step() method.

    Returns dict with trial_map (1-based trial index -> Same/Different),
    buffer_map, step_log, and final_kv.
    """
    n = block.n
    agent = WorkingMemoryAgent(
        llm=llm,
        condition_id=condition_id,
        temperature=temperature,
        debug=debug,
        system_prompt_override=wm_system_prompt_for(condition_id, n),
    )

    # Initial instruction
    agent.step(
        f"This is a {n}-back task. You will see letters one at a time. "
        f"For the first {n} letter(s), respond 'no response'. "
        f"After that, respond 'same' if the current letter matches the letter "
        f"{n} position(s) back, otherwise 'different'. "
        f"Update your working memory each turn to track the recent sequence.",
        allow_tools=False,
        max_tokens=512,
    )

    trial_map: Dict[int, str] = {}
    buffer_map: Dict[int, Optional[str]] = {i: None for i in range(1, n + 1)}

    for pos_idx, letter in enumerate(block.full_sequence):
        global_pos = pos_idx + 1  # 1-based

        if debug:
            print(f"  Letter {global_pos}: {letter}")

        response_text = agent.step(
            f"Next letter: {letter}",
            allow_tools=True,
            max_tokens=512,
        )

        classification = _parse_classification(response_text)

        if debug:
            cls_str = classification or f"(unparsed: {response_text[:60]!r})"
            print(f"    -> {cls_str}")
            print(f"    Memory: {agent.wm.snapshot()}")

        if global_pos <= n:
            # Buffer period
            buffer_map[global_pos] = classification
        else:
            # Trial period
            trial_i = global_pos - n  # 1-based trial index
            if classification in ("Same", "Different"):
                trial_map[trial_i] = classification

    return {
        "trial_map": trial_map,
        "buffer_map": buffer_map,
        "step_log": agent.get_step_log(),
        "final_kv": agent.wm.store,
        "slot_utilization": len(agent.wm.store) / MAX_KEYS,
    }


def run_summarizer_nback_block(
    llm: LLM,
    block: NBackBlock,
    condition_id: str,
    temperature: float,
    debug: bool,
) -> Dict[str, Any]:
    n = block.n
    agent = SummarizerAgent(
        llm=llm,
        condition_id=condition_id,
        temperature=temperature,
        debug=debug,
        system_prompt_override=summarizer_system_prompt(
            TASK_DESC_BY_N[n], condition_id=condition_id
        ),
    )

    agent.step(
        f"This is a {n}-back task. You will see letters one at a time. "
        f"For the first {n} letter(s), respond 'no response'. "
        f"After that, respond 'same' if the current letter matches the letter "
        f"{n} position(s) back, otherwise 'different'. "
        f"Keep a compact running summary of the recent sequence needed to answer.",
        max_tokens=512,
    )

    trial_map: Dict[int, str] = {}
    buffer_map: Dict[int, Optional[str]] = {i: None for i in range(1, n + 1)}

    for pos_idx, letter in enumerate(block.full_sequence):
        global_pos = pos_idx + 1

        if debug:
            print(f"  Letter {global_pos}: {letter}")

        response_text = agent.step(f"Next letter: {letter}", max_tokens=512)
        classification = _parse_classification(response_text)

        if debug:
            cls_str = classification or f"(unparsed: {response_text[:60]!r})"
            print(f"    -> {cls_str}")
            print(f"    Summary: {agent.summary}")

        if global_pos <= n:
            buffer_map[global_pos] = classification
        else:
            trial_i = global_pos - n
            if classification in ("Same", "Different"):
                trial_map[trial_i] = classification

    return {
        "trial_map": trial_map,
        "buffer_map": buffer_map,
        "step_log": agent.get_step_log(),
        "final_summary": agent.summary,
        "summary_length_words": agent.summary_length_words(),
    }


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    n_repeat: int = 1,
    n_repeats_per_participant: int = 1,
    stimuli_seed: int = 123,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    rng_master = random.Random(stimuli_seed)

    # Pre-generate seeds so parallelization is reproducible
    combos: List[tuple] = []
    for pid in range(1, int(n_repeat) + 1):
        for rep in range(1, int(n_repeats_per_participant) + 1):
            seed = rng_master.randrange(1_000_000_000)
            combos.append((pid, rep, seed))

    workers = resolve_worker_count(len(combos), max_parallel=max_parallel_participants)

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME}.jsonl"
    sink = JsonlSink(jsonl_path)

    def _one_combo(i: int) -> List[Dict[str, Any]]:
        pid, rep, seed = combos[i]
        rng = random.Random(seed)
        blocks = {n_level: generate_block(n_level, rng) for n_level in N_LEVELS}
        combo_rows: List[Dict[str, Any]] = []

        for cond_id in CONDITIONS:
            for n_level in N_LEVELS:
                block = blocks[n_level]

                if debug:
                    print(f"\n{'=' * 50}")
                    print(f"{TASK_NAME} | {cond_id} | p{pid} | {n_level}-back")
                    print(f"  sequence: {' '.join(block.full_sequence)}")
                    print(f"{'=' * 50}")

                block_result = run_nback_block(
                    llm=llm,
                    block=block,
                    condition_id=cond_id,
                    temperature=temperature,
                    debug=debug,
                )

                scored = score_block(block, block_result["trial_map"])

                row = {
                    "participant_id": pid,
                    "repeat_index": rep,
                    "condition_id": cond_id,
                    "condition_name": CONDITIONS[cond_id]["name"],
                    "n_level": n_level,
                    "buffer_letters": block.buffer_letters,
                    "trial_letters": block.trial_letters,
                    "trial_is_target": block.trial_is_target,
                    "model_parsed": block_result["trial_map"],
                    "model_parsed_buffer": block_result["buffer_map"],
                    "final_kv": block_result["final_kv"],
                    "slot_utilization": block_result["slot_utilization"],
                    "answered": scored["answered"],
                    "correct": scored["correct"],
                    "acc_over_answered": scored["accuracy_over_answered"],
                    "acc_over_14": scored["accuracy_over_14"],
                    "precision": scored["precision"],
                    "recall": scored["recall"],
                    "precision_z": scored["precision_z"],
                    "recall_z": scored["recall_z"],
                    "d_prime": scored["d_prime"],
                    "per_trial": scored["per_trial"],
                }
                sink.append(row)
                combo_rows.append(row)
        return combo_rows

    all_combo_results = map_participants(
        list(range(len(combos))),
        _one_combo,
        max_workers=workers,
    )
    results: List[Dict[str, Any]] = [
        row for combo_rows in all_combo_results for row in combo_rows
    ]

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in CONDITIONS:
        rows = [r for r in results if r["condition_id"] == cond_id]
        cond_summary = summarize_nback_condition(
            rows, cond_id, CONDITIONS[cond_id]["name"]
        )
        util_vals = [float(r["slot_utilization"]) for r in rows]
        cond_summary["metrics"]["slot_utilization"] = (
            float(sum(util_vals) / len(util_vals)) if util_vals else 0.0
        )
        cond_summaries.append(cond_summary)
        save_acc_by_n_figure(
            out_dir,
            TASK_NAME,
            cond_id,
            cond_summary["breakdown"]["acc_over_14_by_n"],
        )

    summary: Dict[str, Any] = {
        "task": TASK_NAME,
        "meta": {
            "matched_html": True,
            "html_constants": {
                "TRIALS_PER_BLOCK": TRIALS_PER_BLOCK,
                "TARGET_RATIO": TARGET_RATIO,
                "STIMULUS_MS": STIMULUS_MS,
                "ISI_MS": ISI_MS,
                "BLOCK_INTRO_MS": BLOCK_INTRO_MS,
                "N_LEVELS": N_LEVELS,
                "CONSONANTS": CONSONANTS,
            },
            "temperature": float(temperature),
            "n_repeat": int(n_repeat),
            "n_repeats_per_participant": int(n_repeats_per_participant),
            "seed": int(stimuli_seed),
            "conditions": {k: v["name"] for k, v in CONDITIONS.items()},
            "total_records": len(results),
        },
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary


def evaluate_summarizer(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    n_repeat: int = 1,
    n_repeats_per_participant: int = 1,
    stimuli_seed: int = 123,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    rng_master = random.Random(stimuli_seed)

    combos: List[tuple] = []
    for pid in range(1, int(n_repeat) + 1):
        for rep in range(1, int(n_repeats_per_participant) + 1):
            seed = rng_master.randrange(1_000_000_000)
            combos.append((pid, rep, seed))

    workers = resolve_worker_count(len(combos), max_parallel=max_parallel_participants)

    jsonl_path = out_dir / "tasks" / f"{TASK_NAME_SUM}.jsonl"
    sink = JsonlSink(jsonl_path)

    def _one_combo(i: int) -> List[Dict[str, Any]]:
        pid, rep, seed = combos[i]
        rng = random.Random(seed)
        blocks = {n_level: generate_block(n_level, rng) for n_level in N_LEVELS}
        combo_rows: List[Dict[str, Any]] = []

        for cond_id in SUMMARIZER_CONDITIONS:
            for n_level in N_LEVELS:
                block = blocks[n_level]

                if debug:
                    print(f"\n{'=' * 50}")
                    print(f"{TASK_NAME_SUM} | {cond_id} | p{pid} | {n_level}-back")
                    print(f"  sequence: {' '.join(block.full_sequence)}")
                    print(f"{'=' * 50}")

                block_result = run_summarizer_nback_block(
                    llm=llm,
                    block=block,
                    condition_id=cond_id,
                    temperature=temperature,
                    debug=debug,
                )

                scored = score_block(block, block_result["trial_map"])

                row = {
                    "participant_id": pid,
                    "repeat_index": rep,
                    "condition_id": cond_id,
                    "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
                    "n_level": n_level,
                    "buffer_letters": block.buffer_letters,
                    "trial_letters": block.trial_letters,
                    "trial_is_target": block.trial_is_target,
                    "model_parsed": block_result["trial_map"],
                    "model_parsed_buffer": block_result["buffer_map"],
                    "final_summary": block_result["final_summary"],
                    "summary_length_words": block_result["summary_length_words"],
                    "answered": scored["answered"],
                    "correct": scored["correct"],
                    "acc_over_answered": scored["accuracy_over_answered"],
                    "acc_over_14": scored["accuracy_over_14"],
                    "precision": scored["precision"],
                    "recall": scored["recall"],
                    "precision_z": scored["precision_z"],
                    "recall_z": scored["recall_z"],
                    "d_prime": scored["d_prime"],
                    "per_trial": scored["per_trial"],
                }
                sink.append(row)
                combo_rows.append(row)
        return combo_rows

    all_combo_results = map_participants(
        list(range(len(combos))),
        _one_combo,
        max_workers=workers,
    )
    results: List[Dict[str, Any]] = [
        row for combo_rows in all_combo_results for row in combo_rows
    ]

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in SUMMARIZER_CONDITIONS:
        rows = [r for r in results if r["condition_id"] == cond_id]
        cond_summary = summarize_nback_condition(
            rows, cond_id, SUMMARIZER_CONDITIONS[cond_id]["name"]
        )
        slw_vals = [float(r["summary_length_words"]) for r in rows]
        cond_summary["metrics"]["summary_length_words"] = (
            float(sum(slw_vals) / len(slw_vals)) if slw_vals else 0.0
        )
        cond_summaries.append(cond_summary)
        save_acc_by_n_figure(
            out_dir,
            TASK_NAME_SUM,
            cond_id,
            cond_summary["breakdown"]["acc_over_14_by_n"],
        )

    summary: Dict[str, Any] = {
        "task": TASK_NAME_SUM,
        "meta": {
            "matched_html": True,
            "html_constants": {
                "TRIALS_PER_BLOCK": TRIALS_PER_BLOCK,
                "TARGET_RATIO": TARGET_RATIO,
                "STIMULUS_MS": STIMULUS_MS,
                "ISI_MS": ISI_MS,
                "BLOCK_INTRO_MS": BLOCK_INTRO_MS,
                "N_LEVELS": N_LEVELS,
                "CONSONANTS": CONSONANTS,
            },
            "temperature": float(temperature),
            "n_repeat": int(n_repeat),
            "n_repeats_per_participant": int(n_repeats_per_participant),
            "seed": int(stimuli_seed),
            "conditions": {k: v["name"] for k, v in SUMMARIZER_CONDITIONS.items()},
            "total_records": len(results),
        },
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME_SUM}_summary.json", summary)
    return summary
