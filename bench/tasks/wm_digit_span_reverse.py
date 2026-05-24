from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.io import write_json, JsonlSink
from ..core.llm import LLM
from ..core.parallel import map_participants, resolve_worker_count
from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
from ..core.working_memory import MAX_KEYS
from .digit_span_forward import (
    _mean,
    _merge_exact_by_span_across_participants,
    exact_match,
    parse_pressed_digits,
)
from .digit_span_reverse import (
    FORMAT_RULES as REVERSE_FORMAT_RULES,
)
from .digit_span_reverse import (
    TASK_DESC as REVERSE_TASK_DESC,
)
from .digit_span_reverse import (
    generate_trials,
    save_exact_by_span_figure,
)
from .wm_digit_span_forward import (
    CONDITIONS,
    DIGIT_SUMMARIZE_ENCODE_PROMPT,
    WM_SYSTEM_PROMPTS,
    _indexed_sequence,
    encode_digit_span,
)
from .wm_prompt_parts import (
    SUMMARIZER_CONDITIONS,
    summarizer_recall_prompt,
    summarizer_system_prompt,
    wm_recall_prompt,
)

TASK_NAME = "wm_digit_span_reverse"
TASK_NAME_SUM = "sum_digit_span_reverse"

REVERSE_RECALL_PROMPT = wm_recall_prompt(
    task_prompt=REVERSE_TASK_DESC,
    wm_recall_instructions=(
        "First read all chunks in left-to-right order to recover the full digit sequence. "
        "Then output every digit in reverse order from your memory chunks, followed by <<S>>. "
    ),
    format_rules=REVERSE_FORMAT_RULES
    + "\nAfter the final digit, output:\npress <<S>>.",
)

SUM_REVERSE_RECALL_PROMPT = summarizer_recall_prompt(
    task_prompt=REVERSE_TASK_DESC,
    recall_instructions=(
        "First read your summary to recover the full digit sequence. Then output "
        "every digit in reverse order, followed by <<S>>. No extra text, no explanations."
    ),
    format_rules=REVERSE_FORMAT_RULES
    + "\nAfter the final digit, output:\npress <<S>>.",
)


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    stimuli_seed: int = 42,
    min_span: int = 2,
    max_span: int = 100,
    sequences_per_span: int = 20,
    n_participants: int = 1,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    n_participants = max(1, int(n_participants))
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    work_items: List[Tuple[int, int, str, int, Dict[str, Any]]] = []
    for participant_id in range(1, n_participants + 1):
        participant_seed = int(stimuli_seed) + participant_id - 1
        trials = generate_trials(
            min_span, max_span, sequences_per_span, seed=participant_seed
        )
        trials_by_span: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for t in trials:
            trials_by_span[t["span_length"]].append(t)
        for cond_id in CONDITIONS:
            for span in range(min_span, max_span + 1):
                span_trials = trials_by_span.get(span, [])[:sequences_per_span]
                if not span_trials:
                    break
                for trial in span_trials:
                    work_items.append(
                        (participant_id, participant_seed, cond_id, span, trial)
                    )

    workers = resolve_worker_count(
        len(work_items), max_parallel=max_parallel_participants
    )

    def _one_trial(i: int) -> Dict[str, Any]:
        participant_id, participant_seed, cond_id, span, trial = work_items[i]
        digits = trial["digits"]
        gold = trial.get("target_reverse", list(reversed(digits)))
        agent = WorkingMemoryAgent(
            llm=llm,
            condition_id=cond_id,
            temperature=temperature,
            debug=debug,
            system_prompt_override=WM_SYSTEM_PROMPTS[cond_id],
        )

        if debug:
            print(f"\n{'=' * 50}")
            print(
                f"{TASK_NAME} | {cond_id} | p{participant_id} | span={span} seq={trial['sequence_index']}"
            )
            print(f"  digits: {digits}  (target reversed: {gold})")
            print(f"{'=' * 50}")

        candidates_text, encoding_log = encode_digit_span(
            agent, llm, digits, temperature
        )
        recall_raw = agent.recall(recall_prompt=REVERSE_RECALL_PROMPT)
        pred = parse_pressed_digits(recall_raw)
        has_s = "<<S>>" in recall_raw or "<<s>>" in recall_raw.lower()
        final_kv = agent.wm.store
        m = {
            "exact": exact_match(pred, gold),
            "has_S": 1.0 if has_s else 0.0,
            "slot_utilization": float(len(final_kv) / MAX_KEYS),
        }
        row = {
            "id": f"{TASK_NAME}:{cond_id}:p{participant_id}:span{span}:seq{trial['sequence_index']}",
            "participant_id": participant_id,
            "stimuli_seed": participant_seed,
            "condition_id": cond_id,
            "condition_name": CONDITIONS[cond_id]["name"],
            "span_length": span,
            "sequence_index": trial["sequence_index"],
            "digits": digits,
            "gold": gold,
            "pred": pred,
            "candidates_text": candidates_text,
            "encoding_log": encoding_log,
            "final_kv": final_kv,
            "recall_raw": recall_raw,
            "metrics": m,
        }
        sink.append(row)
        return row

    all_rows: List[Dict[str, Any]] = map_participants(
        list(range(len(work_items))),
        _one_trial,
        max_workers=workers,
    )

    cond_digit_spans: Dict[str, List[float]] = defaultdict(list)
    cond_exact: Dict[str, List[float]] = defaultdict(list)
    cond_slot_utilization: Dict[str, List[float]] = defaultdict(list)
    cond_has_s: Dict[str, List[float]] = defaultdict(list)
    cond_per_span_list: Dict[str, List[Dict[Any, float]]] = defaultdict(list)

    rows_by_pc: Dict[Tuple[int, str], Dict[int, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in all_rows:
        rows_by_pc[(r["participant_id"], r["condition_id"])][r["span_length"]].append(r)

    for participant_id in range(1, n_participants + 1):
        for cond_id in CONDITIONS:
            by_span = rows_by_pc.get((participant_id, cond_id), {})
            kept_rows: List[Dict[str, Any]] = []
            by_span_exact: Dict[int, List[float]] = defaultdict(list)
            digit_span = 0
            for span in range(min_span, max_span + 1):
                span_rows = sorted(
                    by_span.get(span, []), key=lambda r: r["sequence_index"]
                )
                if not span_rows:
                    break
                kept_rows.extend(span_rows)
                for r in span_rows:
                    by_span_exact[span].append(r["metrics"]["exact"])
                if any(r["metrics"]["exact"] == 1.0 for r in span_rows):
                    digit_span = span
                else:
                    break

            per_span = {
                k: sum(v) / len(v) for k, v in sorted(by_span_exact.items())
            }
            n = len(kept_rows)
            cond_digit_spans[cond_id].append(float(digit_span))
            cond_exact[cond_id].append(
                sum(r["metrics"]["exact"] for r in kept_rows) / n if n else 0.0
            )
            cond_has_s[cond_id].append(
                sum(r["metrics"]["has_S"] for r in kept_rows) / n if n else 0.0
            )
            cond_slot_utilization[cond_id].append(
                sum(r["metrics"]["slot_utilization"] for r in kept_rows) / n
                if n
                else 0.0
            )
            cond_per_span_list[cond_id].append(per_span)

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in CONDITIONS:
        merged_span = _merge_exact_by_span_across_participants(
            cond_per_span_list[cond_id]
        )
        cond_summaries.append(
            {
                "condition": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "n": n_participants,
                "metrics": {
                    "digit_span": _mean(cond_digit_spans[cond_id]),
                    "exact": _mean(cond_exact[cond_id]),
                    "has_S": _mean(cond_has_s[cond_id]),
                    "slot_utilization": _mean(cond_slot_utilization[cond_id]),
                },
                "breakdown": {"exact_by_span": merged_span},
            }
        )
        save_exact_by_span_figure(
            out_dir, TASK_NAME, cond_id, merged_span, n_participants
        )

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "stimuli_seed_base": int(stimuli_seed),
        "participant_seeds": [
            int(stimuli_seed) + pid - 1 for pid in range(1, n_participants + 1)
        ],
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary


def evaluate_summarizer(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    stimuli_seed: int = 42,
    min_span: int = 2,
    max_span: int = 100,
    sequences_per_span: int = 20,
    n_participants: int = 1,
    temperature: float = 0.0,
    debug: bool = False,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    n_participants = max(1, int(n_participants))
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME_SUM}.jsonl")

    work_items: List[Tuple[int, int, str, int, Dict[str, Any]]] = []
    for participant_id in range(1, n_participants + 1):
        participant_seed = int(stimuli_seed) + participant_id - 1
        trials = generate_trials(
            min_span, max_span, sequences_per_span, seed=participant_seed
        )
        trials_by_span: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for t in trials:
            trials_by_span[t["span_length"]].append(t)
        for cond_id in SUMMARIZER_CONDITIONS:
            for span in range(min_span, max_span + 1):
                span_trials = trials_by_span.get(span, [])[:sequences_per_span]
                if not span_trials:
                    break
                for trial in span_trials:
                    work_items.append(
                        (participant_id, participant_seed, cond_id, span, trial)
                    )

    workers = resolve_worker_count(
        len(work_items), max_parallel=max_parallel_participants
    )

    def _one_trial(i: int) -> Dict[str, Any]:
        participant_id, participant_seed, cond_id, span, trial = work_items[i]
        digits = trial["digits"]
        gold = trial.get("target_reverse", list(reversed(digits)))
        agent = SummarizerAgent(
            llm=llm,
            condition_id=cond_id,
            temperature=temperature,
            debug=debug,
            system_prompt_override=summarizer_system_prompt(
                REVERSE_TASK_DESC, condition_id=cond_id
            ),
        )

        if debug:
            print(f"\n{'=' * 50}")
            print(
                f"{TASK_NAME_SUM} | {cond_id} | p{participant_id} | span={span} seq={trial['sequence_index']}"
            )
            print(f"  digits: {digits}  (target reversed: {gold})")
            print(f"{'=' * 50}")

        encode_input = DIGIT_SUMMARIZE_ENCODE_PROMPT.format(
            indexed_sequence=_indexed_sequence(digits)
        )
        encoding_log = agent.encode(encode_input)
        recall_raw = agent.recall(recall_prompt=SUM_REVERSE_RECALL_PROMPT)
        pred = parse_pressed_digits(recall_raw)
        has_s = "<<S>>" in recall_raw or "<<s>>" in recall_raw.lower()
        summary_length_words = agent.summary_length_words()
        m = {
            "exact": exact_match(pred, gold),
            "has_S": 1.0 if has_s else 0.0,
            "summary_length_words": float(summary_length_words),
        }
        row = {
            "id": f"{TASK_NAME_SUM}:{cond_id}:p{participant_id}:span{span}:seq{trial['sequence_index']}",
            "participant_id": participant_id,
            "stimuli_seed": participant_seed,
            "condition_id": cond_id,
            "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
            "span_length": span,
            "sequence_index": trial["sequence_index"],
            "digits": digits,
            "gold": gold,
            "pred": pred,
            "encoding_log": encoding_log,
            "final_summary": agent.summary,
            "recall_raw": recall_raw,
            "metrics": m,
        }
        sink.append(row)
        return row

    all_rows: List[Dict[str, Any]] = map_participants(
        list(range(len(work_items))),
        _one_trial,
        max_workers=workers,
    )

    cond_digit_spans: Dict[str, List[float]] = defaultdict(list)
    cond_exact: Dict[str, List[float]] = defaultdict(list)
    cond_summary_length: Dict[str, List[float]] = defaultdict(list)
    cond_has_s: Dict[str, List[float]] = defaultdict(list)
    cond_per_span_list: Dict[str, List[Dict[Any, float]]] = defaultdict(list)

    rows_by_pc: Dict[Tuple[int, str], Dict[int, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in all_rows:
        rows_by_pc[(r["participant_id"], r["condition_id"])][r["span_length"]].append(r)

    for participant_id in range(1, n_participants + 1):
        for cond_id in SUMMARIZER_CONDITIONS:
            by_span = rows_by_pc.get((participant_id, cond_id), {})
            kept_rows: List[Dict[str, Any]] = []
            by_span_exact: Dict[int, List[float]] = defaultdict(list)
            digit_span = 0
            for span in range(min_span, max_span + 1):
                span_rows = sorted(
                    by_span.get(span, []), key=lambda r: r["sequence_index"]
                )
                if not span_rows:
                    break
                kept_rows.extend(span_rows)
                for r in span_rows:
                    by_span_exact[span].append(r["metrics"]["exact"])
                if any(r["metrics"]["exact"] == 1.0 for r in span_rows):
                    digit_span = span
                else:
                    break

            per_span = {
                k: sum(v) / len(v) for k, v in sorted(by_span_exact.items())
            }
            n = len(kept_rows)
            cond_digit_spans[cond_id].append(float(digit_span))
            cond_exact[cond_id].append(
                sum(r["metrics"]["exact"] for r in kept_rows) / n if n else 0.0
            )
            cond_has_s[cond_id].append(
                sum(r["metrics"]["has_S"] for r in kept_rows) / n if n else 0.0
            )
            cond_summary_length[cond_id].append(
                sum(r["metrics"]["summary_length_words"] for r in kept_rows) / n
                if n
                else 0.0
            )
            cond_per_span_list[cond_id].append(per_span)

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in SUMMARIZER_CONDITIONS:
        merged_span = _merge_exact_by_span_across_participants(
            cond_per_span_list[cond_id]
        )
        cond_summaries.append(
            {
                "condition": cond_id,
                "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
                "n": n_participants,
                "metrics": {
                    "digit_span": _mean(cond_digit_spans[cond_id]),
                    "exact": _mean(cond_exact[cond_id]),
                    "has_S": _mean(cond_has_s[cond_id]),
                    "summary_length_words": _mean(cond_summary_length[cond_id]),
                },
                "breakdown": {"exact_by_span": merged_span},
            }
        )
        save_exact_by_span_figure(
            out_dir, TASK_NAME_SUM, cond_id, merged_span, n_participants
        )

    summary = {
        "task": TASK_NAME_SUM,
        "n_participants": n_participants,
        "stimuli_seed_base": int(stimuli_seed),
        "participant_seeds": [
            int(stimuli_seed) + pid - 1 for pid in range(1, n_participants + 1)
        ],
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME_SUM}_summary.json", summary)
    return summary
