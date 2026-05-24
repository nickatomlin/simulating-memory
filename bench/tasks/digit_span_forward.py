from __future__ import annotations
import random, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count
from ..core.plotting import save_fig

from .human_simulation_prefixes import HUMAN_SIM_INTRO_C2, HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from .stimulus_prompt_shell import DIGIT_SPAN_SEP_BEFORE_RULES, wrap_stimulus_prompt

_TASK_DIR = Path(__file__).resolve().parent
_C4_HUMAN_TRIALS = (_TASK_DIR / "prompts" / "digit_span_forward_c4_human.txt").read_text(encoding="utf-8")

TASK_NAME="digit_span_forward"

####
# PROMPTS (verbatim from generate_digit_span.ipynb)
####

TASK_DESC = """You will see a sequence of digits presented one at a time. Your task is to remember the digits in the exact order they appear. After the sequence ends, type the digits in the same order. The sequences will gradually become longer. Try to remember them as accurately as possible.
"""

HUMAN_PROMPT = """The human will see a sequence of digits presented one at a time. Their task is to remember the digits in the exact order they appear. Then, the sequence will disappear. After the sequence disappears, they will be asked to type the digits in the same order as they appeared. The sequences will gradually become longer. They will be asked to remember them as accurately as possible.
"""

FORMAT_RULES = """Output ONLY lines of the form:
press <<D>>.
... (one per digit in order)

For example, if the digits are the following: [4, 8, 2], the response should be:
press <<4>>.
press <<8>>.
press <<2>>.
"""

# Legacy C4 few-shot examples (replaced by real human trial transcripts in prompts/).
# ICL_EXAMPLES = """Here are example results from previous human participants.
#
# Example A:
# The digits are the following: [4, 8, 2]
# Response:
# press <<4>>.
# press <<8>>.
# press <<2>>.
# (Correct)
#
# Example B:
# The digits are the following: [7, 3, 9, 1, 6, 4, 8]
# Response:
# press <<7>>.
# press <<3>>.
# press <<9>>.
# press <<1>>.
# (Wrong)
#
# Example C:
# The digits are the following: [5, 2, 8, 6, 1, 9, 4, 7, 3]
# Response:
# press <<5>>.
# press <<2>>.
# press <<8>>.
# press <<6>>.
# press <<1>>.
# press <<9>>.
# (Wrong)
# """

CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {
        "name": "Just describe the task",
        "prompt_prefix": TASK_DESC,
    },
    "C2": {
        "name": "Task + simulate human psychology experiment",
        "prompt_prefix": (
            HUMAN_SIM_INTRO_C2 + HUMAN_PROMPT
        ),
    },
    "C3": {
        "name": "Task + human simulation + limited memory emphasis",
        "prompt_prefix": HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN + HUMAN_PROMPT,
    },
    "C4": {
        "name": "Task + human simulation + limited memory + in-context examples",
        "prompt_prefix": (
            HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
            + HUMAN_PROMPT
            + "\n"
            + _C4_HUMAN_TRIALS
        ),
    },
}

PRESS_RE = re.compile(r"press\s*(?:<<\s*)?([0-9])\s*(?:\s*>>)?\s*\.?", re.IGNORECASE)


def generate_trials(min_span:int=2,max_span:int=20,sequences_per_span:int=20,seed:Optional[int]=42)->List[Dict[str,Any]]:
    if seed is not None:
        random.seed(seed)
    trials=[]
    for span in range(min_span,max_span+1):
        for seq_idx in range(1,sequences_per_span+1):
            digits=[random.randint(0,9) for _ in range(span)]
            trials.append({
                "span_length": span,
                "sequence_index": seq_idx,
                "digits": digits,
                "text": f"The digits are the following: {digits}",
            })
    return trials

def parse_pressed_digits(answer: str) -> List[int]:
    # Some runs use `press D.` while others use `press <<D>>.`.
    # Be permissive: support optional `<< >>` and optional trailing dot.
    return [int(m.group(1)) for m in PRESS_RE.finditer(answer or "")]

def exact_match(pred: List[int], gold: List[int]) -> float:
    return 1.0 if pred==gold else 0.0

def prefix_accuracy(pred: List[int], gold: List[int]) -> float:
    if not gold:
        return 0.0
    k=0
    for p,g in zip(pred,gold):
        if p==g:
            k+=1
        else:
            break
    return k/len(gold)


def build_prompt(condition_id: str, stimulus_text: str) -> str:
    """Condition prefix, then stimuli framing (C1 vs human-facing), then formatting rules."""
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    return wrap_stimulus_prompt(
        prefix,
        condition_id,
        stimulus_text.strip(),
        FORMAT_RULES,
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _merge_exact_by_span_across_participants(per_participant: List[Dict[Any, float]]) -> Dict[str, float]:
    """Average exact-by-span curves across participants (each span averaged over participants who reached it)."""
    by_span: Dict[Any, List[float]] = defaultdict(list)
    for d in per_participant:
        for k, v in d.items():
            by_span[k].append(float(v))
    return {str(k): _mean(by_span[k]) for k in sorted(by_span.keys())}


def save_exact_by_span_figure(
    out_dir: Path,
    task_name: str,
    condition_id: str,
    merged_span: Dict[str, float],
    n_participants: int,
) -> None:
    spans = [int(k) for k in merged_span.keys()]
    vals = [merged_span[str(s)] for s in spans]
    fig = plt.figure()
    plt.plot(spans, vals, marker="o")
    plt.xlabel("Span length")
    plt.ylabel("Exact match rate")
    plt.title(f"Forward digit span ({condition_id}): mean exact by span (n={n_participants})")
    save_fig(fig, out_dir / "figures" / task_name / f"{condition_id}_exact_by_span.png")


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    stimuli_seed: int = 42,
    min_span: int = 2,
    max_span: int = 20,
    sequences_per_span: int = 2,
    n_participants: int = 1,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """
    For each of n_participants independent runs, digit sequences use a distinct RNG seed:
    participant_seed = stimuli_seed + participant_id - 1 (participant_id in 1..n; n=1 gives base seed).

    ``sequences_per_span`` (trials per span step) is independent of ``n_participants``; the CLI
    ``--repeat`` flag only overrides participant count.

    Per-condition summary metrics are means across participants; jsonl rows include
    participant_id and stimuli_seed.
    """
    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []

    # Per condition: collect lists for averaging
    cond_digit_spans: Dict[str, List[float]] = defaultdict(list)
    cond_exact: Dict[str, List[float]] = defaultdict(list)
    cond_prefix: Dict[str, List[float]] = defaultdict(list)
    cond_per_span_list: Dict[str, List[Dict[Any, float]]] = defaultdict(list)
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    def _one_participant(
        participant_id: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[float, float, float, Dict[Any, float]]]]:
        participant_seed = int(stimuli_seed) + participant_id - 1
        trials = generate_trials(min_span, max_span, sequences_per_span, seed=participant_seed)

        trials_by_span: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for t in trials:
            trials_by_span[t["span_length"]].append(t)

        rows_out: List[Dict[str, Any]] = []
        contrib: Dict[str, Tuple[float, float, float, Dict[Any, float]]] = {}

        for cond_id in ["C1", "C2", "C3", "C4"]:
            rows: List[Dict[str, Any]] = []
            by_span_exact: Dict[int, List[float]] = defaultdict(list)

            digit_span = 0

            for span in range(min_span, max_span + 1):
                span_trials = trials_by_span.get(span, [])
                if not span_trials:
                    break

                span_trials = span_trials[:sequences_per_span]
                any_correct = False

                for t in span_trials:
                    prompt = build_prompt(cond_id, t["text"])
                    resp = llm.generate(
                        prompt,
                        temperature=float(model_cfg.get("temperature", 0.0)),
                        max_tokens=int(model_cfg.get("max_tokens", 512)),
                        top_p=float(model_cfg.get("top_p", 1.0)),
                        seed=model_cfg.get("seed"),
                    )

                    pred = parse_pressed_digits(resp.text)
                    gold = t["digits"]
                    m = {
                        "exact": exact_match(pred, gold),
                        "prefix_acc": prefix_accuracy(pred, gold),
                    }
                    by_span_exact[span].append(m["exact"])

                    if m["exact"] == 1.0:
                        any_correct = True

                    row = {
                        "id": f"{TASK_NAME}:{cond_id}:p{participant_id}:span{span}:seq{t['sequence_index']}",
                        "participant_id": participant_id,
                        "stimuli_seed": participant_seed,
                        "condition_id": cond_id,
                        "condition_name": CONDITIONS[cond_id]["name"],
                        "span_length": span,
                        "sequence_index": t["sequence_index"],
                        "prompt": prompt,
                        "gold": gold,
                        "pred": pred,
                        "raw": resp.text,
                        "metrics": m,
                    }
                    sink.append(row)
                    rows.append(row)

                if any_correct:
                    digit_span = span
                else:
                    break

            per_span = {k: sum(v) / len(v) for k, v in sorted(by_span_exact.items())}
            overall_exact = (
                sum(r["metrics"]["exact"] for r in rows) / len(rows) if rows else 0.0
            )
            overall_prefix = (
                sum(r["metrics"]["prefix_acc"] for r in rows) / len(rows) if rows else 0.0
            )

            contrib[cond_id] = (
                float(digit_span),
                overall_exact,
                overall_prefix,
                per_span,
            )
            rows_out.extend(rows)

        return rows_out, contrib

    for part_result in map_participants(
        list(range(1, n_participants + 1)),
        _one_participant,
        max_workers=workers,
    ):
        p_rows, p_contrib = part_result
        all_rows.extend(p_rows)
        for cond_id, (ds, oe, op, ps) in p_contrib.items():
            cond_digit_spans[cond_id].append(ds)
            cond_exact[cond_id].append(oe)
            cond_prefix[cond_id].append(op)
            cond_per_span_list[cond_id].append(ps)

    for cond_id in ["C1", "C2", "C3", "C4"]:
        merged_span = _merge_exact_by_span_across_participants(cond_per_span_list[cond_id])
        cond_summaries.append(
            {
                "condition": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "n": n_participants,
                "metrics": {
                    "digit_span": _mean(cond_digit_spans[cond_id]),
                    "exact": _mean(cond_exact[cond_id]),
                    "prefix_acc": _mean(cond_prefix[cond_id]),
                },
                "breakdown": {"exact_by_span": merged_span},
            }
        )

        save_exact_by_span_figure(out_dir, TASK_NAME, cond_id, merged_span, n_participants)

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "stimuli_seed_base": int(stimuli_seed),
        "participant_seeds": [int(stimuli_seed) + pid - 1 for pid in range(1, n_participants + 1)],
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
