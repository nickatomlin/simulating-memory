from __future__ import annotations
import random, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count
from ..core.plotting import save_fig, plt

from .human_simulation_prefixes import HUMAN_SIM_INTRO_C2, HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from .stimulus_prompt_shell import DIGIT_SPAN_SEP_BEFORE_RULES, wrap_stimulus_prompt

_TASK_DIR = Path(__file__).resolve().parent
_C4_HUMAN_TRIALS = (_TASK_DIR / "prompts" / "digit_span_reverse_c4_human.txt").read_text(encoding="utf-8")

TASK_NAME="digit_span_reverse"

TASK_DESC = """You will see a sequence of digits presented one at a time. Your task is to remember the digits and enter them in reverse order. After the sequence ends, type the digits from last to first. The sequences will gradually become longer. Try to remember them as accurately as possible.

For example, if the digits are the following: [4, 8, 2]
You should answer:
press <<2>>.
press <<8>>.
press <<4>>.
"""

HUMAN_PROMPT = """The human will see a sequence of digits presented one at a time. Their task is to remember the digits and enter them in reverse order. Then, the sequence will disappear. After the sequence disappears, they will be asked to type the digits from last to first. The sequences will gradually become longer. They will be asked to remember them as accurately as possible.

For example, if the digits are the following: [4, 8, 2]
You should answer:
press <<2>>.
press <<8>>.
press <<4>>.
"""

FORMAT_RULES = """Output ONLY lines in this format:
press <<D>>.
... (one per digit in reverse order)
No extra text.
"""

####
# PROMPTS (verbatim from reverse_digit_span.ipynb)
####

# Legacy C4 few-shot examples (replaced by real human trial transcripts in prompts/).
# ICL_EXAMPLES = """Here is an example result from a previous human participant.
#
# Example A:
# The digits are the following: [4, 8, 2]
# Response:
# press <<2>>.
# press <<8>>.
# press <<4>>.
# (Correct)
#
# Example B:
# The digits are the following: [7, 3, 9, 1, 6, 4, 8]
# Response:
# press <<4>>.
# press <<6>>.
# press <<1>>.
# press <<9>>.
# press <<3>>.
# (Wrong)
#
# Example C:
# The digits are the following: [5, 2, 8, 6, 1, 9, 4, 7, 3]
# Response:
# press <<7>>.
# press <<4>>.
# press <<9>>.
# press <<1>>.
# (Wrong)
# """

CONDITIONS = {
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

def generate_trials(min_span:int=2,max_span:int=100,sequences_per_span:int=20,seed:Optional[int]=42)->List[Dict[str,Any]]:
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
                "target_reverse": list(reversed(digits)),
                "text": f"The digits are the following: {digits}",
            })
    return trials

def parse_pressed_digits(answer: str) -> List[int]:
    return [int(m.group(1)) for m in PRESS_RE.finditer(answer or "")]

def exact_match(pred: List[int], gold: List[int]) -> float:
    return 1.0 if pred==gold else 0.0


def build_prompt(condition_id: str, trial: Dict[str, Any]) -> str:
    """Condition prefix, then stimuli framing (C1 vs human-facing), then formatting rules."""
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    return wrap_stimulus_prompt(
        prefix,
        condition_id,
        trial["text"].strip(),
        FORMAT_RULES,
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _merge_exact_by_span_across_participants(per_participant: List[Dict[Any, float]]) -> Dict[str, float]:
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
    plt.title(f"Reverse digit span ({condition_id}): mean exact by span (n={n_participants})")
    save_fig(fig, out_dir / "figures" / task_name / f"{condition_id}_exact_by_span.png")


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    stimuli_seed: int = 42,
    min_span: int = 2,
    max_span: int = 100,
    sequences_per_span: int = 2,
    n_participants: int = 1,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Each participant uses participant_seed = stimuli_seed + participant_id - 1 for RNG (distinct digit lists).
    ``sequences_per_span`` is independent of ``n_participants``; CLI ``--repeat`` only sets participant count.
    Summary metrics are means across participants.
    """
    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)
    all_rows: List[Dict[str, Any]] = []

    cond_digit_spans: Dict[str, List[float]] = defaultdict(list)
    cond_exact: Dict[str, List[float]] = defaultdict(list)
    cond_per_span_list: Dict[str, List[Dict[Any, float]]] = defaultdict(list)
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    def _one_participant(
        participant_id: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[float, float, Dict[Any, float]]]]:
        participant_seed = int(stimuli_seed) + participant_id - 1
        trials = generate_trials(min_span, max_span, sequences_per_span, seed=participant_seed)

        trials_by_span: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for t in trials:
            trials_by_span[t["span_length"]].append(t)

        rows_out: List[Dict[str, Any]] = []
        contrib: Dict[str, Tuple[float, float, Dict[Any, float]]] = {}

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
                    prompt = build_prompt(cond_id, t)
                    resp = llm.generate(
                        prompt,
                        temperature=float(model_cfg.get("temperature", 0.0)),
                        max_tokens=int(model_cfg.get("max_tokens", 512)),
                        top_p=float(model_cfg.get("top_p", 1.0)),
                        seed=model_cfg.get("seed"),
                    )

                    pred = parse_pressed_digits(resp.text)
                    gold = t["target_reverse"]
                    m = {
                        "exact": exact_match(pred, gold),
                    }
                    by_span_exact[span].append(m["exact"])

                    if m["exact"] == 1.0:
                        any_correct = True

                    row = {
                        "id": f"{TASK_NAME}:{cond_id}:p{participant_id}:span{span}:seq{t['sequence_index']}",
                        "participant_id": participant_id,
                        "stimuli_seed": participant_seed,
                        "condition_id": cond_id,
                        "condition_name": CONDITIONS[cond_id]['name'],
                        "span_length": span,
                        "sequence_index": t['sequence_index'],
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
            contrib[cond_id] = (float(digit_span), overall_exact, per_span)
            rows_out.extend(rows)

        return rows_out, contrib

    for part_result in map_participants(
        list(range(1, n_participants + 1)),
        _one_participant,
        max_workers=workers,
    ):
        p_rows, p_contrib = part_result
        all_rows.extend(p_rows)
        for cond_id, (ds, oe, ps) in p_contrib.items():
            cond_digit_spans[cond_id].append(ds)
            cond_exact[cond_id].append(oe)
            cond_per_span_list[cond_id].append(ps)

    cond_summaries: List[Dict[str, Any]] = []
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
