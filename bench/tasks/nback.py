from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count
from ..core.plotting import save_fig

from .human_simulation_prefixes import HUMAN_SIM_INTRO_C2, HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from .stimulus_prompt_shell import wrap_stimulus_prompt

TASK_NAME = "nback"


# ============================
# 0) Match the HTML constants
# ============================

CONSONANTS = list("BCDFGHJKLMNPQRSTVWXYZ")
TRIALS_PER_BLOCK = 14
TARGET_RATIO = 1 / 3
STIMULUS_MS = 500
ISI_MS = 2000
# Only 1/2/3-back (0-back removed; matches single-block benchmark)
N_LEVELS = [1, 2, 3]
BLOCK_INTRO_MS = 2500


# ============================
# 1) Four prompt conditions (task/human/ICL text is per n-back level)
# ============================

TASK_DESC_BY_N: Dict[int, str] = {
    1: """You will be shown a sequence of letters. After every letter, you will have to decide whether it matches the letter one turn back. In each block, respond with "no response" to the first letter. Once enough letters have appeared, respond to each new letter as "same" or "different".

Example Question:
A → A → B → C → C
Responses:
no response, same, different, different, same
Explanation:
The second A matches the previous A → same
The second C matches the previous C → same
""",
    2: """You will be shown a sequence of letters. After every letter, you will have to decide whether it matches the letter two turns back. In each block, respond with "no response" to the first two letters. Once enough letters have appeared, respond to each new letter as "same" or "different".

Example Question:
A → B → A → C → A
Responses:
no response, no response, same, different, same
Explanation:
The third A matches the letter two steps earlier (A) → same
The last A matches the letter two steps earlier (A) → same
""",
    3: """You will be shown a sequence of letters. After every letter, you will have to decide whether it matches the letter three turns back. In each block, respond with "no response" to the first three letters. Once enough letters have appeared, respond to each new letter as "same" or "different".

Example Question:
A → B → C → A → D → C
Responses:
no response, no response, no response, same, different, same
Explanation:
The fourth A matches the letter three steps earlier (A) → same
The sixth C matches the letter three steps earlier (C) → same
""",
}

HUMAN_PROMPT_BY_N: Dict[int, str] = {
    1: """The human will be shown a sequence of letters. After every letter, the human will have to decide whether it matches the letter one turn back. In each block, respond with "no response" to the first letter. Once enough letters have appeared, respond to each new letter as "same" or "different".

Example Question:
A → A → B → C → C
Responses:
no response, same, different, different, same
Explanation:
The second A matches the previous A → same
The second C matches the previous C → same
""",
    2: """The human will be shown a sequence of letters. After every letter, the human will have to decide whether it matches the letter two turns back. In each block, respond with "no response" to the first two letters. Once enough letters have appeared, respond to each new letter as "same" or "different".

Example Question:
A → B → A → C → A
Responses:
no response, no response, same, different, same
Explanation:
The third A matches the letter two steps earlier (A) → same
The last A matches the letter two steps earlier (A) → same
""",
    3: """The human will be shown a sequence of letters. After every letter, the human will have to decide whether it matches the letter three turns back. In each block, respond with "no response" to the first three letters. Once enough letters have appeared, respond to each new letter as "same" or "different".

Example Question:
A → B → C → A → D → C
Responses:
no response, no response, no response, same, different, same
Explanation:
The fourth A matches the letter three steps earlier (A) → same
The sixth C matches the letter three steps earlier (C) → same
""",
}


def format_format_rules(n: int) -> str:
    """Output-format instructions for this block's n only (n + 14 lines total)."""
    if n not in (1, 2, 3):
        raise ValueError(f"n must be 1, 2, or 3, got {n}")
    total = n + TRIALS_PER_BLOCK
    return f"""Output ONLY lines in this exact format — one line per letter in the sequence shown for the block, in order.
- Answer "no response" to the very first {n} letters.
- For each of the remaining 14 letters, answer "same" or "different".

Numbering uses one index running through the whole sequence:
trial 1: no response
... 
trial {n + 1}: same
trial {n + 2}: different
...
trial {total}: same

So you output exactly {total} lines. After the first {n} lines (all "no response"), do not use "no response" again — only "same" or "different". Use the phrases "no response", "same", or "different" after the colon. No extra text, no explanations, no blank lines.
"""


ICL_EXAMPLES_BY_N: Dict[int, str] = {
    1: """Here are example outputs in the required line format.
Example (1-back):
Sequence:
A → A → B → C → C → C → C
Responses:
No response (Correct), Same (Correct), Different (Correct), Different (Correct), Same (Correct), Same (Correct), Same (Correct)
""",
    2: """Here are example outputs in the required line format.
Example (2-back):
Sequence:
A → B → A → C → A → C → A → C → A
Responses:
No response (Correct), No response (Correct), Same (Correct), Different (Correct), Same (Correct), Same (Correct), Same (Correct), Different (Wrong), Same (Correct)
""",
    3: """Here are example outputs in the required line format.
Example (3-back):
Sequence:
A → B → C → A → D → C → B → A → C → A
Responses:
No response (Correct), No response (Correct), No response (Correct), Same (Correct), Different (Correct), Same (Correct), Different (Correct), Different (Correct), Same (Correct), Different (Correct)
""",
}


def prompt_prefix_for(condition_id: str, n: int) -> str:
    """Instruction prefix for condition C1..C4 and this block's n-back level."""
    h = HUMAN_PROMPT_BY_N[n]
    if condition_id == "C1":
        return TASK_DESC_BY_N[n]
    if condition_id == "C2":
        return HUMAN_SIM_INTRO_C2 + h
    if condition_id == "C3":
        return HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN + h
    if condition_id == "C4":
        return (
            HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
            + h
            + "\n"
            + ICL_EXAMPLES_BY_N[n]
        )
    raise KeyError(f"Unknown condition_id: {condition_id}")


CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {"name": "Just describe the task"},
    "C2": {"name": "Task + simulate human psychology experiment"},
    "C3": {"name": "Task + human simulation + limited memory emphasis"},
    "C4": {"name": "Task + human simulation + limited memory + in-context examples"},
}


# ============================
# 2) Stimulus generation (match HTML generateBlock)
# ============================

def _random_choice(rng: random.Random, arr: List[str]) -> str:
    return arr[rng.randrange(len(arr))]


def _shuffle(rng: random.Random, arr: List[int]) -> List[int]:
    a = arr[:]
    rng.shuffle(a)
    return a


@dataclass
class NBackBlock:
    n: int
    buffer_letters: List[str]
    trial_letters: List[str]
    trial_is_target: List[bool]
    full_sequence: List[str]


def generate_block(n: int, rng: random.Random) -> NBackBlock:
    """
    Port of the HTML generateBlock(n) for n >= 1:
    - numTargets = round(TRIALS_PER_BLOCK * TARGET_RATIO)
    - Build totalLen=14+n, choose target indices among slots [n..n+13]
      If target at i: letters[i] = letters[i-n]
      Else: letters[i] random consonant != letters[i-n]
    """
    if n < 1:
        raise ValueError("n must be >= 1 (0-back removed)")

    num_targets = round(TRIALS_PER_BLOCK * TARGET_RATIO)

    buffer_count = n
    total_len = TRIALS_PER_BLOCK + buffer_count

    valid_target_slots = [i + buffer_count for i in range(TRIALS_PER_BLOCK)]
    shuffled = _shuffle(rng, valid_target_slots)
    target_indices = set(shuffled[: min(num_targets, len(shuffled))])

    letters: List[str] = []
    others = CONSONANTS[:]

    for i in range(total_len):
        if i < buffer_count:
            letters.append(_random_choice(rng, others))
        elif i in target_indices:
            letters.append(letters[i - n])
        else:
            prev = letters[i - n]
            pool = [c for c in others if c != prev]
            letters.append(_random_choice(rng, pool if pool else others))

    trial_letters = letters[buffer_count:]
    trial_targets = [(buffer_count + j) in target_indices for j in range(TRIALS_PER_BLOCK)]
    buffer_letters = letters[:buffer_count]

    return NBackBlock(
        n=n,
        buffer_letters=buffer_letters,
        trial_letters=trial_letters,
        trial_is_target=trial_targets,
        full_sequence=letters,
    )


# ============================
# 3) Prompt building
# ============================

def build_block_prompt(condition_id: str, block: NBackBlock) -> str:
    """Prefix from ``prompt_prefix_for``, then stimuli framing (C1 vs human-facing), then format rules."""
    n = block.n
    prefix = prompt_prefix_for(condition_id, n)
    seq_lines = "\n".join([f"{i + 1}: {ch}" for i, ch in enumerate(block.full_sequence)])
    stimulus = "Sequence:\n" + seq_lines + "\n"
    return wrap_stimulus_prompt(prefix, condition_id, stimulus, format_format_rules(n))



# ============================
# 4) Parsing + scoring
# ============================

_TRIAL_LINE_RE = re.compile(r"^trial\s+(\d+):\s*(.+?)\s*$", re.IGNORECASE)
_NORM = NormalDist()


def _norm_no_response_label(s: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", s.strip()).lower()
    if t in ("no response", "no-response", "noresponse"):
        return "No response"
    return None


def _norm_same_different_label(s: str) -> Optional[str]:
    t = re.sub(r"\s+", " ", s.strip()).lower()
    if t == "same":
        return "Same"
    if t in ("different", "non-target", "non target", "nontarget"):
        return "Different"
    # Legacy outputs from older prompts (Target / Non-target)
    if t == "target":
        return "Same"
    return None


def _non_empty_content_lines(text: str) -> List[str]:
    """Split into stripped non-empty lines; skip markdown fence lines."""
    out: List[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("```"):
            continue
        out.append(s)
    return out


def _apply_sequence_line(
    idx: int,
    rest: str,
    n: int,
    trial_map: Dict[int, str],
    buffer_map: Dict[int, Optional[str]],
) -> None:
    """Map global position idx (1..n+14) to buffer or trial slots."""
    if 1 <= idx <= n:
        buffer_map[idx] = _norm_no_response_label(rest)
    elif n + 1 <= idx <= n + TRIALS_PER_BLOCK:
        trial_i = idx - n
        lab = _norm_same_different_label(rest)
        if lab is not None:
            trial_map[trial_i] = lab


def parse_sequence_responses(text: str, n: int) -> Tuple[Dict[int, str], Dict[int, Optional[str]]]:
    """
    Parse one line per sequence position. Expected: trials 1..n -> no response;
    trials n+1 .. n+14 -> same / different.

    Accepts either:
    - Explicit lines: ``trial k: no response`` / ``same`` / ``different``, or
    - Plain lines only (no ``trial`` prefix): first line = position 1, second = position 2, ...

    If any line uses the ``trial N:`` form, only those lines are used (unprefixed lines ignored).
    If none do, all non-empty lines are taken in order as positions 1..K.

    Returns:
      - trial_map: trial index 1..14 -> 'Same'/'Different' (only from valid lines)
      - buffer_map: position 1..n -> 'No response' if line matches, else None if wrong/missing
    """
    trial_map: Dict[int, str] = {}
    buffer_map: Dict[int, Optional[str]] = {i: None for i in range(1, n + 1)}

    lines = _non_empty_content_lines(text)
    indexed: Dict[int, str] = {}
    plain: List[str] = []
    for line in lines:
        m = _TRIAL_LINE_RE.match(line)
        if m:
            indexed[int(m.group(1))] = m.group(2).strip()
        else:
            plain.append(line)

    if indexed:
        for idx, rest in sorted(indexed.items()):
            _apply_sequence_line(idx, rest, n, trial_map, buffer_map)
    else:
        for i, line in enumerate(plain, start=1):
            _apply_sequence_line(i, line, n, trial_map, buffer_map)

    return trial_map, buffer_map


def parse_responses(text: str, n: int) -> Dict[int, str]:
    """Backward-compatible name: returns trial 1..14 -> Same/Different."""
    trial_map, _ = parse_sequence_responses(text, n)
    return trial_map


def score_block(block: NBackBlock, resp_map: Dict[int, str]) -> Dict[str, Any]:
    correct = 0
    answered = 0
    tp = 0
    fp = 0
    fn = 0
    per_trial = []

    for i in range(1, TRIALS_PER_BLOCK + 1):
        expected_is_target = block.trial_is_target[i - 1]
        expected_label = "Same" if expected_is_target else "Different"
        got = resp_map.get(i)
        is_corr = got == expected_label
        if got is not None:
            answered += 1
            if is_corr:
                correct += 1

        if expected_is_target:
            if got == "Same":
                tp += 1
            else:
                fn += 1
        else:
            if got == "Same":
                fp += 1

        per_trial.append(
            {
                "trial": i,
                "letter": block.trial_letters[i - 1],
                "target": expected_is_target,
                "expected_label": expected_label,
                "model_label": got,
                "correct": is_corr if got is not None else None,
            }
        )

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None

    # Requested scoring: d' = Z(precision) - Z(recall).
    # For undefined precision/recall, use 0.5 (neutral Z=0).
    p_for_z = 0.5 if precision is None else float(precision)
    r_for_z = 0.5 if recall is None else float(recall)
    eps = 1e-6
    p_for_z = min(max(p_for_z, eps), 1.0 - eps)
    r_for_z = min(max(r_for_z, eps), 1.0 - eps)
    precision_z = _NORM.inv_cdf(p_for_z)
    recall_z = _NORM.inv_cdf(r_for_z)
    d_prime = precision_z - recall_z

    return {
        "answered": answered,
        "correct": correct,
        "accuracy_over_answered": (correct / answered) if answered else None,
        "accuracy_over_14": correct / TRIALS_PER_BLOCK,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "precision_z": precision_z,
        "recall_z": recall_z,
        "d_prime": d_prime,
        "per_trial": per_trial,
    }


def _rng_nback_session(stimuli_seed: int, pid: int, rep: int) -> random.Random:
    """Deterministic RNG per (participant, repeat) so sessions can run in parallel."""
    payload = f"nback:{stimuli_seed}:{pid}:{rep}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def summarize_nback_condition(
    rows: List[Dict[str, Any]],
    condition_id: str,
    condition_name: str,
) -> Dict[str, Any]:
    by_n = defaultdict(list)
    by_n_answered = defaultdict(list)
    by_n_d_prime = defaultdict(list)
    for r in rows:
        by_n[int(r["n_level"])].append(float(r["acc_over_14"]))
        by_n_answered[int(r["n_level"])].append(float(r["answered"]) / float(TRIALS_PER_BLOCK))
        by_n_d_prime[int(r["n_level"])].append(float(r["d_prime"]))

    acc_over_14_vals = [float(r["acc_over_14"]) for r in rows]
    acc_over_answered_vals = [
        float(r["acc_over_answered"]) for r in rows if r["acc_over_answered"] is not None
    ]
    answered_rate_vals = [float(r["answered"]) / float(TRIALS_PER_BLOCK) for r in rows]
    d_prime_vals = [float(r["d_prime"]) for r in rows]

    acc_over_14_by_n = {str(n): _mean(by_n[n]) for n in sorted(by_n.keys())}
    answered_rate_by_n = {str(n): _mean(by_n_answered[n]) for n in sorted(by_n_answered.keys())}
    d_prime_by_n = {str(n): _mean(by_n_d_prime[n]) for n in sorted(by_n_d_prime.keys())}

    return {
        "condition": condition_id,
        "condition_name": condition_name,
        "n": len(rows),
        "metrics": {
            "d_prime_mean": _mean(d_prime_vals),
            "acc_over_14": _mean(acc_over_14_vals),
            "acc_over_answered": _mean(acc_over_answered_vals),
            "answered_rate": _mean(answered_rate_vals),
        },
        "breakdown": {
            "d_prime_by_n": d_prime_by_n,
            "acc_over_14_by_n": acc_over_14_by_n,
            "answered_rate_by_n": answered_rate_by_n,
        },
    }


def save_acc_by_n_figure(
    out_dir: Path,
    task_name: str,
    condition_id: str,
    acc_over_14_by_n: Dict[str, float],
) -> None:
    xs = [int(k) for k in acc_over_14_by_n.keys()]
    ys = [float(acc_over_14_by_n[str(k)]) for k in xs]
    fig = plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xticks(N_LEVELS)
    plt.xlabel("N level")
    plt.ylabel("Accuracy over 14 trials")
    plt.title(f"N-back ({condition_id}): accuracy by N level")
    save_fig(fig, out_dir / "figures" / task_name / f"{condition_id}_acc_by_n.png")


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    n_repeat: int = 1,
    n_repeats_per_participant: int = 1,
    stimuli_seed: int = 123,
    temperature: float = 0.7,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """``n_repeat`` = number of simulated participants (same as CLI ``--repeat`` / YAML ``n_repeat``)."""
    n_rep = int(n_repeat)
    n_reps = int(n_repeats_per_participant)
    total_sessions = n_rep * n_reps
    workers = resolve_worker_count(total_sessions, max_parallel=max_parallel_participants)
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    def _one_session(session_idx: int) -> List[Dict[str, Any]]:
        pid = session_idx // n_reps + 1
        rep = session_idx % n_reps + 1
        rng = _rng_nback_session(stimuli_seed, pid, rep)
        blocks = {n: generate_block(n, rng) for n in N_LEVELS}
        chunk: List[Dict[str, Any]] = []
        for cond_id, cond in CONDITIONS.items():
            for n in N_LEVELS:
                block = blocks[n]
                prompt = build_block_prompt(cond_id, block)
                resp = llm.generate(
                    prompt,
                    temperature=float(temperature),
                    max_tokens=int(model_cfg.get("max_tokens", 512)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    seed=model_cfg.get("seed"),
                )
                raw = (resp.text or "").strip()

                resp_map, buffer_parsed = parse_sequence_responses(raw, block.n)
                scored = score_block(block, resp_map)

                row = {
                    "participant_id": pid,
                    "repeat_index": rep,
                    "condition_id": cond_id,
                    "condition_name": cond["name"],
                    "n_level": n,
                    "prompt": prompt,
                    "buffer_letters": block.buffer_letters,
                    "trial_letters": block.trial_letters,
                    "trial_is_target": block.trial_is_target,
                    "model_raw": raw,
                    "model_parsed": resp_map,
                    "model_parsed_buffer": buffer_parsed,
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
                chunk.append(row)
        return chunk

    results = [
        row
        for part in map_participants(
            list(range(total_sessions)),
            _one_session,
            max_workers=workers,
        )
        for row in part
    ]

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in CONDITIONS.keys():
        rows = [r for r in results if r["condition_id"] == cond_id]
        cond_summary = summarize_nback_condition(rows, cond_id, CONDITIONS[cond_id]["name"])
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
