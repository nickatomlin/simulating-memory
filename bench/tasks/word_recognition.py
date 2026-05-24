from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import matplotlib.pyplot as plt

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count
from ..core.plotting import save_fig

from .human_simulation_prefixes import HUMAN_SIM_INTRO_C2, HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from .stimulus_prompt_shell import wrap_stimulus_prompt

TASK_NAME = "word_recognition"

####
# PROMPTS (verbatim from word_recognition.ipynb)
####

TASK_DESC = """Words will appear. For each word, decide whether it has already appeared earlier in the list. Select "old" if the word has appeared before. Select "new" if this is the first time you have seen the word. The first word is always "new".
"""

HUMAN_PROMPT="""Words will appear one at a time. For each word, the human will be asked to decide whether it has already appeared earlier in the list. The human will be asked to select "old" if the word has appeared before. The human will be asked to select "new" if this is their first time seeing the word. The first word is always "new".
"""

FORMAT_RULES = """Output ONLY lines in this exact format:
trial 1: new
trial 2: old
trial 3: new
...

One line per trial. Use exactly "old" or "new". No extra text.
"""

ICL_EXAMPLES = """Here are example results from previous participants.
Example 1:
Trials:
1: apple
2: chair
3: apple
4: river
Response:
trial 1: New (Correct)
trial 2: New (Correct)
trial 3: Old (Correct)
trial 4: New (Correct)

Example 2:
Trials:
1: apple
2: chair
3: apple
4: river
5: chair
6: artist 
Response:
trial 1: New (Correct)
trial 2: New (Correct)
trial 3: Old (Correct)
trial 4: New (Correct)
trial 5: New (Wrong)
trial 6: Old (Correct)
"""

CONDITIONS = {
    "C1": {
        "name": "Just describe the task",
        "prompt_prefix": TASK_DESC,
    },
    "C2": {
        "name": "Task + simulate human psychology experiment",
        "prompt_prefix": HUMAN_SIM_INTRO_C2 + HUMAN_PROMPT,
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
            + "\n\n"
            + ICL_EXAMPLES
        ),
    },
}


def load_words(words_json_path: str) -> List[str]:
    data = json.loads(Path(words_json_path).read_text(encoding="utf-8"))
    words = data.get("words", [])
    if not isinstance(words, list):
        raise ValueError("words.json must contain a list under key 'words'.")
    return [str(w).strip() for w in words if isinstance(w, str) and w.strip()]


def generate_one_game(
    word_pool: List[str],
    rng: random.Random,
    max_trials: int,
) -> List[Dict[str, Any]]:
    """
    Generate one game sequence matching HTML logic:
    - Trial 1: always New (pick from unseen, add to seen).
    - Trial 2+: 50% Old (pick from seen), 50% New (pick from unseen, add to seen).
    - Stop when unseen is empty or max_trials reached.
    """
    seen: List[str] = []
    unseen = word_pool[:]
    rng.shuffle(unseen)
    trials: List[Dict[str, Any]] = []

    for t in range(max_trials):
        if t == 0:
            is_old = False
        else:
            is_old = rng.random() < 0.5

        if is_old:
            if not seen:
                is_old = False
            else:
                word = rng.choice(seen)

        if not is_old:
            if not unseen:
                break
            word = unseen.pop()
            seen.append(word)

        trials.append({
            "trial_index": len(trials) + 1,
            "word": word,
            "is_old": is_old,
        })

    return trials


def generate_runs(
    all_words: List[str],
    n_repeat: int,
    seed: int,
    max_trials_per_game: int = 200,
) -> List[Dict[str, Any]]:
    if len(all_words) < 2:
        raise ValueError("Need at least 2 words.")
    rng = random.Random(seed)
    runs = []
    for r in range(n_repeat):
        trials = generate_one_game(all_words, rng, max_trials_per_game)
        runs.append({"repeat_index": r + 1, "trials": trials})
    return runs


def build_prompt(condition_id: str, run: Dict[str, Any]) -> str:
    trials = run["trials"]
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    trials_block = "\n".join([f"{t['trial_index']}: {t['word']}" for t in trials])
    stimulus = (
        "\n"
        + trials_block
        + "\n\n\n"
    )
    return wrap_stimulus_prompt(prefix, condition_id, stimulus, FORMAT_RULES)


build_run_prompt = build_prompt


LINE_RE = re.compile(r"^trial\s+(\d+):\s*(Old|New)\s*$", re.IGNORECASE)


def parse_responses(text: str) -> Dict[int, str]:
    """Returns mapping trial_index (1-based) -> 'Old' or 'New'."""
    out: Dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        val = m.group(2).capitalize()
        if val in ("Old", "New"):
            out[idx] = val
    return out


MAX_ERRORS_BEFORE_STOP = 3


def score_game(
    trials: List[Dict[str, Any]],
    resp_map: Dict[int, str],
    max_errors: int = MAX_ERRORS_BEFORE_STOP,
) -> Dict[str, Any]:
    """
    Score = number of correct answers when the participant has made max_errors mistakes.
    Task continues until max_errors incorrect answers; score is count of correct up to that point.
    """
    score = 0
    error_count = 0
    third_error_at: Optional[int] = None
    per_trial = []

    for t in trials:
        idx = t["trial_index"]
        word = t["word"]
        expected_old = t["is_old"]
        expected = "Old" if expected_old else "New"
        got = resp_map.get(idx)

        if got is None:
            is_correct = None
        else:
            is_correct = got == expected
            if is_correct:
                score += 1
            else:
                error_count += 1
                if error_count == max_errors:
                    third_error_at = idx

        per_trial.append({
            "trial": idx,
            "word": word,
            "expected": expected,
            "model_response": got,
            "correct": is_correct,
        })
        if third_error_at is not None:
            break

    return {
        "score": score,
        "first_error_at": third_error_at,  # kept for compatibility; now "error_at_stop"
        "n_trials": len(trials),
        "per_trial": per_trial,
        "errors_at_stop": error_count,
    }


def summarize_game_rows(
    rows: List[Dict[str, Any]],
    condition_id: str,
    condition_name: str,
) -> Dict[str, Any]:
    scores = [int(r["metrics"]["score"]) for r in rows]
    return {
        "condition": condition_id,
        "condition_name": condition_name,
        "n": len(rows),
        "metrics": {
            "score_mean": sum(scores) / len(scores) if scores else 0.0,
            "score_min": min(scores) if scores else 0,
            "score_max": max(scores) if scores else 0,
        },
    }


def save_score_by_condition_figure(
    out_dir: Path,
    task_name: str,
    cond_summaries: List[Dict[str, Any]],
) -> None:
    cond_ids = [c["condition"] for c in cond_summaries]
    means = [c["metrics"]["score_mean"] for c in cond_summaries]
    fig = plt.figure()
    plt.bar(cond_ids, means, color="steelblue", edgecolor="black")
    plt.xlabel("Condition")
    plt.ylabel("Mean score (correct when 3 errors reached)")
    plt.title("Word recognition: mean score by condition")
    save_fig(fig, out_dir / "figures" / task_name / "score_by_condition.png")


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    words_json_path: str,
    n_repeat: int = 1,
    max_trials_per_game: int = 200,
    stimuli_seed: int = 42,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """``n_repeat`` = number of participants; each plays one game per condition (4 × n_repeat LLM calls)."""
    words = load_words(words_json_path)
    runs = generate_runs(
        words,
        n_repeat=n_repeat,
        seed=stimuli_seed,
        max_trials_per_game=max_trials_per_game,
    )

    n_runs = len(runs)
    workers = resolve_worker_count(n_runs, max_parallel=max_parallel_participants)

    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    for cond_id in ["C1", "C2", "C3", "C4"]:
        rows: List[Dict[str, Any]] = []
        scores: List[int] = []

        def _one_run(run_idx: int) -> Dict[str, Any]:
            run = runs[run_idx]
            prompt = build_prompt(cond_id, run)
            resp = llm.generate(
                prompt,
                temperature=float(model_cfg.get("temperature", 0.0)),
                max_tokens=int(model_cfg.get("max_tokens", 512)),
                top_p=float(model_cfg.get("top_p", 1.0)),
                seed=model_cfg.get("seed"),
            )
            resp_map = parse_responses(resp.text)
            trials = run["trials"]
            scored = score_game(trials, resp_map)
            row = {
                "id": f"{TASK_NAME}:{cond_id}:{run['repeat_index']}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "repeat_index": run["repeat_index"],
                "prompt": prompt,
                "gold_trials": trials,
                "raw": resp.text,
                "metrics": {
                    "score": scored["score"],
                    "first_error_at": scored["first_error_at"],
                    "n_trials": scored["n_trials"],
                },
                "per_trial": scored["per_trial"],
            }
            sink.append(row)
            return row

        row_dicts = map_participants(
            list(range(n_runs)),
            _one_run,
            max_workers=workers,
        )
        for row in row_dicts:
            rows.append(row)
            scores.append(row["metrics"]["score"])

        cond_summaries.append(summarize_game_rows(rows, cond_id, CONDITIONS[cond_id]["name"]))
        all_rows.extend(rows)

    # Figure: mean score by condition
    save_score_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {
        "task": TASK_NAME,
        "n_participants": int(n_repeat),
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
