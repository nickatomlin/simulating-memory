from __future__ import annotations
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count
from ..core.plotting import save_fig

from .human_simulation_prefixes import HUMAN_SIM_INTRO_C2, HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from .stimulus_prompt_shell import wrap_stimulus_prompt

TASK_NAME = "variable_mapping"

####
# PROMPTS (verbatim from variable_mapping.ipynb)
####

TASK_DESC = """You will see a series of sentences describing where people live. Try to remember where each person lives. Pay attention: people will occasionally move to a new city! After every two sentences, you will be asked: “Where does [Name] live?” When that happens, respond with the city where the person currently lives.
"""

HUMAN_PROMPT = """The human will see a series of sentences describing where people live. Sentences are presented one at a time. Each sentence disappears before the next sentence/question appears, and the human cannot see previous sentences again. They are asked to try to remember where each person lives. Pay attention: people will occasionally move to a new city. After every two sentences, the human will be asked: “Where does [Name] live?” When that happens, respond with the city where the person currently lives."""


FORMAT_RULES = """Output ONLY lines in this format:
Question 1: A
Question 2: B
...

One line per question. No extra text.
"""

ICL_EXAMPLES = """Here are example results from previous human participants.

Alice lives in Sydney.
Alice moved to Reno.
Question 1:
Where does Alice live?
A) Sydney  B) Reno  C) Paris  D) Tokyo
Bob lives in Sydney.
Alice moved to Tokyo.
Question 2:
Where does Alice live?
A) Sydney  B) Reno  C) Paris  D) Tokyo
Bob moved to Reno.
Carol lives in Sydney.
Question 3:
Where does Bob live?
A) Sydney  B) Reno  C) Paris  D) Tokyo

Response:
Question 1: B (Correct)
Question 2: D (Correct)
Question 3: A (Wrong)
"""

CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {
        "name": "Just describe the task",
        "prompt_prefix": TASK_DESC,
    },
    "C2": {
        "name": "Task + simulate human",
        "prompt_prefix": HUMAN_SIM_INTRO_C2 + HUMAN_PROMPT,
    },
    "C3": {
        "name": "Task + human + limited memory",
        "prompt_prefix": HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN + HUMAN_PROMPT,
    },
    "C4": {
        "name": "Task + human + limited memory + examples",
        "prompt_prefix": (
            HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
            + HUMAN_PROMPT
            + "\n\n"
            + ICL_EXAMPLES
        ),
    },
}


TURNS_PER_QUESTION = 2
N_QUESTIONS = 10
N_ASSIGNMENTS = TURNS_PER_QUESTION * N_QUESTIONS
# Distinct name→city relations to build before random moves; score ceiling is this value per run.
TARGET_RELATIONS = 10


def load_json_list(path: str, key: str) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    val = data.get(key, [])
    if not isinstance(val, list):
        raise ValueError(f"{path} must contain a list under key '{key}'.")
    return [str(x).strip() for x in val if isinstance(x, str) and x.strip()]


def generate_one_run(
    names: List[str],
    cities: List[str],
    rng: random.Random,
) -> Dict[str, Any]:
    mapping: Dict[str, str] = {}
    assignments: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []

    # First `cap` turns each introduce a new person so len(mapping) reaches `cap` by turn `cap`;
    # the last question can then have relation_count == cap (score ceiling). Later turns randomize names.
    cap = min(TARGET_RELATIONS, len(names))
    intro_order = names[:]
    rng.shuffle(intro_order)
    intro_order = intro_order[:cap]

    for turn in range(1, N_ASSIGNMENTS + 1):
        if turn <= cap:
            name = intro_order[turn - 1]
        else:
            name = rng.choice(names)
        had_before = name in mapping and mapping[name]

        if not had_before:
            city = rng.choice(cities)
            mapping[name] = city
            statement = f"{name} lives in {city}."
        else:
            current = mapping[name]
            others = [c for c in cities if c != current]
            city = rng.choice(others) if others else current
            mapping[name] = city
            statement = f"{name} moved to {city}."

        assignments.append(
            {
                "turn": turn,
                "name": name,
                "city": city,
                "statement": statement,
            }
        )

        if turn % TURNS_PER_QUESTION == 0:
            mapped_names = [n for n in mapping if mapping[n] and mapping[n] in cities]
            if not mapped_names:
                continue
            q_name = rng.choice(mapped_names)
            correct_city = mapping[q_name]
            wrong = [c for c in cities if c != correct_city]
            rng.shuffle(wrong)
            options = [correct_city] + wrong[:3]
            rng.shuffle(options)
            questions.append(
                {
                    "question_index": len(questions) + 1,
                    "name": q_name,
                    "correct_city": correct_city,
                    "options": options,
                    "relation_count": len(mapping),
                }
            )

    return {"assignments": assignments, "questions": questions}


def generate_runs(
    names: List[str],
    cities: List[str],
    n_repeat: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not names or not cities:
        raise ValueError("Need names and cities.")
    rng = random.Random(seed)
    runs: List[Dict[str, Any]] = []
    for r in range(n_repeat):
        run = generate_one_run(names, cities, rng)
        runs.append({"repeat_index": r + 1, **run})
    return runs


def build_prompt(condition_id: str, run: Dict[str, Any]) -> str:
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    assign_block = "\n".join(a["statement"] for a in run["assignments"])
    q_lines: List[str] = []
    for q in run["questions"]:
        opts = " ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(q["options"]))
        q_lines.append(
            f"Question {q['question_index']}: Where does {q['name']} live? {opts}"
        )
    stimulus = (
        "Assignments (in order):\n"
        + assign_block
        + "\n\n"
        + "\n".join(q_lines)
    )
    return wrap_stimulus_prompt(prefix, condition_id, stimulus, FORMAT_RULES)


def build_incremental_prompt(condition_id: str, run: Dict[str, Any], question_index: int) -> str:
    """
    Build one prompt for a single question with cumulative history:
    question k sees first 2*k assignment statements and only Question k.
    Same stimuli shell as ``build_prompt`` (C1 vs C2–C4 framing via ``wrap_stimulus_prompt``).
    """
    if question_index < 1 or question_index > len(run["questions"]):
        raise ValueError(f"Invalid question_index={question_index}")

    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    n_assignments = TURNS_PER_QUESTION * question_index
    assign_block = "\n".join(a["statement"] for a in run["assignments"][:n_assignments])
    q = run["questions"][question_index - 1]
    opts = " ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(q["options"]))
    q_line = f"Question {q['question_index']}: Where does {q['name']} live? {opts}"

    stimulus = (
        f"Assignments so far (first {n_assignments}, in order):\n"
        + assign_block
        + "\n\n"
        + q_line
        + "\n\nYour answer:\n"
    )
    incremental_format_rules = (
        "Output ONLY one line in this format:\n"
        f"Question {q['question_index']}: \n"
        "No extra text.\n"
    )
    return wrap_stimulus_prompt(
        prefix,
        condition_id,
        stimulus,
        incremental_format_rules,
    )


LINE_RE = re.compile(r"^question\s+(\d+):\s*([A-Da-d])(?:\b.*)?$", re.IGNORECASE)


def parse_answers(text: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        out[idx] = m.group(2).upper()
    return out


def score_run(questions: List[Dict[str, Any]], answer_map: Dict[int, str]) -> Dict[str, Any]:
    """
    Score = maximum number of name–city relations that existed when the model
    last answered a question correctly. The game ends on the first incorrect
    or invalid answer.
    """
    score = 0
    first_error_at: Optional[int] = None

    for q in questions:
        idx = q["question_index"]
        correct_city = q["correct_city"]
        options = q["options"]
        letter = answer_map.get(idx)

        if letter and 0 <= ord(letter) - ord("A") < len(options):
            chosen = options[ord(letter) - ord("A")]
            is_correct = chosen == correct_city
        else:
            is_correct = False

        if not is_correct:
            first_error_at = idx
            break

        score = q["relation_count"]

    return {
        "score": score,
        "first_error_at": first_error_at,
        "n_questions": len(questions),
    }


def summarize_variable_mapping_rows(
    rows: List[Dict[str, Any]],
    condition_id: str,
    condition_name: str,
    *,
    n_participants: int,
    n_runs_per_participant: int,
) -> Dict[str, Any]:
    run_scores = [int(row["metrics"]["score"]) for row in rows]
    participant_scores: List[int] = []
    for p in range(n_participants):
        start = p * n_runs_per_participant
        end = start + n_runs_per_participant
        best = max(run_scores[start:end]) if end <= len(run_scores) else 0
        participant_scores.append(best)
    mean_score = float(sum(participant_scores) / len(participant_scores)) if participant_scores else 0.0
    return {
        "condition": condition_id,
        "condition_name": condition_name,
        "n": len(rows),
        "n_participants": n_participants,
        "n_runs_per_participant": n_runs_per_participant,
        "metrics": {
            "score_mean": mean_score,
            "score_min": int(min(participant_scores)) if participant_scores else 0,
            "score_max": int(max(participant_scores)) if participant_scores else 0,
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
    plt.bar(cond_ids, means, color="seagreen", edgecolor="black")
    plt.xlabel("Condition")
    plt.ylabel("Mean score (best of 3 runs per participant)")
    plt.title("Variable mapping: mean score by condition")
    save_fig(fig, out_dir / "figures" / task_name / "score_by_condition.png")


N_RUNS_PER_PARTICIPANT = 3


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    names_json_path: str,
    city_json_path: str,
    n_repeat: int = 1,
    n_runs_per_participant: int = N_RUNS_PER_PARTICIPANT,
    stimuli_seed: int = 42,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """
    ``n_repeat`` is the number of participants (same as CLI ``--repeat`` / ``n_participants`` in YAML).
    Each participant completes ``n_runs_per_participant`` runs; scoring uses the best run per participant.
    """
    names = load_json_list(names_json_path, "names")
    cities = load_json_list(city_json_path, "cities")
    n_participants = n_repeat
    total_runs = n_participants * n_runs_per_participant
    runs = generate_runs(names, cities, n_repeat=total_runs, seed=stimuli_seed)
    workers = resolve_worker_count(total_runs, max_parallel=max_parallel_participants)

    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    for cond_id in ["C1", "C2", "C3", "C4"]:
        rows: List[Dict[str, Any]] = []
        run_scores: List[int] = []

        def _one_run(run_idx: int) -> Dict[str, Any]:
            run = runs[run_idx]
            answer_map: Dict[int, str] = {}
            per_question_raw: List[Dict[str, Any]] = []
            incremental_prompts: List[Dict[str, Any]] = []

            for q in run["questions"]:
                q_idx = int(q["question_index"])
                q_prompt = build_incremental_prompt(cond_id, run, q_idx)
                resp = llm.generate(
                    q_prompt,
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    max_tokens=int(model_cfg.get("max_tokens", 128)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    seed=model_cfg.get("seed"),
                )
                parsed = parse_answers(resp.text)
                if q_idx in parsed:
                    answer_map[q_idx] = parsed[q_idx]
                per_question_raw.append({"question_index": q_idx, "raw": resp.text})
                incremental_prompts.append({"question_index": q_idx, "prompt": q_prompt})

            scored = score_run(run["questions"], answer_map)
            participant_id = run_idx // n_runs_per_participant

            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{participant_id}:run{run_idx % n_runs_per_participant}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "participant_id": participant_id,
                "run_index": run_idx,
                "repeat_index": run["repeat_index"],
                "prompt": build_prompt(cond_id, run),
                "incremental_prompts": incremental_prompts,
                "assignments": run["assignments"],
                "questions": run["questions"],
                "raw": "\n".join(
                    f"Question {x['question_index']} raw:\n{x['raw']}" for x in per_question_raw
                ),
                "raw_by_question": per_question_raw,
                "parsed_answers": answer_map,
                "metrics": scored,
            }
            sink.append(row)
            return row

        row_dicts = map_participants(
            list(range(total_runs)),
            _one_run,
            max_workers=workers,
        )
        for row in row_dicts:
            rows.append(row)
            run_scores.append(int(row["metrics"]["score"]))

        cond_summaries.append(
            summarize_variable_mapping_rows(
                rows,
                cond_id,
                CONDITIONS[cond_id]["name"],
                n_participants=n_participants,
                n_runs_per_participant=n_runs_per_participant,
            )
        )
        all_rows.extend(rows)

    # Figure: mean score by condition (mean of best-of-3 per participant)
    save_score_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {
        "task": TASK_NAME,
        "n_participants": int(n_repeat),
        "n_runs_per_participant": int(n_runs_per_participant),
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
