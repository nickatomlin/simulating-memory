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

TASK_NAME = "map_task"

####
# PROMPTS (verbatim from map_task.ipynb)
####

TASK_DESC = """You will study a map of locations and roads. Some of these locations are connected to each other by roads. Memorize which locations are connected then answer five questions about possible routes. There will be three trials in total.
"""


HUMAN_PROMPT = """The human will study a map of locations. Some of these locations are connected to each other by roads. The human will have one minute to memorize which locations are connected, and then the map will disappear. After the map disappears, the human will be asked to answer five questions about ways to get from one location to another. Note that you can only travel along the roads that connect those locations. There will be three trials in total.
"""

FORMAT_RULES = """Output ONLY lines in this format:
Question 1: A
Question 2: B
...

One line per question. No extra text.
"""

ICL_EXAMPLES = """Here are example results from previous human participants.
MAP 
Locations:
- Loft
- Clinic
- Gallery
- Market
- Stadium
- Ferry

Paths:
- Loft <-> Market
- Market <-> Ferry
- Ferry <-> Clinic
- Clinic <-> Gallery
- Gallery <-> Stadium
- Market <-> Stadium
- Loft <-> Gallery

QUESTIONS
Q1) You are at Loft. How do you get to Clinic?
A) Loft -> Market -> Ferry -> Clinic
B) Loft -> Stadium -> Clinic
Answer: A (Correct)

Q2) You are at Stadium. How do you get to Clinic?
A) Stadium -> Gallery -> Clinic
B) Stadium -> Market -> Loft -> Clinic
Answer: A (Correct)

Q3) You are at Market. How do you get to Gallery?
A) Market -> Loft -> Gallery
B) Market -> Clinic -> Gallery
Answer: B (Wrong)

Q4) You are at Ferry. How do you get to Stadium?
A) Ferry -> Clinic -> Gallery -> Stadium
B) Ferry -> Market -> Gallery -> Stadium
Answer: A (Correct)

Q5) You are at Loft. How do you get to Stadium?
A) Loft -> Market -> Stadium
B) Loft -> Gallery -> Stadium
Answer: A  (Wrong)
"""


CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {"name": "Just describe the task", "prompt_prefix": TASK_DESC},
    "C2": {
        "name": "Task + simulate human",
        "prompt_prefix": HUMAN_SIM_INTRO_C2 + " " + HUMAN_PROMPT,
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


def load_data(data_json_path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(data_json_path).read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"{data_json_path} must contain a list under key 'items'.")
    return items


def map_to_text(item: Dict[str, Any]) -> str:
    locs = item.get("locations") or []
    edges = item.get("edges") or []
    loc_line = "Locations: " + ", ".join(locs) + "."
    road_str = "; ".join(f"{e[0]} — {e[1]}" for e in edges)
    road_line = (
        "Roads (you can travel directly between these pairs): " + road_str + "."
    )
    return loc_line + "\n" + road_line


def build_prompt(condition_id: str, item: Dict[str, Any]) -> str:
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    map_block = map_to_text(item)
    q_lines: List[str] = []
    for i, q in enumerate(item.get("questions", []), start=1):
        choices = q.get("choices", [])
        opts = " ".join(
            f"{c.get('choice_id', '')}. " + " -> ".join(c.get("route", []))
            for c in choices
        )
        q_lines.append(f"Question {i}: {q.get('prompt', '')} {opts}")
    stimulus = (
        "Map:\n"
        + map_block
        + "\n\n------\n"
        + "Questions:\n"
        + "\n".join(q_lines)
    )
    return wrap_stimulus_prompt(prefix, condition_id, stimulus, FORMAT_RULES)


LINE_RE = re.compile(r"^question\s+(\d+):\s*([A-Ba-b])\s*$", re.IGNORECASE)


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


def score_map(
    questions: List[Dict[str, Any]],
    answer_map: Dict[int, str],
) -> Dict[str, Any]:
    correct = 0
    for i, q in enumerate(questions, start=1):
        ans = (q.get("answer") or "").strip().upper()
        if answer_map.get(i) == ans:
            correct += 1
    n = len(questions)
    return {
        "correct": correct,
        "total": n,
        "accuracy": correct / n if n else 0.0,
    }


def summarize_map_rows(
    rows: List[Dict[str, Any]],
    condition_id: str,
    condition_name: str,
) -> Dict[str, Any]:
    accuracies = [float(r["metrics"]["accuracy"]) for r in rows]
    correct_total = sum(int(r["metrics"]["correct"]) for r in rows) if rows else 0
    total_questions = sum(int(r["metrics"]["total"]) for r in rows) if rows else 0
    return {
        "condition": condition_id,
        "condition_name": condition_name,
        "n": len(rows),
        "metrics": {
            "accuracy_mean": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "correct_total": correct_total,
            "total_questions": total_questions,
        },
    }


def save_accuracy_by_condition_figure(
    out_dir: Path,
    task_name: str,
    cond_summaries: List[Dict[str, Any]],
) -> None:
    cond_ids = [c["condition"] for c in cond_summaries]
    means = [c["metrics"]["accuracy_mean"] for c in cond_summaries]
    fig = plt.figure()
    plt.bar(cond_ids, means, color="darkorange", edgecolor="black")
    plt.xlabel("Condition")
    plt.ylabel("Mean accuracy (per map)")
    plt.title("Map task: mean accuracy by condition")
    save_fig(fig, out_dir / "figures" / task_name / "accuracy_by_condition.png")


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    data_json_path: str,
    n_participants: int = 1,
    n_maps: Optional[int] = None,
    stimuli_seed: int = 42,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """Each participant completes every map in the selected set (``n_maps`` subsets the pool)."""
    all_items = load_data(data_json_path)
    if not all_items:
        raise ValueError(f"No maps in {data_json_path}.")

    rng = random.Random(stimuli_seed)
    n = min(n_maps, len(all_items)) if n_maps is not None else len(all_items)
    items = rng.sample(all_items, n)

    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)

    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    for cond_id in ["C1", "C2", "C3", "C4"]:
        rows: List[Dict[str, Any]] = []
        accuracies: List[float] = []
        corrects: List[int] = []

        def _one_participant(participant_index: int) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for item in items:
                map_id = item.get("map_id", "")
                questions = item.get("questions", [])
                prompt = build_prompt(cond_id, item)
                resp = llm.generate(
                    prompt,
                    temperature=float(model_cfg.get("temperature", 0.0)),
                    max_tokens=int(model_cfg.get("max_tokens", 512)),
                    top_p=float(model_cfg.get("top_p", 1.0)),
                    seed=model_cfg.get("seed"),
                )
                answer_map = parse_answers(resp.text)
                scored = score_map(questions, answer_map)
                row = {
                    "id": f"{TASK_NAME}:{cond_id}:p{participant_index}:{map_id}",
                    "condition_id": cond_id,
                    "condition_name": CONDITIONS[cond_id]["name"],
                    "participant_id": participant_index - 1,
                    "repeat_index": participant_index,
                    "map_id": map_id,
                    "difficulty": item.get("difficulty", ""),
                    "prompt": prompt,
                    "questions": questions,
                    "raw": resp.text,
                    "metrics": scored,
                }
                sink.append(row)
                out.append(row)
            return out

        per_p = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        for part_rows in per_p:
            for row in part_rows:
                rows.append(row)
                accuracies.append(row["metrics"]["accuracy"])
                corrects.append(row["metrics"]["correct"])

        cond_summaries.append(summarize_map_rows(rows, cond_id, CONDITIONS[cond_id]["name"]))
        all_rows.extend(rows)

    # Figure: mean accuracy by condition
    save_accuracy_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "n_maps_per_participant": len(items),
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
