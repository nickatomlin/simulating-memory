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

TASK_NAME = "craft_task"

####
# PROMPTS (verbatim from craft_task.ipynb)
####

TASK_DESC = """You will study a set of materials and crafting rules. Memorize how items combine. Answer five questions from memory. There are three trials in total.
"""

HUMAN_PROMPT = """The human will study a set of materials and crafting rules. The human will have one minute to memorize how items combine, and then the rules will disappear. After the rules disappear, they will be asked to answer five questions from memory. There will be three trials in total.
"""

FORMAT_RULES = """Output ONLY lines in this format:
Question 1: A
Question 2: B
...
One line per question. No extra text.
"""

ICL_EXAMPLES = """Here are example results from previous human participants.

Rules: 
"A and B combine to form C.",
"C and D combine to form E.",
"B and F combine to form D.",
"E and F combine to form A."

Question 1: "Which pair produces D?"
"A": "B and F",
"B": "C and B"
Answer 1: A (Correct)

Question 2: "Which pair produces E?"
"A": "C and D",
"B": "A and F"
Answer 2: B (Wrong)
"""

CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {"name": "Just describe the task", "prompt_prefix": TASK_DESC},
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


def load_data(data_json_path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(data_json_path).read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"{data_json_path} must contain a list under key 'items'.")
    return items


def craft_to_text(item: Dict[str, Any]) -> str:
    materials = item.get("items") or []
    rules = item.get("rules_text") or []
    mat_line = "Materials: " + ", ".join(materials) + "."
    rule_lines = "\n".join("- " + r for r in rules)
    return mat_line + "\nCrafting rules:\n" + rule_lines


def _format_mcq_choices(choices: Dict[str, Any]) -> str:
    """Build 'A. ... B. ...' for the prompt. Supports A/B or Choice 1/Choice 2 (craft_task.json)."""
    if not choices:
        return "A.  B."
    a = choices.get("A")
    b = choices.get("B")
    if a is None and b is None:
        a = choices.get("Choice 1")
        b = choices.get("Choice 2")
    if a is None and b is None:
        vals = [str(v) for v in choices.values() if v is not None]
        if len(vals) >= 2:
            a, b = vals[0], vals[1]
        elif len(vals) == 1:
            a = vals[0]
            b = ""
        else:
            a, b = "", ""
    return f"A. {a or ''} B. {b or ''}"


def build_prompt(condition_id: str, item: Dict[str, Any]) -> str:
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    craft_block = craft_to_text(item)
    q_lines: List[str] = []
    for i, q in enumerate(item.get("questions", []), start=1):
        choices = q.get("choices") or {}
        opts = _format_mcq_choices(choices)
        q_lines.append(f"Question {i}: {q.get('prompt', '')} {opts}")
    stimulus = (
        "Crafting DAG:\n"
        + craft_block
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


def score_craft(
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


def summarize_craft_rows(
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
    plt.bar(cond_ids, means, color="teal", edgecolor="black")
    plt.xlabel("Condition")
    plt.ylabel("Mean accuracy (per task)")
    plt.title("Craft task: mean accuracy by condition")
    save_fig(fig, out_dir / "figures" / task_name / "accuracy_by_condition.png")


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    data_json_path: str,
    n_participants: int = 1,
    n_tasks: Optional[int] = None,
    stimuli_seed: int = 42,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """Run craft_task for conditions C1..C4.

    Each *participant* completes every selected crafting task (each task has 5 MCQs).
    By default all tasks in the JSON are used (e.g. 3 tasks × 5 = 15 questions per
    participant per condition). ``n_tasks`` optionally limits how many tasks are sampled
    for quick runs.
    """
    all_items = load_data(data_json_path)
    if not all_items:
        raise ValueError(f"No tasks in {data_json_path}.")

    rng = random.Random(stimuli_seed)
    n = min(n_tasks, len(all_items)) if n_tasks is not None else len(all_items)
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
                task_id = item.get("task_id", "")
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
                scored = score_craft(questions, answer_map)
                row = {
                    "id": f"{TASK_NAME}:{cond_id}:p{participant_index}:{task_id}",
                    "condition_id": cond_id,
                    "condition_name": CONDITIONS[cond_id]["name"],
                    "participant_id": participant_index - 1,
                    "repeat_index": participant_index,
                    "task_id": task_id,
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

        cond_summaries.append(summarize_craft_rows(rows, cond_id, CONDITIONS[cond_id]["name"]))
        all_rows.extend(rows)

    # Figure: mean accuracy by condition
    save_accuracy_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "n_tasks_per_participant": len(items),
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
