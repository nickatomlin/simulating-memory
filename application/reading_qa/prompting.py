from __future__ import annotations

import re
from typing import Dict, List, Optional

from bench.tasks.human_simulation_prefixes import HUMAN_SIM_INTRO_C2, HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from bench.tasks.stimulus_prompt_shell import DIGIT_SPAN_SEP_BEFORE_RULES, wrap_stimulus_prompt

from .data import Question


ANSWER_RE = re.compile(r"^question\s+(\d+):\s*([A-Da-d])\s*$", re.IGNORECASE)
DIFF_RE = re.compile(r"^difficulty:\s*([0-9]{1,2})\s*$", re.IGNORECASE)

TASK_DESC = "Read a passage, answer all multiple-choice questions, then rate reading difficulty."
HUMAN_PROMPT = (
    "The human will have three minutes to read a passage, after which the text will disappear. The human will then be asked to answer ten questions about the text based on their memory and rate reading difficulty."
)

FORMAT_RULES = """Output ONLY lines in this exact format:
Question 1: A
Question 2: B
...
Difficulty: 7
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
    # Keep C4 for compatibility with existing experiment structure.
    "C4": {
        "name": "Task + human + limited memory",
        "prompt_prefix": HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN + HUMAN_PROMPT,
    },
}


def build_prompt(condition_id: str, reading_text: str, questions: List[Question]) -> str:
    lines: List[str] = []
    lines.append("Passage:")
    lines.append(reading_text)
    lines.append("")
    lines.append("Questions:")
    for i, q in enumerate(questions, start=1):
        opts = " ".join(f"{key}) {q.options[key]}" for key in ["A", "B", "C", "D"])
        lines.append(f"Question {i}: {q.question} {opts}")
    lines.append("")
    lines.append("On a scale 1-10, how difficult is the reading?")

    stimulus = "\n".join(lines)
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    return wrap_stimulus_prompt(
        prefix,
        condition_id,
        stimulus,
        FORMAT_RULES,
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


def parse_answers_and_difficulty(text: str) -> Dict[str, object]:
    answers: Dict[int, str] = {}
    difficulty: Optional[int] = None
    parse_errors: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        answer_match = ANSWER_RE.match(line)
        if answer_match:
            q_idx = int(answer_match.group(1))
            answers[q_idx] = answer_match.group(2).upper()
            continue
        diff_match = DIFF_RE.match(line)
        if diff_match:
            difficulty = int(diff_match.group(1))

    if difficulty is None:
        parse_errors.append("difficulty_missing")
    elif difficulty < 1 or difficulty > 10:
        parse_errors.append("difficulty_out_of_range")
        difficulty = None

    return {
        "answers": answers,
        "difficulty": difficulty,
        "parse_errors": parse_errors,
    }
