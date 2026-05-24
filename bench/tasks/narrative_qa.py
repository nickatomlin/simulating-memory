from __future__ import annotations
import hashlib
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

TASK_NAME = "narrative_qa"

####
# PROMPTS (verbatim from narrative_qa.ipynb)
####

TASK_DESC = """Read a passage, and then answer ten questions."""

HUMAN_PROMPT = """The human will have three minutes to read a passage, after which the text will disappear. The human will then be asked to answer ten questions about the text.
"""

FORMAT_RULES = """Output ONLY lines in this format:
Question 1: A
Question 2: B
...
One line per question. No extra text.
"""

ICL_EXAMPLES = """Here are example results from previous human participants.
Example 1:
Story: "On a chilly autumn morning, Clara woke up feeling unusually restless. The golden leaves outside her window danced in the wind, signaling the arrival of fall. She had planned a quiet day at home, but an old letter she found tucked inside a dusty book changed everything. The letter was from her grandfather, written decades ago, inviting her to visit a small village called Willowbrook. Intrigued and eager for adventure, Clara packed a small bag with essentials and set off on her journey. The first event occurred when she boarded the early train from her city to the countryside, the rhythmic clatter of the wheels soothing yet filled with anticipation. Upon arrival, Clara stepped onto the platform of Willowbrook, greeted by the crisp air and the scent of pine trees. She noticed a quaint café nearby and decided to stop for breakfast, savoring warm pastries and strong coffee while studying a map of the village. Event three unfolded as she wandered through the narrow streets, admiring colorful houses and chatting with friendly locals who shared stories about her grandfather's kindness and community spirit. Encouraged, Clara made her way to the village library, where she hoped to find more clues about her grandfather's past. Inside, the librarian handed her a faded photo album, revealing images of her grandfather as a young man, participating in village festivals and helping neighbors. This was event four. Inspired, Clara decided to explore the nearby forest mentioned in the letter. As she walked beneath towering oaks and maples, the sunlight filtering through the leaves created a mosaic of light on the ground. Suddenly, event five occurred: she stumbled upon a hidden path marked by a wooden sign pointing to an old cabin. Curious, Clara followed the trail and arrived at the cabin, its weathered door slightly ajar. Inside, she found artifacts and journals belonging to her grandfather, detailing his dreams and adventures. Event six was the discovery of a map inside one journal, indicating a secret spot in the forest where her grandfather had planted a time capsule. Excited, Clara set off to find this location, navigating through thick underbrush and crossing a small creek. After some searching, event seven happened when she uncovered a metal box buried beneath a large oak tree. Inside were letters, photographs, and mementos from her grandfather's youth, offering a glimpse into his life and values. Feeling a deep connection, Clara decided to share these treasures with the villagers. Event eight was organizing a small gathering at the village hall, where she presented the time capsule contents and listened to stories from elderly residents who remembered her grandfather fondly. The community's warmth made Clara feel at home. As the sun began to set, event nine unfolded when Clara took a quiet walk along the riverbank, reflecting on the day's discoveries and the unexpected bond she had formed with the village. She realized that this journey had changed her perspective on family and belonging. Finally, event ten occurred as Clara returned to the train station, ready to head back to the city but promising herself to visit Willowbrook again soon. The letter had sparked an unexpected journey, transforming a restless morning into a memorable adventure filled with connection, history, and newfound purpose."

Question 1: "What did Clara do first after waking up on the chilly autumn morning?"
"A": "Found an old letter inside a dusty book",
"B": "Packed a bag with essentials",
"C": "Boarded the early train to the countryside",
"D": "Stopped at a café for breakfast"
Answer 1: A (Correct)

Question 2: "What was Clara's immediate action after reading the letter from her grandfather?"
"A": "She packed a small bag with essentials",
"B": "She boarded the train to Willowbrook",
"C": "She visited the village library",
"D": "She stopped at a café for breakfast"
Answer 2: A (Correct)

Question 3: "Which event happened right after Clara arrived at Willowbrook?"
"A": "She stopped at a café for breakfast",
"B": "She explored the nearby forest",
"C": "She visited the village library",
"D": "She boarded the early train"
Answer 3: B (Wrong)
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


def load_data(data_json_path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(data_json_path).read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"{data_json_path} must contain a list under key 'items'.")
    return items


def build_prompt(condition_id: str, item: Dict[str, Any]) -> str:
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    title = item.get("title", "Story")
    story_text = item.get("story_text", "")
    story_block = f"Title: {title}\n\n{story_text}"
    q_lines: List[str] = []
    for i, q in enumerate(item.get("questions", []), start=1):
        opts_dict = q.get("options", q.get("choices", {}))
        opts = " ".join(f"{k}) {opts_dict.get(k, '')}" for k in ["A", "B", "C", "D"])
        q_lines.append(f"Question {i}: {q.get('question', '')} {opts}")
    stimulus = (
        "Story:\n"
        + story_block
        + "\n\n------\n"
        + "Questions:\n"
        + "\n".join(q_lines)
    )
    return wrap_stimulus_prompt(prefix, condition_id, stimulus, FORMAT_RULES)


LINE_RE = re.compile(r"^question\s+(\d+):\s*([A-Da-d])\s*$", re.IGNORECASE)


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


def score_story(
    questions: List[Dict[str, Any]],
    answer_map: Dict[int, str],
) -> Dict[str, Any]:
    correct = 0
    for i, q in enumerate(questions, start=1):
        ans = (q.get("answer") or "").upper()
        if answer_map.get(i) == ans:
            correct += 1
    total = len(questions)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def summarize_story_rows(
    rows: List[Dict[str, Any]],
    condition_id: str,
    condition_name: str,
) -> Dict[str, Any]:
    accuracies = [float(r["metrics"]["accuracy"]) for r in rows]
    corrects = [int(r["metrics"]["correct"]) for r in rows]
    return {
        "condition": condition_id,
        "condition_name": condition_name,
        "n": len(rows),
        "metrics": {
            "accuracy_mean": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "correct_mean": sum(corrects) / len(corrects) if corrects else 0.0,
            "total_per_story": rows[0]["metrics"]["total"] if rows else 0,
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
    plt.bar(cond_ids, means, color="mediumpurple", edgecolor="black")
    plt.xlabel("Condition")
    plt.ylabel("Mean accuracy (correct / 10)")
    plt.title("Narrative QA: mean accuracy by condition")
    save_fig(fig, out_dir / "figures" / task_name / "accuracy_by_condition.png")


def _rng_for_item_draw(stimuli_seed: int, participant_index: int, cond_id: str) -> random.Random:
    payload = f"narrative_qa:{stimuli_seed}:{participant_index}:{cond_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return random.Random(seed_int)


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    data_json_path: str,
    n_participants: int = 1,
    n_stories: Optional[int] = None,
    stimuli_seed: int = 42,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """Each participant independently draws one story per condition from the pool.

    ``n_stories`` optionally limits pool size (random sample); ``None`` uses all stories.
    """
    all_items = load_data(data_json_path)
    if not all_items:
        raise ValueError(f"No stories in {data_json_path}.")

    rng = random.Random(stimuli_seed)
    pool = list(all_items)
    if n_stories is not None:
        nk = min(int(n_stories), len(pool))
        pool = rng.sample(pool, nk)

    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)

    all_rows: List[Dict[str, Any]] = []
    cond_summaries: List[Dict[str, Any]] = []
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    for cond_id in ["C1", "C2", "C3", "C4"]:
        rows: List[Dict[str, Any]] = []
        accuracies: List[float] = []
        corrects: List[int] = []

        def _one_participant(participant_index: int) -> Dict[str, Any]:
            pick_rng = _rng_for_item_draw(stimuli_seed, participant_index, cond_id)
            item = pick_rng.choice(pool)
            story_id = item.get("story_id", "")
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
            scored = score_story(questions, answer_map)
            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{participant_index}:{story_id}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "participant_id": participant_index - 1,
                "repeat_index": participant_index,
                "story_id": story_id,
                "title": item.get("title", ""),
                "prompt": prompt,
                "questions": questions,
                "raw": resp.text,
                "metrics": scored,
            }
            sink.append(row)
            return row

        row_dicts = map_participants(
            list(range(1, n_participants + 1)),
            _one_participant,
            max_workers=workers,
        )
        for row in row_dicts:
            rows.append(row)
            accuracies.append(row["metrics"]["accuracy"])
            corrects.append(row["metrics"]["correct"])

        cond_summaries.append(summarize_story_rows(rows, cond_id, CONDITIONS[cond_id]["name"]))
        all_rows.extend(rows)

    # Figure: mean accuracy by condition
    save_accuracy_by_condition_figure(out_dir, TASK_NAME, cond_summaries)

    summary = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary
