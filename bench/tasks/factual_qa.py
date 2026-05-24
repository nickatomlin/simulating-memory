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
from .stimulus_prompt_shell import DIGIT_SPAN_SEP_BEFORE_RULES, wrap_stimulus_prompt

TASK_NAME = "factual_qa"

####
# PROMPTS (verbatim from factual_qa.ipynb)
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
Readings: Facebook, Inc. v. Duguid, 592 U.S. ___ (2021), was a United States Supreme Court case concerning how to interpret the term \"automatic telephone dialing system\" under the Telephone Consumer Protection Act of 1991 (TCPA). The dispute arose over whether Facebook’s security-alert text messaging system qualified as an autodialer under the statute and thus could expose the company to liability for sending unsolicited text messages.\n\nThe TCPA was enacted in 1991 to reduce the growing number of unsolicited calls consumers were receiving. Among its provisions, the law restricted the use of automated telephone dialing systems, often called autodialers, to contact consumers on services that might cost the recipient money, including cell phones and text messaging. Violations of these restrictions could be investigated and fined by the Federal Communications Commission (FCC).\n\nOver time, Congress amended the TCPA. One important amendment, adopted in 2015, created an exception that allowed automated calls to numbers associated with services that charge the called party, when those calls were made to collect debts owed to or guaranteed by the federal government.\n\nSince the law’s enactment, the TCPA’s autodialer restrictions have been challenged on First Amendment grounds, with plaintiffs arguing that the statute impermissibly restricted speech. Federal courts of appeals generally upheld the statute, finding it consistent with free speech protections.\n\nThe 2015 amendment, however, introduced a new constitutional argument. By exempting automated calls for federal debt collection, the law appeared to favor one type of speech over others, potentially making the autodialer restriction a content-based regulation of speech and therefore more vulnerable under the First Amendment.\n\nAt the same time that Facebook, Inc. v. Duguid was proceeding, another case, Barr v. American Association of Political Consultants, Inc., was advancing toward the Supreme Court to address whether this 2015 amendment rendered the autodialer provisions unconstitutional.\n\nThe factual dispute in Facebook, Inc. v. Duguid began in 2014, when Noah Duguid started receiving text messages on his cell phone from Facebook. These messages warned him of suspicious activity on a Facebook account, but Duguid did not have a Facebook account at all.\n\nAfter attempting unsuccessfully to get Facebook to stop the messages, Duguid filed a class-action lawsuit in March 2015 in the United States District Court for the Northern District of California. He alleged that Facebook had violated the TCPA’s autodialer provision by using an automatic telephone dialing system (ATDS) to send him unwanted text messages.\n\nFacebook moved to dismiss the lawsuit on two main grounds. First, it argued that its login or security notification system was not an ATDS because it did not generate phone numbers randomly or sequentially; instead, it sent targeted messages to specific numbers associated with particular user accounts or entries in a database. Second, Facebook challenged the constitutionality of the TCPA’s autodialer provision as modified by the 2015 amendment, contending that the law had become a content-based restriction on speech that violated the First Amendment.\n\nBecause the case now raised a constitutional question, the federal government intervened to defend the TCPA and support dismissal.\n\nThe district court ruled in Facebook’s favor and dismissed Duguid’s complaint. The court concluded that Duguid had not adequately alleged that Facebook’s system qualified as an ATDS under the statutory definition, which referred to equipment that could store or produce telephone numbers using a random or sequential number generator.\n\nDuguid appealed to the United States Court of Appeals for the Ninth Circuit. On appeal, Facebook again argued that its notification system did not meet the statutory definition of an ATDS, and the federal government again appeared to defend the TCPA.\n\nThe Ninth Circuit, however, relied on its prior decision in Marks v.,
   
Question 1: What was the central legal issue in Facebook, Inc. v. Duguid?
"A": "Whether Facebook could charge users for security notifications",
"B": "The definition and function of auto dialers under the TCPA",
"C": "The legality of social media advertising practices",
"D": "Whether text messages are protected speech under the First Amendment"
Response 1: B (Correct)


Question 2: What precedent from Marks v. Crunch San Diego, LLC did the Ninth Circuit rely on regarding the definition of an ATDS?
"A": "That an ATDS only includes devices dialing random numbers during business hours",
"B": "That an ATDS is limited strictly to devices using rotary-dial technology",
"C": "That an ATDS includes devices that can dial stored numbers and those that send automated, unsolicited, and unwanted messages",
"D": "That an ATDS must be approved by the FCC before use"
Response 2: D (Wrong)
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
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"{data_json_path} must contain a list under key 'results'.")
    return results


def build_prompt(condition_id: str, doc: Dict[str, Any]) -> str:
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    title = doc.get("title", "Reading")
    excerpt = doc.get("excerpt_first_600_words_plus_sentence", "")
    reading_block = f"Title: {title}\n\n{excerpt}"
    q_lines: List[str] = []
    for i, q in enumerate(doc.get("questions", []), start=1):
        choices = q.get("choices", {})
        opts = " ".join(f"{k}) {choices.get(k, '')}" for k in ["A", "B", "C", "D"])
        q_lines.append(f"Question {i}: {q.get('question', '')} {opts}")
    stimulus = (
        "Reading:\n"
        + reading_block
        + "\n\n------\n"
        + "Questions:\n"
        + "\n".join(q_lines)
    )
    return wrap_stimulus_prompt(
        prefix,
        condition_id,
        stimulus,
        FORMAT_RULES,
        sep_before_rules=DIGIT_SPAN_SEP_BEFORE_RULES,
    )


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


def score_doc(
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


def summarize_doc_rows(
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
            "total_per_doc": rows[0]["metrics"]["total"] if rows else 0,
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
    plt.bar(cond_ids, means, color="coral", edgecolor="black")
    plt.xlabel("Condition")
    plt.ylabel("Mean accuracy (correct / 10)")
    plt.title("Factual QA: mean accuracy by condition")
    save_fig(fig, out_dir / "figures" / task_name / "accuracy_by_condition.png")


def _rng_for_item_draw(stimuli_seed: int, participant_index: int, cond_id: str) -> random.Random:
    payload = f"factual_qa:{stimuli_seed}:{participant_index}:{cond_id}".encode("utf-8")
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
    n_docs: Optional[int] = None,
    stimuli_seed: int = 42,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    """Each participant independently draws one document per condition from the pool.

    ``n_docs`` optionally limits pool size (random sample of that many docs); ``None`` uses all.
    """
    all_docs = load_data(data_json_path)
    if not all_docs:
        raise ValueError(f"No documents in {data_json_path}.")

    rng = random.Random(stimuli_seed)
    pool = list(all_docs)
    if n_docs is not None:
        nk = min(int(n_docs), len(pool))
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
            doc = pick_rng.choice(pool)
            doc_id = doc.get("doc_id", 0)
            questions = doc.get("questions", [])
            prompt = build_prompt(cond_id, doc)
            resp = llm.generate(
                prompt,
                temperature=float(model_cfg.get("temperature", 0.0)),
                max_tokens=int(model_cfg.get("max_tokens", 512)),
                top_p=float(model_cfg.get("top_p", 1.0)),
                seed=model_cfg.get("seed"),
            )
            answer_map = parse_answers(resp.text)
            scored = score_doc(questions, answer_map)
            row = {
                "id": f"{TASK_NAME}:{cond_id}:p{participant_index}:{doc_id}",
                "condition_id": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "participant_id": participant_index - 1,
                "repeat_index": participant_index,
                "doc_id": doc_id,
                "title": doc.get("title", ""),
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

        cond_summaries.append(summarize_doc_rows(rows, cond_id, CONDITIONS[cond_id]["name"]))
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
