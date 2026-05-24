from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.io import write_json, JsonlSink
from ..core.llm import LLM
from ..core.parallel import map_participants, resolve_worker_count
from ..core.wm_agent import SummarizerAgent, WorkingMemoryAgent
from ..core.working_memory import MAX_KEYS
from .semantic_story_recall import (
    FORMAT_RULES,
    HUMAN_PROMPT,
    TASK_CORE,
    _load_sentence_transformer,
    _mean_optional_float,
    bleu_score,
    embedding_similarity,
    generate_story_stimuli,
)
from .wm_prompt_parts import (
    CONDITIONS,
    SUMMARIZER_CONDITIONS,
    summarizer_recall_prompt,
    summarizer_system_prompt,
    wm_recall_prompt,
    wm_system_prompts,
)

# ---------------------------------------------------------------------------
# LLM-as-judge prompts and helpers (previously in semantic_story_recall.py)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert evaluator for a memory recall experiment. You will receive the original story and a participant's recall. Score the recall on three dimensions (each 1-5) and provide a short rationale."""

JUDGE_USER_TEMPLATE = """\
## Original story
{story}

## Participant's recall
{recall}

Score the recall on these dimensions (1 = worst, 5 = best):
1. **coverage** — How many key events and relationships from the story are recalled?
2. **faithfulness** — How accurate is the recall, without distortions?
3. **hallucination** — How much invented detail not in the story? (5 = many hallucinations, 1 = none)

Return ONLY a JSON object with keys: coverage, faithfulness, hallucination, rationale_bullets (an array of 2-6 short strings). Example:
{{"coverage": 3, "faithfulness": 4, "hallucination": 2, "rationale_bullets": ["Recalled main events", "Missed ending"]}}"""


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Try to extract JSON from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Try to find a JSON object
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"parse_error": text[:200]}


def _normalize_judge_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    if "parse_error" in d:
        return {
            "coverage": 1,
            "faithfulness": 1,
            "hallucination": 5,
            "rationale_bullets": ["Judge output could not be parsed."],
            "judge_parse_error": d["parse_error"],
        }
    out: Dict[str, Any] = {}
    for k in ("coverage", "faithfulness", "hallucination"):
        v = d.get(k)
        try:
            v = int(v)
            v = max(1, min(5, v))
        except (TypeError, ValueError):
            v = 1 if k != "hallucination" else 5
        out[k] = v
    out["rationale_bullets"] = d.get("rationale_bullets", [])
    return out


TASK_NAME = "wm_semantic_story_recall"
TASK_NAME_SUM = "sum_semantic_story_recall"

WM_SYSTEM_PROMPTS = wm_system_prompts(
    task_prompt=TASK_CORE,
    human_task_prompt=HUMAN_PROMPT,
)

STORY_RECALL_PROMPT = wm_recall_prompt(
    task_prompt=TASK_CORE,
    wm_recall_instructions=(
        "write everything you can remember about the story (events, details, and key "
        "information). Try to be as accurate and detailed as possible. Do NOT add explanations "
        "about your memory slots or extra meta-commentary."
    ),
    format_rules=FORMAT_RULES,
)

SUM_SYSTEM_PROMPT = summarizer_system_prompt(task_prompt=TASK_CORE)

SUM_STORY_RECALL_PROMPT = summarizer_recall_prompt(
    task_prompt=TASK_CORE,
    recall_instructions=(
        "write everything you can remember about the story (events, details, and key "
        "information). Try to be as accurate and detailed as possible. Do NOT add explanations "
        "about your summary or extra meta-commentary."
    ),
    format_rules=FORMAT_RULES,
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s+|$)")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on .!? followed by whitespace or end-of-string."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    repeats_per_stimulus: int = 1,
    temperature: float = 0.0,
    debug: bool = False,
    story: Optional[str] = None,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    stimuli = generate_story_stimuli(data_dir)
    embed_model = _load_sentence_transformer()
    if story is not None:
        key = story.lower().replace(" ", "_")
        stimuli = [s for s in stimuli if s.stimulus_id.startswith(key)]
        if not stimuli:
            valid = [s.stimulus_id for s in generate_story_stimuli(data_dir)]
            raise ValueError(f"Unknown story {story!r}. Valid stimulus IDs: {valid}")

    n_runs = len(stimuli) * int(repeats_per_stimulus)
    workers = resolve_worker_count(n_runs, max_parallel=max_parallel_participants)

    all_rows: List[Dict[str, Any]] = []
    jsonl_path = out_dir / "tasks" / f"{TASK_NAME}.jsonl"
    sink = JsonlSink(jsonl_path)

    def _one_run(args: tuple) -> Dict[str, Any]:
        cond_id, stim, rep = args
        sentences = split_sentences(stim.text)

        agent = WorkingMemoryAgent(
            llm=llm,
            condition_id=cond_id,
            temperature=temperature,
            debug=debug,
            system_prompt_override=WM_SYSTEM_PROMPTS[cond_id],
        )

        if debug:
            print(f"\n{'=' * 60}")
            print(f"{TASK_NAME} | {cond_id} | story={stim.story_name} | rep={rep}")
            print(f"  {len(sentences)} sentences")
            print(f"{'=' * 60}")

        # Study phase: encode full story at once
        encoding_log = agent.encode(stim.text)
        turn_logs = [encoding_log]

        # Recall phase: from slots only, using story-specific prompt
        recall_text = agent.recall(recall_prompt=STORY_RECALL_PROMPT)

        # Judge phase: LLM-as-judge
        judge_prompt = JUDGE_USER_TEMPLATE.format(
            story=stim.text,
            recall=recall_text,
        )
        normalized: Dict[str, Any] = {
            "coverage": 1,
            "faithfulness": 1,
            "hallucination": 5,
            "rationale_bullets": [
                "Judge call failed.",
                "Default pessimistic scores were applied for robustness.",
            ],
            "judge_parse_error": "judge_call_failed",
        }
        judge_raw = ""
        try:
            for _ in range(3):
                judge_resp = llm.generate(
                    judge_prompt,
                    system=JUDGE_SYSTEM,
                    temperature=0.0,
                    max_tokens=512,
                )
                judge_raw = judge_resp.text or ""
                parsed = _safe_json_loads(judge_raw)
                normalized = _normalize_judge_metrics(parsed)
                if "judge_parse_error" not in normalized:
                    break
        except Exception as e:
            normalized["judge_exception"] = str(e)[:300]

        final_kv = agent.wm.store
        slot_utilization = len(final_kv) / MAX_KEYS

        metrics = dict(normalized)
        metrics["embeddingSimilarity"] = embedding_similarity(
            embed_model, stim.text, recall_text
        )
        metrics["bleuScore"] = bleu_score(stim.text, recall_text)
        metrics["slot_utilization"] = slot_utilization

        row = {
            "id": f"{TASK_NAME}:{cond_id}:{stim.stimulus_id}:{rep}",
            "stimulus_id": stim.stimulus_id,
            "story_name": stim.story_name,
            "story_source_file": stim.story_source_file,
            "condition": cond_id,
            "condition_name": CONDITIONS[cond_id]["name"],
            "repeat_index": rep,
            "n_sentences": len(sentences),
            "final_kv": final_kv,
            "recall_text": recall_text,
            "judge_raw": judge_raw,
            "turn_logs": turn_logs,
            "metrics": metrics,
        }
        sink.append(row)
        return row

    for cond_id in CONDITIONS:
        combos = [
            (cond_id, stim, rep)
            for stim in stimuli
            for rep in range(1, int(repeats_per_stimulus) + 1)
        ]
        cond_rows = map_participants(
            list(range(len(combos))),
            lambda i, combos=combos: _one_run(combos[i]),
            max_workers=workers,
        )
        all_rows.extend(cond_rows)

    # Summarize per condition
    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == cond_id]
        cov = [float(r["metrics"].get("coverage", 1.0)) for r in rows]
        fai = [float(r["metrics"].get("faithfulness", 1.0)) for r in rows]
        hal = [float(r["metrics"].get("hallucination", 5.0)) for r in rows]
        util = [float(r["metrics"].get("slot_utilization", 0.0)) for r in rows]
        sims = [r["metrics"].get("embeddingSimilarity") for r in rows]
        bleus = [r["metrics"].get("bleuScore") for r in rows]
        cond_summaries.append(
            {
                "condition": cond_id,
                "condition_name": CONDITIONS[cond_id]["name"],
                "n": len(rows),
                "metrics": {
                    "embeddingSimilarity": _mean_optional_float(sims),
                    "bleuScore": _mean_optional_float(bleus),
                    "coverage": _mean(cov),
                    "faithfulness": _mean(fai),
                    "hallucination": _mean(hal),
                    "slot_utilization": _mean(util),
                },
            }
        )

    summary: Dict[str, Any] = {
        "task": TASK_NAME,
        "repeats_per_stimulus": int(repeats_per_stimulus),
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)
    return summary


def evaluate_summarizer(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    repeats_per_stimulus: int = 1,
    temperature: float = 0.0,
    debug: bool = False,
    story: Optional[str] = None,
    max_parallel_participants: Optional[int] = None,
) -> Dict[str, Any]:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    stimuli = generate_story_stimuli(data_dir)
    embed_model = _load_sentence_transformer()
    if story is not None:
        key = story.lower().replace(" ", "_")
        stimuli = [s for s in stimuli if s.stimulus_id.startswith(key)]
        if not stimuli:
            valid = [s.stimulus_id for s in generate_story_stimuli(data_dir)]
            raise ValueError(f"Unknown story {story!r}. Valid stimulus IDs: {valid}")

    n_runs = len(stimuli) * int(repeats_per_stimulus)
    workers = resolve_worker_count(n_runs, max_parallel=max_parallel_participants)

    all_rows: List[Dict[str, Any]] = []
    jsonl_path = out_dir / "tasks" / f"{TASK_NAME_SUM}.jsonl"
    sink = JsonlSink(jsonl_path)

    def _one_run(args: tuple) -> Dict[str, Any]:
        cond_id, stim, rep = args
        sentences = split_sentences(stim.text)

        agent = SummarizerAgent(
            llm=llm,
            condition_id=cond_id,
            temperature=temperature,
            debug=debug,
            system_prompt_override=summarizer_system_prompt(
                TASK_CORE, condition_id=cond_id
            ),
        )

        if debug:
            print(f"\n{'=' * 60}")
            print(f"{TASK_NAME_SUM} | {cond_id} | story={stim.story_name} | rep={rep}")
            print(f"  {len(sentences)} sentences")
            print(f"{'=' * 60}")

        encoding_log = agent.encode(stim.text)
        turn_logs = [encoding_log]

        recall_text = agent.recall(recall_prompt=SUM_STORY_RECALL_PROMPT)

        judge_prompt = JUDGE_USER_TEMPLATE.format(
            story=stim.text,
            recall=recall_text,
        )
        normalized: Dict[str, Any] = {
            "coverage": 1,
            "faithfulness": 1,
            "hallucination": 5,
            "rationale_bullets": [
                "Judge call failed.",
                "Default pessimistic scores were applied for robustness.",
            ],
            "judge_parse_error": "judge_call_failed",
        }
        judge_raw = ""
        try:
            for _ in range(3):
                judge_resp = llm.generate(
                    judge_prompt,
                    system=JUDGE_SYSTEM,
                    temperature=0.0,
                    max_tokens=512,
                )
                judge_raw = judge_resp.text or ""
                parsed = _safe_json_loads(judge_raw)
                normalized = _normalize_judge_metrics(parsed)
                if "judge_parse_error" not in normalized:
                    break
        except Exception as e:
            normalized["judge_exception"] = str(e)[:300]

        metrics = dict(normalized)
        metrics["embeddingSimilarity"] = embedding_similarity(
            embed_model, stim.text, recall_text
        )
        metrics["bleuScore"] = bleu_score(stim.text, recall_text)
        metrics["summary_length_words"] = agent.summary_length_words()

        row = {
            "id": f"{TASK_NAME_SUM}:{cond_id}:{stim.stimulus_id}:{rep}",
            "stimulus_id": stim.stimulus_id,
            "story_name": stim.story_name,
            "story_source_file": stim.story_source_file,
            "condition": cond_id,
            "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
            "repeat_index": rep,
            "n_sentences": len(sentences),
            "final_summary": agent.summary,
            "recall_text": recall_text,
            "judge_raw": judge_raw,
            "turn_logs": turn_logs,
            "metrics": metrics,
        }
        sink.append(row)
        return row

    for cond_id in SUMMARIZER_CONDITIONS:
        combos = [
            (cond_id, stim, rep)
            for stim in stimuli
            for rep in range(1, int(repeats_per_stimulus) + 1)
        ]
        cond_rows = map_participants(
            list(range(len(combos))),
            lambda i, combos=combos: _one_run(combos[i]),
            max_workers=workers,
        )
        all_rows.extend(cond_rows)

    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in SUMMARIZER_CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == cond_id]
        cov = [float(r["metrics"].get("coverage", 1.0)) for r in rows]
        fai = [float(r["metrics"].get("faithfulness", 1.0)) for r in rows]
        hal = [float(r["metrics"].get("hallucination", 5.0)) for r in rows]
        slw = [float(r["metrics"].get("summary_length_words", 0.0)) for r in rows]
        sims = [r["metrics"].get("embeddingSimilarity") for r in rows]
        bleus = [r["metrics"].get("bleuScore") for r in rows]
        cond_summaries.append(
            {
                "condition": cond_id,
                "condition_name": SUMMARIZER_CONDITIONS[cond_id]["name"],
                "n": len(rows),
                "metrics": {
                    "embeddingSimilarity": _mean_optional_float(sims),
                    "bleuScore": _mean_optional_float(bleus),
                    "coverage": _mean(cov),
                    "faithfulness": _mean(fai),
                    "hallucination": _mean(hal),
                    "summary_length_words": _mean(slw),
                },
            }
        )

    summary: Dict[str, Any] = {
        "task": TASK_NAME_SUM,
        "repeats_per_stimulus": int(repeats_per_stimulus),
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME_SUM}_summary.json", summary)
    return summary
