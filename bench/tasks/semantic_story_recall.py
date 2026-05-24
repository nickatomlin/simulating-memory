from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.llm import LLM
from ..core.io import write_json, JsonlSink
from ..core.parallel import map_participants, resolve_worker_count

from .human_simulation_prefixes import HUMAN_SIM_INTRO_C2, HUMAN_SIM_INTRO_C3_C4_BEFORE_HUMAN
from .stimulus_prompt_shell import wrap_stimulus_prompt

TASK_NAME = "semantic_story_recall"

# Matches semantic-memory-story-recall.html: tokenize()
_TOKEN_SPLIT_RE = re.compile(r'[.,!?;:"\'\[\]{}()\-\n\r]+')

# First 200 tokens for embedding (same as HTML truncateWords maxWords=200)
EMBED_MAX_WORDS = 200


# =========================
# Prompts (copied verbatim from semantic_memory_story_recall.ipynb)
# =========================

TASK_CORE = """Read a story, then recall the story using the same words that appear in the story whenever possible.  For example, if the story is in first person you should also use first person. 
"""

HUMAN_PROMPT = """The human will have five minutes to read a story. The story will then be hidden.  Then, the human will be asked to type as much as they remember in the text box. They will be asked to recall the story using the same words that appear in the story whenever possible. For example, if the story is in first person they should also use first person. 
"""

FORMAT_RULES = """Write your recall freely in plain text. There is no required line format, numbering, or structure—just reproduce what you remember from the story above."""

ICL_EXAMPLES = """Here are example results from previous human participants.

Story A: On the final Friday, a handwritten sign appeared in the window:
PAY WHAT YOU CAN. TAKE WHAT YOU NEED.
Rosa expected a quiet day.
Instead, people came.
A student left three crumpled dollars and a thank-you note. A tired nurse paid extra and said nothing. A child traded a drawing of the bakery—too many windows, a smiling sun—for a chocolate croissant. Rosa taped the drawing to the wall behind the counter.
Recall A: There is a sign saying pay what you can and take what you need. A student paid a few dollars with a thank you note.

Story B: On a gray Tuesday, rain threading the air like static, Lin noticed something different. When the doors opened, a single piece of paper slid out and landed at her feet.
Recall B: Lin noticed that there is a paper slid out.
"""


CONDITIONS: Dict[str, Dict[str, str]] = {
    "C1": {
        "name": "Just describe the task",
        "prompt_prefix": TASK_CORE,
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
            + "\n"
            + ICL_EXAMPLES
        ),
    },
}


# def make_study_prompt(condition_id: str, story_text: str) -> str:
#     return (
#         CONDITIONS[condition_id]["prompt_prefix"]
#         + "\n---\n"
#         + "READ PHASE (you have 5 minutes in the original task):\n"
#         + "Read the story below carefully. "
#         + "STORY:\n"
#         + story_text
#     )


# def make_recall_prompt(condition_id: str) -> str:
#     return (
#         CONDITIONS[condition_id]["prompt_prefix"]
#         + "\n---\n"
#         + "RECALL PHASE:\n"
#         + "The story is now hidden. Type as much as you remember.\n"
#         + "Begin your recall now:"
#     )

def build_prompt(condition_id: str, story_text: str) -> str:
    """Single request: read story and recall in one context; stimuli framing matches craft_task."""
    prefix = CONDITIONS[condition_id]["prompt_prefix"]
    stimulus = (
        "Here is the story:\n"
        + story_text
        + "\n\n\n"
        + "Begin your recall now:"
    )
    return wrap_stimulus_prompt(prefix, condition_id, stimulus, FORMAT_RULES)


make_study_recall_prompt = build_prompt


# =========================
# Stimuli generation (full story) using packaged transcripts
# =========================


def _load_transcripts(data_dir: Path) -> List[Tuple[str, str, str]]:
    items = [
        ("Pieman", "pieman_transcript.txt"),
        ("Oregon Trail", "oregontrail_transcript.txt"),
        ("Baseball", "baseball_transcript.txt"),
        ("Eyespy", "eyespy_transcript.txt"),
    ]
    out: List[Tuple[str, str, str]] = []
    for name, fname in items:
        p = data_dir / fname
        out.append((name, fname, p.read_text(encoding="utf-8")))
    return out


@dataclass
class StoryStimulus:
    stimulus_id: str
    story_name: str
    story_source_file: str
    text: str
    difficulty: Dict[str, Any]


def generate_story_stimuli(data_dir: Path) -> List[StoryStimulus]:
    transcripts = _load_transcripts(data_dir)
    stimuli: List[StoryStimulus] = []
    for name, src, text in transcripts:
        sid = f"{name.lower().replace(' ', '_')}_full"
        stimuli.append(
            StoryStimulus(
                stimulus_id=sid,
                story_name=name,
                story_source_file=src,
                text=text,
                difficulty={"mode": "full_story"},
            )
        )
    return stimuli


def tokenize_story_recall(text: str) -> List[str]:
    """Align with semantic-memory-story-recall.html tokenize()."""
    if not text:
        return []
    t = text.lower()
    t = _TOKEN_SPLIT_RE.sub(" ", t)
    return [w for w in t.split() if w]


def truncate_to_max_words(text: str, max_words: int = EMBED_MAX_WORDS) -> str:
    """Align with HTML truncateWords: first max_words tokens joined by space."""
    t = tokenize_story_recall(text)
    return " ".join(t[:max_words])


def _load_sentence_transformer() -> Optional[Any]:
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        print("SentenceTransformer not found")
        return None


def bleu_score(
    story: str,
    recall: str,
) -> Optional[float]:
    """
    Sentence-level BLEU (sacrebleu) on space-joined tokens from tokenize_story_recall,
    matching the story-recall tokenizer. Uses effective_order=True so short recalls are
    not forced to 0. Returns a value in [0, 1], or None if sacrebleu is unavailable.
    """
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        return None

    ref = " ".join(tokenize_story_recall(story))
    hyp = " ".join(tokenize_story_recall(recall))
    if not ref or not hyp:
        return 0.0 if not hyp else None

    bleu = BLEU(effective_order=True, tokenize="none")
    try:
        out = bleu.sentence_score(hyp, [ref])
        s = float(out.score)
        if s > 1.0:
            s = s / 100.0
        return max(0.0, min(1.0, s))
    except Exception:
        return None


def embedding_similarity(
    model: Optional[Any],
    story: str,
    recall: str,
) -> Optional[float]:
    """
    Cosine similarity of mean-pooled normalized embeddings (all-MiniLM-L6-v2),
    after truncating story and recall to 200 words — same as semantic-memory-story-recall.html.
    Returns None if model unavailable, inputs empty after truncate, or encode fails.
    """
    s = truncate_to_max_words(story, EMBED_MAX_WORDS)
    r = truncate_to_max_words(recall, EMBED_MAX_WORDS)
    if not s or not r:
        return None
    if model is None:
        return None
    try:
        emb = model.encode(
            [s, r],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        a = np.asarray(emb[0], dtype=np.float64)
        b = np.asarray(emb[1], dtype=np.float64)
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return None
        dot = float(np.dot(a, b))
        return max(0.0, min(1.0, dot))
    except Exception:
        return None


def _rng_for_story_draw(stimuli_seed: int, cond_id: str, participant_index: int) -> random.Random:
    """Stable RNG for which story a participant sees (reproducible across processes)."""
    payload = f"{stimuli_seed}:{cond_id}:{participant_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return random.Random(seed_int)


def _mean_optional_float(vals: List[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in vals if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def evaluate(
    llm: LLM,
    out_dir: Path,
    *,
    model_cfg: Dict[str, Any],
    n_participants: int = 1,
    stimuli_seed: int = 42,
    max_parallel_participants: Optional[int] = None,
    recall_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Run semantic story recall for conditions C1..C4.

    For each condition, ``n_participants`` independent simulated participants each read and
    recall **one** story drawn uniformly at random from the transcript pool (reproducible via
    ``stimuli_seed``).

    Metrics:
    - embeddingSimilarity: cosine similarity of all-MiniLM-L6-v2 embeddings (200-word truncate).
    - bleuScore: sentence-level BLEU (sacrebleu, effective_order) on tokenizer-aligned strings.
    """

    data_dir = Path(__file__).resolve().parents[2] / "data"
    stimuli = generate_story_stimuli(data_dir)
    embed_model = _load_sentence_transformer()

    # Long recalls need a high ceiling; cap to what your API/model allows (e.g. 16384 for many OpenAI models).
    _max_tokens = int(
        recall_max_tokens
        if recall_max_tokens is not None
        else model_cfg.get("max_tokens", 16384)
    )

    all_rows: List[Dict[str, Any]] = []
    n_participants = max(1, int(n_participants))
    workers = resolve_worker_count(n_participants, max_parallel=max_parallel_participants)
    sink = JsonlSink(out_dir / "tasks" / f"{TASK_NAME}.jsonl")

    for cond_id, cond in CONDITIONS.items():

        def _one_participant(participant_index: int) -> Dict[str, Any]:
            rng = _rng_for_story_draw(stimuli_seed, cond_id, participant_index)
            stim = rng.choice(stimuli)
            recall_prompt = build_prompt(cond_id, stim.text)
            recall_resp = llm.generate(
                recall_prompt,
                temperature=float(model_cfg.get("temperature", 0)),
                max_tokens=_max_tokens,
                top_p=float(model_cfg.get("top_p", 1.0)),
                seed=model_cfg.get("seed"),
            )
            recall_text = (recall_resp.text or "").strip()

            emb_sim = embedding_similarity(embed_model, stim.text, recall_text)
            bleu = bleu_score(stim.text, recall_text)
            metrics = {
                "embeddingSimilarity": emb_sim,
                "bleuScore": bleu,
            }

            row = {
                "stimulus_id": stim.stimulus_id,
                "story_name": stim.story_name,
                "story_source_file": stim.story_source_file,
                "difficulty": stim.difficulty,
                "condition": cond_id,
                "condition_name": cond["name"],
                "repeat_index": participant_index,
                "stimuli_seed": int(stimuli_seed),
                "recall_prompt": recall_prompt,
                "llm_recall": recall_text,
                "metrics": metrics,
            }
            sink.append(row)
            return row

        all_rows.extend(
            map_participants(
                list(range(1, n_participants + 1)),
                _one_participant,
                max_workers=workers,
            )
        )

    cond_summaries: List[Dict[str, Any]] = []
    for cond_id in CONDITIONS.keys():
        rows = [r for r in all_rows if r["condition"] == cond_id]
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
                },
            }
        )

    summary: Dict[str, Any] = {
        "task": TASK_NAME,
        "n_participants": n_participants,
        "stimuli_seed": int(stimuli_seed),
        "conditions": cond_summaries,
    }

    write_json(out_dir / "tasks" / f"{TASK_NAME}_summary.json", summary)

    return summary
