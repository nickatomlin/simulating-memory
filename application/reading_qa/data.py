from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


LEVELS = ["biography", "distractor", "reading_level", "redundant"]


@dataclass(frozen=True)
class Question:
    q_id: str
    question: str
    options: Dict[str, str]
    answer: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    levels: Dict[str, str]
    questions: List[Question]


def _read_level_text(raw: Dict[str, Any], level: str) -> str:
    if level == "biography":
        return str(raw.get("biography_text", "")).strip()
    if level == "distractor":
        return str(((raw.get("distractor") or {}).get("updated_biography", ""))).strip()
    if level == "reading_level":
        return str(((raw.get("reading_level") or {}).get("updated_biography", ""))).strip()
    if level == "redundant":
        # Source key in JSON is "redundancy"; internally we expose this as "redundant".
        return str(((raw.get("redundancy") or {}).get("updated_biography", ""))).strip()
    raise ValueError(f"Unknown level: {level}")


def _parse_questions(raw_questions: List[Dict[str, Any]], path: Path) -> List[Question]:
    parsed: List[Question] = []
    for idx, q in enumerate(raw_questions, start=1):
        question_text = str(q.get("question", "")).strip()
        options_raw = q.get("options", {})
        answer = str(q.get("answer", "")).strip().upper()
        if not question_text:
            raise ValueError(f"{path}: question {idx} is missing text")
        if not isinstance(options_raw, dict):
            raise ValueError(f"{path}: question {idx} options must be an object")
        options: Dict[str, str] = {}
        for key in ["A", "B", "C", "D"]:
            val = str(options_raw.get(key, "")).strip()
            if not val:
                raise ValueError(f"{path}: question {idx} is missing option {key}")
            options[key] = val
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"{path}: question {idx} has invalid answer {answer!r}")
        parsed.append(
            Question(
                q_id=str(q.get("q_id", f"Q{idx:02d}")),
                question=question_text,
                options=options,
                answer=answer,
            )
        )
    if not parsed:
        raise ValueError(f"{path}: no questions found")
    return parsed


def load_documents(documents_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(documents_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        title = str(raw.get("title", "")).strip() or path.stem
        doc_id = str(raw.get("biography_id", "")).strip() or path.stem
        levels: Dict[str, str] = {}
        for level in LEVELS:
            text = _read_level_text(raw, level)
            if not text:
                raise ValueError(f"{path}: level {level!r} is empty")
            levels[level] = text
        questions = _parse_questions(raw.get("questions", []), path)
        docs.append(Document(doc_id=doc_id, title=title, levels=levels, questions=questions))
    if not docs:
        raise ValueError(f"No json documents found in {documents_dir}")
    return docs
