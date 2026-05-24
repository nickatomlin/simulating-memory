from __future__ import annotations
from typing import Dict

MAX_KEYS = 4
NUM_SLOTS = MAX_KEYS  # backward-compat alias


class WorkingMemory:
    """Key-value working memory with a maximum of MAX_KEYS entries,
    mirroring the ~4-chunk limit of human short-term memory (Cowan, 2001)."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def write_key(self, key: str, value: str) -> str:
        if key not in self._store and len(self._store) >= MAX_KEYS:
            return (
                f"Error: memory is full ({MAX_KEYS} keys). "
                "Delete an existing key first or overwrite one."
            )
        self._store[key] = str(value)
        return f"Key '{key}' written."

    def clear_key(self, key: str) -> str:
        if key not in self._store:
            return f"Error: key '{key}' not found."
        del self._store[key]
        return f"Key '{key}' deleted."

    @property
    def store(self) -> Dict[str, str]:
        return dict(self._store)

    def snapshot(self) -> str:
        if not self._store:
            return "  (empty)"
        lines = []
        for k, v in self._store.items():
            lines.append(f'  "{k}": {repr(v)}')
        return "\n".join(lines)

    def to_recall_text(self) -> str:
        if not self._store:
            return "(memory is empty)"
        return "\n".join(f"{k}: {v}" for k, v in self._store.items())
