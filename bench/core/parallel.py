from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")


def resolve_worker_count(n: int, *, max_parallel: Optional[int] = None) -> int:
    """Thread count for participant-level parallelism: at least 1, at most ``n``."""
    if n <= 1:
        return 1
    if max_parallel is not None:
        return max(1, min(n, int(max_parallel)))
    env = os.environ.get("BENCH_MAX_PARALLEL_PARTICIPANTS", "").strip()
    if env:
        try:
            return max(1, min(n, int(env)))
        except ValueError:
            pass
    return n


def map_participants(
    participant_ids: Sequence[int],
    fn: Callable[[int], T],
    *,
    max_workers: int,
) -> List[T]:
    """Run ``fn(pid)`` for each id; results are in the same order as ``participant_ids``."""
    ids = list(participant_ids)
    if not ids:
        return []
    workers = max(1, min(len(ids), int(max_workers)))
    if workers <= 1:
        return [fn(i) for i in ids]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, ids))
