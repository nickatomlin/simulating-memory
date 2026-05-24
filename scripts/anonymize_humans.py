"""Anonymize the human run files under runs/human/.

What this strips
----------------
- ``ip_address``               (PII)
- ``user_agent``               (browser/OS fingerprint, soft PII)
- ``completion_code``          (Prolific completion code; sensitive)

What this rewrites
------------------
- ``participant_id``   ->  ``participant-NNNN``  (4-digit, stable across tasks)
- ``session_id``       ->  ``session-NNNN-SS``   (per-participant counter)
- ``run_id``           ->  ``run-NNNN-RR``       (per-participant counter)

Mappings are deterministic given the *sorted order of original UUIDs* so the
result is reproducible (rerunning this on the same input gives the same
output), and joins across task folders continue to work.

The script is idempotent: if it detects already-anonymized IDs it leaves
the file alone.

Usage
-----
    python scripts/anonymize_humans.py            # rewrite in place
    python scripts/anonymize_humans.py --dry-run  # show what would change
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMAN_ROOT = ROOT / "runs" / "human"

PII_TOP_LEVEL = ("ip_address", "user_agent", "completion_code")

ANON_PARTICIPANT_RE = re.compile(r"^participant-\d{4}$")
ANON_SESSION_RE = re.compile(r"^session-\d{4}-\d{2}$")
ANON_RUN_RE = re.compile(r"^run-\d{4}-\d{2}$")


def load_all() -> list[tuple[Path, dict]]:
    out = []
    for p in sorted(HUMAN_ROOT.rglob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                out.append((p, json.load(f)))
        except json.JSONDecodeError:
            print(f"  skip (invalid JSON): {p}")
    return out


def build_id_maps(records: list[tuple[Path, dict]]) -> tuple[dict, dict, dict]:
    """Return (participant_map, session_map, run_map) keyed by original IDs."""
    # Collect raw participant -> set of sessions / runs.
    sessions_by_pid: dict[str, set[str]] = {}
    runs_by_pid: dict[str, set[str]] = {}
    for _, rec in records:
        pid = rec.get("participant_id")
        if not isinstance(pid, str):
            continue
        if ANON_PARTICIPANT_RE.match(pid):
            continue  # already-anon record: don't claim a new pseudonym
        sessions_by_pid.setdefault(pid, set())
        runs_by_pid.setdefault(pid, set())
        sid = rec.get("session_id")
        rid = rec.get("run_id")
        if isinstance(sid, str) and not ANON_SESSION_RE.match(sid):
            sessions_by_pid[pid].add(sid)
        if isinstance(rid, str) and not ANON_RUN_RE.match(rid):
            runs_by_pid[pid].add(rid)

    # Stable ordering: sort participants by their original UUID string.
    pmap: dict[str, str] = {}
    for i, pid in enumerate(sorted(sessions_by_pid.keys()), start=1):
        pmap[pid] = f"participant-{i:04d}"

    smap: dict[str, str] = {}
    rmap: dict[str, str] = {}
    for pid, anon_pid in pmap.items():
        pnum = anon_pid.split("-")[-1]
        for j, sid in enumerate(sorted(sessions_by_pid[pid]), start=1):
            smap[sid] = f"session-{pnum}-{j:02d}"
        for j, rid in enumerate(sorted(runs_by_pid[pid]), start=1):
            rmap[rid] = f"run-{pnum}-{j:02d}"
    return pmap, smap, rmap


def anonymize_record(rec: dict, pmap: dict, smap: dict, rmap: dict) -> dict:
    new = {}
    for k, v in rec.items():
        if k in PII_TOP_LEVEL:
            continue
        if k == "participant_id" and isinstance(v, str) and v in pmap:
            new[k] = pmap[v]
        elif k == "session_id" and isinstance(v, str) and v in smap:
            new[k] = smap[v]
        elif k == "run_id" and isinstance(v, str) and v in rmap:
            new[k] = rmap[v]
        else:
            new[k] = v
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write changes; print a summary instead.")
    args = parser.parse_args()

    if not HUMAN_ROOT.is_dir():
        raise SystemExit(f"missing directory: {HUMAN_ROOT}")

    records = load_all()
    print(f"loaded {len(records)} JSON records")

    pmap, smap, rmap = build_id_maps(records)
    print(f"  participants: {len(pmap)}")
    print(f"  sessions:     {len(smap)}")
    print(f"  runs:         {len(rmap)}")

    changed = 0
    stripped_counts = {k: 0 for k in PII_TOP_LEVEL}
    for path, rec in records:
        for k in PII_TOP_LEVEL:
            if k in rec:
                stripped_counts[k] += 1
        new_rec = anonymize_record(rec, pmap, smap, rmap)
        if new_rec == rec:
            continue
        changed += 1
        if not args.dry_run:
            with path.open("w", encoding="utf-8") as f:
                json.dump(new_rec, f, ensure_ascii=False)

    print(f"files changed: {changed}/{len(records)}")
    print(f"fields stripped: {stripped_counts}")
    if args.dry_run:
        print("(dry-run; nothing written)")
    else:
        print("done.")


if __name__ == "__main__":
    main()
