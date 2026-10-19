"""Import the July human export and two Gemini Compactor runs for public release.

Reads a local paper checkout; never calls an API. Existing public participant
pseudonyms are reconstructed from the original export deterministically.
New IDs are assigned deterministically. The private ID maps are never written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.core.scoring import HUMAN_TASK_DIR, TASKS
from scripts.anonymize_humans import build_id_maps
from scripts.release_privacy import clean_record, completion_key

PRIVATE_KEYS = {"ip_address", "user_agent", "completion_code", "api_key", "authorization"}
MODELS = ["google_gemini-3.1-pro-preview", "google_gemini-3.6-flash"]


def sanitize(value, maps):
    if isinstance(value, dict):
        return {key: sanitize(maps.get(key, {}).get(item, item) if isinstance(item, str) else item, maps)
                for key, item in value.items() if key.lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [sanitize(item, maps) for item in value]
    return value


def identity_maps(paper_root, public_root, records):
    # Reconstruct the original public pseudonyms without needing a private
    # filename-to-identity lookup in the release.
    original = [(p, json.loads(p.read_text())) for p in sorted((paper_root / "runs-human/runs").rglob("*.json"))]
    maps = dict(zip(("participant_id", "session_id", "run_id"), build_id_maps(original)))
    pmap = maps["participant_id"]
    next_pid = max((int(v.split("-")[1]) for v in pmap.values()), default=0) + 1
    for pid in sorted({r["participant_id"] for _, r in records}):
        if pid not in pmap:
            pmap[pid] = f"participant-{next_pid:04d}"
            next_pid += 1
    for field, prefix in [("session_id", "session"), ("run_id", "run")]:
        mapping = maps[field]
        for pid in sorted({r["participant_id"] for _, r in records}):
            pnum = pmap[pid].split("-")[1]
            start = f"{prefix}-{pnum}-"
            index = max((int(v.rsplit("-", 1)[1]) for v in mapping.values() if v.startswith(start)), default=0) + 1
            ids = {r[field] for _, r in records if r["participant_id"] == pid and r.get(field)}
            for original in sorted(ids):
                if original not in mapping:
                    mapping[original] = f"{start}{index:02d}"
                    index += 1
    return maps


def import_data(paper_root: Path, public_root: Path = ROOT, *, dry_run=False):
    source = paper_root / "runs-human/2026-07-27/runs"
    records = []
    for folder in HUMAN_TASK_DIR.values():
        paths = sorted((source / folder).glob("run-*.json"))
        if not paths:
            raise ValueError(f"No source records in {source / folder}")
        records.extend((path, json.loads(path.read_text())) for path in paths)
    maps = identity_maps(paper_root, public_root, records)
    identities = {key: value for mapping in maps.values() for key, value in mapping.items()}
    order = {str(p): index for index, (p, _) in enumerate(
        sorted(records, key=lambda pr: completion_key(pr[1], pr[0].name)), 1)}
    outputs = []
    counts = Counter()
    participants = {task: set() for task in HUMAN_TASK_DIR.values()}
    for path, record in records:
        clean = clean_record(record, identities)
        clean["completion_order"] = order[str(path)]
        content = json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n"
        # Check values too: a raw participant ID must not survive in a nested field.
        if any(original in content for original in maps["participant_id"]):
            raise ValueError(f"Unredacted participant ID in {path.name}")
        target = public_root / "runs/human-2026-07-27" / path.parent.name / f"{clean['run_id']}.json"
        outputs.append((path, target, content.encode("utf-8")))
        counts[path.parent.name] += 1
        if record.get("status") == "completed":
            participants[path.parent.name].add(clean["participant_id"])
    for task, ids in participants.items():
        if len(ids) != 100:
            raise ValueError(f"Expected 100 completed participants for {task}, found {len(ids)}")
    gemini_root = paper_root / "runs/wm-slot-ablation/2026-07-27/4-slots"
    for model in MODELS:
        names = ["config_snapshot.json", "summary.json"]
        names += [f"tasks/wm_{task}{suffix}" for task in TASKS for suffix in (".jsonl", "_summary.json")]
        for name in names:
            path = gemini_root / model / name
            if not path.is_file():
                raise ValueError(f"Missing model data: {path}")
            if path.suffix == ".jsonl":
                rows = [sanitize(json.loads(line), {}) for line in path.read_text().splitlines() if line.strip()]
                content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            else:
                content = json.dumps(sanitize(json.loads(path.read_text()), {}), indent=2, ensure_ascii=False) + "\n"
            outputs.append((path, public_root / "runs/compactor" / model / name, content.encode("utf-8")))
    manifest = {
        "source_repository": "LLM_memory_bench",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paper_root, text=True).strip(),
        "human_export": "2026-07-27", "human_selection": "latest completed session for single-session tasks; best score for variable mapping, QA, map and craft",
        "records_per_task": dict(sorted(counts.items())),
        "completed_participants_per_task": {k: len(v) for k, v in participants.items()},
        "unique_completed_participants": len(set().union(*participants.values())),
        "anonymization": "Consistent participant/session/run pseudonyms, including filenames. Network/browser identifiers, completion codes, feedback and absolute collection timestamps are excluded. Completion order is retained for session selection. Private ID maps are not released.",
        "model_runs": MODELS,
        "limitations": ["Free recall has one missing selected human score (usable n=99).",
                        "Selected craft cohort includes two sessions using older crafting systems; scorer reports this and supports current-only filtering.",
                        "Gemini data are saved four-slot Compactor runs, not corrected context-isolation reruns.",
                        "Gemini prompt-only raw runs are unavailable; no supplied summary points are substituted.",
                        "Historical Compactor digit-span files encode many sequences under one participant ID."],
        "files": [{"source_directory": str(p.parent.relative_to(paper_root)), "path": str(t.relative_to(public_root)),
                   "sha256": hashlib.sha256(data).hexdigest()} for p, t, data in outputs],
    }
    if not dry_run:
        for _, target, data in outputs:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (public_root / "runs/data_update_2026-07-27.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = import_data(args.paper_root.resolve(), dry_run=args.dry_run)
    print(f"{'Prepared' if args.dry_run else 'Imported'} {len(manifest['files'])} files: "
          f"{sum(manifest['records_per_task'].values())} human records and two Gemini Compactor runs.")
    print("100 completed participants per task; 110 distinct participants across tasks.")


if __name__ == "__main__":
    main()
