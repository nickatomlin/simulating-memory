"""Privacy helpers for releasing human experimental records.

Keep participant pseudonyms, experimental responses, durations and completion
order. Strip network/browser details, credentials and absolute collection times.
Private identifier maps are accepted in memory and never written to the release.
"""
from __future__ import annotations

from datetime import datetime, timezone

PRIVATE_FIELDS = {
    "ip", "ipaddress", "remoteaddr", "useragent", "completioncode",
    "apikey", "authorization", "prolificid", "prolificpid", "workerid", "email",
    "request", "feedbacktext",
}
ABSOLUTE_TIME_FIELDS = {
    "startedat", "completedat", "createdat", "updatedat", "startedatms",
    "finishedatms", "timestamp",
}


def field_name(key):
    return str(key).lower().replace("_", "").replace("-", "")


def clean_record(value, identities=None):
    identities = identities or {}
    if isinstance(value, dict):
        return {key: clean_record(item, identities) for key, item in value.items()
                if field_name(key) not in PRIVATE_FIELDS | ABSOLUTE_TIME_FIELDS}
    if isinstance(value, list):
        return [clean_record(item, identities) for item in value]
    if isinstance(value, str):
        return identities.get(value, value)
    return value


def completion_key(record, filename=""):
    if record.get("completion_order") is not None:
        return float(record["completion_order"]), filename
    for field in ("completed_at", "updated_at", "created_at", "started_at"):
        value = record.get(field)
        if value:
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).timestamp(), filename
            except ValueError:
                pass
    return 0., filename
