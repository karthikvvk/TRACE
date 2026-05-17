"""time_utils.py — Shared time helper utilities."""

from datetime import datetime, timezone


def time_bucket_for_hour(hour: int) -> str:
    """Map a UTC hour to a fuzzy time bucket."""
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat()
