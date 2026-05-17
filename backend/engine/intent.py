"""
intent.py — Rule-based intent classifier (Phase 1/2).

Classifies browser events and chat messages into structured intent objects.
No LLM needed at this phase — URL patterns + keyword matching is enough.
Phase 4: add LLM path (Ollama) for ambiguous cases.
"""

import re
from datetime import datetime, timezone
from typing import Any

from backend.utils.time_utils import time_bucket_for_hour

# ── Task-creation language patterns ─────────────────────────────────────────
_TASK_TRIGGERS = re.compile(
    r"\b(remind me|don't forget|i need to|i should|i have to|todo|to-do|"
    r"later i'll|don't let me forget|must|schedule|book|call|email|send|"
    r"follow up|follow-up|check on|ping|draft|write|review|fix|submit)\b",
    re.IGNORECASE,
)

# ── Intent categories from URL patterns ─────────────────────────────────────
_URL_INTENT_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(google|bing|duckduckgo|yahoo)\.com/search"), "research", "search"),
    (re.compile(r"(youtube\.com|youtu\.be)"), "entertainment", "video"),
    (re.compile(r"(github\.com|gitlab\.com|stackoverflow\.com)"), "development", "code"),
    (re.compile(r"(linkedin\.com)"), "career", "networking"),
    (re.compile(r"(amazon|flipkart|myntra|ebay|etsy)\.(com|in)"), "shopping", "ecommerce"),
    (re.compile(r"(booking\.com|airbnb|makemytrip|skyscanner|expedia)"), "travel", "booking"),
    (re.compile(r"(gmail|outlook|mail\.yahoo|proton\.me)"), "communication", "email"),
    (re.compile(r"(notion\.so|obsidian\.md|roamresearch|logseq)"), "productivity", "notes"),
    (re.compile(r"(figma\.com|canva\.com|sketch\.com)"), "design", "tools"),
    (re.compile(r"(twitter|x\.com|instagram|facebook|threads)"), "social", "social_media"),
    (re.compile(r"(docs\.google|sheets\.google|slides\.google)"), "productivity", "docs"),
    (re.compile(r"(calendar\.google|cal\.com|fantastical)"), "planning", "calendar"),
    (re.compile(r"(coursera|udemy|edx|brilliant\.org|khanacademy)"), "learning", "education"),
]


def classify_url_intent(url: str) -> dict[str, Any]:
    """Match a URL against known patterns and return an intent object."""
    for pattern, intent, category in _URL_INTENT_RULES:
        if pattern.search(url):
            return {"intent": intent, "category": category, "source": "url_rule"}
    return {"intent": "browsing", "category": "general", "source": "url_rule"}


def classify_text_intent(text: str) -> dict[str, Any]:
    """
    Detect task-creation intent in user text.
    Returns is_task=True plus the raw text for task creation downstream.
    """
    if _TASK_TRIGGERS.search(text):
        return {"is_task": True, "intent": "task_creation", "source": "text_rule"}
    return {"is_task": False, "intent": "chat", "source": "text_rule"}


def classify_intent(event: dict) -> dict[str, Any]:
    """
    Master classifier — dispatches based on event type.
    Attaches a time_bucket to every classified event.
    """
    now = datetime.now(timezone.utc)
    bucket = time_bucket_for_hour(now.hour)

    event_type = event.get("type", "unknown")
    result: dict[str, Any] = {"time_bucket": bucket}

    if event_type == "navigation":
        url = event.get("url", "")
        result.update(classify_url_intent(url))

    elif event_type == "chat":
        text = event.get("text", "")
        result.update(classify_text_intent(text))

    elif event_type == "search":
        result["intent"] = "research"
        result["category"] = event.get("category", "general")
        result["source"] = "search_event"

    else:
        result["intent"] = "unknown"
        result["category"] = "unknown"
        result["source"] = "fallback"

    return result


# Module-level convenience alias
class IntentClassifier:
    def classify(self, event: dict) -> dict[str, Any]:
        return classify_intent(event)
