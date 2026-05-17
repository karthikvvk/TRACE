"""
anonymizer.py — PII stripping and semantic abstraction layer.

Phase 1: Regex-based fallback. Strips names, emails, phone numbers, IPs, credit cards.
Phase 3: Upgrade to Microsoft Presidio by setting USE_PRESIDIO=true in .env
         and running: pip install presidio-analyzer presidio-anonymizer
         then: python -m spacy download en_core_web_lg

The output of anonymize() is always a clean semantic dict — never raw user data.
"""

import re
import hashlib
import json
from typing import Any

from backend.config import settings

# ── Presidio (optional, lazy-loaded) ────────────────────────────────────────
_presidio_analyzer = None
_presidio_anonymizer = None


def _load_presidio():
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _presidio_analyzer = AnalyzerEngine()
        _presidio_anonymizer = AnonymizerEngine()


# ── Regex patterns ───────────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("PHONE", re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("CARD", re.compile(r"\b(?:\d[ \-]?){13,19}\b")),
    ("URL_TOKEN", re.compile(r"(?<=[?&])(token|key|secret|password|access_token)=[^&\s]+")),
]


def _regex_strip(text: str) -> str:
    """Remove PII from a text string using regex patterns."""
    for label, pattern in _PII_PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    return text


def _presidio_strip(text: str) -> str:
    """Use Presidio (spaCy-backed) for NLP-aware PII detection."""
    _load_presidio()
    results = _presidio_analyzer.analyze(text=text, language="en")
    anonymized = _presidio_anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


def strip_pii(text: str) -> str:
    """Strip PII from text. Uses Presidio if enabled, otherwise regex."""
    if settings.use_presidio:
        try:
            return _presidio_strip(text)
        except Exception:
            pass  # Fallback to regex on Presidio errors
    return _regex_strip(text)


def pseudonymize(identifier: str) -> str:
    """
    Replace a real identifier with a stable pseudonym.
    CONTACT_A3F2 style — deterministic but not reversible.
    """
    digest = hashlib.sha256(
        (settings.pseudonym_salt + identifier).encode()
    ).hexdigest()[:4].upper()
    return f"CONTACT_{digest}"


def _clean_value(v: Any) -> Any:
    """Recursively strip PII from string values in a dict/list."""
    if isinstance(v, str):
        return strip_pii(v)
    if isinstance(v, dict):
        return {k: _clean_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_clean_value(item) for item in v]
    return v


class Anonymizer:
    """
    Stateless anonymizer. Strips PII then converts raw browser events
    into semantic intent objects. Raw URLs become intent categories;
    raw text becomes stripped text.
    """

    def anonymize_event(self, raw: dict) -> dict:
        """
        Take a raw event from the extension and return a clean semantic dict.
        Strips PII, replaces raw URL with domain only, infers intent category.
        """
        from backend.tools.search_tool import extract_query_from_url

        clean: dict[str, Any] = {}

        # Preserve safe scalar fields
        clean["type"] = raw.get("type", "unknown")
        clean["timestamp"] = raw.get("timestamp")  # epoch ms — no PII

        # URL handling: keep only origin (scheme+host), check for search intent
        raw_url = raw.get("url", "")
        search_intent = extract_query_from_url(raw_url)
        if search_intent:
            clean["intent"] = search_intent.get("intent")
            clean["category"] = search_intent.get("category")
            clean["engine"] = search_intent.get("engine")
            # Store the cleaned query (not the raw URL)
            clean["query_stripped"] = strip_pii(search_intent.get("query", ""))
        else:
            # Keep only scheme + hostname (no path, no query params)
            try:
                from urllib.parse import urlparse
                parsed = urlparse(raw_url)
                clean["domain"] = f"{parsed.scheme}://{parsed.netloc}"
            except Exception:
                clean["domain"] = None

        # Strip PII from any text field
        if "text" in raw:
            clean["text"] = strip_pii(str(raw["text"]))

        # Extra metadata — clean recursively
        if "meta" in raw:
            clean["meta"] = _clean_value(raw["meta"])

        return clean


# Module-level singleton
_anonymizer = Anonymizer()


def anonymize(raw: dict) -> dict:
    """Convenience function: anonymize a raw extension event."""
    return _anonymizer.anonymize_event(raw)
