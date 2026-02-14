"""
Rule-based intent classification for pre-LLM orchestration.
"""

from __future__ import annotations

import re

from backend.app.brain.models import Intent


_GREETING_RE = re.compile(r"^(hi|hello|hey|yo|good\s+morning|good\s+afternoon|good\s+evening)\b[!.,\s]*$", re.I)
_MEMORY_QUERY_PATTERNS = [
    r"\bwhat\s+is\s+my\b",
    r"\bwhat\s+are\s+my\b",
    r"\bwhen\s+(?:is|was)\s+my\b",
    r"\bdo\s+you\s+remember\b",
    r"\bwhat\s+do\s+i\s+like\b",
    r"\bwhat\s+is\s+my\s+favorite\b",
]
_MEMORY_UPDATE_PATTERNS = [
    r"\bi\s+like\b",
    r"\bi\s+love\b",
    r"\bi\s+prefer\b",
    r"\bi\s+dislike\b",
    r"\bi\s+hate\b",
    r"\bmy\s+\w+\s+(?:is|was|will\s+be|on)\b",
    r"\bi\s+have\b",
    r"\bi\s+am\b",
    r"\bi'm\b",
]
_GUESS_PATTERNS = [
    r"\bguess\b",
    r"\bwhat\s+would\s+i\s+like\b",
    r"\bguess\s+my\b",
]


def classify_intent(message: str) -> Intent:
    text = (message or "").strip()
    if not text:
        return Intent.OTHER
    if _GREETING_RE.match(text):
        return Intent.GREETING

    lowered = text.lower()
    if any(re.search(p, lowered) for p in _GUESS_PATTERNS):
        return Intent.GUESS_REQUEST
    if any(re.search(p, lowered) for p in _MEMORY_QUERY_PATTERNS):
        return Intent.MEMORY_QUERY
    if any(re.search(p, lowered) for p in _MEMORY_UPDATE_PATTERNS):
        return Intent.MEMORY_UPDATE
    if "?" in lowered:
        return Intent.FACTUAL_QUESTION
    return Intent.OTHER
