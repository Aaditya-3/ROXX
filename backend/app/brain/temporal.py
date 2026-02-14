"""
Temporal parsing utilities that normalize relative phrases to absolute dates.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta


def _iso_date(dt: datetime) -> str:
    return dt.date().isoformat()


def parse_date_from_text(text: str, today: date) -> str | None:
    """
    Convert natural language date fragments into YYYY-MM-DD.
    Returns None when parsing fails.
    """
    source = (text or "").strip()
    if not source:
        return None
    lower = source.lower()

    # Fast ISO fallback.
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", source)
    if m:
        return m.group(1)

    # Numeric date fallbacks.
    m = re.search(r"\b(20\d{2})/(\d{1,2})/(\d{1,2})\b", source)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            pass
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", source)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            pass

    # Natural language parsing.
    try:
        import dateparser  # type: ignore
        from dateparser.search import search_dates  # type: ignore

        base = datetime.combine(today, time.min)
        parsed = search_dates(
            source,
            settings={
                "RELATIVE_BASE": base,
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
            languages=["en"],
        )
        if parsed:
            return _iso_date(parsed[0][1])

        dt = dateparser.parse(
            source,
            settings={
                "RELATIVE_BASE": base,
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
            languages=["en"],
        )
        if dt is not None:
            return _iso_date(dt)
    except Exception:
        pass

    # Rule-based fallback for relative day words.
    if "day after tomorrow" in lower:
        return (today + timedelta(days=2)).isoformat()
    if "tomorrow" in lower:
        return (today + timedelta(days=1)).isoformat()
    if "today" in lower:
        return today.isoformat()
    if "yesterday" in lower:
        return (today - timedelta(days=1)).isoformat()

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for wd, wd_idx in weekday_map.items():
        if wd not in lower:
            continue
        delta = (wd_idx - today.weekday()) % 7
        if re.search(rf"\b(?:next|upcoming)\s+{wd}\b", lower):
            if delta == 0:
                delta = 7
        elif re.search(rf"\bthis\s+{wd}\b", lower):
            pass
        elif delta == 0:
            delta = 7
        return (today + timedelta(days=delta)).isoformat()
    return None
