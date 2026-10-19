"""The duration grammar shared by ``--since`` and ``captures.max_age``: ``N`` + ``m|h|d``."""

from __future__ import annotations

import re
from datetime import timedelta

#: How an error names the accepted forms.
DURATION_FORMS = "30m, 24h or 14d"

_DURATION = re.compile(r"(\d+)([mhd])")
_UNITS = {"m": timedelta(minutes=1), "h": timedelta(hours=1), "d": timedelta(days=1)}


def parse_duration(text: str) -> timedelta:
    """Parse ``30m`` / ``24h`` / ``14d`` (surrounding whitespace allowed) into a timedelta.

    Raises:
        ValueError: when ``text`` is not that form, or is too large for a timedelta.
    """
    match = _DURATION.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"expected a duration like {DURATION_FORMS}, got {text!r}")
    try:
        return _UNITS[match.group(2)] * int(match.group(1))
    except OverflowError as exc:
        raise ValueError(f"duration {text!r} is too large") from exc


def format_duration(delta: timedelta) -> str:
    """Render ``delta`` in the largest whole unit :func:`parse_duration` reads (``14d``, ``36h``)."""
    seconds = int(delta.total_seconds())
    for suffix, unit in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % unit == 0:
            return f"{seconds // unit}{suffix}"
    return f"{seconds}s"
