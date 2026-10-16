"""The duration grammar shared by ``--since`` and ``captures.max_age``."""

from __future__ import annotations

from datetime import timedelta

import pytest

from evalshift_cli.utils.duration import format_duration, parse_duration


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30m", timedelta(minutes=30)),
        ("24h", timedelta(hours=24)),
        ("14d", timedelta(days=14)),
        (" 7d ", timedelta(days=7)),
        ("0d", timedelta(0)),
    ],
)
def test_parse_duration_accepts_minutes_hours_days(text: str, expected: timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "14", "2w", "1.5d", "-1d", "d", "14 d", "P14D"])
def test_parse_duration_rejects_anything_else(text: str) -> None:
    with pytest.raises(ValueError, match="expected a duration like 30m, 24h or 14d"):
        parse_duration(text)


def test_parse_duration_overflow_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="too large"):
        parse_duration("9999999999999d")


@pytest.mark.parametrize(
    ("delta", "text"),
    [(timedelta(days=14), "14d"), (timedelta(hours=36), "36h"), (timedelta(minutes=90), "90m")],
)
def test_format_duration_uses_the_largest_whole_unit(delta: timedelta, text: str) -> None:
    assert format_duration(delta) == text
