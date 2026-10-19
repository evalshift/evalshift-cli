"""What makes a promoted case stale (spec §3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.staleness import (
    NO_TOOLS_ROUTE,
    StalenessRules,
    case_instant,
    examples_route,
    is_reviewed,
    parse_instant,
)
from evalshift_cli.suite.models import ExpectedToolCall, SuiteExample

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
#: A non-empty toolset (expected_tools may not sit beside an empty one).
_REF = "sha256:" + "cd" * 32


def _example(*tools: str, reviewed: bool = False) -> SuiteExample:
    calls = [
        ExpectedToolCall(tool_name=t, provenance="reviewed" if reviewed else "captured")
        for t in tools
    ]
    return SuiteExample(id="c", toolset_ref=_REF, expected_tools=calls or None)


def _case(*, captured_at: str = "", promoted_at: str = "") -> PromotedCase:
    return PromotedCase(
        name="c",
        suite="s",
        from_capture="cap",
        captured_at=captured_at,
        promoted_at=promoted_at,
        example=_example(),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-01T12:00:00+00:00", datetime(2026, 10, 1, 12, tzinfo=UTC)),
        ("2026-10-01T12:00:00Z", datetime(2026, 10, 1, 12, tzinfo=UTC)),
        ("2026-10-01T14:00:00+02:00", datetime(2026, 10, 1, 12, tzinfo=UTC)),
        ("2026-10-01T12:00:00", datetime(2026, 10, 1, 12, tzinfo=UTC)),
    ],
)
def test_parse_instant_normalises_to_utc(text: str, expected: datetime) -> None:
    assert parse_instant(text) == expected


@pytest.mark.parametrize(
    "text", ["", "   ", "yesterday", "2026-13-01", "0001-01-01T00:00:00+05:00"]
)
def test_parse_instant_is_none_for_anything_unparseable(text: str) -> None:
    assert parse_instant(text) is None


def test_case_instant_prefers_the_capture_time() -> None:
    case = _case(captured_at="2026-10-01T00:00:00+00:00", promoted_at="2026-10-05T00:00:00+00:00")
    assert case_instant(case) == datetime(2026, 10, 1, tzinfo=UTC)


def test_case_instant_falls_back_to_the_promotion_time() -> None:
    case = _case(promoted_at="2026-10-05T00:00:00+00:00")
    assert case_instant(case) == datetime(2026, 10, 5, tzinfo=UTC)


def test_case_instant_is_none_when_neither_parses() -> None:
    assert case_instant(_case()) is None


def test_is_reviewed_reads_any_expectation() -> None:
    assert is_reviewed(_example("a", reviewed=True))
    assert not is_reviewed(_example("a"))
    assert not is_reviewed(_example())


def test_is_reviewed_reads_every_round() -> None:
    # A mark in a later round pins the case too: is_reviewed guards deletion.
    captured = ExpectedToolCall(tool_name="a", provenance="captured")
    reviewed = ExpectedToolCall(tool_name="b", provenance="reviewed")
    example = SuiteExample(
        id="c",
        toolset_ref=_REF,
        expected_tools=[captured],
        expected_tool_rounds=[[captured], [reviewed]],
    )
    assert is_reviewed(example)
    unreviewed = example.model_copy(update={"expected_tool_rounds": [[captured], [captured]]})
    assert not is_reviewed(unreviewed)


def test_route_is_the_sorted_union_of_expected_tool_names() -> None:
    assert examples_route([_example("b", "a")]) == "a,b"
    assert examples_route([_example("a"), _example("c")]) == "a,c"
    assert examples_route([_example()]) == NO_TOOLS_ROUTE


def test_an_old_case_is_age_stale() -> None:
    rules = StalenessRules(now=NOW, max_age=timedelta(days=14))
    assert rules.stale_reason(NOW - timedelta(days=15), "m") == "age"
    assert rules.stale_reason(NOW - timedelta(days=13), "m") is None
    assert rules.stale_reason(None, "m") is None


def test_another_model_is_model_stale_before_age() -> None:
    rules = StalenessRules(now=NOW, max_age=timedelta(days=14), source_model="claude-sonnet-5-5")
    old = NOW - timedelta(days=30)
    assert rules.stale_reason(old, "claude-sonnet-5-20260301") == "model"
    assert rules.stale_reason(old, "claude-sonnet-5-5") == "age"
    assert rules.stale_reason(NOW, "") is None


def test_no_limits_means_nothing_is_stale() -> None:
    assert StalenessRules(now=NOW).stale_reason(NOW - timedelta(days=999), "x") is None
