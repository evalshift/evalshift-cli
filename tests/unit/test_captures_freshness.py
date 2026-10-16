"""``suite_freshness``: what doctor and compare say about a committed capture suite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from evalshift_cli.captures.freshness import SuiteFreshness, suite_freshness
from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.promote import write_promoted_case
from evalshift_cli.suite.models import SuiteExample

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


def _write(base: Path, i: int, *, days: int, model: str = "claude-sonnet-5") -> None:
    write_promoted_case(
        PromotedCase(
            name=f"cap_{i}",
            suite="alpha",
            from_capture=f"cap_{i}",
            captured_at=(NOW - timedelta(days=days)).isoformat(),
            source_model=model,
            example=SuiteExample(id=f"cap_{i}", inputs={"input": f"q{i}"}, tools=[]),
        ),
        base=base,
    )


def _check(base: Path, **limits: Any) -> SuiteFreshness:
    options: dict[str, Any] = {
        "max_age": timedelta(days=14),
        "max_cases": 5,
        "source_model": "claude-sonnet-5",
        **limits,
    }
    fresh = suite_freshness("alpha", base / "suites" / "alpha", now=NOW, **options)
    assert fresh is not None
    return fresh


def test_no_case_files_means_nothing_to_report(tmp_path: Path) -> None:
    assert (
        suite_freshness(
            "alpha",
            tmp_path / "suites" / "alpha",
            max_age=timedelta(days=14),
            max_cases=5,
            source_model="claude-sonnet-5",
            now=NOW,
        )
        is None
    )


def test_a_fresh_suite_has_no_findings(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=2)
    _write(tmp_path, 1, days=12)
    fresh = _check(tmp_path)
    assert fresh.findings == ()
    assert fresh.describe(NOW) == "alpha: 2 cases, captured 2d–12d ago"  # noqa: RUF001 — doctor copy


def test_a_suite_past_max_age_says_how_to_refresh(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=23)
    [finding] = _check(tmp_path).findings
    assert finding.kind == "stale_age"
    assert finding.message == (
        "alpha: newest case captured 23d ago (captures.max_age 14d); "
        "refresh with evalshift capture sync"
    )


def test_cases_on_another_model_are_counted(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=1)
    _write(tmp_path, 1, days=1, model="claude-sonnet-4-5")
    [finding] = _check(tmp_path).findings
    assert finding.kind == "stale_model"
    assert finding.message.startswith(
        "alpha: 1 of 2 cases recorded on another model than claude-sonnet-5"
    )


def test_no_case_on_the_source_model_is_its_own_finding(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=1, model="gpt-5")
    [finding] = _check(tmp_path).findings
    assert finding.kind == "no_source_model_case"
    assert "no case recorded on claude-sonnet-5 (seen: gpt-5 ×1)" in finding.message


def test_a_suite_over_its_cap_is_reported(tmp_path: Path) -> None:
    for i in range(3):
        _write(tmp_path, i, days=1)
    [finding] = _check(tmp_path, max_cases=2).findings
    assert finding.message == (
        "alpha: 3 cases, over captures.max_cases 2; evalshift capture sync trims it"
    )


def test_an_unreadable_case_file_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=1)
    (tmp_path / "suites" / "alpha" / "broken.json").write_text("{", encoding="utf-8")
    fresh = _check(tmp_path)
    assert fresh.cases == 1
    assert [f.kind for f in fresh.findings] == ["unreadable"]


def test_a_legacy_case_ages_from_its_promotion(tmp_path: Path) -> None:
    write_promoted_case(
        PromotedCase(
            name="old",
            suite="alpha",
            from_capture="old",
            promoted_at=(NOW - timedelta(days=30)).isoformat(),
            example=SuiteExample(id="old", inputs={"input": "q"}, tools=[]),
        ),
        base=tmp_path,
    )
    assert [f.kind for f in _check(tmp_path).findings] == ["stale_age"]
