"""Weeks of production captures in a bucket stay one bounded, fresh suite (spec, Testing)."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.captures.store_uri import StoreURI
from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config
from evalshift_cli.cli.main import app
from tests.unit.fake_remote_store import (
    TOOLSET_KEY,
    TOOLSET_PAYLOAD,
    FakeRemoteStore,
    capture_payload,
)

pytestmark = pytest.mark.integration

runner = CliRunner()


def _day(day: int, month: int = 6) -> datetime:
    return datetime(2026, month, day, 12, tzinfo=UTC)


class _Bucket:
    """A fake bucket the SDK 'writes' to between syncs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.store = FakeRemoteStore({TOOLSET_KEY: TOOLSET_PAYLOAD})
        monkeypatch.setattr(capture_module, "open_store", lambda parsed: self._open(parsed))

    def _open(self, parsed: StoreURI) -> FakeRemoteStore:
        return self.store

    def record(self, cid: str, when: datetime, *, model: str = "claude-sonnet-5-20260301") -> None:
        key = f"captures/planner/{cid}.json"
        self.store.objects[key] = capture_payload(
            cid,
            "planner",
            model_input=f"plan my day {cid}",
            created_at=when.isoformat(),
            model_id=model,
        )
        self.store.modified[key] = when


def _config(path: Path, *, source_model: str = "claude-sonnet-5") -> None:
    text = render_minimal_config(profile="model-upgrade")
    assert CAPTURES_SCAFFOLD + "\n" in text
    text = text.replace(CAPTURES_SCAFFOLD + "\n", "")
    text, n = re.subn(r"(?m)^  source_model: .*$", f"  source_model: {source_model}", text)
    assert n == 1
    text += "\ncaptures:\n  store: s3://acme-evals/testapp\n  max_age: 14d\n  max_cases: 4\n"
    path.write_text(text, encoding="utf-8")


def _sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now: datetime) -> Any:
    monkeypatch.setattr(capture_module, "_now", lambda: now)
    result = runner.invoke(
        app,
        ["capture", "sync", "--config", str(tmp_path / "evalshift.yaml"), "--base", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    return result


def _said(result: Any) -> str:
    """The command's stdout with Rich's line wrapping undone."""
    return " ".join(result.stdout.split())


def _cases(tmp_path: Path) -> set[str]:
    return {p.stem for p in (tmp_path / "suites" / "planner").glob("*.json")}


def test_weeks_of_captures_stay_one_bounded_fresh_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml")

    # Week 1: six captures, a suite of four.
    for i in range(6):
        bucket.record(f"w1_{i}", _day(10))
    _sync(tmp_path, monkeypatch, _day(12))
    week_1 = _cases(tmp_path)
    assert len(week_1) == 4

    # Week 2, inside max_age: more captures, but nothing is stale -- the suite holds still.
    for i in range(6):
        bucket.record(f"w2_{i}", _day(17))
    result = _sync(tmp_path, monkeypatch, _day(19))
    assert _cases(tmp_path) == week_1
    # 8 eligible = six week-2 captures + the two week-1 captures the cap left out.
    assert "planner: 4 cases — kept 4, retired 0, added 0 of 8 eligible" in _said(result)

    # Two weeks later: week-1 cases are stale and fresh captures replace them.
    for i in range(3):
        bucket.record(f"w4_{i}", _day(1, month=7))
    _sync(tmp_path, monkeypatch, _day(2, month=7))
    cases = _cases(tmp_path)
    assert len(cases) == 4
    assert {f"w4_{i}" for i in range(3)} <= cases
    # Only three replacements exist, so one stale week-1 case survives (replacement-only).
    assert len(cases & week_1) == 1

    # A fresh checkout: suites committed, local mirror gone. Promoted captures are not
    # downloaded again, and nothing outside max_age is -- only the one new capture is.
    shutil.rmtree(tmp_path / "captures")
    bucket.store.gets.clear()
    bucket.record("w5_0", _day(2, month=7))
    result = _sync(tmp_path, monkeypatch, _day(2, month=7))
    assert [k for k in bucket.store.gets if k.startswith("captures/")] == [
        "captures/planner/w5_0.json"
    ]
    assert "(skipped 4 promoted, 11 older than captures.max_age (14d))" in _said(result)
    # The new capture takes the stale week-1 survivor's slot.
    assert _cases(tmp_path) == {"w4_0", "w4_1", "w4_2", "w5_0"}


def test_a_production_migration_retires_the_old_models_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml")
    for i in range(4):
        bucket.record(f"old_{i}", _day(10))
    _sync(tmp_path, monkeypatch, _day(12))

    _config(tmp_path / "evalshift.yaml", source_model="claude-sonnet-5-5")
    for i in range(2):
        bucket.record(f"new_{i}", _day(13), model="claude-sonnet-5-5-20260601")
    result = _sync(tmp_path, monkeypatch, _day(14))

    assert {"new_0", "new_1"} <= _cases(tmp_path)
    assert len(_cases(tmp_path)) == 4
    assert "retired 2 (2 on another model)" in _said(result)


def test_a_wrong_source_model_trips_the_guardrail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml", source_model="gpt-9")
    for i in range(3):
        bucket.record(f"cap_{i}", _day(10))

    result = _sync(tmp_path, monkeypatch, _day(12))

    assert len(_cases(tmp_path)) == 3
    assert "not filtering by model — check defaults.source_model" in _said(result)


def test_a_source_model_typo_neither_empties_nor_retires_an_established_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml")
    for i in range(4):
        bucket.record(f"est_{i}", _day(10))
    _sync(tmp_path, monkeypatch, _day(12))
    established = _cases(tmp_path)
    assert len(established) == 4

    _config(tmp_path / "evalshift.yaml", source_model="claude-sonet-5")
    for i in range(2):
        bucket.record(f"more_{i}", _day(13))
    result = _sync(tmp_path, monkeypatch, _day(14))

    # The guardrail switches the model check off, so all four cases stay fresh: zero free slots
    # under max_cases 4, nothing stale to replace -- the suite holds, nothing on "another model".
    said = _said(result)
    assert "none of 6 case(s) and capture(s) were recorded on claude-sonet-5" in said
    assert "not filtering by model — check defaults.source_model" in said
    assert "planner: 4 cases — kept 4, retired 0, added 0 of 2 eligible" in said
    assert "on another model" not in said
    assert _cases(tmp_path) == established
