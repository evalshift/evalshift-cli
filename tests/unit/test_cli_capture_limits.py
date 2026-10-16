"""``capture sync`` with ``captures.max_age`` / ``captures.max_cases`` set (bounded suites)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config
from evalshift_cli.cli.main import app
from evalshift_cli.suite.loader import load_jsonl
from tests.unit.fake_remote_store import TOOLSET_HEX, TOOLSET_PAYLOAD, capture_payload

runner = CliRunner()

WEEK_1 = "2026-06-10T12:00:00+00:00"
WEEK_3 = "2026-06-29T12:00:00+00:00"


def _at(monkeypatch: pytest.MonkeyPatch, day: int) -> None:
    """Pin sync's clock to noon UTC on 2026-06-<day>."""
    monkeypatch.setattr(capture_module, "_now", lambda: datetime(2026, 6, day, 12, tzinfo=UTC))


def _put(
    base: Path,
    cid: str,
    *,
    suite: str = "alpha",
    created_at: str = WEEK_1,
    model_id: str = "claude-sonnet-5-20260301",
    text: str | None = None,
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> None:
    """Write one local capture and the shared toolset sidecar it points at."""
    sidecar = base / "toolsets" / f"{TOOLSET_HEX}.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(TOOLSET_PAYLOAD)
    path = base / "captures" / suite / f"{cid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        capture_payload(
            cid,
            suite,
            model_input=text if text is not None else f"question {cid}?",
            created_at=created_at,
            model_id=model_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
    )


def _config(
    path: Path,
    *,
    max_age: str | None = "14d",
    max_cases: int | None = 3,
    source_model: str = "claude-sonnet-5",
) -> Path:
    text = render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
    text = re.sub(r"(?m)^  source_model: .*$", f"  source_model: {source_model}", text)
    limits = (f"  max_age: {max_age}\n" if max_age else "") + (
        f"  max_cases: {max_cases}\n" if max_cases else ""
    )
    if limits:
        text += "\ncaptures:\n" + limits
    path.write_text(text, encoding="utf-8")
    return path


def _sync(tmp_path: Path, *args: str) -> Any:
    result = runner.invoke(
        app,
        [
            "capture",
            "sync",
            "--config",
            str(tmp_path / "evalshift.yaml"),
            "--base",
            str(tmp_path),
            *args,
        ],
    )
    assert result.exit_code == 0, result.stdout
    return result


def _flat(result: Any) -> str:
    return " ".join(result.stdout.split())


def _cases(tmp_path: Path, suite: str = "alpha") -> set[str]:
    return {p.stem for p in (tmp_path / "suites" / suite).glob("*.json")}


def test_a_first_sync_caps_the_suite_at_max_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert len(load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").examples) == 3
    assert "alpha: 3 cases — kept 0, retired 0, added 3 of 5 eligible" in _flat(result)


def test_syncing_again_with_nothing_new_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    suite_dir = tmp_path / "suites" / "alpha"
    before = {p.name: p.read_bytes() for p in suite_dir.iterdir()}

    result = _sync(tmp_path)

    assert {p.name: p.read_bytes() for p in suite_dir.iterdir()} == before
    assert "alpha: 3 cases — kept 3, retired 0, added 0 of 2 eligible" in _flat(result)


def test_stale_cases_are_swapped_for_fresh_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"old_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new_0", created_at=WEEK_3)
    _put(tmp_path, "new_1", created_at=WEEK_3)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"old_2", "new_0", "new_1"}
    assert "alpha: 3 cases — kept 1, retired 2 (2 older than 14d), added 2 of 2 eligible" in _flat(
        result
    )


def test_a_stale_suite_stays_full_when_nothing_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"old_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert "alpha: 3 cases — kept 3, retired 0, added 0 of 0 eligible" in _flat(result)


def test_cases_recorded_on_the_old_model_leave_after_a_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"old_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new_0", created_at="2026-06-13T12:00:00+00:00", model_id="claude-sonnet-5-5")
    _config(tmp_path / "evalshift.yaml", source_model="claude-sonnet-5-5")
    _at(monkeypatch, 14)

    result = _sync(tmp_path)

    assert "new_0" in _cases(tmp_path)
    assert len(_cases(tmp_path)) == 3
    assert "retired 1 (1 on another model), added 1 of 1 eligible" in _flat(result)


def test_a_source_model_no_capture_ran_on_filters_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", source_model="gpt-9")
    _at(monkeypatch, 12)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    flat = _flat(result)
    assert "alpha: none of 3 case(s) and capture(s) were recorded on gpt-9" in flat
    assert "(seen: claude-sonnet-5-20260301 ×3)" in flat
    assert "not filtering by model — check defaults.source_model" in flat


def test_captures_outside_the_window_are_not_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(tmp_path, "old")
    _put(tmp_path, "new", created_at=WEEK_3)
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"new"}
    assert "added 1 of 1 eligible; not eligible: 1 older than 14d" in _flat(result)


def test_max_age_alone_filters_without_capping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"new_{i}", created_at=WEEK_3)
    _put(tmp_path, "old")
    _config(tmp_path / "evalshift.yaml", max_cases=None)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 5
    assert "alpha: not eligible: 1 older than 14d" in _flat(result)


def test_an_over_cap_suite_is_trimmed_without_new_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    for path in (tmp_path / "captures" / "alpha").glob("*.json"):
        path.unlink()
    _config(tmp_path / "evalshift.yaml", max_cases=3)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert len(load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").examples) == 3
    assert "alpha: 3 cases — kept 3, retired 2 (2 over max_cases), added 0 of 0 eligible" in _flat(
        result
    )


def test_suite_flag_leaves_other_suites_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for suite in ("alpha", "beta"):
        for i in range(5):
            _put(tmp_path, f"{suite}_{i}", suite=suite)
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _config(tmp_path / "evalshift.yaml", max_cases=3)

    _sync(tmp_path, "--suite", "alpha")

    assert len(_cases(tmp_path, "alpha")) == 3
    assert len(_cases(tmp_path, "beta")) == 5


def test_a_conversation_enters_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for turn in range(3):
        _put(tmp_path, f"conv_t{turn}", conversation_id="c1", turn_index=turn)
    _put(tmp_path, "solo")
    _config(tmp_path / "evalshift.yaml", max_cases=2)
    _at(monkeypatch, 12)

    _sync(tmp_path)

    assert _cases(tmp_path) == {"solo"}


def test_the_same_question_captured_fresh_replaces_its_stale_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(tmp_path, "old", text="where is order 1?")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new", text="where is order 1?", created_at=WEEK_3)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"new"}
    assert "retired 1 (1 older than 14d), added 1 of 1 eligible" in _flat(result)


def test_without_limits_sync_promotes_everything_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None, source_model="gpt-9")
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 5
    assert "cases — kept" not in _flat(result)
    assert "not filtering by model" not in _flat(result)


def test_max_age_alone_keeps_a_stale_case_from_gaining_a_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without max_cases nothing is retired, so the fresh copy would sit next to the stale one.
    _put(tmp_path, "old", text="where is order 1?")
    _config(tmp_path / "evalshift.yaml", max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new", text="where is order 1?", created_at=WEEK_3)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"old"}
    assert len(load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").examples) == 1
    assert "skipped 1 duplicate capture(s)" in _flat(result)


def test_a_reviewed_stale_case_keeps_its_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reviewed case is never retired, so its fresh twin must not be added beside it.
    _put(tmp_path, "old", text="where is order 1?")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    case_path = tmp_path / "suites" / "alpha" / "old.json"
    data = json.loads(case_path.read_text(encoding="utf-8"))
    data["example"]["expected_tools"] = [{"tool_name": "lookup", "provenance": "reviewed"}]
    case_path.write_text(json.dumps(data), encoding="utf-8")
    _put(tmp_path, "new", text="where is order 1?", created_at=WEEK_3)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"old"}
    assert "alpha: 1 cases — kept 1, retired 0, added 0 of 0 eligible" in _flat(result)


def test_reviewed_cases_over_the_cap_stay_and_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    for case_path in (tmp_path / "suites" / "alpha").glob("*.json"):
        data = json.loads(case_path.read_text(encoding="utf-8"))
        data["example"]["expected_tools"] = [{"tool_name": "lookup", "provenance": "reviewed"}]
        case_path.write_text(json.dumps(data), encoding="utf-8")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=2)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert (
        "alpha: 1 reviewed case(s) over captures.max_cases 2; reviewed cases are never "
        "retired, so it stays over." in _flat(result)
    )


def test_a_suite_whose_cases_cannot_be_read_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    (tmp_path / "suites" / "alpha" / "broken.json").write_text("{", encoding="utf-8")
    _put(tmp_path, "cap_new")
    _config(tmp_path / "evalshift.yaml", max_cases=2)

    result = runner.invoke(
        app,
        ["capture", "sync", "--config", str(tmp_path / "evalshift.yaml"), "--base", str(tmp_path)],
    )

    assert _cases(tmp_path) == {"cap_0", "cap_1", "cap_2", "broken"}
    assert "alpha: could not read its promoted cases" in _flat(result)
