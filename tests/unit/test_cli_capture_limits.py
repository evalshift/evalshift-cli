"""``capture sync`` with ``captures.max_age`` / ``captures.max_cases`` set (bounded suites)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands._suites import parse_suites_region
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
    text: Any = None,
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


def test_a_conversation_too_long_for_the_slots_left_enters_as_its_leading_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for turn in range(3):
        _put(tmp_path, f"conv_t{turn}", conversation_id="c1", turn_index=turn)
    _put(tmp_path, "solo")
    _config(tmp_path / "evalshift.yaml", max_cases=2)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    suite_dir = tmp_path / "suites" / "alpha"
    before = {p.name: p.read_bytes() for p in suite_dir.iterdir()}

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"solo", "conv_t0"}
    assert {p.name: p.read_bytes() for p in suite_dir.iterdir()} == before
    assert "alpha: 2 cases — kept 2, retired 0, added 0 of 2 eligible" in _flat(result)


def test_a_turn_whose_earlier_turns_are_not_local_is_not_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Turns 0-1 were promoted elsewhere, or fell outside the fetch budget: turns 2-3 recorded
    # a bare string, so their history could only be rebuilt from turns that are not here.
    for turn in (2, 3):
        _put(tmp_path, f"conv_t{turn}", conversation_id="c1", turn_index=turn)
    _put(tmp_path, "solo")
    _config(tmp_path / "evalshift.yaml", max_cases=5)
    _at(monkeypatch, 12)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"solo"}
    assert "not eligible: 2 missing earlier turns" in _flat(result)


def test_a_turn_that_recorded_its_own_history_needs_no_earlier_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages = [
        {"role": "user", "content": "convert 10 EUR"},
        {"role": "assistant", "content": "10 EUR is 10.87 USD"},
        {"role": "user", "content": "and to GBP?"},
    ]
    _put(tmp_path, "conv_t1", conversation_id="c1", turn_index=1, text=messages)
    _config(tmp_path / "evalshift.yaml", max_cases=5)
    _at(monkeypatch, 12)

    _sync(tmp_path)

    assert _cases(tmp_path) == {"conv_t1"}


def test_without_limits_a_turn_missing_earlier_turns_still_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(tmp_path, "conv_t2", conversation_id="c1", turn_index=2)
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)

    _sync(tmp_path)

    assert _cases(tmp_path) == {"conv_t2"}


def test_build_warnings_name_only_the_captures_that_became_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = capture_module.build_conversation_examples

    def warn_every_capture(*args: Any, **kwargs: Any) -> Any:
        return [
            (record, replace(built, warnings=[*built.warnings, "WARNED"]))
            for record, built in real(*args, **kwargs)
        ]

    monkeypatch.setattr(capture_module, "build_conversation_examples", warn_every_capture)
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_cases=2)
    _at(monkeypatch, 12)

    result = _sync(tmp_path)

    warned = set(re.findall(r"(cap_\d): WARNED", result.stdout))
    assert warned == _cases(tmp_path)
    assert len(warned) == 2


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
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    before = golden.read_bytes()
    (tmp_path / "suites" / "alpha" / "broken.json").write_text("{", encoding="utf-8")
    _put(tmp_path, "cap_new")
    _config(tmp_path / "evalshift.yaml", max_cases=2)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"cap_0", "cap_1", "cap_2", "broken"}
    assert golden.read_bytes() == before
    assert "alpha: could not read its promoted cases" in _flat(result)


def test_a_stray_unreadable_suite_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    for path in (tmp_path / "captures" / "alpha").glob("*.json"):
        path.unlink()
    stray = tmp_path / "suites" / "aaa" / "notes.json"
    stray.parent.mkdir(parents=True)
    stray.write_text('{"note": "not a case"}', encoding="utf-8")
    _config(tmp_path / "evalshift.yaml", max_cases=3)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert len(load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").examples) == 3
    assert not (tmp_path / "suites" / "aaa" / "golden.jsonl").exists()
    flat = _flat(result)
    assert "aaa: could not read its promoted cases" in flat
    assert "✓ promoted 0 capture(s) into 1 suite(s)" in flat
    assert "alpha: 3 cases — kept 3, retired 2 (2 over max_cases)" in flat
    assert "  aaa:" not in (tmp_path / "evalshift.yaml").read_text(encoding="utf-8")


def _put_team_support(base: Path, cid: str) -> None:
    """A capture of suite ``team/support``, in the directory the SDK sanitises it to."""
    sidecar = base / "toolsets" / f"{TOOLSET_HEX}.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(TOOLSET_PAYLOAD)
    path = base / "captures" / "team_support" / f"{cid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        capture_payload(
            cid,
            "team/support",
            model_input=f"question {cid}?",
            created_at=WEEK_1,
            model_id="claude-sonnet-5-20260301",
        )
    )


def test_a_suite_named_with_a_slash_is_planned_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(4):
        _put_team_support(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put_team_support(tmp_path, "cap_new")

    result = _sync(tmp_path)

    assert len(_cases(tmp_path, "team_support")) == 3
    flat = _flat(result)
    assert flat.count("cases — kept") == 1
    assert "team/support: 3 cases — kept 3" in flat
    config = (tmp_path / "evalshift.yaml").read_text(encoding="utf-8")
    assert "team_support:" not in config
    assert config.count("team/support") >= 1


def test_a_suite_named_with_a_slash_keeps_its_name_without_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put_team_support(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    for path in (tmp_path / "captures" / "team_support").glob("*.json"):
        path.unlink()
    _config(tmp_path / "evalshift.yaml", max_cases=3)

    result = _sync(tmp_path, "--suite", "team/support")

    assert len(_cases(tmp_path, "team_support")) == 3
    assert "team/support: 3 cases — kept 3, retired 2 (2 over max_cases)" in _flat(result)
    assert "team_support:" not in (tmp_path / "evalshift.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize("old", ["a_old", "z_old"])
def test_force_replaces_a_stale_twin_whatever_the_capture_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: str
) -> None:
    _put(tmp_path, old, text="where is order 1?")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "b_new", text="where is order 1?", created_at=WEEK_3)
    _at(monkeypatch, 30)

    _sync(tmp_path, "--force")

    assert _cases(tmp_path) == {"b_new"}


def test_help_describes_bounded_suites() -> None:
    result = runner.invoke(app, ["capture", "sync", "--help"])
    flat = " ".join(result.stdout.split())
    assert "Promote every capture" not in flat
    assert "captures.max_age" in flat
    assert "captures.max_cases" in flat


def _mark_reviewed(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["example"]["expected_tools"] = [{"tool_name": "lookup", "provenance": "reviewed"}]
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_copied_case_file_never_redirects_a_retirement_to_the_reviewed_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_cases=4)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    suite = tmp_path / "suites" / "alpha"
    original = suite / "cap_0.json"
    # A copy whose question was edited but whose "name" still says cap_0.
    variant = json.loads(original.read_text(encoding="utf-8"))
    variant["example"]["id"] = "variant"
    variant["example"]["inputs"] = {"input": "a variant question?"}
    (suite / "variant.json").write_text(json.dumps(variant), encoding="utf-8")
    _mark_reviewed(original)
    _config(tmp_path / "evalshift.yaml", max_cases=3)

    result = _sync(tmp_path)

    assert original.exists()
    assert "reviewed" in original.read_text(encoding="utf-8")
    assert len(_cases(tmp_path)) == 3
    assert "alpha: 3 cases — kept 3, retired 1 (1 over max_cases), added 0 of 0 eligible" in (
        _flat(result)
    )


def test_a_renamed_case_file_is_retired_from_where_it_lives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    suite = tmp_path / "suites" / "alpha"
    (suite / "cap_0.json").rename(suite / "refund_case.json")
    _config(tmp_path / "evalshift.yaml", max_cases=2)

    first = _sync(tmp_path)
    second = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 2
    assert "alpha: 2 cases — kept 2, retired 1 (1 over max_cases)" in _flat(first)
    assert "alpha: 2 cases — kept 2, retired 0," in _flat(second)


def test_a_retirement_that_deletes_nothing_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _config(tmp_path / "evalshift.yaml", max_cases=2)
    monkeypatch.setattr(capture_module, "retire_promoted_case", lambda *a, **k: False)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    flat = _flat(result)
    assert "alpha: 3 cases — kept 3, retired 0, added 0 of 0 eligible" in flat
    assert "alpha: could not delete cap_0.json; not counted as retired." in flat


def test_a_capture_whose_case_name_is_taken_neither_adds_nor_retires_its_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(tmp_path, "old", text="where is order 1?")
    _put(tmp_path, "other", text="something else?")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    suite = tmp_path / "suites" / "alpha"
    # `other`'s case now lives under the name a later capture is given.
    moved = json.loads((suite / "other.json").read_text(encoding="utf-8"))
    moved["name"] = "new"
    (suite / "other.json").unlink()
    (suite / "new.json").write_text(json.dumps(moved), encoding="utf-8")
    _put(tmp_path, "new", text="where is order 1?", created_at=WEEK_3)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"old", "new"}
    assert json.loads((suite / "new.json").read_text(encoding="utf-8"))["from_capture"] == "other"
    flat = _flat(result)
    assert "alpha: 2 cases — kept 2, retired 0, added 0 of 0 eligible" in flat
    assert "skipped 3 already-promoted" in flat  # old, other, and new (its name is taken)


def test_a_write_that_fails_neither_counts_as_added_nor_retires_its_twin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(tmp_path, "old", text="where is order 1?")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new", text="where is order 1?", created_at=WEEK_3)
    _at(monkeypatch, 30)
    real_write = capture_module.write_promoted_case

    def write(case: Any, **kwargs: Any) -> Path:
        if case.name == "new":  # another process wrote it first
            raise FileExistsError(case.name)
        return real_write(case, **kwargs)

    monkeypatch.setattr(capture_module, "write_promoted_case", write)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"old"}
    flat = _flat(result)
    assert "alpha: 1 cases — kept 1, retired 0, added 0 of 1 eligible" in flat
    assert "promoted 0 capture(s)" in flat


def test_a_renamed_legacy_case_is_backfilled_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(2):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_cases=5)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    suite = tmp_path / "suites" / "alpha"
    legacy = json.loads((suite / "cap_0.json").read_text(encoding="utf-8"))
    legacy.pop("captured_at")
    legacy.pop("source_model")
    (suite / "cap_0.json").unlink()
    (suite / "refund_case.json").write_text(json.dumps(legacy), encoding="utf-8")

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"refund_case", "cap_1"}
    assert json.loads((suite / "refund_case.json").read_text(encoding="utf-8"))["captured_at"]
    assert "backfilled capture time and model on 1 case(s)" in _flat(result)


def test_a_duplicate_kept_beside_a_replaceable_case_still_blocks_a_fresh_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two cases share their content (kept with --keep-duplicates): a_old is replaceable,
    # b_old is reviewed. A fresh copy would duplicate b_old, which never leaves.
    _put(tmp_path, "a_old", text="where is order 1?")
    _put(tmp_path, "b_old", text="where is order 1?")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path, "--keep-duplicates")
    _mark_reviewed(tmp_path / "suites" / "alpha" / "b_old.json")
    _put(tmp_path, "c_new", text="where is order 1?", created_at=WEEK_3)
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert "c_new" not in _cases(tmp_path)
    assert "skipped 1 duplicate capture(s)" in _flat(result)


def _wired(tmp_path: Path) -> set[str]:
    text = (tmp_path / "evalshift.yaml").read_text(encoding="utf-8")
    return set(parse_suites_region(text))


def test_a_suite_dir_without_captures_is_not_wired_back_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"a_{i}", suite="alpha")
        _put(tmp_path, f"o_{i}", suite="old")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    # The team drops `old`: its captures and its suites entry go, its directory stays.
    for path in (tmp_path / "captures" / "old").glob("*.json"):
        path.unlink()
    _config(tmp_path / "evalshift.yaml", max_cases=2)

    _sync(tmp_path)

    assert _wired(tmp_path) == {"alpha"}
    assert len(_cases(tmp_path, "old")) == 2
    assert len(load_jsonl(tmp_path / "suites" / "old" / "golden.jsonl").examples) == 2


def test_a_wired_suite_without_captures_stays_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"a_{i}", suite="alpha")
        _put(tmp_path, f"o_{i}", suite="old")
    config = _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    for path in (tmp_path / "captures" / "old").glob("*.json"):
        path.unlink()
    config.write_text(
        config.read_text(encoding="utf-8") + "\ncaptures:\n  max_cases: 2\n", encoding="utf-8"
    )

    _sync(tmp_path)

    assert _wired(tmp_path) == {"alpha", "old"}
    assert len(_cases(tmp_path, "old")) == 2
