"""``capture sync --repair`` and the warning that points at it.

The broken cases here are what CLI 1.3.0 promoted from a Claude follow-up turn: every block
list went through ``str()``. The store fixture plays S3; ``--base`` plays a fresh clone.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.remote import RemoteStore
from evalshift_cli.captures.store_uri import StoreURI
from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config
from evalshift_cli.cli.main import app
from evalshift_cli.suite.models import SuiteExample
from tests.unit.fake_remote_store import (
    TOOLSET_HEX,
    TOOLSET_KEY,
    TOOLSET_PAYLOAD,
    TOOLSET_REF,
    FakeRemoteStore,
    capture_payload,
)

runner = CliRunner()

CLAUDE_INPUT: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "evalshift"}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "results"}],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "A regression tool."}]},
    {"role": "user", "content": "And the price?"},
]
BROKEN_HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {"role": "assistant", "content": str(CLAUDE_INPUT[2]["content"])},
    {"role": "user", "content": str(CLAUDE_INPUT[3]["content"])},
    {"role": "assistant", "content": str(CLAUDE_INPUT[4]["content"])},
]
#: What ``CLAUDE_INPUT`` promotes to today: the history a repair rebuilds ``BROKEN_HISTORY`` into.
REBUILT_HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {
        "role": "assistant",
        "content": "Let me look.",
        "tool_calls": [{"id": "toolu_1", "name": "search", "arguments": {"q": "evalshift"}}],
    },
    {"role": "tool", "content": "results", "tool_call_id": "toolu_1"},
    {"role": "assistant", "content": "A regression tool."},
]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeRemoteStore:
    fake = FakeRemoteStore({TOOLSET_KEY: TOOLSET_PAYLOAD})

    def _open(parsed: StoreURI) -> RemoteStore:
        return fake

    monkeypatch.setattr(capture_module, "open_store", _open)
    return fake


def _config(base: Path) -> Path:
    path = base / "evalshift.yaml"
    path.write_text(
        render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
        + "\ncaptures:\n  store: s3://b/p\n",
        encoding="utf-8",
    )
    return path


def _sync(base: Path, *flags: str) -> Any:
    return runner.invoke(
        app,
        ["capture", "sync", *flags, "--config", str(_config(base)), "--base", str(base)],
    )


def _out(result: Any) -> str:
    return " ".join(result.output.split())


def _put(
    store: FakeRemoteStore,
    capture_id: str,
    *,
    model_input: Any = None,
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> None:
    store.objects[f"captures/alpha/{capture_id}.json"] = capture_payload(
        capture_id,
        "alpha",
        model_input=CLAUDE_INPUT if model_input is None else model_input,
        conversation_id=conversation_id,
        turn_index=turn_index,
    )


def _write_local_capture(base: Path, capture_id: str, *, toolset: bool = True) -> None:
    captures = base / "captures" / "alpha"
    captures.mkdir(parents=True, exist_ok=True)
    (captures / f"{capture_id}.json").write_bytes(
        capture_payload(capture_id, "alpha", model_input=CLAUDE_INPUT)
    )
    if not toolset:
        return
    toolsets = base / "toolsets"
    toolsets.mkdir(parents=True, exist_ok=True)
    (toolsets / f"{TOOLSET_HEX}.json").write_bytes(TOOLSET_PAYLOAD)


def _write_case(
    base: Path,
    capture_id: str,
    *,
    name: str | None = None,
    suite: str = "alpha",
    inputs: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    tags: list[str] | None = None,
    expected_tools: list[dict[str, Any]] | None = None,
    conversation_id: str | None = None,
) -> Path:
    case_name = name or capture_id
    example = SuiteExample.model_validate(
        {
            "id": case_name,
            "inputs": {"input": "And the price?"} if inputs is None else inputs,
            "tags": tags or ["captured"],
            "history": BROKEN_HISTORY if history is None else history,
            "expected_tools": expected_tools,
            "toolset_ref": TOOLSET_REF,
            "conversation_id": conversation_id,
        }
    )
    # Provenance matches what ``capture_payload`` records, so sync's backfill leaves the
    # file alone and the byte-for-byte "untouched" checks below measure repair only.
    case = PromotedCase(
        name=case_name,
        suite=suite,
        from_capture=capture_id,
        example=example,
        conversation_id=conversation_id,
        captured_at="2026-06-16T12:00:00+00:00",
        source_model="m",
    )
    path = base / "suites" / suite / f"{case_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(case.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _stored(path: Path) -> PromotedCase:
    return PromotedCase.model_validate_json(path.read_text(encoding="utf-8"))


# --- without --repair ------------------------------------------------------------------------


def test_a_fresh_clone_sync_warns_without_fetching_the_broken_cases(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _put(store, "cap_1")
    result = _sync(tmp_path)
    assert result.exit_code == 0, result.output
    assert (
        "alpha: 1 promoted case(s) carry message blocks an earlier sync stringified "
        "(e.g. cap_1); run evalshift capture sync --repair to rebuild them."
    ) in _out(result)
    assert "captures/alpha/cap_1.json" not in store.gets
    assert case.read_bytes() == before


@pytest.mark.parametrize("flags", [(), ("--force",)])
def test_sync_leaves_candidates_for_repair_even_with_force(
    tmp_path: Path, store: FakeRemoteStore, flags: tuple[str, ...]
) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _write_local_capture(tmp_path, "cap_1")
    result = _sync(tmp_path, "--offline", *flags)
    assert result.exit_code == 0, result.output
    assert "left 1 case(s) for --repair" in _out(result)
    assert case.read_bytes() == before


# --- --repair ---------------------------------------------------------------------------------


def test_repair_from_a_fresh_clone_refetches_and_rebuilds_only_the_broken_case(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    broken = _write_case(
        tmp_path,
        "cap_1",
        tags=["captured", "billing"],
        expected_tools=[{"tool_name": "search", "arguments": {"q": "evalshift"}}],
    )
    healthy = _write_case(tmp_path, "cap_2", inputs={"input": "hand edited"}, history=[])
    healthy_before = healthy.read_bytes()
    _put(store, "cap_1")
    _put(store, "cap_2", model_input="where is order cap_2?")

    result = _sync(tmp_path, "--repair")

    assert result.exit_code == 0, result.output
    assert "repaired 1 case(s)" in _out(result)
    assert "captures/alpha/cap_1.json" in store.gets
    assert "captures/alpha/cap_2.json" not in store.gets
    fixed = _stored(broken).example
    assert fixed.history is not None
    assert [m.role for m in fixed.history] == ["system", "user", "assistant", "tool", "assistant"]
    assert fixed.history[2].tool_calls is not None
    assert fixed.history[2].tool_calls[0].id == "toolu_1"
    assert fixed.history[3].tool_call_id == "toolu_1"
    assert fixed.tags == ["captured", "billing"]
    assert fixed.expected_tools is not None
    assert fixed.expected_tools[0].tool_name == "search"
    assert healthy.read_bytes() == healthy_before
    golden = (tmp_path / "suites" / "alpha" / "golden.jsonl").read_text(encoding="utf-8")
    assert "'type'" not in golden


def test_repair_keeps_a_custom_name_and_writes_no_second_case(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    _write_case(tmp_path, "cap_1", name="billing_q")
    _put(store, "cap_1")
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    suite_dir = tmp_path / "suites" / "alpha"
    assert sorted(p.name for p in suite_dir.iterdir()) == ["billing_q.json", "golden.jsonl"]
    fixed = _stored(suite_dir / "billing_q.json")
    assert fixed.example.id == "billing_q"
    assert fixed.example.history is not None
    assert fixed.example.history[3].role == "tool"


def test_repair_fetches_conversation_siblings_without_rewriting_them(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    sibling = _write_case(
        tmp_path,
        "cap_t0",
        inputs={"input": "What is evalshift?"},
        history=[],
        conversation_id="conv",
    )
    sibling_before = sibling.read_bytes()
    _write_case(tmp_path, "cap_t1", conversation_id="conv")
    _put(store, "cap_t0", model_input="What is evalshift?", conversation_id="conv", turn_index=0)
    _put(store, "cap_t1", conversation_id="conv", turn_index=1)

    result = _sync(tmp_path, "--repair")

    assert result.exit_code == 0, result.output
    assert {"captures/alpha/cap_t0.json", "captures/alpha/cap_t1.json"} <= set(store.gets)
    assert sibling.read_bytes() == sibling_before
    assert "repaired 1 case(s)" in _out(result)


def test_repair_names_cases_whose_capture_is_gone(tmp_path: Path, store: FakeRemoteStore) -> None:
    case = _write_case(tmp_path, "cap_gone")
    before = case.read_bytes()
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    assert (
        "can't repair 1 case(s) — their capture is no longer local or in the store "
        "(cap_gone): delete the case file or re-capture."
    ) in _out(result)
    assert case.read_bytes() == before


def test_repair_is_not_hidden_by_since(tmp_path: Path, store: FakeRemoteStore) -> None:
    _write_case(tmp_path, "cap_1")
    _put(store, "cap_1")
    store.modified["captures/alpha/cap_1.json"] = datetime(2020, 1, 1, tzinfo=UTC)
    result = _sync(tmp_path, "--repair", "--since", "1h")
    assert result.exit_code == 0, result.output
    assert "repaired 1 case(s)" in _out(result)


def test_repair_offline_uses_local_captures_only(tmp_path: Path, store: FakeRemoteStore) -> None:
    _write_case(tmp_path, "cap_1")
    _write_local_capture(tmp_path, "cap_1")
    result = _sync(tmp_path, "--repair", "--offline")
    assert result.exit_code == 0, result.output
    assert store.gets == []
    assert "repaired 1 case(s)" in _out(result)


def test_repair_twice_is_a_no_op(tmp_path: Path, store: FakeRemoteStore) -> None:
    case = _write_case(tmp_path, "cap_1")
    _put(store, "cap_1")
    assert _sync(tmp_path, "--repair").exit_code == 0
    after_first = case.read_bytes()
    second = _sync(tmp_path, "--repair")
    assert second.exit_code == 0, second.output
    assert "no promoted cases need repair." in _out(second)
    assert case.read_bytes() == after_first


def test_repair_keeps_the_capture_provenance_it_backfills(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    """A 1.3.0-era case has no provenance; repairing it must not drop what sync backfilled."""
    case = _write_case(tmp_path, "cap_1")
    legacy = json.loads(case.read_text(encoding="utf-8"))
    del legacy["captured_at"], legacy["source_model"]
    case.write_text(json.dumps(legacy), encoding="utf-8")
    _put(store, "cap_1")

    first = _sync(tmp_path, "--repair")

    assert first.exit_code == 0, first.output
    assert "repaired 1 case(s)" in _out(first)
    assert _stored(case).captured_at == "2026-06-16T12:00:00+00:00"
    assert _stored(case).source_model == "m"
    after_first = case.read_bytes()
    second = _sync(tmp_path)
    assert second.exit_code == 0, second.output
    assert "backfilled" not in _out(second)
    assert case.read_bytes() == after_first


def test_repair_leaves_a_case_whose_rebuild_is_blocked(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _put(store, "cap_1")
    del store.objects[TOOLSET_KEY]  # no sidecar anywhere: the rebuild is refused (no_toolset)
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    assert "can't repair cap_1:" in _out(result)
    assert case.read_bytes() == before


def test_repair_survives_a_corrupt_case_file_elsewhere(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    broken = _write_case(tmp_path, "cap_1")
    bad = tmp_path / "suites" / "beta" / "bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    _put(store, "cap_1")
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "to check for repairs: ValidationError" in out
    assert "repaired 1 case(s)" in out
    history = _stored(broken).example.history
    assert history is not None
    assert history[3].role == "tool"


def test_repair_reports_when_there_is_nothing_to_do(tmp_path: Path, store: FakeRemoteStore) -> None:
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    assert "no promoted cases need repair." in _out(result)


@pytest.mark.parametrize("duplicate", ["cap_a", "cap_c"])
@pytest.mark.parametrize("flags", [("--offline",), ("--offline", "--repair")])
def test_a_duplicate_of_a_candidate_is_skipped_wherever_it_sorts(
    tmp_path: Path, store: FakeRemoteStore, flags: tuple[str, ...], duplicate: str
) -> None:
    """CLI 1.3.0 skipped the duplicate of cap_b; rebuilt, it still is one -- even built first."""
    _write_case(tmp_path, "cap_b")
    _write_local_capture(tmp_path, "cap_b")
    _write_local_capture(tmp_path, duplicate)  # cap_a is built and dedup-checked before cap_b
    result = _sync(tmp_path, *flags)
    assert result.exit_code == 0, result.output
    assert "skipped 1 duplicate capture(s)" in _out(result)
    assert not (tmp_path / "suites" / "alpha" / f"{duplicate}.json").exists()


def test_a_no_op_rebuild_is_not_left_for_repair_and_force_overwrites_it(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    _put(store, "cap_1", model_input={})  # promotes with empty inputs: genuinely input-less
    assert _sync(tmp_path).exit_code == 0
    case = tmp_path / "suites" / "alpha" / "cap_1.json"
    first = _stored(case)
    plain = _sync(tmp_path, "--offline")
    assert plain.exit_code == 0, plain.output
    assert "case(s) for --repair" not in _out(plain)
    assert "skipped 1 already-promoted" in _out(plain)
    forced = _sync(tmp_path, "--offline", "--force")
    assert forced.exit_code == 0, forced.output
    assert "case(s) for --repair" not in _out(forced)
    assert "promoted 1 capture(s)" in _out(forced)
    assert _stored(case).promoted_at != first.promoted_at


def test_a_blocked_rebuild_of_a_stringified_case_is_left_for_repair(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _write_local_capture(tmp_path, "cap_1", toolset=False)
    result = _sync(tmp_path, "--offline")
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "left 1 case(s) for --repair" in out
    assert "skipped cap_1" not in out
    assert case.read_bytes() == before


@pytest.mark.parametrize("flags", [("--offline",), ("--offline", "--repair")])
def test_a_blocked_rebuild_of_an_input_less_case_is_an_ordinary_skip(
    tmp_path: Path, store: FakeRemoteStore, flags: tuple[str, ...]
) -> None:
    case = _write_case(tmp_path, "cap_1", inputs={}, history=[])
    before = case.read_bytes()
    _write_local_capture(tmp_path, "cap_1", toolset=False)
    result = _sync(tmp_path, *flags)
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "skipped 1 capture(s) with no usable recorded toolset" in out
    assert "case(s) for --repair" not in out
    assert "can't repair" not in out
    assert case.read_bytes() == before


def test_repair_names_a_duplicate_it_leaves_behind(tmp_path: Path, store: FakeRemoteStore) -> None:
    """A fresh-clone plain sync promoted cap_2 before --repair turned cap_1 into its twin."""
    broken = _write_case(tmp_path, "cap_1")
    twin = _write_case(tmp_path, "cap_2", history=REBUILT_HISTORY)
    _put(store, "cap_1")
    _put(store, "cap_2")
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "repaired 1 case(s)" in out
    assert (
        "alpha: repaired case cap_1 duplicates cap_2 — delete one (duplicate examples inflate n)"
    ) in out
    assert out.count("duplicates cap_2") == 1
    assert broken.is_file()
    assert twin.is_file()


def test_repair_rewrites_nothing_when_the_rebuild_is_identical(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    _put(store, "cap_1", model_input=[])  # promotes with empty inputs: genuinely input-less
    assert _sync(tmp_path).exit_code == 0
    case = tmp_path / "suites" / "alpha" / "cap_1.json"
    before = case.read_bytes()
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "cap_1: could not recover structured inputs" in out
    assert "repaired 0 case(s)" in out
    assert "can't repair" not in out
    assert case.read_bytes() == before


def test_repair_is_scoped_to_the_suite_option(tmp_path: Path, store: FakeRemoteStore) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _put(store, "cap_1")
    result = _sync(tmp_path, "--repair", "--suite", "beta")
    assert result.exit_code == 0, result.output
    assert "no promoted cases need repair." in _out(result)
    assert "captures/alpha/cap_1.json" not in store.gets
    assert case.read_bytes() == before


def test_repair_shows_the_rebuild_warnings(tmp_path: Path, store: FakeRemoteStore) -> None:
    _write_case(tmp_path, "cap_1")
    legacy = {"role": "function", "content": "legacy"}  # never aliased: dropped with a warning
    _put(store, "cap_1", model_input=[CLAUDE_INPUT[0], legacy, *CLAUDE_INPUT[1:]])
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "cap_1: dropped 1 history message(s) with unrecognised role(s)" in out
    assert "repaired 1 case(s)" in out


def test_repair_warns_once_for_a_pair_it_repaired_both_of(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    first = _write_case(tmp_path, "cap_1")  # promoted together under --keep-duplicates
    second = _write_case(tmp_path, "cap_2")
    _put(store, "cap_1")
    _put(store, "cap_2")
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "repaired 2 case(s)" in out
    assert out.count("— delete one (duplicate examples inflate n)") == 1
    assert "alpha: repaired case cap_1 duplicates cap_2" in out
    assert first.is_file()
    assert second.is_file()


def test_repair_keeps_a_custom_input_key(tmp_path: Path, store: FakeRemoteStore) -> None:
    case = _write_case(tmp_path, "cap_1", inputs={"question": "And the price?"})
    _put(store, "cap_1")
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    assert "repaired 1 case(s)" in _out(result)
    fixed = _stored(case).example
    assert fixed.inputs == {"question": "And the price?"}
    assert fixed.history is not None
    assert [m.role for m in fixed.history] == ["system", "user", "assistant", "tool", "assistant"]


def test_repair_with_force_still_only_rebuilds_a_candidates_inputs_and_history(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    case = _write_case(
        tmp_path,
        "cap_1",
        tags=["captured", "billing"],
        expected_tools=[{"tool_name": "search", "arguments": {"q": "evalshift"}}],
    )
    _put(store, "cap_1")
    result = _sync(tmp_path, "--repair", "--force")
    assert result.exit_code == 0, result.output
    assert "repaired 1 case(s)" in _out(result)
    fixed = _stored(case).example
    assert fixed.tags == ["captured", "billing"]
    assert fixed.expected_tools is not None
    assert [(t.tool_name, t.arguments) for t in fixed.expected_tools] == [
        ("search", {"q": "evalshift"})
    ]
    assert fixed.history is not None
    assert fixed.history[3].role == "tool"


@pytest.mark.parametrize("flags", [(), ("--repair",)])
def test_a_fresh_clone_sync_skips_the_duplicate_of_a_stringified_case(
    tmp_path: Path, store: FakeRemoteStore, flags: tuple[str, ...]
) -> None:
    """cap_dup was de-duplicated against cap_1 when CLI 1.3.0 promoted it, so it never was.

    A plain sync never fetches cap_1 (it is promoted), so only the offline reconstruction of
    cap_1's stringified case can tell that cap_dup's correct rebuild is the same turn.
    """
    _write_case(tmp_path, "cap_1")
    _put(store, "cap_1")
    _put(store, "cap_dup")
    result = _sync(tmp_path, *flags)
    assert result.exit_code == 0, result.output
    assert "skipped 1 duplicate capture(s)" in _out(result)
    if not flags:
        assert "captures/alpha/cap_1.json" not in store.gets
    suite_dir = tmp_path / "suites" / "alpha"
    assert sorted(p.name for p in suite_dir.iterdir()) == ["cap_1.json", "golden.jsonl"]


@pytest.mark.parametrize(
    "t1_input",
    [{}, [{"role": "system", "content": "sys"}]],
    ids=["no-messages-list", "messages-list-without-a-user-turn"],
)
def test_repair_keeps_a_reconstructed_history_whose_sibling_turn_is_gone(
    tmp_path: Path, store: FakeRemoteStore, t1_input: Any
) -> None:
    """t1's own capture yields no history, so it came from t0 -- whose capture expired.

    Either t1 recorded no messages list at all, or one with no user turn to split on: both
    leave promotion to reconstruct t1's history from its sibling turns.
    """
    _put(store, "cap_t0", model_input="What is evalshift?", conversation_id="conv", turn_index=0)
    _put(store, "cap_t1", model_input=t1_input, conversation_id="conv", turn_index=1)
    first = _sync(tmp_path)
    assert first.exit_code == 0, first.output
    t1 = tmp_path / "suites" / "alpha" / "cap_t1.json"
    assert _stored(t1).example.history  # reconstructed from t0
    before = t1.read_bytes()
    del store.objects["captures/alpha/cap_t0.json"]  # the store's lifecycle rule expired it
    shutil.rmtree(tmp_path / "captures")  # a fresh clone

    result = _sync(tmp_path, "--repair")

    assert result.exit_code == 0, result.output
    out = _out(result)
    assert (
        "can't repair cap_t1: conversation turn(s) cap_t0 are no longer local or in the store, "
        "so its history can't be rebuilt"
    ) in out
    assert "repaired 0 case(s)" in out
    assert t1.read_bytes() == before


def test_repair_rebuilds_a_turn_that_recorded_its_messages_even_with_a_sibling_gone(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    _write_case(
        tmp_path,
        "cap_t0",
        inputs={"input": "What is evalshift?"},
        history=[],
        conversation_id="conv",
    )
    broken = _write_case(tmp_path, "cap_t1", conversation_id="conv")
    _put(store, "cap_t1", conversation_id="conv", turn_index=1)  # cap_t0's capture is gone

    result = _sync(tmp_path, "--repair")

    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "repaired 1 case(s)" in out
    assert "can't repair cap_t1" not in out
    history = _stored(broken).example.history
    assert history is not None
    assert [m.role for m in history] == ["system", "user", "assistant", "tool", "assistant"]
