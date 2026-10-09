"""``capture sync --repair`` and the warning that points at it.

The broken cases here are what CLI 1.3.0 promoted from a Claude follow-up turn: every block
list went through ``str()``. The store fixture plays S3; ``--base`` plays a fresh clone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.remote import RemoteStore
from evalshift_cli.captures.store_uri import StoreURI
from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import render_minimal_config
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
        render_minimal_config(profile="model-upgrade") + "\ncaptures:\n  store: s3://b/p\n",
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


def _write_local_capture(base: Path, capture_id: str) -> None:
    captures = base / "captures" / "alpha"
    captures.mkdir(parents=True, exist_ok=True)
    (captures / f"{capture_id}.json").write_bytes(
        capture_payload(capture_id, "alpha", model_input=CLAUDE_INPUT)
    )
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
    case = PromotedCase(
        name=case_name,
        suite=suite,
        from_capture=capture_id,
        example=example,
        conversation_id=conversation_id,
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


@pytest.mark.parametrize("flags", [("--offline",), ("--offline", "--repair")])
def test_a_duplicate_of_a_candidate_is_still_skipped(
    tmp_path: Path, store: FakeRemoteStore, flags: tuple[str, ...]
) -> None:
    """CLI 1.3.0 skipped cap_2 as a duplicate of cap_1; rebuilt, it still is one."""
    _write_case(tmp_path, "cap_1")
    _write_local_capture(tmp_path, "cap_1")
    _write_local_capture(tmp_path, "cap_2")
    result = _sync(tmp_path, *flags)
    assert result.exit_code == 0, result.output
    assert "skipped 1 duplicate capture(s)" in _out(result)
    assert not (tmp_path / "suites" / "alpha" / "cap_2.json").exists()


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
