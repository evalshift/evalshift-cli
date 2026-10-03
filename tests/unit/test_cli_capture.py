"""Tests for the ``evalshift capture`` command group."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.captures.toolset import EMPTY_TOOLSET_FINGERPRINT, fingerprint_tools
from evalshift_cli.cli.commands.capture import _build_suite_entries, _declared_tool_properties
from evalshift_cli.cli.commands.init import render_minimal_config
from evalshift_cli.cli.main import app
from evalshift_cli.config.loader import load_config
from evalshift_cli.suite.loader import load_jsonl

runner = CliRunner()


def _write_min_config(path: Path) -> None:
    """Write a minimal init-style config (with the managed suites markers)."""
    path.write_text(render_minimal_config(profile="model-upgrade"), encoding="utf-8")


def _write_empty_capture(base: Path, *, capture_id: str, suite: str) -> Path:
    """Write a capture whose trace has no events (unpromotable)."""
    payload = {
        "schema_version": "2.0.0",
        "capture_id": capture_id,
        "suite": suite,
        "input_hash": "h",
        "code_version": "",
        "created_at": "2026-06-16T12:00:00+00:00",
        "trace": {
            "run_id": capture_id,
            "prompt_id": suite,
            "example_id": capture_id,
            "role": "source",
            "events": [],
        },
    }
    suite_dir = base / "captures" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / f"{capture_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _events(
    *, tools: list[str], model_input: Any = "where is order 12345?"
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "model_call",
            "sequence_index": 0,
            "timestamp": "2026-06-16T12:00:00+00:00",
            "metadata": {},
            "model_id": "m",
            "input": model_input,
            "output": "out",
            "toolset_ref": "sha256:" + "ab" * 32,
            "tools_offered": list(tools),
        },
    ]
    idx = 1
    for name in tools:
        events.append(
            {
                "type": "tool_call",
                "sequence_index": idx,
                "timestamp": "2026-06-16T12:00:02+00:00",
                "metadata": {},
                "name": name,
                "arguments": {"customer_id": "c42"},
                "call_id": f"call_{idx}",
            },
        )
        idx += 1
    events.append(
        {
            "type": "final_output",
            "sequence_index": idx,
            "timestamp": "2026-06-16T12:00:03+00:00",
            "metadata": {},
            "text": "done",
        },
    )
    return events


_TOOLSET_REF = "sha256:" + "ab" * 32


def _write_toolset_sidecar(base: Path, *, ref: str = _TOOLSET_REF) -> None:
    """Ensure ``ref``'s sidecar exists under ``base`` so promotion doesn't refuse it.

    ``build_example_from_capture`` now checks the sidecar exists (not just that
    ``toolset_ref`` is set) before promoting -- see ``captures/promote.py``.
    Every capture this file writes with a real ``toolset_ref`` uses the one
    shared placeholder value, so one idempotent sidecar write per ``base``
    covers every test; content is a minimal, valid, empty toolset since no
    test here asserts on resolved tool bodies.
    """
    toolsets_dir = base / "toolsets"
    toolsets_dir.mkdir(parents=True, exist_ok=True)
    (toolsets_dir / f"{ref.removeprefix('sha256:')}.json").write_text(
        '{"tools": []}', encoding="utf-8"
    )


def _write_capture(
    base: Path,
    *,
    capture_id: str,
    suite: str = "support_agent",
    tools: list[str] | None = None,
    model_input: Any = "where is order 12345?",
    events: list[dict[str, Any]] | None = None,
    created_at: str = "2026-06-16T12:00:00+00:00",
    conversation_id: str | None = None,
    turn_index: int | None = None,
    input_hash: str | None = None,
) -> Path:
    _write_toolset_sidecar(base)
    payload = {
        "schema_version": "2.0.0",
        "capture_id": capture_id,
        "suite": suite,
        # Unique per capture by default — pass an explicit input_hash to
        # simulate re-capturing the same input (sync dedupes on it).
        "input_hash": input_hash or f"hash-{capture_id}",
        "code_version": "v1",
        "created_at": created_at,
        "trace": {
            "run_id": capture_id,
            "prompt_id": suite,
            "example_id": capture_id,
            "role": "source",
            "events": events
            if events is not None
            else _events(
                tools=tools if tools is not None else ["search_orders"], model_input=model_input
            ),
        },
        "conversation_id": conversation_id,
        "turn_index": turn_index,
    }
    suite_dir = base / "captures" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / f"{capture_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _messages_events(messages: list[dict[str, Any]], *, final_text: str = "done") -> Any:
    return [
        {
            "type": "model_call",
            "sequence_index": 0,
            "timestamp": "2026-06-16T12:00:00+00:00",
            "metadata": {},
            "model_id": "m",
            "input": messages,
            "output": "out",
            "toolset_ref": "sha256:" + "ab" * 32,
            "tools_offered": [],
        },
        {
            "type": "final_output",
            "sequence_index": 1,
            "timestamp": "2026-06-16T12:00:03+00:00",
            "metadata": {},
            "text": final_text,
        },
    ]


def _invoke(args: list[str], base: Path) -> Any:
    return runner.invoke(app, ["capture", *args, "--base", str(base)])


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_empty(tmp_path: Path) -> None:
    result = _invoke(["list"], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "no captures" in result.stdout.lower()


def test_list_shows_captures(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="support_agent")
    _write_capture(tmp_path, capture_id="cap_2", suite="other")

    result = _invoke(["list"], tmp_path)

    assert result.exit_code == 0, result.stdout
    # capture_id is the no-wrap column, so both rows survive terminal width.
    # Exact field values (suite, hashes) are covered by the --json test.
    assert "cap_1" in result.stdout
    assert "cap_2" in result.stdout
    assert "captures" in result.stdout


def test_list_json(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", tools=["search_orders", "send_email"])

    result = _invoke(["list", "--json"], tmp_path)

    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert rows[0]["capture_id"] == "cap_1"
    assert rows[0]["n_tools"] == 2
    assert rows[0]["promoted"] is False


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


def test_promote_writes_case_and_golden(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="support_agent")

    result = _invoke(
        ["promote", "cap_1", "--as", "case1", "--input-var", "query"],
        tmp_path,
    )

    assert result.exit_code == 0, result.stdout
    case_path = tmp_path / "suites" / "support_agent" / "case1.json"
    assert case_path.exists()
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    assert case.from_capture == "cap_1"
    assert case.example.inputs == {"query": "where is order 12345?"}

    golden = tmp_path / "suites" / "support_agent" / "golden.jsonl"
    suite = load_jsonl(golden)
    assert suite.ids() == {"case1"}
    # The promote output points the user at the run bridge.
    assert "--suite-name" in result.stdout


def test_promote_missing_capture(tmp_path: Path) -> None:
    result = _invoke(["promote", "cap_nope", "--as", "x"], tmp_path)
    assert result.exit_code == 1
    assert "cap_nope" in result.stdout


def test_promote_refuses_overwrite_without_force(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1")
    assert _invoke(["promote", "cap_1", "--as", "case1"], tmp_path).exit_code == 0

    again = _invoke(["promote", "cap_1", "--as", "case1"], tmp_path)
    assert again.exit_code == 1
    assert "force" in again.stdout.lower()

    forced = _invoke(["promote", "cap_1", "--as", "case1", "--force"], tmp_path)
    assert forced.exit_code == 0, forced.stdout


def test_promote_warns_on_unrecoverable_inputs(tmp_path: Path) -> None:
    # An opaque (list) model input can't map to template variables.
    _write_capture(tmp_path, capture_id="cap_1", model_input=["opaque", "payload"])

    result = _invoke(["promote", "cap_1", "--as", "case1"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "inputs" in result.stdout.lower()


# ---------------------------------------------------------------------------
# tool-argument unwrapping -- ``_declared_tool_properties`` sources the
# declared schema from the capture's OWN recorded toolset, not a
# config-level ``tools_path`` (deleted).
# ---------------------------------------------------------------------------

_ARCHIVE_PROJECT_TOOL: dict[str, Any] = {
    "name": "archive_project",
    "description": "Archive a project.",
    "input_schema": {
        "type": "object",
        "properties": {"project_name": {"type": "string"}},
        "required": ["project_name"],
    },
}
# `load_toolset` now verifies a sidecar's `tools` against the ref that named it (the
# hardening pass) -- so this must be `_ARCHIVE_PROJECT_TOOL`'s own real fingerprint, not an
# arbitrary placeholder, or `_declared_tool_properties` (which every test below exercises
# via `capture promote`/`sync`) silently fails to resolve it and the unwrap heuristic these
# tests are actually about never fires.
_ARCHIVE_TOOL_REF = fingerprint_tools([_ARCHIVE_PROJECT_TOOL])


def _write_tool_sidecar(base: Path, *, ref: str, tools: list[dict[str, Any]]) -> None:
    """Write a real toolset sidecar declaring ``tools``' schemas at ``ref``."""
    toolsets_dir = base / "toolsets"
    toolsets_dir.mkdir(parents=True, exist_ok=True)
    (toolsets_dir / f"{ref.removeprefix('sha256:')}.json").write_text(
        json.dumps({"tools": tools}), encoding="utf-8"
    )


def _write_archive_tool_sidecar(base: Path, *, ref: str = _ARCHIVE_TOOL_REF) -> None:
    """A single-tool sidecar declaring ``archive_project(project_name)``."""
    _write_tool_sidecar(base, ref=ref, tools=[_ARCHIVE_PROJECT_TOOL])


def _wrapped_call_events(
    *,
    toolset_ref: str,
    wrapped_arguments: dict[str, Any],
    tool_name: str = "archive_project",
    model_input: str = "archive project apollo",
) -> list[dict[str, Any]]:
    """Events for a single tool call whose arguments were recorded wrapped in one key."""
    return [
        {
            "type": "model_call",
            "sequence_index": 0,
            "timestamp": "2026-06-16T12:00:00+00:00",
            "metadata": {},
            "model_id": "m",
            "input": model_input,
            "output": "done",
            "toolset_ref": toolset_ref,
            "tools_offered": [tool_name],
        },
        {
            "type": "tool_call",
            "sequence_index": 1,
            "timestamp": "2026-06-16T12:00:02+00:00",
            "metadata": {},
            "name": tool_name,
            "arguments": wrapped_arguments,
            "call_id": "call_1",
        },
        {
            "type": "final_output",
            "sequence_index": 2,
            "timestamp": "2026-06-16T12:00:03+00:00",
            "metadata": {},
            "text": "done",
        },
    ]


def test_promote_unwraps_arguments_using_the_captures_own_recorded_toolset(
    tmp_path: Path,
) -> None:
    """The declared schema comes from the capture's ``toolset_ref`` sidecar --
    never a config file -- so the promoter can recognise and undo a
    wrapper-key recording with no ``evalshift.yaml`` in sight.
    """
    _write_archive_tool_sidecar(tmp_path)
    events = _wrapped_call_events(
        toolset_ref=_ARCHIVE_TOOL_REF,
        wrapped_arguments={"tool_args": {"project_name": "apollo"}},
    )
    _write_capture(tmp_path, capture_id="cap_1", suite="ops", events=events)

    result = _invoke(["promote", "cap_1", "--as", "case1"], tmp_path)

    assert result.exit_code == 0, result.stdout
    case_path = tmp_path / "suites" / "ops" / "case1.json"
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    assert case.example.expected_tools is not None
    call = case.example.expected_tools[0]
    assert call.tool_name == "archive_project"
    assert call.arguments == {"project_name": "apollo"}
    assert "tool_args" in result.stdout  # the unwrap warning names the wrapper key


def test_promote_leaves_arguments_untouched_when_they_already_match_the_schema(
    tmp_path: Path,
) -> None:
    """A recording that already matches the declared schema is never rewritten."""
    _write_archive_tool_sidecar(tmp_path)
    events = _wrapped_call_events(
        toolset_ref=_ARCHIVE_TOOL_REF,
        wrapped_arguments={"project_name": "apollo"},
    )
    _write_capture(tmp_path, capture_id="cap_1", suite="ops", events=events)

    result = _invoke(["promote", "cap_1", "--as", "case1"], tmp_path)

    assert result.exit_code == 0, result.stdout
    case_path = tmp_path / "suites" / "ops" / "case1.json"
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    assert case.example.expected_tools is not None
    assert case.example.expected_tools[0].arguments == {"project_name": "apollo"}
    assert "tool_args" not in result.stdout


def test_promote_leaves_arguments_untouched_when_no_sidecar_resolves(tmp_path: Path) -> None:
    """No resolvable schema is evidence of nothing -- the recording is left as-is."""
    unresolvable_ref = "sha256:" + "99" * 32
    events = _wrapped_call_events(
        toolset_ref=unresolvable_ref,
        wrapped_arguments={"tool_args": {"project_name": "apollo"}},
    )
    _write_capture(tmp_path, capture_id="cap_1", suite="ops", events=events)

    # build_example_from_capture already blocks promotion outright when a
    # capture's own sidecar can't be resolved -- so exercise the properties
    # lookup at capture-sync's aggregate scope instead, where a *different*,
    # resolvable capture is promoted successfully and must not be affected
    # by the one with a dangling ref.
    _write_archive_tool_sidecar(tmp_path)
    resolvable_events = _wrapped_call_events(
        toolset_ref=_ARCHIVE_TOOL_REF,
        wrapped_arguments={"tool_args": {"project_name": "apollo"}},
    )
    _write_capture(tmp_path, capture_id="cap_2", suite="ops", events=resolvable_events)
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    golden = load_jsonl(tmp_path / "suites" / "ops" / "golden.jsonl")
    by_id = {ex.id: ex for ex in golden.examples}
    assert by_id["cap_2"].expected_tools is not None
    assert by_id["cap_2"].expected_tools[0].arguments == {"project_name": "apollo"}


def test_sync_unwraps_each_captures_arguments_against_its_own_toolset(tmp_path: Path) -> None:
    """Two captures synced together, each offering a DIFFERENT toolset: unwrapping
    must use each capture's own recorded schema and never conflate the two --
    the failure mode a single global (config-sourced) schema could not avoid.
    """
    send_email_tool: dict[str, Any] = {
        "name": "send_email",
        "description": "Send an email.",
        "input_schema": {"type": "object", "properties": {"to": {"type": "string"}}},
    }
    # Real fingerprint, not an arbitrary placeholder -- same reason as _ARCHIVE_TOOL_REF
    # above (load_toolset now verifies content against the ref that named it).
    other_ref = fingerprint_tools([send_email_tool])
    _write_archive_tool_sidecar(tmp_path)
    _write_tool_sidecar(
        tmp_path,
        ref=other_ref,
        tools=[send_email_tool],
    )
    _write_capture(
        tmp_path,
        capture_id="cap_archive",
        suite="ops",
        events=_wrapped_call_events(
            toolset_ref=_ARCHIVE_TOOL_REF,
            wrapped_arguments={"tool_args": {"project_name": "apollo"}},
        ),
    )
    _write_capture(
        tmp_path,
        capture_id="cap_email",
        suite="ops",
        events=_wrapped_call_events(
            toolset_ref=other_ref,
            wrapped_arguments={"payload": {"to": "a@example.com"}},
            tool_name="send_email",
            model_input="email a@example.com",
        ),
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    golden = load_jsonl(tmp_path / "suites" / "ops" / "golden.jsonl")
    by_id = {ex.id: ex for ex in golden.examples}
    assert by_id["cap_archive"].expected_tools is not None
    assert by_id["cap_archive"].expected_tools[0].arguments == {"project_name": "apollo"}
    assert by_id["cap_email"].expected_tools is not None
    assert by_id["cap_email"].expected_tools[0].arguments == {"to": "a@example.com"}


def _envelope_offering(*, capture_id: str, toolset_ref: str, tool_name: str) -> CaptureEnvelope:
    """A minimal envelope whose one model_call recorded ``tool_name`` under ``toolset_ref``."""
    payload = {
        "schema_version": "2.0.0",
        "capture_id": capture_id,
        "suite": "s",
        "input_hash": f"hash-{capture_id}",
        "code_version": "",
        "created_at": "2026-06-16T12:00:00+00:00",
        "trace": {
            "run_id": capture_id,
            "prompt_id": "s",
            "example_id": capture_id,
            "role": "source",
            "events": [
                {
                    "type": "model_call",
                    "sequence_index": 0,
                    "timestamp": "2026-06-16T12:00:00+00:00",
                    "metadata": {},
                    "model_id": "m",
                    "input": "hi",
                    "output": "out",
                    "toolset_ref": toolset_ref,
                    "tools_offered": [tool_name],
                },
            ],
        },
    }
    return CaptureEnvelope.model_validate(payload)


def test_declared_tool_properties_is_deterministic_on_a_cross_toolset_name_collision(
    tmp_path: Path,
) -> None:
    """M1: ``refs`` used to be iterated as a plain ``set`` -- non-deterministic
    last-write-wins whenever two toolsets in the same batch declare the same
    tool name with different argument shapes. That collision is exactly the
    path that writes ``expected_tools`` *arguments*, so two ``capture sync``
    runs over identical captures could write different golden data.

    8 colliding refs make an accidental sorted-order match by pure hash-seed
    luck astronomically unlikely (empirically 0/5 fresh-process trials with
    this exact shape), so this reds reliably pre-fix and greens once
    ``_declared_tool_properties`` iterates ``sorted(refs)``.

    Each ref below is the REAL fingerprint of its own distinct ``tools_i``
    (``load_toolset`` now verifies content against the ref that named it --
    the hardening pass -- so an arbitrary placeholder ref would make every
    one of these 8 sidecars fail to resolve, and this test would spuriously
    green with ``result is None`` regardless of iteration order).
    """
    tool_variants = [
        {
            "name": "act",
            "description": "d",
            "input_schema": {"type": "object", "properties": {f"prop_{i}": {"type": "string"}}},
        }
        for i in range(8)
    ]
    refs = [fingerprint_tools([tool]) for tool in tool_variants]
    for ref, tool in zip(refs, tool_variants, strict=True):
        _write_tool_sidecar(tmp_path, ref=ref, tools=[tool])
    envelopes = [
        _envelope_offering(capture_id=f"cap_{i}", toolset_ref=ref, tool_name="act")
        for i, ref in enumerate(refs)
    ]

    result = _declared_tool_properties(envelopes, base=tmp_path)

    assert result is not None
    winner_index = refs.index(sorted(refs)[-1])
    assert result["act"] == frozenset({f"prop_{winner_index}"})


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def test_clean_promoted_only_by_default(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_promoted")
    _write_capture(tmp_path, capture_id="cap_unpromoted")
    assert _invoke(["promote", "cap_promoted", "--as", "c1"], tmp_path).exit_code == 0

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "captures" / "support_agent" / "cap_promoted.json").exists()
    assert (tmp_path / "captures" / "support_agent" / "cap_unpromoted.json").exists()
    # suites are never touched by clean
    assert (tmp_path / "suites" / "support_agent" / "c1.json").exists()


def test_clean_all_removes_everything(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1")
    _write_capture(tmp_path, capture_id="cap_2")

    result = _invoke(["clean", "--all", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "captures" / "support_agent" / "cap_1.json").exists()
    assert not (tmp_path / "captures" / "support_agent" / "cap_2.json").exists()


def test_clean_nothing_to_do(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1")  # not promoted
    result = _invoke(["clean", "--yes"], tmp_path)
    assert result.exit_code == 0
    assert (tmp_path / "captures" / "support_agent" / "cap_1.json").exists()


# ---------------------------------------------------------------------------
# clean — confirmation must say what will actually be deleted (C2
# aggravating factor): the sweep can delete toolset sidecars too, not just
# capture files, so the prompt must not talk only about "N promoted
# captures". And the sweep must never run un-confirmed, even on the
# "nothing to clean" branch (no captures matched) -- it can still delete
# sidecars, and previously did so with zero confirmation at all.
# ---------------------------------------------------------------------------


def test_clean_confirmation_mentions_toolset_sidecars(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_promoted")
    assert _invoke(["promote", "cap_promoted", "--as", "c1"], tmp_path).exit_code == 0

    result = runner.invoke(app, ["capture", "clean", "--base", str(tmp_path)], input="n\n")

    assert "sidecar" in result.stdout.lower()


def test_clean_nothing_to_clean_still_confirms_before_sweeping_an_orphan(tmp_path: Path) -> None:
    """The exact regression: no captures matched (the early 'nothing to
    clean' branch), but a real orphaned sidecar still sits on disk. Declining
    must abort before it is swept -- previously this branch never asked at
    all.
    """
    _write_toolset_sidecar(tmp_path, ref=_ORPHAN_REF)  # referenced by nothing

    result = runner.invoke(app, ["capture", "clean", "--base", str(tmp_path)], input="n\n")

    assert result.exit_code != 0
    hex_name = _ORPHAN_REF.removeprefix("sha256:")
    assert (tmp_path / "toolsets" / f"{hex_name}.json").exists()


def test_clean_nothing_to_clean_and_nothing_to_sweep_never_prompts(tmp_path: Path) -> None:
    """No captures, no sidecars at all: truly a no-op, must not hang on a
    prompt (no ``input=`` given -- CliRunner would error if one were asked)."""
    result = runner.invoke(app, ["capture", "clean", "--base", str(tmp_path)])
    assert result.exit_code == 0, result.stdout


# ---------------------------------------------------------------------------
# clean — orphan toolset-sidecar sweep (V5)
#
# ``_write_capture`` (above) always points its captures at the one shared
# ``_TOOLSET_REF`` sidecar ``_write_toolset_sidecar`` writes. These tests use
# a second, dedicated ref so "still referenced" and "truly orphaned" can be
# told apart on purpose.
# ---------------------------------------------------------------------------

_ORPHAN_REF = "sha256:" + "cd" * 32


def test_clean_sweeps_a_sidecar_referenced_by_nothing(tmp_path: Path) -> None:
    """A sidecar no capture or promoted suite references is deleted and reported."""
    _write_toolset_sidecar(tmp_path, ref=_ORPHAN_REF)

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    hex_name = _ORPHAN_REF.removeprefix("sha256:")
    assert not (tmp_path / "toolsets" / f"{hex_name}.json").exists()
    # Rich soft-wraps the long hex filename across lines at terminal width;
    # collapse before searching so the wrap point doesn't break the assertion.
    assert hex_name in result.stdout.replace("\n", "")  # every deletion is reported


def test_clean_aborts_the_sweep_when_a_promoted_case_file_is_corrupt(tmp_path: Path) -> None:
    """C2 end-to-end: a corrupt promoted-case file must abort the sweep,
    naming the file, rather than silently sweeping every sidecar it (and its
    now-unreadable siblings) referenced."""
    _write_capture(tmp_path, capture_id="cap_1")
    assert _invoke(["promote", "cap_1", "--as", "c1"], tmp_path).exit_code == 0
    (tmp_path / "suites" / "support_agent" / "c1.json").write_text("not json", encoding="utf-8")

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code != 0
    assert "c1.json" in result.stdout
    # Nothing was swept: the shared sidecar the now-corrupt case named survives.
    hex_name = _TOOLSET_REF.removeprefix("sha256:")
    assert (tmp_path / "toolsets" / f"{hex_name}.json").exists()


def test_clean_default_promoted_scope_keeps_a_sidecar_a_promoted_suite_still_references(
    tmp_path: Path,
) -> None:
    """The exact destructive path V5 found.

    Promoting a capture, then cleaning it (the default, promoted-only
    scope), must never take the now-promoted suite's sidecar down with it --
    a naive sweep that refcounts only ``<base>/captures/`` sees the count
    drop to zero the moment the capture is deleted and would delete a
    sidecar the promoted ``golden.jsonl`` still references.
    """
    _write_capture(tmp_path, capture_id="cap_promoted", suite="support_agent")
    assert _invoke(["promote", "cap_promoted", "--as", "c1"], tmp_path).exit_code == 0

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    # The capture itself is gone (default promoted scope)...
    assert not (tmp_path / "captures" / "support_agent" / "cap_promoted.json").exists()
    # ...but its sidecar survives: the promoted suite's golden.jsonl still
    # names it, and every future `run` of that suite depends on it.
    hex_name = _TOOLSET_REF.removeprefix("sha256:")
    assert (tmp_path / "toolsets" / f"{hex_name}.json").exists()


def test_clean_all_scope_still_keeps_a_sidecar_a_promoted_suite_references(
    tmp_path: Path,
) -> None:
    """``--all`` empties ``<base>/captures/`` entirely -- the promoted suite must still protect it."""
    _write_capture(tmp_path, capture_id="cap_1")
    assert _invoke(["promote", "cap_1", "--as", "c1"], tmp_path).exit_code == 0

    result = _invoke(["clean", "--all", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "captures" / "support_agent" / "cap_1.json").exists()
    hex_name = _TOOLSET_REF.removeprefix("sha256:")
    assert (tmp_path / "toolsets" / f"{hex_name}.json").exists()


def test_clean_keeps_a_sidecar_referenced_by_a_surviving_unpromoted_capture(
    tmp_path: Path,
) -> None:
    """Default scope leaves an unpromoted capture on disk; its sidecar must survive too.

    Otherwise promoting it later fails with a missing sidecar `capture clean`
    deleted out from under it.
    """
    _write_capture(tmp_path, capture_id="cap_promoted", suite="support_agent")
    _write_capture(tmp_path, capture_id="cap_unpromoted", suite="support_agent")
    assert _invoke(["promote", "cap_promoted", "--as", "c1"], tmp_path).exit_code == 0

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "captures" / "support_agent" / "cap_unpromoted.json").exists()
    hex_name = _TOOLSET_REF.removeprefix("sha256:")
    assert (tmp_path / "toolsets" / f"{hex_name}.json").exists()


def test_clean_all_sweeps_a_sidecar_once_its_last_reference_is_gone(tmp_path: Path) -> None:
    """Positive control: with nothing promoted, ``--all`` truly orphans the sidecar and it goes."""
    _write_capture(tmp_path, capture_id="cap_1", suite="support_agent")  # never promoted

    result = _invoke(["clean", "--all", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    hex_name = _TOOLSET_REF.removeprefix("sha256:")
    assert not (tmp_path / "toolsets" / f"{hex_name}.json").exists()


def test_clean_sweep_ignores_the_suite_scope_argument(tmp_path: Path) -> None:
    """The toolsets namespace is global: scoping ``clean`` to one suite must not
    make the sweep blind to a sidecar a DIFFERENT suite's promoted case still uses.
    """
    _write_capture(tmp_path, capture_id="cap_1", suite="other_suite")
    assert _invoke(["promote", "cap_1", "--as", "c1"], tmp_path).exit_code == 0

    # Clean scoped to a suite with nothing in it -- no captures match, so the
    # early "nothing to clean" branch runs, but the sweep must still protect
    # other_suite's sidecar.
    result = _invoke(["clean", "support_agent", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    hex_name = _TOOLSET_REF.removeprefix("sha256:")
    assert (tmp_path / "toolsets" / f"{hex_name}.json").exists()


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def test_diff_reports_tool_difference(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_a", tools=["search_orders"])
    _write_capture(tmp_path, capture_id="cap_b", tools=["issue_refund"])

    result = _invoke(["diff", "cap_a", "cap_b"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "search_orders" in result.stdout or "issue_refund" in result.stdout


def test_diff_identical_captures(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_a", tools=["search_orders"])
    _write_capture(tmp_path, capture_id="cap_b", tools=["search_orders"])

    result = _invoke(["diff", "cap_a", "cap_b"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "no trace differences" in result.stdout.lower()


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_sync_promotes_all_and_wires_config(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    _write_capture(tmp_path, capture_id="cap_2", suite="alpha", model_input="cancel order 99")
    _write_capture(tmp_path, capture_id="cap_3", suite="beta")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    # golden.jsonl built for every suite.
    for suite, n in (("alpha", 2), ("beta", 1)):
        golden = tmp_path / "suites" / suite / "golden.jsonl"
        assert load_jsonl(golden).ids().__len__() == n

    # suites: block wired into the config and resolvable.
    cfg = load_config(config)
    assert set(cfg.suites) == {"alpha", "beta"}
    assert cfg.suites["alpha"].source == "captured"
    resolved = config.resolve().parent / cfg.suites["alpha"].path
    assert resolved.is_file()


def test_sync_skips_empty_events(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_ok", suite="alpha")
    _write_empty_capture(tmp_path, capture_id="cap_empty", suite="ghost")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "no events" in result.stdout.lower()
    cfg = load_config(config)
    assert "ghost" not in cfg.suites
    assert "alpha" in cfg.suites


def test_sync_print_does_not_touch_config(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    before = config.read_text(encoding="utf-8")

    result = _invoke(["sync", "--print", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert config.read_text(encoding="utf-8") == before  # config untouched
    assert "suites:" in result.stdout
    # ...but cases + golden are still written.
    assert (tmp_path / "suites" / "alpha" / "golden.jsonl").is_file()


def test_sync_idempotent_skips_already_promoted(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    again = _invoke(["sync", "--config", str(config)], tmp_path)

    assert again.exit_code == 0, again.stdout
    assert "already-promoted" in again.stdout.lower()


def test_sync_warns_on_unreadable_capture(tmp_path: Path) -> None:
    """A capture that fails envelope validation is surfaced, not silently dropped.

    Regression: SDK <= 1.0 wrote error events with an empty message for
    exceptions whose str() is empty (e.g. CancelledError); the file failed
    validation and sync dropped it without a trace.
    """
    _write_capture(tmp_path, capture_id="cap_ok", suite="alpha")
    bad_dir = tmp_path / "captures" / "ghost"
    bad_dir.mkdir(parents=True)
    payload = {
        "schema_version": "2.0.0",
        "capture_id": "cap_bad",
        "suite": "ghost",
        "trace": {
            "run_id": "cap_bad",
            "prompt_id": "ghost",
            "example_id": "cap_bad",
            "role": "source",
            "events": [
                {
                    "type": "error",
                    "sequence_index": 0,
                    "timestamp": "2026-06-16T12:00:00+00:00",
                    "metadata": {},
                    "message": "",
                    "category": "CancelledError",
                },
            ],
        },
    }
    (bad_dir / "cap_bad.json").write_text(json.dumps(payload), encoding="utf-8")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "unreadable" in result.stdout.lower()
    assert "cap_bad" in result.stdout
    cfg = load_config(config)
    assert "ghost" not in cfg.suites
    assert "alpha" in cfg.suites


def test_sync_no_captures(tmp_path: Path) -> None:
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0
    assert "no captures" in result.stdout.lower()


def test_sync_prints_block_when_markers_absent(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    config.write_text("version: 1\n", encoding="utf-8")  # no managed markers

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "suites:" in result.stdout
    # config left untouched since it had no markers to replace.
    assert config.read_text(encoding="utf-8") == "version: 1\n"


# ---------------------------------------------------------------------------
# sync — conversation-aware promotion
# ---------------------------------------------------------------------------


def test_sync_promotes_conversation_with_reconstructed_history(tmp_path: Path) -> None:
    """A 2-turn conversation + a standalone capture in the same suite.

    golden.jsonl should carry 3 rows; the second conversation turn should
    carry a reconstructed history (turn 1's user text + assistant reply);
    the promoted-case JSON files should carry conversation provenance.
    """
    _write_capture(
        tmp_path,
        capture_id="cap_t0",
        suite="support_agent",
        events=[
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": "2026-06-16T12:00:00+00:00",
                "metadata": {},
                "model_id": "m",
                "input": "hi",
                "output": "out",
                "toolset_ref": "sha256:" + "ab" * 32,
                "tools_offered": [],
            },
            {
                "type": "final_output",
                "sequence_index": 1,
                "timestamp": "2026-06-16T12:00:03+00:00",
                "metadata": {},
                "text": "hello, how can I help?",
            },
        ],
        conversation_id="conv_1",
        turn_index=0,
    )
    _write_capture(
        tmp_path,
        capture_id="cap_t1",
        suite="support_agent",
        events=[
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": "2026-06-16T12:00:05+00:00",
                "metadata": {},
                "model_id": "m",
                "input": "where is order 12345?",
                "output": "out",
                "toolset_ref": "sha256:" + "ab" * 32,
                "tools_offered": [],
            },
            {
                "type": "final_output",
                "sequence_index": 1,
                "timestamp": "2026-06-16T12:00:06+00:00",
                "metadata": {},
                "text": "Your order ships tomorrow.",
            },
        ],
        conversation_id="conv_1",
        turn_index=1,
    )
    _write_capture(tmp_path, capture_id="cap_solo", suite="support_agent")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    golden = tmp_path / "suites" / "support_agent" / "golden.jsonl"
    suite = load_jsonl(golden)
    assert suite.ids() == {"cap_t0", "cap_t1", "cap_solo"}

    turn1 = next(e for e in suite.examples if e.id == "cap_t1")
    assert turn1.conversation_id == "conv_1"
    assert turn1.turn_index == 1
    assert turn1.history is not None
    assert [m.role for m in turn1.history] == ["user", "assistant"]
    assert turn1.history[0].content == "hi"
    assert turn1.history[1].content == "hello, how can I help?"

    turn0 = next(e for e in suite.examples if e.id == "cap_t0")
    assert turn0.conversation_id == "conv_1"
    assert turn0.turn_index == 0

    solo = next(e for e in suite.examples if e.id == "cap_solo")
    assert solo.conversation_id is None

    # provenance also lands in the canonical case JSON files.
    case_t1 = PromotedCase.model_validate_json(
        (tmp_path / "suites" / "support_agent" / "cap_t1.json").read_text(encoding="utf-8"),
    )
    assert case_t1.conversation_id == "conv_1"
    assert case_t1.turn_index == 1


def test_sync_conversation_dedup_skips_already_promoted(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_t0",
        suite="support_agent",
        conversation_id="conv_1",
        turn_index=0,
    )
    _write_capture(
        tmp_path,
        capture_id="cap_t1",
        suite="support_agent",
        conversation_id="conv_1",
        turn_index=1,
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    again = _invoke(["sync", "--config", str(config)], tmp_path)

    assert again.exit_code == 0, again.stdout
    assert "already-promoted" in again.stdout.lower()


# ---------------------------------------------------------------------------
# promote — conversation sibling hint
# ---------------------------------------------------------------------------


def test_promote_single_with_unpromoted_siblings_prints_hint(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_t0",
        suite="support_agent",
        conversation_id="conv_1",
        turn_index=0,
    )
    _write_capture(
        tmp_path,
        capture_id="cap_t1",
        suite="support_agent",
        conversation_id="conv_1",
        turn_index=1,
    )

    result = _invoke(["promote", "cap_t0", "--as", "case_t0"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "conv_1" in result.stdout
    assert "capture sync" in result.stdout


def test_promote_single_without_siblings_no_hint(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_solo", suite="support_agent")

    result = _invoke(["promote", "cap_solo", "--as", "case_solo"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "unpromoted sibling" not in result.stdout.lower()


def test_promote_single_uses_own_messages_list_only(tmp_path: Path) -> None:
    """Single promote does not reconstruct cross-capture history."""
    _write_capture(
        tmp_path,
        capture_id="cap_t0",
        suite="support_agent",
        conversation_id="conv_1",
        turn_index=0,
    )
    _write_capture(
        tmp_path,
        capture_id="cap_t1",
        suite="support_agent",
        conversation_id="conv_1",
        turn_index=1,
    )

    result = _invoke(["promote", "cap_t1", "--as", "case_t1"], tmp_path)

    assert result.exit_code == 0, result.stdout
    case = PromotedCase.model_validate_json(
        (tmp_path / "suites" / "support_agent" / "case_t1.json").read_text(encoding="utf-8"),
    )
    # No history reconstruction — cap_t1 recorded a bare string, not a
    # messages list, so history stays None for a lone `promote`.
    assert case.example.history is None
    assert case.conversation_id == "conv_1"
    assert case.turn_index == 1


def test_sync_dedupes_identical_input_hash(tmp_path: Path) -> None:
    # Re-exercising an agent on the same data produces captures with the
    # same content — sync must promote only the first.
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha", input_hash="same")
    _write_capture(tmp_path, capture_id="cap_2", suite="alpha", input_hash="same")
    _write_capture(tmp_path, capture_id="cap_3", suite="alpha", model_input="different question")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "duplicate" in result.stdout.lower()
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert len(load_jsonl(golden).ids()) == 2


def test_sync_dedupes_same_content_despite_distinct_input_hash(tmp_path: Path) -> None:
    # The SDK folds conversation_id into input_hash, so re-captured identical
    # inputs carry DIFFERENT hashes. Dedup must key on the built example's
    # content, not the envelope hash.
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha", input_hash="h1")
    _write_capture(tmp_path, capture_id="cap_2", suite="alpha", input_hash="h2")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "duplicate" in result.stdout.lower()
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert len(load_jsonl(golden).ids()) == 1


def test_sync_keep_duplicates_flag_promotes_all(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha", input_hash="same")
    _write_capture(tmp_path, capture_id="cap_2", suite="alpha", input_hash="same")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--keep-duplicates", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert len(load_jsonl(golden).ids()) == 2


def test_sync_same_hash_in_different_suites_not_deduped(tmp_path: Path) -> None:
    # Dedup is per-suite: identical inputs in unrelated suites both promote.
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha", input_hash="same")
    _write_capture(tmp_path, capture_id="cap_2", suite="beta", input_hash="same")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert len(load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").ids()) == 1
    assert len(load_jsonl(tmp_path / "suites" / "beta" / "golden.jsonl").ids()) == 1


def test_sync_dedupes_against_already_promoted_cases(tmp_path: Path) -> None:
    # Dedup must span sync runs. `cap_a` is captured after `cap_z` was already
    # promoted, but sorts first, so the in-run pass would promote `cap_a` and
    # skip `cap_z` -- whose case file is already on disk, leaving both.
    _write_capture(tmp_path, capture_id="cap_z", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    first = _invoke(["sync", "--config", str(config)], tmp_path)
    assert first.exit_code == 0, first.stdout

    _write_capture(tmp_path, capture_id="cap_a", suite="alpha")
    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "duplicate" in result.stdout.lower()
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert load_jsonl(golden).ids() == {"cap_z"}
    assert not (tmp_path / "suites" / "alpha" / "cap_a.json").exists()


def test_sync_is_idempotent_across_runs(tmp_path: Path) -> None:
    # Seeding from disk must not make a capture a duplicate of its own
    # promoted case: re-syncing the same captures changes nothing.
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    _write_capture(tmp_path, capture_id="cap_2", suite="alpha", model_input="another question")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert load_jsonl(golden).ids() == {"cap_1", "cap_2"}


def test_sync_force_refreshes_its_own_promoted_case(tmp_path: Path) -> None:
    # Seeding dedup from disk must not make a capture a duplicate of the case
    # it wrote itself: --force still re-promotes it with the new options.
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0

    result = _invoke(
        ["sync", "--force", "--tag", "rerun", "--config", str(config)],
        tmp_path,
    )

    assert result.exit_code == 0, result.stdout
    # Same replayed content as the case on disk, so only the owner check keeps
    # it out of the duplicate bucket.
    assert "duplicate" not in result.stdout.lower()
    assert "promoted 1 capture" in result.stdout
    case_path = tmp_path / "suites" / "alpha" / "cap_1.json"
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    assert "rerun" in case.example.tags


def test_sync_skips_capture_already_promoted_under_a_custom_name(tmp_path: Path) -> None:
    # `capture promote --as my_case` writes the case under a name sync would
    # never pick, so sync must recognise the content as already promoted
    # rather than writing a second case file for the same capture.
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["promote", "cap_1", "--as", "my_case"], tmp_path).exit_code == 0

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert load_jsonl(golden).ids() == {"my_case"}
    assert not (tmp_path / "suites" / "alpha" / "cap_1.json").exists()


def test_sync_keep_duplicates_ignores_already_promoted_cases(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_z", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0

    _write_capture(tmp_path, capture_id="cap_a", suite="alpha")
    result = _invoke(["sync", "--keep-duplicates", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert load_jsonl(golden).ids() == {"cap_a", "cap_z"}


# ---------------------------------------------------------------------------
# Promotion hygiene — errored turns and duplicate conversation turns
# ---------------------------------------------------------------------------


def _errored_events(model_input: Any = "any duplicated projects?") -> list[dict[str, Any]]:
    return [
        {
            "type": "model_call",
            "sequence_index": 0,
            "timestamp": "2026-06-16T12:00:00+00:00",
            "metadata": {},
            "model_id": "m",
            "input": model_input,
            "output": "",
            "toolset_ref": "sha256:" + "ab" * 32,
            "tools_offered": [],
        },
        {
            "type": "error",
            "sequence_index": 1,
            "timestamp": "2026-06-16T12:00:01+00:00",
            "metadata": {},
            "message": "400 Bad Request. CachedContent model mismatch.",
        },
    ]


def test_promote_refuses_an_errored_capture(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_err", suite="alpha", events=_errored_events())

    result = _invoke(["promote", "cap_err", "--as", "case_err"], tmp_path)

    assert result.exit_code == 1
    assert "error event" in result.stdout
    # --allow-errored is the one flag that *can* rescue this refusal, so the
    # hint belongs here (and only here -- see the no-toolset test below).
    assert "--allow-errored" in result.stdout
    assert not (tmp_path / "suites" / "alpha" / "case_err.json").exists()


def test_promote_refuses_a_capture_with_an_unresolvable_toolset_ref(tmp_path: Path) -> None:
    """toolset_ref is set, but no sidecar exists on disk for it -- refused, not
    silently promoted with a reference nothing can ever resolve. Confirms --base
    (here, tmp_path) is what the sidecar-existence check resolves against, end to
    end through the CLI, not just at the build_example_from_capture unit level."""
    dangling_ref = "sha256:" + "77" * 32
    _write_capture(
        tmp_path,
        capture_id="cap_dangling",
        suite="alpha",
        events=[
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": "2026-06-16T12:00:00+00:00",
                "metadata": {},
                "model_id": "m",
                "input": "hi",
                "output": "out",
                "toolset_ref": dangling_ref,
                "tools_offered": ["search_orders"],
            },
            {
                "type": "final_output",
                "sequence_index": 1,
                "timestamp": "2026-06-16T12:00:03+00:00",
                "metadata": {},
                "text": "hello",
            },
        ],
    )

    result = _invoke(["promote", "cap_dangling", "--as", "case_dangling"], tmp_path)

    assert result.exit_code == 1
    assert "cap_dangling" in result.stdout
    assert dangling_ref in result.stdout
    assert not (tmp_path / "suites" / "alpha" / "case_dangling.json").exists()


def test_promote_no_toolset_refusal_does_not_suggest_allow_errored(tmp_path: Path) -> None:
    """--allow-errored only rescues an errored *turn*; it can never make a
    capture with no resolvable toolset promotable. Offering it here would send
    the operator down a dead end instead of at the real fix (re-capture)."""
    dangling_ref = "sha256:" + "88" * 32
    _write_capture(
        tmp_path,
        capture_id="cap_no_toolset",
        suite="alpha",
        events=[
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": "2026-06-16T12:00:00+00:00",
                "metadata": {},
                "model_id": "m",
                "input": "hi",
                "output": "out",
                "toolset_ref": dangling_ref,
                "tools_offered": ["search_orders"],
            },
            {
                "type": "final_output",
                "sequence_index": 1,
                "timestamp": "2026-06-16T12:00:03+00:00",
                "metadata": {},
                "text": "hello",
            },
        ],
    )

    result = _invoke(["promote", "cap_no_toolset", "--as", "case_nt"], tmp_path)

    assert result.exit_code == 1
    assert "--allow-errored" not in result.stdout
    assert "re-capture" in result.stdout.lower()
    assert not (tmp_path / "suites" / "alpha" / "case_nt.json").exists()


def test_promote_allow_errored_promotes_without_expected_no_tools(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_err", suite="alpha", events=_errored_events())

    result = _invoke(
        ["promote", "cap_err", "--as", "case_err", "--allow-errored"],
        tmp_path,
    )

    assert result.exit_code == 0, result.stdout
    case_path = tmp_path / "suites" / "alpha" / "case_err.json"
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    assert case.example.expected_no_tools is False


def test_sync_skips_errored_captures(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_ok", suite="alpha")
    _write_capture(tmp_path, capture_id="cap_err", suite="alpha", events=_errored_events())
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "error event" in result.stdout
    assert load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").ids() == {"cap_ok"}


def test_sync_allow_errored_promotes_them(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_ok", suite="alpha")
    _write_capture(tmp_path, capture_id="cap_err", suite="alpha", events=_errored_events())
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--allow-errored", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").ids() == {
        "cap_err",
        "cap_ok",
    }


def test_sync_skips_captures_with_no_toolset(tmp_path: Path) -> None:
    """capture sync's summary must count and describe a missing-toolset_ref
    refusal separately from an errored-capture refusal -- --allow-errored
    cannot rescue either, but only the latter's message may say so."""
    _write_capture(tmp_path, capture_id="cap_ok", suite="alpha")
    _write_capture(
        tmp_path,
        capture_id="cap_no_toolset",
        suite="alpha",
        events=[
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": "2026-06-16T12:00:00+00:00",
                "metadata": {},
                "model_id": "m",
                "input": "hi",
                "output": "out",
                "toolset_ref": None,
                "tools_offered": None,
            },
            {
                "type": "final_output",
                "sequence_index": 1,
                "timestamp": "2026-06-16T12:00:03+00:00",
                "metadata": {},
                "text": "hello",
            },
        ],
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").ids() == {"cap_ok"}
    assert "no toolset_ref" in result.stdout  # per-item warning names the actual reason
    assert "no usable" in result.stdout  # summary: its own counter/message
    assert "errored capture" not in result.stdout  # must not be lumped into that bucket


def test_sync_warns_when_two_captures_claim_the_same_turn(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_first",
        suite="alpha",
        conversation_id="75",
        turn_index=0,
    )
    _write_capture(
        tmp_path,
        capture_id="cap_retry",
        suite="alpha",
        conversation_id="75",
        turn_index=0,
        model_input="a different question",
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "conversation 75 turn 0" in result.stdout
    assert "cap_first" in result.stdout
    assert "cap_retry" in result.stdout


# ---------------------------------------------------------------------------
# --rounds
# ---------------------------------------------------------------------------


def _multi_round_events() -> list[dict[str, Any]]:
    """Two archives in round 1, one get_projects in round 2, a text-only round 3."""
    return [
        {
            "type": "model_call",
            "sequence_index": 0,
            "timestamp": "2026-06-16T12:00:00+00:00",
            "metadata": {},
            "model_id": "m",
            "input": "yes",
            "output": "",
            "toolset_ref": "sha256:" + "ab" * 32,
            "tools_offered": ["archive_project", "get_projects"],
        },
        {
            "type": "tool_call",
            "sequence_index": 1,
            "timestamp": "2026-06-16T12:00:01+00:00",
            "metadata": {},
            "name": "archive_project",
            "arguments": {"project_name": "Series A Fundraise"},
            "call_id": "call_a",
        },
        {
            "type": "tool_call",
            "sequence_index": 2,
            "timestamp": "2026-06-16T12:00:01+00:00",
            "metadata": {},
            "name": "archive_project",
            "arguments": {"project_name": "Q2 Product Launch"},
            "call_id": "call_b",
        },
        {
            "type": "model_call",
            "sequence_index": 3,
            "timestamp": "2026-06-16T12:00:02+00:00",
            "metadata": {},
            "model_id": "m",
            "input": "yes",
            "output": "",
            "toolset_ref": "sha256:" + "ab" * 32,
            "tools_offered": ["archive_project", "get_projects"],
        },
        {
            "type": "tool_call",
            "sequence_index": 4,
            "timestamp": "2026-06-16T12:00:03+00:00",
            "metadata": {},
            "name": "get_projects",
            "arguments": {},
            "call_id": "call_c",
        },
        {
            "type": "final_output",
            "sequence_index": 5,
            "timestamp": "2026-06-16T12:00:04+00:00",
            "metadata": {},
            "text": "Done.",
        },
    ]


def _flat(text: str) -> str:
    """Collapse Rich's soft-wrapping so a warning phrase can be asserted on."""
    return " ".join(text.split())


_MULTI_TOOLSET_ROUND_2_REF = "sha256:" + "cd" * 32


def _multi_toolset_round_events() -> list[dict[str, Any]]:
    """Round 1 offers one toolset, round 2 switches to a different one -- the
    exact per-call-toolset-switching shape I1 is about (the SDK stamps
    toolset_ref per call, so this is a legitimate recording, not a fixture
    mistake)."""
    events = _multi_round_events()
    events[3] = {**events[3], "toolset_ref": _MULTI_TOOLSET_ROUND_2_REF}
    return events


def test_sync_refuses_a_capture_that_switched_toolsets_mid_run(tmp_path: Path) -> None:
    """I1 at the capture-sync summary: routed to its own message, not lumped
    into the errored-capture bucket."""
    _write_capture(
        tmp_path,
        capture_id="cap_multi_toolset",
        suite="main_chat",
        events=_multi_toolset_round_events(),
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "suites" / "main_chat" / "golden.jsonl").exists()
    assert "switched toolsets" in _flat(result.stdout)
    assert "errored capture" not in result.stdout


def test_promote_refuses_a_capture_that_switched_toolsets_mid_run(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_multi_toolset",
        suite="main_chat",
        events=_multi_toolset_round_events(),
    )

    result = _invoke(["promote", "cap_multi_toolset", "--as", "case1"], tmp_path)

    assert result.exit_code == 1
    assert "switched toolsets" in _flat(result.stdout)


def test_sync_defaults_to_first_round(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_multi",
        suite="main_chat",
        events=_multi_round_events(),
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "agent round(s)" in _flat(result.stdout)

    suite = load_jsonl(tmp_path / "suites" / "main_chat" / "golden.jsonl")
    names = [t.tool_name for t in suite.examples[0].expected_tools or []]
    assert names == ["archive_project", "archive_project"]


def _tool_result_event(name: str, index: int, call_id: str, result: Any) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:01+00:00",
        "metadata": {},
        "name": name,
        "call_id": call_id,
        "result": result,
    }


def _multi_round_events_with_results() -> list[dict[str, Any]]:
    """``_multi_round_events`` with every call's recorded result — what a
    teacher-forced ``--rounds all`` promotion needs."""
    results = {
        "call_a": {"ok": True},
        "call_b": {"ok": True},
        "call_c": {"projects": []},
    }
    out: list[dict[str, Any]] = []
    for event in _multi_round_events():
        out.append({**event, "sequence_index": len(out)})
        if event["type"] == "tool_call":
            call_id = str(event["call_id"])
            out.append(
                _tool_result_event(str(event["name"]), len(out), call_id, results[call_id]),
            )
    return out


def test_sync_rounds_all_scopes_expected_tools_to_round_one(tmp_path: Path) -> None:
    """--rounds all no longer flattens; without recorded results it also
    cannot cover a single round, so it says so."""
    _write_capture(
        tmp_path,
        capture_id="cap_multi",
        suite="main_chat",
        events=_multi_round_events(),
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--rounds", "all", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "replay stays single-shot" in _flat(result.stdout)
    suite = load_jsonl(tmp_path / "suites" / "main_chat" / "golden.jsonl")
    names = [t.tool_name for t in suite.examples[0].expected_tools or []]
    assert names == ["archive_project", "archive_project"]
    assert suite.examples[0].tool_result_fixtures is None


def test_promote_rounds_all_carries_tool_result_fixtures(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_multi",
        suite="main_chat",
        events=_multi_round_events_with_results(),
    )

    result = _invoke(["promote", "cap_multi", "--as", "case1", "--rounds", "all"], tmp_path)

    assert result.exit_code == 0, result.stdout
    suite = load_jsonl(tmp_path / "suites" / "main_chat" / "golden.jsonl")
    example = suite.examples[0]
    assert [t.tool_name for t in example.expected_tools or []] == [
        "archive_project",
        "archive_project",
    ]
    assert [[f.tool_name for f in r] for r in example.tool_result_fixtures or []] == [
        ["archive_project", "archive_project"],
        ["get_projects"],
    ]
    assert example.rounds_to_replay() == 3


def test_promote_records_every_round_on_the_case(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_multi",
        suite="main_chat",
        events=_multi_round_events(),
    )

    result = _invoke(["promote", "cap_multi", "--as", "case1"], tmp_path)

    assert result.exit_code == 0, result.stdout
    case_path = tmp_path / "suites" / "main_chat" / "case1.json"
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    rounds = case.example.expected_tool_rounds or []
    assert [[t.tool_name for t in r] for r in rounds] == [
        ["archive_project", "archive_project"],
        ["get_projects"],
    ]


def _requested_events(names: list[str]) -> list[dict[str, Any]]:
    """A capture whose model_call records the calls the model asked for."""
    events = _events(tools=names)
    events[0]["requested_tool_calls"] = [
        {"name": name, "arguments": {"customer_id": "c42"}} for name in names
    ]
    return events


def test_promote_records_the_promotion_source_on_the_case(tmp_path: Path) -> None:
    """Reports need to know which yardstick a case was promoted against."""
    _write_capture(tmp_path, capture_id="cap_req", events=_requested_events(["search_orders"]))
    _write_capture(tmp_path, capture_id="cap_exec", tools=["search_orders"])

    assert _invoke(["promote", "cap_req", "--as", "req"], tmp_path).exit_code == 0
    assert _invoke(["promote", "cap_exec", "--as", "exe"], tmp_path).exit_code == 0

    suite_dir = tmp_path / "suites" / "support_agent"
    requested = PromotedCase.model_validate_json((suite_dir / "req.json").read_text("utf-8"))
    executed = PromotedCase.model_validate_json((suite_dir / "exe.json").read_text("utf-8"))
    assert requested.promotion_source == "requested"
    assert executed.promotion_source == "executed"


def test_sync_records_the_promotion_source_on_every_case(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_req", events=_requested_events(["search_orders"]))

    result = _invoke(["sync", "--print"], tmp_path)

    assert result.exit_code == 0, result.stdout
    case_path = tmp_path / "suites" / "support_agent" / "cap_req.json"
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    assert case.promotion_source == "requested"


def test_rounds_rejects_an_unknown_value(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1")

    result = _invoke(["promote", "cap_1", "--as", "case1", "--rounds", "some"], tmp_path)

    assert result.exit_code != 0
    assert "--rounds" in result.output


def test_sync_reports_wired_generation_config(tmp_path: Path) -> None:
    events = _events(tools=[])
    events[0]["metadata"] = {"generation_config": {"response_mime_type": "application/json"}}
    _write_capture(tmp_path, capture_id="cap_gen", events=events)
    result = _invoke(["sync", "--print"], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "wired generation config for 1 case(s)" in result.stdout


def test_sync_without_generation_config_omits_the_fragment(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_plain")
    result = _invoke(["sync", "--print"], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "wired generation config" not in result.stdout


# ---------------------------------------------------------------------------
# sync — per-suite evaluator generation
# ---------------------------------------------------------------------------


def _write_tool_free_capture(
    base: Path,
    *,
    capture_id: str,
    suite: str,
    model_input: str = "summarize the quarter",
) -> None:
    """Write a capture whose agent was offered no tools at all.

    The empty toolset is a real, fingerprinted value, so the sidecar is written
    under its own fingerprint (and verifies against it) rather than under the
    shared placeholder ref the tool-calling fixtures use.
    """
    toolsets_dir = base / "toolsets"
    toolsets_dir.mkdir(parents=True, exist_ok=True)
    hex_ref = EMPTY_TOOLSET_FINGERPRINT.removeprefix("sha256:")
    (toolsets_dir / f"{hex_ref}.json").write_text('{"tools": []}', encoding="utf-8")
    _write_capture(
        base,
        capture_id=capture_id,
        suite=suite,
        model_input=model_input,
        events=[
            {
                "type": "model_call",
                "sequence_index": 0,
                "timestamp": "2026-06-16T12:00:00+00:00",
                "metadata": {},
                "model_id": "m",
                "input": model_input,
                "output": "out",
                "toolset_ref": EMPTY_TOOLSET_FINGERPRINT,
                "tools_offered": [],
            },
            {
                "type": "final_output",
                "sequence_index": 1,
                "timestamp": "2026-06-16T12:00:03+00:00",
                "metadata": {},
                "text": "done",
            },
        ],
    )


def _suite_entry_text(config_text: str, name: str) -> str:
    """Slice one suite's raw entry (its key line plus its indented body)."""
    lines = config_text.splitlines()
    start = lines.index(f"  {name}:")
    entry = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("    "):
            break
        entry.append(line)
    return "\n".join(entry)


def test_sync_wires_tool_evaluators_for_a_tool_calling_suite(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    cfg = load_config(config)
    override = cfg.suites["alpha"].evaluators
    assert override is not None
    assert override.tool_selection is not None
    assert [e.name for e in override.tool_selection] == ["routing"]
    # The fixture's tool call recorded arguments, so they are scored too.
    assert override.tool_arguments is not None
    assert [e.name for e in override.tool_arguments] == ["routing_args"]
    assert override.tool_arguments[0].against == "expected"
    # ...and the resolved set is what evaluate/report/bundle will actually use.
    resolved = cfg.evaluators_for("alpha")
    assert [e.name for e in resolved.tool_selection] == ["routing"]


def test_sync_leaves_a_tool_free_suite_without_tool_evaluators(tmp_path: Path) -> None:
    """A prose suite must not inherit a tool evaluator with an empty denominator."""
    _write_tool_free_capture(tmp_path, capture_id="cap_1", suite="prose")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    cfg = load_config(config)
    assert cfg.suites["prose"].evaluators is None
    assert cfg.evaluators_for("prose").tool_selection == []


def test_sync_of_one_suite_leaves_the_others_untouched(tmp_path: Path) -> None:
    """The partition guarantee: regenerating one suite never disturbs another."""
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    _write_tool_free_capture(tmp_path, capture_id="cap_2", suite="beta")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    beta_before = _suite_entry_text(config.read_text(encoding="utf-8"), "beta")

    result = _invoke(["sync", "--suite", "alpha", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    after = config.read_text(encoding="utf-8")
    assert _suite_entry_text(after, "beta") == beta_before
    assert "alpha" in load_config(config).suites


def test_resync_of_unchanged_captures_is_byte_identical(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    _write_tool_free_capture(tmp_path, capture_id="cap_2", suite="beta")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    first = config.read_text(encoding="utf-8")

    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0

    assert config.read_text(encoding="utf-8") == first


def test_sync_freezes_an_unmanaged_suite_and_prints_what_it_would_write(
    tmp_path: Path,
) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    # Hand-edit the entry and freeze it: sync must not regenerate either change.
    frozen = config.read_text(encoding="utf-8").replace(
        "  alpha:\n    source: captured",
        "  alpha:\n    managed: false\n    source: jsonl",
    )
    config.write_text(frozen, encoding="utf-8")
    entry_before = _suite_entry_text(frozen, "alpha")

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert _suite_entry_text(config.read_text(encoding="utf-8"), "alpha") == entry_before
    assert "alpha" in result.stdout
    assert "managed" in result.stdout


# ---------------------------------------------------------------------------
# sync — CI pin drift warning
# ---------------------------------------------------------------------------


def _write_stale_workflow(root: Path, version: str = "0.0.1") -> Path:
    path = root / ".github" / "workflows" / "evalshift.yml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "on: push\njobs:\n  evalshift:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - uses: evalshift/evalshift-action@v0\n"
        f'        with:\n          evalshift-version: "{version}"\n',
        encoding="utf-8",
    )
    return path


def test_sync_warns_when_ci_pins_an_older_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evalshift_cli.cli.commands.capture.__version__", "1.2.3")
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    _write_stale_workflow(tmp_path)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    # The warning comes after the successful write, never instead of it.
    wired = result.stdout.index("wired 1 suite(s)")
    warned = result.stdout.index("CI installs evalshift 0.0.1")
    assert wired < warned
    assert ".github/workflows/evalshift.yml, job evalshift" in result.stdout
    assert 'evalshift-version: "1.2.3"' in result.stdout
    assert load_config(config).suites.keys() == {"alpha"}


def test_sync_print_still_warns_about_a_stale_ci_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("evalshift_cli.cli.commands.capture.__version__", "1.2.3")
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    _write_stale_workflow(tmp_path)

    result = _invoke(["sync", "--config", str(config), "--print"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "CI installs evalshift 0.0.1" in result.stdout


def test_sync_is_silent_when_no_workflow_uses_the_action(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "CI installs" not in result.stdout


# ---------------------------------------------------------------------------
# sync — hand-written trace_invariants survive regeneration
# ---------------------------------------------------------------------------


def test_sync_keeps_a_managed_suites_trace_invariants(tmp_path: Path) -> None:
    """Review Focus 5: a contract is human-owned; regeneration must not drop it."""
    config_path = tmp_path / "evalshift.yaml"
    suite_path = tmp_path / "suites" / "checkout" / "golden.jsonl"
    suite_path.parent.mkdir(parents=True)
    suite_path.write_text("", encoding="utf-8")
    rules = [
        {
            "name": "payments",
            "rules": [{"id": "no-v1", "type": "forbidden", "tools": ["refund_v1"]}],
        }
    ]
    existing = {
        "checkout": {
            "source": "captured",
            "path": "suites/checkout/golden.jsonl",
            "evaluators": {"tool_selection": [{"name": "old"}], "trace_invariants": rules},
        },
    }

    entries, frozen = _build_suite_entries(
        {"checkout": suite_path},
        {"checkout": None},
        config_path=config_path,
        existing=existing,
    )

    assert frozen == {}
    assert entries["checkout"]["evaluators"] == {"trace_invariants": rules}


def test_sync_without_prior_rules_is_unchanged(tmp_path: Path) -> None:
    config_path = tmp_path / "evalshift.yaml"
    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text("", encoding="utf-8")
    entries, _ = _build_suite_entries(
        {"s": suite_path},
        {"s": None},
        config_path=config_path,
        existing={},
    )
    assert "evaluators" not in entries["s"]


def test_sync_preview_of_a_frozen_suite_shows_its_trace_invariants_kept(tmp_path: Path) -> None:
    """The ``managed: false`` preview is what sync *would* write -- rules included."""
    config_path = tmp_path / "evalshift.yaml"
    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text("", encoding="utf-8")
    rules = [{"name": "p", "rules": [{"id": "r", "type": "forbidden", "tools": ["t"]}]}]
    prior = {
        "managed": False,
        "source": "captured",
        "path": "golden.jsonl",
        "evaluators": {"trace_invariants": rules},
    }

    entries, frozen = _build_suite_entries(
        {"s": suite_path}, {"s": None}, config_path=config_path, existing={"s": prior}
    )

    assert entries["s"] is prior  # the frozen entry itself is written back untouched
    assert frozen["s"]["evaluators"] == {"trace_invariants": rules}


_HAND_WRITTEN_RULES = """\
      trace_invariants:
        - name: payments
          owner: payments-team
          rules:
            - id: no-v1
              type: forbidden
              tools:
                - refund_v1
            - id: refund-args
              type: arguments
              tool: refund
              json_schema:
                type: object
                required:
                  - order_id
"""


def test_sync_round_trips_hand_written_trace_invariants_through_the_config(
    tmp_path: Path,
) -> None:
    """End to end: rules hand-added to a managed entry survive a re-sync on disk."""
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    text = config.read_text(encoding="utf-8")
    assert "    evaluators:\n" in text
    config.write_text(
        text.replace("    evaluators:\n", "    evaluators:\n" + _HAND_WRITTEN_RULES, 1),
        encoding="utf-8",
    )
    expected = load_config(config).suites["alpha"].evaluators
    assert expected is not None
    assert expected.trace_invariants is not None

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    override = load_config(config).suites["alpha"].evaluators
    assert override is not None
    assert override.trace_invariants == expected.trace_invariants
    # Every other key is still regenerated.
    assert override.tool_selection is not None
    assert [e.name for e in override.tool_selection] == ["routing"]
    # ...and the rewritten region is stable from here on.
    settled = config.read_text(encoding="utf-8")
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    assert config.read_text(encoding="utf-8") == settled
