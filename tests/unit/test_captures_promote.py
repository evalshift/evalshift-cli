"""Tests for promoting a capture into a golden suite case.

``build_example_from_capture`` is the pure mapping from a recorded
:class:`CaptureEnvelope` to a :class:`SuiteExample`. ``write_promoted_case``
and ``rebuild_golden_jsonl`` are the disk side that makes a promoted case
runnable via the existing ``suite/loader.load_jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import litellm
import pytest

from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.captures.promote import (
    BuiltExample,
    PromoteOptions,
    _history_message,
    _tool_rounds,
    build_conversation_examples,
    build_example_from_capture,
    capture_provenance,
    duplicate_turn_warnings,
    example_content_key,
    rebuild_golden_jsonl,
    retire_promoted_case,
    write_promoted_case,
)
from evalshift_cli.captures.reader import CaptureRecord
from evalshift_cli.captures.toolset import EMPTY_TOOLSET_FINGERPRINT
from evalshift_cli.suite.loader import load_jsonl
from evalshift_cli.suite.models import HistoryToolCall, SuiteExample

FIXTURES = Path(__file__).parent / "fixtures" / "captures"
MULTI_ROUND_CAPTURE = FIXTURES / "multi_round_tools.json"


def load_capture_fixture(name: str) -> CaptureEnvelope:
    """Load a checked-in capture fixture as a validated envelope."""
    return CaptureEnvelope.model_validate_json((FIXTURES / name).read_text(encoding="utf-8"))


_TOOLSET_REF = "sha256:" + "ab" * 32
# Used by exactly one test (TestToolset::test_toolset_ref_is_carried_from_the_first_model_call)
# to prove the *specific* ref on the event is what rides onto the example, not a constant --
# needs its own sidecar below for the same reason _TOOLSET_REF does.
_OTHER_TOOLSET_REF = "sha256:" + "cd" * 32
# The checked-in multi_round_tools.json fixture's own recorded ref -- read from the fixture
# itself (rather than retyped) so it can never drift from what the JSON actually carries; a
# third, independent value this file needs a sidecar for.
_MULTI_ROUND_TOOLSET_REF = json.loads(MULTI_ROUND_CAPTURE.read_text(encoding="utf-8"))["trace"][
    "events"
][0]["toolset_ref"]


@pytest.fixture(autouse=True)
def _toolset_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every ``toolset_ref`` this file's fixtures use a real sidecar on disk.

    ``build_example_from_capture`` refuses to promote a ``toolset_ref`` whose
    sidecar doesn't resolve (see its own docstring) -- existence only, no
    parsing of tool bodies, so a minimal ``{"tools": []}`` placeholder is
    sufficient content for every ref this file uses; no test here asserts on
    resolved tool bodies. Autouse + ``base=None`` (every ``build_example_from_
    capture``/``build_conversation_examples`` call in this file omits ``base``)
    means this needs zero changes to the ~90 existing call sites: they all
    resolve through ``capture_base()`` -> ``$EVALSHIFT_DIR`` -> here.
    """
    toolsets_dir = tmp_path / "toolsets"
    toolsets_dir.mkdir()
    for ref in (_TOOLSET_REF, _OTHER_TOOLSET_REF, _MULTI_ROUND_TOOLSET_REF):
        (toolsets_dir / f"{ref.removeprefix('sha256:')}.json").write_text(
            '{"tools": []}', encoding="utf-8"
        )
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path))


def _model_call(
    index: int,
    *,
    model_input: Any,
    toolset_ref: str | None = _TOOLSET_REF,
    tools_offered: list[str] | None = None,
    requested_tool_calls: list[dict[str, Any]] | None = None,
    model_id: str = "m",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "model_call",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:00+00:00",
        "metadata": {},
        "model_id": model_id,
        "input": model_input,
        "output": "out",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "toolset_ref": toolset_ref,
        "tools_offered": tools_offered if tools_offered is not None else ["search_orders"],
    }
    # Omitted, never written as null: "absent" is what every pre-2.1.0 capture
    # looks like, and it is the case the executed-call path must keep serving.
    if requested_tool_calls is not None:
        event["requested_tool_calls"] = requested_tool_calls
    return event


# Sentinel for _tool_call's call_id: the default derives one from name+index,
# while an explicit None omits the field entirely -- what a capture from an SDK
# (or provider) that reported no call ids looks like, and the case the
# pair-by-name fallback exists for.
_AUTO_CALL_ID = "<auto>"


def _tool_call(
    name: str,
    index: int,
    *,
    args: dict[str, Any] | None = None,
    call_id: str | None = _AUTO_CALL_ID,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "tool_call",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:02+00:00",
        "metadata": {},
        "name": name,
        "arguments": args if args is not None else {"q": "x"},
    }
    if call_id == _AUTO_CALL_ID:
        event["call_id"] = f"call_{name}_{index}"
    elif call_id is not None:
        event["call_id"] = call_id
    return event


def _tool_result(
    name: str,
    index: int,
    *,
    result: Any = None,
    error: str | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "tool_result",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:02+00:00",
        "metadata": {},
        "name": name,
        "result": result,
        "error": error,
    }
    if call_id is not None:
        event["call_id"] = call_id
    return event


def _error(message: str, index: int) -> dict[str, Any]:
    return {
        "type": "error",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:04+00:00",
        "metadata": {},
        "message": message,
    }


def _final(text: str, index: int) -> dict[str, Any]:
    return {
        "type": "final_output",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:03+00:00",
        "metadata": {},
        "text": text,
    }


def _envelope(
    *,
    capture_id: str = "cap_abc",
    suite: str = "support_agent",
    events: list[dict[str, Any]] | None = None,
    input_hash: str = "hash123",
    created_at: str = "2026-06-16T12:00:00+00:00",
    conversation_id: str | None = None,
    turn_index: int | None = None,
    parent_capture_id: str | None = None,
) -> CaptureEnvelope:
    if events is None:
        events = [
            _model_call(0, model_input="where is order 12345?"),
            _tool_call("search_orders", 1, args={"customer_id": "c42"}),
            _final("Your order ships tomorrow.", 2),
        ]
    payload = {
        "schema_version": "2.0.0",
        "capture_id": capture_id,
        "suite": suite,
        "input_hash": input_hash,
        "code_version": "v1",
        "created_at": created_at,
        "trace": {
            "run_id": capture_id,
            "prompt_id": suite,
            "example_id": capture_id,
            "role": "source",
            "events": events,
        },
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "parent_capture_id": parent_capture_id,
    }
    return CaptureEnvelope.model_validate(payload)


def _record(envelope: CaptureEnvelope, *, path: Path | None = None) -> CaptureRecord:
    return CaptureRecord(path=path or Path(f"{envelope.capture_id}.json"), envelope=envelope)


def _messages_call(
    index: int,
    *,
    messages: list[dict[str, Any]],
    input_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "type": "model_call",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:00+00:00",
        "metadata": {},
        "model_id": "m",
        "input": messages,
        "output": "out",
        "input_tokens": input_tokens,
        "toolset_ref": "sha256:" + "ab" * 32,
        "tools_offered": [],
    }


# ---------------------------------------------------------------------------
# build_example_from_capture — the field mapping
# ---------------------------------------------------------------------------


def test_build_defaults_id_to_capture_id() -> None:
    built = build_example_from_capture(_envelope(), PromoteOptions())
    assert isinstance(built, BuiltExample)
    assert built.example.id == "cap_abc"


def test_build_uses_name_override_for_id() -> None:
    built = build_example_from_capture(_envelope(), PromoteOptions(name="case1"))
    assert built.example.id == "case1"


def test_build_tags_include_captured_and_suite() -> None:
    built = build_example_from_capture(
        _envelope(suite="support_agent"),
        PromoteOptions(tags=("regression",)),
    )
    assert built.example.tags == ["captured", "support_agent", "regression"]


def test_build_tools_become_expected_tools_in_order() -> None:
    events = [
        _model_call(0, model_input="q"),
        _tool_call("lookup_customer", 1),
        _tool_call("issue_refund", 2),
        _final("done", 3),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    ex = built.example
    assert ex.expected_no_tools is False
    assert [t.tool_name for t in ex.expected_tools or []] == ["lookup_customer", "issue_refund"]
    assert all(t.match_strategy == "subset" for t in ex.expected_tools or [])


def test_build_no_tools_sets_expected_no_tools() -> None:
    """Tools were offered (the default ``_model_call`` fixture) and none were called."""
    events = [_model_call(0, model_input="what is your refund policy?"), _final("We offer...", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.expected_no_tools is True
    assert built.example.expected_tools is None


# ---------------------------------------------------------------------------
# toolset_ref -- carried from the event's first-class field (V-note), mirroring
# the *shape* of the generation_config carry, not its metadata storage.
# ---------------------------------------------------------------------------


class TestToolset:
    def test_toolset_ref_is_carried_from_the_first_model_call(self) -> None:
        """A populated toolset promotes correctly: the ref rides onto the example."""
        events = [
            _model_call(0, model_input="q", toolset_ref="sha256:" + "cd" * 32),
            _tool_call("search_orders", 1),
            _final("done", 2),
        ]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.blocked is None
        assert built.example.toolset_ref == "sha256:" + "cd" * 32
        assert built.example.tools is None

    def test_the_empty_toolset_promotes_correctly(self) -> None:
        """A capture whose call was offered zero tools still promotes -- the empty
        toolset is a first-class value, not a reason to refuse."""
        events = [
            _model_call(0, model_input="what is your refund policy?", tools_offered=[]),
            _final("We offer...", 1),
        ]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.blocked is None
        assert built.example.toolset_ref == _TOOLSET_REF
        # No tools offered -> expected_no_tools must NOT be asserted (see
        # TestExpectedNoToolsRequiresOffered below): a text-only reply here is
        # not evidence the model correctly withheld a call it could have made.
        assert built.example.expected_no_tools is False

    def test_a_missing_toolset_ref_is_refused(self) -> None:
        """Refuse to promote a capture missing toolset_ref, naming the capture and
        telling the user to re-capture."""
        events = [
            _model_call(0, model_input="q", toolset_ref=None),
            _final("done", 1),
        ]
        built = build_example_from_capture(
            _envelope(capture_id="cap_no_toolset", events=events),
            PromoteOptions(),
        )
        assert built.blocked is not None
        assert "cap_no_toolset" in built.blocked
        assert "re-capture" in built.blocked.lower()

    def test_a_missing_toolset_ref_still_yields_a_loadable_example(self) -> None:
        """Even blocked, .example must satisfy SuiteExample's own exactly-one-of --
        build_example_from_capture must never crash on a refused capture."""
        events = [_model_call(0, model_input="q", toolset_ref=None), _final("done", 1)]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.blocked is not None
        assert built.example.toolset_ref is None
        assert built.example.tools == []

    def test_a_missing_toolset_ref_is_refused_even_with_allow_errored(self) -> None:
        """--allow-errored only concerns error events; it cannot rescue a capture
        the SDK never recorded a toolset for."""
        events = [_model_call(0, model_input="q", toolset_ref=None), _final("done", 1)]
        built = build_example_from_capture(
            _envelope(events=events), PromoteOptions(allow_errored=True)
        )
        assert built.blocked is not None

    def test_an_unresolvable_toolset_ref_is_refused(self) -> None:
        """A toolset_ref with no sidecar on disk is refused rather than promoted
        cleanly -- the risk table's commitment ('promote refuses an unresolvable
        ref rather than degrading'), and the exact shape a sidecar deleted by
        `capture clean`, moved, or never written produces. Existence-only: this
        must not require the sidecar to contain valid tool bodies, only to exist."""
        unresolvable_ref = "sha256:" + "99" * 32
        events = [
            _model_call(0, model_input="q", toolset_ref=unresolvable_ref),
            _tool_call("search_orders", 1),
            _final("done", 2),
        ]
        built = build_example_from_capture(
            _envelope(capture_id="cap_dangling", events=events),
            PromoteOptions(),
        )
        assert built.blocked is not None
        assert "cap_dangling" in built.blocked
        assert unresolvable_ref in built.blocked
        assert "re-capture" in built.blocked.lower()
        assert built.blocked_reason == "no_toolset"

    def test_an_unresolvable_toolset_ref_is_refused_even_with_allow_errored(self) -> None:
        """Same non-rescuability as a missing toolset_ref: --allow-errored only
        concerns error events, never a dangling toolset_ref."""
        unresolvable_ref = "sha256:" + "99" * 32
        events = [
            _model_call(0, model_input="q", toolset_ref=unresolvable_ref),
            _final("done", 1),
        ]
        built = build_example_from_capture(
            _envelope(events=events), PromoteOptions(allow_errored=True)
        )
        assert built.blocked is not None
        assert built.blocked_reason == "no_toolset"

    # -----------------------------------------------------------------------
    # I1: a capture that switched toolsets mid-run. The SDK stamps
    # toolset_ref per call, so more than one distinct ref across a capture's
    # model_call events is legitimate -- the exact workload per-call toolset
    # capture exists for -- but SuiteExample carries exactly one toolset for
    # its whole trace, and expected_tool_rounds retains every round
    # regardless of --rounds. Promoting such a capture as a single case would
    # assert a later round's calls against a toolset the example doesn't
    # carry: dispatched (or teacher-forced) with the wrong toolset, silently
    # recreating the mismatched-manifest bug this feature exists to prevent.
    # -----------------------------------------------------------------------

    def test_a_capture_that_switched_toolsets_mid_run_is_refused(self) -> None:
        events = [
            _model_call(0, model_input="do things", toolset_ref=_TOOLSET_REF),
            _tool_call("search_orders", 1),
            _model_call(2, model_input="do more", toolset_ref=_OTHER_TOOLSET_REF),
            _tool_call("get_projects", 3),
            _final("done", 4),
        ]
        built = build_example_from_capture(
            _envelope(capture_id="cap_switched", events=events), PromoteOptions()
        )
        assert built.blocked is not None
        assert "cap_switched" in built.blocked
        assert built.blocked_reason == "multi_toolset"

    def test_a_capture_that_switched_toolsets_mid_run_is_refused_even_with_rounds_all(
        self,
    ) -> None:
        """Latent under --rounds all too -- worse, even: expected_tools itself
        (not just expected_tool_rounds) would then carry round 2's calls
        against round 1's ref. The refusal must not depend on --rounds."""
        events = [
            _model_call(0, model_input="do things", toolset_ref=_TOOLSET_REF),
            _tool_call("search_orders", 1),
            _model_call(2, model_input="do more", toolset_ref=_OTHER_TOOLSET_REF),
            _tool_call("get_projects", 3),
            _final("done", 4),
        ]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions(rounds="all"))
        assert built.blocked is not None
        assert built.blocked_reason == "multi_toolset"

    def test_a_capture_reusing_the_same_toolset_across_rounds_is_not_refused(self) -> None:
        """Positive control: the common case -- one toolset for the whole
        capture, spanning several rounds -- must promote normally. (The
        checked-in multi_round_tools.json fixture records the identical ref
        on all three of its model_call events.)"""
        env = load_capture_fixture("multi_round_tools.json")
        built = build_example_from_capture(env, PromoteOptions())
        assert built.blocked is None

    def test_multi_toolset_refusal_is_not_rescued_by_allow_errored(self) -> None:
        """--allow-errored only concerns error events; a capture that
        switched toolsets mid-run has no error event to rescue."""
        events = [
            _model_call(0, model_input="do things", toolset_ref=_TOOLSET_REF),
            _tool_call("search_orders", 1),
            _model_call(2, model_input="do more", toolset_ref=_OTHER_TOOLSET_REF),
            _tool_call("get_projects", 3),
            _final("done", 4),
        ]
        built = build_example_from_capture(
            _envelope(events=events), PromoteOptions(allow_errored=True)
        )
        assert built.blocked is not None
        assert built.blocked_reason == "multi_toolset"

    def test_blocked_reason_distinguishes_multi_toolset_from_the_others(self) -> None:
        multi_toolset = build_example_from_capture(
            _envelope(
                events=[
                    _model_call(0, model_input="q", toolset_ref=_TOOLSET_REF),
                    _model_call(1, model_input="q2", toolset_ref=_OTHER_TOOLSET_REF),
                    _final("d", 2),
                ],
            ),
            PromoteOptions(),
        )
        no_toolset = build_example_from_capture(
            _envelope(events=[_model_call(0, model_input="q", toolset_ref=None), _final("d", 1)]),
            PromoteOptions(),
        )
        errored = build_example_from_capture(
            _envelope(events=[_model_call(0, model_input="q"), _error("boom", 1)]),
            PromoteOptions(),
        )
        assert multi_toolset.blocked_reason == "multi_toolset"
        assert no_toolset.blocked_reason == "no_toolset"
        assert errored.blocked_reason == "errored"

    def test_blocked_reason_distinguishes_no_toolset_from_errored_turn(self) -> None:
        """capture sync's summary (cli/commands/capture.py) routes each refusal to
        an accurate counter/message via blocked_reason, not by pattern-matching
        built.blocked's text -- both refusal kinds must be tagged correctly, and
        distinctly, for that routing to work."""
        no_toolset = build_example_from_capture(
            _envelope(events=[_model_call(0, model_input="q", toolset_ref=None), _final("d", 1)]),
            PromoteOptions(),
        )
        errored = build_example_from_capture(
            _envelope(events=[_model_call(0, model_input="q"), _error("boom", 1)]),
            PromoteOptions(),
        )
        assert no_toolset.blocked_reason == "no_toolset"
        assert errored.blocked_reason == "errored"

    def test_toolset_ref_round_trips_through_golden_jsonl(self, tmp_path: Path) -> None:
        events = [_model_call(0, model_input="q"), _final("done", 1)]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        case = PromotedCase(
            name=built.example.id,
            suite="support_agent",
            from_capture="cap_abc",
            example=built.example,
        )
        case_path = write_promoted_case(case, base=tmp_path)
        golden = rebuild_golden_jsonl(case_path.parent)
        suite = load_jsonl(golden)
        [loaded] = suite.examples
        assert loaded.toolset_ref == _TOOLSET_REF


class TestExpectedNoToolsRequiresOffered:
    """expected_no_tools must mean 'tools were offered and none were called', and
    must not be set at all when no tools were offered (V-note)."""

    def test_absent_when_nothing_was_offered_even_with_no_calls_and_no_errors(self) -> None:
        events = [
            _model_call(0, model_input="hi", tools_offered=[]),
            _final("hello", 1),
        ]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.example.expected_no_tools is False

    def test_true_when_tools_were_offered_and_none_were_called(self) -> None:
        events = [
            _model_call(0, model_input="hi", tools_offered=["search_orders", "issue_refund"]),
            _final("hello", 1),
        ]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.example.expected_no_tools is True

    def test_false_when_tools_were_offered_and_called(self) -> None:
        events = [
            _model_call(0, model_input="q", tools_offered=["search_orders"]),
            _tool_call("search_orders", 1),
            _final("done", 2),
        ]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.example.expected_no_tools is False

    def test_still_absent_on_an_allowed_errored_turn_with_no_tools_offered(self) -> None:
        """An errored turn is never evidence calling nothing was correct -- doubly so
        when nothing was even offered."""
        events = [
            _model_call(0, model_input="hi", tools_offered=[]),
            _error("boom", 1),
        ]
        built = build_example_from_capture(
            _envelope(events=events),
            PromoteOptions(allow_errored=True),
        )
        assert built.example.expected_no_tools is False


def test_build_strict_args_uses_exact_match() -> None:
    built = build_example_from_capture(_envelope(), PromoteOptions(strict_args=True))
    assert (built.example.expected_tools or [])[0].match_strategy == "exact"
    assert (built.example.expected_tools or [])[0].arguments == {"customer_id": "c42"}


def test_build_names_only_drops_arguments() -> None:
    built = build_example_from_capture(_envelope(), PromoteOptions(names_only=True))
    assert (built.example.expected_tools or [])[0].arguments is None


def test_build_tool_count_only_when_requested() -> None:
    assert (
        build_example_from_capture(_envelope(), PromoteOptions()).example.expected_tool_count
        is None
    )
    counted = build_example_from_capture(_envelope(), PromoteOptions(tool_count=True))
    assert counted.example.expected_tool_count == 1


def test_build_expected_from_final_output() -> None:
    built = build_example_from_capture(_envelope(), PromoteOptions())
    assert built.example.expected == {"final_output": "Your order ships tomorrow."}


def test_build_inputs_from_dict_model_input() -> None:
    events = [
        _model_call(0, model_input={"query": "hello", "lang": "en"}),
        _tool_call("search_orders", 1),
        _final("done", 2),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"query": "hello", "lang": "en"}
    assert built.warnings == []


def test_build_inputs_from_string_model_input_uses_input_var() -> None:
    events = [
        _model_call(0, model_input="where is order 12345?"),
        _tool_call("search_orders", 1),
        _final("done", 2),
    ]
    built = build_example_from_capture(
        _envelope(events=events),
        PromoteOptions(input_var="query"),
    )
    assert built.example.inputs == {"query": "where is order 12345?"}


def test_build_inputs_empty_with_warning_when_unrecoverable() -> None:
    events = [_tool_call("search_orders", 0), _final("done", 1)]  # no model_call
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {}
    assert built.warnings  # at least one warning surfaced


# ---------------------------------------------------------------------------
# messages-aware input recovery
# ---------------------------------------------------------------------------


def test_build_messages_list_splits_history_and_current_turn() -> None:
    messages = [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello, how can I help?"},
        {"role": "user", "content": "where is order 12345?"},
    ]
    events = [
        _messages_call(0, messages=messages),
        _tool_call("search_orders", 1),
        _final("Your order ships tomorrow.", 2),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    ex = built.example
    assert ex.inputs == {"input": "where is order 12345?"}
    assert ex.history is not None
    assert [m.role for m in ex.history] == ["system", "user", "assistant"]
    assert ex.history[0].content == "You are a support agent."
    assert ex.history[1].content == "hi"
    assert ex.history[2].content == "hello, how can I help?"


def test_build_messages_list_model_role_coerced_to_assistant() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "model", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["user", "assistant"]
    assert built.example.history[1].content == "hello"


def test_promote_preserves_tool_turns_in_history() -> None:
    messages = [
        {"role": "system", "content": "You are Kaila."},
        {"role": "user", "content": "list my projects"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "get_projects", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"projects": []}'},
        {"role": "assistant", "content": "You have none."},
        {"role": "user", "content": "yes"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    history = built.example.history
    assert history is not None
    assert [m.role for m in history] == ["system", "user", "assistant", "tool", "assistant"]
    assert history[2].tool_calls is not None
    assert history[2].tool_calls[0].name == "get_projects"
    assert history[3].tool_call_id == "c1"
    assert not any("tool-role history is not replayed" in w for w in built.warnings)


def test_promote_synthesises_an_id_for_an_unpaired_tool_result() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": "bye"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["user", "tool"]
    assert built.example.history[1].tool_call_id == "_pos1"
    unpaired = [w for w in built.warnings if "no tool_call_id" in w]
    assert len(unpaired) == 1
    assert "1 tool result(s)" in unpaired[0]


def test_promote_pairs_an_idless_tool_result_with_the_preceding_assistant_call() -> None:
    """Gemini records no function-response ids; the result answers the call just before it."""
    messages = [
        {"role": "user", "content": "How far is 3.5 miles in km?"},
        {
            "role": "model",
            "content": "",
            "tool_calls": [
                {"id": "call_1389050", "name": "convert_units", "arguments": {"value": 3.5}}
            ],
        },
        {
            "role": "tool",
            "content": '{"result": 5.633}',
            "tool_call_id": None,
            "name": "convert_units",
        },
        {"role": "model", "content": "3.5 miles is approximately 5.63 kilometers."},
        {"role": "user", "content": "What's 120 euros in yen?"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert built.example.history[2].tool_call_id == "call_1389050"
    assert not any("_pos" in (m.tool_call_id or "") for m in built.example.history)
    assert not any("synthetic ids were assigned" in w for w in built.warnings)
    paired = [w for w in built.warnings if "paired" in w]
    assert len(paired) == 1
    assert "1 tool result(s)" in paired[0]


def test_promote_pairs_idless_tool_results_with_parallel_calls_in_order() -> None:
    messages = [
        {"role": "user", "content": "convert both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "convert_units", "arguments": {"value": 1}},
                {"id": "c2", "name": "convert_units", "arguments": {"value": 2}},
            ],
        },
        {"role": "tool", "content": "one"},
        {"role": "tool", "content": "two"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert built.example.history[2].tool_call_id == "c1"
    assert built.example.history[3].tool_call_id == "c2"


def test_promote_pairing_skips_call_ids_already_answered_by_a_keyed_result() -> None:
    messages = [
        {"role": "user", "content": "convert both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "convert_units", "arguments": {"value": 1}},
                {"id": "c2", "name": "convert_units", "arguments": {"value": 2}},
            ],
        },
        {"role": "tool", "content": "one", "tool_call_id": "c1"},
        {"role": "tool", "content": "two"},
        {"role": "user", "content": "thanks"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert built.example.history[3].tool_call_id == "c2"


def test_promote_synthesises_one_shared_id_when_neither_call_nor_result_has_one() -> None:
    messages = [
        {"role": "user", "content": "convert"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "convert_units", "arguments": {"value": 1}}],
        },
        {"role": "tool", "content": "one"},
        {"role": "user", "content": "thanks"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    history = built.example.history
    assert history is not None
    assert history[1].tool_calls is not None
    call_id = history[1].tool_calls[0].id
    assert call_id is not None
    assert history[2].tool_call_id == call_id


def test_promote_does_not_pair_an_idless_result_across_a_later_user_turn() -> None:
    """A result after the next user turn answers nothing in the prefix; fall back to a positional id."""
    messages = [
        {"role": "user", "content": "convert"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "convert_units", "arguments": {}}],
        },
        {"role": "tool", "content": "one", "tool_call_id": "c1"},
        {"role": "user", "content": "again"},
        {"role": "tool", "content": "stray"},
        {"role": "user", "content": "thanks"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert built.example.history[4].tool_call_id == "_pos4"
    assert any("synthetic ids were assigned" in w for w in built.warnings)


def test_build_messages_list_unrecognised_role_dropped_with_aggregated_warning() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "function", "content": "legacy output 1"},
        {"role": "function", "content": "legacy output 2"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["user", "assistant"]
    dropped = [w for w in built.warnings if "unrecognised role" in w]
    assert len(dropped) == 1
    assert "dropped 2 history message(s)" in dropped[0]
    assert "function" in dropped[0]


def test_build_messages_list_no_user_message_falls_back_empty_with_warning() -> None:
    messages = [{"role": "system", "content": "sys only"}]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {}
    assert built.example.history is None
    assert built.warnings


def test_build_non_messages_list_falls_back_to_existing_behavior() -> None:
    events = [
        _model_call(0, model_input=["opaque", "payload"]),
        _final("done", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {}
    assert built.example.history is None
    assert built.warnings


def test_build_dict_input_unchanged_history_none() -> None:
    events = [
        _model_call(0, model_input={"query": "hello"}),
        _final("done", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"query": "hello"}
    assert built.example.history is None


def test_build_str_input_unchanged_history_none() -> None:
    events = [
        _model_call(0, model_input="hello"),
        _final("done", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"input": "hello"}
    assert built.example.history is None


def test_build_none_input_unchanged_history_none() -> None:
    events = [
        _model_call(0, model_input=None),
        _final("done", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {}
    assert built.example.history is None
    assert built.warnings


# ---------------------------------------------------------------------------
# context-loss warning
# ---------------------------------------------------------------------------


def test_context_loss_warning_fires_on_large_gap() -> None:
    # Recorded 13180 input tokens but only a few chars recovered -> ~5 est tokens.
    events = [
        _messages_call(0, messages=[{"role": "user", "content": "1pm"}], input_tokens=13180),
        _final("ok", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    loss_warnings = [w for w in built.warnings if "input tokens but only" in w]
    assert len(loss_warnings) == 1
    assert "13180 input tokens" in loss_warnings[0]
    assert "system prompt / conversation history was probably not captured" in loss_warnings[0]


def test_context_loss_warning_does_not_fire_when_gap_small() -> None:
    # Ratio >= 4 but the absolute gap is below the min-gap threshold.
    events = [
        _messages_call(
            0,
            messages=[{"role": "user", "content": "x" * 40}],  # ~10 est tokens
            input_tokens=45,  # ratio 4.5, gap only 35 tokens (< 300)
        ),
        _final("ok", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert not any("input tokens but only" in w for w in built.warnings)


def test_context_loss_warning_does_not_fire_when_ratio_below_threshold() -> None:
    events = [
        _messages_call(
            0,
            messages=[{"role": "user", "content": "x" * 4000}],  # ~1000 est tokens
            input_tokens=2000,  # ratio 2, well over the min gap but ratio < 4
        ),
        _final("ok", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert not any("input tokens but only" in w for w in built.warnings)


def test_context_loss_warning_skipped_when_input_tokens_zero() -> None:
    events = [
        _messages_call(0, messages=[{"role": "user", "content": "1pm"}], input_tokens=0),
        _final("ok", 1),
    ]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert not any("input tokens but only" in w for w in built.warnings)


# ---------------------------------------------------------------------------
# empty-ground-truth warning
# ---------------------------------------------------------------------------


def test_empty_ground_truth_warning_fires_when_no_output_and_no_tools() -> None:
    events = [_model_call(0, model_input="hi")]  # no final_output, empty output, no tool calls
    events[0]["output"] = ""
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert any("no scoreable ground truth" in w for w in built.warnings)


def test_empty_ground_truth_warning_absent_when_final_output_present() -> None:
    events = [_model_call(0, model_input="hi"), _final("hello", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert not any("no scoreable ground truth" in w for w in built.warnings)


def test_empty_ground_truth_warning_absent_for_pure_tool_call_turn() -> None:
    events = [_model_call(0, model_input="hi"), _tool_call("search_orders", 1)]
    events[0]["output"] = ""
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert not any("no scoreable ground truth" in w for w in built.warnings)


def test_empty_ground_truth_warning_absent_when_model_call_output_nonempty() -> None:
    events = [_model_call(0, model_input="hi")]  # output defaults to "out" (non-empty)
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert not any("no scoreable ground truth" in w for w in built.warnings)


# ---------------------------------------------------------------------------
# build_conversation_examples — conversation grouping
# ---------------------------------------------------------------------------


def test_conversation_grouping_orders_by_turn_index() -> None:
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[
            _messages_call(0, messages=[{"role": "user", "content": "hi"}]),
            _final("hello, how can I help?", 1),
        ],
    )
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[
            _model_call(0, model_input="where is order 12345?"),
            _final("Your order ships tomorrow.", 1),
        ],
    )
    # Feed out of order to prove ordering is by turn_index, not input order.
    records = [_record(turn1), _record(turn0)]
    opts = PromoteOptions()

    results = build_conversation_examples(records, opts)

    assert [r.envelope.capture_id for r, _ in results] == ["cap_t0", "cap_t1"]
    ex0, ex1 = (b.example for _, b in results)
    assert ex0.conversation_id == "conv_1"
    assert ex0.turn_index == 0
    assert ex1.conversation_id == "conv_1"
    assert ex1.turn_index == 1


def test_conversation_grouping_none_turn_index_falls_back_to_created_at() -> None:
    first = _envelope(
        capture_id="cap_a",
        conversation_id="conv_1",
        turn_index=None,
        created_at="2026-06-16T12:00:00+00:00",
        events=[_model_call(0, model_input="a"), _final("resp a", 1)],
    )
    second = _envelope(
        capture_id="cap_b",
        conversation_id="conv_1",
        turn_index=None,
        created_at="2026-06-16T12:05:00+00:00",
        events=[_model_call(0, model_input="b"), _final("resp b", 1)],
    )
    records = [_record(second), _record(first)]

    results = build_conversation_examples(records, PromoteOptions())

    assert [r.envelope.capture_id for r, _ in results] == ["cap_a", "cap_b"]
    # group ordinal used as turn_index when envelope turn_index is None.
    assert [b.example.turn_index for _, b in results] == [0, 1]


def test_conversation_grouping_reconstructs_history_from_final_output() -> None:
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[
            _model_call(0, model_input="hi"),
            _final("hello, how can I help?", 1),
        ],
    )
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[
            _model_call(0, model_input="where is order 12345?"),
            _final("Your order ships tomorrow.", 1),
        ],
    )
    records = [_record(turn0), _record(turn1)]

    results = build_conversation_examples(records, PromoteOptions())

    _, built1 = results[1]
    assert built1.example.history is not None
    assert [m.role for m in built1.example.history] == ["user", "assistant"]
    assert built1.example.history[0].content == "hi"
    assert built1.example.history[1].content == "hello, how can I help?"
    assert built1.example.inputs == {"input": "where is order 12345?"}


def test_conversation_grouping_reconstructs_history_fallback_to_model_call_output() -> None:
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[_model_call(0, model_input="hi")],  # no final_output; output="out"
    )
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[_model_call(0, model_input="next"), _final("resp", 1)],
    )
    records = [_record(turn0), _record(turn1)]

    results = build_conversation_examples(records, PromoteOptions())

    _, built1 = results[1]
    assert built1.example.history is not None
    assert built1.example.history[1].content == "out"


def test_conversation_grouping_missing_assistant_text_skips_with_warning() -> None:
    no_output_call = _model_call(0, model_input="hi")
    no_output_call["output"] = ""  # no final_output and empty model_call output
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[no_output_call],
    )
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[_model_call(0, model_input="next"), _final("resp", 1)],
    )
    records = [_record(turn0), _record(turn1)]

    results = build_conversation_examples(records, PromoteOptions())

    _, built1 = results[1]
    assert built1.example.history is not None
    # only the user message from turn0 is present; assistant msg skipped.
    assert [m.role for m in built1.example.history] == ["user"]
    assert any("assistant" in w.lower() for w in built1.warnings)


def test_conversation_grouping_own_messages_list_kept_verbatim() -> None:
    """A turn that recorded its own full messages list is the source of truth.

    Even though a prior turn exists in the group, this turn's recovered
    history (from its own recorded messages) must not be overwritten by
    cross-capture reconstruction.
    """
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[_model_call(0, model_input="hi"), _final("hello", 1)],
    )
    turn1_messages = [
        {"role": "system", "content": "custom system prompt"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "next"},
    ]
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[_messages_call(0, messages=turn1_messages), _final("resp", 1)],
    )
    records = [_record(turn0), _record(turn1)]

    results = build_conversation_examples(records, PromoteOptions())

    _, built1 = results[1]
    assert built1.example.history is not None
    assert [m.role for m in built1.example.history] == ["system", "user", "assistant"]
    assert built1.example.history[0].content == "custom system prompt"


def test_conversation_grouping_seeds_system_prompt_from_prior_turn() -> None:
    """Reconstruction seeds history with the system prompt a prior turn recovered."""
    turn0_messages = [
        {"role": "system", "content": "You are a scheduling assistant."},
        {"role": "user", "content": "hi"},
    ]
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[_messages_call(0, messages=turn0_messages), _final("hello", 1)],
    )
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[_model_call(0, model_input="next"), _final("resp", 1)],
    )
    records = [_record(turn0), _record(turn1)]

    results = build_conversation_examples(records, PromoteOptions())

    _, built1 = results[1]
    assert built1.example.history is not None
    assert [m.role for m in built1.example.history] == ["system", "user", "assistant"]
    assert built1.example.history[0].content == "You are a scheduling assistant."
    assert built1.example.history[1].content == "hi"
    assert built1.example.history[2].content == "hello"


def test_conversation_grouping_system_prompt_mismatch_uses_earliest_with_warning() -> None:
    turn0_messages = [
        {"role": "system", "content": "prompt A"},
        {"role": "user", "content": "hi"},
    ]
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[_messages_call(0, messages=turn0_messages), _final("hello", 1)],
    )
    turn1_messages = [
        {"role": "system", "content": "prompt B"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[_messages_call(0, messages=turn1_messages), _final("sure", 1)],
    )
    turn2 = _envelope(
        capture_id="cap_t2",
        conversation_id="conv_1",
        turn_index=2,
        events=[_model_call(0, model_input="next"), _final("resp", 1)],
    )
    records = [_record(turn0), _record(turn1), _record(turn2)]

    results = build_conversation_examples(records, PromoteOptions())

    _, built2 = results[2]
    assert built2.example.history is not None
    assert built2.example.history[0].role == "system"
    assert built2.example.history[0].content == "prompt A"
    mismatch_warnings = [w for w in built2.warnings if "system prompt" in w]
    assert len(mismatch_warnings) == 1
    assert "cap_t0" in mismatch_warnings[0]
    assert "cap_t1" in mismatch_warnings[0]


def test_conversation_grouping_no_prior_system_prompt_reconstructs_without_seed() -> None:
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[_model_call(0, model_input="hi"), _final("hello", 1)],
    )
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[_model_call(0, model_input="next"), _final("resp", 1)],
    )
    records = [_record(turn0), _record(turn1)]

    results = build_conversation_examples(records, PromoteOptions())

    _, built1 = results[1]
    assert built1.example.history is not None
    assert [m.role for m in built1.example.history] == ["user", "assistant"]
    assert not any("system prompt" in w for w in built1.warnings)


def test_context_loss_warning_cleared_when_reconstruction_supplies_prefix() -> None:
    """The pre-reconstruction context-loss warning is re-evaluated with the rebuilt prefix."""
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[_model_call(0, model_input="hi"), _final("y" * 4000, 1)],  # ~1000 est tokens
    )
    turn1_call = _model_call(0, model_input="next")
    turn1_call["input_tokens"] = 2000  # ratio 2 vs the reconstructed prefix -> below threshold
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[turn1_call, _final("resp", 1)],
    )
    # Sanity: promoted alone (no reconstruction), the warning fires.
    alone = build_example_from_capture(turn1, PromoteOptions())
    assert any("input tokens but only" in w for w in alone.warnings)

    results = build_conversation_examples([_record(turn0), _record(turn1)], PromoteOptions())

    _, built1 = results[1]
    assert not any("input tokens but only" in w for w in built1.warnings)


def test_context_loss_warning_persists_when_reconstruction_still_short() -> None:
    turn0 = _envelope(
        capture_id="cap_t0",
        conversation_id="conv_1",
        turn_index=0,
        events=[_model_call(0, model_input="hi"), _final("hello", 1)],
    )
    turn1_call = _model_call(0, model_input="next")
    turn1_call["input_tokens"] = 13180  # still a huge gap over the tiny reconstructed prefix
    turn1 = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=1,
        events=[turn1_call, _final("resp", 1)],
    )

    results = build_conversation_examples([_record(turn0), _record(turn1)], PromoteOptions())

    _, built1 = results[1]
    loss_warnings = [w for w in built1.warnings if "input tokens but only" in w]
    assert len(loss_warnings) == 1


def test_conversation_grouping_none_conversation_id_untouched() -> None:
    standalone = _envelope(capture_id="cap_solo", conversation_id=None)
    records = [_record(standalone)]

    results = build_conversation_examples(records, PromoteOptions())

    assert len(results) == 1
    rec, built = results[0]
    assert rec.envelope.capture_id == "cap_solo"
    assert built.example.conversation_id is None
    assert built.example.turn_index is None


def test_conversation_grouping_single_group_matches_per_capture_build() -> None:
    """Sanity: a lone conversation member should build the same as a direct call."""
    solo = _envelope(
        capture_id="cap_solo",
        conversation_id="conv_solo",
        turn_index=0,
    )
    records = [_record(solo)]

    results = build_conversation_examples(records, PromoteOptions())

    assert len(results) == 1
    _, built = results[0]
    assert built.example.conversation_id == "conv_solo"
    assert built.example.turn_index == 0


# ---------------------------------------------------------------------------
# write_promoted_case + rebuild_golden_jsonl — the disk side
# ---------------------------------------------------------------------------


def _case(name: str, example_id: str, *, suite: str = "support_agent") -> PromotedCase:
    built = build_example_from_capture(
        _envelope(capture_id=example_id, suite=suite),
        PromoteOptions(name=name),
    )
    return PromotedCase(
        name=name,
        suite=suite,
        from_capture=example_id,
        promoted_at="2026-06-16T12:00:00+00:00",
        source_input_hash="hash123",
        code_version="v1",
        example=built.example,
    )


def test_write_promoted_case_writes_under_suite_dir(tmp_path: Path) -> None:
    path = write_promoted_case(_case("case1", "cap_1"), base=tmp_path)
    assert path == tmp_path / "suites" / "support_agent" / "case1.json"
    reloaded = PromotedCase.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded.from_capture == "cap_1"
    assert reloaded.example.id == "case1"


def test_write_promoted_case_refuses_overwrite_without_force(tmp_path: Path) -> None:
    write_promoted_case(_case("case1", "cap_1"), base=tmp_path)
    with pytest.raises(FileExistsError):
        write_promoted_case(_case("case1", "cap_1"), base=tmp_path)
    # force overwrites cleanly
    write_promoted_case(_case("case1", "cap_1"), base=tmp_path, force=True)


def test_rebuild_golden_jsonl_emits_loadable_sorted_index(tmp_path: Path) -> None:
    write_promoted_case(_case("zeta", "cap_z"), base=tmp_path)
    write_promoted_case(_case("alpha", "cap_a"), base=tmp_path)
    suite_dir = tmp_path / "suites" / "support_agent"

    golden = rebuild_golden_jsonl(suite_dir)

    assert golden == suite_dir / "golden.jsonl"
    lines = [
        json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [row["id"] for row in lines] == ["alpha", "zeta"]  # sorted by id
    # The regenerated index is a valid suite the existing loader accepts.
    suite = load_jsonl(golden)
    assert suite.ids() == {"alpha", "zeta"}


def test_rebuild_golden_jsonl_is_deterministic(tmp_path: Path) -> None:
    write_promoted_case(_case("b", "cap_b"), base=tmp_path)
    write_promoted_case(_case("a", "cap_a"), base=tmp_path)
    suite_dir = tmp_path / "suites" / "support_agent"

    first = rebuild_golden_jsonl(suite_dir).read_text(encoding="utf-8")
    second = rebuild_golden_jsonl(suite_dir).read_text(encoding="utf-8")
    assert first == second


# ---------------------------------------------------------------------------
# The frozen multi-round capture — the regression anchor for the agent
# tool-eval fidelity work. A faithful, shortened redaction of the real
# cap_9c97e4dc trace: three model_call rounds, archives in round 1, a
# get_projects in round 2, a text-only round 3. The nested ``tool_args``
# argument shape is preserved verbatim — later phases depend on it.
# ---------------------------------------------------------------------------


def test_multi_round_fixture_validates_as_a_capture_envelope() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    kinds = [e.type for e in env.trace.events]
    assert kinds.count("model_call") == 3
    assert kinds.count("tool_call") == 3
    assert env.conversation_id == "74"
    assert env.turn_index == 12


# ---------------------------------------------------------------------------
# Round-scoped ground truth
# ---------------------------------------------------------------------------


def test_tool_rounds_splits_on_each_model_call() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    rounds = _tool_rounds(env.trace.events)
    assert [[c.name for c in r] for r in rounds] == [
        ["archive_project", "archive_project"],
        ["get_projects"],
    ]


def test_tool_rounds_ignores_the_trailing_text_only_round() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    assert len(_tool_rounds(env.trace.events)) == 2  # not 3


def test_tool_rounds_is_empty_when_no_tools_were_called() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    text_only = [e for e in env.trace.events if e.type == "model_call"]
    assert _tool_rounds(text_only) == []


def test_promote_scopes_expected_tools_to_the_first_round_by_default() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions())
    assert built.example.expected_tools is not None
    assert [t.tool_name for t in built.example.expected_tools] == [
        "archive_project",
        "archive_project",
    ]


def test_promote_records_every_round_regardless_of_scoping() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions())
    assert built.example.expected_tool_rounds is not None
    assert [[t.tool_name for t in r] for r in built.example.expected_tool_rounds] == [
        ["archive_project", "archive_project"],
        ["get_projects"],
    ]


def test_promote_rounds_all_still_scopes_expected_tools_to_round_one() -> None:
    """--rounds all no longer flattens: expected_tools is always round 1.

    The flattened list was a yardstick no replay could meet; the later rounds
    now travel as expected_tool_rounds + tool_result_fixtures instead, scored
    round by round.
    """
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))
    assert built.example.expected_tools is not None
    assert [t.tool_name for t in built.example.expected_tools] == [
        "archive_project",
        "archive_project",
    ]


def test_promote_warns_when_later_rounds_are_dropped() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions())
    assert any("2 agent round(s)" in w for w in built.warnings)


def test_promote_rounds_all_does_not_warn_about_dropped_rounds() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))
    assert not any("agent round(s)" in w for w in built.warnings)


def test_promote_tool_count_counts_the_scoped_calls_only() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions(tool_count=True))
    assert built.example.expected_tool_count == 2


def test_a_single_round_capture_is_unaffected_by_scoping() -> None:
    env = _envelope(
        events=[
            _model_call(0, model_input="where is order 12345?"),
            _tool_call("search_orders", 1),
            _tool_call("send_email", 2),
            _final("done", 3),
        ],
    )
    first = build_example_from_capture(env, PromoteOptions())
    every = build_example_from_capture(env, PromoteOptions(rounds="all"))

    names = ["search_orders", "send_email"]
    assert [t.tool_name for t in first.example.expected_tools or []] == names
    assert [t.tool_name for t in every.example.expected_tools or []] == names
    assert not any("agent round(s)" in w for w in first.warnings)


def test_a_capture_with_no_tools_records_no_rounds() -> None:
    built = build_example_from_capture(
        _envelope(events=[_model_call(0, model_input="hi")]), PromoteOptions()
    )
    assert built.example.expected_tool_rounds is None
    assert built.example.expected_tools is None
    assert built.example.expected_no_tools is True


# ---------------------------------------------------------------------------
# Teacher-forced replay — tool_result_fixtures
#
# --rounds all now carries the recorded tool results alongside
# expected_tool_rounds, so the runner can replay round k with rounds 0..k-1
# fed back verbatim. Alignment is positional and load-time validated (see
# suite/models.py), so everything below is really one assertion: the fixture
# at [k][i] is the result of the call at expected_tool_rounds[k][i].
# ---------------------------------------------------------------------------


def test_rounds_first_writes_no_fixtures() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions())
    assert built.example.tool_result_fixtures is None
    assert built.example.rounds_to_replay() == 1


def test_rounds_all_carries_every_recorded_result_as_a_fixture() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert [[f.tool_name for f in r] for r in fixtures] == [
        ["archive_project", "archive_project"],
        ["get_projects"],
    ]
    assert [f.result for f in fixtures[0]] == [
        {"success": True, "title": "Series A Fundraise", "status": "archived"},
        {"success": True, "title": "Q2 Product Launch", "status": "archived"},
    ]
    assert fixtures[1][0].result == {
        "success": True,
        "projects": [{"id": 57, "title": "evalshift updates"}],
    }
    # Two covered rounds plus the answer round after them.
    assert built.example.rounds_to_replay() == 3


def test_fixtures_pair_by_call_id_not_by_position() -> None:
    """Results recorded out of order still land on the call they answer."""
    env = _envelope(
        events=[
            _model_call(0, model_input="hi"),
            _tool_call("search_orders", 1, call_id="c1"),
            _tool_call("search_orders", 2, call_id="c2"),
            _tool_result("search_orders", 3, result="second", call_id="c2"),
            _tool_result("search_orders", 4, result="first", call_id="c1"),
            _model_call(5, model_input="hi"),
            _final("done", 6),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert [f.result for f in fixtures[0]] == ["first", "second"]


def test_fixtures_fall_back_to_name_within_the_round_when_ids_are_absent() -> None:
    env = _envelope(
        events=[
            _model_call(0, model_input="hi"),
            _tool_call("search_orders", 1, call_id=None),
            _tool_call("send_email", 2, call_id=None),
            _tool_result("send_email", 3, result="sent"),
            _tool_result("search_orders", 4, result="found"),
            _final("done", 5),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert [(f.tool_name, f.result) for f in fixtures[0]] == [
        ("search_orders", "found"),
        ("send_email", "sent"),
    ]


def test_a_name_match_never_reaches_across_a_round_boundary() -> None:
    """Round 2's result must not be conscripted to answer round 1's call."""
    env = _envelope(
        events=[
            _model_call(0, model_input="hi"),
            _tool_call("search_orders", 1, call_id=None),
            _model_call(2, model_input="hi"),
            _tool_call("search_orders", 3, call_id=None),
            _tool_result("search_orders", 4, result="round 2 only"),
            _final("done", 5),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    assert built.example.tool_result_fixtures is None
    assert any(
        "round 1 has 1 tool call(s) with no recorded result; replay stays single-shot" in w
        for w in built.warnings
    )


def test_an_unpaired_call_stops_fixture_coverage_at_that_round() -> None:
    env = _envelope(
        events=[
            _model_call(0, model_input="hi"),
            _tool_call("search_orders", 1),
            _tool_result("search_orders", 2, result="found", call_id="call_search_orders_1"),
            _model_call(3, model_input="hi"),
            _tool_call("send_email", 4),
            _tool_call("send_email", 5),
            _tool_result("send_email", 6, result="sent", call_id="call_send_email_4"),
            _model_call(7, model_input="hi"),
            _final("done", 8),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert [[f.tool_name for f in r] for r in fixtures] == [["search_orders"]]
    assert built.example.rounds_to_replay() == 2
    assert any(
        "round 2 has 1 tool call(s) with no recorded result; replay will cover rounds 1..1" in w
        for w in built.warnings
    )


def test_fixtures_carry_a_recorded_error_verbatim() -> None:
    env = _envelope(
        events=[
            _model_call(0, model_input="hi"),
            _tool_call("search_orders", 1),
            _tool_result("search_orders", 2, error="404 not found", call_id="call_search_orders_1"),
            _final("done", 3),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert fixtures[0][0].error == "404 not found"
    assert fixtures[0][0].result is None


def test_requested_calls_pair_their_results_by_call_id() -> None:
    env = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[
                    _requested("search_orders", call_id="r1"),
                    _requested("search_orders", call_id="r2"),
                ],
            ),
            _tool_call("search_orders", 1, call_id="r1"),
            _tool_call("search_orders", 2, call_id="r2"),
            _tool_result("search_orders", 3, result="second", call_id="r2"),
            _tool_result("search_orders", 4, result="first", call_id="r1"),
            _model_call(5, model_input="hi", requested_tool_calls=[]),
            _final("done", 6),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    assert built.promotion_source == "requested"
    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert [f.result for f in fixtures[0]] == ["first", "second"]


def test_requested_calls_without_ids_pair_by_name_within_the_round() -> None:
    env = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[_requested("search_orders"), _requested("send_email")],
            ),
            _tool_call("search_orders", 1, call_id=None),
            _tool_call("send_email", 2, call_id=None),
            _tool_result("send_email", 3, result="sent"),
            _tool_result("search_orders", 4, result="found"),
            _model_call(5, model_input="hi", requested_tool_calls=[]),
            _final("done", 6),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    assert built.promotion_source == "requested"
    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert [(f.tool_name, f.result) for f in fixtures[0]] == [
        ("search_orders", "found"),
        ("send_email", "sent"),
    ]


def test_tool_count_under_rounds_all_totals_the_replayed_rounds() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions(rounds="all", tool_count=True))
    assert built.example.expected_tool_count == 3


def test_tool_count_under_rounds_all_includes_the_round_that_stopped_coverage() -> None:
    env = _envelope(
        events=[
            _model_call(0, model_input="hi"),
            _tool_call("search_orders", 1),
            _tool_result("search_orders", 2, result="found", call_id="call_search_orders_1"),
            _model_call(3, model_input="hi"),
            _tool_call("send_email", 4, call_id=None),
            _model_call(5, model_input="hi"),
            _tool_call("archive_project", 6, call_id=None),
            _final("done", 7),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all", tool_count=True))

    # Round 1 is covered and round 2 is the answer round the replay reaches;
    # round 3 is never replayed, so its call is not in the denominator.
    assert built.example.rounds_to_replay() == 2
    assert built.example.expected_tool_count == 2


def test_tool_count_under_rounds_first_still_counts_round_one_only() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions(tool_count=True))
    assert built.example.expected_tool_count == 2


def test_a_capture_with_no_toolset_carries_no_fixtures() -> None:
    """The inert placeholder example must stay inert (see the I2 comment)."""
    env = _envelope(
        events=[
            _model_call(0, model_input="hi", toolset_ref=None),
            _tool_call("search_orders", 1),
            _tool_result("search_orders", 2, result="found", call_id="call_search_orders_1"),
            _final("done", 3),
        ],
    )
    built = build_example_from_capture(env, PromoteOptions(rounds="all"))

    assert built.blocked is not None
    assert built.example.tool_result_fixtures is None


def test_conversation_examples_pass_fixtures_through() -> None:
    record = _record(
        _envelope(
            capture_id="cap_conv",
            conversation_id="c1",
            turn_index=0,
            events=[
                _model_call(0, model_input="hi"),
                _tool_call("search_orders", 1),
                _tool_result("search_orders", 2, result="found", call_id="call_search_orders_1"),
                _model_call(3, model_input="hi"),
                _final("done", 4),
            ],
        ),
    )

    [(_, built)] = build_conversation_examples([record], PromoteOptions(rounds="all"))

    fixtures = built.example.tool_result_fixtures
    assert fixtures is not None
    assert [f.result for f in fixtures[0]] == ["found"]


def test_example_content_key_ignores_fixtures() -> None:
    """Two captures with the same replayed content are one case, whatever
    their tools returned."""
    env = load_capture_fixture("multi_round_tools.json")
    first = build_example_from_capture(env, PromoteOptions())
    every = build_example_from_capture(env, PromoteOptions(rounds="all"))

    assert first.example.tool_result_fixtures is None
    assert every.example.tool_result_fixtures is not None
    assert example_content_key(first.example) == example_content_key(every.example)


def test_the_rounds_first_warning_points_at_teacher_forced_replay() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions())

    [warning] = [w for w in built.warnings if "agent round(s)" in w]
    assert "--rounds all" in warning
    assert "teacher-forced" in warning
    assert "flatten" not in warning


# ---------------------------------------------------------------------------
# Promotion hygiene — errored turns, failed tool results, duplicate turns
# ---------------------------------------------------------------------------


def _errored_capture(
    *,
    capture_id: str = "cap_errored",
    conversation_id: str | None = "75",
    turn_index: int | None = 0,
) -> CaptureEnvelope:
    """A capture whose turn died before the agent acted (a Gemini 400)."""
    return _envelope(
        capture_id=capture_id,
        suite="main_chat",
        conversation_id=conversation_id,
        turn_index=turn_index,
        events=[
            _model_call(0, model_input="Can you check if I have any duplicated projects?"),
            _error("400 Bad Request. CachedContent model mismatch.", 1),
        ],
    )


def test_promote_blocks_a_capture_whose_turn_errored() -> None:
    built = build_example_from_capture(_errored_capture(), PromoteOptions())
    assert built.blocked is not None
    assert "error event" in built.blocked


def test_an_errored_capture_never_asserts_expected_no_tools() -> None:
    built = build_example_from_capture(_errored_capture(), PromoteOptions())
    assert built.example.expected_no_tools is False


def test_allow_errored_promotes_but_still_warns() -> None:
    built = build_example_from_capture(
        _errored_capture(),
        PromoteOptions(allow_errored=True),
    )
    assert built.blocked is None
    assert any("error event" in w for w in built.warnings)
    assert built.example.expected_no_tools is False


def test_a_clean_capture_is_never_blocked() -> None:
    built = build_example_from_capture(_envelope(), PromoteOptions())
    assert built.blocked is None
    assert not any("error event" in w for w in built.warnings)


def test_a_blocked_turn_does_not_seed_later_turns_history() -> None:
    """A turn that never ran must not contribute a user message downstream."""
    errored = _record(_errored_capture(capture_id="cap_errored", turn_index=0))
    retry = _record(
        _envelope(
            capture_id="cap_retry",
            suite="main_chat",
            conversation_id="75",
            turn_index=1,
            events=[
                _model_call(0, model_input="Can you check if I have any duplicated projects?"),
                _tool_call("get_projects", 1, args={}),
                _final("You have two duplicates.", 2),
            ],
        ),
    )

    results = {
        r.envelope.capture_id: b
        for r, b in build_conversation_examples([errored, retry], PromoteOptions())
    }
    history = results["cap_retry"].example.history or []
    assert history == []
    assert not any("no recoverable assistant reply" in w for w in results["cap_retry"].warnings)


def test_promote_warns_about_failed_tool_results() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    # Mark the second archive's result as a failure, mirroring the real
    # cap_9c97e4dc trace where archive_project("Home Purchase") 404'd.
    env.trace.events[4].result = {"success": False, "error": "Project not found"}

    built = build_example_from_capture(env, PromoteOptions())
    assert any("1 recorded tool result(s) failed" in w for w in built.warnings)
    assert built.blocked is None  # a warning, not a block


def test_promote_warns_about_a_tool_result_carrying_an_error_field() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    env.trace.events[4].error = "Project not found"

    built = build_example_from_capture(env, PromoteOptions())
    assert any("archive_project" in w and "failed" in w for w in built.warnings)


def test_promote_does_not_warn_when_every_tool_result_succeeded() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions())
    assert not any("failed" in w for w in built.warnings)


def test_promote_reports_conversation_and_turn_for_duplicate_detection() -> None:
    env = load_capture_fixture("multi_round_tools.json")
    built = build_example_from_capture(env, PromoteOptions())
    assert built.example.conversation_id == "74"
    assert built.example.turn_index == 12


def test_sync_warns_when_two_captures_claim_the_same_turn() -> None:
    first = load_capture_fixture("multi_round_tools.json")
    second = CaptureEnvelope.model_validate(
        {**json.loads(MULTI_ROUND_CAPTURE.read_text(encoding="utf-8")), "capture_id": "cap_retry"},
    )
    # Same conversation_id (74) and turn_index (12) on both.

    warnings = duplicate_turn_warnings([first, second])
    assert len(warnings) == 1
    assert "conversation 74 turn 12" in warnings[0]
    assert first.capture_id in warnings[0]
    assert "cap_retry" in warnings[0]


def test_no_duplicate_warning_for_distinct_turns() -> None:
    first = load_capture_fixture("multi_round_tools.json")
    second = CaptureEnvelope.model_validate(
        {
            **json.loads(MULTI_ROUND_CAPTURE.read_text(encoding="utf-8")),
            "capture_id": "cap_other",
            "turn_index": 13,
        },
    )
    assert duplicate_turn_warnings([first, second]) == []


def test_no_duplicate_warning_without_conversation_provenance() -> None:
    a = _envelope(capture_id="cap_a")
    b = _envelope(capture_id="cap_b")
    assert duplicate_turn_warnings([a, b]) == []


# ---------------------------------------------------------------------------
# Argument unwrapping
# ---------------------------------------------------------------------------
#
# A capture SDK that decorates a Python function records that *function's*
# parameters. When the function takes a single dict (``def archive_project(
# tool_args: dict)``), every recorded call looks like
# ``{"tool_args": {"project_name": ...}}`` while the model only ever saw the
# declared schema's flat properties. Ground truth in that shape can never be
# matched by any model, so `tool_arguments` scores 0 for a reason that is not
# the model's fault. Unwrapping is gated on the declared schema: without it,
# or when the schema does not confirm the shape, the recording is left alone.

_ARCHIVE_SCHEMA = {"archive_project": frozenset({"project_name", "reason"})}


def _wrapped_envelope(args: dict[str, Any]) -> CaptureEnvelope:
    return _envelope(
        events=[
            _model_call(0, model_input="yes"),
            _tool_call("archive_project", 1, args=args),
            _final("Archived.", 2),
        ],
    )


def test_wrapper_argument_is_unwrapped_when_the_schema_confirms_it() -> None:
    envelope = _wrapped_envelope({"tool_args": {"project_name": "Series A Fundraise"}})
    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties=_ARCHIVE_SCHEMA),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"project_name": "Series A Fundraise"}
    assert any("tool_args" in w for w in built.warnings)


def test_unwrapping_also_applies_to_expected_tool_rounds() -> None:
    envelope = _wrapped_envelope({"tool_args": {"project_name": "Q2 Product Launch"}})
    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties=_ARCHIVE_SCHEMA),
    )

    rounds = built.example.expected_tool_rounds
    assert rounds is not None
    assert rounds[0][0].arguments == {"project_name": "Q2 Product Launch"}


def test_arguments_are_left_alone_without_a_declared_schema() -> None:
    """No schema means no evidence — never guess at the recorded shape."""
    envelope = _wrapped_envelope({"tool_args": {"project_name": "X"}})
    built = build_example_from_capture(envelope, PromoteOptions())

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"tool_args": {"project_name": "X"}}
    assert not any("tool_args" in w for w in built.warnings)


def test_a_declared_single_dict_parameter_is_not_unwrapped() -> None:
    """``tool_args`` really being a declared property is a legitimate schema."""
    envelope = _wrapped_envelope({"tool_args": {"project_name": "X"}})
    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties={"archive_project": frozenset({"tool_args"})}),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"tool_args": {"project_name": "X"}}


def test_flat_arguments_matching_the_schema_are_untouched() -> None:
    envelope = _wrapped_envelope({"project_name": "X"})
    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties=_ARCHIVE_SCHEMA),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"project_name": "X"}
    assert not any("tool_args" in w for w in built.warnings)


def test_an_inner_dict_the_schema_rejects_is_not_unwrapped() -> None:
    """Unwrapping must produce arguments the tool actually declares."""
    envelope = _wrapped_envelope({"tool_args": {"totally_unknown": 1}})
    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties=_ARCHIVE_SCHEMA),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"tool_args": {"totally_unknown": 1}}


def test_an_empty_wrapper_unwraps_to_empty_arguments() -> None:
    """``get_projects(tool_args={})`` is the no-argument call, not a mismatch."""
    envelope = _envelope(
        events=[
            _model_call(0, model_input="list them"),
            _tool_call("get_projects", 1, args={"tool_args": {}}),
            _final("Here.", 2),
        ],
    )
    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties={"get_projects": frozenset({"status"})}),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {}


def test_an_undeclared_tool_is_left_alone() -> None:
    envelope = _wrapped_envelope({"tool_args": {"project_name": "X"}})
    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties={"some_other_tool": frozenset({"x"})}),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"tool_args": {"project_name": "X"}}


def test_names_only_still_drops_arguments_entirely() -> None:
    envelope = _wrapped_envelope({"tool_args": {"project_name": "X"}})
    built = build_example_from_capture(
        envelope,
        PromoteOptions(names_only=True, tool_properties=_ARCHIVE_SCHEMA),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments is None


def _model_call_with_output(index: int, *, output: Any) -> dict[str, Any]:
    return {
        "type": "model_call",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:00+00:00",
        "metadata": {},
        "model_id": "m",
        "input": "hi",
        "output": output,
        "toolset_ref": "sha256:" + "ab" * 32,
        "tools_offered": ["lookup_customer"],
    }


def test_expected_falls_back_to_the_last_model_call_output() -> None:
    envelope = _envelope(
        events=[
            _model_call_with_output(0, output=""),
            _tool_call("lookup_customer", 1),
            _model_call_with_output(2, output="Your order ships tomorrow."),
        ],
    )
    built = build_example_from_capture(envelope, PromoteOptions())
    assert built.example.expected == {"final_output": "Your order ships tomorrow."}


def test_final_output_event_still_wins_over_the_fallback() -> None:
    envelope = _envelope(
        events=[
            _model_call_with_output(0, output="draft text"),
            _final("Your order ships tomorrow.", 1),
        ],
    )
    built = build_example_from_capture(envelope, PromoteOptions())
    assert built.example.expected == {"final_output": "Your order ships tomorrow."}


def test_no_recoverable_text_warns_instead_of_going_silent() -> None:
    envelope = _envelope(
        events=[
            _model_call_with_output(0, output=""),
            _tool_call("lookup_customer", 1),
        ],
    )
    built = build_example_from_capture(envelope, PromoteOptions())
    assert built.example.expected is None
    assert any("no text ground truth" in w for w in built.warnings)


def test_non_string_model_output_is_not_treated_as_text() -> None:
    envelope = _envelope(events=[_model_call_with_output(0, output={"parts": ["hi"]})])
    built = build_example_from_capture(envelope, PromoteOptions())
    assert built.example.expected is None


class TestGenerationConfig:
    """The first model_call's recorded generation_config rides onto the example."""

    def test_generation_config_copied_from_first_model_call(self) -> None:
        cfg = {"temperature": 0.0, "response_mime_type": "application/json"}
        mc = _model_call(0, model_input="q")
        mc["metadata"] = {"generation_config": cfg}
        envelope = _envelope(events=[mc, _final("ok", 1)])
        built = build_example_from_capture(envelope, PromoteOptions())
        assert built.example.generation_config == cfg

    def test_no_generation_config_key_means_none(self) -> None:
        envelope = _envelope()
        built = build_example_from_capture(envelope, PromoteOptions())
        assert built.example.generation_config is None

    def test_non_dict_or_empty_generation_config_ignored(self) -> None:
        for bad in ("nope", [], {}):
            mc = _model_call(0, model_input="q")
            mc["metadata"] = {"generation_config": bad}
            envelope = _envelope(events=[mc, _final("ok", 1)])
            built = build_example_from_capture(envelope, PromoteOptions())
            assert built.example.generation_config is None

    def test_generation_config_round_trips_through_golden_jsonl(self, tmp_path: Path) -> None:
        cfg = {"response_mime_type": "application/json", "response_schema": {"type": "object"}}
        mc = _model_call(0, model_input="q")
        mc["metadata"] = {"generation_config": cfg}
        envelope = _envelope(events=[mc, _final("ok", 1)])
        built = build_example_from_capture(envelope, PromoteOptions())
        case = PromotedCase(
            name=envelope.capture_id,
            suite=envelope.suite,
            from_capture=envelope.capture_id,
            promoted_at="2026-06-16T12:00:05+00:00",
            source_input_hash=envelope.input_hash,
            code_version=envelope.code_version,
            example=built.example,
        )
        case_path = write_promoted_case(case, base=tmp_path)
        golden = rebuild_golden_jsonl(case_path.parent)
        suite = load_jsonl(golden)
        [loaded] = suite.examples
        assert loaded.generation_config == cfg


def test_build_marks_expected_arguments_as_captured_ground_truth() -> None:
    """Promotion transcribes the source's own call, and says so on the row.

    Nothing here has been checked by a human, so ``against: expected`` scoring
    of these arguments measures target-deviation-from-source; the report
    discloses that on the strength of this field.
    """
    built = build_example_from_capture(_envelope(), PromoteOptions())
    assert all(t.provenance == "captured" for t in built.example.expected_tools or [])
    assert all(
        t.provenance == "captured" for r in built.example.expected_tool_rounds or [] for t in r
    )


# ---------------------------------------------------------------------------
# Requested vs executed tool calls
# ---------------------------------------------------------------------------
#
# A capture records three distinguishable things: the tools OFFERED to the
# model, the calls the model REQUESTED in its response, and the calls the app
# actually EXECUTED. Only the requested list is a yardstick a candidate model
# can be held to -- the executed calls have passed through the app's own
# filtering, re-ordering, and (see the unwrapping section above) its function
# signatures. So when the capture carries requested calls, they are the ground
# truth; a capture that predates the field keeps the executed-call behaviour
# unchanged.


def _requested(
    name: str, args: dict[str, Any] | None = None, *, call_id: str | None = None
) -> dict[str, Any]:
    call: dict[str, Any] = {"name": name, "arguments": args if args is not None else {"q": "x"}}
    if call_id is not None:
        call["call_id"] = call_id
    return call


def test_requested_calls_become_expected_tools_verbatim() -> None:
    envelope = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[_requested("search_orders", {"customer_id": "c42"})],
            ),
            _final("done", 1),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions())

    assert built.promotion_source == "requested"
    assert built.example.expected_tools is not None
    [call] = built.example.expected_tools
    assert call.tool_name == "search_orders"
    assert call.arguments == {"customer_id": "c42"}
    assert call.provenance == "captured"


def test_requested_calls_skip_wrapper_unwrapping() -> None:
    """The model's own arguments need no unwrapping — no wrapper signature saw them.

    The executed-call path unwraps ``{"tool_args": {...}}`` because a decorated
    Python function's parameters stood between the model and the recording.
    Nothing stands between the model and its own requested call, so the same
    shape here is what the model really produced and must be kept verbatim.
    """
    envelope = _envelope(
        events=[
            _model_call(
                0,
                model_input="yes",
                requested_tool_calls=[
                    _requested("archive_project", {"tool_args": {"project_name": "X"}}),
                ],
            ),
            _final("Archived.", 1),
        ],
    )

    built = build_example_from_capture(
        envelope,
        PromoteOptions(tool_properties=_ARCHIVE_SCHEMA),
    )

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"tool_args": {"project_name": "X"}}
    assert not any("tool_args" in w for w in built.warnings)


def test_each_model_call_is_a_round_and_empty_rounds_are_dropped() -> None:
    envelope = _envelope(
        events=[
            _model_call(0, model_input="hi", requested_tool_calls=[_requested("search_orders")]),
            _model_call(1, model_input="hi", requested_tool_calls=[_requested("issue_refund")]),
            # The final, text-producing round requested nothing -- dropped
            # exactly as a tool-less executed round is.
            _model_call(2, model_input="hi", requested_tool_calls=[]),
            _final("done", 3),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions())

    rounds = built.example.expected_tool_rounds
    assert rounds is not None
    assert [[c.tool_name for c in r] for r in rounds] == [["search_orders"], ["issue_refund"]]
    # --rounds first still scopes expected_tools to round 1.
    assert [c.tool_name for c in built.example.expected_tools or []] == ["search_orders"]


def test_requested_rounds_under_rounds_all_keep_expected_tools_at_round_one() -> None:
    envelope = _envelope(
        events=[
            _model_call(0, model_input="hi", requested_tool_calls=[_requested("search_orders")]),
            _model_call(1, model_input="hi", requested_tool_calls=[_requested("issue_refund")]),
            _final("done", 2),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions(rounds="all"))

    assert [c.tool_name for c in built.example.expected_tools or []] == ["search_orders"]
    assert [[c.tool_name for c in r] for r in built.example.expected_tool_rounds or []] == [
        ["search_orders"],
        ["issue_refund"],
    ]


def test_a_model_that_requested_nothing_still_asserts_expected_no_tools() -> None:
    envelope = _envelope(
        events=[
            _model_call(0, model_input="hi", requested_tool_calls=[], tools_offered=["search"]),
            _final("No tool needed.", 1),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions())

    assert built.promotion_source == "requested"
    assert built.example.expected_tools is None
    assert built.example.expected_no_tools is True


def test_requested_calls_count_as_scoreable_ground_truth() -> None:
    """A turn whose model asked for tools is scoreable even if the app ran none."""
    envelope = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[_requested("search_orders")],
            ),
        ],
    )
    envelope.trace.events[0].output = ""

    built = build_example_from_capture(envelope, PromoteOptions())

    assert not any("no scoreable ground truth" in w for w in built.warnings)


def test_absent_requested_calls_keep_the_executed_behaviour() -> None:
    """The default fixture predates the field: nothing about promotion changes."""
    built = build_example_from_capture(_envelope(), PromoteOptions())

    assert built.promotion_source == "executed"
    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].tool_name == "search_orders"
    assert built.example.expected_tools[0].arguments == {"customer_id": "c42"}


def test_a_mixed_capture_falls_back_to_executed_calls_and_warns() -> None:
    """The usual cause is an app that omits the argument on its final, text-only call.

    The SDK records ``None`` for an omitted argument, so a capture goes mixed
    the moment one ``record_model_call`` in the run leaves it off — most often
    the last, answer-producing call. The warning has to name *that* fix
    (``[]``, not omission) rather than send the reader looking for two SDK
    versions installed side by side.
    """
    envelope = _envelope(
        events=[
            _model_call(0, model_input="hi", requested_tool_calls=[_requested("issue_refund")]),
            _tool_call("search_orders", 1, args={"customer_id": "c42"}),
            _model_call(2, model_input="hi"),
            _tool_call("search_orders", 3, args={"customer_id": "c99"}),
            _final("done", 4),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions())

    assert built.promotion_source == "executed"
    assert [c.tool_name for c in built.example.expected_tools or []] == ["search_orders"]
    [warning] = [w for w in built.warnings if "requested_tool_calls" in w]
    assert "1 of 2" in warning
    # Names the real fix: record the field on every call, [] for a round that
    # requested nothing. None means "not recorded", never "nothing requested".
    assert "every model call" in warning
    assert "[]" in warning
    assert "not recorded" in warning
    assert "fell back to the executed tool calls for the whole capture" in warning
    # And no longer sends the reader hunting for a second SDK install.
    assert "evalshift-sdk" not in warning


def test_requested_calls_win_over_disagreeing_executed_calls_and_warn() -> None:
    envelope = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[
                    _requested("search_orders", {"customer_id": "c42"}),
                    _requested("issue_refund", {"ticket_id": "T-1"}),
                ],
            ),
            # The app ran only one of the two the model asked for.
            _tool_call("search_orders", 1, args={"customer_id": "c42"}),
            _final("done", 2),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions())

    assert built.promotion_source == "requested"
    assert [c.tool_name for c in built.example.expected_tools or []] == [
        "search_orders",
        "issue_refund",
    ]
    [disagreement] = [w for w in built.warnings if "issue_refund" in w and "search_orders" in w]
    assert "requested" in disagreement and "executed" in disagreement


def test_matching_requested_and_executed_calls_emit_no_disagreement_warning() -> None:
    envelope = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[_requested("search_orders", {"customer_id": "c42"})],
            ),
            _tool_call("search_orders", 1, args={"customer_id": "c42"}),
            _final("done", 2),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions())

    assert built.promotion_source == "requested"
    assert not any("disagree" in w for w in built.warnings)


def test_differing_arguments_alone_still_warn() -> None:
    envelope = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[_requested("search_orders", {"customer_id": "c42"})],
            ),
            _tool_call("search_orders", 1, args={"customer_id": "REDACTED"}),
            _final("done", 2),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions())

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments == {"customer_id": "c42"}
    assert any("search_orders" in w and "argument" in w for w in built.warnings)


def test_names_only_drops_requested_arguments_too() -> None:
    envelope = _envelope(
        events=[
            _model_call(
                0,
                model_input="hi",
                requested_tool_calls=[_requested("search_orders", {"customer_id": "c42"})],
            ),
            _final("done", 1),
        ],
    )

    built = build_example_from_capture(envelope, PromoteOptions(names_only=True))

    assert built.example.expected_tools is not None
    assert built.example.expected_tools[0].arguments is None


def test_conversation_grouping_carries_the_promotion_source() -> None:
    envelope = _envelope(
        capture_id="cap_t1",
        conversation_id="conv_1",
        turn_index=0,
        events=[
            _model_call(0, model_input="hi", requested_tool_calls=[_requested("search_orders")]),
            _final("done", 1),
        ],
    )

    [(_, built)] = build_conversation_examples([_record(envelope)], PromoteOptions())

    assert built.promotion_source == "requested"


# ---------------------------------------------------------------------------
# Cost at promotion
# ---------------------------------------------------------------------------
# The SDK never prices anything: ``model_call.cost_usd`` is 0.0 unless the
# app's own instrumentation set it, and the provider client wrappers record
# tokens but leave cost at 0 by design. Pricing is the CLI's job, at promote
# time, from litellm's price table. A model litellm cannot price (local /
# self-hosted) is the normal case for open-source models, not a failure: it
# stays at 0 with no tag and no warning.


def _litellm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """The number litellm itself puts on these tokens -- never a hardcoded dollar figure."""
    in_cost, out_cost = litellm.cost_per_token(
        model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
    )
    return float(in_cost) + float(out_cost)


class TestCostAtPromotion:
    def test_tokens_without_cost_are_priced_from_litellm(self) -> None:
        events = [
            _model_call(
                0, model_input="q", model_id="gpt-4o-mini", input_tokens=1200, output_tokens=300
            ),
            _final("done", 1),
        ]

        built = build_example_from_capture(_envelope(events=events), PromoteOptions())

        expected = _litellm_cost("gpt-4o-mini", 1200, 300)
        assert expected > 0
        assert built.cost_usd == pytest.approx(expected)
        assert built.cost_source == "estimated"

    def test_unpriced_model_stays_at_zero_with_no_tag_and_no_noise(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # litellm's pricer opens a socket to a local Ollama daemon for
        # ``ollama/...`` ids and prints a provider banner for bare unknown
        # ones -- an unpriced model must never reach it at all.
        def _never(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
            raise AssertionError("litellm.cost_per_token must not be called for an unpriced model")

        monkeypatch.setattr(litellm, "cost_per_token", _never)
        events = [
            _model_call(
                0, model_input="q", model_id="llama3.1:8b", input_tokens=900, output_tokens=200
            ),
            _final("done", 1),
        ]

        with caplog.at_level(logging.DEBUG):
            built = build_example_from_capture(_envelope(events=events), PromoteOptions())

        assert built.cost_usd == 0.0
        assert built.cost_source is None
        assert built.blocked is None
        assert not [w for w in built.warnings if "cost" in w.lower()]
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_recorded_cost_is_left_untouched(self) -> None:
        events = [
            _model_call(
                0,
                model_input="q",
                model_id="gpt-4o-mini",
                input_tokens=1200,
                output_tokens=300,
                cost_usd=0.0123,
            ),
            _final("done", 1),
        ]

        built = build_example_from_capture(_envelope(events=events), PromoteOptions())

        assert built.cost_usd == 0.0123
        assert built.cost_source == "recorded"

    def test_zero_tokens_and_zero_cost_estimate_nothing(self) -> None:
        events = [_model_call(0, model_input="q", model_id="gpt-4o-mini"), _final("done", 1)]

        built = build_example_from_capture(_envelope(events=events), PromoteOptions())

        assert built.cost_usd == 0.0
        assert built.cost_source is None

    def test_several_model_calls_are_summed_per_event(self) -> None:
        # Each event is priced by its own model_id; a recorded cost is kept
        # as-is; an unpriced event contributes 0; one estimated event makes
        # the whole figure "estimated".
        events = [
            _model_call(
                0, model_input="q", model_id="gpt-4o-mini", input_tokens=1000, output_tokens=100
            ),
            _tool_call("search_orders", 1),
            _model_call(
                2, model_input="q", model_id="m", input_tokens=500, output_tokens=50, cost_usd=0.01
            ),
            _model_call(
                3, model_input="q", model_id="llama3.1:8b", input_tokens=400, output_tokens=40
            ),
            _final("done", 4),
        ]

        built = build_example_from_capture(_envelope(events=events), PromoteOptions())

        assert built.cost_usd == pytest.approx(_litellm_cost("gpt-4o-mini", 1000, 100) + 0.01)
        assert built.cost_source == "estimated"

    def test_recorded_plus_unpriced_is_tagged_recorded(self) -> None:
        events = [
            _model_call(
                0, model_input="q", model_id="m", input_tokens=500, output_tokens=50, cost_usd=0.02
            ),
            _tool_call("search_orders", 1),
            _model_call(
                2, model_input="q", model_id="llama3.1:8b", input_tokens=400, output_tokens=40
            ),
            _final("done", 3),
        ]

        built = build_example_from_capture(_envelope(events=events), PromoteOptions())

        assert built.cost_usd == 0.02
        assert built.cost_source == "recorded"

    def test_cost_round_trips_through_the_promoted_case_file(self, tmp_path: Path) -> None:
        events = [
            _model_call(
                0, model_input="q", model_id="gpt-4o-mini", input_tokens=1200, output_tokens=300
            ),
            _final("done", 1),
        ]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        case = PromotedCase(
            name=built.example.id,
            suite="support_agent",
            from_capture="cap_abc",
            cost_usd=built.cost_usd,
            cost_source=built.cost_source,
            example=built.example,
        )

        path = write_promoted_case(case, base=tmp_path)
        reloaded = PromotedCase.model_validate_json(path.read_text(encoding="utf-8"))

        assert reloaded.cost_usd == pytest.approx(built.cost_usd)
        assert reloaded.cost_source == "estimated"
        # The run-facing example never carries the capture's cost: it is
        # provenance of the recorded run, not something a replay reproduces.
        assert "cost_usd" not in built.example.model_dump()

    def test_promoted_case_written_before_cost_fields_still_parses(self) -> None:
        legacy = json.loads(_case("legacy", "ex_1").model_dump_json())
        legacy.pop("cost_usd")
        legacy.pop("cost_source")

        case = PromotedCase.model_validate(legacy)

        assert case.cost_usd == 0.0
        assert case.cost_source is None

    def test_conversation_grouping_carries_the_cost(self) -> None:
        envelope = _envelope(
            capture_id="cap_t1",
            conversation_id="conv_1",
            turn_index=0,
            events=[
                _model_call(
                    0,
                    model_input="hi",
                    model_id="gpt-4o-mini",
                    input_tokens=1200,
                    output_tokens=300,
                ),
                _final("done", 1),
            ],
        )

        [(_, built)] = build_conversation_examples([_record(envelope)], PromoteOptions())

        assert built.cost_usd == pytest.approx(_litellm_cost("gpt-4o-mini", 1200, 300))
        assert built.cost_source == "estimated"


# ---------------------------------------------------------------------------
# provider-native message shapes
# ---------------------------------------------------------------------------

_CLAUDE_FOLLOW_UP: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "search", "signature": "sig_abc"},
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

_IMAGE_TURN: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
            {"type": "text", "text": "What is in this screenshot?"},
        ],
    }
]


def test_claude_blocks_become_tool_calls_and_tool_messages_in_history() -> None:
    events = [_messages_call(0, messages=_CLAUDE_FOLLOW_UP), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is None
    assert built.example.inputs == {"input": "And the price?"}
    history = built.example.history
    assert history is not None
    assert [m.role for m in history] == ["system", "user", "assistant", "tool", "assistant"]
    assert history[2].content == "Let me look."
    assert history[2].tool_calls == [
        HistoryToolCall(id="toolu_1", name="search", arguments={"q": "evalshift"})
    ]
    assert history[3].tool_call_id == "toolu_1"
    assert history[3].content == "results"
    assert not any("'type'" in m.content or "sig_abc" in m.content for m in history)


def test_openai_chat_history_with_null_content_no_longer_fails() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res"},
        {"role": "assistant", "content": "a1", "refusal": None, "tool_calls": None},
        {"role": "user", "content": "q2"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"input": "q2"}
    history = built.example.history
    assert history is not None
    assert history[2].tool_calls == [
        HistoryToolCall(id="call_1", name="search", arguments={"q": "x"})
    ]
    assert history[3].tool_call_id == "call_1"


def test_an_assistant_message_without_a_content_key_still_recovers_inputs() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "s", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res"},
        {"role": "user", "content": "q2"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"input": "q2"}
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["user", "assistant", "tool"]


def test_normalizer_warnings_reach_the_built_example() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "not json"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res"},
        {"role": "user", "content": "q2"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is None
    assert (
        "recorded message 2: tool call `search` (call_1) had arguments that are not a JSON "
        "object; replayed with {}"
    ) in built.warnings
    assert built.example.history is not None
    assert built.example.history[1].tool_calls == [
        HistoryToolCall(id="call_1", name="search", arguments={})
    ]


def test_langchain_roles_recover_inputs() -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "human", "content": "hi"}]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"input": "hi"}
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["system"]


def test_non_text_content_blocks_the_capture() -> None:
    events = [_messages_call(0, messages=_IMAGE_TURN), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked_reason == "unreplayable_content"
    assert built.blocked == (
        "capture 'cap_abc' carries content a text replay can't carry — recorded message 1 "
        "(user) carries a `image` block. A replay without it is not ground truth; re-capture "
        "without it or hand-write the case."
    )


def test_unreplayable_content_counts_the_rest() -> None:
    turn = [
        {"role": "user", "content": [{"type": "image"}, {"type": "document"}, {"type": "file"}]},
    ]
    events = [_messages_call(0, messages=turn), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is not None
    assert "carries a `image` block (+2 more)." in built.blocked


def test_unreplayable_content_wins_over_errored_even_with_allow_errored() -> None:
    events = [_messages_call(0, messages=_IMAGE_TURN), _error("boom", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions(allow_errored=True))
    assert built.blocked_reason == "unreplayable_content"


def test_history_message_refuses_non_string_content() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        _history_message(
            {"role": "assistant", "content": [{"type": "text", "text": "x"}]}, "assistant"
        )


# ---------------------------------------------------------------------------
# system and developer messages: one leading system prompt per replay
# ---------------------------------------------------------------------------

_LATE_SYSTEM_WARNING = (
    "dropped 1 system message(s) after the first turn — a replay carries one leading system prompt"
)


def test_a_developer_message_merges_into_the_leading_system_prompt() -> None:
    """An OpenAI Responses call with ``instructions=`` plus a developer input."""
    messages = [
        {"role": "system", "content": "instructions text"},
        {"role": "developer", "content": "Format answers as JSON."},
        {"role": "user", "content": "And the price?"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is None
    assert built.example.inputs == {"input": "And the price?"}
    assert built.example.history is not None
    assert [(m.role, m.content) for m in built.example.history] == [
        ("system", "instructions text\nFormat answers as JSON.")
    ]
    assert not any("system message(s)" in w for w in built.warnings)


def test_a_developer_message_after_the_first_turn_is_dropped_with_a_warning() -> None:
    messages = [
        {"role": "user", "content": "What is evalshift?"},
        {"role": "assistant", "content": "A regression tool."},
        {"role": "developer", "content": "Answer in one sentence from now on."},
        {"role": "user", "content": "And the price?"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is None
    assert built.example.history is not None
    assert [(m.role, m.content) for m in built.example.history] == [
        ("user", "What is evalshift?"),
        ("assistant", "A regression tool."),
    ]
    assert _LATE_SYSTEM_WARNING in built.warnings


def test_a_langchain_system_message_after_the_first_turn_is_dropped_with_a_warning() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "human", "content": "q1"},
        {"role": "ai", "content": "a1"},
        {"role": "system", "content": "be brief"},
        {"role": "human", "content": "q2"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is None
    assert built.example.inputs == {"input": "q2"}
    assert built.example.history is not None
    assert [(m.role, m.content) for m in built.example.history] == [
        ("system", "sys"),
        ("user", "q1"),
        ("assistant", "a1"),
    ]
    assert built.warnings.count(_LATE_SYSTEM_WARNING) == 1


def test_late_system_messages_are_counted_in_one_warning() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "developer", "content": "d1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "s2"},
        {"role": "user", "content": "q2"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["system", "user", "assistant"]
    late = [w for w in built.warnings if "after the first turn" in w]
    assert late == [
        "dropped 2 system message(s) after the first turn — a replay carries one leading "
        "system prompt"
    ]


# ---------------------------------------------------------------------------
# invalid_case: the safety net under SuiteExample's validators
# ---------------------------------------------------------------------------


def _empty_toolset_envelope_with_a_tool_call() -> CaptureEnvelope:
    """A capture whose model was offered no tools, yet the app recorded a tool call.

    ``SuiteExample`` rejects tool-call ground truth beside the empty toolset's fingerprint,
    which no promotion rule anticipates -- the case the safety net exists for.
    """
    toolsets = Path(os.environ["EVALSHIFT_DIR"]) / "toolsets"
    hex_ref = EMPTY_TOOLSET_FINGERPRINT.removeprefix("sha256:")
    (toolsets / f"{hex_ref}.json").write_text('{"tools": []}', encoding="utf-8")
    events = [
        _model_call(0, model_input="hi", toolset_ref=EMPTY_TOOLSET_FINGERPRINT, tools_offered=[]),
        _tool_call("search_orders", 1),
        _final("done", 2),
    ]
    return _envelope(events=events)


def test_a_capture_that_cannot_form_a_valid_case_is_blocked_as_invalid_case() -> None:
    built = build_example_from_capture(_empty_toolset_envelope_with_a_tool_call(), PromoteOptions())
    assert built.blocked_reason == "invalid_case"
    assert built.blocked == (
        "capture 'cap_abc' could not be turned into a valid case — Value error, "
        "toolset_ref=<empty toolset> (no tools offered) is incompatible with a non-empty "
        "expected_tools list. Hand-write the case or report this capture."
    )
    placeholder = built.example
    assert placeholder.id == "cap_abc"
    assert placeholder.inputs == {}
    assert placeholder.tags == ["captured", "support_agent"]
    assert placeholder.tools == []
    assert placeholder.toolset_ref is None
    assert placeholder.expected_tools is None


def test_allow_errored_does_not_rescue_an_invalid_case() -> None:
    built = build_example_from_capture(
        _empty_toolset_envelope_with_a_tool_call(), PromoteOptions(allow_errored=True)
    )
    assert built.blocked_reason == "invalid_case"


def test_an_invalid_case_in_a_conversation_does_not_seed_later_turns() -> None:
    bad = _empty_toolset_envelope_with_a_tool_call().model_copy(
        update={"conversation_id": "conv", "turn_index": 0}
    )
    later = _envelope(
        capture_id="cap_t1",
        events=[_model_call(0, model_input="q2"), _final("a2", 1)],
        conversation_id="conv",
        turn_index=1,
    )
    pairs = build_conversation_examples([_record(bad), _record(later)], PromoteOptions())
    by_id = {record.envelope.capture_id: built for record, built in pairs}
    assert by_id["cap_abc"].blocked_reason == "invalid_case"
    assert by_id["cap_t1"].blocked is None
    assert by_id["cap_t1"].example.history == []


class TestCaptureProvenance:
    def test_reads_created_at_and_the_first_model_calls_model(self) -> None:
        envelope = _envelope(
            events=[
                _model_call(0, model_input="hi", model_id="claude-sonnet-5-20260301"),
                _model_call(1, model_input="again", model_id="claude-haiku-5"),
                _final("done", 2),
            ],
            created_at="2026-10-01T12:00:00+00:00",
        )
        assert capture_provenance(envelope) == (
            "2026-10-01T12:00:00+00:00",
            "claude-sonnet-5-20260301",
        )

    def test_a_capture_without_a_model_call_has_no_model(self) -> None:
        assert capture_provenance(_envelope(events=[_final("done", 0)]))[1] == ""


class TestRetirePromotedCase:
    def test_deletes_the_case_file(self, tmp_path: Path) -> None:
        case = PromotedCase(
            name="cap_1",
            suite="alpha",
            from_capture="cap_1",
            example=SuiteExample(id="cap_1", tools=[]),
        )
        path = write_promoted_case(case, base=tmp_path)
        assert retire_promoted_case("alpha", "cap_1", base=tmp_path) is True
        assert not path.exists()

    def test_an_absent_case_is_not_an_error(self, tmp_path: Path) -> None:
        assert retire_promoted_case("alpha", "cap_nope", base=tmp_path) is False
