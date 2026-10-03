"""Unit tests for :mod:`evalshift_cli.suite.models`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evalshift_cli.captures.toolset import EMPTY_TOOLSET_FINGERPRINT
from evalshift_cli.evaluators.tool_models import ToolSpec
from evalshift_cli.suite.models import (
    ChatMessage,
    ExpectedToolCall,
    HistoryToolCall,
    Suite,
    SuiteExample,
    ToolResultFixture,
)

_TOOLSET_REF = "sha256:" + "ab" * 32


def _ex(**kw: Any) -> SuiteExample:
    """Build a ``SuiteExample``, defaulting ``tools=[]`` when the test doesn't care.

    ``toolset_ref`` / ``tools`` are exactly-one-of and required (V7 of
    ``PER_CALL_TOOLSET_CAPTURE_PLAN.md``); most of the tests in this file are
    about some other field entirely, so this fills in the trivial "no tools
    offered" toolset unless the caller specifies one explicitly. Tests that
    ARE about the toolset fields (``TestToolsetFields`` below) construct
    ``SuiteExample`` directly instead of going through this helper.
    """
    if "tools" not in kw and "toolset_ref" not in kw:
        kw["tools"] = []
    return SuiteExample(**kw)


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_valid_roles(self) -> None:
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="hi")
            assert msg.role == role
            assert msg.content == "hi"

    def test_content_defaults_empty(self) -> None:
        msg = ChatMessage(role="user")
        assert msg.content == ""

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage.model_validate({"role": "function", "content": "x"})

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ChatMessage.model_validate({"role": "user", "content": "hi", "rogue_key": True})

    def test_history_assistant_message_can_carry_tool_calls(self) -> None:
        msg = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[HistoryToolCall(id="call_a", name="get_projects", arguments={})],
        )
        assert ChatMessage.model_validate(msg.model_dump()) == msg

    def test_history_tool_message_requires_a_call_id(self) -> None:
        with pytest.raises(ValidationError, match="tool_call_id"):
            ChatMessage(role="tool", content="{}")

    def test_tool_call_id_is_rejected_off_a_tool_message(self) -> None:
        with pytest.raises(ValidationError, match="tool_call_id"):
            ChatMessage(role="assistant", content="hi", tool_call_id="c1")

    def test_only_assistant_messages_may_carry_tool_calls(self) -> None:
        with pytest.raises(ValidationError, match="tool_calls"):
            ChatMessage(
                role="user",
                content="hi",
                tool_calls=[HistoryToolCall(name="x", arguments={})],
            )

    def test_tool_call_name_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            HistoryToolCall(name="", arguments={})


# ---------------------------------------------------------------------------
# SuiteExample
# ---------------------------------------------------------------------------


class TestSuiteExample:
    def test_minimal_construction(self) -> None:
        ex = _ex(id="ex1")
        assert ex.id == "ex1"
        assert ex.inputs == {}
        assert ex.tags == []
        assert ex.expected is None

    def test_with_all_fields(self) -> None:
        ex = _ex(
            id="ex1",
            inputs={"name": "Alex", "tone": "formal"},
            tags=["formal", "english"],
            expected={"summary": "Hi Alex"},
        )
        assert ex.inputs["name"] == "Alex"
        assert "english" in ex.tags
        assert ex.expected == {"summary": "Hi Alex"}

    def test_empty_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            _ex(id="")

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SuiteExample.model_validate(
                {"id": "ex1", "inputs": {}, "rogue_key": True},
            )

    def test_inputs_accept_arbitrary_json_values(self) -> None:
        ex = _ex(
            id="ex1",
            inputs={
                "string": "x",
                "number": 1.5,
                "bool": True,
                "null": None,
                "list": [1, 2, 3],
                "object": {"nested": "yes"},
            },
        )
        assert ex.inputs["object"]["nested"] == "yes"
        assert ex.inputs["list"] == [1, 2, 3]

    def test_equality(self) -> None:
        a = _ex(id="ex1", inputs={"k": 1}, tags=["t"])
        b = _ex(id="ex1", inputs={"k": 1}, tags=["t"])
        c = _ex(id="ex2", inputs={"k": 1}, tags=["t"])
        assert a == b
        assert a != c

    def test_single_turn_example_without_new_fields_still_parses(self) -> None:
        """Back-compat: pre-multi-turn suite rows must still load unchanged.

        Still needs a toolset (``tools``/``toolset_ref`` are an unrelated,
        orthogonal requirement -- see ``TestToolsetFields`` below) -- this
        test is only about the multi-turn fields staying optional.
        """
        ex = SuiteExample.model_validate(
            {"id": "ex1", "inputs": {"name": "Alex"}, "tags": ["formal"], "tools": []},
        )
        assert ex.history is None
        assert ex.conversation_id is None
        assert ex.turn_index is None

    def test_history_round_trips(self) -> None:
        history = [
            ChatMessage(role="system", content="Be terse."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello."),
        ]
        ex = _ex(
            id="ex1",
            history=history,
            conversation_id="conv_1",
            turn_index=2,
        )
        assert ex.history == history
        assert ex.conversation_id == "conv_1"
        assert ex.turn_index == 2
        recreated = SuiteExample.model_validate(ex.model_dump())
        assert recreated == ex

    def test_history_round_trips_a_full_tool_exchange(self) -> None:
        ex = _ex(
            id="x",
            history=[
                ChatMessage(role="system", content="You are Kaila."),
                ChatMessage(role="user", content="list my projects"),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[HistoryToolCall(id="c1", name="get_projects", arguments={})],
                ),
                ChatMessage(role="tool", tool_call_id="c1", content='{"projects": []}'),
                ChatMessage(role="assistant", content="You have no projects."),
            ],
        )
        assert SuiteExample.model_validate(ex.model_dump()) == ex

    def test_history_empty_list_allowed_and_not_normalised(self) -> None:
        ex = _ex(id="ex1", history=[])
        assert ex.history == []
        assert ex.history is not None

    def test_history_system_message_must_be_first(self) -> None:
        with pytest.raises(ValidationError, match="system"):
            _ex(
                id="ex1",
                history=[
                    ChatMessage(role="user", content="Hi"),
                    ChatMessage(role="system", content="Be terse."),
                ],
            )

    def test_history_rejects_two_system_messages(self) -> None:
        with pytest.raises(ValidationError, match="system"):
            _ex(
                id="ex1",
                history=[
                    ChatMessage(role="system", content="Be terse."),
                    ChatMessage(role="system", content="Also be nice."),
                ],
            )

    def test_history_zero_system_messages_ok(self) -> None:
        ex = _ex(
            id="ex1",
            history=[
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello."),
            ],
        )
        assert len(ex.history or []) == 2

    def test_history_system_first_of_multiple_is_ok(self) -> None:
        ex = _ex(
            id="ex1",
            history=[
                ChatMessage(role="system", content="Be terse."),
                ChatMessage(role="user", content="Hi"),
                ChatMessage(role="assistant", content="Hello."),
            ],
        )
        assert ex.history is not None
        assert ex.history[0].role == "system"

    def test_turn_index_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _ex(id="ex1", turn_index=-1)

    def test_turn_index_zero_allowed(self) -> None:
        ex = _ex(id="ex1", turn_index=0)
        assert ex.turn_index == 0

    def test_unknown_key_still_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SuiteExample.model_validate({"id": "ex1", "rogue_key": True})

    def test_expected_tool_rounds_defaults_to_none_and_round_shape_round_trips(self) -> None:
        assert _ex(id="a").expected_tool_rounds is None

        ex = _ex(
            id="b",
            # Not tools=[] (the _ex default): this row asserts non-empty tool
            # ground truth below, now correctly rejected against an empty
            # toolset (I2) -- toolset_ref is the neutral choice since this
            # test is about expected_tool_rounds' shape, not toolset fields.
            toolset_ref=_TOOLSET_REF,
            expected_tools=[ExpectedToolCall(tool_name="x")],
            expected_tool_rounds=[
                [ExpectedToolCall(tool_name="x")],
                [ExpectedToolCall(tool_name="y")],
            ],
        )
        assert SuiteExample.model_validate(ex.model_dump()) == ex

    def test_expected_no_tools_rejects_expected_tool_rounds(self) -> None:
        with pytest.raises(ValidationError, match="expected_no_tools"):
            _ex(
                id="c",
                expected_no_tools=True,
                expected_tool_rounds=[[ExpectedToolCall(tool_name="x")]],
            )


# ---------------------------------------------------------------------------
# SuiteExample.toolset_ref / .tools -- exactly-one-of (V7)
#
# Every model call records the toolset it was offered; a suite example is the
# promoted or hand-authored record of one such call, so it must carry that
# toolset too. Two spellings exist because there are two genuinely different
# authors: `capture sync` (a deduplicating machine) writes a content-addressed
# `toolset_ref`; a person hand-editing JSONL writes inline `tools`. Neither
# present is a load error, not a default -- a suite that forgot its toolset
# must fail immediately and visibly, same as an empty `id` would.
# ---------------------------------------------------------------------------


class TestToolsetFields:
    def test_toolset_ref_alone_is_valid(self) -> None:
        ex = SuiteExample(id="ex1", toolset_ref=_TOOLSET_REF)
        assert ex.toolset_ref == _TOOLSET_REF
        assert ex.tools is None

    def test_tools_alone_is_valid(self) -> None:
        tools = [ToolSpec(name="search_orders", description="Look up orders.", input_schema={})]
        ex = SuiteExample(id="ex1", tools=tools)
        assert ex.tools == tools
        assert ex.toolset_ref is None

    def test_empty_tools_list_alone_is_valid(self) -> None:
        """The empty toolset is a first-class value -- 'no tools were offered' is a real
        assertion, not an absence."""
        ex = SuiteExample(id="ex1", tools=[])
        assert ex.tools == []

    def test_neither_field_fails_to_load(self) -> None:
        """Forgetting the toolset is a load error, not a silent default."""
        with pytest.raises(ValidationError, match=r"toolset_ref|tools"):
            SuiteExample(id="ex1")

    def test_both_fields_present_is_also_rejected(self) -> None:
        """Exactly one, not at-most-one: the two spellings are mutually exclusive."""
        with pytest.raises(ValidationError, match=r"toolset_ref|tools"):
            SuiteExample(id="ex1", toolset_ref=_TOOLSET_REF, tools=[])

    def test_round_trips_through_dump(self) -> None:
        ex = SuiteExample(id="ex1", toolset_ref=_TOOLSET_REF)
        assert SuiteExample.model_validate(ex.model_dump()) == ex


# ---------------------------------------------------------------------------
# I2: `tools=[]` (inline "no tools offered") rejects tool-call ground truth.
#
# `_check_tool_expectations_consistent` already rejected expected_no_tools=True
# paired with expected_tools/expected_tool_count/expected_tool_rounds, but
# accepted tools=[] paired with the same -- the hand-authored mirror of the
# motivating bug (a suite asserting a toolset it did not have).
#
# A `toolset_ref` pointing at an arbitrary sidecar can't be inspected here --
# resolving what it actually contains needs disk I/O, which this load-time
# pydantic validator deliberately does not do (that's `load_toolset`'s job,
# checked at resolution time instead -- see captures/reader.py). But the
# EMPTY toolset is the one exception: it has exactly one possible
# fingerprint (a property of the hashing algorithm, not of any one sidecar),
# so a `toolset_ref` equal to that fingerprint is exactly as inspectable,
# with zero I/O, as `tools == []` is -- and is now checked identically.
# ---------------------------------------------------------------------------


class TestEmptyToolsetRejectsToolGroundTruth:
    def test_rejects_nonempty_expected_tools(self) -> None:
        with pytest.raises(ValidationError, match="no tools offered"):
            SuiteExample(id="ex1", tools=[], expected_tools=[ExpectedToolCall(tool_name="x")])

    def test_rejects_positive_expected_tool_count(self) -> None:
        with pytest.raises(ValidationError, match="no tools offered"):
            SuiteExample(id="ex1", tools=[], expected_tool_count=3)

    def test_allows_zero_expected_tool_count(self) -> None:
        """0 is not ground truth for a tool call -- it's consistent with never
        calling anything, same as expected_no_tools's own ``!= 0`` check."""
        ex = SuiteExample(id="ex1", tools=[], expected_tool_count=0)
        assert ex.expected_tool_count == 0

    def test_rejects_nonempty_expected_tool_rounds(self) -> None:
        with pytest.raises(ValidationError, match="no tools offered"):
            SuiteExample(
                id="ex1",
                tools=[],
                expected_tool_rounds=[[ExpectedToolCall(tool_name="x")]],
            )

    def test_a_populated_inline_toolset_is_unaffected(self) -> None:
        """Positive control: only the EMPTY inline toolset is incompatible --
        a real one may still carry expected_tools normally."""
        tools = [ToolSpec(name="x", description="d", input_schema={})]
        ex = SuiteExample(id="ex1", tools=tools, expected_tools=[ExpectedToolCall(tool_name="x")])
        assert ex.expected_tools is not None

    def test_a_toolset_ref_naming_an_unknown_toolset_is_unaffected_by_this_check(self) -> None:
        """The documented asymmetry: an arbitrary toolset_ref can't be
        inspected at load time (resolving what it actually contains needs
        disk I/O), so this validator lets it through -- even one that will
        eventually turn out to resolve to zero tools once loaded. Only the
        ONE ref value this validator can know without I/O -- the empty
        toolset's fingerprint -- is checked (see the class below)."""
        ex = SuiteExample(
            id="ex1",
            toolset_ref=_TOOLSET_REF,
            expected_tools=[ExpectedToolCall(tool_name="x")],
        )
        assert ex.expected_tools is not None


class TestEmptyToolsetRefRejectsToolGroundTruth:
    """The ``toolset_ref`` mirror of ``TestEmptyToolsetRejectsToolGroundTruth`` above.

    The empty toolset's fingerprint is a fixed, known constant (a property of
    the hashing algorithm itself -- :data:`~evalshift_cli.captures.toolset.EMPTY_TOOLSET_FINGERPRINT`),
    so a ``toolset_ref`` equal to it asserts exactly what inline ``tools=[]``
    asserts, with no sidecar I/O needed to know that. Before this fix, this
    spelling of "no tools offered" escaped `_check_tool_expectations_consistent`
    entirely -- an agent with an empty toolset_ref and ground truth for tool
    calls it could never make loaded clean.
    """

    def test_rejects_nonempty_expected_tools(self) -> None:
        with pytest.raises(ValidationError, match="no tools offered"):
            SuiteExample(
                id="ex1",
                toolset_ref=EMPTY_TOOLSET_FINGERPRINT,
                expected_tools=[ExpectedToolCall(tool_name="x")],
            )

    def test_rejects_positive_expected_tool_count(self) -> None:
        with pytest.raises(ValidationError, match="no tools offered"):
            SuiteExample(id="ex1", toolset_ref=EMPTY_TOOLSET_FINGERPRINT, expected_tool_count=3)

    def test_allows_zero_expected_tool_count(self) -> None:
        ex = SuiteExample(id="ex1", toolset_ref=EMPTY_TOOLSET_FINGERPRINT, expected_tool_count=0)
        assert ex.expected_tool_count == 0

    def test_rejects_nonempty_expected_tool_rounds(self) -> None:
        with pytest.raises(ValidationError, match="no tools offered"):
            SuiteExample(
                id="ex1",
                toolset_ref=EMPTY_TOOLSET_FINGERPRINT,
                expected_tool_rounds=[[ExpectedToolCall(tool_name="x")]],
            )

    def test_error_wording_matches_the_inline_spelling_beyond_the_field_name(self) -> None:
        """A user who hits this should not be able to tell which spelling they
        used from the wording alone, beyond which field is named (per the
        hardening brief) -- assert the two raised messages are identical once
        each one's own field=value prefix is stripped out.

        Compares ``.errors()[0]["msg"]`` (the ``ValueError`` text this
        validator actually raises), not ``str(exc.value)`` -- pydantic's
        ``ValidationError.__str__`` also echoes the whole input dict, which
        legitimately differs between the two calls (one passed ``tools``, the
        other ``toolset_ref``) for reasons that have nothing to do with this
        validator's own wording.
        """
        with pytest.raises(ValidationError) as inline_exc:
            SuiteExample(id="ex1", tools=[], expected_tool_count=3)
        with pytest.raises(ValidationError) as ref_exc:
            SuiteExample(id="ex1", toolset_ref=EMPTY_TOOLSET_FINGERPRINT, expected_tool_count=3)

        inline_msg = inline_exc.value.errors()[0]["msg"].replace(
            "tools=[] (no tools offered)", "<REASON>"
        )
        ref_msg = ref_exc.value.errors()[0]["msg"].replace(
            "toolset_ref=<empty toolset> (no tools offered)", "<REASON>"
        )
        assert "<REASON>" in inline_msg, inline_msg
        assert "<REASON>" in ref_msg, ref_msg
        assert inline_msg == ref_msg

    def test_a_populated_toolset_ref_is_unaffected(self) -> None:
        """Positive control: only the ref naming the EMPTY toolset is
        incompatible -- any other ref (even one this validator can't
        resolve) may still carry expected_tools normally."""
        ex = SuiteExample(
            id="ex1",
            toolset_ref=_TOOLSET_REF,
            expected_tools=[ExpectedToolCall(tool_name="x")],
        )
        assert ex.expected_tools is not None


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class TestSuite:
    def test_default_is_empty(self) -> None:
        suite = Suite()
        assert len(suite) == 0
        assert suite.ids() == set()
        assert suite.by_tag("anything") == []

    def test_construction_from_examples(self) -> None:
        examples = [
            _ex(id="ex1", inputs={"n": 1}),
            _ex(id="ex2", inputs={"n": 2}),
        ]
        suite = Suite(examples=examples)
        assert len(suite) == 2
        assert suite.ids() == {"ex1", "ex2"}

    def test_by_tag_returns_only_matching_examples(self) -> None:
        suite = Suite(
            examples=[
                _ex(id="a", tags=["formal"]),
                _ex(id="b", tags=["casual"]),
                _ex(id="c", tags=["formal", "english"]),
                _ex(id="d", tags=[]),
            ],
        )
        formal = suite.by_tag("formal")
        assert {e.id for e in formal} == {"a", "c"}

    def test_by_tag_supports_multi_tag_membership(self) -> None:
        suite = Suite(
            examples=[
                _ex(id="a", tags=["formal", "english"]),
            ],
        )
        # Same example appears in both slices.
        assert suite.by_tag("formal") == suite.by_tag("english")

    def test_by_tag_returns_examples_in_suite_order(self) -> None:
        suite = Suite(
            examples=[
                _ex(id="z", tags=["t"]),
                _ex(id="a", tags=["t"]),
                _ex(id="m", tags=["t"]),
            ],
        )
        assert [e.id for e in suite.by_tag("t")] == ["z", "a", "m"]

    def test_by_tag_unknown_tag_returns_empty(self) -> None:
        suite = Suite(examples=[_ex(id="a", tags=["x"])])
        assert suite.by_tag("y") == []

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate example ids: \['ex1'\]"):
            Suite(
                examples=[
                    _ex(id="ex1"),
                    _ex(id="ex2"),
                    _ex(id="ex1"),
                ],
            )

    def test_round_trip_through_dump(self) -> None:
        original = Suite(
            examples=[
                _ex(id="a", inputs={"k": 1}, tags=["t"]),
                _ex(id="b", inputs={"k": 2}),
            ],
        )
        recreated = Suite.model_validate(original.model_dump())
        assert recreated == original

    def test_extra_top_level_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Suite.model_validate(
                {"examples": [{"id": "a"}], "rogue": 42},
            )


class TestExpectedToolCallProvenance:
    """Where a row's argument ground truth came from, disclosed on the row.

    ``capture sync`` promotes ``arguments`` verbatim from the source model's
    own recorded call, so scoring ``against: expected`` pins the source at 1.0
    by construction. The field is what lets the report say so -- and what a
    human flips once they have actually checked the row.
    """

    def test_defaults_to_captured(self) -> None:
        assert ExpectedToolCall(tool_name="x").provenance == "captured"

    def test_accepts_reviewed(self) -> None:
        call = ExpectedToolCall(tool_name="x", provenance="reviewed")
        assert call.provenance == "reviewed"
        assert ExpectedToolCall.model_validate(call.model_dump()) == call

    def test_rejects_an_unknown_provenance(self) -> None:
        with pytest.raises(ValidationError):
            ExpectedToolCall.model_validate({"tool_name": "x", "provenance": "guessed"})


# ---------------------------------------------------------------------------
# tool_result_fixtures (teacher-forced multi-round replay)
# ---------------------------------------------------------------------------


def _call(name: str, **arguments: Any) -> ExpectedToolCall:
    return ExpectedToolCall(tool_name=name, arguments=arguments or None)


def _agent_ex(**kw: Any) -> SuiteExample:
    """``_ex`` for tool ground truth: a real toolset ref, not the empty one."""
    return SuiteExample(toolset_ref=_TOOLSET_REF, **kw)


class TestToolResultFixtures:
    """Recorded tool results ride on the example, aligned by position with
    ``expected_tool_rounds`` — see "Agent rounds and what a replay can
    reproduce" in ``docs/agents.md``."""

    def test_absent_by_default_and_means_single_shot(self) -> None:
        ex = _agent_ex(id="e", expected_tool_rounds=[[_call("a")]])
        assert ex.tool_result_fixtures is None
        assert ex.rounds_to_replay() == 1

    def test_rounds_to_replay_is_covered_rounds_plus_the_answer_round(self) -> None:
        ex = _agent_ex(
            id="e",
            expected_tool_rounds=[[_call("a"), _call("b")], [_call("c")]],
            tool_result_fixtures=[
                [
                    ToolResultFixture(tool_name="a", result={"ok": 1}),
                    ToolResultFixture(tool_name="b", result="plain"),
                ],
                [ToolResultFixture(tool_name="c", error="boom")],
            ],
        )
        assert ex.rounds_to_replay() == 3

    def test_partial_coverage_replays_through_the_first_uncovered_round(self) -> None:
        ex = _agent_ex(
            id="e",
            expected_tool_rounds=[[_call("a")], [_call("c")]],
            tool_result_fixtures=[[ToolResultFixture(tool_name="a")]],
        )
        assert ex.rounds_to_replay() == 2

    def test_fixture_defaults(self) -> None:
        fx = ToolResultFixture(tool_name="a")
        assert fx.result is None
        assert fx.error is None

    def test_fixtures_require_expected_tool_rounds(self) -> None:
        with pytest.raises(ValidationError, match="expected_tool_rounds"):
            _agent_ex(id="e", tool_result_fixtures=[[ToolResultFixture(tool_name="a")]])

    def test_more_fixture_rounds_than_expected_rounds_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="2 round"):
            _agent_ex(
                id="e",
                expected_tool_rounds=[[_call("a")]],
                tool_result_fixtures=[
                    [ToolResultFixture(tool_name="a")],
                    [ToolResultFixture(tool_name="a")],
                ],
            )

    def test_round_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="round 1"):
            _agent_ex(
                id="e",
                expected_tool_rounds=[[_call("a"), _call("b")]],
                tool_result_fixtures=[[ToolResultFixture(tool_name="a")]],
            )

    def test_tool_name_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"round 1.*position 1"):
            _agent_ex(
                id="e",
                expected_tool_rounds=[[_call("a")]],
                tool_result_fixtures=[[ToolResultFixture(tool_name="zzz")]],
            )

    def test_round_trips_through_json(self) -> None:
        ex = _agent_ex(
            id="e",
            expected_tool_rounds=[[_call("a", q=1)]],
            tool_result_fixtures=[[ToolResultFixture(tool_name="a", result=[1, 2])]],
        )
        again = SuiteExample.model_validate_json(ex.model_dump_json())
        assert again == ex
        assert again.tool_result_fixtures is not None
        assert again.tool_result_fixtures[0][0].result == [1, 2]

    def test_fixture_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValidationError):
            ToolResultFixture(tool_name="a", call_id="x")  # type: ignore[call-arg]
