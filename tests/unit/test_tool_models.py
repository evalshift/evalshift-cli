"""Unit tests for :mod:`evalshift_cli.evaluators.tool_models`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalshift_cli.evaluators.tool_models import ToolCall, ToolSpec, ToolTrace

# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_minimum_valid(self) -> None:
        spec = ToolSpec(
            name="search_db",
            description="Search the customer database.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert spec.name == "search_db"
        assert spec.input_schema["properties"]["query"]["type"] == "string"

    def test_to_anthropic_shape(self) -> None:
        spec = ToolSpec(name="x", description="y", input_schema={"a": 1})
        out = spec.to_anthropic()
        assert out == {"name": "x", "description": "y", "input_schema": {"a": 1}}

    def test_to_openai_shape(self) -> None:
        spec = ToolSpec(name="x", description="y", input_schema={"a": 1})
        out = spec.to_openai()
        assert out["type"] == "function"
        assert out["function"]["name"] == "x"
        assert out["function"]["parameters"] == {"a": 1}

    def test_from_dict_anthropic_shape(self) -> None:
        spec = ToolSpec.from_dict(
            {"name": "x", "description": "y", "input_schema": {"a": 1}},
        )
        assert spec.name == "x"
        assert spec.input_schema == {"a": 1}

    def test_from_dict_openai_shape(self) -> None:
        spec = ToolSpec.from_dict(
            {
                "type": "function",
                "function": {
                    "name": "x",
                    "description": "y",
                    "parameters": {"a": 1},
                },
            },
        )
        assert spec.name == "x"
        assert spec.input_schema == {"a": 1}

    def test_from_dict_round_trip_anthropic(self) -> None:
        original = ToolSpec(name="x", description="y", input_schema={"a": 1})
        roundtripped = ToolSpec.from_dict(original.to_anthropic())
        assert roundtripped == original

    def test_from_dict_round_trip_openai(self) -> None:
        original = ToolSpec(name="x", description="y", input_schema={"a": 1})
        roundtripped = ToolSpec.from_dict(original.to_openai())
        assert roundtripped == original

    def test_from_dict_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'name'"):
            ToolSpec.from_dict({"description": "y"})

    def test_from_dict_openai_with_non_dict_function_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'function' object"):
            ToolSpec.from_dict({"function": "not a dict"})

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolSpec(name="", description="y", input_schema={})

    def test_empty_description_accepted(self) -> None:
        """A recorded tool with no description fingerprints fine in the SDK.

        The CLI is not the source of truth about what production offered, so
        it does not get to reject an empty description the SDK already
        accepted and fingerprinted (V7 reconciliation, tool_models.py:46-49
        vs. from_dict's "" default at :93/:100).
        """
        spec = ToolSpec(name="x", description="", input_schema={})
        assert spec.description == ""

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ToolSpec.model_validate(
                {"name": "x", "description": "y", "input_schema": {}, "rogue": True},
            )


class TestToolSpecStrict:
    """``strict`` (structured tool arguments) survives both wire shapes.

    The SDK records ``strict`` verbatim when production set it; dropping it on
    replay would change what the target model is actually asked to do.
    """

    def test_defaults_to_false(self) -> None:
        assert ToolSpec(name="x", description="y", input_schema={}).strict is False

    def test_non_strict_serialisation_is_unchanged(self) -> None:
        spec = ToolSpec(name="x", description="y", input_schema={"a": 1})
        assert spec.to_anthropic() == {"name": "x", "description": "y", "input_schema": {"a": 1}}
        assert spec.to_openai() == {
            "type": "function",
            "function": {"name": "x", "description": "y", "parameters": {"a": 1}},
        }

    def test_to_anthropic_emits_top_level_strict(self) -> None:
        spec = ToolSpec(name="x", description="y", input_schema={"a": 1}, strict=True)
        assert spec.to_anthropic() == {
            "name": "x",
            "description": "y",
            "input_schema": {"a": 1},
            "strict": True,
        }

    def test_to_openai_emits_function_strict(self) -> None:
        spec = ToolSpec(name="x", description="y", input_schema={"a": 1}, strict=True)
        assert spec.to_openai() == {
            "type": "function",
            "function": {
                "name": "x",
                "description": "y",
                "parameters": {"a": 1},
                "strict": True,
            },
        }

    def test_from_dict_reads_top_level_strict(self) -> None:
        spec = ToolSpec.from_dict(
            {"name": "x", "description": "y", "input_schema": {}, "strict": True}
        )
        assert spec.strict is True

    def test_from_dict_reads_openai_function_strict(self) -> None:
        spec = ToolSpec.from_dict(
            {
                "type": "function",
                "function": {"name": "x", "description": "y", "parameters": {}, "strict": True},
            }
        )
        assert spec.strict is True

    def test_from_dict_absent_strict_is_false(self) -> None:
        assert ToolSpec.from_dict({"name": "x", "input_schema": {}}).strict is False
        assert ToolSpec.from_dict({"type": "function", "function": {"name": "x"}}).strict is False

    def test_from_dict_non_true_strict_is_false(self) -> None:
        """Only a literal ``true`` counts — the SDK writes the key only when true."""
        assert (
            ToolSpec.from_dict({"name": "x", "input_schema": {}, "strict": False}).strict is False
        )
        assert (
            ToolSpec.from_dict({"name": "x", "input_schema": {}, "strict": "yes"}).strict is False
        )

    def test_round_trips_through_both_shapes(self) -> None:
        original = ToolSpec(name="x", description="y", input_schema={"a": 1}, strict=True)
        assert ToolSpec.from_dict(original.to_anthropic()) == original
        assert ToolSpec.from_dict(original.to_openai()) == original


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_minimum_valid(self) -> None:
        call = ToolCall(tool_name="search", arguments={"q": "ACME"}, sequence_index=0)
        assert call.tool_name == "search"
        assert call.call_id is None
        assert call.parent_call_id is None

    def test_negative_sequence_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(tool_name="x", arguments={}, sequence_index=-1)

    def test_empty_tool_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(tool_name="", arguments={}, sequence_index=0)

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ToolCall.model_validate(
                {
                    "tool_name": "x",
                    "arguments": {},
                    "sequence_index": 0,
                    "rogue": True,
                },
            )

    def test_round_trip_through_json(self) -> None:
        original = ToolCall(
            tool_name="search",
            arguments={"q": "ACME", "limit": 5},
            call_id="tc_1",
            parent_call_id="tc_root",
            sequence_index=2,
        )
        rebuilt = ToolCall.model_validate_json(original.model_dump_json())
        assert rebuilt == original


# ---------------------------------------------------------------------------
# ToolTrace
# ---------------------------------------------------------------------------


def _call(name: str = "search", *, idx: int = 0, parent: str | None = None) -> ToolCall:
    return ToolCall(
        tool_name=name,
        arguments={},
        sequence_index=idx,
        parent_call_id=parent,
    )


class TestToolTrace:
    def test_default_is_empty(self) -> None:
        trace = ToolTrace()
        assert trace.calls == []
        assert trace.final_text is None
        assert trace.raised_refusal is False
        assert trace.call_count == 0
        assert trace.tool_names == []
        assert trace.tool_name_set == set()
        assert trace.has_parallel_calls() is False

    def test_call_count_and_tool_names(self) -> None:
        trace = ToolTrace(
            calls=[
                _call("a", idx=0),
                _call("b", idx=1),
                _call("a", idx=2),
            ],
        )
        assert trace.call_count == 3
        assert trace.tool_names == ["a", "b", "a"]
        assert trace.tool_name_set == {"a", "b"}

    def test_calls_by_tool(self) -> None:
        trace = ToolTrace(
            calls=[
                _call("a", idx=0),
                _call("b", idx=1),
                _call("a", idx=2),
            ],
        )
        result = trace.calls_by_tool("a")
        assert [c.sequence_index for c in result] == [0, 2]
        assert trace.calls_by_tool("never") == []

    def test_has_parallel_calls_when_two_top_level(self) -> None:
        trace = ToolTrace(
            calls=[
                _call("a", idx=0),
                _call("b", idx=1),
            ],
        )
        assert trace.has_parallel_calls() is True

    def test_has_parallel_calls_when_chain(self) -> None:
        # Two calls but one is nested under the other → not parallel.
        trace = ToolTrace(
            calls=[
                _call("a", idx=0),
                _call("b", idx=1, parent="a_id"),
            ],
        )
        assert trace.has_parallel_calls() is False

    def test_has_parallel_calls_single_top_level(self) -> None:
        trace = ToolTrace(calls=[_call("a", idx=0)])
        assert trace.has_parallel_calls() is False

    def test_duplicate_sequence_index_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate sequence_index"):
            ToolTrace(
                calls=[
                    _call("a", idx=0),
                    _call("b", idx=0),  # same index — invalid
                ],
            )

    def test_round_trip_through_json(self) -> None:
        original = ToolTrace(
            calls=[
                ToolCall(
                    tool_name="search",
                    arguments={"q": "ACME"},
                    call_id="tc_1",
                    sequence_index=0,
                ),
                ToolCall(
                    tool_name="email",
                    arguments={"to": "ops@example.com"},
                    sequence_index=1,
                ),
            ],
            final_text="Done.",
            raised_refusal=False,
        )
        rebuilt = ToolTrace.model_validate_json(original.model_dump_json())
        assert rebuilt == original

    def test_refusal_state(self) -> None:
        trace = ToolTrace(
            calls=[],
            raised_refusal=True,
            refusal_text="I can't help with that.",
        )
        assert trace.raised_refusal is True
        assert trace.refusal_text is not None
        assert trace.call_count == 0

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ToolTrace.model_validate(
                {"calls": [], "rogue": True},
            )


# ---------------------------------------------------------------------------
# Multi-round traces (teacher-forced replay)
# ---------------------------------------------------------------------------


def _tc(name: str, seq: int, round_index: int = 0) -> ToolCall:
    return ToolCall(tool_name=name, sequence_index=seq, round_index=round_index)


class TestToolTraceRounds:
    def test_defaults_describe_a_single_round(self) -> None:
        trace = ToolTrace(calls=[_tc("a", 0)])
        assert trace.round_count == 1
        assert trace.calls[0].round_index == 0
        assert trace.rounds() == [trace]

    def test_pre_round_json_still_loads_as_one_round(self) -> None:
        legacy = {"calls": [{"tool_name": "a", "sequence_index": 0}], "final_text": "x"}
        trace = ToolTrace.model_validate(legacy)
        assert trace.round_count == 1
        assert trace.calls[0].round_index == 0

    def test_round_index_must_be_below_round_count(self) -> None:
        with pytest.raises(ValidationError, match="round_index"):
            ToolTrace(calls=[_tc("a", 0, round_index=1)])
        with pytest.raises(ValidationError, match="round_index"):
            ToolTrace(round_count=2, calls=[_tc("a", 0, round_index=2)])

    def test_round_index_is_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(tool_name="a", sequence_index=0, round_index=-1)

    def test_round_count_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            ToolTrace(round_count=0)

    def test_round_slices_and_renumbers(self) -> None:
        trace = ToolTrace(
            round_count=3,
            calls=[_tc("a", 0, 0), _tc("b", 1, 0), _tc("c", 2, 1)],
            final_text="done",
        )
        r0, r1, r2 = trace.rounds()
        assert r0.tool_names == ["a", "b"]
        assert [c.sequence_index for c in r0.calls] == [0, 1]
        assert r0.round_count == 1
        assert r0.final_text is None
        assert r1.tool_names == ["c"]
        assert r1.calls[0].sequence_index == 0
        assert r1.calls[0].round_index == 0
        assert r2.calls == []
        # The final text belongs to the last round only.
        assert r2.final_text == "done"
        assert trace.round(1) == r1

    def test_refusal_flags_travel_with_the_last_round(self) -> None:
        trace = ToolTrace(round_count=2, raised_refusal=True, refusal_text="no")
        first, last = trace.rounds()
        assert not first.raised_refusal
        assert last.raised_refusal and last.refusal_text == "no"

    def test_round_out_of_range(self) -> None:
        with pytest.raises(IndexError):
            ToolTrace().round(1)

    def test_has_parallel_calls_is_per_round(self) -> None:
        """Two sequential rounds of one call each are not a parallel fan-out."""
        trace = ToolTrace(round_count=2, calls=[_tc("a", 0, 0), _tc("b", 1, 1)])
        assert not trace.has_parallel_calls()
        assert not any(r.has_parallel_calls() for r in trace.rounds())
        fanout = ToolTrace(round_count=2, calls=[_tc("a", 0, 0), _tc("b", 1, 0)])
        assert fanout.has_parallel_calls()

    def test_ends_in_tool_calls_when_the_last_round_called(self) -> None:
        """A response that called a tool is still waiting on its results."""
        assert ToolTrace(calls=[_tc("a", 0)], final_text="Let me check!").ends_in_tool_calls
        assert ToolTrace(round_count=2, calls=[_tc("a", 0, 0), _tc("b", 1, 1)]).ends_in_tool_calls

    def test_does_not_end_in_tool_calls_when_the_last_round_answered(self) -> None:
        assert not ToolTrace(final_text="hi").ends_in_tool_calls
        assert not ToolTrace(
            round_count=2, calls=[_tc("a", 0, 0)], final_text="done"
        ).ends_in_tool_calls
