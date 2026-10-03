"""Tests for the pure trace-rule checker."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import TypeAdapter

from evalshift_cli.config.models import InvariantRule
from evalshift_cli.evaluators.invariants import (
    ObservedCall,
    Response,
    check_rules,
    side_score,
)

_RULE = TypeAdapter(InvariantRule)


def _rule(**payload: Any) -> InvariantRule:
    return _RULE.validate_python(payload)


def _response(
    *calls: str | tuple[str, dict[str, Any]],
    context: tuple[str, ...] = (),
    round_index: int = 0,
) -> Response:
    observed = tuple(
        ObservedCall(name=c, arguments={}) if isinstance(c, str) else ObservedCall(*c)
        for c in calls
    )
    return Response(round_index=round_index, context=context, calls=observed)


class TestForbidden:
    def test_each_forbidden_call_is_one_violation(self) -> None:
        rule = _rule(id="no-v1", type="forbidden", tools=["refund_v1"])
        violations = check_rules([rule], [_response("lookup", "refund_v1", "refund_v1")])
        assert [v.rule_id for v in violations] == ["no-v1", "no-v1"]
        assert violations[0].tool == "refund_v1"

    def test_a_forbidden_tool_in_context_is_not_the_models_violation(self) -> None:
        rule = _rule(id="no-v1", type="forbidden", tools=["refund_v1"])
        assert check_rules([rule], [_response("lookup", context=("refund_v1",))]) == []


class TestRequired:
    def test_missing_tool_is_one_violation_per_tool(self) -> None:
        rule = _rule(id="must", type="required", tools=["book", "confirm"])
        violations = check_rules([rule], [_response("book")])
        assert [(v.rule_id, v.tool) for v in violations] == [("must", "confirm")]

    def test_a_call_in_any_round_satisfies_it(self) -> None:
        rule = _rule(id="must", type="required", tools=["book"])
        responses = [_response("search"), _response("book", round_index=1)]
        assert check_rules([rule], responses) == []

    def test_a_call_only_in_context_does_not_satisfy_it(self) -> None:
        rule = _rule(id="must", type="required", tools=["book"])
        assert len(check_rules([rule], [_response("search", context=("book",))])) == 1


class TestOrder:
    _RULE_ARGS: ClassVar[dict[str, Any]] = {
        "id": "auth-first",
        "type": "order",
        "before": "authenticate",
        "after": "charge",
    }

    def test_after_without_before_is_a_violation(self) -> None:
        violations = check_rules([_rule(**self._RULE_ARGS)], [_response("charge")])
        assert [v.tool for v in violations] == ["charge"]

    def test_before_earlier_in_the_same_response_satisfies_it(self) -> None:
        assert check_rules([_rule(**self._RULE_ARGS)], [_response("authenticate", "charge")]) == []

    def test_before_later_in_the_same_response_does_not(self) -> None:
        violations = check_rules([_rule(**self._RULE_ARGS)], [_response("charge", "authenticate")])
        assert len(violations) == 1

    def test_before_in_context_satisfies_it(self) -> None:
        responses = [_response("charge", context=("authenticate",))]
        assert check_rules([_rule(**self._RULE_ARGS)], responses) == []

    def test_a_trace_that_never_calls_after_is_vacuously_fine(self) -> None:
        assert check_rules([_rule(**self._RULE_ARGS)], [_response("lookup")]) == []

    def test_the_models_own_earlier_round_is_not_context(self) -> None:
        """Teacher-forced: round 1 never saw round 0's own calls, only the recorded ones."""
        responses = [_response("authenticate"), _response("charge", round_index=1)]
        violations = check_rules([_rule(**self._RULE_ARGS)], responses)
        assert [v.round_index for v in violations] == [1]


class TestCallCount:
    _MAX_ONE: ClassVar[dict[str, Any]] = {
        "id": "one",
        "type": "call_count",
        "tool": "charge",
        "max_calls": 1,
    }
    _MIN_ONE: ClassVar[dict[str, Any]] = {
        "id": "some",
        "type": "call_count",
        "tool": "charge",
        "min_calls": 1,
    }

    def test_the_call_past_the_maximum_is_the_violation(self) -> None:
        violations = check_rules([_rule(**self._MAX_ONE)], [_response("charge", "charge")])
        assert len(violations) == 1
        assert "call #2" in violations[0].detail

    def test_context_calls_count_toward_the_maximum(self) -> None:
        responses = [_response("charge", context=("charge",))]
        assert len(check_rules([_rule(**self._MAX_ONE)], responses)) == 1

    def test_context_alone_over_the_maximum_is_not_the_models_violation(self) -> None:
        responses = [_response("lookup", context=("charge", "charge"))]
        assert check_rules([_rule(**self._MAX_ONE)], responses) == []

    def test_each_round_counts_from_its_own_context(self) -> None:
        """Round 0's own call is an alternative never fed back to round 1."""
        responses = [_response("charge"), _response("charge", round_index=1)]
        assert check_rules([_rule(**self._MAX_ONE)], responses) == []

    def test_max_zero_forbids_the_tool(self) -> None:
        rule = _rule(id="none", type="call_count", tool="charge", max_calls=0)
        assert len(check_rules([rule], [_response("charge")])) == 1

    def test_fewer_than_the_minimum_is_one_violation(self) -> None:
        rule = _rule(id="two", type="call_count", tool="charge", min_calls=2)
        violations = check_rules([rule], [_response("charge")])
        assert len(violations) == 1
        assert "1 call(s)" in violations[0].detail
        assert "at least 2" in violations[0].detail

    def test_context_calls_count_toward_the_minimum(self) -> None:
        """Unlike ``required``: the conversation already made the call."""
        responses = [_response("lookup", context=("charge",))]
        assert check_rules([_rule(**self._MIN_ONE)], responses) == []

    def test_the_minimum_is_judged_on_the_last_response(self) -> None:
        """The model's own round 0 was never fed back; the recorded round was."""
        own_round_zero_only = [
            _response("charge"),
            _response("lookup", context=("authenticate",), round_index=1),
        ]
        violations = check_rules([_rule(**self._MIN_ONE)], own_round_zero_only)
        assert [v.round_index for v in violations] == [1]
        recorded = [_response("lookup"), _response("lookup", context=("charge",), round_index=1)]
        assert check_rules([_rule(**self._MIN_ONE)], recorded) == []

    def test_exact_is_both_bounds(self) -> None:
        rule = _rule(id="exactly-one", type="call_count", tool="charge", min_calls=1, max_calls=1)
        assert check_rules([rule], [_response("charge")]) == []
        assert len(check_rules([rule], [_response()])) == 1
        assert len(check_rules([rule], [_response("charge", "charge")])) == 1


class TestArguments:
    _SCHEMA: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["amount"],
        "properties": {"amount": {"type": "number", "minimum": 0.01, "maximum": 5000}},
    }

    def _rule(self) -> InvariantRule:
        return _rule(id="sane", type="arguments", tool="charge", json_schema=self._SCHEMA)

    def test_valid_arguments_pass(self) -> None:
        assert check_rules([self._rule()], [_response(("charge", {"amount": 12.5}))]) == []

    def test_out_of_range_names_the_field(self) -> None:
        violations = check_rules([self._rule()], [_response(("charge", {"amount": 9000}))])
        assert len(violations) == 1
        assert "amount" in violations[0].detail
        assert "5000" in violations[0].detail

    def test_missing_required_field(self) -> None:
        violations = check_rules([self._rule()], [_response(("charge", {}))])
        assert "amount" in violations[0].detail

    def test_unparseable_arguments_fail_a_schema_that_requires_a_field(self) -> None:
        violations = check_rules([self._rule()], [_response(("charge", {"_parse_error": True}))])
        assert len(violations) == 1

    def test_other_tools_are_not_checked(self) -> None:
        assert check_rules([self._rule()], [_response(("refund", {"amount": -1}))]) == []


class TestSideScore:
    def test_share_of_rules_never_broken(self) -> None:
        rules = [
            _rule(id="a", type="forbidden", tools=["x"]),
            _rule(id="b", type="forbidden", tools=["y"]),
        ]
        violations = check_rules(rules, [_response("x", "x")])
        assert side_score(rules, violations) == pytest.approx(0.5)

    def test_clean_is_one(self) -> None:
        rules = [_rule(id="a", type="forbidden", tools=["x"])]
        assert side_score(rules, []) == 1.0


def test_violation_to_dict_is_json_ready() -> None:
    rule = _rule(id="no-v1", type="forbidden", tools=["refund_v1"])
    (violation,) = check_rules([rule], [_response("refund_v1", round_index=2)])
    assert violation.to_dict() == {
        "rule_id": "no-v1",
        "rule_type": "forbidden",
        "tool": "refund_v1",
        "round_index": 2,
        "detail": "called forbidden tool 'refund_v1'",
    }
