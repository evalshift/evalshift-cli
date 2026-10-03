"""Tests for the trace-invariants evaluators."""

from __future__ import annotations

from typing import Any

import pytest

from evalshift_cli.config.models import TraceInvariantsEvaluatorConfig
from evalshift_cli.evaluators.failures import CATEGORY_LABELS, INVARIANT_VIOLATION
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.evaluators.trace_invariants import (
    KIND,
    ImportedTraceInvariantsEvaluator,
    TraceInvariantsEvaluator,
    build_trace_invariants_evaluator,
    replayed_responses,
)
from evalshift_cli.suite.models import ChatMessage
from evalshift_cli.traces.models import AgentTrace
from tests.unit.suite_examples import suite_example

_RULES: list[dict[str, Any]] = [
    {"id": "auth-first", "type": "order", "before": "authenticate", "after": "charge"},
    {"id": "one-charge", "type": "call_count", "tool": "charge", "max_calls": 1},
]


def _config(**overrides: Any) -> TraceInvariantsEvaluatorConfig:
    return TraceInvariantsEvaluatorConfig.model_validate(
        {"name": "payments", "rules": _RULES, **overrides},
    )


def _trace(*calls: str | tuple[str, int], round_count: int = 1) -> ToolTrace:
    """Calls as names, or (name, round_index) pairs for a multi-round replay."""
    built = [
        ToolCall(
            tool_name=c if isinstance(c, str) else c[0],
            arguments={},
            sequence_index=i,
            round_index=0 if isinstance(c, str) else c[1],
        )
        for i, c in enumerate(calls)
    ]
    return ToolTrace(calls=built, round_count=round_count)


async def _score(
    source: ToolTrace,
    target: ToolTrace,
    *,
    config: TraceInvariantsEvaluatorConfig | None = None,
    **example: Any,
) -> Any:
    evaluator = TraceInvariantsEvaluator(config or _config())
    return await evaluator.score_pair(
        run_id="r",
        prompt_id="checkout",
        example=suite_example(id="ex1", **example),
        source_trace=source,
        target_trace=target,
    )


class TestRecordShape:
    async def test_clean_pair_is_equivalent(self) -> None:
        record = await _score(_trace("authenticate", "charge"), _trace("authenticate", "charge"))
        assert record.kind == KIND
        assert (record.source_score, record.target_score, record.delta) == (1.0, 1.0, 0.0)
        assert record.metadata["target_violations"] == []
        assert "failure_categories" not in record.metadata

    async def test_target_violation_is_categorised_and_scored(self) -> None:
        record = await _score(_trace("authenticate", "charge"), _trace("charge", "charge"))
        assert record.target_score == pytest.approx(0.0)  # both rules broken
        assert record.delta == pytest.approx(-1.0)
        assert record.metadata["failure_categories"] == [INVARIANT_VIOLATION]
        rule_ids = {v["rule_id"] for v in record.metadata["target_violations"]}
        assert rule_ids == {"auth-first", "one-charge"}
        assert "auth-first" in record.explanation

    async def test_a_violation_the_source_shares_still_marks_the_target(self) -> None:
        record = await _score(_trace("charge"), _trace("charge"))
        assert record.delta == 0.0
        assert record.target_score < 1.0
        assert record.metadata["failure_categories"] == [INVARIANT_VIOLATION]
        assert record.metadata["source_violations"]

    async def test_owner_and_rules_checked_ride_on_the_record(self) -> None:
        record = await _score(_trace(), _trace(), config=_config(owner="@payments"))
        assert record.metadata["owner"] == "@payments"
        assert record.metadata["rules_checked"] == ["auth-first", "one-charge"]


class TestAppliesTo:
    def test_globs_select_prompts(self) -> None:
        evaluator = TraceInvariantsEvaluator(_config(applies_to=["check*"]))
        assert evaluator.applies("checkout")
        assert not evaluator.applies("support")

    def test_default_applies_everywhere(self) -> None:
        assert TraceInvariantsEvaluator(_config()).applies("anything")


class TestTeacherForcedContext:
    """Review Focus 2: round k's context is the *recorded* rounds, not the model's own."""

    def _example(self) -> dict[str, Any]:
        return {
            "expected_tool_rounds": [
                [{"tool_name": "authenticate"}],
                [{"tool_name": "charge"}],
            ],
            "tool_result_fixtures": [
                [{"tool_name": "authenticate", "result": "ok"}],
                [{"tool_name": "charge", "result": "ok"}],
            ],
        }

    async def test_recorded_round_zero_satisfies_order_in_round_one(self) -> None:
        # Target skipped auth in its own round 0 (called lookup) but round 1 was
        # answered after the *recorded* authenticate.
        target = _trace(("lookup", 0), ("charge", 1), round_count=3)
        record = await _score(target, target, **self._example())
        assert record.metadata["target_violations"] == []

    async def test_own_round_zero_alternative_does_not_count_toward_round_one(self) -> None:
        target = _trace(("charge", 0), ("charge", 1), round_count=3)
        record = await _score(target, target, **self._example())
        violations = record.metadata["target_violations"]
        # Round 0: charge without auth (order). Round 1: context has the recorded
        # authenticate only, so its single charge is the first -- no call_count break.
        assert [(v["rule_id"], v["round_index"]) for v in violations] == [("auth-first", 0)]

    async def test_round_two_counts_the_recorded_charge(self) -> None:
        target = _trace(("authenticate", 0), ("charge", 2), round_count=3)
        record = await _score(target, target, **self._example())
        violations = record.metadata["target_violations"]
        assert [(v["rule_id"], v["round_index"]) for v in violations] == [("one-charge", 2)]

    def test_history_tool_calls_are_context(self) -> None:
        example = suite_example(
            id="ex1",
            history=[
                ChatMessage(role="user", content="pay"),
                ChatMessage(
                    role="assistant",
                    tool_calls=[{"id": "c1", "name": "authenticate", "arguments": {}}],
                ),
                ChatMessage(role="tool", tool_call_id="c1", content="ok"),
            ],
        )
        (response,) = replayed_responses(_trace("charge"), example)
        assert response.context == ("authenticate",)


class TestImported:
    def _trace(self, role: str, *names: str) -> AgentTrace:
        return AgentTrace.model_validate(
            {
                "run_id": "r",
                "prompt_id": "checkout",
                "example_id": "ex1",
                "role": role,
                "events": [
                    {
                        "type": "tool_call",
                        "sequence_index": i,
                        "timestamp": "2026-10-02T12:00:00Z",
                        "name": name,
                        "arguments": {},
                    }
                    for i, name in enumerate(names)
                ],
            },
        )

    async def test_imported_traces_are_one_response_in_sequence_order(self) -> None:
        evaluator = ImportedTraceInvariantsEvaluator(_config(traces="imported"))
        record = await evaluator.score_trace_pair(
            run_id="r",
            source_trace=self._trace("source", "authenticate", "charge"),
            target_trace=self._trace("target", "charge", "authenticate", "charge"),
        )
        assert (record.prompt_id, record.example_id, record.kind) == ("checkout", "ex1", KIND)
        rule_ids = sorted(v["rule_id"] for v in record.metadata["target_violations"])
        assert rule_ids == ["auth-first", "one-charge"]


def test_the_builder_routes_on_traces() -> None:
    assert isinstance(build_trace_invariants_evaluator(_config()), TraceInvariantsEvaluator)
    assert isinstance(
        build_trace_invariants_evaluator(_config(traces="imported")),
        ImportedTraceInvariantsEvaluator,
    )


def test_the_category_has_a_display_label() -> None:
    assert CATEGORY_LABELS[INVARIANT_VIOLATION] == "Broke a hand-written trace rule"
