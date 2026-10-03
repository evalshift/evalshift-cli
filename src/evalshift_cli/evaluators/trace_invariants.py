"""Evaluator: hand-written trace rules, checked on both sides of every pair.

Every other tool evaluator asks "did the target do what the source did?",
which lets a wrong call the source also made pass as correct. This one asks
"did each side keep the rules?" -- the source is not the yardstick, the
rules are. See :mod:`evalshift_cli.evaluators.invariants` for what each rule
means, and ``migration_policy.max_invariant_violations`` for how a broken
rule fails the run.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any

from evalshift_cli.config.models import TraceInvariantsEvaluatorConfig
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import INVARIANT_VIOLATION
from evalshift_cli.evaluators.invariants import (
    ObservedCall,
    Response,
    Violation,
    check_rules,
    side_score,
)
from evalshift_cli.evaluators.tool_models import ToolTrace
from evalshift_cli.suite.models import SuiteExample
from evalshift_cli.traces.models import AgentTrace

#: Stable evaluator-type slug stamped onto every record. The analysis
#: layer selects the ``max_invariant_violations`` budget on this.
KIND = "trace_invariants"


def replayed_responses(trace: ToolTrace, example: SuiteExample) -> list[Response]:
    """One :class:`Response` per replayed round, with what that round was shown.

    Round *k* was asked after the example's ``history`` and the recorded
    rounds ``0..k-1`` the replay fed back -- exactly the prefix
    ``runner/orchestrator.py:build_round_messages`` builds -- never after the
    model's own earlier rounds.

    Args:
        trace: One side's replayed trace; a single-round trace is one response.
        example: The suite example that was replayed. Its ``history``,
            ``expected_tool_rounds`` and ``tool_result_fixtures`` decide each
            round's context.

    Returns:
        One response per round of ``trace``, in round order.
    """
    history = tuple(
        call.name for message in example.history or [] for call in message.tool_calls or []
    )
    recorded = example.expected_tool_rounds or []
    fed_back = min(len(recorded), len(example.tool_result_fixtures or []))
    responses: list[Response] = []
    for index, round_trace in enumerate(trace.rounds()):
        earlier = tuple(
            call.tool_name
            for round_calls in recorded[: min(index, fed_back)]
            for call in round_calls
        )
        responses.append(
            Response(
                round_index=index,
                context=history + earlier,
                calls=tuple(ObservedCall(c.tool_name, c.arguments) for c in round_trace.calls),
            ),
        )
    return responses


def imported_responses(trace: AgentTrace) -> list[Response]:
    """An imported trace is the agent's own whole run: one response, no context.

    Args:
        trace: One side's imported trace.

    Returns:
        A single response holding every tool call in sequence order.
    """
    return [
        Response(
            round_index=0,
            context=(),
            calls=tuple(ObservedCall(c.name, c.arguments) for c in trace.tool_calls),
        ),
    ]


class _TraceInvariantsBase:
    kind = KIND

    def __init__(self, config: TraceInvariantsEvaluatorConfig) -> None:
        self.config = config
        self.name = config.name

    def applies(self, prompt_id: str) -> bool:
        """Whether ``prompt_id`` matches one of the configured ``applies_to`` globs."""
        return any(fnmatchcase(prompt_id, pattern) for pattern in self.config.applies_to)

    def _record(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example_id: str,
        source: list[Response],
        target: list[Response],
    ) -> EvalRecord:
        rules = self.config.rules
        source_violations = check_rules(rules, source)
        target_violations = check_rules(rules, target)
        source_score = side_score(rules, source_violations)
        target_score = side_score(rules, target_violations)
        metadata: dict[str, Any] = {
            "rules_checked": [rule.id for rule in rules],
            "source_violations": [v.to_dict() for v in source_violations],
            "target_violations": [v.to_dict() for v in target_violations],
        }
        if self.config.owner:
            metadata["owner"] = self.config.owner
        if target_violations:
            metadata["failure_categories"] = [INVARIANT_VIOLATION]
        return EvalRecord(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example_id,
            evaluator_name=self.name,
            kind=KIND,
            source_score=source_score,
            target_score=target_score,
            delta=target_score - source_score,
            explanation=_explanation(target_violations),
            metadata=metadata,
        )


class TraceInvariantsEvaluator(_TraceInvariantsBase):
    """Rules over a replayed (source, target) :class:`ToolTrace` pair."""

    async def score_pair(
        self,
        *,
        run_id: str,
        prompt_id: str,
        example: SuiteExample,
        source_trace: ToolTrace,
        target_trace: ToolTrace,
    ) -> EvalRecord:
        """Check every rule on both sides of one replayed pair.

        Args:
            run_id: The run the pair belongs to.
            prompt_id: The prompt the example was replayed against.
            example: The replayed suite example (supplies round context).
            source_trace: The source model's replayed trace.
            target_trace: The target model's replayed trace.

        Returns:
            One record: each side scored as the share of rules it kept, with
            both sides' violations in ``metadata``.
        """
        return self._record(
            run_id=run_id,
            prompt_id=prompt_id,
            example_id=example.id,
            source=replayed_responses(source_trace, example),
            target=replayed_responses(target_trace, example),
        )


class ImportedTraceInvariantsEvaluator(_TraceInvariantsBase):
    """Rules over an imported (source, target) :class:`AgentTrace` pair."""

    async def score_trace_pair(
        self,
        *,
        run_id: str,
        source_trace: AgentTrace,
        target_trace: AgentTrace,
    ) -> EvalRecord:
        """Check every rule on both sides of one imported pair.

        Args:
            run_id: The run the pair belongs to.
            source_trace: The source side's imported trace; its
                ``prompt_id`` / ``example_id`` key the record.
            target_trace: The target side's imported trace.

        Returns:
            One record: each side scored as the share of rules it kept, with
            both sides' violations in ``metadata``.
        """
        return self._record(
            run_id=run_id,
            prompt_id=source_trace.prompt_id,
            example_id=source_trace.example_id,
            source=imported_responses(source_trace),
            target=imported_responses(target_trace),
        )


def build_trace_invariants_evaluator(
    config: TraceInvariantsEvaluatorConfig,
) -> TraceInvariantsEvaluator | ImportedTraceInvariantsEvaluator:
    """The evaluator for ``config.traces``; ``evaluate`` routes on its method.

    Args:
        config: The configured ``trace_invariants`` evaluator.

    Returns:
        A :class:`TraceInvariantsEvaluator` for ``traces: replayed``, an
        :class:`ImportedTraceInvariantsEvaluator` for ``traces: imported``.
    """
    if config.traces == "imported":
        return ImportedTraceInvariantsEvaluator(config)
    return TraceInvariantsEvaluator(config)


def _explanation(violations: list[Violation]) -> str:
    if not violations:
        return "target kept every trace rule"
    broken = sorted({v.rule_id for v in violations})
    return f"target broke {len(broken)} trace rule(s): {', '.join(broken)}"


__all__ = [
    "KIND",
    "ImportedTraceInvariantsEvaluator",
    "TraceInvariantsEvaluator",
    "build_trace_invariants_evaluator",
    "imported_responses",
    "replayed_responses",
]
