"""Deterministic trace rules: what a tool-calling model must and must not do.

The checker behind the ``trace_invariants`` evaluator, kept free of traces,
suites and records so the rule semantics can be read and tested in one place.

The unit of checking is a :class:`Response` -- one model turn -- carrying the
tool calls that were already in its context when it answered. Context calls
were not made by the model under test, so no rule is ever *violated* by one,
but they satisfy an ``order`` rule's ``before`` and count toward both
``call_count`` bounds: on a teacher-forced replay, round *k* was answered after the
recorded rounds before it, and those calls really happened in that
conversation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from jsonschema import Draft202012Validator

from evalshift_cli.config.models import (
    ArgumentsRule,
    CallCountRule,
    ForbiddenToolsRule,
    InvariantRule,
    RequiredToolsRule,
    ToolOrderRule,
)


@dataclass(frozen=True, slots=True)
class ObservedCall:
    """One tool call the model under test made."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Response:
    """One model response and the tool calls already in its context.

    Attributes:
        round_index: Which replayed round this response answered; ``0`` for a
            single-shot replay and for an imported trace.
        context: Tool names called before this response by someone other than
            the model under test -- recorded history, and the recorded earlier
            rounds a teacher-forced replay fed back -- in order.
        calls: The model's own calls, in the order it emitted them.
    """

    round_index: int
    context: tuple[str, ...]
    calls: tuple[ObservedCall, ...]


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule, attributed to one call (or one missing tool)."""

    rule_id: str
    rule_type: str
    tool: str
    round_index: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """The JSON shape stored under a record's ``*_violations`` metadata."""
        return asdict(self)


def check_rules(rules: Sequence[InvariantRule], responses: Sequence[Response]) -> list[Violation]:
    """Every violation of every rule across ``responses``, rule by rule.

    Args:
        rules: The rules to check, in configured order.
        responses: One side's model responses, in round order.

    Returns:
        The violations grouped by rule (in ``rules`` order), each rule's in
        the order they occur in the trace; empty when every rule holds.
    """
    violations: list[Violation] = []
    for rule in rules:
        violations.extend(_check(rule, responses))
    return violations


def side_score(rules: Sequence[InvariantRule], violations: Sequence[Violation]) -> float:
    """Share of ``rules`` broken zero times; ``1.0`` when there are no rules.

    Args:
        rules: The rules that were checked.
        violations: What :func:`check_rules` returned for them.

    Returns:
        A score in ``[0.0, 1.0]``. A rule broken many times costs the same
        as a rule broken once; a violation naming no rule in ``rules`` costs
        nothing.
    """
    if not rules:
        return 1.0
    broken = {violation.rule_id for violation in violations}
    kept = sum(1 for rule in rules if rule.id not in broken)
    return kept / len(rules)


def _check(rule: InvariantRule, responses: Sequence[Response]) -> list[Violation]:
    if isinstance(rule, ForbiddenToolsRule):
        return _forbidden(rule, responses)
    if isinstance(rule, RequiredToolsRule):
        return _required(rule, responses)
    if isinstance(rule, ToolOrderRule):
        return _order(rule, responses)
    if isinstance(rule, CallCountRule):
        return _call_count(rule, responses)
    return _arguments(rule, responses)


def _forbidden(rule: ForbiddenToolsRule, responses: Sequence[Response]) -> list[Violation]:
    banned = set(rule.tools)
    return [
        Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            tool=call.name,
            round_index=response.round_index,
            detail=f"called forbidden tool {call.name!r}",
        )
        for response in responses
        for call in response.calls
        if call.name in banned
    ]


def _required(rule: RequiredToolsRule, responses: Sequence[Response]) -> list[Violation]:
    called = {call.name for response in responses for call in response.calls}
    last_round = responses[-1].round_index if responses else 0
    return [
        Violation(
            rule_id=rule.id,
            rule_type=rule.type,
            tool=tool,
            round_index=last_round,
            detail=f"never called required tool {tool!r}",
        )
        for tool in rule.tools
        if tool not in called
    ]


def _order(rule: ToolOrderRule, responses: Sequence[Response]) -> list[Violation]:
    violations: list[Violation] = []
    for response in responses:
        seen_before = rule.before in response.context
        for call in response.calls:
            if call.name == rule.before:
                seen_before = True
            elif call.name == rule.after and not seen_before:
                violations.append(
                    Violation(
                        rule_id=rule.id,
                        rule_type=rule.type,
                        tool=call.name,
                        round_index=response.round_index,
                        detail=f"called {rule.after!r} before any {rule.before!r}",
                    ),
                )
    return violations


def _call_count(rule: CallCountRule, responses: Sequence[Response]) -> list[Violation]:
    violations: list[Violation] = []
    final_count = 0
    for response in responses:
        count = response.context.count(rule.tool)
        for call in response.calls:
            if call.name != rule.tool:
                continue
            count += 1
            if rule.max_calls is not None and count > rule.max_calls:
                violations.append(
                    Violation(
                        rule_id=rule.id,
                        rule_type=rule.type,
                        tool=call.name,
                        round_index=response.round_index,
                        detail=f"call #{count} to {rule.tool!r}; at most {rule.max_calls} allowed",
                    ),
                )
        # What the conversation holds once this response is done. Only the
        # last response's value is judged: an earlier round cannot know what
        # later rounds will add, and its own calls were never fed forward.
        final_count = count
    if rule.min_calls is not None and final_count < rule.min_calls:
        violations.append(
            Violation(
                rule_id=rule.id,
                rule_type=rule.type,
                tool=rule.tool,
                round_index=responses[-1].round_index if responses else 0,
                detail=(
                    f"{final_count} call(s) to {rule.tool!r} by the end; "
                    f"at least {rule.min_calls} required"
                ),
            ),
        )
    return violations


def _arguments(rule: ArgumentsRule, responses: Sequence[Response]) -> list[Violation]:
    validator = Draft202012Validator(rule.json_schema)
    violations: list[Violation] = []
    for response in responses:
        for call in response.calls:
            if call.name != rule.tool:
                continue
            errors = sorted(
                validator.iter_errors(call.arguments),
                key=lambda error: [str(part) for part in error.absolute_path],
            )
            if not errors:
                continue
            first = errors[0]
            where = "/".join(str(part) for part in first.absolute_path) or "(arguments)"
            violations.append(
                Violation(
                    rule_id=rule.id,
                    rule_type=rule.type,
                    tool=call.name,
                    round_index=response.round_index,
                    detail=f"{where}: {first.message}",
                ),
            )
    return violations


__all__ = ["ObservedCall", "Response", "Violation", "check_rules", "side_score"]
