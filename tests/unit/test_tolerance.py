"""Tests for the shared within-tolerance predicate."""

from __future__ import annotations

from evalshift_cli.analysis.tolerance import (
    is_semantic,
    is_semantic_regression,
    within_tolerance,
)
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION


def _record(
    *,
    kind: str = "semantic",
    name: str = "semantic.cosine",
    target: float = 0.95,
    flagged: bool = False,
) -> EvalRecord:
    return EvalRecord(
        run_id="r1",
        prompt_id="p",
        example_id="ex1",
        evaluator_name=name,
        kind=kind,
        source_score=1.0,
        target_score=target,
        delta=target - 1.0,
        metadata={"failure_categories": [SEMANTIC_REGRESSION]} if flagged else {},
    )


def test_unflagged_semantic_drift_is_within_tolerance() -> None:
    assert within_tolerance(_record(target=0.904))


def test_flagged_semantic_drift_is_not() -> None:
    record = _record(target=0.75, flagged=True)
    assert is_semantic_regression(record)
    assert not within_tolerance(record)


def test_other_kinds_are_never_within_tolerance() -> None:
    assert not within_tolerance(_record(kind="llm_judge", name="llm_judge.equivalence"))
    assert not within_tolerance(_record(kind="", name="structural.length"))


def test_legacy_rows_fall_back_to_the_name_prefix() -> None:
    legacy = _record(kind="", name="semantic.cosine")
    assert is_semantic(legacy)
    assert within_tolerance(legacy)


def test_a_renamed_semantic_evaluator_is_selected_by_kind() -> None:
    assert within_tolerance(_record(kind="semantic", name="meaning"))


def test_a_malformed_category_list_is_not_a_regression() -> None:
    record = _record().model_copy(update={"metadata": {"failure_categories": "SEMANTIC"}})
    assert not is_semantic_regression(record)
    assert within_tolerance(record)
