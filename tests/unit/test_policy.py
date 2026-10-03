"""Tests for migration-policy verdict generation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evalshift_cli.analysis.policy import (
    BudgetResult,
    MigrationDecision,
    _metrics,
    evaluate_migration_policy,
    inconclusive_decision,
    unmeasured_gating_evaluators,
)
from evalshift_cli.analysis.statistics import (
    ADVISORY_NOTE_PREFIX,
    UNMEASURED_NOTE_PREFIX,
    ComparisonResult,
)
from evalshift_cli.config.models import MigrationPolicy, SliceMigrationPolicy
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import (
    INVARIANT_VIOLATION,
    TOOL_GROUND_TRUTH_MISS,
    TOOL_SELECTION_DRIFT,
)
from evalshift_cli.evaluators.tool_arguments import KIND as KIND_ARGUMENTS
from evalshift_cli.evaluators.tool_selection import KIND_CONFORMANCE, KIND_DIVERGENCE
from evalshift_cli.runner.models import Call


def _comparison(
    *,
    severity: str,
    slice_name: str = "all",
    evaluator_name: str = "structural.length",
    prompt_id: str = "p",
    delta_avg_score: float = -0.1,
    notes: list[str] | None = None,
) -> ComparisonResult:
    return ComparisonResult(
        prompt_id=prompt_id,
        evaluator_name=evaluator_name,
        slice_name=slice_name,
        n=30,
        test="paired_t",
        statistic=1.0,
        p_value=0.01,
        p_value_corrected=0.01,
        effect_size=-0.8 if severity not in {"improved", "none"} else 0.2,
        effect_size_ci_low=-1.0,
        effect_size_ci_high=-0.2,
        delta_avg_score=delta_avg_score,
        severity=severity,  # type: ignore[arg-type]
        notes=notes or [],
    )


def _record(
    *,
    example_id: str,
    delta: float,
    evaluator_name: str = "structural.length",
    category: str | None = None,
    blocking: bool = True,
) -> EvalRecord:
    metadata = {"failure_categories": [category]} if category else {}
    return EvalRecord(
        run_id="r1",
        prompt_id="p",
        example_id=example_id,
        evaluator_name=evaluator_name,
        source_score=1.0,
        target_score=max(0.0, 1.0 + delta),
        delta=delta,
        metadata=metadata,
        blocking=blocking,
    )


def _call(
    *,
    example_id: str,
    role: str,
    cost_usd: float,
    latency_ms: int,
    error: str | None = None,
) -> Call:
    return Call(
        run_id="r1",
        prompt_id="p",
        example_id=example_id,
        model_id=role,
        role=role,  # type: ignore[arg-type]
        text="" if error else "ok",
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        error=error,
    )


class TestEvaluateMigrationPolicy:
    def test_fail_when_blocking_regression_exceeds_budget(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_critical_regressions=0),
            comparisons=[_comparison(severity="critical", evaluator_name="tool_selection.routing")],
            records=[
                _record(
                    example_id="ex1",
                    delta=-1.0,
                    evaluator_name="tool_selection.routing",
                    category="TOOL_SELECTION_DRIFT",
                ),
            ],
            calls=[],
        )

        assert decision.verdict == "fail"
        assert decision.blocking_regressions
        assert decision.failure_categories[0].category == "TOOL_SELECTION_DRIFT"

    def test_conditional_pass_when_only_specific_slice_regresses(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_overall_regression_rate=1.0,
                max_critical_regressions=1,
                min_equivalence_rate=0.0,
            ),
            comparisons=[
                _comparison(severity="high", slice_name="billing"),
                _comparison(severity="none", slice_name="support", delta_avg_score=0.0),
            ],
            records=[
                _record(example_id="ex1", delta=-0.8, category="SEMANTIC_REGRESSION"),
                _record(example_id="ex2", delta=0.0),
            ],
            calls=[],
        )

        assert decision.verdict == "conditional_pass"
        assert decision.slices["billing"].verdict == "fail"
        assert decision.slices["support"].verdict == "pass"
        assert "support" in decision.recommendations[0]

    def test_pass_when_all_budgets_are_met(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"ex{i}", delta=0.0) for i in range(20)],
            calls=[
                _call(example_id="ex1", role="source", cost_usd=1.0, latency_ms=100),
                _call(example_id="ex1", role="target", cost_usd=1.1, latency_ms=110),
            ],
        )

        assert decision.verdict == "pass"
        assert decision.overall.regression_rate == pytest.approx(0.0)
        assert all(result.passed for result in decision.budget_results)

    def test_unflagged_semantic_drift_is_equivalent_not_regression(self) -> None:
        # cosine ~0.98 (delta -0.02) within min_similarity: no SEMANTIC_REGRESSION
        # flag, so it must NOT count toward the regression budget and instead
        # count as equivalent.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                _record(example_id="ex1", delta=-0.02, evaluator_name="semantic.cosine"),
            ],
            calls=[],
        )

        assert decision.overall.regression_rate == pytest.approx(0.0)
        assert decision.overall.equivalent_rate == pytest.approx(1.0)

    def test_improvements_satisfy_equivalence_budget(self) -> None:
        # A target that IMPROVES on every record has equivalent_rate 0.0 but
        # must still pass the default min_equivalence_rate=0.95, because the
        # budget is a floor on the non-regression (equivalent-or-better) rate.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="improved", delta_avg_score=0.2)],
            records=[
                EvalRecord(
                    run_id="r1",
                    prompt_id="p",
                    example_id=f"ex{i}",
                    evaluator_name="structural.length",
                    source_score=0.6,
                    target_score=0.9,
                    delta=0.3,
                    metadata={},
                )
                for i in range(20)
            ],
            calls=[],
        )

        equivalence = next(b for b in decision.budget_results if b.name == "min_equivalence_rate")
        assert decision.overall.equivalent_rate == pytest.approx(0.0)
        assert decision.overall.improved_rate == pytest.approx(1.0)
        assert equivalence.observed == pytest.approx(1.0)
        assert equivalence.passed

    def test_flagged_semantic_drift_counts_as_regression(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                _record(
                    example_id="ex1",
                    delta=-0.5,
                    evaluator_name="semantic.cosine",
                    category="SEMANTIC_REGRESSION",
                ),
            ],
            calls=[],
        )

        assert decision.overall.regression_rate == pytest.approx(1.0)
        assert decision.overall.equivalent_rate == pytest.approx(0.0)

    def test_non_semantic_regression_still_counts_by_delta(self) -> None:
        # Structural records are unaffected: any delta<0 is a regression
        # regardless of failure_categories.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_overall_regression_rate=1.0),
            comparisons=[_comparison(severity="low")],
            records=[
                _record(example_id="ex1", delta=-0.1, evaluator_name="structural.length"),
            ],
            calls=[],
        )

        assert decision.overall.regression_rate == pytest.approx(1.0)

    def test_inconclusive_when_all_comparisons_are_insufficient(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="insufficient", delta_avg_score=0.0)],
            records=[],
            calls=[],
        )

        assert decision.verdict == "inconclusive"


class TestAdvisoryEvaluators:
    """Advisory (blocking=False) records inform but never gate the verdict."""

    def test_advisory_records_do_not_gate(self) -> None:
        # Every advisory record regresses; every blocking record is clean.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[
                    _record(
                        example_id=f"j{i}",
                        delta=-1.0,
                        evaluator_name="llm_judge.equivalence",
                        blocking=False,
                    )
                    for i in range(4)
                ],
                *[_record(example_id=f"s{i}", delta=0.0) for i in range(40)],
            ],
            calls=[],
        )
        assert decision.verdict == "pass"
        assert decision.overall.regression_rate == pytest.approx(0.0)
        assert decision.advisory is not None
        assert decision.advisory.regression_rate == pytest.approx(1.0)

    def test_advisory_comparisons_go_to_advisory_regressions(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[
                _comparison(severity="critical", evaluator_name="semantic.cosine"),
                _comparison(severity="none", delta_avg_score=0.0),
            ],
            records=[
                _record(
                    example_id="c1",
                    delta=-0.05,
                    evaluator_name="semantic.cosine",
                    blocking=False,
                ),
                *[_record(example_id=f"s{i}", delta=0.0) for i in range(40)],
            ],
            calls=[],
        )
        assert decision.verdict == "pass"
        assert decision.blocking_regressions == []
        assert [r.evaluator_name for r in decision.advisory_regressions] == ["semantic.cosine"]

    def test_no_advisory_records_means_advisory_none(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"s{i}", delta=0.0) for i in range(40)],
            calls=[],
        )
        assert decision.advisory is None
        assert decision.advisory_regressions == []


class TestWilsonBudgets:
    """Rate budgets are CI-aware: small-n breaches are inconclusive, not fails."""

    def test_small_n_breach_is_inconclusive(self) -> None:
        # n=8, 3 regressions → observed 0.375 > 0.30 budget, but the 95%
        # Wilson interval reaches below the budget: can't confirm a breach.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_overall_regression_rate=0.30,
                min_equivalence_rate=0.0,
                max_critical_regressions=100,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[_record(example_id=f"r{i}", delta=-0.5) for i in range(3)],
                *[_record(example_id=f"e{i}", delta=0.0) for i in range(5)],
            ],
            calls=[],
        )
        assert decision.verdict == "inconclusive"
        assert decision.reason is not None
        assert "n=8" in decision.reason
        # The budget's denominator IS the run-level n here — restating it
        # per-budget would read as a second, different sample.
        assert "over n=8" not in decision.reason
        budget = next(b for b in decision.budget_results if b.name == "max_overall_regression_rate")
        assert budget.passed is False
        assert budget.conclusive is False
        assert budget.ci_low is not None and budget.ci_low < 0.30
        # Small-n inconclusive (with blocking records) is fixed by more
        # examples — the recommendation must keep saying so.
        assert decision.recommendations == [
            "Collect more examples before making a migration decision.",
        ]

    def test_large_n_breach_fails(self) -> None:
        # n=400, 160 regressions (0.40): Wilson low ~0.35 > 0.30 → real breach.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_overall_regression_rate=0.30,
                min_equivalence_rate=0.0,
                max_critical_regressions=1000,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[_record(example_id=f"r{i}", delta=-0.5) for i in range(160)],
                *[_record(example_id=f"e{i}", delta=0.0) for i in range(240)],
            ],
            calls=[],
        )
        assert decision.verdict == "fail"
        budget = next(b for b in decision.budget_results if b.name == "max_overall_regression_rate")
        assert budget.conclusive is True
        assert budget.passed is False

    def test_clean_small_n_passes(self) -> None:
        # n=8, zero regressions: wide CI must NOT block a clean run.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_overall_regression_rate=0.03,
                min_equivalence_rate=0.95,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(8)],
            calls=[],
        )
        assert decision.verdict == "pass"

    def test_min_equivalence_small_n_breach_is_inconclusive(self) -> None:
        # non-regression rate 0.625 < 0.65 budget at n=8: CI straddles.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.65,
                max_critical_regressions=100,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[_record(example_id=f"r{i}", delta=-0.5) for i in range(3)],
                *[_record(example_id=f"e{i}", delta=0.0) for i in range(5)],
            ],
            calls=[],
        )
        assert decision.verdict == "inconclusive"


class TestAllAdvisoryVerdict:
    def test_zero_blocking_records_is_inconclusive_not_pass(self) -> None:
        # Every evaluator advisory → nothing gates → there is no evidence
        # either way. That must read "inconclusive", never "pass".
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[
                _comparison(severity="critical", evaluator_name="semantic.cosine"),
            ],
            records=[
                _record(
                    example_id=f"a{i}",
                    delta=-0.1,
                    evaluator_name="semantic.cosine",
                    blocking=False,
                )
                for i in range(9)
            ],
            calls=[],
        )
        assert decision.verdict == "inconclusive"
        assert decision.reason is not None
        assert "advisory" in decision.reason

    def test_conclusive_cost_breach_fails_even_with_no_blocking_records(self) -> None:
        # Cost/latency budgets are derived from calls, not evaluator records:
        # a 100% cost increase against a 20% budget is a hard, conclusive
        # breach whether or not any evaluator gates quality. Reporting
        # "inconclusive" would hide a real failure the slices already flag.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_cost_increase=0.2),
            comparisons=[
                _comparison(severity="critical", evaluator_name="semantic.cosine"),
            ],
            records=[
                _record(
                    example_id=f"a{i}",
                    delta=-0.1,
                    evaluator_name="semantic.cosine",
                    blocking=False,
                )
                for i in range(9)
            ],
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.01, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.02, latency_ms=100),
            ],
        )
        assert decision.verdict == "fail"
        # The verdict changed, but the run still gated nothing on quality —
        # say so, and keep pointing at the config fix.
        assert decision.reason is not None
        assert "advisory" in decision.reason
        assert (
            "Set blocking: true on at least one trusted evaluator in "
            "evalshift.yaml to get a pass/fail verdict."
        ) in decision.recommendations

    def test_record_budgets_are_inconclusive_when_no_records_scored(self) -> None:
        # min_equivalence_rate reads "observed 1.00, passed" on zero records
        # because the rate defaults to 0/0 = 0. It must not also claim to be
        # conclusive — there is nothing behind the number.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[
                _comparison(severity="critical", evaluator_name="semantic.cosine"),
            ],
            records=[
                _record(
                    example_id=f"a{i}",
                    delta=-0.1,
                    evaluator_name="semantic.cosine",
                    blocking=False,
                )
                for i in range(9)
            ],
            calls=[],
        )
        by_name = {b.name: b for b in decision.budget_results}
        for name in (
            "max_overall_regression_rate",
            "min_equivalence_rate",
            "max_critical_regressions",
            "max_tool_argument_drift",
        ):
            assert by_name[name].conclusive is False, name
        # The call-derived budgets are unmeasured here too: this run has no
        # calls at all, so their 0.0 is the "nothing to divide" default.
        for name in ("max_cost_increase", "max_latency_increase"):
            assert by_name[name].conclusive is False, name

    def test_zero_blocking_records_recommendation_says_enable_blocking(self) -> None:
        # "Collect more examples" is wrong advice here: more advisory
        # examples can never produce a verdict. Point at the actual fix.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[
                _comparison(severity="critical", evaluator_name="semantic.cosine"),
            ],
            records=[
                _record(
                    example_id=f"a{i}",
                    delta=-0.1,
                    evaluator_name="semantic.cosine",
                    blocking=False,
                )
                for i in range(9)
            ],
            calls=[],
        )
        assert decision.recommendations == [
            "Set blocking: true on at least one trusted evaluator in "
            "evalshift.yaml to get a pass/fail verdict.",
        ]


_RESOLVED_POLICY_KEYS = {
    "max_overall_regression_rate",
    "max_critical_regressions",
    "min_equivalence_rate",
    "max_tool_argument_drift",
    "max_tool_divergence",
    "max_invariant_violations",
    "tool_argument_drift_floor",
    "max_cost_increase",
    "max_latency_increase",
    "fail_on_dropped_params",
    "slices",
}


class TestDecisionPolicy:
    """``decision.policy`` carries the resolved budgets the verdict was made under.

    The hosted server gates pull requests on its own, separately-edited policy
    unless the bundle tells it otherwise; riding the CLI's resolved policy
    inside every pushed bundle is what makes the two agree. See
    ``evaluate_migration_policy`` / ``inconclusive_decision``.
    """

    def test_evaluate_migration_policy_stamps_the_resolved_policy(self) -> None:
        policy = MigrationPolicy(max_overall_regression_rate=0.05)
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=policy,
            comparisons=[],
            records=[_record(example_id="e0", delta=0.0)],
            calls=[],
        )
        assert decision.policy == policy.model_dump(mode="json")
        assert decision.policy is not None
        assert set(decision.policy) == _RESOLVED_POLICY_KEYS

    def test_inconclusive_decision_has_no_policy(self) -> None:
        """The no-``migration_policy`` path: nothing was resolved to stamp."""
        decision = inconclusive_decision(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            comparisons=[],
            records=[_record(example_id="e0", delta=0.0)],
            calls=[],
        )
        assert decision.policy is None

    def test_round_trips_through_to_dict_and_from_dict(self) -> None:
        policy = MigrationPolicy(slices={"checkout": SliceMigrationPolicy()})
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=policy,
            comparisons=[],
            records=[_record(example_id="e0", delta=0.0)],
            calls=[],
        )
        restored = MigrationDecision.from_dict(
            json.loads(json.dumps(decision.to_dict())),
        )
        assert restored == decision
        assert restored.policy == policy.model_dump(mode="json")

    def test_from_dict_of_a_decision_without_a_policy_key_still_loads(self) -> None:
        """A ``migration_decision.json`` written before this field existed."""
        payload = inconclusive_decision(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            comparisons=[],
            records=[_record(example_id="e0", delta=0.0)],
            calls=[],
        ).to_dict()
        assert "policy" in payload
        del payload["policy"]
        restored = MigrationDecision.from_dict(payload)
        assert restored.policy is None


class TestFromDict:
    """``migration_decision.json`` is what every downstream surface describes."""

    def _decision(self) -> MigrationDecision:
        return evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(slices={"checkout": SliceMigrationPolicy()}),
            comparisons=[
                _comparison(severity="critical", slice_name="checkout"),
                _comparison(severity="low"),
            ],
            records=[
                _record(example_id="a", delta=-0.4),
                _record(example_id="b", delta=0.0),
            ],
            calls=[],
        )

    def test_round_trips_through_to_dict(self) -> None:
        decision = self._decision()
        assert MigrationDecision.from_dict(decision.to_dict()) == decision

    def test_survives_a_json_round_trip(self) -> None:
        """The real path: written by ``analyze``, read back by ``report``."""
        decision = self._decision()
        restored = MigrationDecision.from_dict(json.loads(json.dumps(decision.to_dict())))
        assert restored == decision
        assert restored.slices["checkout"].budget_results

    def test_rejects_a_payload_that_is_not_a_decision(self) -> None:
        """Callers treat this as "nothing persisted" and fall back."""
        with pytest.raises(ValueError, match="not a migration decision"):
            MigrationDecision.from_dict({"run_id": "r1"})

    def test_rejects_an_unknown_field(self) -> None:
        payload = self._decision().to_dict()
        payload["overall"]["invented_rate"] = 1.0
        with pytest.raises(ValueError, match="not a migration decision"):
            MigrationDecision.from_dict(payload)


# ---------------------------------------------------------------------------
# Evaluator-kind row selection (policy metrics must not depend on user names)
# ---------------------------------------------------------------------------


def _arg_record(
    example_id: str, target_score: float, *, kind: str = "tool_arguments"
) -> EvalRecord:
    return EvalRecord(
        run_id="r",
        prompt_id="p",
        example_id=example_id,
        evaluator_name="routing_args",
        kind=kind,
        source_score=1.0,
        target_score=target_score,
        delta=target_score - 1.0,
    )


class TestToolArgumentDriftSelection:
    def test_counts_rows_under_any_evaluator_name(self) -> None:
        metrics = _metrics(records=[_arg_record("e1", 0.0), _arg_record("e2", 1.0)], calls=[])
        assert metrics.tool_argument_drift_rate == pytest.approx(0.5)

    def test_ignores_other_evaluator_kinds(self) -> None:
        records = [
            EvalRecord(
                run_id="r",
                prompt_id="p",
                example_id="e1",
                evaluator_name="routing",
                kind="tool_selection",
                source_score=1.0,
                target_score=0.0,
                delta=-1.0,
            ),
        ]
        assert _metrics(records=records, calls=[]).tool_argument_drift_rate == 0.0

    def test_legacy_records_without_a_kind_still_feed_the_metric(self) -> None:
        """Rows checkpointed before ``kind`` existed keep the old name prefix."""
        legacy = EvalRecord(
            run_id="r",
            prompt_id="p",
            example_id="e1",
            evaluator_name="tool_arguments.routing_args",
            source_score=1.0,
            target_score=0.0,
            delta=-1.0,
        )
        assert _metrics(records=[legacy], calls=[]).tool_argument_drift_rate == pytest.approx(1.0)

    def test_semantic_regression_is_selected_by_kind(self) -> None:
        """A semantic row under a custom name is still judged by min_similarity."""
        drifted = EvalRecord(
            run_id="r",
            prompt_id="p",
            example_id="e1",
            evaluator_name="my_similarity",
            kind="semantic",
            source_score=1.0,
            target_score=0.93,
            delta=-0.07,
            metadata={"failure_categories": []},
        )
        metrics = _metrics(records=[drifted], calls=[])
        # No SEMANTIC_REGRESSION flag → within min_similarity → equivalent.
        assert metrics.regression_rate == 0.0
        assert metrics.equivalent_rate == pytest.approx(1.0)

    def test_the_budget_now_binds_under_a_user_chosen_name(self) -> None:
        """The end-to-end claim: a renamed evaluator can fail the migration.

        Stands in for the plan's ``analyze --run`` check against a real
        project. Before ``kind``, ``routing_args`` matched no prefix, the rate
        read 0.0, and ``max_tool_argument_drift`` never bound however wrong
        the arguments were.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_tool_argument_drift=0.01),
            comparisons=[],
            records=[_arg_record("e1", 0.0), _arg_record("e2", 1.0)],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.observed == pytest.approx(0.5)
        assert not drift.passed
        assert decision.verdict == "fail"


class TestToolArgumentDriftFloor:
    """Argument scoring is continuous; the drift *rate* must not be binary.

    Counting every ``delta < 0`` weighs a 0.98 exactly like a 0.0, which makes
    the shipped ``max_tool_argument_drift: 0.01`` budget unreachable for any
    migration between two different models.
    """

    def test_a_near_miss_argument_is_not_counted_as_drift(self) -> None:
        metrics = _metrics(records=[_arg_record("e1", 0.95), _arg_record("e2", 1.0)], calls=[])
        assert metrics.tool_argument_drift_rate == 0.0

    def test_a_materially_wrong_argument_is_counted_as_drift(self) -> None:
        metrics = _metrics(records=[_arg_record("e1", 0.0), _arg_record("e2", 1.0)], calls=[])
        assert metrics.tool_argument_drift_rate == pytest.approx(0.5)

    def test_the_floor_is_configurable(self) -> None:
        records = [_arg_record("e1", 0.5), _arg_record("e2", 1.0)]
        assert _metrics(records=records, calls=[], drift_floor=0.9).tool_argument_drift_rate == 0.5
        assert _metrics(records=records, calls=[], drift_floor=0.4).tool_argument_drift_rate == 0.0

    def test_an_improved_argument_score_is_never_drift(self) -> None:
        """The floor narrows the count; it must not widen it past ``delta < 0``."""
        better = EvalRecord(
            run_id="r",
            prompt_id="p",
            example_id="e1",
            evaluator_name="routing_args",
            kind="tool_arguments",
            source_score=0.2,
            target_score=0.8,
            delta=0.6,
        )
        assert _metrics(records=[better], calls=[]).tool_argument_drift_rate == 0.0

    def test_the_configured_floor_reaches_the_budget(self) -> None:
        """End-to-end: a near miss no longer breaches a 0.01 budget."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.01,
                tool_argument_drift_floor=0.9,
            ),
            comparisons=[],
            records=[_arg_record("e1", 0.95), _arg_record("e2", 1.0)],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.observed == 0.0
        assert drift.passed

    def test_a_slice_inherits_the_floor_from_the_top_level_policy(self) -> None:
        """A slice override must not silently reset the floor to its default."""
        from evalshift_cli.analysis.policy import _slice_policy

        base = MigrationPolicy(tool_argument_drift_floor=0.4)
        resolved = _slice_policy(base, SliceMigrationPolicy(max_tool_argument_drift=0.5))
        assert resolved.tool_argument_drift_floor == pytest.approx(0.4)

    def test_a_slice_can_override_the_floor(self) -> None:
        from evalshift_cli.analysis.policy import _slice_policy

        base = MigrationPolicy(tool_argument_drift_floor=0.4)
        resolved = _slice_policy(base, SliceMigrationPolicy(tool_argument_drift_floor=0.95))
        assert resolved.tool_argument_drift_floor == pytest.approx(0.95)


class TestToolArgumentDriftConclusiveness:
    """The drift budget has its own denominator, so ``n_records`` can't judge it.

    Every other record-derived budget is measured exactly when the scope
    scored a record. Tool-argument drift is not: its rate is counted over
    ``tool_arguments`` rows only. A suite that scores plenty of records but
    configures no tool-argument evaluator reads ``observed 0.00, passed`` —
    a confidently clean row for something that was never measured.
    ``conclusive`` is the only field that tells those apart, so it must
    follow the tool-argument row count.
    """

    def test_scored_records_without_tool_argument_rows_are_not_conclusive(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_tool_argument_drift=0.01),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(20)],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        # Plenty of scored records — just none of the kind this budget counts.
        assert decision.overall.n_records == 20
        assert drift.observed == 0.0
        assert drift.passed is True
        assert drift.conclusive is False

    def test_tool_argument_rows_with_no_drift_are_conclusive(self) -> None:
        """A real clean measurement must keep reading as one."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.01,
                tool_argument_drift_floor=0.9,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record("e1", 1.0), _arg_record("e2", 0.95)],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.observed == 0.0
        assert drift.passed is True
        assert drift.conclusive is True

    def test_a_material_drift_still_fails_conclusively(self) -> None:
        """Regression guard: the fix must not weaken the gate.

        A drifted tool-argument row is measured evidence, so the breach has
        to stay a conclusive ``fail`` — never softened to inconclusive.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.01,
                tool_argument_drift_floor=0.9,
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.0,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record("e1", 0.0), _arg_record("e2", 1.0)],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.observed == pytest.approx(0.5)
        assert drift.passed is False
        assert drift.conclusive is True
        assert decision.verdict == "fail"

    def test_the_count_is_scoped_per_slice(self) -> None:
        """A slice's drift row is measured only if *that slice* scored one."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_tool_argument_drift=0.01),
            comparisons=[
                _comparison(
                    severity="none",
                    slice_name="prose",
                    evaluator_name="structural.length",
                    delta_avg_score=0.0,
                ),
                _comparison(
                    severity="none",
                    slice_name="tools",
                    evaluator_name="routing_args",
                    delta_avg_score=0.0,
                ),
            ],
            records=[
                _record(example_id="e1", delta=0.0, evaluator_name="structural.length"),
                _arg_record("e2", 1.0),
            ],
            calls=[],
        )
        prose = next(
            b
            for b in decision.slices["prose"].budget_results
            if b.name == "max_tool_argument_drift"
        )
        tools = next(
            b
            for b in decision.slices["tools"].budget_results
            if b.name == "max_tool_argument_drift"
        )
        assert decision.slices["prose"].metrics.n_records == 1
        assert prose.conclusive is False
        assert tools.conclusive is True


class TestToolArgumentDriftInterval:
    """Drift is a binomial proportion, so it carries a Wilson interval too.

    Drifted tool-argument rows over tool-argument rows is exactly the shape
    ``max_overall_regression_rate`` already gets an interval for, and the
    hosted gate (``_PROPORTION_BUDGETS``) has always counted it as one. While
    the CLI computed none, a thin-sample drift breach read ``fail`` locally and
    ``inconclusive`` hosted off the very denominator the CLI itself uploaded.
    """

    def test_the_budget_reports_an_interval_around_its_rate(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.30,
                tool_argument_drift_floor=0.9,
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.0,
                max_critical_regressions=100,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[_arg_record(f"d{i}", 0.0) for i in range(3)],
                *[_arg_record(f"k{i}", 1.0) for i in range(5)],
            ],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.observed == pytest.approx(0.375)
        assert drift.ci_low is not None
        assert drift.ci_high is not None
        assert drift.ci_low < drift.observed < drift.ci_high

    def test_a_thin_sample_breach_is_no_longer_a_confident_fail(self) -> None:
        """3 drifted of 8 breaches 0.30, but the lower bound (~0.137) does not.

        The accepted loosening: a breach the sample cannot confirm stops being
        a local ``fail`` and reads ``inconclusive``, which is what the hosted
        gate already said about the same numbers.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.30,
                tool_argument_drift_floor=0.9,
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.0,
                max_critical_regressions=100,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[_arg_record(f"d{i}", 0.0) for i in range(3)],
                *[_arg_record(f"k{i}", 1.0) for i in range(5)],
            ],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.passed is False
        assert drift.conclusive is False
        assert drift.ci_low is not None and drift.ci_low < 0.30
        assert decision.verdict == "inconclusive"
        assert decision.reason is not None
        # The reason speaks the reader's language, not the config's.
        assert "tool-argument drift" in decision.reason
        assert "max_tool_argument_drift" not in decision.reason

    def test_a_well_sampled_breach_is_still_a_conclusive_fail(self) -> None:
        """160 drifted of 400: the lower bound (~0.353) clears 0.30."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.30,
                tool_argument_drift_floor=0.9,
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.0,
                max_critical_regressions=1000,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[_arg_record(f"d{i}", 0.0) for i in range(160)],
                *[_arg_record(f"k{i}", 1.0) for i in range(240)],
            ],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.passed is False
        assert drift.conclusive is True
        assert drift.ci_low is not None and drift.ci_low > 0.30
        assert decision.verdict == "fail"

    def test_a_held_budget_stays_conclusive_even_when_the_interval_spans_it(self) -> None:
        """The CLI's rule is asymmetric on purpose — do not adopt the server's.

        0 drifted of 8 gives an upper bound of ~0.324, which straddles the
        0.30 budget. The server's symmetric ``ci_low <= allowed <= ci_high``
        would call that unresolved; the CLI holds that a within-budget
        observation is conclusive by construction, so a wide interval can
        never block a clean run. The server is being moved onto this rule.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.30,
                tool_argument_drift_floor=0.9,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record(f"k{i}", 1.0) for i in range(8)],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.observed == 0.0
        assert drift.passed is True
        assert drift.ci_high is not None and drift.ci_high > 0.30
        assert drift.conclusive is True
        assert decision.verdict == "pass"


class TestWilsonConstant:
    """Both engines must emit byte-identical bounds, so both must use one z.

    The CLI shipped the rounded ``1.96`` (nominal coverage 95.0004%); the
    server uses the exact two-sided 95% normal quantile. The bounds differ by
    under 1e-5, which no verdict in the suite turns on, but a difference that
    small is exactly the kind that shows up as an unexplained mismatch between
    a local report and the hosted one.
    """

    def test_the_z_is_the_exact_two_sided_95_percent_quantile(self) -> None:
        from evalshift_cli.analysis.policy import _WILSON_Z

        assert _WILSON_Z == 1.959963984540054

    def test_the_bounds_match_the_servers_to_full_precision(self) -> None:
        from evalshift_cli.analysis.policy import _wilson_interval

        # Recomputed from the server's ``_wilson_interval`` at z=1.959963984540054.
        # The tolerance is tighter than the 9.03e-6 gap the rounded z produces,
        # so a drift back to 1.96 fails here rather than silently in a report.
        low, high = _wilson_interval(1, 10)
        assert low == pytest.approx(0.017876213095072868, abs=1e-12)
        assert high == pytest.approx(0.4041500267952385, abs=1e-12)


class TestCallBudgetConclusiveness:
    """The cost/latency budgets are call-derived, so calls can be missing too.

    ``_relative_increase`` returns 0.0 whenever either side has no error-free
    call, and 0.0 clears every ``ge=0.0`` budget. That renders as
    ``observed 0.00, passed`` — "the target did not cost more" — derived from
    zero samples. Only ``conclusive`` can tell that apart from a genuine
    measurement of no increase.
    """

    def _decision(
        self,
        *,
        calls: list[Call],
        policy: MigrationPolicy | None = None,
    ) -> MigrationDecision:
        return evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=policy
            or MigrationPolicy(max_overall_regression_rate=1.0, min_equivalence_rate=0.0),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id="e1", delta=0.0)],
            calls=calls,
        )

    def test_no_calls_at_all_leaves_both_budgets_unmeasured(self) -> None:
        decision = self._decision(calls=[])
        by_name = {b.name: b for b in decision.budget_results}
        for name in ("max_cost_increase", "max_latency_increase"):
            assert by_name[name].observed == 0.0, name
            assert by_name[name].passed is True, name
            assert by_name[name].conclusive is False, name

    def test_all_target_calls_errored_leaves_both_budgets_unmeasured(self) -> None:
        """Source side is healthy; there is simply nothing to compare it to."""
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.01, latency_ms=100),
                _call(example_id="e2", role="source", cost_usd=0.02, latency_ms=120),
                _call(
                    example_id="e1",
                    role="target",
                    cost_usd=0.0,
                    latency_ms=0,
                    error="rate limited",
                ),
                _call(
                    example_id="e2",
                    role="target",
                    cost_usd=0.0,
                    latency_ms=0,
                    error="rate limited",
                ),
            ],
        )
        by_name = {b.name: b for b in decision.budget_results}
        for name in ("max_cost_increase", "max_latency_increase"):
            assert by_name[name].observed == 0.0, name
            assert by_name[name].conclusive is False, name

    def test_healthy_calls_with_no_increase_are_conclusive(self) -> None:
        """A real measurement of "no increase" must keep reading as one."""
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.01, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.01, latency_ms=100),
            ],
        )
        by_name = {b.name: b for b in decision.budget_results}
        for name in ("max_cost_increase", "max_latency_increase"):
            assert by_name[name].observed == 0.0, name
            assert by_name[name].passed is True, name
            assert by_name[name].conclusive is True, name

    def test_a_real_cost_increase_still_fails_conclusively(self) -> None:
        """Regression guard: the fix must not weaken the gate.

        Measured calls are real evidence, so a breach stays a conclusive
        ``fail`` — never softened to inconclusive.
        """
        decision = self._decision(
            policy=MigrationPolicy(
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.0,
                max_cost_increase=0.10,
            ),
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.01, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.02, latency_ms=100),
            ],
        )
        cost = next(b for b in decision.budget_results if b.name == "max_cost_increase")
        assert cost.observed == pytest.approx(1.0)
        assert cost.passed is False
        assert cost.conclusive is True
        assert decision.verdict == "fail"


class TestZeroValuedCallBudgets:
    """Paired calls are necessary for a cost/latency ratio, not sufficient.

    ``_relative_increase`` also returns its ``0.0`` default when *both* roles
    average zero — an unpriced model pair, or calls whose ``latency_ms`` is 0.
    The call lists are non-empty there, so the older pairing check said
    "measured" and the row rendered "observed 0.00, passed, conclusive" for a
    cost nobody ever priced.
    """

    def _decision(
        self,
        *,
        calls: list[Call],
        policy: MigrationPolicy | None = None,
        comparisons: list[ComparisonResult] | None = None,
    ) -> MigrationDecision:
        return evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=policy
            or MigrationPolicy(max_overall_regression_rate=1.0, min_equivalence_rate=0.0),
            comparisons=comparisons or [_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id="e1", delta=0.0)],
            calls=calls,
        )

    def _unpriced_calls(self) -> list[Call]:
        return [
            _call(example_id="e1", role="source", cost_usd=0.0, latency_ms=0),
            _call(example_id="e1", role="target", cost_usd=0.0, latency_ms=0),
        ]

    def test_zero_on_both_sides_is_not_conclusive(self) -> None:
        decision = self._decision(calls=self._unpriced_calls())
        by_name = {b.name: b for b in decision.budget_results}
        for name in ("max_cost_increase", "max_latency_increase"):
            assert by_name[name].observed == 0.0, name
            assert by_name[name].passed is True, name
            assert by_name[name].conclusive is False, name

    def test_each_field_is_judged_on_its_own_values(self) -> None:
        """Unpriced models still produce real latencies — one, not both."""
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.0, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.0, latency_ms=110),
            ],
        )
        by_name = {b.name: b for b in decision.budget_results}
        assert by_name["max_cost_increase"].conclusive is False
        assert by_name["max_latency_increase"].conclusive is True

    def test_a_free_source_against_a_priced_target_stays_conclusive(self) -> None:
        """Only one side has to have measured something for the ratio to mean one."""
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.0, latency_ms=0),
                _call(example_id="e1", role="target", cost_usd=0.02, latency_ms=100),
            ],
            policy=MigrationPolicy(
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.0,
                max_cost_increase=2.0,
                max_latency_increase=2.0,
            ),
        )
        by_name = {b.name: b for b in decision.budget_results}
        assert by_name["max_cost_increase"].observed == pytest.approx(1.0)
        assert by_name["max_cost_increase"].conclusive is True
        assert by_name["max_latency_increase"].conclusive is True

    def test_a_priced_source_against_a_free_target_stays_conclusive(self) -> None:
        """A target that costs nothing is a real, clean measurement."""
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.02, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.0, latency_ms=0),
            ],
        )
        by_name = {b.name: b for b in decision.budget_results}
        assert by_name["max_cost_increase"].observed == 0.0
        assert by_name["max_cost_increase"].conclusive is True
        assert by_name["max_latency_increase"].conclusive is True

    def test_slice_budgets_are_unmeasured_too(self) -> None:
        """Slices read the same run-level calls, so they inherit the verdict."""
        decision = self._decision(
            calls=self._unpriced_calls(),
            comparisons=[
                _comparison(severity="none", slice_name="tools", delta_avg_score=0.0),
            ],
        )
        by_name = {b.name: b for b in decision.slices["tools"].budget_results}
        for name in ("max_cost_increase", "max_latency_increase"):
            assert by_name[name].conclusive is False, name

    def test_the_verdict_does_not_move(self) -> None:
        """Regression guard: ``conclusive`` is the only field that changes.

        The row still observes ``0.00``, which clears its budget, so
        ``_verdict_for`` never sees it — an unmeasured cost must not start
        blocking migrations that were previously clean.
        """
        decision = self._decision(calls=self._unpriced_calls())
        assert decision.verdict == "pass"
        assert decision.reason is None


class TestZeroValuedCallBudgetWarning:
    """``conclusive: false`` with a full raw.jsonl needs a reason beside it.

    A run with no calls explains itself. A run with calls whose cost is zero
    does not: the same flag, with evidence sitting right there, reads as a
    mystery unless something says the values were all zero.
    """

    def _decision(
        self,
        *,
        calls: list[Call],
        comparisons: list[ComparisonResult] | None = None,
    ) -> MigrationDecision:
        return evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_overall_regression_rate=1.0, min_equivalence_rate=0.0),
            comparisons=comparisons or [_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id="e1", delta=0.0)],
            calls=calls,
        )

    def test_an_unpriced_cost_budget_says_why(self) -> None:
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.0, latency_ms=100),
                _call(example_id="e2", role="source", cost_usd=0.0, latency_ms=120),
                _call(example_id="e1", role="target", cost_usd=0.0, latency_ms=90),
                _call(example_id="e2", role="target", cost_usd=0.0, latency_ms=95),
            ],
        )
        assert (
            "The cost increase budget could not be measured: all 4 error-free "
            "calls across both models recorded a cost of 0, so its observed "
            "0.00 is a default, not a measurement."
        ) in decision.recommendations
        assert not any("latency" in r for r in decision.recommendations)

    def test_an_untimed_latency_budget_says_why(self) -> None:
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.01, latency_ms=0),
                _call(example_id="e1", role="target", cost_usd=0.01, latency_ms=0),
            ],
        )
        assert (
            "The latency increase budget could not be measured: all 2 error-free "
            "calls across both models recorded a latency of 0, so its observed "
            "0.00 is a default, not a measurement."
        ) in decision.recommendations
        assert not any("cost" in r for r in decision.recommendations)

    def test_missing_calls_are_not_double_reported(self) -> None:
        """No calls at all is already visible; a second note is noise."""
        decision = self._decision(calls=[])
        assert not any("error-free calls" in r for r in decision.recommendations)

    def test_a_measured_ratio_is_not_warned(self) -> None:
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.01, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.01, latency_ms=100),
            ],
        )
        assert not any("error-free calls" in r for r in decision.recommendations)

    def test_the_note_is_emitted_once_per_run_not_once_per_scope(self) -> None:
        """Every scope reads the same run-level calls — repeating it is noise."""
        decision = self._decision(
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.0, latency_ms=0),
                _call(example_id="e1", role="target", cost_usd=0.0, latency_ms=0),
            ],
            comparisons=[
                _comparison(severity="none", slice_name="tools", delta_avg_score=0.0),
                _comparison(severity="none", slice_name="prose", delta_avg_score=0.0),
            ],
        )
        assert (
            sum("cost increase budget could not be measured" in r for r in decision.recommendations)
            == 1
        )
        assert (
            sum(
                "latency increase budget could not be measured" in r
                for r in decision.recommendations
            )
            == 1
        )

    def test_the_verdict_advice_still_comes_first(self) -> None:
        """Appended, never substituted — same rule as the granularity notes."""
        decision = self._decision(calls=self._unpriced())
        assert decision.recommendations[0] == "Safe to migrate under the configured policy."

    def _unpriced(self) -> list[Call]:
        return [
            _call(example_id="e1", role="source", cost_usd=0.0, latency_ms=0),
            _call(example_id="e1", role="target", cost_usd=0.0, latency_ms=0),
        ]


class TestUnmeasuredComparisons:
    """A blocking evaluator that scored nothing must not read as a pass."""

    def _unmeasured(self, evaluator_name: str) -> ComparisonResult:
        return _comparison(
            severity="insufficient",
            evaluator_name=evaluator_name,
            notes=[f"{UNMEASURED_NOTE_PREFIX} this evaluator scored no comparable pair"],
        )

    def test_pass_is_downgraded_when_a_blocking_evaluator_measured_nothing(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_overall_regression_rate=1.0, min_equivalence_rate=0.0),
            comparisons=[
                _comparison(severity="none", evaluator_name="tool_selection.routing"),
                self._unmeasured("llm_judge.equivalence"),
            ],
            records=[_record(example_id="ex1", delta=0.0)],
            calls=[],
        )
        assert decision.verdict == "conditional_pass"
        assert any("llm_judge.equivalence" in r for r in decision.recommendations)

    def test_an_all_unmeasured_run_names_them_instead_of_asking_for_more_data(self) -> None:
        # `_verdict_for` already returns inconclusive when every comparison is
        # insufficient, so this run never reaches the pass downgrade — but
        # "collect more examples" is the wrong diagnosis for rows that were
        # never applicable in the first place.
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[
                self._unmeasured("llm_judge.equivalence"),
                self._unmeasured("semantic.cosine"),
            ],
            records=[_record(example_id="ex1", delta=0.0)],
            calls=[],
        )
        assert decision.verdict == "inconclusive"
        assert any("llm_judge.equivalence, semantic.cosine" in r for r in decision.recommendations)
        assert not any("Collect more examples" in r for r in decision.recommendations)


# ---------------------------------------------------------------------------
# Sub-granular rate budgets (a budget finer than one row cannot be represented)
# ---------------------------------------------------------------------------


class TestSubGranularBudgetWarning:
    """A rate budget below its own ``1/denominator`` step has zero tolerance.

    Rate ceilings are counted over whole rows, so their achievable values are
    multiples of ``1/denominator``. A budget below that step cannot be
    represented: the next value up from ``0.0`` already breaches it, so the
    gate the user configured as "1%" is really "any at all". The arithmetic is
    correct — the honesty problem is that nothing says so.
    """

    def test_drift_budget_below_one_record_is_warned(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_tool_argument_drift=0.01),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record(f"e{i}", 1.0) for i in range(10)],
            calls=[],
        )
        assert (
            "The tool-argument drift budget of 1% (max_tool_argument_drift in "
            "evalshift.yaml) is below the 10% granularity of 10 tool-argument "
            "comparisons — effective tolerance is zero at this sample size."
        ) in decision.recommendations

    def test_a_representable_budget_is_not_warned(self) -> None:
        """10 rows put 0.2 exactly on the grid — nothing to flag."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.2,
                max_overall_regression_rate=0.2,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record(f"e{i}", 1.0) for i in range(10)],
            calls=[],
        )
        assert not any("granularity" in r for r in decision.recommendations)

    def test_an_unmeasured_denominator_is_not_double_reported(self) -> None:
        """Zero rows is already ``conclusive=False``; a second note is noise."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.01,
                max_overall_regression_rate=0.2,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(10)],
            calls=[],
        )
        assert not any("max_tool_argument_drift" in r for r in decision.recommendations)

    def test_an_intentional_zero_budget_is_not_warned(self) -> None:
        """``0.0`` already means "any at all fails" — that is not a mistake."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.0,
                max_overall_regression_rate=0.0,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record(f"e{i}", 1.0) for i in range(10)],
            calls=[],
        )
        assert not any("granularity" in r for r in decision.recommendations)

    def test_the_regression_rate_budget_is_warned_too(self) -> None:
        """Not a tool-argument quirk: every rate ceiling has the same grid."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.0,
                max_overall_regression_rate=0.03,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(20)],
            calls=[],
        )
        assert (
            "The overall regression rate budget of 3% (max_overall_regression_rate "
            "in evalshift.yaml) is below the 5% granularity of 20 scored "
            "comparisons — effective tolerance is zero at this sample size."
        ) in decision.recommendations

    def test_a_slice_budget_names_its_own_scope_and_denominator(self) -> None:
        """A slice's grid is coarser still — its own row count decides."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_tool_argument_drift=0.01,
                max_overall_regression_rate=1.0,
            ),
            comparisons=[
                _comparison(
                    severity="none",
                    slice_name="tools",
                    evaluator_name="routing_args",
                    delta_avg_score=0.0,
                ),
            ],
            records=[_arg_record(f"e{i}", 1.0) for i in range(4)],
            calls=[],
        )
        assert (
            "The tool-argument drift budget of 1% (max_tool_argument_drift in "
            "evalshift.yaml) in the 'tools' slice is below the 25% granularity "
            "of 4 tool-argument comparisons — effective tolerance is zero at "
            "this sample size."
        ) in decision.recommendations

    def test_count_and_ratio_budgets_are_never_warned(self) -> None:
        """``1/n`` says nothing about a count or a ratio of two averages."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_cost_increase=0.001,
                max_latency_increase=0.001,
                max_critical_regressions=0,
                max_tool_argument_drift=0.01,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record(f"e{i}", 1.0) for i in range(10)],
            calls=[
                _call(example_id="e1", role="source", cost_usd=0.01, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.01, latency_ms=100),
            ],
        )
        warnings = [r for r in decision.recommendations if "granularity" in r]
        assert warnings, "the rate ceilings should still have warned"
        assert not any(
            name in r
            for r in warnings
            for name in ("max_cost_increase", "max_latency_increase", "max_critical_regressions")
        )

    def test_the_equivalence_floor_is_not_warned(self) -> None:
        """``min_equivalence_rate`` is a floor, and the wording would be false.

        Below one row's granularity a floor collapses to *maximally lax* —
        only a 0% equivalence rate could fail it — which is the opposite of
        "effective tolerance is zero". Saying it anyway, in a change whose
        whole point is honesty, would be a new lie for an old one.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                min_equivalence_rate=0.01,
                max_overall_regression_rate=0.2,
                max_tool_argument_drift=0.2,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_arg_record(f"e{i}", 1.0) for i in range(10)],
            calls=[],
        )
        assert not any("min_equivalence_rate" in r for r in decision.recommendations)


class TestBudgetDenominators:
    """Every budget reports the sample ``observed`` was measured over.

    ``BUNDLE_SPEC.md`` gives ``denominator`` three distinct readings and the
    governed gate keys on all three: ``0`` means counted-and-empty (so
    ``observed`` is a default), a positive integer means that many units were
    counted, and ``null`` means no sample size was reported at all — which
    only bundles predating the field may say. The CLI always knows its own
    denominators, so every budget it emits carries an integer.
    """

    def test_record_budgets_report_the_scored_record_count(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(12)],
            calls=[],
        )
        by_name = {b.name: b for b in decision.budget_results}
        assert decision.overall.n_records == 12
        for name in (
            "max_overall_regression_rate",
            "min_equivalence_rate",
            "max_critical_regressions",
        ):
            assert by_name[name].denominator == 12, name

    def test_errored_rows_are_not_counted_in_the_denominator(self) -> None:
        """The denominator must be the *scored* rows, not every row handed in.

        A broken measurement is the only row left to exclude: a pair an
        evaluator measured nothing on now writes no row at all.
        """
        scored = [_record(example_id=f"e{i}", delta=0.0) for i in range(4)]
        errored = _record(example_id="broken", delta=0.0)
        errored.error = "evaluator error: boom"
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[*scored, errored],
            calls=[],
        )
        by_name = {b.name: b for b in decision.budget_results}
        assert by_name["max_overall_regression_rate"].denominator == 4

    def test_tool_drift_reports_its_own_denominator(self) -> None:
        """Drift is counted over ``tool_arguments`` rows, not over every row."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_tool_argument_drift=0.5),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[
                *[_arg_record(f"a{i}", 1.0) for i in range(3)],
                *[_record(example_id=f"e{i}", delta=0.0) for i in range(7)],
            ],
            calls=[],
        )
        by_name = {b.name: b for b in decision.budget_results}
        assert by_name["max_tool_argument_drift"].denominator == 3
        assert by_name["max_overall_regression_rate"].denominator == 10

    def test_a_budget_that_measured_nothing_reports_zero_not_null(self) -> None:
        """``0`` and ``null`` are different statements; the CLI only ever says ``0``.

        A scope that scored rows but ran no ``tool_arguments`` evaluator
        counted zero of them. ``null`` would tell the server "no sample size
        reported", which sends it back to pre-P6a behaviour on a bundle that
        knows the answer.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(6)],
            calls=[],
        )
        drift = next(b for b in decision.budget_results if b.name == "max_tool_argument_drift")
        assert drift.denominator == 0
        assert drift.conclusive is False

    def test_call_ratio_budgets_count_the_error_free_calls(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(4)],
            calls=[
                _call(example_id="e0", role="source", cost_usd=0.01, latency_ms=100),
                _call(example_id="e1", role="source", cost_usd=0.02, latency_ms=120),
                _call(example_id="e0", role="target", cost_usd=0.01, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=0.02, latency_ms=90, error="boom"),
            ],
        )
        by_name = {b.name: b for b in decision.budget_results}
        # Three error-free calls: two source, one target. The errored target
        # call is excluded from the averages, so it is excluded from the count.
        assert by_name["max_cost_increase"].denominator == 3
        assert by_name["max_latency_increase"].denominator == 3

    def test_one_sided_calls_leave_the_ratio_denominator_at_zero(self) -> None:
        """The ratio divides one role's average by the other's.

        With no target call there is nothing to divide, so nothing was counted
        — ``0``, matching the ``conclusive: false`` the same absence produces.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(4)],
            calls=[_call(example_id="e0", role="source", cost_usd=0.01, latency_ms=100)],
        )
        by_name = {b.name: b for b in decision.budget_results}
        for name in ("max_cost_increase", "max_latency_increase"):
            assert by_name[name].denominator == 0, name
            assert by_name[name].conclusive is False, name

    def test_an_all_zero_ratio_still_reports_the_calls_it_counted(self) -> None:
        """Both-sides-zero is unmeasured (phase P5) but not uncounted.

        ``conclusive`` carries the doubt; the denominator stays honest about
        how many calls were averaged, so the two fields say different things
        rather than one of them lying.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(4)],
            calls=[
                _call(example_id="e0", role="source", cost_usd=0.0, latency_ms=100),
                _call(example_id="e0", role="target", cost_usd=0.0, latency_ms=100),
            ],
        )
        cost = next(b for b in decision.budget_results if b.name == "max_cost_increase")
        assert cost.denominator == 2
        assert cost.conclusive is False

    def test_slice_budgets_report_the_slice_denominator(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(slices={"short": SliceMigrationPolicy()}),
            comparisons=[
                _comparison(severity="none", delta_avg_score=0.0),
                _comparison(
                    severity="none",
                    slice_name="short",
                    evaluator_name="structural.length",
                    delta_avg_score=0.0,
                ),
            ],
            records=[_record(example_id=f"e{i}", delta=0.0) for i in range(9)],
            calls=[],
        )
        slice_budgets = {b.name: b for b in decision.slices["short"].budget_results}
        assert slice_budgets["max_overall_regression_rate"].denominator == 9

    def test_an_older_decision_without_denominators_round_trips_as_null(self) -> None:
        """``null`` survives ``from_dict`` — it must never be read as ``0``."""
        payload = json.loads(
            json.dumps(
                evaluate_migration_policy(
                    run_id="r1",
                    source_model="src",
                    target_model="tgt",
                    policy=MigrationPolicy(),
                    comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
                    records=[_record(example_id="e0", delta=0.0)],
                    calls=[],
                ).to_dict()
            )
        )
        for budget in payload["budget_results"]:
            budget.pop("denominator")
        restored = MigrationDecision.from_dict(payload)
        assert all(b.denominator is None for b in restored.budget_results)


# ---------------------------------------------------------------------------
# The two tool-selection axes (S3)
# ---------------------------------------------------------------------------


def _conformance_record(
    example_id: str,
    source_score: float,
    target_score: float,
    *,
    evaluator_name: str = "routing",
) -> EvalRecord:
    """A ``tool_selection.conformance`` row, categorised as the evaluator does.

    Mirrors ``ToolSelectionEvaluator._record``: a target below the source is
    drift, and *both* sides below full conformance is a shared ground-truth
    miss. The real labels are pinned end-to-end by
    ``tests/integration/test_scoring_semantics.py``.
    """
    categories = [
        *([TOOL_SELECTION_DRIFT] if target_score < source_score else []),
        *([TOOL_GROUND_TRUTH_MISS] if source_score < 1.0 and target_score < 1.0 else []),
    ]
    return EvalRecord(
        run_id="r",
        prompt_id="p",
        example_id=example_id,
        evaluator_name=evaluator_name,
        kind=KIND_CONFORMANCE,
        source_score=source_score,
        target_score=target_score,
        delta=target_score - source_score,
        metadata={"failure_categories": categories} if categories else {},
    )


def _divergence_record(
    example_id: str,
    target_score: float,
    *,
    evaluator_name: str = "routing",
) -> EvalRecord:
    """A ``tool_selection.divergence`` row: source is its own baseline at 1.0."""
    return EvalRecord(
        run_id="r",
        prompt_id="p",
        example_id=example_id,
        evaluator_name=evaluator_name,
        kind=KIND_DIVERGENCE,
        source_score=1.0,
        target_score=target_score,
        delta=target_score - 1.0,
        metadata={"failure_categories": [TOOL_SELECTION_DRIFT]} if target_score < 1.0 else {},
    )


def _decide(
    records: list[EvalRecord],
    *,
    policy: MigrationPolicy | None = None,
    comparisons: list[ComparisonResult] | None = None,
) -> MigrationDecision:
    return evaluate_migration_policy(
        run_id="r1",
        source_model="src",
        target_model="tgt",
        policy=policy or MigrationPolicy(),
        comparisons=comparisons if comparisons is not None else [],
        records=records,
        calls=[],
    )


def _budgets(decision: MigrationDecision) -> dict[str, BudgetResult]:
    return {b.name: b for b in decision.budget_results}


class TestToolSelectionKindRegistration:
    """S3 step 1 — both axis slugs must be registered in the policy layer.

    ``evaluators/base.py`` documents why selection is by ``kind`` and not by
    the user-chosen evaluator name: a rename must not unhook a budget. The
    other half of that rule is that a slug the policy layer never registered
    leaves its budget permanently unbound — which is what ``tool_selection``
    was before this change, registered nowhere at all.
    """

    def test_the_policy_layer_registers_the_evaluators_own_slugs(self) -> None:
        from evalshift_cli.analysis.policy import _TOOL_CONFORMANCE_KIND, _TOOL_DIVERGENCE_KIND

        assert _TOOL_CONFORMANCE_KIND == KIND_CONFORMANCE
        assert _TOOL_DIVERGENCE_KIND == KIND_DIVERGENCE

    def test_a_row_with_no_kind_slug_does_not_crash_the_new_selectors(self) -> None:
        """Neither axis has a legacy name prefix, so the fallback must miss cleanly.

        ``_is_kind`` indexes ``_LEGACY_KIND_PREFIXES`` directly, so asking it
        about a slug with no legacy form raises ``KeyError`` on any row
        checkpointed before ``kind`` existed.
        """
        legacy = EvalRecord(
            run_id="r",
            prompt_id="p",
            example_id="e1",
            evaluator_name="routing",
            source_score=1.0,
            target_score=1.0,
            delta=0.0,
        )
        budgets = _budgets(_decide([legacy]))
        assert budgets["max_tool_divergence"].denominator == 0
        assert budgets["max_overall_regression_rate"].denominator == 1


class TestToolDivergenceBudget:
    """S3 step 2 — ``max_tool_divergence`` over the divergence axis.

    Shaped exactly like ``max_tool_argument_drift``: a rate on its own
    denominator, with a Wilson interval and the same measured/unmeasured rule.
    """

    def test_the_rate_is_divergent_rows_over_divergence_rows(self) -> None:
        decision = _decide(
            [
                _divergence_record("e1", 0.0),
                _divergence_record("e2", 0.0),
                _divergence_record("e3", 1.0),
                _divergence_record("e4", 1.0),
            ],
        )
        assert _budgets(decision)["max_tool_divergence"].observed == pytest.approx(0.5)

    def test_the_budget_binds_under_a_user_chosen_evaluator_name(self) -> None:
        """The point of selecting on ``kind``: a rename cannot unhook the gate."""
        decision = _decide(
            [_divergence_record(f"e{i}", 0.0, evaluator_name="my_router") for i in range(10)],
            policy=MigrationPolicy(max_tool_divergence=0.1),
        )
        budget = _budgets(decision)["max_tool_divergence"]
        assert budget.observed == pytest.approx(1.0)
        assert not budget.passed
        assert budget.conclusive
        assert decision.verdict == "fail"

    def test_conformance_rows_are_not_counted_as_divergence(self) -> None:
        decision = _decide(
            [
                _conformance_record("e1", 1.0, 0.0),
                _conformance_record("e2", 1.0, 0.0),
                _divergence_record("e3", 1.0),
            ],
        )
        budget = _budgets(decision)["max_tool_divergence"]
        assert budget.denominator == 1
        assert budget.observed == 0.0

    def test_a_scope_with_no_divergence_rows_measured_nothing(self) -> None:
        """``0`` and a clean pass are different statements — see the drift budget."""
        budget = _budgets(_decide([_record(example_id=f"e{i}", delta=0.0) for i in range(6)]))[
            "max_tool_divergence"
        ]
        assert budget.denominator == 0
        assert budget.conclusive is False

    def test_a_thin_sample_breach_is_not_a_confident_fail(self) -> None:
        decision = _decide(
            [_divergence_record("e1", 0.0), *[_divergence_record(f"e{i}", 1.0) for i in range(3)]],
            policy=MigrationPolicy(max_tool_divergence=0.1),
        )
        budget = _budgets(decision)["max_tool_divergence"]
        assert budget.observed == pytest.approx(0.25)
        assert not budget.passed
        assert not budget.conclusive
        assert decision.verdict == "inconclusive"

    def test_an_unconfirmed_breach_names_its_own_denominator(self) -> None:
        """The reason must say which sample is thin — the divergence rows.

        The run-level ``n`` counts every blocking record, but this budget is
        counted over its own axis rows only. "n=44 is too small" about a
        budget measured over 4 rows sends the user growing the wrong sample.
        """
        decision = _decide(
            [
                *[_record(example_id=f"ok{i}", delta=0.0) for i in range(40)],
                _divergence_record("d0", 0.0),
                *[_divergence_record(f"d{i}", 1.0) for i in range(1, 4)],
            ],
            policy=MigrationPolicy(max_tool_divergence=0.1),
        )
        assert decision.verdict == "inconclusive"
        assert decision.reason is not None
        assert "n=44" in decision.reason
        assert "over n=4" in decision.reason
        # Prose, not config identifiers — and the budget explains itself.
        assert "max_tool_divergence" not in decision.reason
        assert "tool-selection divergence" in decision.reason
        assert "different tools" in decision.reason

    def test_a_held_budget_reports_an_interval(self) -> None:
        budget = _budgets(_decide([_divergence_record(f"e{i}", 1.0) for i in range(8)]))[
            "max_tool_divergence"
        ]
        assert budget.ci_low is not None
        assert budget.ci_high is not None
        assert budget.passed
        assert budget.conclusive

    def test_a_slice_can_override_the_budget(self) -> None:
        from evalshift_cli.analysis.policy import _slice_policy

        resolved = _slice_policy(
            MigrationPolicy(max_tool_divergence=0.2),
            SliceMigrationPolicy(max_tool_divergence=0.5),
        )
        assert resolved.max_tool_divergence == pytest.approx(0.5)

    def test_a_sub_granular_budget_is_warned_like_the_other_rate_ceilings(self) -> None:
        decision = _decide(
            [_divergence_record(f"e{i}", 1.0) for i in range(10)],
            policy=MigrationPolicy(max_tool_divergence=0.01),
        )
        assert any(
            "The tool-selection divergence budget of 1% (max_tool_divergence in "
            "evalshift.yaml) is below the 10% granularity of 10 tool-selection "
            "comparisons" in note
            for note in decision.recommendations
        ), decision.recommendations


class TestSharedGroundTruthMissIsNotEquivalence:
    """S3 step 3 — a zero delta both models earned by failing is not evidence.

    Ground truth captured from the source model that the source model then
    fails means the harness is misconfigured. Its ``0.0 / 0.0`` reads as a
    zero delta, which ``_is_equivalent`` filed as equivalence — the single
    number that let the personalButler run claim ``equivalent_rate: 1.0``.
    """

    def test_a_shared_miss_leaves_the_equivalence_denominator(self) -> None:
        decision = _decide(
            [
                *[_conformance_record(f"e{i}", 0.0, 0.0) for i in range(10)],
                *[_divergence_record(f"e{i}", 0.0) for i in range(9)],
                _divergence_record("e9", 1.0),
            ],
        )
        assert decision.overall.n_records == 10
        assert decision.overall.equivalent_rate == pytest.approx(0.1)
        assert decision.overall.regression_rate == pytest.approx(0.9)

    def test_the_equivalence_budget_denominator_shrinks_with_them(self) -> None:
        decision = _decide(
            [
                *[_conformance_record(f"c{i}", 0.0, 0.0) for i in range(10)],
                *[_divergence_record(f"d{i}", 1.0) for i in range(4)],
            ],
        )
        budget = _budgets(decision)["min_equivalence_rate"]
        assert budget.denominator == 4
        assert budget.observed == pytest.approx(1.0)

    def test_a_partial_miss_shared_by_both_models_is_excluded_too(self) -> None:
        """The zero delta is the artefact whatever height both sides missed at."""
        decision = _decide(
            [_conformance_record("e1", 0.5, 0.5), _divergence_record("e2", 1.0)],
        )
        assert decision.overall.n_records == 1

    def test_a_conformance_row_the_target_lost_ground_on_is_still_a_regression(self) -> None:
        """Both sides below 1.0 *and* the target below the source is a real finding.

        Excluding on the ground-truth flag alone would delete it.
        """
        decision = _decide([_conformance_record("e1", 0.8, 0.3)])
        assert decision.overall.n_records == 1
        assert decision.overall.regression_rate == pytest.approx(1.0)

    def test_a_conformance_row_the_target_improved_on_is_still_counted(self) -> None:
        decision = _decide([_conformance_record("e1", 0.2, 0.6)])
        assert decision.overall.n_records == 1
        assert decision.overall.improved_rate == pytest.approx(1.0)

    def test_a_divergence_row_at_zero_on_both_sides_is_never_excluded(self) -> None:
        """The exclusion is a conformance rule; divergence has no ground truth."""
        decision = _decide([_divergence_record("e1", 1.0)])
        assert decision.overall.n_records == 1

    def test_the_excluded_rows_are_named_in_the_recommendations(self) -> None:
        decision = _decide(
            [
                *[_conformance_record(f"c{i}", 0.0, 0.0) for i in range(3)],
                _divergence_record("d0", 1.0),
            ],
        )
        assert any(
            "3 tool-selection conformance comparisons are excluded" in note
            for note in decision.recommendations
        ), decision.recommendations

    def test_the_excluded_rows_still_report_their_failure_category(self) -> None:
        """Excluded from the rates, never from the evidence."""
        decision = _decide(
            [
                *[_conformance_record(f"c{i}", 0.0, 0.0) for i in range(3)],
                _divergence_record("d0", 1.0),
            ],
        )
        counts = {c.category: c.count for c in decision.failure_categories}
        assert counts[TOOL_GROUND_TRUTH_MISS] == 3

    def test_a_run_that_is_nothing_but_shared_misses_says_so(self) -> None:
        """Not ``pass``, and not the "every evaluator is advisory" reason either."""
        decision = _decide([_conformance_record(f"c{i}", 0.0, 0.0) for i in range(4)])
        assert decision.overall.n_records == 0
        assert decision.verdict == "inconclusive"
        assert decision.reason is not None
        assert "ground truth" in decision.reason
        assert "advisory" not in decision.reason

    def test_an_all_excluded_run_is_not_told_to_collect_more_examples(self) -> None:
        """The ``reason`` was right and the advice beside it was the opposite.

        More examples from a misconfigured harness produce more shared misses,
        which are excluded in turn — the denominator stays empty however many
        are added. The fix is the setup, and the recommendation must say so.
        """
        decision = _decide([_conformance_record(f"c{i}", 0.0, 0.0) for i in range(4)])
        assert not any("Collect more examples" in r for r in decision.recommendations)
        assert any("Fix the eval harness" in r for r in decision.recommendations)

    def test_small_n_inconclusive_still_asks_for_more_examples(self) -> None:
        """The replacement is scoped to the empty-by-exclusion case only."""
        decision = _decide(
            [
                *[_divergence_record(f"r{i}", 0.5) for i in range(3)],
                *[_divergence_record(f"e{i}", 1.0) for i in range(5)],
            ],
            policy=MigrationPolicy(
                max_overall_regression_rate=0.30,
                min_equivalence_rate=0.0,
                max_critical_regressions=100,
            ),
            comparisons=[_comparison(severity="none", delta_avg_score=0.0)],
        )
        assert decision.verdict == "inconclusive"
        assert decision.recommendations == [
            "Collect more examples before making a migration decision.",
        ]


class TestDenominatorsAfterTheAxisSplit:
    """S3 step 4 — every denominator re-checked now that cardinality doubled."""

    def test_both_axes_count_toward_the_overall_regression_denominator(self) -> None:
        """An overall rate is over measurements, not over examples.

        Conformance and divergence ask different questions against different
        baselines, so a five-example suite legitimately contributes ten rows.
        """
        decision = _decide(
            [
                *[_conformance_record(f"e{i}", 1.0, 1.0) for i in range(5)],
                *[_divergence_record(f"e{i}", 1.0) for i in range(5)],
            ],
        )
        budgets = _budgets(decision)
        for name in (
            "max_overall_regression_rate",
            "min_equivalence_rate",
            "max_critical_regressions",
        ):
            assert budgets[name].denominator == 10, name

    def test_tool_argument_drift_is_unmoved_by_the_second_axis(self) -> None:
        decision = _decide(
            [
                *[_arg_record(f"a{i}", 1.0) for i in range(3)],
                *[_conformance_record(f"e{i}", 1.0, 1.0) for i in range(5)],
                *[_divergence_record(f"e{i}", 1.0) for i in range(5)],
            ],
        )
        assert _budgets(decision)["max_tool_argument_drift"].denominator == 3

    def test_the_divergence_denominator_is_the_divergence_rows_only(self) -> None:
        decision = _decide(
            [
                *[_arg_record(f"a{i}", 1.0) for i in range(3)],
                *[_conformance_record(f"e{i}", 1.0, 1.0) for i in range(5)],
                *[_divergence_record(f"e{i}", 1.0) for i in range(4)],
            ],
        )
        assert _budgets(decision)["max_tool_divergence"].denominator == 4

    def test_the_call_ratio_denominators_are_unmoved_by_the_second_axis(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[],
            records=[
                *[_conformance_record(f"e{i}", 1.0, 1.0) for i in range(3)],
                *[_divergence_record(f"e{i}", 1.0) for i in range(3)],
            ],
            calls=[
                _call(example_id="e0", role="source", cost_usd=1.0, latency_ms=10),
                _call(example_id="e0", role="target", cost_usd=1.0, latency_ms=10),
            ],
        )
        budgets = _budgets(decision)
        assert budgets["max_cost_increase"].denominator == 2
        assert budgets["max_latency_increase"].denominator == 2


# ---------------------------------------------------------------------------
# "Measured nothing" is a property of the budget, not of the reader
# ---------------------------------------------------------------------------


class TestBudgetMeasured:
    """``conclusive`` is False for two unrelated reasons; only one is blindness.

    A *breached* budget is inconclusive when its CI still includes the ceiling
    — the sample measured something, it just cannot resolve the breach. A
    *passing* budget is conclusive by construction however wide its interval,
    so ``passed and not conclusive`` is reachable only by the ``0/0`` default.
    """

    def _budget(self, *, passed: bool, conclusive: bool) -> BudgetResult:
        return BudgetResult(
            name="max_tool_divergence",
            observed=0.0 if passed else 0.5,
            allowed=0.2,
            passed=passed,
            conclusive=conclusive,
            denominator=0 if passed and not conclusive else 30,
        )

    def test_a_budget_that_passed_on_an_empty_sample_measured_nothing(self) -> None:
        assert not self._budget(passed=True, conclusive=False).measured

    def test_a_breach_the_ci_cannot_confirm_still_measured_something(self) -> None:
        assert self._budget(passed=False, conclusive=False).measured

    def test_a_resolved_budget_measured_something(self) -> None:
        assert self._budget(passed=True, conclusive=True).measured
        assert self._budget(passed=False, conclusive=True).measured

    def test_the_property_is_not_serialised_into_the_decision(self) -> None:
        """``BudgetResult`` is the bundle's budget object, ``additionalProperties: false``."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none")],
            records=[_record(example_id="ex1", delta=0.0)],
            calls=[],
        )
        for row in decision.to_dict()["budget_results"]:
            assert "measured" not in row

    def test_a_divergence_budget_with_no_rows_is_unmeasured_in_a_real_decision(self) -> None:
        """The exact shape the shipped run had: seven budgets, one of them blind."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[_comparison(severity="none")],
            records=[_record(example_id="ex1", delta=0.0)],
            calls=[],
        )
        divergence = next(b for b in decision.budget_results if b.name == "max_tool_divergence")
        assert divergence.passed
        assert not divergence.measured


class TestUnmeasuredGatingEvaluators:
    """One definition of "which gates were blind", shared with the prose channel."""

    def _unmeasured(self, evaluator_name: str) -> ComparisonResult:
        return _comparison(
            severity="insufficient",
            evaluator_name=evaluator_name,
            notes=[f"{UNMEASURED_NOTE_PREFIX} this evaluator scored no comparable pair"],
        )

    def _advisory_unmeasured(self, evaluator_name: str) -> ComparisonResult:
        return _comparison(
            severity="insufficient",
            evaluator_name=evaluator_name,
            notes=[
                f"{UNMEASURED_NOTE_PREFIX} this evaluator scored no comparable pair",
                f"{ADVISORY_NOTE_PREFIX} blocking is false in evalshift.yaml — "
                "this axis reports and never gates",
            ],
        )

    def test_it_names_every_gating_evaluator_that_scored_nothing(self) -> None:
        names = unmeasured_gating_evaluators(
            comparisons=[
                self._unmeasured("semantic.cosine"),
                self._unmeasured("llm_judge.equivalence"),
                _comparison(severity="none", evaluator_name="tool_selection.routing"),
            ],
            records=[_record(example_id="ex1", delta=0.0)],
        )
        assert names == ["llm_judge.equivalence", "semantic.cosine"]

    def test_an_advisory_evaluator_never_counts_as_a_blind_gate(self) -> None:
        names = unmeasured_gating_evaluators(
            comparisons=[
                self._unmeasured("semantic.cosine"),
                self._unmeasured("llm_judge.equivalence"),
            ],
            records=[
                _record(
                    example_id="ex1", delta=0.0, evaluator_name="semantic.cosine", blocking=False
                ),
                _record(example_id="ex1", delta=0.0, evaluator_name="llm_judge.equivalence"),
            ],
        )
        assert names == ["llm_judge.equivalence"]

    def test_an_advisory_evaluator_with_no_rows_is_known_from_its_note(self) -> None:
        """Zero records leave no ``blocking`` flag to read, so the advisory
        note the analysis stage stamps on the synthesized comparison is the
        only surviving trace of the config — exactly the case this function
        used to misfire on, naming a ``blocking: false`` evaluator as a
        blind gate."""
        names = unmeasured_gating_evaluators(
            comparisons=[
                self._advisory_unmeasured("semantic.cosine"),
                self._unmeasured("llm_judge.equivalence"),
            ],
            records=[_record(example_id="ex1", delta=0.0)],
        )
        assert names == ["llm_judge.equivalence"]

    def test_advisory_silence_never_demotes_a_pass(self) -> None:
        """An advisory evaluator that scored nothing gates nothing by design:
        it must not push a clean pass to conditional_pass, and the prose
        must not call it blocking."""
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(),
            comparisons=[
                _comparison(severity="none", delta_avg_score=0.0),
                self._advisory_unmeasured("semantic.cosine"),
            ],
            records=[_record(example_id="ex1", delta=0.0)],
            calls=[],
        )
        assert decision.verdict == "pass"
        assert not any("semantic.cosine" in r for r in decision.recommendations)

    def test_it_is_the_same_set_the_recommendations_name(self) -> None:
        """The prose channel and the facts channel cannot be allowed to drift."""
        comparisons = [
            _comparison(severity="none", evaluator_name="tool_selection.routing"),
            self._unmeasured("llm_judge.equivalence"),
            self._unmeasured("semantic.cosine"),
        ]
        records = [_record(example_id="ex1", delta=0.0)]
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(max_overall_regression_rate=1.0, min_equivalence_rate=0.0),
            comparisons=comparisons,
            records=records,
            calls=[],
        )
        names = unmeasured_gating_evaluators(comparisons=comparisons, records=records)
        assert ", ".join(names) in " ".join(decision.recommendations)


class TestSliceBudgetsBlock:
    """A slice budget gates the run exactly like an overall one.

    A slice budget the user wrote is a gate they asked for. Demoting its
    breach to ``conditional_pass`` made the CLI answer ``conditional_pass``
    where the hosted gate answered ``fail`` on the same bundle.
    """

    def _policy(self, *, slice_ceiling: float) -> MigrationPolicy:
        return MigrationPolicy(
            max_overall_regression_rate=0.5,
            max_critical_regressions=0,
            min_equivalence_rate=0.0,
            slices={
                "billing": SliceMigrationPolicy(max_overall_regression_rate=slice_ceiling),
            },
        )

    def _comparisons(self) -> list[ComparisonResult]:
        return [
            _comparison(severity="none", delta_avg_score=0.0),
            _comparison(
                severity="none",
                slice_name="billing",
                evaluator_name="billing.judge",
                delta_avg_score=0.0,
            ),
        ]

    def _records(self, *, scoped: int, regressed: int) -> list[EvalRecord]:
        clean = [_record(example_id=f"ok{i}", delta=0.0) for i in range(20)]
        return clean + [
            _record(
                example_id=f"bill{i}",
                delta=-0.5 if i < regressed else 0.0,
                evaluator_name="billing.judge",
            )
            for i in range(scoped)
        ]

    def test_a_conclusive_slice_budget_breach_fails_the_run(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=self._policy(slice_ceiling=0.0),
            comparisons=self._comparisons(),
            records=self._records(scoped=20, regressed=20),
            calls=[],
        )

        assert decision.verdict == "fail"
        assert decision.slices["billing"].verdict == "fail"
        # The overall budgets are all green, so the prose has to name the
        # slice budget that actually blocked or the fail is undebuggable.
        assert any(
            "'billing' slice breached its overall regression rate budget" in rec
            for rec in decision.recommendations
        )
        assert not any("max_overall_regression_rate" in rec for rec in decision.recommendations)

    def test_an_unconfirmed_slice_budget_breach_is_inconclusive(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=self._policy(slice_ceiling=0.5),
            comparisons=self._comparisons(),
            records=self._records(scoped=3, regressed=2),
            calls=[],
        )

        assert decision.verdict == "inconclusive"
        assert decision.reason is not None
        assert "overall regression rate" in decision.reason
        assert "'billing' slice" in decision.reason
        assert "max_overall_regression_rate" not in decision.reason
        # The slice's own sample, not the run-level n=23.
        assert "over n=3" in decision.reason

    def test_a_slice_budget_that_holds_leaves_the_run_passing(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=self._policy(slice_ceiling=1.0),
            comparisons=self._comparisons(),
            records=self._records(scoped=20, regressed=20),
            calls=[],
        )

        assert decision.verdict == "pass"

    def test_a_slice_row_that_restates_an_overall_one_adds_no_line(self) -> None:
        """The call-derived budgets are run-level, so every slice copies them.

        One breach, one line — not one per slice of a run that happens to
        have three.
        """
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(
                max_overall_regression_rate=1.0,
                min_equivalence_rate=0.0,
                max_cost_increase=0.0,
                slices={"billing": SliceMigrationPolicy()},
            ),
            comparisons=self._comparisons(),
            records=self._records(scoped=20, regressed=0),
            calls=[
                _call(example_id="e1", role="source", cost_usd=1.0, latency_ms=100),
                _call(example_id="e1", role="target", cost_usd=2.0, latency_ms=100),
            ],
        )

        assert decision.verdict == "fail"
        assert not [r for r in decision.recommendations if "slice breached its" in r]


def _argument_record(
    example_id: str,
    *,
    source_score: float = 1.0,
    target_score: float = 1.0,
    provenance: str | None = "captured",
) -> EvalRecord:
    """A ``tool_arguments`` row, stamped as the evaluator stamps it.

    ``provenance=None`` is an ``against: source`` row: it never claimed to
    measure correctness, so it carries no ground-truth stamp.
    """
    metadata: dict[str, Any] = (
        {"against": "expected", "gt_provenance": provenance}
        if provenance is not None
        else {"per_call": []}
    )
    return EvalRecord(
        run_id="r",
        prompt_id="p",
        example_id=example_id,
        evaluator_name="routing_args",
        kind=KIND_ARGUMENTS,
        source_score=source_score,
        target_score=target_score,
        delta=target_score - source_score,
        metadata=metadata,
    )


def _has_provenance_note(decision: MigrationDecision) -> bool:
    return any("source model's own recorded calls" in note for note in decision.recommendations)


class TestSourceDerivedGroundTruthDisclosure:
    """`capture sync` promotes the source's own arguments as ground truth.

    So ``against: expected`` pins ``source_score`` at 1.0 by construction and
    degenerates to target-deviation-from-source. No scoring changes — but a
    run that never says so lets ``source_score: 1.0`` read as evidence.
    """

    def test_disclosed_when_every_scored_row_is_source_derived(self) -> None:
        decision = _decide([_argument_record(f"e{i}") for i in range(3)])
        assert _has_provenance_note(decision), decision.recommendations
        assert any("3 " in note for note in decision.recommendations)
        # The note names the axis in words, not by its record-kind slug.
        assert not any("tool_arguments" in note for note in decision.recommendations)

    def test_silent_once_a_human_has_reviewed_one_row(self) -> None:
        """One checked row means the gate is no longer uniformly degenerate."""
        decision = _decide(
            [
                *[_argument_record(f"e{i}") for i in range(3)],
                _argument_record("e3", provenance="reviewed"),
            ],
        )
        assert not _has_provenance_note(decision), decision.recommendations

    def test_silent_on_a_row_with_mixed_expectations(self) -> None:
        decision = _decide([_argument_record("e0", provenance="mixed")])
        assert not _has_provenance_note(decision), decision.recommendations

    def test_silent_for_against_source_rows(self) -> None:
        decision = _decide([_argument_record(f"e{i}", provenance=None) for i in range(3)])
        assert not _has_provenance_note(decision), decision.recommendations

    def test_silent_when_no_tool_arguments_evaluator_ran(self) -> None:
        decision = _decide([_record(example_id="e1", delta=0.0)])
        assert not _has_provenance_note(decision), decision.recommendations

    def test_disclosed_without_a_migration_policy_too(self) -> None:
        """The degeneracy is a property of the suite, not of the configured gate."""
        decision = inconclusive_decision(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            comparisons=[],
            records=[_argument_record("e0")],
            calls=[],
        )
        assert _has_provenance_note(decision), decision.recommendations


class TestFailOnDroppedParams:
    """``fail_on_dropped_params`` turns a report caveat into a gate.

    A team replaying captures that pinned ``response_format`` may consider a
    target that cannot honour it unmigratable regardless of how the scores
    came out — the arm did not run the experiment they wrote.
    """

    def _decide(
        self, *, policy: MigrationPolicy, dropped: dict[str, list[str]] | None
    ) -> MigrationDecision:
        return evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=policy,
            comparisons=[_comparison(severity="none")],
            records=[_record(example_id="ex1", delta=0.0)],
            calls=[],
            dropped_params=dropped,
        )

    def test_passes_when_the_knob_is_off(self) -> None:
        decision = self._decide(
            policy=MigrationPolicy(),
            dropped={"tgt": ["tool_choice"]},
        )
        assert decision.verdict == "pass"

    def test_fails_and_names_the_model_and_params(self) -> None:
        decision = self._decide(
            policy=MigrationPolicy(fail_on_dropped_params=True),
            dropped={"tgt": ["response_format", "tool_choice"]},
        )
        assert decision.verdict == "fail"
        assert decision.reason is not None
        assert "tgt" in decision.reason
        assert "response_format" in decision.reason
        assert "tool_choice" in decision.reason

    def test_no_effect_when_nothing_was_dropped(self) -> None:
        decision = self._decide(policy=MigrationPolicy(fail_on_dropped_params=True), dropped={})
        assert decision.verdict == "pass"
        assert decision.reason is None

    def test_defaults_to_no_record_for_callers_that_do_not_pass_one(self) -> None:
        """Old artefacts have no ``dropped_params``; the gate must stay silent."""
        decision = self._decide(policy=MigrationPolicy(fail_on_dropped_params=True), dropped=None)
        assert decision.verdict == "pass"

    def test_keeps_an_existing_reason_alongside_its_own(self) -> None:
        decision = evaluate_migration_policy(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            policy=MigrationPolicy(fail_on_dropped_params=True),
            comparisons=[],
            records=[],
            calls=[],
            dropped_params={"tgt": ["tool_choice"]},
        )
        assert decision.verdict == "fail"
        assert decision.reason is not None
        # The "no blocking evaluator records" explanation survives.
        assert "no blocking evaluator records" in decision.reason
        assert "tool_choice" in decision.reason


class TestAMultiRoundDivergenceCountsAsDiverged:
    """``max_tool_divergence`` counts ``delta < 0``, and that is the whole rule.

    A teacher-forced replay scores the mean over its rounds, so an example
    that matched the source in round 1 and diverged in round 2 lands at
    ``0.5`` rather than ``0.0``. The budget must still count it: "the target
    called different tools somewhere in the loop" is the finding, and a
    threshold on the *mean* would let a four-round example hide three
    divergences behind one match. No policy code implements this — the
    evaluator's mean does — so this test is what locks the semantics in.
    """

    def test_a_partially_diverged_row_is_counted(self) -> None:
        diverged_in_one_round = EvalRecord(
            run_id="r",
            prompt_id="p",
            example_id="e1",
            evaluator_name="routing",
            kind=KIND_DIVERGENCE,
            source_score=1.0,
            target_score=0.5,
            delta=-0.5,
            metadata={
                "mode": "exact",
                "failure_categories": [TOOL_SELECTION_DRIFT],
                "rounds": [
                    {"round": 0, "source_names": ["a"], "target_names": ["a"]},
                    {"round": 1, "source_names": ["b"], "target_names": ["c"]},
                ],
            },
        )
        budget = _budgets(_decide([diverged_in_one_round, _divergence_record("e2", 1.0)]))[
            "max_tool_divergence"
        ]
        assert budget.denominator == 2
        assert budget.observed == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Hand-written trace rules: the ``max_invariant_violations`` budget
# ---------------------------------------------------------------------------


def _invariant_record(
    example_id: str,
    source_score: float,
    target_score: float,
    *,
    blocking: bool = True,
    rule_id: str = "no-v1",
) -> EvalRecord:
    def broken(score: float) -> list[dict[str, Any]]:
        return (
            []
            if score == 1.0
            else [
                {
                    "rule_id": rule_id,
                    "rule_type": "forbidden",
                    "tool": "refund_v1",
                    "round_index": 0,
                    "detail": "x",
                }
            ]
        )

    metadata: dict[str, Any] = {
        "rules_checked": [rule_id],
        "source_violations": broken(source_score),
        "target_violations": broken(target_score),
    }
    if target_score < 1.0:
        metadata["failure_categories"] = [INVARIANT_VIOLATION]
    return EvalRecord(
        run_id="r",
        prompt_id="p",
        example_id=example_id,
        evaluator_name="payments",
        kind="trace_invariants",
        source_score=source_score,
        target_score=target_score,
        delta=target_score - source_score,
        metadata=metadata,
        blocking=blocking,
    )


def _insufficient(evaluator_name: str = "payments") -> ComparisonResult:
    """One comparison too small for a paired test -- what n < 5 produces."""
    return _comparison(evaluator_name=evaluator_name, severity="insufficient")


class TestInvariantViolationBudget:
    def test_counts_examples_whose_target_broke_a_rule(self) -> None:
        decision = _decide(
            [
                _invariant_record("e1", 1.0, 0.0),
                _invariant_record("e2", 1.0, 1.0),
                _invariant_record("e3", 1.0, 0.5),
            ],
            policy=MigrationPolicy(max_invariant_violations=5),
        )
        budget = _budgets(decision)["max_invariant_violations"]
        assert (budget.observed, budget.allowed, budget.denominator) == (2.0, 5.0, 3)
        assert budget.passed and budget.conclusive

    def test_a_violation_the_source_shares_still_counts(self) -> None:
        decision = _decide([_invariant_record("e1", 0.0, 0.0)])
        assert _budgets(decision)["max_invariant_violations"].observed == 1.0
        assert decision.verdict == "fail"

    def test_a_broken_rule_fails_a_suite_too_small_for_statistics(self) -> None:
        """Review Focus 3: every comparison insufficient must not soften a broken rule."""
        decision = _decide(
            [_invariant_record("e1", 1.0, 0.0)],
            comparisons=[_insufficient()],
        )
        assert decision.verdict == "fail"
        assert decision.reason is not None
        assert "max_invariant_violations" in decision.reason

    def test_no_invariant_rows_emits_no_budget_row(self) -> None:
        """Review Focus 4: projects without trace rules see no new row."""
        decision = _decide([_divergence_record("e1", 1.0)])
        assert "max_invariant_violations" not in _budgets(decision)

    def test_advisory_rules_never_gate(self) -> None:
        decision = _decide([_invariant_record("e1", 1.0, 0.0, blocking=False)])
        assert "max_invariant_violations" not in _budgets(decision)
        assert decision.verdict != "fail"

    def test_selected_by_kind_not_name(self) -> None:
        record = _invariant_record("e1", 1.0, 0.0).model_copy(update={"evaluator_name": "anything"})
        assert _budgets(_decide([record]))["max_invariant_violations"].observed == 1.0

    def test_slice_override_applies(self) -> None:
        records = [_invariant_record("e1", 1.0, 0.0)]
        comparisons = [
            _comparison(evaluator_name="payments", slice_name="checkout", severity="none"),
            _comparison(evaluator_name="payments", slice_name="all", severity="none"),
        ]
        decision = _decide(
            records,
            policy=MigrationPolicy(
                max_invariant_violations=0,
                slices={"checkout": SliceMigrationPolicy(max_invariant_violations=3)},
            ),
            comparisons=comparisons,
        )
        checkout = {b.name: b for b in decision.slices["checkout"].budget_results}
        assert checkout["max_invariant_violations"].passed
        assert decision.verdict == "fail"  # the overall budget of 0 still breaches

    def test_shared_violation_recommends_reviewing_the_rule(self) -> None:
        decision = _decide([_invariant_record("e1", 0.0, 0.0)])
        assert any("no-v1" in line and "owner" in line for line in decision.recommendations)

    def test_the_budget_has_a_label_and_meaning(self) -> None:
        from evalshift_cli.analysis.policy import BUDGET_LABELS, BUDGET_MEANINGS

        assert BUDGET_LABELS["max_invariant_violations"] == "Trace-rule violations"
        assert "trace rule" in BUDGET_MEANINGS["max_invariant_violations"]
