"""Tests for the deterministic half of the insights package.

``build_facts`` and ``fallback_insight`` are pure — no network, no model —
which is what makes the generator testable without mocking a provider.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from evalshift_cli.analysis.policy import (
    BlockingRegression,
    BudgetResult,
    FailureCategoryCount,
    _dropped_params_reason,
    _invariant_reason,
    evaluate_migration_policy,
)
from evalshift_cli.analysis.statistics import ComparisonResult
from evalshift_cli.config.models import MigrationPolicy, SliceMigrationPolicy
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.insights.facts import (
    BUDGET_BREACHED,
    BUDGET_UNCONFIRMED,
    BUDGET_UNMEASURED,
    BUDGET_WITHIN,
    _scoped_budget_label,
    build_facts,
    describe_reply,
)
from evalshift_cli.insights.generator import banned_identifiers
from evalshift_cli.insights.templates import fallback_insight
from evalshift_cli.reports.economics import PromptEconomics
from evalshift_cli.runner.models import Call
from tests.unit.insights_factories import (
    COST_TARGET,
    SEVERITY_REASON,
    budgets,
    comparisons,
    converter_decision,
    decision,
    records,
    role,
    slice_invariant_decision,
    unmeasured_budget,
    unmeasured_comparison,
)


def test_figures_are_pre_rendered_as_display_strings(sample_run: dict[str, Any]) -> None:
    """The model copies these verbatim; it never sees a raw float to round."""
    facts = build_facts(**sample_run)
    assert facts.rendered["cost_source_total_usd"] == "$0.0101"
    assert facts.rendered["cost_target_total_usd"] == "$0.0204"
    assert facts.rendered["cost_delta_pct"] == "+102%"
    assert facts.rendered["effect_size"] == "−2.51"
    assert facts.rendered["p_value"] == "< 0.0001"
    assert facts.rendered["n_examples"] == "21"


def test_a_resolvable_p_value_is_printed_in_full(sample_run: dict[str, Any]) -> None:
    """``0.0000`` would be wrong in a way a reader cannot detect."""
    sample_run["comparisons"] = comparisons(p_corrected=0.0312)
    assert build_facts(**sample_run).rendered["p_value"] == "0.0312"


def test_headline_stats_come_from_the_worst_comparison(sample_run: dict[str, Any]) -> None:
    """``rendered`` is figures only, so the evaluator name lives beside it."""
    facts = build_facts(**sample_run)
    assert facts.worst_evaluator == "semantic.cosine"


def test_verdict_and_budget_counts_are_rendered(sample_run: dict[str, Any]) -> None:
    facts = build_facts(**sample_run)
    assert facts.verdict == "pass"
    assert facts.rendered["verdict"] == "PASS"
    assert facts.rendered["budgets_passed"] == "6"
    assert facts.rendered["budgets_total"] == "6"
    assert facts.rendered["cost_ceiling_pct"] == "+200%"
    assert facts.rendered["latency_ceiling_pct"] == "+30%"


def test_delta_spread_is_rendered_from_the_examples(sample_run: dict[str, Any]) -> None:
    facts = build_facts(**sample_run)
    assert facts.rendered["worst_delta"] == "−0.0800"
    assert facts.rendered["median_delta"] == "−0.0300"
    assert facts.rendered["best_delta"] == "+0.0200"


def test_latency_is_not_reported_as_a_measurement_on_a_cached_replay(
    sample_run: dict[str, Any],
) -> None:
    """No role measured live latency; a percentage there is a fiction."""
    assert build_facts(**sample_run).rendered["latency_delta_pct"] == "not comparable"


def test_latency_is_not_comparable_when_only_one_side_ran_live(
    sample_run: dict[str, Any],
) -> None:
    # A target served from the cache (or with some rounds replayed) has no
    # live latency sample; its 0.0 average is "unmeasured", not "instant", so
    # it must not render as a -100% latency change.
    sample_run["economics"] = PromptEconomics(
        source=role(live_calls=21, cached_calls=0, latency_ms_avg=1000.0),
        target=role(
            live_calls=0,
            cached_calls=21,
            latency_ms_avg=0.0,
            total_cost_usd=COST_TARGET,
        ),
    )
    assert build_facts(**sample_run).rendered["latency_delta_pct"] == "not comparable"


def test_latency_delta_is_rendered_when_both_sides_ran_live(
    sample_run: dict[str, Any],
) -> None:
    sample_run["economics"] = PromptEconomics(
        source=role(live_calls=21, cached_calls=0, latency_ms_avg=1000.0),
        target=role(
            live_calls=21,
            cached_calls=0,
            latency_ms_avg=1473.0,
            total_cost_usd=COST_TARGET,
        ),
    )
    assert build_facts(**sample_run).rendered["latency_delta_pct"] == "+47.3%"


def test_allowed_numbers_covers_every_rendered_figure(sample_run: dict[str, Any]) -> None:
    facts = build_facts(**sample_run)
    for value in facts.rendered.values():
        assert value in facts.allowed_numbers


def test_allowed_numbers_admits_bare_numerals_and_counting(
    sample_run: dict[str, Any],
) -> None:
    """``+102%`` also admits ``102``, and "15 of 21" must be writable."""
    facts = build_facts(**sample_run)
    assert "102" in facts.allowed_numbers
    assert "0.0101" in facts.allowed_numbers
    assert all(str(index) in facts.allowed_numbers for index in range(22))


def test_allowed_numbers_admits_the_numerals_inside_model_ids(
    sample_run: dict[str, Any],
) -> None:
    """Otherwise naming ``gemini-3.1-flash`` reads as an invented figure."""
    facts = build_facts(**sample_run)
    assert "3.1" in facts.allowed_numbers
    assert "2.5" in facts.allowed_numbers


def test_budget_limits_and_failure_categories_are_carried(
    sample_run: dict[str, Any],
) -> None:
    sample_run["decision"] = decision(
        verdict="conditional_pass",
        blocking=[
            BlockingRegression(
                prompt_id="replay",
                evaluator_name="semantic.cosine",
                slice_name="all",
                severity="high",
                delta_avg_score=-0.031,
                effect_size=-2.51,
            ),
        ],
        categories=[FailureCategoryCount(category="missing_field", count=15)],
    )
    facts = build_facts(**sample_run)
    assert facts.budget_limits["max_overall_regression_rate"] == "3%"
    assert facts.blocking_evaluators == ["semantic.cosine"]
    assert facts.failure_categories == [("missing_field", 15)]
    assert "15" in facts.allowed_numbers


def test_the_trace_rule_budget_limit_is_a_count_not_a_rate(
    sample_run: dict[str, Any],
) -> None:
    """``max_invariant_violations`` counts examples, so its limit renders bare.

    Rendered as a rate, an allowance of three violations would read "300%".
    """
    sample_run["decision"] = decision(
        budget_results=[
            *budgets(),
            BudgetResult(
                name="max_invariant_violations",
                observed=1.0,
                allowed=3.0,
                passed=True,
                conclusive=True,
                denominator=4,
            ),
        ],
    )
    facts = build_facts(**sample_run)
    assert facts.budget_limits["max_invariant_violations"] == "3"


def test_regression_samples_are_capped_and_truncated(sample_run: dict[str, Any]) -> None:
    facts = build_facts(**sample_run)
    assert len(facts.regression_samples) <= 8
    for sample in facts.regression_samples:
        assert len(sample.source_output) <= 2000
        assert len(sample.target_output) <= 2000
        assert len(sample.input_text) <= 2000


def test_regression_samples_are_worst_first(sample_run: dict[str, Any]) -> None:
    facts = build_facts(**sample_run)
    deltas = [sample.delta for sample in facts.regression_samples]
    assert deltas == sorted(deltas)


def test_a_clean_run_has_no_regression_samples(passing_run: dict[str, Any]) -> None:
    assert build_facts(**passing_run).regression_samples == []


def test_fallback_insight_uses_only_rendered_figures(sample_run: dict[str, Any]) -> None:
    facts = build_facts(**sample_run)
    insight = fallback_insight(facts)
    assert "$0.0101" in insight.economics_summary
    assert "$0.0204" in insight.economics_summary
    assert insight.model == "none"
    assert insight.findings == []


def test_the_fallback_calls_the_costs_totals(sample_run: dict[str, Any]) -> None:
    """Each figure is one model's whole-run cost; "across 42 calls" read as per call."""
    summary = fallback_insight(build_facts(**sample_run)).economics_summary
    assert summary.startswith("Total cost $0.0101 → $0.0204 (")
    assert "per call" not in summary


def test_fallback_handles_a_run_with_no_regressions(passing_run: dict[str, Any]) -> None:
    """A clean run must not produce prose about regressions that do not exist."""
    insight = fallback_insight(build_facts(**passing_run))
    assert "regression" not in insight.advisory_summary.lower() or "0" in insight.advisory_summary


def test_fallback_never_emits_an_empty_prose_field(passing_run: dict[str, Any]) -> None:
    """Server-side ``min_length=1`` — an empty summary is a 400 at finalize."""
    insight = fallback_insight(build_facts(**passing_run))
    for text in (
        insight.verdict_summary,
        insight.advisory_summary,
        insight.economics_summary,
        insight.recommendation,
        insight.model,
    ):
        assert text.strip()
        assert len(text) <= 2000


def test_fallback_recommendation_tracks_the_verdict(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = decision(verdict="fail")
    insight = fallback_insight(build_facts(**sample_run))
    assert "not" in insight.recommendation.lower()


def test_fallback_names_the_top_failure_category_in_words(
    sample_run: dict[str, Any],
) -> None:
    """A machine label like TOOL_SELECTION_DRIFT never reaches the reader."""
    sample_run["decision"] = decision(
        verdict="fail",
        categories=[FailureCategoryCount(category="TOOL_SELECTION_DRIFT", count=8)],
    )
    insight = fallback_insight(build_facts(**sample_run))
    assert "Different tools chosen" in insight.advisory_summary
    assert "TOOL_SELECTION_DRIFT" not in insight.advisory_summary


# ---------------------------------------------------------------------------
# S4 — the narrative may not assert equivalence it cannot support
# ---------------------------------------------------------------------------

#: The sentence a real run shipped over a suite where nine of ten examples
#: called an entirely different tool. It is the acceptance test for this
#: phase: no facts block may make it writable.
SHIPPED_EQUIVALENCE_CLAIM = (
    "The target model achieved a 100% equivalence rate with the source model, "
    "indicating no loss in output quality or behavioral consistency."
)


def _unmeasured_run(sample_run: dict[str, Any]) -> dict[str, Any]:
    """The reference run with every blocking row gone from the rates.

    Either nothing gated quality or every gating row was excluded as a shared
    ground-truth miss — both land on ``n_records == 0``, where ``_rate``
    returns ``0.0`` for want of a denominator.
    """
    sample_run["decision"] = decision(
        verdict="inconclusive",
        n_records=0,
        equivalent_rate=0.0,
        categories=[FailureCategoryCount(category="TOOL_GROUND_TRUTH_MISS", count=10)],
    )
    return sample_run


def test_rates_over_an_empty_denominator_are_not_rendered_as_figures(
    sample_run: dict[str, Any],
) -> None:
    """``0%`` is a default, not a measurement, and reads as "nothing regressed"."""
    from evalshift_cli.insights.facts import NOT_MEASURED

    facts = build_facts(**_unmeasured_run(sample_run))
    for key in ("equivalence_rate_pct", "regression_rate_pct", "improved_rate_pct"):
        assert facts.rendered[key] == NOT_MEASURED, key
        assert not any(char.isdigit() for char in facts.rendered[key]), key


def test_the_shipped_equivalence_claim_is_not_writable(sample_run: dict[str, Any]) -> None:
    """The permit-list is what stops the model restating an unmeasured rate."""
    from evalshift_cli.insights.generator import validate_numbers

    facts = build_facts(**_unmeasured_run(sample_run))
    assert "100%" not in facts.allowed_numbers
    assert validate_numbers(SHIPPED_EQUIVALENCE_CLAIM, facts.allowed_numbers) == ["100%"]


def test_the_facts_say_why_the_rates_are_missing(sample_run: dict[str, Any]) -> None:
    """A blank is ambiguous; the reason has to travel with the absence."""
    facts = build_facts(**_unmeasured_run(sample_run))
    assert "not measured" in facts.rates_basis.lower()
    # Digit-free on purpose: anything numeric here would widen the permit-list.
    assert not any(char.isdigit() for char in facts.rates_basis)


def test_a_measured_run_still_reports_its_rates(sample_run: dict[str, Any]) -> None:
    """The marker must not leak onto runs that did measure something."""
    facts = build_facts(**sample_run)
    assert facts.rendered["equivalence_rate_pct"] == "71.4%"
    assert facts.rates_basis == ""


def test_the_fallback_does_not_report_unmeasured_rates_as_zero(
    sample_run: dict[str, Any],
) -> None:
    """``Regression rate 0%, equivalence 0%`` over a run that measured nothing."""
    insight = fallback_insight(build_facts(**_unmeasured_run(sample_run)))
    assert "0%" not in insight.advisory_summary
    assert "not measured" in insight.advisory_summary.lower()


# ---------------------------------------------------------------------------
# A gate that measured nothing is not a gate that passed
# ---------------------------------------------------------------------------

#: The recommendation a real run shipped (``r_20260823_main_chat_8de58b``)
#: while its own report body, two sections above it, named the two blocking
#: evaluators that had scored nothing. The acceptance case for this phase.
SHIPPED_ALL_CONSTRAINTS_CLAIM = (
    "Proceed with migration as all hard constraints are met, but perform "
    "additional manual validation on tool-calling logic."
)


def _blind_gate_run(sample_run: dict[str, Any]) -> dict[str, Any]:
    """The reference run with one dead budget and two dead blocking evaluators.

    ``n_records`` stays positive on purpose: the tool evaluators scored
    plenty, so the run-level ``rates_basis`` guard never fires and every rate
    renders as a real figure. That is exactly the shape of the shipped run —
    the run-level flag cannot see an evaluator that died on its own.
    """
    sample_run["decision"] = decision(
        verdict="conditional_pass",
        budget_results=[*budgets(), unmeasured_budget()],
    )
    sample_run["comparisons"] = [
        unmeasured_comparison("llm_judge.equivalence"),
        unmeasured_comparison("semantic.cosine"),
    ]
    return sample_run


def test_a_budget_that_measured_nothing_is_not_counted_as_passed(
    sample_run: dict[str, Any],
) -> None:
    """ "7 of 7 budgets passed" over a budget that was handed an empty sample."""
    facts = build_facts(**_blind_gate_run(sample_run))
    assert facts.rendered["budgets_total"] == "7"
    assert facts.rendered["budgets_passed"] == "6"
    assert facts.rendered["budgets_unmeasured"] == "1"


def test_the_facts_name_the_budget_that_measured_nothing(
    sample_run: dict[str, Any],
) -> None:
    """A count alone cannot be acted on; the narrative has to say which gate."""
    facts = build_facts(**_blind_gate_run(sample_run))
    # Display name, not the config key — this list flows into reader prose.
    assert facts.unmeasured_budgets == ["Tool-selection divergence"]


def test_the_facts_name_the_blocking_evaluators_that_scored_nothing(
    sample_run: dict[str, Any],
) -> None:
    """The same two names the decision's own recommendations carry."""
    facts = build_facts(**_blind_gate_run(sample_run))
    assert facts.unmeasured_evaluators == ["llm_judge.equivalence", "semantic.cosine"]


def test_an_advisory_evaluator_that_scored_nothing_is_not_called_a_gate(
    sample_run: dict[str, Any],
) -> None:
    """Advisory silence gates nothing by design — listing it dilutes the signal."""
    run = _blind_gate_run(sample_run)
    run["records"] = records(advisory=("semantic.cosine",))
    facts = build_facts(**run)
    assert facts.unmeasured_evaluators == ["llm_judge.equivalence"]


def test_the_facts_say_why_coverage_is_incomplete(sample_run: dict[str, Any]) -> None:
    """A name without a reason is a fact the model will explain for itself."""
    facts = build_facts(**_blind_gate_run(sample_run))
    assert "measured nothing" in facts.coverage_basis.lower()
    # Digit-free on purpose: the basis is not in the permit-list, so a numeral
    # here is one the validator would reject out of the model's own prose.
    assert not any(char.isdigit() for char in facts.coverage_basis)


def test_a_fully_measured_run_carries_no_coverage_marker(
    sample_run: dict[str, Any],
) -> None:
    """The marker must not leak onto runs where every gate really scored."""
    facts = build_facts(**sample_run)
    assert facts.coverage_basis == ""
    assert facts.unmeasured_budgets == []
    assert facts.unmeasured_evaluators == []
    assert facts.rendered["budgets_passed"] == "6"
    assert facts.rendered["budgets_total"] == "6"
    assert facts.rendered["budgets_unmeasured"] == "0"


def test_the_fallback_does_not_report_a_blind_gate_as_a_clean_sweep(
    sample_run: dict[str, Any],
) -> None:
    """The templated voice shipped the same claim the generated one did."""
    insight = fallback_insight(build_facts(**_blind_gate_run(sample_run)))
    assert "7 of 7" not in insight.verdict_summary
    assert "6 of 7" in insight.verdict_summary
    assert "Tool-selection divergence" in insight.verdict_summary
    assert "max_tool_divergence" not in insight.verdict_summary
    assert "llm_judge.equivalence" in insight.verdict_summary


def test_the_fallback_is_unchanged_when_every_gate_measured(
    sample_run: dict[str, Any],
) -> None:
    """No caveat where there is nothing to caveat."""
    insight = fallback_insight(build_facts(**sample_run))
    assert "6 of 6 budgets passed" in insight.verdict_summary
    assert "measured nothing" not in insight.verdict_summary


def _critical_semantic() -> BlockingRegression:
    return BlockingRegression(
        prompt_id="replay",
        evaluator_name="semantic.cosine",
        slice_name="all",
        severity="critical",
        delta_avg_score=-0.05,
        effect_size=-2.15,
    )


def test_a_severity_fail_never_claims_a_budget_breach(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = decision(verdict="fail", blocking=[_critical_semantic()])
    recommendation = fallback_insight(build_facts(**sample_run)).recommendation
    assert recommendation == (
        "Do not migrate under the configured policy — semantic.cosine regressed at "
        "blocking severity although no budget was breached. Review its regressed "
        "examples, then re-run."
    )


def test_a_budget_fail_names_the_budget(sample_run: dict[str, Any]) -> None:
    breached = BudgetResult(
        name="max_overall_regression_rate",
        observed=0.5,
        allowed=0.03,
        passed=False,
        conclusive=True,
        ci_low=0.3,
        ci_high=0.7,
    )
    sample_run["decision"] = decision(
        verdict="fail",
        blocking=[_critical_semantic()],
        budget_results=[breached, *budgets()[1:]],
    )
    facts = build_facts(**sample_run)
    assert facts.failed_budgets == ["Overall regression rate"]
    assert fallback_insight(facts).recommendation.startswith(
        "Do not migrate under the configured policy — the overall regression rate "
        "budget was breached.",
    )


def test_a_fail_with_no_named_cause_points_at_what_the_reader_can_see(
    sample_run: dict[str, Any],
) -> None:
    """``decision.reason`` can be ``None`` on a fail, so the advice never points there."""
    sample_run["decision"] = decision(verdict="fail")
    assert fallback_insight(build_facts(**sample_run)).recommendation == (
        "Do not migrate under the configured policy — see the verdict's recommendations "
        "and the budget table, address the cause, then re-run."
    )


# ---------------------------------------------------------------------------
# Slice budgets gate the run too
# ---------------------------------------------------------------------------


def _slice_comparison(
    *,
    evaluator_name: str,
    slice_name: str,
    severity: str = "none",
) -> ComparisonResult:
    blocking = severity in {"critical", "high"}
    return ComparisonResult(
        prompt_id="replay",
        evaluator_name=evaluator_name,
        slice_name=slice_name,
        n=20,
        test="paired_t",
        statistic=1.0,
        p_value=0.01 if blocking else 0.6,
        p_value_corrected=0.01 if blocking else 0.6,
        effect_size=-0.9 if blocking else 0.0,
        effect_size_ci_low=-1.2 if blocking else -0.3,
        effect_size_ci_high=-0.6 if blocking else 0.3,
        delta_avg_score=-0.2 if blocking else 0.0,
        severity=severity,  # type: ignore[arg-type]
        notes=[],
    )


def _scored(evaluator_name: str, *, n: int, delta: float) -> list[EvalRecord]:
    return [
        EvalRecord(
            run_id="r_20260803_abcdef",
            prompt_id="replay",
            example_id=f"{evaluator_name}_{index}",
            evaluator_name=evaluator_name,
            source_score=1.0,
            target_score=1.0 + delta,
            delta=delta,
        )
        for index in range(n)
    ]


def _slice_budget_run(
    sample_run: dict[str, Any],
    *,
    extra: list[ComparisonResult],
    policy: MigrationPolicy | None = None,
    calls: list[Call] | None = None,
) -> dict[str, Any]:
    """A run whose overall budgets hold and whose 'security' slice budget breaks.

    Decided by the real policy rather than a hand-built decision, so the
    facts are read off exactly the shape ``evaluate_migration_policy``
    produces: the slice row lives under ``decision.slices``, never in the
    overall ``budget_results``.
    """
    comparisons = [
        _slice_comparison(evaluator_name="structural.length", slice_name="all"),
        _slice_comparison(evaluator_name="security.judge", slice_name="security"),
        *extra,
    ]
    records = [
        *_scored("structural.length", n=20, delta=0.0),
        *_scored("security.judge", n=20, delta=-0.5),
    ]
    sample_run["decision"] = evaluate_migration_policy(
        run_id="r_20260803_abcdef",
        source_model="src",
        target_model="tgt",
        policy=policy
        or MigrationPolicy(
            max_overall_regression_rate=0.5,
            min_equivalence_rate=0.0,
            slices={"security": SliceMigrationPolicy(max_overall_regression_rate=0.0)},
        ),
        comparisons=comparisons,
        records=records,
        calls=calls or [],
    )
    sample_run["comparisons"] = comparisons
    sample_run["records"] = records
    return sample_run


def test_a_slice_budget_fail_names_the_slice_budget(sample_run: dict[str, Any]) -> None:
    run = _slice_budget_run(sample_run, extra=[])
    assert run["decision"].verdict == "fail"
    assert run["decision"].reason is None

    facts = build_facts(**run)

    assert facts.failed_budgets == ["'security' slice overall regression rate"]
    assert fallback_insight(facts).recommendation == (
        "Do not migrate under the configured policy — the 'security' slice overall "
        "regression rate budget was breached. Address the blocking evaluators, then re-run."
    )


def test_a_slice_budget_fail_with_a_slice_only_blocking_row_blames_the_budget(
    sample_run: dict[str, Any],
) -> None:
    """Slice severity only demotes to conditional_pass; the budget is what failed."""
    run = _slice_budget_run(
        sample_run,
        extra=[
            _slice_comparison(
                evaluator_name="structural.length",
                slice_name="security",
                severity="high",
            ),
        ],
    )
    assert run["decision"].verdict == "fail"

    recommendation = fallback_insight(build_facts(**run)).recommendation

    assert "'security' slice overall regression rate budget was breached" in recommendation
    assert "no budget was breached" not in recommendation


def test_a_slice_row_that_restates_an_overall_breach_is_named_once(
    sample_run: dict[str, Any],
) -> None:
    """An inheriting slice copies the run-level cost row; one breach, one name."""
    run = _slice_budget_run(
        sample_run,
        extra=[],
        policy=MigrationPolicy(
            max_overall_regression_rate=1.0,
            min_equivalence_rate=0.0,
            max_cost_increase=0.0,
            slices={"security": SliceMigrationPolicy()},
        ),
        calls=[
            Call(
                run_id="r_20260803_abcdef",
                prompt_id="replay",
                example_id="e1",
                model_id=model_role,
                role=model_role,  # type: ignore[arg-type]
                text="ok",
                cost_usd=cost,
                latency_ms=100,
            )
            for model_role, cost in (("source", 1.0), ("target", 2.0))
        ],
    )
    assert run["decision"].verdict == "fail"

    assert build_facts(**run).failed_budgets == ["Cost increase"]


def test_a_severity_fail_names_only_the_overall_blocking_evaluators(
    sample_run: dict[str, Any],
) -> None:
    """A slice-only blocking row did not fail the run, so it is not named as why."""
    slice_only = BlockingRegression(
        prompt_id="replay",
        evaluator_name="structural.length",
        slice_name="security",
        severity="high",
        delta_avg_score=-0.2,
        effect_size=-0.9,
    )
    sample_run["decision"] = decision(verdict="fail", blocking=[_critical_semantic(), slice_only])

    recommendation = fallback_insight(build_facts(**sample_run)).recommendation

    assert recommendation == (
        "Do not migrate under the configured policy — semantic.cosine regressed at "
        "blocking severity although no budget was breached. Review its regressed "
        "examples, then re-run."
    )


def test_the_critical_count_reads_the_blocking_regressions(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = decision(verdict="fail", blocking=[_critical_semantic()])
    facts = build_facts(**sample_run)
    assert facts.rendered["critical_regressions"] == "1"
    assert "1 blocking regression (1 critical)" in fallback_insight(facts).verdict_summary


# ---------------------------------------------------------------------------
# Budget status and the decision's reason — what a FAIL's cause is read from
# ---------------------------------------------------------------------------


def _status_of(facts_statuses: list[tuple[str, str, str]], label: str) -> tuple[str, str]:
    (hit,) = [(limit, status) for name, limit, status in facts_statuses if name == label]
    return hit


def test_every_gating_budget_carries_a_status(sample_run: dict[str, Any]) -> None:
    """The converter run: one breached, one blind, the rest held."""
    sample_run["decision"] = converter_decision()
    facts = build_facts(**sample_run)

    assert len(facts.budget_statuses) == 8
    assert _status_of(facts.budget_statuses, "Latency increase") == ("+30%", BUDGET_BREACHED)
    assert _status_of(facts.budget_statuses, "Trace-rule violations") == ("1", BUDGET_WITHIN)
    assert _status_of(facts.budget_statuses, "Cost increase") == ("+200%", BUDGET_WITHIN)
    assert _status_of(facts.budget_statuses, "Tool-selection divergence") == (
        "20%",
        BUDGET_UNMEASURED,
    )
    assert facts.failed_budgets == ["Latency increase"]


def test_an_unconfirmed_breach_has_its_own_status(sample_run: dict[str, Any]) -> None:
    over = BudgetResult(
        name="max_overall_regression_rate",
        observed=0.1,
        allowed=0.03,
        passed=False,
        conclusive=False,
        ci_low=0.0,
        ci_high=0.3,
    )
    sample_run["decision"] = decision(verdict="inconclusive", budget_results=[over, *budgets()[1:]])
    facts = build_facts(**sample_run)
    assert _status_of(facts.budget_statuses, "Overall regression rate") == (
        "3%",
        BUDGET_UNCONFIRMED,
    )
    assert facts.failed_budgets == []


def test_a_slice_budget_is_listed_under_its_scope(sample_run: dict[str, Any]) -> None:
    """A slice row is a gating budget too, and its limit is a copyable figure."""
    # A slice limit no overall figure shares, so only the status lines can
    # have admitted it to the allow-list.
    policy = MigrationPolicy(
        max_overall_regression_rate=0.5,
        min_equivalence_rate=0.0,
        slices={"security": SliceMigrationPolicy(max_overall_regression_rate=0.07)},
    )
    facts = build_facts(**_slice_budget_run(sample_run, extra=[], policy=policy))
    assert _status_of(facts.budget_statuses, "'security' slice overall regression rate") == (
        "7%",
        BUDGET_BREACHED,
    )
    assert "7%" not in facts.budget_limits.values()
    assert {"7%", "7"} <= facts.allowed_numbers


def test_the_decision_reason_travels_with_the_facts(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = replace(decision(verdict="fail"), reason=SEVERITY_REASON)
    facts = build_facts(**sample_run)
    assert facts.verdict_reason == SEVERITY_REASON
    # Its figures are facts the model may copy.
    assert {"-2.15", "2.15", "0.0004"} <= facts.allowed_numbers


def test_a_decision_without_a_reason_renders_none(sample_run: dict[str, Any]) -> None:
    assert build_facts(**sample_run).verdict_reason == ""


def test_a_trace_rule_reason_reaches_the_facts_without_its_config_key(
    sample_run: dict[str, Any],
) -> None:
    """The policy quotes ``migration_policy.max_invariant_violations``; the facts must not.

    The reason lands in a copy-verbatim block, and that key is a banned
    identifier — a faithful copy would be rejected on exactly these runs.
    """
    breach = BudgetResult(
        name="max_invariant_violations", observed=2.0, allowed=1.0, passed=False, denominator=10
    )
    sample_run["decision"] = replace(decision(verdict="fail"), reason=_invariant_reason([breach]))
    facts = build_facts(**sample_run)

    assert "the trace-rule violations budget" in facts.verdict_reason
    assert "migration_policy." not in facts.verdict_reason
    assert not [t for t in banned_identifiers(facts) if t in facts.verdict_reason]
    assert {"2", "10", "1"} <= facts.allowed_numbers


def test_a_slice_trace_rule_reason_names_the_slice_budget_not_the_overall_one(
    sample_run: dict[str, Any],
) -> None:
    """The policy quotes the overall key beside the slice's limit; the facts must scope it.

    Overall limit 5 held; the 'security' slice's 0 broke. Rewritten as "the
    trace-rule violations budget", the reason pins the slice's 0 on the overall
    row — the one the FACTS block marks ``within`` — and a summary that copies
    the reason, as instructed, blames a budget that held.
    """
    sample_run["decision"] = slice_invariant_decision()
    facts = build_facts(**sample_run)

    assert (
        "in the 'security' slice, over the 0 allowed by the 'security' slice "
        "trace-rule violations budget." in facts.verdict_reason
    )
    assert "allowed by the trace-rule violations budget" not in facts.verdict_reason
    assert "migration_policy." not in facts.verdict_reason
    # Exactly the label the FACTS block lists the slice row under.
    assert "'security' slice trace-rule violations" in facts.failed_budgets
    assert ("Trace-rule violations", "5", BUDGET_WITHIN) in facts.budget_statuses


def test_a_slice_whose_name_has_an_apostrophe_keeps_its_scoped_budget(
    sample_run: dict[str, Any],
) -> None:
    """The scope is read up to "' slice, over", not up to the first quote.

    Stopping at the first quote, scope "o'brien" fell back to the overall
    wording and pinned the slice's limit on the overall row.
    """
    row = BudgetResult(
        name="max_invariant_violations",
        observed=1.0,
        allowed=0.0,
        passed=False,
        scope="o'brien",
        denominator=4,
    )
    sample_run["decision"] = replace(decision(verdict="fail"), reason=_invariant_reason([row]))
    facts = build_facts(**sample_run)

    assert f"{_scoped_budget_label(row)} budget" in facts.verdict_reason
    assert "'o'brien' slice trace-rule violations budget" in facts.verdict_reason
    assert "allowed by the trace-rule violations budget" not in facts.verdict_reason
    assert "migration_policy." not in facts.verdict_reason


def test_a_dropped_params_reason_reaches_the_facts_in_plain_words(
    sample_run: dict[str, Any],
) -> None:
    """Config vocabulary is not banned, so it would otherwise ship to readers."""
    reason = _dropped_params_reason({"anthropic/claude-sonnet-5-5": ["temperature"]})
    sample_run["decision"] = replace(decision(verdict="fail"), reason=reason)
    facts = build_facts(**sample_run)

    assert "migration_policy." not in facts.verdict_reason
    assert "drop_params" not in facts.verdict_reason
    assert "anthropic/claude-sonnet-5-5 dropped temperature" in facts.verdict_reason


def test_a_text_reply_is_described_by_its_text() -> None:
    assert describe_reply(_call(text="You have 3 tasks.")) == "You have 3 tasks."


def test_a_tool_call_reply_is_described_by_its_calls() -> None:
    """A replay of round 1 ends at the tool call, so its text is empty.

    Shown as a blank, the narrative model reported "the target returned empty
    outputs" for replies that each called a tool.
    """
    reply = _call(trace=_trace(("get_tasks", {"status": "pending"})))
    assert describe_reply(reply) == '[tool calls] get_tasks({"status": "pending"})'


def test_a_reply_with_calls_and_text_carries_both() -> None:
    reply = _call(
        text="Checking both.",
        trace=_trace(("search_notes", {"query": "PO"}), ("get_tasks", {})),
    )
    assert describe_reply(reply) == (
        '[tool calls] search_notes({"query": "PO"}); get_tasks({}) [text] Checking both.'
    )


def test_only_a_reply_with_neither_is_called_empty() -> None:
    assert describe_reply(_call()) == "[empty reply: no text and no tool calls]"
    assert describe_reply(_call(trace=_trace())) == "[empty reply: no text and no tool calls]"


def test_a_failed_call_is_not_called_empty() -> None:
    assert describe_reply(_call(error="timeout")) == "[the call failed]"


def _call(**overrides: Any) -> Call:
    fields: dict[str, Any] = {
        "run_id": "r_20260803_abcdef",
        "prompt_id": "replay",
        "example_id": "e1",
        "model_id": "anthropic/claude-sonnet-5",
        "role": "target",
    }
    return Call(**{**fields, **overrides})


def _trace(*calls: tuple[str, dict[str, Any]]) -> ToolTrace:
    return ToolTrace(
        calls=[
            ToolCall(tool_name=name, arguments=arguments, sequence_index=index)
            for index, (name, arguments) in enumerate(calls)
        ]
    )
