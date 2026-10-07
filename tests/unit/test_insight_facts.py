"""Tests for the deterministic half of the insights package.

``build_facts`` and ``fallback_insight`` are pure — no network, no model —
which is what makes the generator testable without mocking a provider.
"""

from __future__ import annotations

from typing import Any

from evalshift_cli.analysis.policy import BlockingRegression, BudgetResult, FailureCategoryCount
from evalshift_cli.insights.facts import build_facts
from evalshift_cli.insights.templates import fallback_insight
from evalshift_cli.reports.economics import PromptEconomics
from tests.unit.insights_factories import (
    COST_TARGET,
    budgets,
    comparisons,
    decision,
    records,
    role,
    unmeasured_budget,
    unmeasured_comparison,
)


def test_figures_are_pre_rendered_as_display_strings(sample_run: dict[str, Any]) -> None:
    """The model copies these verbatim; it never sees a raw float to round."""
    facts = build_facts(**sample_run)
    assert facts.rendered["cost_source_usd"] == "$0.0101"
    assert facts.rendered["cost_target_usd"] == "$0.0204"
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
