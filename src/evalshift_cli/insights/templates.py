"""Deterministic prose, built from the same figures the model is given.

This is what ships when generation is skipped (``--no-insights``, no API key)
or fails validation twice, so it has to read acceptably on its own. It
returns **no findings**: a behavioral finding is exactly what a template
cannot produce, and inventing one would be worse than omitting it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from evalshift_cli.evaluators.failures import category_label
from evalshift_cli.insights.facts import NOT_AVAILABLE, Facts
from evalshift_cli.insights.models import (
    FALLBACK_MODEL,
    MAX_SUMMARY_CHARS,
    Insight,
    clamp_text,
)

_RECOMMENDATIONS: dict[str, str] = {
    "pass": (
        "No policy budget was breached — safe to migrate under the configured policy. "
        "Skim the sampled regressions before promoting."
    ),
    "conditional_pass": (
        "Regressions were detected but no budget was breached. Review the sampled "
        "regressions and the blocking evaluators before promoting."
    ),
    "inconclusive": (
        "The evidence is not conclusive at this sample size. Capture more examples "
        "and re-run before deciding."
    ),
}


def fallback_insight(facts: Facts) -> Insight:
    """Build a templated narrative from ``facts``, with no generated prose."""
    rendered = facts.rendered
    return Insight(
        model=FALLBACK_MODEL,
        generated_at=datetime.now(UTC),
        verdict_summary=_clamp(_verdict_summary(facts)),
        advisory_summary=_clamp(_advisory_summary(facts)),
        economics_summary=_clamp(_economics_summary(rendered)),
        recommendation=_clamp(_recommendation(facts)),
        findings=[],
    )


def _recommendation(facts: Facts) -> str:
    """The verdict's advice, naming what actually failed the run.

    Keyed on the verdict alone, a ``fail`` once said "at least one regression
    budget was breached" over a run whose every budget held: the fail came
    from a blocking evaluator's severity. The facts carry the cause, so the
    sentence names it: a confirmed budget breach first (overall or slice),
    then an overall-slice blocking regression.
    """
    if facts.verdict != "fail":
        return _RECOMMENDATIONS.get(facts.verdict, _RECOMMENDATIONS["inconclusive"])
    if facts.failed_budgets:
        labels = ", ".join(label[:1].lower() + label[1:] for label in facts.failed_budgets)
        noun = "budget was" if len(facts.failed_budgets) == 1 else "budgets were"
        return (
            f"Do not migrate under the configured policy — the {labels} {noun} breached. "
            "Address the blocking evaluators, then re-run."
        )
    if facts.overall_blocking_evaluators:
        names = facts.overall_blocking_evaluators
        whose = "its" if len(names) == 1 else "their"
        return (
            "Do not migrate under the configured policy — "
            f"{', '.join(names)} regressed at blocking severity "
            f"although no budget was breached. Review {whose} regressed examples, then re-run."
        )
    # Not "see the verdict's reason": a fail can carry none (a slice-driven
    # demotion overridden by dropped params, say), and advice pointing at an
    # empty field is a dead end.
    return (
        "Do not migrate under the configured policy — see the verdict's recommendations "
        "and the budget table, address the cause, then re-run."
    )


def _verdict_summary(facts: Facts) -> str:
    rendered = facts.rendered
    blocking = rendered["blocking_regressions"]
    noun = "regression" if blocking == "1" else "regressions"
    parts = [
        f"{rendered['verdict']} under the configured policy. "
        f"{rendered['budgets_passed']} of {rendered['budgets_total']} budgets passed, "
        f"with {blocking} blocking {noun} ({rendered['critical_regressions']} critical) "
        f"across {rendered['n_examples']} examples.",
    ]
    # `budgets_passed` already excludes the gates that measured nothing, so
    # without this the count reads as a budget that *failed* rather than one
    # that was never evaluated — and the templated voice shipped the same
    # clean-sweep claim the generated one did.
    if facts.coverage_basis:
        if facts.unmeasured_budgets:
            noun = "budget" if len(facts.unmeasured_budgets) == 1 else "budgets"
            parts.append(
                f"{rendered['budgets_unmeasured']} {noun} measured nothing: "
                f"{', '.join(facts.unmeasured_budgets)}.",
            )
        if facts.unmeasured_evaluators:
            # Worded as the decision's own recommendations word it, so the two
            # surfaces a reader sees side by side cannot contradict each other.
            parts.append(
                "These blocking evaluators scored no comparable pair and did not "
                f"gate this run: {', '.join(facts.unmeasured_evaluators)}.",
            )
        parts.append("Treat their silence as unknown, not as equivalence.")
    return " ".join(parts)


def _advisory_summary(facts: Facts) -> str:
    rendered = facts.rendered
    # The three rates share one denominator. With none behind them they are
    # 0% by default, and "Regression rate 0%, equivalence 0%" over a run that
    # compared nothing is the same false comfort in the templated voice.
    parts = [
        f"Outcome rates are {rendered['regression_rate_pct']}: {facts.rates_basis}"
        if facts.rates_basis
        else (
            f"Regression rate {rendered['regression_rate_pct']}, "
            f"equivalence {rendered['equivalence_rate_pct']}, "
            f"improved {rendered['improved_rate_pct']}."
        ),
    ]
    if rendered["effect_size"] != NOT_AVAILABLE:
        parts.append(
            f"Largest effect size {rendered['effect_size']} at p {rendered['p_value']} "
            f"on {facts.worst_evaluator}.",
        )
    if rendered["worst_delta"] != NOT_AVAILABLE:
        parts.append(
            f"Per-example score deltas span {rendered['worst_delta']} to "
            f"{rendered['best_delta']}, median {rendered['median_delta']}.",
        )
    if facts.blocking_evaluators:
        parts.append(f"Blocking evaluators: {', '.join(facts.blocking_evaluators)}.")
    if facts.failure_categories:
        category, count = facts.failure_categories[0]
        parts.append(f"Most common failure: {category_label(category)} ({count} examples).")
    return " ".join(parts)


def _economics_summary(rendered: dict[str, str]) -> str:
    return (
        f"{rendered['cost_source_usd']} → {rendered['cost_target_usd']} across "
        f"{rendered['n_calls']} calls. Cost {rendered['cost_delta_pct']}, "
        f"latency {rendered['latency_delta_pct']}. Ceilings: cost "
        f"{rendered['cost_ceiling_pct']}, latency {rendered['latency_ceiling_pct']}."
    )


def _clamp(text: str) -> str:
    return clamp_text(text, MAX_SUMMARY_CHARS)


__all__ = ["fallback_insight"]
