"""Migration-policy evaluation.

This module turns statistical comparisons plus per-example records into the
product-level answer EvalShift exists to provide: whether the target model is
safe to migrate to under the configured regression budget.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from evalshift_cli.analysis.statistics import (
    ADVISORY_NOTE_PREFIX,
    UNMEASURED_NOTE_PREFIX,
    ComparisonResult,
)
from evalshift_cli.config.models import MigrationPolicy, SliceMigrationPolicy
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import (
    BROKEN_HARNESS_CAUSES,
    SEMANTIC_REGRESSION,
    TOOL_GROUND_TRUTH_MISS,
)
from evalshift_cli.runner.models import Call

MigrationVerdict = Literal["pass", "conditional_pass", "fail", "inconclusive"]

_REGRESSION_SEVERITIES = frozenset({"critical", "high", "medium", "low"})
_BLOCKING_SEVERITIES = frozenset({"critical", "high"})
_TOOL_ARGUMENT_KIND = "tool_arguments"
_SEMANTIC_KIND = "semantic"
# The two axes ``tool_selection`` scores. They are separate measurements
# against separate baselines — conformance grades each side absolutely
# against the recorded ground truth, divergence grades the target against a
# source fixed at 1.0 — so they select separately here too. Registering them
# is what binds them to a budget at all: the family slug ``tool_selection``
# appeared in no selector, which is why nothing in this module ever looked at
# a tool-selection row.
_TOOL_CONFORMANCE_KIND = "tool_selection.conformance"
_TOOL_DIVERGENCE_KIND = "tool_selection.divergence"
# Hand-written trace rules. Selected by slug like every other budget; no
# legacy name prefix exists, because no row predating the slug can be one.
_INVARIANTS_KIND = "trace_invariants"
# Legacy selection, kept only for records checkpointed before ``EvalRecord.kind``
# existed. Never add a new metric on top of these: an evaluator's ``name`` is
# whatever the user typed in evalshift.yaml, so a name-prefix filter silently
# stops matching the moment they rename it — which is how the tool-argument
# budget shipped unbound for every project scaffolded by ``evalshift init``.
#
# The two tool-selection axes deliberately have no entry: they never existed
# under a name prefix, so a row that predates them cannot belong to either.
_LEGACY_KIND_PREFIXES = {
    _TOOL_ARGUMENT_KIND: "tool_arguments.",
    _SEMANTIC_KIND: "semantic.",
}

# Rate *ceilings*, and the noun naming the rows each one is counted over. A
# rate over whole rows can only land on multiples of ``1/denominator``, so a
# ceiling finer than that step cannot be represented: the first non-zero value
# on the grid already breaches it, and a budget written as "1%" enforces "any
# at all". See :func:`_granularity_warnings`.
#
# Ceilings only. ``min_equivalence_rate`` shares the regression rate's
# denominator but is a *floor*, and a sub-granular floor collapses to
# maximally lax — only a 0% rate could fail it — which is the opposite of zero
# tolerance, so the warning's wording would be false. ``max_critical_regressions``
# is a count on a granularity of 1, and the cost/latency budgets are ratios of
# two averages with no row denominator at all.
_RATE_CEILING_DENOMINATORS: dict[str, str] = {
    "max_overall_regression_rate": "scored comparisons",
    "max_tool_argument_drift": "tool-argument comparisons",
    "max_tool_divergence": "tool-selection comparisons",
}

# Display names for the policy budgets, keyed on the ``evalshift.yaml`` field.
# Public: the HTML report's budget table shows the same names, so the prose a
# reader gets here and the row they look up never use two vocabularies.
BUDGET_LABELS: dict[str, str] = {
    "max_overall_regression_rate": "Overall regression rate",
    "max_critical_regressions": "Critical regressions",
    "min_equivalence_rate": "Equivalent-or-better rate",
    "max_tool_argument_drift": "Tool-argument drift",
    "max_tool_divergence": "Tool-selection divergence",
    "max_invariant_violations": "Trace-rule violations",
    "max_cost_increase": "Cost increase",
    "max_latency_increase": "Latency increase",
}

# What each budget actually measures, in the reader's terms. Appended in
# parentheses the first time prose names a budget: "tool-selection divergence"
# alone still assumes the reader knows the methodology.
BUDGET_MEANINGS: dict[str, str] = {
    "max_overall_regression_rate": "the share of scored comparisons where the target did worse",
    "max_critical_regressions": "the number of critical regressions",
    "min_equivalence_rate": (
        "the share of comparisons where the target matched or beat the source"
    ),
    "max_tool_argument_drift": ("how often the target filled tool arguments differently"),
    "max_tool_divergence": "how often the target called different tools than the source",
    "max_invariant_violations": (
        "the number of examples where the target broke a hand-written trace rule"
    ),
    "max_cost_increase": "how much more the target model costs to run",
    "max_latency_increase": "how much slower the target model answers",
}

# The budgets whose values are whole counts, not rates — rendered bare where
# every other budget renders as a percentage.
_COUNT_BUDGETS = frozenset({"max_critical_regressions", "max_invariant_violations"})

# Two-sided 95% normal quantile — the confidence level of every Wilson interval
# below. The exact quantile rather than the textbook ``1.96`` (nominal coverage
# 95.0004%) so this engine and the hosted gate, which uses the same constant,
# emit identical bounds; a rounded z put them up to 9e-6 apart, which reads as
# an unexplained mismatch between a local report and the hosted one. Not
# configurable, on either side: a per-project confidence level would let a
# policy edit change what ``conclusive`` means.
_WILSON_Z = 1.959963984540054

# The call-derived budgets, and the ``Call`` field each one averages over both
# roles. Unlike the rate ceilings above these have no row denominator — their
# denominator is the *source average*, which can itself be zero. See
# :func:`_has_measured_ratio` and :func:`_zero_valued_call_warnings`.
_CALL_RATIO_FIELDS: dict[str, str] = {
    "max_cost_increase": "cost_usd",
    "max_latency_increase": "latency_ms",
}


@dataclass(frozen=True, slots=True)
class PolicyMetricSummary:
    """Aggregate migration outcome rates for a scope."""

    n_records: int
    improved_rate: float
    equivalent_rate: float
    regression_rate: float
    critical_regressions: int
    tool_argument_drift_rate: float
    cost_increase_rate: float
    latency_increase_rate: float


@dataclass(frozen=True, slots=True)
class BudgetResult:
    """One migration-policy budget check.

    The three *proportion* budgets — ``max_overall_regression_rate``,
    ``min_equivalence_rate`` and ``max_tool_argument_drift``, each a count of
    records over a count of records — carry a 95% Wilson interval
    (``ci_low``/``ci_high``) on the observed rate. The same three the hosted
    gate computes one for, so the two engines resolve a breach alike.
    ``conclusive`` is False when the observed value breaches the budget but
    the interval still includes it — i.e. the sample is too
    small to confirm a real breach, and the verdict becomes ``inconclusive``
    rather than ``fail``. A budget the observation *held* is conclusive by
    construction however wide its interval: a thin sample must never turn a
    clean run into a caveat. It is also False for any record-derived budget on
    a scope that scored zero records: the 0/0 default renders as a clean row
    but measures nothing. ``max_tool_argument_drift`` is counted over
    ``tool_arguments`` rows only, so it needs one of *those* — a scope that
    scored records but ran no tool-argument evaluator is unmeasured too.
    Exact budgets (the critical-regression count, the cost/latency ratios of
    two averages) describe no proportion and have
    ``ci_low``/``ci_high`` of ``None``. The call-derived cost/latency budgets
    follow the same rule from the other side: the ratio reads 0.0 — "no
    increase" — both when a role has no error-free call to average and when
    both roles average zero, so they are conclusive only when at least one
    role contributed a positive average for that field
    (:func:`_has_measured_ratio`).

    ``denominator`` is how many units ``observed`` was computed over — the
    sample behind the number, per ``BUNDLE_SPEC.md``. ``0`` and ``None`` are
    *different* statements there and the hosted gate keys on both: ``0`` means
    the budget was counted against an empty sample, while ``None`` means no
    sample size was reported at all, which only bundles written before the
    field existed may say. This CLI always knows its own denominators, so
    every budget it emits carries an integer; the ``None`` default exists
    solely so :meth:`MigrationDecision.from_dict` can read back a
    ``migration_decision.json`` written by an older version without inventing
    a zero for it. See :func:`_budget_denominators`.
    """

    name: str
    observed: float
    allowed: float
    passed: bool
    scope: str = "overall"
    ci_low: float | None = None
    ci_high: float | None = None
    conclusive: bool = True
    denominator: int | None = None

    @property
    def measured(self) -> bool:
        """Whether this budget was counted over a sample that exists.

        ``conclusive`` is False for two unrelated reasons and only one of them
        is blindness. A *breached* budget is inconclusive when its interval
        still includes the ceiling — it measured something, the sample just
        cannot resolve the breach. A *passing* budget is conclusive by
        construction however wide its interval (see the class docstring), so
        ``passed and not conclusive`` is reachable only by the ``0/0`` default:
        a clean row over an empty sample.

        Deliberately a property and not a field. ``BudgetResult`` is the
        bundle's budget object, whose schema the server owns and which is
        ``additionalProperties: false``; :meth:`MigrationDecision.to_dict`
        walks ``dataclasses.fields``, so this never reaches the wire.
        """
        return self.conclusive or not self.passed


@dataclass(frozen=True, slots=True)
class SliceDecision:
    """Verdict and metrics for one analysis slice."""

    name: str
    verdict: MigrationVerdict
    metrics: PolicyMetricSummary
    budget_results: list[BudgetResult]


@dataclass(frozen=True, slots=True)
class BlockingRegression:
    """A statistical regression that can block migration."""

    prompt_id: str
    evaluator_name: str
    slice_name: str
    severity: str
    delta_avg_score: float
    effect_size: float


@dataclass(frozen=True, slots=True)
class FailureCategoryCount:
    """Count of a deterministic failure category from evaluator metadata."""

    category: str
    count: int


@dataclass(frozen=True, slots=True)
class MigrationDecision:
    """Top-level migration decision persisted as JSON.

    ``overall`` and the record-derived budgets are computed from *blocking*
    evaluator records only; the cost/latency budgets come from the run's
    calls and hold even when nothing gated quality. ``advisory`` mirrors the
    same metric shape for advisory
    (``blocking: false``) evaluators — reported, never gating — and is
    ``None`` when no advisory evaluators ran. ``advisory_regressions``
    likewise holds statistical regressions from advisory evaluators.

    ``recommendations`` is the run's prose channel, and carries two kinds of
    line: what the verdict implies (see :func:`_recommendations`) and any
    warning about how the gate was really set — today, sub-granular rate
    ceilings (see :func:`_granularity_warnings`). It is one list rather than
    two because both surfaces that render it, the terminal and the HTML
    report, read this field; a second channel would have to be plumbed
    through ``migration_decision.json`` and the bundle manifest, whose schema
    the server owns.

    ``policy`` is the resolved ``migration_policy`` the verdict above was
    computed under — ``policy.model_dump(mode="json")``, a plain ``dict``
    rather than the pydantic model, because :meth:`to_dict` is
    ``asdict(self)`` and a dict is what that walk expects. Riding it inside
    every pushed bundle (``hosted/bundle.py``) is what lets the hosted gate
    check a pull request against exactly the budgets this run's own verdict
    used, instead of a separate, web-edited policy nobody configures.
    ``None`` on the ``inconclusive_decision`` path — no ``migration_policy``
    was configured, so there is nothing resolved to stamp — and on a
    ``migration_decision.json`` written before this field existed.
    """

    run_id: str
    source_model: str
    target_model: str
    verdict: MigrationVerdict
    overall: PolicyMetricSummary
    slices: dict[str, SliceDecision]
    budget_results: list[BudgetResult]
    blocking_regressions: list[BlockingRegression]
    failure_categories: list[FailureCategoryCount]
    recommendations: list[str]
    reason: str | None = None
    advisory: PolicyMetricSummary | None = None
    advisory_regressions: list[BlockingRegression] = field(default_factory=list)
    policy: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MigrationDecision:
        """Rebuild a decision from :meth:`to_dict` output.

        ``analyze`` persists its decision to ``migration_decision.json``, and
        that file — not a recomputation from whatever the config says later —
        is the decision every downstream surface must describe. Reading it back
        is what stops a policy edit between ``analyze`` and ``report`` putting
        one verdict in the report's verdict block and another in the prose
        beside it.

        Raises:
            ValueError: If a field is missing or the wrong shape. Callers treat
                that as "no persisted decision" and fall back, so a hand-edited
                artifact degrades the narrative instead of failing the run.
        """
        try:
            return cls(
                run_id=payload["run_id"],
                source_model=payload["source_model"],
                target_model=payload["target_model"],
                verdict=payload["verdict"],
                overall=PolicyMetricSummary(**payload["overall"]),
                slices={
                    name: SliceDecision(
                        name=raw["name"],
                        verdict=raw["verdict"],
                        metrics=PolicyMetricSummary(**raw["metrics"]),
                        budget_results=[BudgetResult(**b) for b in raw["budget_results"]],
                    )
                    for name, raw in (payload.get("slices") or {}).items()
                },
                budget_results=[BudgetResult(**b) for b in payload.get("budget_results") or []],
                blocking_regressions=[
                    BlockingRegression(**r) for r in payload.get("blocking_regressions") or []
                ],
                failure_categories=[
                    FailureCategoryCount(**c) for c in payload.get("failure_categories") or []
                ],
                recommendations=list(payload.get("recommendations") or []),
                reason=payload.get("reason"),
                advisory=(
                    PolicyMetricSummary(**payload["advisory"])
                    if payload.get("advisory") is not None
                    else None
                ),
                advisory_regressions=[
                    BlockingRegression(**r) for r in payload.get("advisory_regressions") or []
                ],
                policy=payload.get("policy"),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"not a migration decision: {exc}") from exc


def _dropped_params_reason(dropped_params: Mapping[str, Sequence[str]]) -> str:
    """One sentence naming every model and the parameters it could not honour."""
    per_model = "; ".join(
        f"{model} dropped {', '.join(params)}" for model, params in sorted(dropped_params.items())
    )
    return (
        "migration_policy.fail_on_dropped_params is set and this run could not "
        f"replay every recorded generation constraint: {per_model}. Those calls "
        "succeeded without the constraint (drop_params), so the scores describe a "
        "different experiment than the capture recorded."
    )


def evaluate_migration_policy(
    *,
    run_id: str,
    source_model: str,
    target_model: str,
    policy: MigrationPolicy,
    comparisons: list[ComparisonResult],
    records: list[EvalRecord],
    calls: list[Call],
    dropped_params: Mapping[str, Sequence[str]] | None = None,
) -> MigrationDecision:
    """Evaluate a run against the configured migration policy.

    Only *blocking* evaluator records/comparisons gate the verdict; advisory
    ones are summarised separately (``advisory`` / ``advisory_regressions``).

    Args:
        dropped_params: ``state.json``'s record of generation parameters
            LiteLLM dropped per model, consulted only when
            ``policy.fail_on_dropped_params`` is set. Optional and defaulted
            because run directories written before the field existed have no
            such record — and "we do not know" must not become "it failed".
    """
    blocking_records = [r for r in records if r.blocking]
    advisory_records = [r for r in records if not r.blocking]
    advisory_evaluators = _advisory_evaluator_names(comparisons=comparisons, records=records)
    gating = [c for c in comparisons if c.evaluator_name not in advisory_evaluators]
    advisory_comparisons = [c for c in comparisons if c.evaluator_name in advisory_evaluators]

    floor = policy.tool_argument_drift_floor
    overall = _metrics(records=blocking_records, calls=calls, drift_floor=floor)
    overall_divergence, overall_divergence_n = _tool_divergence_counts(blocking_records)
    overall_invariants, overall_invariants_n = _invariant_violation_counts(blocking_records)
    overall_denominators = _budget_denominators(
        metrics=overall,
        tool_argument_records=_tool_argument_record_count(blocking_records),
        tool_divergence_records=overall_divergence_n,
        invariant_records=overall_invariants_n,
        calls=calls,
    )
    overall_budgets = _budget_results(
        policy=policy,
        metrics=overall,
        scope="overall",
        denominators=overall_denominators,
        divergence_count=overall_divergence,
        invariant_violations=overall_invariants,
        cost_measured=_has_measured_ratio(calls, field="cost_usd"),
        latency_measured=_has_measured_ratio(calls, field="latency_ms"),
    )
    # Run-level, so computed once and reused by every scope below.
    zero_valued_call_notes = _zero_valued_call_warnings(calls)
    # Likewise run-level: every blocking record belongs to the overall scope,
    # so a per-slice note would only restate a subset of this one count.
    shared_miss_notes = _shared_ground_truth_warnings(blocking_records)
    # Over every scored row, not just the blocking ones: the disclosure is
    # about what the numbers in the report mean, and an advisory argument
    # gate's 1.0 misreads exactly as a blocking one's does.
    source_derived_notes = _source_derived_ground_truth_warnings(records)
    granularity_notes = _granularity_warnings(overall_budgets, overall_denominators)
    blocking = _blocking_regressions(gating)
    categories = _failure_categories(records)
    advisory = (
        _metrics(records=advisory_records, calls=calls, drift_floor=floor)
        if advisory_records
        else None
    )

    slice_names = sorted({c.slice_name for c in comparisons if c.slice_name != "all"})
    slice_decisions: dict[str, SliceDecision] = {}
    for name in slice_names:
        scoped_records = _records_for_slice(blocking_records, gating, name)
        slice_policy = _slice_policy(policy, policy.slices.get(name))
        scoped_metrics = _metrics(
            records=scoped_records,
            calls=calls,
            drift_floor=slice_policy.tool_argument_drift_floor,
        )
        scoped_divergence, scoped_divergence_n = _tool_divergence_counts(scoped_records)
        scoped_invariants, scoped_invariants_n = _invariant_violation_counts(scoped_records)
        scoped_denominators = _budget_denominators(
            metrics=scoped_metrics,
            tool_argument_records=_tool_argument_record_count(scoped_records),
            tool_divergence_records=scoped_divergence_n,
            invariant_records=scoped_invariants_n,
            calls=calls,
        )
        budgets = _budget_results(
            policy=slice_policy,
            metrics=scoped_metrics,
            scope=name,
            denominators=scoped_denominators,
            divergence_count=scoped_divergence,
            invariant_violations=scoped_invariants,
            cost_measured=_has_measured_ratio(calls, field="cost_usd"),
            latency_measured=_has_measured_ratio(calls, field="latency_ms"),
        )
        granularity_notes.extend(_granularity_warnings(budgets, scoped_denominators))
        slice_decisions[name] = SliceDecision(
            name=name,
            verdict=_verdict_for(
                comparisons=[c for c in gating if c.slice_name == name],
                budgets=budgets,
            ),
            metrics=scoped_metrics,
            budget_results=budgets,
        )

    overall_comparisons = [c for c in gating if c.slice_name == "all"]
    # A slice budget is a gate the user wrote, so it blocks on exactly the
    # terms an overall one does: conclusive breach fails, unconfirmed breach
    # is inconclusive. Demoting it to `conditional_pass` is what let this CLI
    # answer `conditional_pass` on a bundle the hosted gate failed.
    # `slices[*].verdict` is unaffected — it still describes its own scope.
    gating_budgets = [
        *overall_budgets,
        *_distinct_slice_budgets(overall_budgets, slice_decisions),
    ]
    # Gating evaluators that ended up with no applicable row at all. Their
    # silence is not evidence of equivalence, so it must never underwrite a
    # verdict — see `_recommendations` and `unmeasured_gating_evaluators`.
    unmeasured = unmeasured_gating_evaluators(comparisons=comparisons, records=records)
    if overall.n_records == 0:
        # Nothing gated quality (every evaluator advisory, or every blocking
        # record errored): there is no evidence either way on correctness, and
        # "pass" on an empty set would be a verdict with zero backing. The
        # cost/latency budgets are call-derived though — they measured
        # something real, so a conclusive breach there still fails the run
        # rather than being softened to "inconclusive".
        verdict: MigrationVerdict = (
            "fail" if any(not b.passed and b.conclusive for b in gating_budgets) else "inconclusive"
        )
        reason: str | None = (
            # The set can also be empty because everything in it measured the
            # harness rather than the migration, and saying "every evaluator
            # is advisory" about a run with a blocking evaluator that scored
            # every pair would be flatly false.
            "every blocking row was a shared ground-truth miss — both models "
            "failed the same recorded ground truth on every scored pair, which "
            "measures the eval harness and not the migration, so nothing is "
            "left to gate quality."
            if shared_miss_notes
            else "no blocking evaluator records — every configured evaluator is "
            "advisory (blocking: false) or all blocking rows errored, so "
            "quality is ungated; only the call-derived cost/latency budgets "
            "are in force."
        )
    else:
        verdict = _verdict_for(comparisons=overall_comparisons, budgets=gating_budgets)
        if verdict == "pass":
            failed_slices = [s for s in slice_decisions.values() if s.verdict == "fail"]
            if failed_slices:
                verdict = "conditional_pass"
            elif unmeasured:
                # The gate these evaluators were configured to enforce never
                # ran. Passing on their silence would be a verdict with no
                # evidence behind it.
                verdict = "conditional_pass"
        elif not overall_comparisons and any(
            s.verdict in {"fail", "conditional_pass"} for s in slice_decisions.values()
        ):
            verdict = "conditional_pass"
        reason = (
            _inconclusive_reason(gating_budgets, n=overall.n_records)
            if verdict == "inconclusive"
            else None
        )

    invariant_breaches = _invariant_breaches(gating_budgets)
    if invariant_breaches:
        # `_verdict_for` already fails on these; repeated here because the
        # slice rules above can still demote its answer to conditional_pass,
        # and a broken trace rule is not softened by any of them.
        verdict = "fail"
        reason = _invariant_reason(invariant_breaches)

    if policy.fail_on_dropped_params and dropped_params:
        # An explicit opt-in, so it overrides every other verdict including a
        # clean pass: the user has said the constraint *is* the contract, and
        # an arm that never received it did not run the experiment they wrote.
        # The existing reason is kept rather than replaced -- a budget breach
        # or an inconclusive sample is still true and still worth naming.
        dropped_reason = _dropped_params_reason(dropped_params)
        reason = f"{reason} {dropped_reason}" if reason else dropped_reason
        verdict = "fail"

    return MigrationDecision(
        run_id=run_id,
        source_model=source_model,
        target_model=target_model,
        verdict=verdict,
        overall=overall,
        slices=slice_decisions,
        budget_results=overall_budgets,
        blocking_regressions=blocking,
        failure_categories=categories,
        recommendations=[
            *_recommendations(
                verdict=verdict,
                slices=slice_decisions,
                # "Enable a blocking evaluator" is only the right advice when
                # the empty set really is a config gap. When the exclusion
                # below emptied it, the note that explains the exclusion is.
                no_blocking_records=overall.n_records == 0 and not shared_miss_notes,
                unmeasured=unmeasured,
                # …and when it did, "collect more examples" is the one piece of
                # advice that cannot work: the extra rows are excluded too.
                shared_ground_truth_only=overall.n_records == 0 and bool(shared_miss_notes),
            ),
            # Which slice budget blocked. The overall rows can be green in a
            # run this fails, so without this line the verdict names no
            # number the reader can go and look at.
            *_slice_budget_notes(gating_budgets),
            # A rule both models broke is still a failure, but it is also the
            # one signal that the rule itself may be what needs a look.
            *_shared_invariant_notes(blocking_records),
            # Appended, never substituted: a sub-granular budget does not
            # change what the run measured, so it must not displace the advice
            # the verdict earned — it explains how the gate was really set.
            *granularity_notes,
            # Same rule, the other direction: these say what the gate could
            # *not* measure, which likewise never replaces the verdict's advice.
            *zero_valued_call_notes,
            # And these say what the gate refused to count as evidence.
            *shared_miss_notes,
            # ...and this, what it counted but could not independently check.
            *source_derived_notes,
        ],
        reason=reason,
        advisory=advisory,
        advisory_regressions=_blocking_regressions(advisory_comparisons),
        policy=policy.model_dump(mode="json"),
    )


def inconclusive_decision(
    *,
    run_id: str,
    source_model: str,
    target_model: str,
    comparisons: list[ComparisonResult],
    records: list[EvalRecord],
    calls: list[Call],
    reason: str = "no migration_policy configured",
) -> MigrationDecision:
    """Build an ``inconclusive`` decision when no migration policy is configured.

    The verdict is fixed to ``inconclusive`` and no budgets are evaluated, but
    the overall outcome metrics and blocking regressions are still computed so
    the bundle always carries a real, renderable decision (never null).
    """
    return MigrationDecision(
        run_id=run_id,
        source_model=source_model,
        target_model=target_model,
        verdict="inconclusive",
        overall=_metrics(records=records, calls=calls),
        slices={},
        budget_results=[],
        blocking_regressions=_blocking_regressions(comparisons),
        failure_categories=_failure_categories(records),
        recommendations=[
            "Configure a migration_policy to get a pass/fail verdict.",
            # True of the suite, not of the gate: a run with no policy still
            # renders these scores, so it owes the same disclosure.
            *_source_derived_ground_truth_warnings(records),
        ],
        reason=reason,
    )


def _is_kind(r: EvalRecord, kind: str) -> bool:
    """Whether ``r`` came from an evaluator of type ``kind``.

    Falls back to the pre-``kind`` name prefix so records checkpointed by an
    older CLI, or replayed from an older bundle, still feed the same metrics.
    """
    if r.kind:
        return r.kind == kind
    prefix = _LEGACY_KIND_PREFIXES.get(kind)
    return prefix is not None and r.evaluator_name.startswith(prefix)


def _is_semantic_regression(r: EvalRecord) -> bool:
    """Whether a semantic-evaluator record breached ``min_similarity``.

    Reuses the ``SEMANTIC_REGRESSION`` flag the evaluator already wrote at
    scoring time, so the policy gate honours the same ``min_similarity``
    threshold as the report instead of re-deriving it from ``delta``.
    """
    cats = r.metadata.get("failure_categories", [])
    return isinstance(cats, list) and SEMANTIC_REGRESSION in cats


def _is_regression(r: EvalRecord) -> bool:
    """Whether a record counts as a regression for policy budgets."""
    if _is_kind(r, _SEMANTIC_KIND):
        return _is_semantic_regression(r)
    return r.delta < 0


def _is_equivalent(r: EvalRecord) -> bool:
    """Whether a record counts as source/target equivalent.

    Semantic drift within ``min_similarity`` is treated as equivalent (not a
    regression), keeping the improved/equivalent/regressed partition intact.
    """
    if _is_kind(r, _SEMANTIC_KIND):
        return not _is_semantic_regression(r)
    return r.delta == 0


def _scored(records: list[EvalRecord]) -> list[EvalRecord]:
    """Rows that actually measured something — i.e. every row that did not error.

    A pair an evaluator measured nothing on no longer produces a row at
    all, so there is nothing left here to filter: absence is the signal, and
    the run's ``EvaluatorCoverage`` is what still counts those pairs. An
    errored row is different — the measurement broke rather than not
    applying — and stays excluded.

    Not what the policy rates are counted over: see :func:`_evidence`, which
    drops the rows that measured the harness rather than the migration.
    """
    return [r for r in records if r.error is None]


def is_shared_ground_truth_miss(r: EvalRecord) -> bool:
    """Whether a conformance row's zero delta is a shared failure, not equivalence.

    Public because the report layer asks the same question of the same rows:
    an aggregate whose every measurement is one of these must not be
    headlined "Equivalent", and one definition of "both sides missed" is what
    keeps the verdict block and the evaluator table from disagreeing.

    The conformance axis grades each side *absolutely* against the ground
    truth the suite recorded, so both sides can miss it at the same height:
    ``0.0 / 0.0`` on an example both models answered with a tool call the
    recording never made. The delta is zero, and :func:`_is_equivalent` filed
    every one of those as equivalent — which is the whole of
    ``equivalent_rate: 1.0`` on a run where nine pairs in ten called entirely
    different tools.

    A shared miss is not evidence about the *migration*. Ground truth
    captured from the source model that the source model then fails is
    evidence about the harness — the wrong toolset attached, the wrong
    prompt, a suite promoted from a different agent. Such rows are excluded
    from every policy rate and reported separately
    (:func:`_shared_ground_truth_warnings`, and the ``TOOL_GROUND_TRUTH_MISS``
    count in ``failure_categories``).

    Reuses the flag the evaluator already wrote, on the same rule as
    :func:`_is_semantic_regression`: one definition of "both sides missed",
    in the evaluator that knows the expectation.

    The ``delta == 0`` guard is what keeps the exclusion from eating real
    findings. A conformance row where both sides missed *and* the target
    missed by more (``0.8 / 0.3``) is a genuine regression the migration
    caused, and a row the target improved on (``0.2 / 0.6``) is a genuine
    improvement; only the shared-height case carries no signal.
    """
    if r.delta != 0 or not _is_kind(r, _TOOL_CONFORMANCE_KIND):
        return False
    cats = r.metadata.get("failure_categories", [])
    return isinstance(cats, list) and TOOL_GROUND_TRUTH_MISS in cats


def _evidence(records: list[EvalRecord]) -> list[EvalRecord]:
    """Scored rows that say something about the migration.

    Every denominator in this module is drawn from here rather than from
    :func:`_scored`, so a row can never be counted by one budget and ignored
    by another.
    """
    return [r for r in _scored(records) if not is_shared_ground_truth_miss(r)]


def _tool_argument_record_count(records: list[EvalRecord]) -> int:
    """How many scored rows the tool-argument drift rate is counted over.

    This is the drift budget's *denominator*, which is not
    ``PolicyMetricSummary.n_records``: a scope can score plenty of rows and
    still have no ``tool_arguments`` evaluator at all. ``_budget_results``
    needs it to tell "measured, zero drift" from "never measured".
    """
    return sum(1 for r in _evidence(records) if _is_kind(r, _TOOL_ARGUMENT_KIND))


def _tool_divergence_counts(records: list[EvalRecord]) -> tuple[int, int]:
    """``(diverged, total)`` over this scope's ``tool_selection.divergence`` rows.

    The divergence budget's numerator *and* denominator, returned together so
    the rate and its Wilson interval can never be computed over different
    samples. Diverged is simply a negative delta: the axis fixes the source at
    1.0, so any target below it called something the source did not.

    Deliberately not a field on :class:`PolicyMetricSummary` — that dataclass
    is serialised verbatim into the bundle manifest, whose schema is owned by
    ``evalshift-server`` and forbids unknown properties. The rate is therefore
    computed here and handed to :func:`_budget_results`, exactly as the
    drift budget's denominator already is.
    """
    rows = [r for r in _evidence(records) if _is_kind(r, _TOOL_DIVERGENCE_KIND)]
    return sum(1 for r in rows if r.delta < 0), len(rows)


def _invariant_violation_counts(records: list[EvalRecord]) -> tuple[int, int]:
    """``(violated, total)`` over this scope's ``trace_invariants`` rows.

    Violated means the target broke at least one rule: any target score below
    1.0, which also catches a row averaged over repeated samples where one
    sample broke a rule. Judged against the rules alone -- a rule the source
    broke too is still broken, which is the reason the evaluator exists.
    """
    rows = [r for r in _evidence(records) if _is_kind(r, _INVARIANTS_KIND)]
    return sum(1 for r in rows if r.target_score < 1.0), len(rows)


def _metrics(
    *,
    records: list[EvalRecord],
    calls: list[Call],
    drift_floor: float = 0.9,
) -> PolicyMetricSummary:
    # Not ``_scored``: a conformance row both models failed at the same
    # height is a broken harness, not a measurement of the migration, and
    # counting it as equivalent is the defect this module exists to stop.
    scored = _evidence(records)
    n = len(scored)
    regressions = [r for r in scored if _is_regression(r)]
    improvements = [r for r in scored if r.delta > 0]
    equivalents = [r for r in scored if _is_equivalent(r)]
    critical = [r for r in scored if str(r.metadata.get("severity", "")) == "critical"]
    tool_argument_records = [r for r in scored if _is_kind(r, _TOOL_ARGUMENT_KIND)]
    # Continuous scores need a materiality threshold: ``delta < 0`` alone
    # counts a 0.98 the same as a 0.0, which makes the shipped 0.01 budget
    # unreachable. Count a call as drifted only when its arguments landed
    # materially below the source's.
    tool_argument_drift = [
        r for r in tool_argument_records if r.delta < 0 and r.target_score < drift_floor
    ]

    return PolicyMetricSummary(
        n_records=n,
        improved_rate=_rate(len(improvements), n),
        equivalent_rate=_rate(len(equivalents), n),
        regression_rate=_rate(len(regressions), n),
        critical_regressions=len(critical),
        tool_argument_drift_rate=_rate(len(tool_argument_drift), len(tool_argument_records)),
        cost_increase_rate=_relative_increase(calls, field="cost_usd"),
        latency_increase_rate=_relative_increase(calls, field="latency_ms"),
    )


def _wilson_interval(count: int, n: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it behaves sanely at the
    small n (8-20 examples) typical of freshly captured suites and at
    observed rates of exactly 0 or 1.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _budget_results(
    *,
    policy: MigrationPolicy,
    metrics: PolicyMetricSummary,
    scope: str,
    denominators: Mapping[str, int],
    divergence_count: int,
    invariant_violations: int,
    cost_measured: bool,
    latency_measured: bool,
) -> list[BudgetResult]:
    """Evaluate every budget for one scope.

    ``denominators`` is this scope's sample size per budget (see
    :func:`_budget_denominators`) — reported on each result *and* used to
    decide whether the drift budget measured anything, so the number a budget
    is judged on is the number it reports. ``divergence_count`` is the
    numerator of ``max_tool_divergence`` over
    ``denominators["max_tool_divergence"]`` (see
    :func:`_tool_divergence_counts`); it arrives as a count rather than a rate
    because the Wilson interval needs the count either way, and rebuilding one
    from the other rounds. ``invariant_violations`` is the numerator of
    ``max_invariant_violations`` over ``denominators["max_invariant_violations"]``
    (see :func:`_invariant_violation_counts`); that budget's row is emitted only
    when the denominator is positive. ``cost_measured`` /
    ``latency_measured`` are the call-derived budgets' conclusiveness flags and
    stay separate from their denominators: those ratios also default to 0.0
    with a perfectly good call count behind them, when both roles average zero
    (see :func:`_has_measured_ratio`).

    All of it is passed in rather than read off ``metrics`` because
    ``PolicyMetricSummary`` is serialised into the bundle manifest, whose
    schema is owned by the server.
    """
    n = metrics.n_records
    # Every record-derived budget reads 0/0 = 0.0 at n == 0, which renders as
    # a clean "observed 0.00, passed" row. Nothing was measured, so those rows
    # are not conclusive.
    measured = n > 0
    # Tool-argument drift is counted over ``tool_arguments`` rows only, so
    # scoring records is not enough: a scope with no such evaluator would
    # otherwise report "observed 0.00, passed, conclusive" for a budget that
    # never measured anything.
    drift_n = denominators["max_tool_argument_drift"]
    drift_measured = measured and drift_n > 0
    regressions = round(metrics.regression_rate * n)
    reg_low, reg_high = _wilson_interval(regressions, n)
    regression_rate_breached = metrics.regression_rate > policy.max_overall_regression_rate
    # A breach only *fails* when the CI confirms it (lower bound clears the
    # budget); otherwise the sample can't resolve the budget → inconclusive.
    # A within-budget observation always passes — a wide CI must not block a
    # clean run.
    non_regression = 1.0 - metrics.regression_rate
    equivalence_breached = non_regression < policy.min_equivalence_rate
    # Drifted tool-argument rows over tool-argument rows is a binomial
    # proportion on its own denominator, so it earns the same interval and the
    # same ceiling rule as the regression rate above — and the hosted gate has
    # always treated it as one.
    drifted = round(metrics.tool_argument_drift_rate * drift_n)
    drift_low, drift_high = _wilson_interval(drifted, drift_n)
    drift_breached = metrics.tool_argument_drift_rate > policy.max_tool_argument_drift
    # Diverged rows over divergence rows — the same shape as drift above, on
    # its own denominator and with the same "counted nothing is not clean"
    # rule. A scope may score plenty of rows and configure ``divergence: off``.
    divergence_n = denominators["max_tool_divergence"]
    divergence_measured = measured and divergence_n > 0
    divergence_rate = _rate(divergence_count, divergence_n)
    divergence_low, divergence_high = _wilson_interval(divergence_count, divergence_n)
    divergence_breached = divergence_rate > policy.max_tool_divergence
    results = [
        BudgetResult(
            name="max_overall_regression_rate",
            observed=metrics.regression_rate,
            allowed=policy.max_overall_regression_rate,
            passed=not regression_rate_breached,
            scope=scope,
            ci_low=reg_low,
            ci_high=reg_high,
            conclusive=measured
            and ((not regression_rate_breached) or reg_low > policy.max_overall_regression_rate),
            denominator=denominators["max_overall_regression_rate"],
        ),
        BudgetResult(
            name="max_critical_regressions",
            observed=float(metrics.critical_regressions),
            allowed=float(policy.max_critical_regressions),
            passed=metrics.critical_regressions <= policy.max_critical_regressions,
            scope=scope,
            conclusive=measured,
            denominator=denominators["max_critical_regressions"],
        ),
        BudgetResult(
            # "Equivalence" budget is satisfied by any non-regression: a record
            # that is equivalent *or improved* counts as passing. Improvements
            # must not fail a migration - the downside is already bounded by
            # ``max_overall_regression_rate``. Non-regression rate is the exact
            # complement of the regression rate, so its CI is the mirrored
            # regression-rate CI.
            name="min_equivalence_rate",
            observed=non_regression,
            allowed=policy.min_equivalence_rate,
            passed=not equivalence_breached,
            scope=scope,
            ci_low=1.0 - reg_high,
            ci_high=1.0 - reg_low,
            conclusive=measured
            and ((not equivalence_breached) or (1.0 - reg_low) < policy.min_equivalence_rate),
            denominator=denominators["min_equivalence_rate"],
        ),
        BudgetResult(
            name="max_tool_argument_drift",
            observed=metrics.tool_argument_drift_rate,
            allowed=policy.max_tool_argument_drift,
            passed=not drift_breached,
            scope=scope,
            ci_low=drift_low,
            ci_high=drift_high,
            conclusive=drift_measured
            and ((not drift_breached) or drift_low > policy.max_tool_argument_drift),
            denominator=drift_n,
        ),
        BudgetResult(
            name="max_tool_divergence",
            observed=divergence_rate,
            allowed=policy.max_tool_divergence,
            passed=not divergence_breached,
            scope=scope,
            ci_low=divergence_low,
            ci_high=divergence_high,
            conclusive=divergence_measured
            and ((not divergence_breached) or divergence_low > policy.max_tool_divergence),
            denominator=divergence_n,
        ),
        BudgetResult(
            name="max_cost_increase",
            observed=metrics.cost_increase_rate,
            allowed=policy.max_cost_increase,
            passed=metrics.cost_increase_rate <= policy.max_cost_increase,
            scope=scope,
            conclusive=cost_measured,
            denominator=denominators["max_cost_increase"],
        ),
        BudgetResult(
            name="max_latency_increase",
            observed=metrics.latency_increase_rate,
            allowed=policy.max_latency_increase,
            passed=metrics.latency_increase_rate <= policy.max_latency_increase,
            scope=scope,
            conclusive=latency_measured,
            denominator=denominators["max_latency_increase"],
        ),
    ]
    invariant_n = denominators["max_invariant_violations"]
    if invariant_n > 0:
        # Emitted only where a trace rule was checked: a "measured nothing"
        # row on every project that never wrote one would be noise in the
        # report and the insights. A count, so no interval: a breach is a
        # breach, and conclusive however small the sample.
        results.append(
            BudgetResult(
                name="max_invariant_violations",
                observed=float(invariant_violations),
                allowed=float(policy.max_invariant_violations),
                passed=invariant_violations <= policy.max_invariant_violations,
                scope=scope,
                conclusive=True,
                denominator=invariant_n,
            ),
        )
    return results


def _budget_denominators(
    *,
    metrics: PolicyMetricSummary,
    tool_argument_records: int,
    tool_divergence_records: int,
    invariant_records: int,
    calls: list[Call],
) -> dict[str, int]:
    """Every budget's own denominator for one scope, keyed by budget name.

    One map with two readers, which is the point: :func:`_budget_results`
    reports it on each ``BudgetResult`` (and decides the drift budget's
    ``conclusive`` from it), and :func:`_granularity_warnings` judges each rate
    ceiling against the grid its denominator implies. Sharing one map is what
    keeps the sample a budget is *reported* to have measured over identical to
    the one it was *judged* on.

    The companion of :data:`_RATE_CEILING_DENOMINATORS`, which names the rate
    ceilings' rows in prose; this covers every budget, because
    ``BUNDLE_SPEC.md`` asks every emitted budget for its sample size.
    """
    return {
        # One denominator, three budgets: the regression rate, its exact
        # complement the non-regression rate, and the critical count are all
        # counted over this scope's scored rows.
        #
        # Counted over *measurements*, not over examples. An evaluator that
        # scores two axes contributes two rows per example, and that is
        # correct here: conformance and divergence ask different questions,
        # a regression on either is a regression, and collapsing them to one
        # row per example would have to pick a winner — which is exactly the
        # mutually-exclusive framing that hid the defect. The rows that do
        # *not* belong are the ones that measured the harness rather than the
        # migration, and :func:`_evidence` has already dropped those.
        "max_overall_regression_rate": metrics.n_records,
        "min_equivalence_rate": metrics.n_records,
        "max_critical_regressions": metrics.n_records,
        # Per-axis budgets keep their own rows: neither moves when the other
        # axis is configured on or off.
        "max_tool_argument_drift": tool_argument_records,
        "max_tool_divergence": tool_divergence_records,
        "max_invariant_violations": invariant_records,
        **{
            name: _call_ratio_denominator(calls, field=call_field)
            for name, call_field in _CALL_RATIO_FIELDS.items()
        },
    }


def _granularity_warnings(
    budgets: list[BudgetResult],
    denominators: Mapping[str, int],
) -> list[str]:
    """Flag every rate ceiling this scope's sample is too coarse to express.

    A rate counted over ``n`` rows can only be a multiple of ``1/n``. When a
    ceiling sits below that step, every representable non-zero value breaches
    it, so the configured budget is silently equivalent to zero tolerance —
    the shipped ``max_tool_argument_drift: 0.01`` on a ten-row starter suite
    is exactly this. The gate is arithmetically right; what is missing is
    anyone saying that "1%" means "any at all" at this size.

    Deliberately silent in three cases:

    * ``allowed == 0`` — an explicit zero-tolerance budget is a choice, not a
      mistake.
    * ``denominator == 0`` — nothing was measured, which ``BudgetResult.conclusive``
      already reports; a second note would double-report one fact.
    * a representable budget — it means what it says.
    """
    out: list[str] = []
    for budget in budgets:
        rows = _RATE_CEILING_DENOMINATORS.get(budget.name)
        n = denominators.get(budget.name, 0)
        if rows is None or n <= 0 or budget.allowed <= 0:
            continue
        granularity = 1 / n
        if budget.allowed >= granularity:
            continue
        scope = "" if budget.scope == "overall" else f" in the '{budget.scope}' slice"
        # The one prose surface that keeps the ``evalshift.yaml`` field name,
        # in parentheses: this note asks the reader to edit that exact field.
        out.append(
            f"The {_label_lower(budget.name)} budget of {_pct(budget.allowed)} "
            f"({budget.name} in evalshift.yaml){scope} is below the "
            f"{_pct(granularity)} granularity of {n} {rows} — effective "
            f"tolerance is zero at this sample size.",
        )
    return out


def _zero_valued_call_warnings(calls: list[Call]) -> list[str]:
    """Explain every call-ratio budget whose 0.00 came out of an all-zero sample.

    :func:`_has_measured_ratio` already reports these as not conclusive. The
    flag alone is enough when there are no calls — an empty ``raw.jsonl``
    explains itself — but not here: the run made calls, they succeeded, and
    the budget still says it measured nothing. Without a line saying every
    value was zero, that reads as a mystery, which is exactly the silent
    signal this fix exists to remove.

    Silent when the pairing check already fails, on the same
    don't-double-report rule as :func:`_granularity_warnings`.

    Emitted once per run, not once per scope: ``evaluate_migration_policy``
    hands every slice the same run-level ``calls`` list, so per-scope notes
    would repeat one identical sentence.
    """
    out: list[str] = []
    for name, call_field in _CALL_RATIO_FIELDS.items():
        if not _has_paired_calls(calls, field=call_field):
            continue
        if _has_measured_ratio(calls, field=call_field):
            continue
        source, target = _paired_call_values(calls, field=call_field)
        noun = "a cost" if call_field == "cost_usd" else "a latency"
        out.append(
            f"The {_label_lower(name)} budget could not be measured: all "
            f"{len(source) + len(target)} error-free calls across both models "
            f"recorded {noun} of 0, so its observed 0.00 is a default, not a "
            f"measurement.",
        )
    return out


def _shared_ground_truth_warnings(records: list[EvalRecord]) -> list[str]:
    """Report the rows :func:`is_shared_ground_truth_miss` kept out of the rates.

    Leaving the denominator must not mean leaving the run. These rows are the
    loudest thing a run can say — the source model failed ground truth
    recorded from that same source model — and dropping them silently would
    trade one invisible defect for another.

    They are also counted, per row, as ``TOOL_GROUND_TRUTH_MISS`` in
    ``failure_categories``; this is the prose half, on the same channel as
    the granularity and unpriced-call notes.
    """
    n = sum(1 for r in _scored(records) if is_shared_ground_truth_miss(r))
    if not n:
        return []
    return [
        f"{n} tool-selection conformance comparisons are excluded from the "
        f"equivalence and regression rates: both models missed the recorded "
        f"ground truth by the same margin, so their zero delta is a shared "
        f"failure rather than evidence the migration is safe. "
        f"{BROKEN_HARNESS_CAUSES}",
    ]


def _source_derived_ground_truth_warnings(records: list[EvalRecord]) -> list[str]:
    """Disclose an ``against: expected`` gate whose ground truth is the source's.

    ``capture sync`` promotes ``expected_tools[].arguments`` verbatim from the
    source model's own recorded call, so on every promoted row the source
    scores 1.0 *by construction* and ``against: expected`` quietly degenerates
    into what ``against: source`` already measured: target deviation from
    source. Nothing is wrong with the number -- but ``source_score: 1.0``
    reads as evidence the source is correct, and on these rows it is not
    evidence of anything.

    Silent as soon as one row is ``reviewed``: a suite a human has started
    checking is no longer uniformly source-derived, and a blanket disclaimer
    over it would understate the rows they did check.

    No scoring change -- prose only, on the same channel as
    :func:`_shared_ground_truth_warnings`.
    """
    stamped = [
        provenance
        for r in _scored(records)
        if _is_kind(r, _TOOL_ARGUMENT_KIND)
        and isinstance(provenance := r.metadata.get("gt_provenance"), str)
    ]
    if not stamped or any(p != "captured" for p in stamped):
        return []
    return [
        f"{len(stamped)} tool-argument comparisons were scored against ground truth "
        f"promoted from the source model's own recorded calls: the source scores 1.0 "
        f"by construction, so this gate measures target deviation from source, not "
        f"correctness. Set provenance: reviewed on a golden row once its arguments "
        f"have been checked by hand.",
    ]


def _advisory_evaluator_names(
    *,
    comparisons: Sequence[ComparisonResult],
    records: Sequence[EvalRecord],
) -> set[str]:
    """Names of the advisory (``blocking: false``) evaluators in a run.

    Read from the rows when there are rows — every record carries the config
    flag — and from the :data:`ADVISORY_NOTE_PREFIX` note when there are
    none: an evaluator that scored nothing writes no records, so the note
    the analysis stage stamped on its synthesized comparison is the only
    trace of the flag left. Deriving from records alone is what once named
    a ``blocking: false`` evaluator as a blind *gate* the moment it scored
    no pair. One function, used by both :func:`evaluate_migration_policy`
    and :func:`unmeasured_gating_evaluators`, so the gating split and the
    blind-gate list can never disagree about who is advisory.
    """
    return {record.evaluator_name for record in records if not record.blocking} | {
        comparison.evaluator_name
        for comparison in comparisons
        if any(note.startswith(ADVISORY_NOTE_PREFIX) for note in comparison.notes)
    }


def unmeasured_gating_evaluators(
    *,
    comparisons: Sequence[ComparisonResult],
    records: Sequence[EvalRecord],
) -> list[str]:
    """Gating evaluators that ran and produced no comparable row at all.

    Their silence is not evidence of equivalence, so it must never underwrite
    a verdict — but advisory silence gates nothing by design, so advisory
    evaluators never belong here. Advisory-ness is read from the records
    when the evaluator has rows and from the comparison's own advisory note
    when it has none — see :func:`_advisory_evaluator_names`.

    Public because two surfaces need the same set and must not be allowed to
    disagree about it: this module's prose channel (:func:`_recommendations`,
    rendered in the terminal and in the report body) and the insights FACTS
    block, which is what a generated narrative is written from. A run that
    named the blind gates in one and claimed a clean sweep in the other is
    exactly the contradiction this function exists to prevent.

    Args:
        comparisons: Every comparison in the run, all slices.
        records: Every scored row, advisory ones included — ``blocking`` is
            what separates a gate from an advisory axis.

    Returns:
        Evaluator names, sorted, deduplicated. Empty when every gate scored.
    """
    advisory = _advisory_evaluator_names(comparisons=comparisons, records=records)
    return sorted(
        {
            comparison.evaluator_name
            for comparison in comparisons
            if comparison.evaluator_name not in advisory
            and any(note.startswith(UNMEASURED_NOTE_PREFIX) for note in comparison.notes)
        },
    )


def _verdict_for(
    *,
    comparisons: list[ComparisonResult],
    budgets: list[BudgetResult],
) -> MigrationVerdict:
    if _invariant_breaches(budgets):
        # Before the small-sample early return: a broken trace rule is an
        # assertion the team wrote down, not a statistic, so it needs no
        # sample size to mean something -- in a slice exactly as overall.
        return "fail"
    if comparisons and all(c.severity == "insufficient" for c in comparisons):
        return "inconclusive"
    if any(not b.passed and b.conclusive for b in budgets):
        return "fail"
    if any(c.severity in _BLOCKING_SEVERITIES for c in comparisons):
        return "fail"
    if any(not b.passed for b in budgets):
        # Breached, but the sample is too small for the CI to confirm it.
        return "inconclusive"
    if any(c.severity in _REGRESSION_SEVERITIES for c in comparisons):
        return "conditional_pass"
    return "pass"


def _distinct_slice_budgets(
    overall: list[BudgetResult],
    slices: Mapping[str, SliceDecision],
) -> list[BudgetResult]:
    """Slice budgets that measure something an overall row does not.

    ``max_cost_increase`` / ``max_latency_increase`` are averaged over the
    run's calls, so an inheriting slice restates the overall row number for
    number — and a run with three slices would report the same breach four
    times. A slice that *overrode* the budget, or one whose record-derived
    rate landed elsewhere, differs and survives.
    """
    seen = {_budget_identity(b) for b in overall}
    kept = []
    for decision in slices.values():
        for budget in decision.budget_results:
            identity = _budget_identity(budget)
            if identity in seen:
                continue
            seen.add(identity)
            kept.append(budget)
    return kept


def _budget_identity(budget: BudgetResult) -> tuple[str, float, float, bool, bool, int | None]:
    """Everything about a budget except which scope reported it."""
    return (
        budget.name,
        budget.observed,
        budget.allowed,
        budget.passed,
        budget.conclusive,
        budget.denominator,
    )


def _slice_budget_notes(budgets: list[BudgetResult]) -> list[str]:
    """One line per slice budget whose breach the sample confirmed.

    Only the conclusive ones: an unconfirmed breach makes the verdict
    ``inconclusive``, and :func:`_inconclusive_reason` already names it there.
    """
    return [
        f"The '{b.scope}' slice breached its {_label_lower(b.name)} budget"
        + (f" ({meaning})" if (meaning := BUDGET_MEANINGS.get(b.name)) else "")
        + f": {_budget_value(b.name, b.observed)} over n={b.denominator} vs the "
        f"{_budget_value(b.name, b.allowed)} "
        + ("minimum" if b.name.startswith("min_") else "limit")
        + f". Keep {b.scope} on the source model or widen that budget."
        for b in budgets
        if b.scope != "overall" and not b.passed and b.conclusive
    ]


def _invariant_breaches(budgets: list[BudgetResult]) -> list[BudgetResult]:
    """The breached ``max_invariant_violations`` rows -- always conclusive."""
    return [b for b in budgets if b.name == "max_invariant_violations" and not b.passed]


def _invariant_reason(breaches: list[BudgetResult]) -> str:
    """Why a broken trace rule failed the run, naming the worst scope."""
    worst = max(breaches, key=lambda b: b.observed)
    where = "" if worst.scope == "overall" else f" in the '{worst.scope}' slice"
    return (
        f"the target broke a hand-written trace rule on {int(worst.observed)} of "
        f"{worst.denominator} examples{where}, over the {int(worst.allowed)} allowed by "
        "migration_policy.max_invariant_violations. Trace rules are assertions, so no "
        "sample size softens them."
    )


def _violated_rule_ids(raw: Any) -> set[str]:
    """The ``rule_id`` of every violation in one side's metadata list."""
    if not isinstance(raw, list):
        return set()
    return {rule for v in raw if isinstance(v, dict) and isinstance(rule := v.get("rule_id"), str)}


def _shared_invariant_notes(records: list[EvalRecord]) -> list[str]:
    """One line when the source broke the same rule on the same example as the target.

    Not an exemption -- the target still fails -- but a shared break is the
    signal that either both models get this wrong or the rule no longer
    matches the toolset, and only the rule's owner can say which.
    """
    shared_rules: set[str] = set()
    examples = 0
    for r in _evidence(records):
        if not _is_kind(r, _INVARIANTS_KIND):
            continue
        common = _violated_rule_ids(r.metadata.get("source_violations")) & _violated_rule_ids(
            r.metadata.get("target_violations")
        )
        if common:
            shared_rules |= common
            examples += 1
    if not examples:
        return []
    return [
        f"The source model also broke {', '.join(sorted(shared_rules))} on {examples} "
        "example(s) where the target did. Either both models break the rule or it no "
        "longer matches the toolset: review it with its owner rather than raising "
        "max_invariant_violations."
    ]


def _pct(value: float) -> str:
    """``0.375`` → ``"37.5%"`` — prose never shows a reader raw fractions."""
    return f"{round(value * 100, 1):g}%"


def _budget_value(name: str, value: float) -> str:
    """One budget figure, in the unit the reader thinks in."""
    if name in _COUNT_BUDGETS:
        return f"{value:g}"
    return _pct(value)


def _budget_bound_noun(name: str) -> str:
    """What the configured limit is, directionally: a floor or a ceiling."""
    return "minimum" if name.startswith("min_") else "budget"


def _label_lower(name: str) -> str:
    """The budget's display name, cased for mid-sentence use."""
    label = BUDGET_LABELS.get(name, name.replace("_", " "))
    return label[:1].lower() + label[1:]


def _budget_label(budget: BudgetResult) -> str:
    """How a budget names itself in prose: words, with its meaning spelled out.

    Never the ``evalshift.yaml`` field name — a report reader is not required
    to know the config vocabulary, so ``max_tool_divergence`` renders as
    "tool-selection divergence (how often the target called different tools
    than the source)".

    Scope-qualified for slice budgets: since they gate the run verdict too
    (see :func:`evaluate_migration_policy`), a bare budget name would send the
    reader to the overall row — which is green in exactly the runs this line
    exists to explain.
    """
    phrase = _label_lower(budget.name)
    meaning = BUDGET_MEANINGS.get(budget.name)
    if meaning is not None:
        phrase = f"{phrase} ({meaning})"
    if budget.scope != "overall":
        phrase = f"{phrase} in the '{budget.scope}' slice"
    return phrase


def _inconclusive_reason(budgets: list[BudgetResult], *, n: int) -> str | None:
    """Human-readable reason for an inconclusive-by-small-n verdict."""
    unresolved = [b for b in budgets if not b.passed and not b.conclusive]
    if not unresolved:
        return None
    parts = []
    for b in unresolved:
        ci = ""
        if b.ci_low is not None and b.ci_high is not None:
            ci = f" (95% CI {_pct(b.ci_low)}-{_pct(b.ci_high)})"
        # A budget counted over its own rows — a slice's scope, or an overall
        # axis budget like tool-selection divergence — has a smaller sample
        # than the leading run-level ``n``, and that smaller sample is the one
        # that failed to resolve it. Name it whenever the two differ.
        scoped_n = (
            ""
            if b.denominator is None or b.denominator == n
            else f" over n={b.denominator} {_RATE_CEILING_DENOMINATORS.get(b.name, 'comparisons')}"
        )
        parts.append(
            f"{_budget_label(b)} was {_budget_value(b.name, b.observed)}{ci}{scoped_n} "
            f"vs the {_budget_value(b.name, b.allowed)} {_budget_bound_noun(b.name)}"
        )
    return (
        f"n={n} is too small to confirm the budget breach — "
        + "; ".join(parts)
        + ". Capture more examples and re-run."
    )


def _blocking_regressions(comparisons: list[ComparisonResult]) -> list[BlockingRegression]:
    return [
        BlockingRegression(
            prompt_id=c.prompt_id,
            evaluator_name=c.evaluator_name,
            slice_name=c.slice_name,
            severity=c.severity,
            delta_avg_score=c.delta_avg_score,
            effect_size=c.effect_size,
        )
        for c in comparisons
        if c.severity in _BLOCKING_SEVERITIES
    ]


def _failure_categories(records: list[EvalRecord]) -> list[FailureCategoryCount]:
    counter: Counter[str] = Counter()
    for r in records:
        raw = r.metadata.get("failure_categories", [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item:
                    counter[item] += 1
    return [
        FailureCategoryCount(category=category, count=count)
        for category, count in counter.most_common()
    ]


def _records_for_slice(
    records: list[EvalRecord],
    comparisons: list[ComparisonResult],
    slice_name: str,
) -> list[EvalRecord]:
    evaluator_names = {c.evaluator_name for c in comparisons if c.slice_name == slice_name}
    if not evaluator_names:
        return []
    return [r for r in records if r.evaluator_name in evaluator_names]


def _slice_policy(
    base: MigrationPolicy,
    override: SliceMigrationPolicy | None,
) -> MigrationPolicy:
    if override is None:
        return base
    return MigrationPolicy(
        max_overall_regression_rate=(
            base.max_overall_regression_rate
            if override.max_overall_regression_rate is None
            else override.max_overall_regression_rate
        ),
        max_critical_regressions=(
            base.max_critical_regressions
            if override.max_critical_regressions is None
            else override.max_critical_regressions
        ),
        min_equivalence_rate=(
            base.min_equivalence_rate
            if override.min_equivalence_rate is None
            else override.min_equivalence_rate
        ),
        max_tool_argument_drift=(
            base.max_tool_argument_drift
            if override.max_tool_argument_drift is None
            else override.max_tool_argument_drift
        ),
        max_tool_divergence=(
            base.max_tool_divergence
            if override.max_tool_divergence is None
            else override.max_tool_divergence
        ),
        max_invariant_violations=(
            base.max_invariant_violations
            if override.max_invariant_violations is None
            else override.max_invariant_violations
        ),
        tool_argument_drift_floor=(
            base.tool_argument_drift_floor
            if override.tool_argument_drift_floor is None
            else override.tool_argument_drift_floor
        ),
        max_cost_increase=(
            base.max_cost_increase
            if override.max_cost_increase is None
            else override.max_cost_increase
        ),
        max_latency_increase=(
            base.max_latency_increase
            if override.max_latency_increase is None
            else override.max_latency_increase
        ),
    )


def _recommendations(
    *,
    verdict: MigrationVerdict,
    slices: dict[str, SliceDecision],
    no_blocking_records: bool = False,
    unmeasured: list[str] | None = None,
    shared_ground_truth_only: bool = False,
) -> list[str]:
    safe = sorted(name for name, decision in slices.items() if decision.verdict == "pass")
    unsafe = sorted(name for name, decision in slices.items() if decision.verdict == "fail")
    # More advisory examples can never gate quality — the fix is config, not
    # sample size. Applies whatever the verdict: a cost/latency fail still
    # leaves correctness unmeasured.
    enable_blocking = (
        "Set blocking: true on at least one trusted evaluator in "
        "evalshift.yaml to get a pass/fail verdict."
    )
    # Distinct from `enable_blocking`: these evaluators *are* blocking and
    # *did* run — they simply found no applicable pair. More examples of the
    # same shape would not help, so never pair this with "collect more".
    unmeasured = unmeasured or []
    nothing_measured = (
        "These blocking evaluators scored no comparable pair and did not gate "
        "this run: " + ", ".join(unmeasured) + ". On an agent suite this usually "
        "means both models answered with tool calls and no prose; treat their "
        "silence as unknown, not as equivalence."
    )
    # A third wrong-advice case, and the sharpest: the rows exist, they scored,
    # and every one of them was excluded for measuring the harness. More
    # examples from that harness are more excluded rows — the denominator stays
    # empty however many are added — so "collect more" is not merely unhelpful
    # here, it is the one instruction guaranteed not to work.
    fix_harness = (
        "Fix the eval harness before collecting more examples: every blocking "
        "row was a shared ground-truth miss, so more pairs from the same setup "
        "measure the same thing."
    )
    if verdict == "pass":
        return ["Safe to migrate under the configured policy."]
    if verdict == "conditional_pass" and unmeasured:
        return [
            nothing_measured,
            "Safe to migrate only where a blocking evaluator actually scored.",
        ]
    if verdict == "inconclusive":
        out = []
        if shared_ground_truth_only:
            out.append(fix_harness)
        if unmeasured:
            out.append(nothing_measured)
        if out:
            return out
        if no_blocking_records:
            return [enable_blocking]
        return ["Collect more examples before making a migration decision."]
    if safe and unsafe:
        primary = (
            "Safe slices: "
            + ", ".join(safe)
            + ". Keep "
            + ", ".join(unsafe)
            + " on the source model."
        )
    else:
        primary = "Do not migrate globally under the configured policy."
    out = [primary]
    if unmeasured:
        out.append(nothing_measured)
    if no_blocking_records:
        out.append(enable_blocking)
    return out


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _paired_call_values(calls: list[Call], *, field: str) -> tuple[list[float], list[float]]:
    """Error-free ``field`` values for the source and target roles.

    Both the cost/latency ratio and its ``conclusive`` flag read this one
    function, so "we had samples to compare" can never disagree with the
    early return in :func:`_relative_increase`.
    """
    source = [float(getattr(c, field)) for c in calls if c.role == "source" and c.error is None]
    target = [float(getattr(c, field)) for c in calls if c.role == "target" and c.error is None]
    return source, target


def _has_paired_calls(calls: list[Call], *, field: str) -> bool:
    """Whether :func:`_relative_increase` had anything to compare for ``field``.

    False means its 0.0 is the "nothing to divide" default rather than a
    measured ratio — the budget row is unmeasured, not clean.

    Necessary but not sufficient for the budget to be conclusive; see
    :func:`_has_measured_ratio`, which is what ``_budget_results`` is given.
    """
    source, target = _paired_call_values(calls, field=field)
    return bool(source) and bool(target)


def _call_ratio_denominator(calls: list[Call], *, field: str) -> int:
    """How many error-free calls the ``field`` ratio was averaged over.

    Zero when either role contributed none: :func:`_relative_increase` divides
    one role's average by the other's, so one-sided data compares nothing and
    its 0.0 is the "nothing to divide" default (:func:`_has_paired_calls`) —
    which is exactly the "counted, and there was nothing to count" that
    ``BUNDLE_SPEC.md`` reserves ``0`` for.

    Counts *calls on both sides*, not pairs. ``BUNDLE_SPEC.md`` calls this
    "paired calls", but the CLI never pairs individual calls: it averages each
    role independently over whatever error-free calls that role produced, and
    the two counts need not match. There is therefore no pair count to report,
    and the number of calls behind the two averages is the honest sample size.
    It is deliberately *not* the same question as :func:`_has_measured_ratio`:
    a run whose calls all carry ``cost_usd = 0`` counted every one of them and
    still measured nothing, so it reports a positive denominator beside
    ``conclusive: false``.
    """
    source, target = _paired_call_values(calls, field=field)
    if not source or not target:
        return 0
    return len(source) + len(target)


def _has_measured_ratio(calls: list[Call], *, field: str) -> bool:
    """Whether :func:`_relative_increase` measured ``field``, rather than defaulting.

    It defaults to 0.0 twice over: when a role has no error-free call at all
    (:func:`_has_paired_calls`), and when both roles average zero — an
    unpriced model pair, whose calls carry ``cost_usd = 0.0`` (or
    ``latency_ms = 0``) on every row. The second case has non-empty call
    lists, so pairing alone reports it as a confident "the target did not
    cost more" for a quantity nobody ever priced.

    Both-sides-zero is therefore treated as **unmeasured**. ``Call.cost_usd``
    is ``0.0`` both when a model is unpriced and when it is genuinely free, and
    nothing else in the record separates them, so a genuinely free pair is
    marked not conclusive too. That is the deliberate trade: a false confident
    pass on an unpriced migration is the worse failure, and "could not tell" is
    recoverable by pricing the models.

    Kept in lockstep with :func:`_relative_increase` — both read
    :func:`_paired_call_values`, and the branch below mirrors its early
    returns, so the flag can never disagree with the number it describes.
    """
    source, target = _paired_call_values(calls, field=field)
    if not source or not target:
        return False
    return (sum(source) / len(source)) > 0 or (sum(target) / len(target)) > 0


def _relative_increase(calls: list[Call], *, field: str) -> float:
    source, target = _paired_call_values(calls, field=field)
    if not source or not target:
        return 0.0
    source_avg = sum(source) / len(source)
    target_avg = sum(target) / len(target)
    if source_avg <= 0:
        return 0.0 if target_avg <= 0 else 1.0
    return max(0.0, (target_avg - source_avg) / source_avg)


__all__ = [
    "BUDGET_LABELS",
    "BUDGET_MEANINGS",
    "BudgetResult",
    "FailureCategoryCount",
    "MigrationDecision",
    "MigrationVerdict",
    "PolicyMetricSummary",
    "SliceDecision",
    "evaluate_migration_policy",
    "inconclusive_decision",
    "is_shared_ground_truth_miss",
    "unmeasured_gating_evaluators",
]
