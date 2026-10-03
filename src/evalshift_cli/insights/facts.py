"""The pre-rendered FACTS block a narrative is written from.

Every figure the prose may mention is rendered here into its final display
string, so the generating model copies rather than calculates. ``allowed_numbers``
is the permit-list the generator validates its output against: a numeric token
outside that set means the model derived something, and in a report that gates
merges a derived number is a defect even when it happens to be right.

Pure — no network, no model. That is what makes the generator testable.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from evalshift_cli.analysis.policy import (
    BUDGET_LABELS,
    MigrationDecision,
    unmeasured_gating_evaluators,
)
from evalshift_cli.analysis.statistics import ComparisonResult
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.insights.models import (
    MAX_REGRESSION_SAMPLES,
    MAX_SAMPLE_TEXT_CHARS,
    NUMERIC_TOKEN_RE,
)
from evalshift_cli.reports.economics import PromptEconomics
from evalshift_cli.runner.models import RunState

#: Below this, a p-value is printed as an inequality. ``0.0000`` is wrong in a
#: way a reader cannot detect.
P_VALUE_FLOOR: float = 0.0001

#: Rendered in place of a ratio that isn't a measurement. Deliberately free of
#: digits so it can never be mistaken for a figure.
NOT_COMPARABLE: str = "not comparable"
NOT_AVAILABLE: str = "n/a"

#: Rendered in place of an outcome rate with no denominator behind it.
#: ``policy._rate`` returns ``0.0`` over an empty denominator, so a run that
#: gated nothing renders "regression rate 0%, equivalence 0%" — figures a
#: model is invited to restate as "no regressions were detected". Digit-free
#: for the same reason as :data:`NOT_COMPARABLE`: it must never enter the
#: permit-list as a number the narrative may then reuse.
NOT_MEASURED: str = "not measured"

#: The sentence appended to the FACTS block when the rates are absent, and
#: the reason they are. Also digit-free.
_RATES_BASIS_UNMEASURED: str = (
    "the outcome rates are not measured on this run — no blocking evaluator "
    "produced a comparable row, or every row it produced was excluded as a "
    "shared ground-truth miss. Do not describe the run as equivalent, "
    "consistent, unchanged or free of regressions: nothing was compared."
)

#: The sentence appended to the FACTS block when some gate measured nothing,
#: and the instruction that goes with it. Digit-free for the same reason as
#: :data:`_RATES_BASIS_UNMEASURED`: it is not in the permit-list, so a numeral
#: here is one the validator would reject out of the model's own prose.
#:
#: Distinct from the rates basis, and not covered by it. That one fires on
#: ``overall.n_records == 0`` — the whole run measured nothing. This one fires
#: when *some* gate did: a run whose tool evaluators scored sixteen rows has a
#: perfectly good denominator, and the two text evaluators that produced no
#: comparable row at all are invisible to a run-level count.
_COVERAGE_BASIS_UNMEASURED: str = (
    "not every gate measured something on this run. Each budget and evaluator "
    "named as unmeasured was handed no comparable row, so it did not pass — it "
    "was blind. Do not write that all budgets passed, that every constraint or "
    "requirement is met, that the migration is safe, or that nothing regressed, "
    "on the strength of a gate that measured nothing. Name the blind gates and "
    "call their silence unknown."
)

#: The two policy budgets that are ratios of increase rather than rates.
_INCREASE_BUDGETS: frozenset[str] = frozenset({"max_cost_increase", "max_latency_increase"})
#: The budgets that are plain counts.
_COUNT_BUDGETS: frozenset[str] = frozenset({"max_critical_regressions", "max_invariant_violations"})


@dataclass(frozen=True, slots=True)
class ExampleFact:
    """One example as the narrative sees it: its worst delta and its text.

    Deliberately not ``reports.json.ExampleRow`` — that row carries the deltas
    but no output text, and ``TopRegression`` carries the text but only for the
    worst five of each prompt. The insights path needs both on every example.
    """

    example_id: str
    worst_delta_score: float | None
    input_text: str | None = None
    source_output: str | None = None
    target_output: str | None = None


@dataclass(frozen=True, slots=True)
class RegressionSample:
    """One regression the model is shown, so it can describe *behavior*."""

    example_id: str
    delta: float
    input_text: str
    source_output: str
    target_output: str


@dataclass(frozen=True, slots=True)
class Facts:
    """Everything a narrative may refer to, already rendered.

    ``rendered`` holds figures only — copy targets for the model.
    ``allowed_numbers`` is every one of them plus its bare-numeral form, so
    the validator accepts ``102`` written out of ``+102%``.
    """

    rendered: dict[str, str]
    allowed_numbers: frozenset[str]
    regression_samples: list[RegressionSample]
    verdict: str
    source_model: str
    target_model: str
    worst_evaluator: str
    blocking_evaluators: list[str]
    failure_categories: list[tuple[str, int]]
    budget_limits: dict[str, str]
    #: Why the three outcome rates read :data:`NOT_MEASURED`, or ``""`` when
    #: they are real. A blank in place of a rate is ambiguous — the reason
    #: has to travel with the absence, or the model fills it in.
    rates_basis: str = ""
    #: Budgets counted over an empty sample: ``0/0`` renders as a clean row and
    #: measures nothing (:attr:`~evalshift_cli.analysis.policy.BudgetResult.measured`).
    unmeasured_budgets: list[str] = field(default_factory=list)
    #: Gating evaluators that produced no comparable row at all. The same set
    #: the decision's own ``recommendations`` name, by construction — see
    #: :func:`~evalshift_cli.analysis.policy.unmeasured_gating_evaluators`.
    unmeasured_evaluators: list[str] = field(default_factory=list)
    #: Why coverage is incomplete, or ``""`` when every gate scored. Same rule
    #: as :attr:`rates_basis`: a name without a reason is a fact the model will
    #: explain for itself.
    coverage_basis: str = ""


def build_facts(
    *,
    decision: MigrationDecision,
    comparisons: Sequence[ComparisonResult],
    economics: PromptEconomics,
    examples: Sequence[ExampleFact],
    records: Sequence[EvalRecord],
    state: RunState,
) -> Facts:
    """Render the run's figures into the block a narrative is written from.

    Args:
        decision: The run's migration decision. Required — callers with no
            configured ``migration_policy`` pass the ``inconclusive_decision``
            the bundle path already builds, rather than ``None``, so the
            narrative always has a verdict and a budget table to describe.
        comparisons: Every statistical comparison in the run. The most
            negative effect size supplies the headline ``effect_size`` and
            ``p_value``.
        economics: The *run-level* rollup (every call, both roles), not one
            prompt's.
        examples: One entry per example, worst delta plus text.
        records: Every scored row, advisory ones included. Required rather
            than optional: ``blocking`` is the only thing that separates a
            gate that measured nothing from an advisory axis that did, and an
            empty default would quietly report every advisory silence as a
            blind gate.
        state: The run's state, for the model ids.
    """
    source = economics.source
    target = economics.target
    deltas = [e.worst_delta_score for e in examples if e.worst_delta_score is not None]
    worst_comparison = _worst_comparison(comparisons)
    budget_limits = {b.name: _budget_limit(b.name, b.allowed) for b in decision.budget_results}
    # The three outcome rates share one denominator, so they are measured or
    # missing together. ``_rate`` returns 0.0 over an empty one, which reads
    # as "nothing regressed" rather than "nothing was compared" — and that is
    # the equivalence claim this phase exists to stop.
    measured = decision.overall.n_records > 0
    rates_basis = "" if measured else _RATES_BASIS_UNMEASURED
    # Per-gate blindness, which the run-level flag above cannot see: a run
    # whose tool evaluators scored plenty has `measured` True while its text
    # evaluators produced no comparable row at all, and a budget counted over
    # an empty sample renders "observed 0.00, passed" either way.
    # Display names, not config keys: this list flows verbatim into both the
    # generation prompt and the templated fallback, so it holds the words a
    # reader is allowed to see.
    unmeasured_budgets = [
        BUDGET_LABELS.get(b.name, b.name) for b in decision.budget_results if not b.measured
    ]
    unmeasured_evaluators = unmeasured_gating_evaluators(
        comparisons=comparisons,
        records=records,
    )
    coverage_basis = (
        _COVERAGE_BASIS_UNMEASURED if (unmeasured_budgets or unmeasured_evaluators) else ""
    )

    rendered: dict[str, str] = {
        "verdict": decision.verdict.replace("_", " ").upper(),
        # Passed *and* measured. A budget handed an empty sample carries
        # `passed=True` by arithmetic — 0/0 is within every ceiling — so
        # counting bare `passed` renders "7 of 7" over a run with a blind
        # gate, which is the clean sweep a narrative then restates.
        "budgets_passed": str(sum(1 for b in decision.budget_results if b.passed and b.measured)),
        "budgets_total": str(len(decision.budget_results)),
        "budgets_unmeasured": str(len(unmeasured_budgets)),
        "blocking_regressions": str(len(decision.blocking_regressions)),
        "critical_regressions": str(decision.overall.critical_regressions),
        "n_examples": str(len(examples)),
        "n_calls": str(source.calls + target.calls),
        "cost_source_usd": _usd(source.total_cost_usd),
        "cost_target_usd": _usd(target.total_cost_usd),
        "cost_delta_pct": _relative_pct(source.total_cost_usd, target.total_cost_usd),
        # Latency averages cover only calls measured fully live on this run
        # (cache hits keep their original latency and are excluded via
        # ``Call.latency_replayed``). A role with no such call averages 0.0,
        # meaning "unmeasured", so either side lacking one leaves no
        # percentage to report — not -100% or a division by zero.
        "latency_delta_pct": (
            _relative_pct(source.latency_ms_avg, target.latency_ms_avg)
            if source.live_calls and target.live_calls
            else NOT_COMPARABLE
        ),
        "cost_ceiling_pct": budget_limits.get("max_cost_increase", NOT_AVAILABLE),
        "latency_ceiling_pct": budget_limits.get("max_latency_increase", NOT_AVAILABLE),
        "regression_rate_pct": (
            _rate_pct(decision.overall.regression_rate) if measured else NOT_MEASURED
        ),
        "equivalence_rate_pct": (
            _rate_pct(decision.overall.equivalent_rate) if measured else NOT_MEASURED
        ),
        "improved_rate_pct": (
            _rate_pct(decision.overall.improved_rate) if measured else NOT_MEASURED
        ),
        "effect_size": (
            _signed(worst_comparison.effect_size, 2)
            if worst_comparison is not None
            else NOT_AVAILABLE
        ),
        "p_value": (
            _p_value(worst_comparison.p_value_corrected)
            if worst_comparison is not None
            else NOT_AVAILABLE
        ),
        "worst_delta": _signed(min(deltas), 4) if deltas else NOT_AVAILABLE,
        "median_delta": _signed(statistics.median(deltas), 4) if deltas else NOT_AVAILABLE,
        "best_delta": _signed(max(deltas), 4) if deltas else NOT_AVAILABLE,
    }

    failure_categories = [(c.category, c.count) for c in decision.failure_categories]
    return Facts(
        rendered=rendered,
        allowed_numbers=_allowed_numbers(
            rendered=rendered,
            budget_limits=budget_limits,
            failure_categories=failure_categories,
            model_ids=(state.models.source, state.models.target),
            n_examples=len(examples),
        ),
        regression_samples=_regression_samples(examples),
        verdict=decision.verdict,
        source_model=state.models.source,
        target_model=state.models.target,
        worst_evaluator=(
            worst_comparison.evaluator_name if worst_comparison is not None else NOT_AVAILABLE
        ),
        blocking_evaluators=_unique(r.evaluator_name for r in decision.blocking_regressions),
        failure_categories=failure_categories,
        budget_limits=budget_limits,
        rates_basis=rates_basis,
        unmeasured_budgets=unmeasured_budgets,
        unmeasured_evaluators=unmeasured_evaluators,
        coverage_basis=coverage_basis,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Unicode minus (U+2212). Matches the design; the validator's token regex
#: accepts it alongside the ASCII hyphen.
_MINUS = "−"


def _usd(value: float) -> str:
    return f"${value:.4f}"


def _signed(value: float, digits: int) -> str:
    """Render a signed figure, using U+2212 for negatives."""
    text = f"{abs(value):.{digits}f}"
    return f"{_MINUS}{text}" if value < 0 else f"+{text}"


def _rate_pct(value: float) -> str:
    """Render a 0-1 rate as an unsigned percentage."""
    return f"{round(value * 100, 1):g}%"


def _signed_pct(value: float) -> str:
    """Render a fractional change as a signed percentage."""
    magnitude = f"{round(abs(value) * 100, 1):g}%"
    return f"{_MINUS}{magnitude}" if value < 0 else f"+{magnitude}"


def _relative_pct(source: float, target: float) -> str:
    """Render ``(target - source) / source``, or say it isn't measurable.

    Not ``policy._relative_increase``: that clamps at zero because a budget
    only cares about increases, which would render a genuinely cheaper target
    as ``+0%``.
    """
    if source <= 0:
        return NOT_COMPARABLE
    return _signed_pct((target - source) / source)


def _p_value(value: float) -> str:
    return f"< {P_VALUE_FLOOR}" if value < P_VALUE_FLOOR else f"{value:.4f}"


def _budget_limit(name: str, allowed: float) -> str:
    if name in _COUNT_BUDGETS:
        return f"{round(allowed):g}"
    if name in _INCREASE_BUDGETS:
        return _signed_pct(allowed)
    return _rate_pct(allowed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _worst_comparison(comparisons: Sequence[ComparisonResult]) -> ComparisonResult | None:
    """The comparison with the most negative effect size, ties broken by p."""
    tested = [c for c in comparisons if c.test != "skipped"]
    if not tested:
        return None
    return min(tested, key=lambda c: (c.effect_size, c.p_value_corrected))


def _regression_samples(examples: Sequence[ExampleFact]) -> list[RegressionSample]:
    """The worst regressions, worst first, with their text bounded."""
    regressed = [e for e in examples if e.worst_delta_score is not None and e.worst_delta_score < 0]
    regressed.sort(key=lambda e: (e.worst_delta_score or 0.0, e.example_id))
    return [
        RegressionSample(
            example_id=e.example_id,
            delta=e.worst_delta_score or 0.0,
            input_text=_bounded(e.input_text),
            source_output=_bounded(e.source_output),
            target_output=_bounded(e.target_output),
        )
        for e in regressed[:MAX_REGRESSION_SAMPLES]
    ]


def _bounded(text: str | None) -> str:
    return (text or "")[:MAX_SAMPLE_TEXT_CHARS]


def _allowed_numbers(
    *,
    rendered: dict[str, str],
    budget_limits: dict[str, str],
    failure_categories: Sequence[tuple[str, int]],
    model_ids: tuple[str, ...],
    n_examples: int,
) -> frozenset[str]:
    """Every figure the narrative may legitimately contain.

    Beyond the rendered figures themselves this admits each one's bare-numeral
    form (``+102%`` also admits ``102``), the counts behind the budget table
    and failure categories, the numerals inside the model ids — writing
    ``gemini-3.1-flash`` must not read as an invented figure — and every
    integer up to ``n_examples`` so the model can write "15 of 21".
    """
    allowed: set[str] = {str(index) for index in range(n_examples + 1)}
    sources: list[str] = [
        *rendered.values(),
        *budget_limits.values(),
        *(str(count) for _, count in failure_categories),
        *model_ids,
    ]
    for text in sources:
        allowed.add(text)
        for token in NUMERIC_TOKEN_RE.findall(text):
            allowed.add(token)
            allowed.add(token.lstrip(f"+{_MINUS}-$").rstrip("%").replace(",", ""))
    return frozenset(allowed)


def _unique(values: Iterable[str]) -> list[str]:
    """Deduplicate preserving first-seen order."""
    return list(dict.fromkeys(values))


__all__ = [
    "NOT_AVAILABLE",
    "NOT_COMPARABLE",
    "NOT_MEASURED",
    "P_VALUE_FLOOR",
    "ExampleFact",
    "Facts",
    "RegressionSample",
    "build_facts",
]
