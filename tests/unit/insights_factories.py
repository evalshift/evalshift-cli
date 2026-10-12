"""Builders for the reference run the insights tests are written against.

The spec's reference run: 21 examples, ``+102%`` cost, ``d = −2.51``. Shared by
the facts/templates suite, the generator suite and the stage suite so the three
cannot disagree about what a run looks like — the generator's fallback
assertions read figures that ``build_facts`` rendered from these very numbers.

:class:`FakeModelClient` lives here for the same reason: **no insights test ever
hits a real model**, and one double shared across the suites keeps that true.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from evalshift_cli.analysis.policy import (
    BlockingRegression,
    BudgetResult,
    FailureCategoryCount,
    MigrationDecision,
    PolicyMetricSummary,
)
from evalshift_cli.analysis.statistics import UNMEASURED_NOTE_PREFIX, ComparisonResult
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.insights.facts import ExampleFact
from evalshift_cli.models.client import CompletionResult
from evalshift_cli.reports.economics import PromptEconomics, RoleEconomics
from evalshift_cli.runner.models import RunModels, RunState


class FakeModelClient:
    """A :class:`~evalshift_cli.models.client.ModelClient` stand-in.

    Replays queued response texts and counts calls, so a test can assert that a
    cache hit cost nothing without mocking a provider.
    """

    def __init__(self) -> None:
        self._responses: list[str] = []
        self.call_count = 0
        self.prompts: list[str] = []

    def queue_responses(self, *responses: str) -> None:
        self._responses.extend(responses)

    async def complete(self, *, model: str, prompt: str, **_: Any) -> CompletionResult:
        self.call_count += 1
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("generator asked for more responses than were queued")
        return CompletionResult(
            text=self._responses.pop(0),
            model_id=model,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_ms=1,
        )


def generation_payload(**overrides: Any) -> str:
    """A valid generation response body, overridable field by field.

    Carries no figures at all by default — the numeric validator holds a
    generation to the *run's* facts, and a canned payload cannot know them.
    """
    body: dict[str, Any] = {
        "verdict_summary": "PASS under the configured policy.",
        "advisory_summary": "Semantic similarity fell on every prompt.",
        "economics_summary": "The target costs more per call than the source.",
        "recommendation": "Safe to migrate under the configured policy.",
        "findings": [],
    }
    body.update(overrides)
    return json.dumps(body)


#: Source total from the reference run in the design spec. Rendered "$0.0101".
COST_SOURCE = 0.01008325
#: Exactly +102% of the source, so ``cost_delta_pct`` renders "+102%".
COST_TARGET = COST_SOURCE * 2.02

#: 21 examples spanning −0.08 → +0.02; median lands on −0.03.
MIXED_DELTAS = [round(-0.08 + index * 0.005, 4) for index in range(21)]
#: Same size, nothing negative.
CLEAN_DELTAS = [round(index * 0.005, 4) for index in range(21)]


def role(**overrides: Any) -> RoleEconomics:
    """One role's rollup, cache-replayed unless overridden."""
    defaults: dict[str, Any] = {
        "calls": 21,
        "live_calls": 0,
        "cached_calls": 21,
        "failed_calls": 0,
        "truncated_calls": 0,
        "empty_output_calls": 0,
        "total_cost_usd": COST_SOURCE,
        "total_input_tokens": 20695,
        "total_output_tokens": 3273,
        "latency_ms_avg": 0.0,
        "latency_ms_p95": 0.0,
    }
    return RoleEconomics(**{**defaults, **overrides})


def economics() -> PromptEconomics:
    """The run-level rollup: a source and a target that differ only in cost."""
    return PromptEconomics(source=role(), target=role(total_cost_usd=COST_TARGET))


def state() -> RunState:
    """A completed run's state, for the model ids the narrative may name."""
    return RunState(
        run_id="r_20260803_abcdef",
        status="completed",
        config_hash="sha256:cafe",
        started_at=datetime(2026, 8, 3, 9, 0, tzinfo=UTC),
        models=RunModels(
            source="gemini/gemini-2.5-flash",
            target="gemini/gemini-3.1-flash-lite-preview",
        ),
        prompt_ids=["replay"],
        suite_path="golden.jsonl",
        total_evaluations=42,
        completed_evaluations=42,
    )


def budgets(*, cost_allowed: float = 2.0, latency_allowed: float = 0.30) -> list[BudgetResult]:
    """The six budgets a fully configured migration policy evaluates."""
    return [
        BudgetResult(
            name="max_overall_regression_rate",
            observed=0.0,
            allowed=0.03,
            passed=True,
            ci_low=0.0,
            ci_high=0.15,
        ),
        BudgetResult(name="max_critical_regressions", observed=0.0, allowed=0.0, passed=True),
        BudgetResult(
            name="min_equivalence_rate",
            observed=1.0,
            allowed=0.95,
            passed=True,
            ci_low=0.85,
            ci_high=1.0,
        ),
        BudgetResult(name="max_tool_argument_drift", observed=0.0, allowed=0.01, passed=True),
        BudgetResult(name="max_cost_increase", observed=1.02, allowed=cost_allowed, passed=True),
        BudgetResult(
            name="max_latency_increase", observed=0.0, allowed=latency_allowed, passed=True
        ),
    ]


def decision(
    *,
    verdict: str = "pass",
    n_records: int = 42,
    regression_rate: float = 0.0,
    equivalent_rate: float = 0.714,
    blocking: list[BlockingRegression] | None = None,
    categories: list[FailureCategoryCount] | None = None,
    budget_results: list[BudgetResult] | None = None,
) -> MigrationDecision:
    """A migration decision, passing under the configured policy by default.

    ``n_records`` is the count the three outcome rates are computed over.
    Pass ``0`` for a run whose every blocking row was dropped — nothing to
    gate quality, and therefore no rate to report: ``policy._rate`` returns
    ``0.0`` on an empty denominator, so all three read ``0%`` and a reader
    cannot tell "nothing regressed" from "nothing was measured".
    """
    metrics = PolicyMetricSummary(
        n_records=n_records,
        improved_rate=0.286,
        equivalent_rate=equivalent_rate,
        regression_rate=regression_rate,
        critical_regressions=0,
        tool_argument_drift_rate=0.0,
        cost_increase_rate=1.02,
        latency_increase_rate=0.0,
    )
    return MigrationDecision(
        run_id="r_20260803_abcdef",
        source_model="gemini/gemini-2.5-flash",
        target_model="gemini/gemini-3.1-flash-lite-preview",
        verdict=verdict,  # type: ignore[arg-type]
        overall=metrics,
        slices={},
        budget_results=budget_results if budget_results is not None else budgets(),
        blocking_regressions=blocking or [],
        failure_categories=categories or [],
        recommendations=["Safe to migrate under the configured policy."],
    )


def comparisons(
    *, effect_size: float = -2.51, p_corrected: float = 0.00002
) -> list[ComparisonResult]:
    """Two comparisons: one regressing evaluator and one improving one."""
    return [
        ComparisonResult(
            prompt_id="replay",
            evaluator_name="semantic.cosine",
            slice_name="all",
            n=21,
            test="wilcoxon",
            statistic=0.0,
            p_value=0.000009,
            p_value_corrected=p_corrected,
            effect_size=effect_size,
            effect_size_ci_low=-3.4,
            effect_size_ci_high=-1.6,
            delta_avg_score=-0.031,
            severity="low",
            notes=[],
        ),
        ComparisonResult(
            prompt_id="replay",
            evaluator_name="llm_judge.equivalence",
            slice_name="all",
            n=21,
            test="wilcoxon",
            statistic=0.0,
            p_value=0.4,
            p_value_corrected=0.4,
            effect_size=0.31,
            effect_size_ci_low=-0.2,
            effect_size_ci_high=0.9,
            delta_avg_score=0.02,
            severity="none",
            notes=[],
        ),
    ]


def unmeasured_budget(name: str = "max_tool_divergence") -> BudgetResult:
    """A budget whose sample was empty.

    ``0/0`` renders ``observed 0.00, passed`` — a clean row over nothing. The
    policy layer marks it ``conclusive=False`` for exactly that reason, which
    is the only thing separating it from a budget that really held.
    """
    return BudgetResult(
        name=name,
        observed=0.0,
        allowed=0.2,
        passed=True,
        conclusive=False,
        denominator=0,
    )


#: A severity-only fail's ``MigrationDecision.reason``, as the policy words it.
#: Carries figures: a generation that copies them must not be rejected.
SEVERITY_REASON = (
    "semantic.cosine regressed at critical severity on prompt replay "
    "(effect size -2.15, adjusted p 0.0004). A critical or high regression from a "
    "blocking evaluator fails the run on its own, whatever the budgets say."
)


def converter_decision() -> MigrationDecision:
    """r_20261012_converter_d418e5: the latency ceiling breached, every other budget held.

    The run whose narrative blamed "Trace-rule violations" — a budget that
    passed at 0 against a limit of 1 — because the prompt listed each budget's
    limit without its result. Six held budgets, one breached, one blind.
    """
    return decision(
        verdict="fail",
        budget_results=[
            *budgets()[:5],
            BudgetResult(name="max_latency_increase", observed=1.753, allowed=0.30, passed=False),
            BudgetResult(name="max_invariant_violations", observed=0.0, allowed=1.0, passed=True),
            unmeasured_budget(),
        ],
    )


def unmeasured_comparison(evaluator_name: str) -> ComparisonResult:
    """An evaluator that was handed pairs and produced no comparable row.

    Shaped exactly as ``analysis.statistics`` emits one: ``n=0``,
    ``test="skipped"``, ``severity="insufficient"`` and the note prefix the
    policy layer selects on.
    """
    return ComparisonResult(
        prompt_id="replay",
        evaluator_name=evaluator_name,
        slice_name="all",
        n=0,
        test="skipped",
        statistic=0.0,
        p_value=1.0,
        p_value_corrected=1.0,
        effect_size=0.0,
        effect_size_ci_low=0.0,
        effect_size_ci_high=0.0,
        delta_avg_score=0.0,
        severity="insufficient",
        notes=[f"{UNMEASURED_NOTE_PREFIX} this evaluator scored no comparable pair"],
    )


def records(*, advisory: tuple[str, ...] = ()) -> list[EvalRecord]:
    """One scored row per evaluator the reference run's comparisons name.

    Only ``blocking`` matters to the facts layer: it is what separates a gate
    that measured nothing from an advisory axis that did. Names in
    ``advisory`` are written ``blocking=False``.
    """
    return [
        EvalRecord(
            run_id="r_20260803_abcdef",
            prompt_id="replay",
            example_id="cap_0000",
            evaluator_name=name,
            source_score=1.0,
            target_score=1.0,
            delta=0.0,
            blocking=name not in advisory,
        )
        for name in ("semantic.cosine", "llm_judge.equivalence")
    ]


def examples(*, deltas: list[float]) -> list[ExampleFact]:
    """One ``ExampleFact`` per delta, worst-first ordering left to the caller."""
    return [
        ExampleFact(
            example_id=f"cap_{index:04d}",
            worst_delta_score=delta,
            # The worst example carries oversized text so the 2000-char cap
            # is exercised by the sampling path.
            input_text="i" * (3000 if index == 0 else 40),
            source_output="s" * (3000 if index == 0 else 40),
            target_output="t" * (3000 if index == 0 else 40),
        )
        for index, delta in enumerate(deltas)
    ]


def sample_run_kwargs() -> dict[str, Any]:
    """``build_facts`` kwargs for the spec's reference run."""
    return {
        "decision": decision(),
        "comparisons": comparisons(),
        "economics": economics(),
        "examples": examples(deltas=MIXED_DELTAS),
        "records": records(),
        "state": state(),
    }


def passing_run_kwargs() -> dict[str, Any]:
    """The same shape with no negative deltas anywhere."""
    return {
        "decision": decision(),
        "comparisons": comparisons(effect_size=0.4, p_corrected=0.6),
        "economics": economics(),
        "examples": examples(deltas=CLEAN_DELTAS),
        "records": records(),
        "state": state(),
    }
