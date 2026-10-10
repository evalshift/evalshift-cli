"""``evaluate`` → ``analyze`` over the run that lied about equivalence.

``r_20260820_project_insights_143a5f`` is replayed from a frozen ``raw.jsonl``
(see :mod:`tests.scoring_fixtures`) through the real stage cores. No model is
called: every output on that run is empty, so both text evaluators
short-circuit before reaching a provider, and both providers are monkeypatched
to explode so that stays true.

What the run shipped, and what this module refuses to let it ship again:

* 40 rows in ``scores.jsonl``, **30 of them skipped** — ``tool_arguments``,
  ``semantic.cosine`` and ``llm_judge.equivalence`` each wrote ten maximum
  scores over a comparison that never happened;
* the remaining ten ``tool_selection`` rows all ``0.0 / 0.0``: both models
  violated the ``expected_no_tools`` ground truth, so ``delta == 0`` and
  ``policy._is_equivalent`` filed each one as equivalent. That answer is not
  wrong, it is *incomplete* — it is the conformance axis, and the run never
  asked the divergence question at all;
* ``equivalent_rate: 1.0`` and ``min_equivalence_rate`` reporting
  ``passed: true, conclusive: true`` over a suite where nine of ten examples
  called an entirely different tool.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from evalshift_cli.analysis.policy import BudgetResult, MigrationDecision
from evalshift_cli.analysis.statistics import UNMEASURED_NOTE_PREFIX, ComparisonResult
from evalshift_cli.cli.commands.analyze import run_analyze
from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME, run_evaluate
from evalshift_cli.cli.commands.report import run_report
from evalshift_cli.evaluators import semantic as semantic_module
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import TOOL_GROUND_TRUTH_MISS
from evalshift_cli.models.client import ModelClient
from evalshift_cli.runner.checkpoint import append_call, read_state, write_state
from evalshift_cli.runner.models import Call, RunModels, RunState
from tests.scoring_fixtures import (
    DIVERGENT_EXAMPLES,
    JUDGE_NAME,
    PROMPT_ID,
    RECORDED_TOOL_CALLS,
    RUN_ID,
    SEMANTIC_NAME,
    SOURCE_MODEL,
    SUITE_TAGS,
    TARGET_MODEL,
    TOOL_ARGUMENTS_NAME,
    TOOL_SELECTION_NAME,
    load_pairs,
)

pytestmark = pytest.mark.integration

# The project's real evaluator and policy configuration, transcribed from its
# ``evalshift.yaml``. ``cache: false`` keeps the run off SQLite; the models are
# never called either way.
CONFIG = f"""
version: 1
project: l-babaliauskas-2/bla-bla-bla

prompts:
  - id: {PROMPT_ID}
    detection: manual
    content: "{{input}}"
    variables: [input]

defaults:
  source_model: {SOURCE_MODEL}
  target_model: {TARGET_MODEL}
  cache: false

evaluators:
  semantic:
    embedding_model: gemini/gemini-embedding-001
    min_similarity: 0.9
    blocking: true
  llm_judge:
    - criterion_name: equivalence
      criterion_prompt: Which output is more complete and correct?
      judge_model: gpt-5.6-terra
      blocking: true
  tool_selection:
    - name: {TOOL_SELECTION_NAME}
      conformance: expected
      divergence: set
  tool_arguments:
    - name: {TOOL_ARGUMENTS_NAME}
      against: expected

migration_policy:
  max_overall_regression_rate: 0.3
  max_critical_regressions: 0
  min_equivalence_rate: 0.7
  max_tool_argument_drift: 0.2
  max_cost_increase: 2
  max_latency_increase: 2
"""

#: Every evaluator on this run except ``tool_selection`` measured nothing on
#: every pair: the outputs are empty and the suite carries no ``expected_tools``.
NON_MEASURING = (TOOL_ARGUMENTS_NAME, SEMANTIC_NAME, JUDGE_NAME)


def _scaffold(tmp_path: Path) -> tuple[Path, Path]:
    """Write the project, the promoted suite, and the frozen run directory."""
    config_path = tmp_path / "evalshift.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")

    pairs = load_pairs()

    # The promoted suite. ``capture sync`` writes ``expected_no_tools: true``
    # for every turn whose recording made no tool call, and no
    # ``expected_tools`` at all — which is the whole setup for this defect.
    # ``inputs`` is the rendered prompt on the real suite; it is irrelevant to
    # scoring and deliberately not carried into the fixture.
    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": pair.example_id,
                    "inputs": {"input": ""},
                    "tags": list(SUITE_TAGS),
                    "expected_no_tools": True,
                    # This run predates per-call toolset capture; tools=[] is
                    # the closest honest filler for a frozen historical
                    # fixture (see tests/scoring_fixtures.py PairFixture).
                    "tools": [],
                },
            )
            + "\n"
            for pair in pairs
        ),
        encoding="utf-8",
    )

    runs_base = tmp_path / ".evalshift" / "runs"
    run_dir = runs_base / RUN_ID
    write_state(
        run_dir,
        RunState(
            run_id=RUN_ID,
            status="completed",
            config_hash="frozen",
            started_at=datetime(2026, 8, 20, tzinfo=UTC),
            models=RunModels(source=SOURCE_MODEL, target=TARGET_MODEL),
            prompt_ids=[PROMPT_ID],
            suite_path=str(suite_path),
            total_evaluations=len(pairs) * 2,
            completed_evaluations=len(pairs) * 2,
        ),
    )
    for pair in pairs:
        for role, model_id, text, trace in (
            ("source", SOURCE_MODEL, pair.source_text, pair.source_trace),
            ("target", TARGET_MODEL, pair.target_text, pair.target_trace),
        ):
            append_call(
                run_dir,
                Call(
                    run_id=RUN_ID,
                    prompt_id=PROMPT_ID,
                    example_id=pair.example_id,
                    model_id=model_id,
                    role=role,  # type: ignore[arg-type]
                    text=text,
                    trace=trace,
                    finish_reason="tool_calls",
                ),
            )
    return config_path, runs_base


@pytest.fixture
def scored_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Run ``evaluate`` over the frozen run. No provider may be touched."""
    # Both model-backed evaluators are blocking; without their keys evaluate
    # refuses before scoring. The mocks below still forbid any real call.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def no_embeddings(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no embedding provider may be called on a tool-only run")

    monkeypatch.setattr(semantic_module.litellm, "aembedding", no_embeddings)
    monkeypatch.setattr(
        ModelClient,
        "complete",
        AsyncMock(side_effect=AssertionError("no judge may be called on a tool-only run")),
    )

    config_path, runs_base = _scaffold(tmp_path)
    run_evaluate(
        run_id=RUN_ID,
        config_path=config_path,
        runs_base=runs_base,
        console=Console(quiet=True),
        quiet=True,
    )
    return config_path, runs_base


def _records(runs_base: Path) -> list[EvalRecord]:
    scores_path = runs_base / RUN_ID / SCORES_FILENAME
    return [
        EvalRecord.model_validate_json(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decision(scored_run: tuple[Path, Path]) -> MigrationDecision:
    config_path, runs_base = scored_run
    result = run_analyze(run_id=RUN_ID, config_path=config_path, runs_base=runs_base)
    assert result.migration_decision is not None
    return result.migration_decision


def _budget(decision: MigrationDecision, name: str) -> BudgetResult:
    return next(b for b in decision.budget_results if b.name == name and b.scope == "overall")


class TestNothingMeasuredWritesNoRow:
    def test_a_skipped_evaluator_writes_no_row(self, scored_run: tuple[Path, Path]) -> None:
        """Thirty of the run's forty rows measured nothing and scored maximum.

        Fails today with all thirty present.
        # S1: ``score``/``score_pair`` return ``None`` and ``evaluate.py``
        # writes no record.
        """
        _, runs_base = scored_run
        fabricated = [
            r.evaluator_name for r in _records(runs_base) if r.evaluator_name in NON_MEASURING
        ]
        assert fabricated == []

    def test_the_run_records_what_it_could_not_measure(self, scored_run: tuple[Path, Path]) -> None:
        """Dropping the rows must not drop the count — 30 of 40 is the headline.

        Coverage is booked per *axis*, not per evaluator: ``tool_selection``
        scores two, and one tally for both could report ``recorded`` above
        ``attempted`` while an axis that measured nothing hid behind one
        that measured everything.
        """
        _, runs_base = scored_run
        coverage = {
            (c.evaluator_name, c.kind): c for c in read_state(runs_base / RUN_ID).evaluator_coverage
        }
        for name in NON_MEASURING:
            entry = next(c for (n, _), c in coverage.items() if n == name)
            assert (entry.attempted, entry.recorded) == (10, 0), name
        for kind in ("tool_selection.conformance", "tool_selection.divergence"):
            entry = coverage[(TOOL_SELECTION_NAME, kind)]
            assert (entry.attempted, entry.recorded) == (10, 10), kind
        assert sum(len(c.unmeasured) for c in coverage.values()) == 30


class TestSilenceIsNotAPass:
    """The ``insufficient`` path is now the *only* guard on a blacked-out run.

    While the fabricated rows existed, an evaluator that measured nothing
    still reached the analysis layer — as ten padded rows that
    ``_one_comparison`` filtered down to ``n=0``. With the rows gone, that
    comparison only exists if the run's :class:`EvaluatorCoverage` puts it
    there, and nothing else stands between a no-measurement run and a
    confident pass.
    """

    def _comparisons(self, scored_run: tuple[Path, Path]) -> dict[str, ComparisonResult]:
        config_path, runs_base = scored_run
        result = run_analyze(run_id=RUN_ID, config_path=config_path, runs_base=runs_base)
        return {c.evaluator_name: c for c in result.comparisons if c.slice_name == "all"}

    def test_an_evaluator_that_measured_nothing_is_still_reported(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        comparisons = self._comparisons(scored_run)
        for name in NON_MEASURING:
            assert name in comparisons, f"{name} vanished from the analysis"
            assert comparisons[name].severity == "insufficient", name
            assert comparisons[name].n == 0, name
            assert any(
                note.startswith(UNMEASURED_NOTE_PREFIX) for note in comparisons[name].notes
            ), name

    def test_the_not_applicable_count_survives_the_deleted_rows(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        notes = self._comparisons(scored_run)[SEMANTIC_NAME].notes
        assert any("10 of 10 rows not applicable" in note for note in notes), notes

    def test_the_verdict_refuses_to_pass_on_silence(self, scored_run: tuple[Path, Path]) -> None:
        assert _decision(scored_run).verdict != "pass"


class TestDivergenceIsARegression:
    def test_nine_of_ten_examples_carry_a_negative_delta(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        """Fails today with an empty set: every row is ``0.0 / 0.0``.

        # S2: the divergence axis grades target against a source baseline of
        # 1.0, so each of the nine lands a -1.0 delta.
        """
        _, runs_base = scored_run
        regressed = {
            r.example_id.removeprefix("cap_")[:10] for r in _records(runs_base) if r.delta < 0
        }
        assert regressed == DIVERGENT_EXAMPLES

    def test_equivalent_rate_is_not_one(self, scored_run: tuple[Path, Path]) -> None:
        """The number that shipped. Fails today at exactly ``1.0``.

        # S3: both-sides-zero conformance rows leave the equivalence
        # denominator, and the divergence axis contributes nine regressions.
        """
        assert _decision(scored_run).overall.equivalent_rate != 1.0

    def test_min_equivalence_rate_does_not_pass_conclusively(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        """The budget that shipped ``observed: 1.0, passed: true, conclusive: true``.

        # S3: ``min_equivalence_rate`` shares the regression rate's
        # denominator, which both new axes and dropped rows perturb.
        """
        budget = _budget(_decision(scored_run), "min_equivalence_rate")
        assert not (budget.passed and budget.conclusive)


class TestTheFixtureNumbersArePinned:
    """The rates and budgets this run must report, to three decimal places.

    Every one of these moved in S3 and every one of them is a number a later
    phase could drift without any test noticing. The run scores twenty rows:
    ten ``tool_selection.conformance`` at ``0.0 / 0.0`` — both models called a
    tool where the recording called none — and ten
    ``tool_selection.divergence``, nine of which put the target somewhere the
    source never went.

    The ten conformance rows are a shared ground-truth miss, so they carry no
    evidence about the migration and leave the rates entirely. What is left is
    the divergence axis: **nine regressions in ten**.
    """

    def test_the_run_scores_ten_rows_on_each_axis(self, scored_run: tuple[Path, Path]) -> None:
        _, runs_base = scored_run
        kinds = Counter(r.kind for r in _records(runs_base))
        assert kinds == {"tool_selection.conformance": 10, "tool_selection.divergence": 10}

    def test_equivalent_rate_is_one_in_ten(self, scored_run: tuple[Path, Path]) -> None:
        """``1.0`` when it shipped, ``0.55`` with both axes counted, ``0.1`` now."""
        overall = _decision(scored_run).overall
        assert overall.n_records == 10
        assert overall.equivalent_rate == pytest.approx(0.1)
        assert overall.regression_rate == pytest.approx(0.9)
        assert overall.improved_rate == 0.0

    def test_min_equivalence_rate_fails_conclusively_over_ten_rows(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        """The exact number that lied. It shipped ``1.0, passed, conclusive``."""
        budget = _budget(_decision(scored_run), "min_equivalence_rate")
        assert budget.observed == pytest.approx(0.1)
        assert budget.allowed == pytest.approx(0.7)
        assert budget.denominator == 10
        assert budget.passed is False
        assert budget.conclusive is True

    def test_the_divergence_budget_fails_on_nine_of_ten(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        budget = _budget(_decision(scored_run), "max_tool_divergence")
        assert budget.observed == pytest.approx(0.9)
        assert budget.denominator == 10
        assert budget.passed is False
        assert budget.conclusive is True

    def test_the_regression_rate_budget_fails_conclusively_too(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        budget = _budget(_decision(scored_run), "max_overall_regression_rate")
        assert budget.observed == pytest.approx(0.9)
        assert budget.denominator == 10
        assert budget.passed is False
        assert budget.conclusive is True

    def test_the_verdict_is_fail(self, scored_run: tuple[Path, Path]) -> None:
        assert _decision(scored_run).verdict == "fail"

    def test_the_ten_excluded_rows_are_surfaced_not_silently_dropped(
        self, scored_run: tuple[Path, Path]
    ) -> None:
        """Leaving the rates must not mean leaving the report.

        S5 turns this same 10/10 into a broken-harness error; until then the
        prose channel and the failure-category count are what carry it.
        """
        decision = _decision(scored_run)
        counts = {c.category: c.count for c in decision.failure_categories}
        assert counts[TOOL_GROUND_TRUTH_MISS] == 10
        assert any(
            "10 tool-selection conformance comparisons are excluded" in note
            for note in decision.recommendations
        ), decision.recommendations


class TestTheReportShowsWhatDiverged:
    """S4. ``grep get_recent_files report.html`` returned nothing on this run.

    Nine of ten examples called an entirely different tool and the report
    named not one of them: with every delta at zero there were no top
    regressions to render, and nothing else in the document has ever carried
    a tool name. The plan's own table — example, source called, target called
    — is the whole finding, and it was invisible.
    """

    @pytest.fixture
    def report_html(self, scored_run: tuple[Path, Path]) -> str:
        config_path, runs_base = scored_run
        run_analyze(run_id=RUN_ID, config_path=config_path, runs_base=runs_base)
        result = run_report(
            run_id=RUN_ID,
            config_path=config_path,
            runs_base=runs_base,
            insights=False,
        )
        return result.html_path.read_text(encoding="utf-8")

    def test_every_example_names_the_tools_each_side_called(self, report_html: str) -> None:
        """All ten pairs, not the five worst — four regressions tie at −1.0."""
        for short_id, (source_tool, target_tool) in RECORDED_TOOL_CALLS.items():
            assert source_tool in report_html, f"{short_id}: source tool missing"
            assert target_tool in report_html, f"{short_id}: target tool missing"

    def test_the_two_axes_are_told_apart(self, report_html: str) -> None:
        """One evaluator name, two measurements, two very different answers."""
        assert "tool_selection.divergence" in report_html
        assert "tool_selection.conformance" in report_html

    def test_the_conformance_axis_is_not_headlined_as_equivalent(self, report_html: str) -> None:
        """Ten rows of ``0.0 / 0.0`` rendered as "✓ Equivalent" beside a ✗."""
        assert "No meaningful difference between models" not in report_html
        assert "Ground truth missed by both" in report_html

    def test_the_excluded_rows_are_explained_in_the_document(self, report_html: str) -> None:
        assert "10 tool-selection conformance comparisons are excluded" in report_html


class TestTheBrokenHarnessIsAnnounced:
    """S5. The source model failed ground truth recorded from itself, 10/10.

    Every rate this run reports is a measurement of the eval harness. The
    conformance exclusion already keeps those rows out of the verdict, but a
    verdict still gets printed — so the run has to say, in the loudest voice
    it has, that the number beside it does not describe the target model.
    """

    @pytest.fixture
    def evaluated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str]:
        # Both model-backed evaluators are blocking; without their keys
        # evaluate refuses before scoring.
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        async def no_embeddings(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("no embedding provider may be called on a tool-only run")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", no_embeddings)
        monkeypatch.setattr(
            ModelClient,
            "complete",
            AsyncMock(side_effect=AssertionError("no judge may be called on a tool-only run")),
        )
        config_path, runs_base = _scaffold(tmp_path)
        console = Console(record=True, width=200)
        result = run_evaluate(
            run_id=RUN_ID,
            config_path=config_path,
            runs_base=runs_base,
            console=console,
        )
        return result, console.export_text()

    def test_evaluate_names_the_source_conformance_rate(self, evaluated: tuple[Any, str]) -> None:
        """The plan's done-when: the warning names the rate, at doctor volume."""
        _, printed = evaluated
        assert "10 of 10" in printed
        assert "100%" in printed
        assert "broken eval harness" in printed

    def test_the_warning_names_the_likely_causes(self, evaluated: tuple[Any, str]) -> None:
        """One wording of this fact, shared with the analyze-time recommendation."""
        _, printed = evaluated
        assert "different toolset, prompt or agent" in printed

    def test_the_check_is_carried_on_the_result_not_only_printed(
        self, evaluated: tuple[Any, str]
    ) -> None:
        """``evalshift compare`` scores quietly inside a Live grid and prints it itself."""
        result, _ = evaluated
        assert result.harness_check is not None
        assert result.harness_check.status == "fail"
