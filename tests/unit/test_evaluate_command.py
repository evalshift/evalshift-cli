"""Tests for the ``evalshift evaluate <run-id>`` command."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from rich.console import Console
from typer.testing import CliRunner

from evalshift_cli.analysis.slicing import build_slices, build_unmeasured
from evalshift_cli.analysis.statistics import analyze
from evalshift_cli.captures.toolset import fingerprint_tools
from evalshift_cli.cli.commands.evaluate import (
    SCORES_FILENAME,
    _build_evaluators,
    _coverage_for,
    _is_agent_trace_evaluator,
    _is_tool_evaluator,
    _pair_calls,
    _PairedCalls,
    _reduce_sample_cells,
    _score_agent_traces,
    _score_all,
    _score_one,
    _ScoredCell,
)
from evalshift_cli.cli.main import app
from evalshift_cli.config.models import (
    EvaluatorsConfig,
    ToolSelectionEvaluatorConfig,
    TraceInvariantsEvaluatorConfig,
)
from evalshift_cli.evaluators.base import EvalRecord, PairedScore
from evalshift_cli.evaluators.failures import INVARIANT_VIOLATION
from evalshift_cli.evaluators.llm_judge import PairwiseJudgeEvaluator
from evalshift_cli.evaluators.semantic import CosineSimilarityEvaluator
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.evaluators.tool_selection import ToolSelectionEvaluator
from evalshift_cli.evaluators.trace_invariants import (
    ImportedTraceInvariantsEvaluator,
    TraceInvariantsEvaluator,
)
from evalshift_cli.models.client import ModelClient
from evalshift_cli.runner.checkpoint import append_call, read_state, write_state
from evalshift_cli.runner.models import Call, EvaluatorCoverage, RunModels, RunState
from evalshift_cli.suite.models import ChatMessage, Suite, SuiteExample
from evalshift_cli.traces.loader import TRACES_FILENAME
from tests.unit.suite_examples import suite_example

runner = CliRunner()


class _SpyEvaluator:
    """Records the ``input_vars``/``history`` it was called with."""

    def __init__(self) -> None:
        self.name = "spy.evaluator"
        self.received_input_vars: dict[str, Any] | None = None
        self.received_history: list[dict[str, str]] | None = None

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore:
        self.received_input_vars = input_vars
        self.received_history = history
        return PairedScore(source_score=1.0, target_score=1.0)


def _write_config(
    tmp_path: Path, with_judge: bool = False, judge_model: str = "gemini-2.5-flash"
) -> Path:
    judge_block = ""
    if with_judge:
        # Indented to sit alongside `structural:` under `evaluators:`.
        judge_block = f"""
          llm_judge:
            - criterion_name: brevity
              criterion_prompt: which is more concise?
              judge_model: {judge_model}"""
    cfg_yaml = f"""
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hi {{name}}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
        evaluators:
          structural:
            - type: length
              min_chars: 1
              max_chars: 100
            - type: regex
              pattern: ".+"
{judge_block}
    """
    path = tmp_path / "evalshift.yaml"
    path.write_text(cfg_yaml, encoding="utf-8")
    return path


def _write_agent_trace_config(tmp_path: Path) -> Path:
    cfg_yaml = """
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hi {name}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
        evaluators:
          agent_trace:
            - name: trace_safety
              dangerous_tools: [issue_refund]
    """
    path = tmp_path / "evalshift.yaml"
    path.write_text(cfg_yaml, encoding="utf-8")
    return path


def _scaffold_run(
    tmp_path: Path,
    completed: bool = True,
    suite_name: str | None = None,
) -> str:
    """Create a minimal completed run dir and return its run_id."""
    runs_base = tmp_path / ".evalshift" / "runs"
    run_id = "r_20260601_aaaaaa"
    run_dir = runs_base / run_id
    state = RunState(
        run_id=run_id,
        status="completed" if completed else "in_progress",
        config_hash="x",
        started_at=datetime(2026, 6, 1, tzinfo=UTC),
        models=RunModels(
            source="gemini/gemini-2.5-flash",
            target="gemini/gemini-2.5-pro",
        ),
        prompt_ids=["greet"],
        suite_path="./golden.jsonl",
        suite_name=suite_name,
        total_evaluations=4,
        completed_evaluations=4,
    )
    write_state(run_dir, state)
    for ex_id in ("ex1", "ex2"):
        for role, model_id, text in (
            ("source", "gemini/gemini-2.5-flash", "Hi Alex (short)"),
            ("target", "gemini/gemini-2.5-pro", "Hi Alex! How are you?"),
        ):
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id=model_id,
                    role=role,  # type: ignore[arg-type]
                    text=text,
                ),
            )
    return run_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestEvaluateHappy:
    def test_writes_scores_jsonl(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _write_config(tmp_path)
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout

        scores_path = tmp_path / ".evalshift" / "runs" / run_id / SCORES_FILENAME
        rows = [
            json.loads(line)
            for line in scores_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # 2 examples x 2 evaluators (length + regex) = 4 rows.
        assert len(rows) == 4
        for row in rows:
            assert row["run_id"] == run_id
            assert row["evaluator_name"].startswith("structural.")
            assert "delta" in row

    def test_blocking_flag_stamped_from_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg_yaml = """
            version: 1
            prompts:
              - id: greet
                detection: manual
                content: "Hi {name}"
                variables: [name]
            defaults:
              source_model: gemini-2.5-flash
              target_model: gemini-2.5-pro
            evaluators:
              structural:
                - type: length
                  min_chars: 1
                - type: regex
                  pattern: ".+"
                  blocking: false
        """
        (tmp_path / "evalshift.yaml").write_text(cfg_yaml, encoding="utf-8")
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout

        scores_path = tmp_path / ".evalshift" / "runs" / run_id / SCORES_FILENAME
        rows = [
            json.loads(line)
            for line in scores_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_name = {row["evaluator_name"]: row for row in rows}
        assert by_name["structural.length"]["blocking"] is True
        assert by_name["structural.regex"]["blocking"] is False

    def test_old_scores_row_without_blocking_loads_as_blocking(self) -> None:
        from evalshift_cli.evaluators.base import EvalRecord

        row = EvalRecord.model_validate(
            {
                "run_id": "r",
                "prompt_id": "p",
                "example_id": "e",
                "evaluator_name": "structural.length",
                "source_score": 1.0,
                "target_score": 1.0,
                "delta": 0.0,
            },
        )
        assert row.blocking is True

    def test_pairs_only_complete_pairs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_config(tmp_path)
        run_id = _scaffold_run(tmp_path)
        # Add an orphan source-only call — should be ignored.
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_orphan",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text="orphan",
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout
        scores_path = run_dir / SCORES_FILENAME
        rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
        # Still 4 (2 paired examples x 2 evaluators); the orphan's not paired.
        assert len(rows) == 4

    def test_upstream_failed_calls_recorded_with_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_config(tmp_path)
        run_id = "r_20260601_bbbbbb"
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(
                    source="gemini/gemini-2.5-flash",
                    target="gemini/gemini-2.5-pro",
                ),
                prompt_ids=["greet"],
                suite_path="./golden.jsonl",
                total_evaluations=2,
                completed_evaluations=2,
            ),
        )
        # Source failed; target succeeded.
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                error="auth failed",
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="Hi",
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout
        rows = [
            json.loads(line)
            for line in (run_dir / SCORES_FILENAME).read_text().splitlines()
            if line.strip()
        ]
        assert all(r["error"] for r in rows)

    def test_truncated_call_recorded_with_error_and_excluded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A target cut off at the token cap must be error-marked (so the
        # analysis layer excludes it) rather than scored as a regression.
        _write_config(tmp_path)
        run_id = "r_20260601_dddddd"
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(
                    source="gemini/gemini-2.5-flash",
                    target="gemini/gemini-2.5-pro",
                ),
                prompt_ids=["greet"],
                suite_path="./golden.jsonl",
                total_evaluations=2,
                completed_evaluations=2,
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text="Hello there",
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="Hel",  # cut off
                finish_reason="length",
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout
        rows = [
            json.loads(line)
            for line in (run_dir / SCORES_FILENAME).read_text().splitlines()
            if line.strip()
        ]
        assert rows
        assert all(r["error"] == "output truncated (token cap)" for r in rows)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestEvaluateErrors:
    def test_unknown_run_id_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_config(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", "r_20260601_nonexis"])
        assert result.exit_code == 1

    def test_no_evaluators_configured_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Config without any evaluator entries.
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: greet, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 1
        assert "no evaluators" in result.stdout.lower()

    def test_no_pairs_in_raw_jsonl(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _write_config(tmp_path)
        # Create a run with state but no pairs (only source calls).
        run_id = "r_20260601_cccccc"
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(source="a", target="b"),
                prompt_ids=["greet"],
                suite_path="x.jsonl",
                total_evaluations=2,
                completed_evaluations=2,
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                model_id="a",
                role="source",
                text="hi",
            ),
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 1
        assert "no (source, target) pairs" in result.stdout

    def test_agent_trace_evaluator_requires_imported_traces(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _write_agent_trace_config(tmp_path)
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["evaluate", run_id])

        assert result.exit_code == 1
        assert "agent_trace evaluators require imported traces" in result.stdout


# ---------------------------------------------------------------------------
# _score_one: input_vars / history plumbing (regression for the input_vars={}
# bug, and multi-turn history threading).
# ---------------------------------------------------------------------------


class TestScoreOneInputVarsAndHistory:
    def _pair(self, example_id: str = "ex1") -> _PairedCalls:
        return _PairedCalls(
            prompt_id="greet",
            example_id=example_id,
            source=Call(
                run_id="r1",
                prompt_id="greet",
                example_id=example_id,
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text="Hi Alex (short)",
            ),
            target=Call(
                run_id="r1",
                prompt_id="greet",
                example_id=example_id,
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="Hi Alex! How are you?",
            ),
        )

    async def test_passes_example_inputs_as_input_vars(self) -> None:
        example = suite_example(id="ex1", inputs={"name": "Alex"})
        spy = _SpyEvaluator()
        await _score_one(spy, self._pair(), "r1", {"ex1": example})  # type: ignore[arg-type]
        assert spy.received_input_vars == {"name": "Alex"}

    async def test_missing_example_falls_back_to_empty_input_vars(self) -> None:
        # No matching example in examples_by_id — must not raise, and must
        # not silently pass stale data either.
        spy = _SpyEvaluator()
        await _score_one(spy, self._pair(), "r1", {})  # type: ignore[arg-type]
        assert spy.received_input_vars == {}

    async def test_passes_history_as_plain_dicts_for_multiturn_example(self) -> None:
        example = suite_example(
            id="ex1",
            inputs={"name": "Alex"},
            history=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="create meeting"),
                ChatMessage(role="assistant", content="What time?"),
            ],
        )
        spy = _SpyEvaluator()
        await _score_one(spy, self._pair(), "r1", {"ex1": example})  # type: ignore[arg-type]
        assert spy.received_history == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "create meeting"},
            {"role": "assistant", "content": "What time?"},
        ]

    async def test_history_none_for_single_turn_example(self) -> None:
        example = suite_example(id="ex1", inputs={"name": "Alex"})
        spy = _SpyEvaluator()
        await _score_one(spy, self._pair(), "r1", {"ex1": example})  # type: ignore[arg-type]
        assert spy.received_history is None


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


@dataclass
class _Inflight:
    """Tracks how many evaluator calls overlap."""

    current: int = 0
    peak: int = 0


class _SlowEvaluator:
    """Sleeps while scoring so overlap is observable."""

    def __init__(self, tracker: _Inflight, *, name: str = "slow.evaluator") -> None:
        self.name = name
        self._tracker = tracker

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore:
        self._tracker.current += 1
        self._tracker.peak = max(self._tracker.peak, self._tracker.current)
        # Later examples finish first, so a correct implementation has to
        # restore ordering rather than rely on completion order.
        await asyncio.sleep(0.02 / (1 + int(example_id.removeprefix("ex"))))
        self._tracker.current -= 1
        return PairedScore(source_score=0.0, target_score=1.0)


class _SometimesUnmeasuredEvaluator:
    name = "custom.semantic"
    kind = "semantic"

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore | None:
        if example_id == "ex1":
            return PairedScore(source_score=1.0, target_score=0.8)
        return None


def _pair_for(example_id: str) -> _PairedCalls:
    return _PairedCalls(
        prompt_id="greet",
        example_id=example_id,
        source=Call(
            run_id="r1",
            prompt_id="greet",
            example_id=example_id,
            model_id="m/source",
            role="source",
            text="src",
        ),
        target=Call(
            run_id="r1",
            prompt_id="greet",
            example_id=example_id,
            model_id="m/target",
            role="target",
            text="tgt",
        ),
    )


class TestScoreAllConcurrency:
    async def test_scores_pairs_concurrently(self) -> None:
        tracker = _Inflight()
        pairs = [_pair_for(f"ex{i}") for i in range(8)]
        await _score_all(
            Console(),
            [_SlowEvaluator(tracker)],  # type: ignore[list-item]
            pairs,
            "r1",
            {},
            concurrency=4,
            quiet=True,
        )
        assert tracker.peak > 1, "evaluator calls ran strictly one at a time"

    async def test_respects_the_concurrency_ceiling(self) -> None:
        tracker = _Inflight()
        pairs = [_pair_for(f"ex{i}") for i in range(8)]
        await _score_all(
            Console(),
            [_SlowEvaluator(tracker)],  # type: ignore[list-item]
            pairs,
            "r1",
            {},
            concurrency=3,
            quiet=True,
        )
        assert tracker.peak <= 3

    async def test_record_order_is_pair_major_evaluator_minor(self) -> None:
        tracker = _Inflight()
        pairs = [_pair_for(f"ex{i}") for i in range(5)]
        evaluators = [
            _SlowEvaluator(tracker, name="slow.a"),
            _SlowEvaluator(tracker, name="slow.b"),
        ]
        records = await _score_all(
            Console(),
            evaluators,  # type: ignore[arg-type]
            pairs,
            "r1",
            {},
            concurrency=8,
            quiet=True,
        )
        assert [(r.example_id, r.evaluator_name) for r in records] == [
            (f"ex{i}", name) for i in range(5) for name in ("slow.a", "slow.b")
        ]

    async def test_progress_path_is_also_concurrent(self) -> None:
        # The non-quiet branch renders a progress bar; it must not fall
        # back to serial scoring.
        tracker = _Inflight()
        pairs = [_pair_for(f"ex{i}") for i in range(6)]
        await _score_all(
            Console(quiet=True),
            [_SlowEvaluator(tracker)],  # type: ignore[list-item]
            pairs,
            "r1",
            {},
            concurrency=4,
            quiet=False,
        )
        assert tracker.peak > 1


class TestTextEvaluatorCoverage:
    async def test_measured_and_unmeasured_text_rows_share_one_axis(self) -> None:
        cells = await _score_all(
            Console(),
            [_SometimesUnmeasuredEvaluator()],  # type: ignore[list-item]
            [_pair_for("ex1"), _pair_for("ex2")],
            "r1",
            {},
            concurrency=2,
            quiet=True,
        )
        records = [cell.record for cell in cells if cell.record is not None]
        coverage = _coverage_for(cells)
        suite = Suite(examples=[suite_example(id="ex1"), suite_example(id="ex2")])

        assert [record.kind for record in records] == ["semantic"]
        assert [(entry.kind, entry.attempted, entry.recorded) for entry in coverage] == [
            ("semantic", 2, 1),
        ]

        comparisons = analyze(
            sliced_by_slice=build_slices(records=records, suite=suite, coverage=coverage),
            unmeasured_by_slice=build_unmeasured(coverage=coverage, suite=suite),
        )

        assert len(comparisons) == 1
        assert comparisons[0].n == 1
        assert any("1 of 2 rows not applicable" in note for note in comparisons[0].notes)

    async def test_coverage_carries_the_evaluators_blocking_flag(self) -> None:
        """An axis with zero rows leaves nothing in ``scores.jsonl`` carrying
        the config ``blocking`` flag — coverage is where it survives."""

        class _AdvisoryEvaluator(_SometimesUnmeasuredEvaluator):
            blocking = False

        cells = await _score_all(
            Console(),
            [_AdvisoryEvaluator()],  # type: ignore[list-item]
            [_pair_for("ex1"), _pair_for("ex2")],
            "r1",
            {},
            concurrency=2,
            quiet=True,
        )
        (entry,) = _coverage_for(cells)
        assert entry.blocking is False

    async def test_coverage_defaults_to_gating_without_a_flag(self) -> None:
        cells = await _score_all(
            Console(),
            [_SometimesUnmeasuredEvaluator()],  # type: ignore[list-item]
            [_pair_for("ex1")],
            "r1",
            {},
            concurrency=2,
            quiet=True,
        )
        (entry,) = _coverage_for(cells)
        assert entry.blocking is True


class TestToolArgumentsEmbeddings:
    """The ``semantic`` argument strategy needs an embeddings function.

    ``ToolArgumentsEvaluator`` has always accepted ``embeddings_fn`` and
    ``_score_semantic`` falls back to exact string equality without one — but
    no caller ever supplied it, so ``strategies: {query: semantic}`` silently
    did nothing. That is exactly what a user reaches for after watching
    ``search_web(query=…)`` fail on one extra word.
    """

    @staticmethod
    def _config(tmp_path: Path, *, semantic: bool) -> Any:
        from evalshift_cli.config.loader import load_config

        semantic_block = "  semantic:\n    embedding_model: text-embedding-3-small\n"
        cfg_yaml = f"""
version: 1
prompts:
  - id: p
    detection: manual
    content: "{{input}}"
    variables: [input]
defaults:
  source_model: gemini-3.1-flash-lite-preview
evaluators:
{semantic_block if semantic else ""}  tool_arguments:
    - name: args
      strategies:
        query: semantic
"""
        path = tmp_path / "evalshift.yaml"
        path.write_text(cfg_yaml, encoding="utf-8")
        return load_config(path)

    def _tool_arguments_evaluator(self, tmp_path: Path, *, semantic: bool) -> Any:
        cfg = self._config(tmp_path, semantic=semantic)
        evaluators = _build_evaluators(cfg.evaluators, tmp_path, judge_client=ModelClient())
        return next(e for e in evaluators if getattr(e, "name", "") == "args")

    def test_is_built_with_an_embeddings_fn(self, tmp_path: Path) -> None:
        ta = self._tool_arguments_evaluator(tmp_path, semantic=True)
        assert ta._embeddings_fn is not None

    def test_has_no_embeddings_fn_without_a_semantic_evaluator(self, tmp_path: Path) -> None:
        """No configured semantic evaluator means no embedding model to borrow."""
        ta = self._tool_arguments_evaluator(tmp_path, semantic=False)
        assert ta._embeddings_fn is None

    async def test_the_embeddings_fn_scores_a_reworded_argument_above_exact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end through the strategy, with the embedding call stubbed."""
        vectors = {
            "weather in Madrid tomorrow August 11 2026": [1.0, 0.0],
            "weather in Madrid tomorrow Tuesday August 11 2026": [0.98, 0.199],
        }

        async def _fake_embed(self: Any, text: str) -> list[float]:
            return vectors[text]

        monkeypatch.setattr(CosineSimilarityEvaluator, "_embed", _fake_embed)
        ta = self._tool_arguments_evaluator(tmp_path, semantic=True)
        score = await ta._score_field(
            "weather in Madrid tomorrow August 11 2026",
            "weather in Madrid tomorrow Tuesday August 11 2026",
            "semantic",
        )
        assert score > 0.9  # exact equality would have scored 0.0

    def test_the_borrowed_evaluator_shares_the_run_cache(self, tmp_path: Path) -> None:
        """The embedder must be the *same instance* ``_run_scoring`` caches.

        The cache is attached after evaluators are built, so an embeddings
        function closing over a separate instance would pay for every
        embedding twice.
        """
        cfg = self._config(tmp_path, semantic=True)
        evaluators = _build_evaluators(cfg.evaluators, tmp_path, judge_client=ModelClient())
        semantic = next(e for e in evaluators if isinstance(e, CosineSimilarityEvaluator))
        semantic.cache = "sentinel"  # type: ignore[assignment]
        ta = next(e for e in evaluators if getattr(e, "name", "") == "args")
        assert ta._embeddings_fn is not None
        assert ta._embeddings_fn.__closure__ is not None
        borrowed = [c.cell_contents for c in ta._embeddings_fn.__closure__]
        assert any(getattr(b, "cache", None) == "sentinel" for b in borrowed)


class TestToolArgumentsToolsetResolver:
    """``auto``'s schema rung needs the toolset the capture recorded.

    The sidecar is already on disk under ``.evalshift/toolsets/``; without a
    resolver wired in, ``auto`` would grade a reworded timestamp as "similar".
    """

    _TOOL: ClassVar[dict[str, Any]] = {
        "name": "add_event",
        "description": "Create a calendar event.",
        "input_schema": {
            "type": "object",
            "properties": {"start_time": {"type": "string", "format": "date-time"}},
        },
    }

    @classmethod
    def _project(cls, tmp_path: Path) -> tuple[Any, Path, str]:
        """Write a config, a golden suite naming a toolset, and its sidecar."""
        from evalshift_cli.config.loader import load_config

        ref = fingerprint_tools([cls._TOOL])
        toolsets = tmp_path / ".evalshift" / "toolsets"
        toolsets.mkdir(parents=True)
        (toolsets / f"{ref.removeprefix('sha256:')}.json").write_text(
            json.dumps({"tools": [cls._TOOL]}),
            encoding="utf-8",
        )
        suite_path = tmp_path / "golden.jsonl"
        suite_path.write_text(
            json.dumps({"id": "ex1", "inputs": {}, "toolset_ref": ref}) + "\n",
            encoding="utf-8",
        )
        cfg_path = tmp_path / "evalshift.yaml"
        cfg_path.write_text(
            """
version: 1
prompts:
  - id: p
    detection: manual
    content: "{input}"
    variables: [input]
defaults:
  source_model: gemini-3.1-flash-lite-preview
evaluators:
  tool_arguments:
    - name: args
""",
            encoding="utf-8",
        )
        return load_config(cfg_path), suite_path, ref

    def test_the_evaluator_gets_a_resolver_that_reads_the_sidecar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg, suite_path, ref = self._project(tmp_path)
        evaluators = _build_evaluators(
            cfg.evaluators,
            tmp_path,
            judge_client=ModelClient(),
            suite_path=suite_path,
        )
        ta = next(e for e in evaluators if getattr(e, "name", "") == "args")
        assert ta._toolset_resolver is not None
        tools = ta._toolset_resolver(suite_example(id="ex1", toolset_ref=ref))
        assert tools is not None
        assert [t.name for t in tools] == ["add_event"]

    def test_a_missing_sidecar_resolves_to_none_instead_of_raising(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scoring must not die because a toolset sidecar was cleaned up."""
        monkeypatch.chdir(tmp_path)
        cfg, suite_path, _ref = self._project(tmp_path)
        evaluators = _build_evaluators(
            cfg.evaluators,
            tmp_path,
            judge_client=ModelClient(),
            suite_path=suite_path,
        )
        ta = next(e for e in evaluators if getattr(e, "name", "") == "args")
        example = suite_example(id="ex1", toolset_ref="sha256:" + "ab" * 32)
        assert ta._toolset_resolver(example) is None

    async def test_a_reworded_timestamp_scores_zero_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg, suite_path, ref = self._project(tmp_path)
        evaluators = _build_evaluators(
            cfg.evaluators,
            tmp_path,
            judge_client=ModelClient(),
            suite_path=suite_path,
        )
        ta = next(e for e in evaluators if getattr(e, "name", "") == "args")
        record = await ta.score_pair(
            run_id="r",
            prompt_id="p",
            example=suite_example(id="ex1", toolset_ref=ref),
            source_trace=ToolTrace(
                calls=[
                    ToolCall(
                        tool_name="add_event",
                        arguments={"start_time": "2026-01-02T09:00:00Z"},
                        sequence_index=0,
                    ),
                ],
            ),
            target_trace=ToolTrace(
                calls=[
                    ToolCall(
                        tool_name="add_event",
                        arguments={"start_time": "2026-01-02T09:00:00+00:00"},
                        sequence_index=0,
                    ),
                ],
            ),
        )
        assert record is not None
        assert record.target_score == pytest.approx(0.0)


class TestCoverageIsBookedPerAxis:
    """``tool_selection`` scores two axes, so it has two denominators.

    Coverage keyed by evaluator alone would let ``recorded`` exceed
    ``attempted`` — two rows per attempt — and would hide an axis that
    measured nothing behind one that measured everything. The C3 invariant
    ("k of n pairs were not measurable") is a statement about one
    measurement, so the tally has to be per measurement.
    """

    def _pair_with_traces(self, example_id: str, source: str, target: str) -> _PairedCalls:
        pair = _pair_for(example_id)
        return _PairedCalls(
            prompt_id=pair.prompt_id,
            example_id=example_id,
            source=pair.source.model_copy(
                update={"trace": ToolTrace(calls=[ToolCall(tool_name=source, sequence_index=0)])},
            ),
            target=pair.target.model_copy(
                update={"trace": ToolTrace(calls=[ToolCall(tool_name=target, sequence_index=0)])},
            ),
        )

    async def _coverage(
        self,
        example: SuiteExample | None,
        **config: str,
    ) -> dict[tuple[str, str], EvaluatorCoverage]:
        evaluator = ToolSelectionEvaluator(
            ToolSelectionEvaluatorConfig(name="routing", **config),  # type: ignore[arg-type]
        )
        pair = self._pair_with_traces("ex1", "get_projects", "add_note")
        cells = await _score_all(
            Console(),
            [evaluator],  # type: ignore[list-item]
            [pair],
            "r1",
            {"ex1": example} if example is not None else {},
            concurrency=2,
            quiet=True,
        )
        return {(c.evaluator_name, c.kind): c for c in _coverage_for(cells)}

    async def test_one_entry_per_axis_never_two_rows_under_one_tally(self) -> None:
        coverage = await self._coverage(suite_example(id="ex1", expected_no_tools=True))

        assert set(coverage) == {
            ("routing", "tool_selection.conformance"),
            ("routing", "tool_selection.divergence"),
        }
        for entry in coverage.values():
            assert (entry.attempted, entry.recorded) == (1, 1)

    async def test_an_unmeasurable_axis_is_visible_next_to_a_measured_one(self) -> None:
        """No ground truth: conformance measures nothing, divergence still does."""
        coverage = await self._coverage(suite_example(id="ex1"))

        conformance = coverage[("routing", "tool_selection.conformance")]
        divergence = coverage[("routing", "tool_selection.divergence")]
        assert (conformance.attempted, conformance.recorded) == (1, 0)
        assert [p.example_id for p in conformance.unmeasured] == ["ex1"]
        assert (divergence.attempted, divergence.recorded) == (1, 1)
        assert divergence.unmeasured == []

    async def test_an_axis_switched_off_is_never_attempted(self) -> None:
        coverage = await self._coverage(suite_example(id="ex1"), conformance="off")

        assert set(coverage) == {("routing", "tool_selection.divergence")}

    async def test_a_broken_pair_errors_on_every_axis(self) -> None:
        """An error is a broken measurement per axis, not one shared row."""
        pair = self._pair_with_traces("ex1", "get_projects", "add_note")
        broken = _PairedCalls(
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            source=pair.source.model_copy(update={"finish_reason": "length"}),
            target=pair.target,
        )
        records = await _score_one(
            ToolSelectionEvaluator(ToolSelectionEvaluatorConfig(name="routing")),  # type: ignore[arg-type]
            broken,
            "r1",
            {},
        )
        assert [r.kind for r in records] == [
            "tool_selection.conformance",
            "tool_selection.divergence",
        ]
        assert all(r.error == "output truncated (token cap)" for r in records)


# ---------------------------------------------------------------------------
# Judge client sharing + runtime temperature-rejection reporting
# ---------------------------------------------------------------------------


_BadRequestError = type("BadRequestError", (Exception,), {})


def _judge_response(text: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class TestJudgeClientSharingAndReporting:
    def test_build_evaluators_threads_shared_judge_client(self, tmp_path: Path) -> None:
        from evalshift_cli.config.loader import load_config

        cfg = load_config(_write_config(tmp_path, with_judge=True))
        judge_client = ModelClient()
        evaluators = _build_evaluators(cfg.evaluators, tmp_path, judge_client=judge_client)
        judges = [e for e in evaluators if isinstance(e, PairwiseJudgeEvaluator)]
        assert judges, "config should have produced a judge"
        assert all(j._client is judge_client for j in judges)

    def test_judge_temperature_rejections_merge_into_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # End-to-end through the command: the judge model 400s the
        # temperature value, the client adapts, scoring completes, and the
        # judge model joins non_deterministic_models in the state written
        # alongside evaluator_coverage.
        from evalshift_cli.models import client as client_module

        # Redirect the default cache path into tmp so the judge's verdict
        # cache can't serve a hit from (or write into) the user's real
        # ~/.evalshift/cache.db — a hit would skip the dispatch this test
        # exists to exercise.
        monkeypatch.setattr(
            "evalshift_cli.cache.schema.DEFAULT_CACHE_PATH",
            tmp_path / "cache.db",
        )
        _write_config(tmp_path, with_judge=True)
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)

        async def fake_acompletion(**kwargs: Any) -> Any:
            if "temperature" in kwargs:
                raise _BadRequestError("'temperature' does not support 0.0 with this model.")
            return _judge_response('{"winner": "A"}')

        monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(
            client_module.litellm,
            "completion_cost",
            lambda completion_response=None, **_: 0.0,
        )

        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout

        state = read_state(tmp_path / ".evalshift" / "runs" / run_id)
        assert "gemini/gemini-2.5-flash" in state.non_deterministic_models
        # Both updates land in the same write.
        assert state.evaluator_coverage is not None

    def _evaluate_with_accepting_judge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, judge_model: str
    ) -> RunState:
        """Score a run whose judge accepts ``temperature`` without complaint."""
        import litellm

        from evalshift_cli.models import client as client_module

        monkeypatch.setattr(
            "evalshift_cli.cache.schema.DEFAULT_CACHE_PATH",
            tmp_path / "cache.db",
        )
        # Pin LiteLLM's reasoning flag: DeepSeek ids think by default, and the
        # answer must not depend on the installed LiteLLM's model map.
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: True)
        _write_config(tmp_path, with_judge=True, judge_model=judge_model)
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)

        async def fake_acompletion(**kwargs: Any) -> Any:
            return _judge_response('{"winner": "A"}')

        monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(
            client_module.litellm,
            "completion_cost",
            lambda completion_response=None, **_: 0.0,
        )

        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout
        return read_state(tmp_path / ".evalshift" / "runs" / run_id)

    def test_temperature_ignoring_judge_joins_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # DeepSeek thinking mode accepts temperature and ignores it, so the
        # client never sees a rejection; the judge is flagged from the model
        # id instead, under its canonical id and exactly once.
        state = self._evaluate_with_accepting_judge(monkeypatch, tmp_path, "deepseek-v4-pro")
        assert state.non_deterministic_models.count("deepseek/deepseek-v4-pro") == 1
        assert state.evaluator_coverage is not None

    def test_temperature_honouring_judge_stays_off_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        state = self._evaluate_with_accepting_judge(monkeypatch, tmp_path, "gemini-2.5-flash")
        assert "gemini/gemini-2.5-flash" not in state.non_deterministic_models


class TestPerSuiteEvaluators:
    """``suites.<name>.evaluators`` decides what a given run is scored with.

    A heterogeneous project — one tool-calling suite, several tool-free ones —
    cannot be scored honestly by a single top-level block, so the run records
    which suite it was and scoring resolves the set from there.
    """

    _CONFIG = """
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hi {name}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
        evaluators:
          structural:
            - type: length
              min_chars: 1
              max_chars: 100
        suites:
          main_chat:
            path: ./golden.jsonl
            evaluators:
              structural:
                - type: regex
                  pattern: ".+"
          briefing:
            path: ./golden.jsonl
    """

    def _evaluator_names(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        suite_name: str | None,
    ) -> set[str]:
        (tmp_path / "evalshift.yaml").write_text(self._CONFIG, encoding="utf-8")
        run_id = _scaffold_run(tmp_path, suite_name=suite_name)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout
        scores_path = tmp_path / ".evalshift" / "runs" / run_id / SCORES_FILENAME
        return {
            json.loads(line)["evaluator_name"]
            for line in scores_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def test_suite_with_a_block_is_scored_with_its_own_evaluators(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        names = self._evaluator_names(monkeypatch, tmp_path, suite_name="main_chat")
        assert names == {"structural.regex"}

    def test_suite_without_a_block_inherits_the_top_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        names = self._evaluator_names(monkeypatch, tmp_path, suite_name="briefing")
        assert names == {"structural.length"}

    def test_a_raw_suite_path_run_uses_the_top_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        names = self._evaluator_names(monkeypatch, tmp_path, suite_name=None)
        assert names == {"structural.length"}


# ---------------------------------------------------------------------------
# samples_per_example (Task 7.1) — per-sample pairing, one row per example
# ---------------------------------------------------------------------------


def _sample_call(*, example_id: str, role: str, sample_index: int, text: str) -> Call:
    return Call(
        run_id="r1",
        prompt_id="greet",
        example_id=example_id,
        model_id=f"m/{role}",
        role=role,  # type: ignore[arg-type]
        text=text,
        sample_index=sample_index,
    )


class _ScoreByTextEvaluator:
    """Target score is parsed from the target text, so each sample scores differently."""

    name = "custom.by_text"
    kind = "by_text"

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore | None:
        if target_output == "skip":
            return None
        if target_output == "boom":
            raise RuntimeError("judge exploded")
        return PairedScore(
            source_score=1.0,
            target_score=float(target_output),
            explanation=f"target said {target_output}",
            metadata={"raw": target_output},
        )


class TestPairCallsPerSample:
    def test_pairs_source_and_target_of_the_same_sample(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r1"
        for sample in (0, 1):
            append_call(
                run_dir,
                _sample_call(example_id="ex1", role="source", sample_index=sample, text="s"),
            )
            append_call(
                run_dir,
                _sample_call(example_id="ex1", role="target", sample_index=sample, text="t"),
            )
        # A target with no matching-sample source is not a pair.
        append_call(
            run_dir, _sample_call(example_id="ex1", role="target", sample_index=2, text="t")
        )

        pairs = _pair_calls(run_dir)

        assert [(p.example_id, p.sample_index) for p in pairs] == [("ex1", 0), ("ex1", 1)]
        for pair in pairs:
            assert pair.source.sample_index == pair.target.sample_index == pair.sample_index


def _pairs_with_samples(targets: dict[str, list[str]]) -> list[_PairedCalls]:
    pairs: list[_PairedCalls] = []
    for example_id, texts in targets.items():
        for sample, text in enumerate(texts):
            pairs.append(
                _PairedCalls(
                    prompt_id="greet",
                    example_id=example_id,
                    source=_sample_call(
                        example_id=example_id, role="source", sample_index=sample, text="s"
                    ),
                    target=_sample_call(
                        example_id=example_id, role="target", sample_index=sample, text=text
                    ),
                    sample_index=sample,
                )
            )
    return pairs


async def _cells(targets: dict[str, list[str]]) -> list[Any]:
    return await _score_all(
        Console(),
        [_ScoreByTextEvaluator()],  # type: ignore[list-item]
        _pairs_with_samples(targets),
        "r1",
        {},
        concurrency=4,
        quiet=True,
    )


class TestSampleReduction:
    async def test_single_sample_output_is_byte_identical_to_before(self) -> None:
        cells = await _cells({"ex1": ["0.5"]})
        (cell,) = cells
        record = cell.record
        assert record is not None
        assert record.explanation == "target said 0.5"
        assert record.metadata == {"raw": "0.5"}
        assert "samples" not in record.metadata

    async def test_scores_are_means_with_within_pair_variance(self) -> None:
        cells = await _cells({"ex1": ["0.2", "0.4", "0.9"]})
        (cell,) = cells
        record = cell.record
        assert record is not None
        assert record.source_score == pytest.approx(1.0)
        assert record.target_score == pytest.approx(0.5)
        assert record.delta == pytest.approx(-0.5)
        samples = record.metadata["samples"]
        assert samples["n"] == 3
        assert samples["scored"] == 3
        assert samples["source_scores"] == [1.0, 1.0, 1.0]
        assert samples["target_scores"] == pytest.approx([0.2, 0.4, 0.9])
        assert samples["deltas"] == pytest.approx([-0.8, -0.6, -0.1])
        # population variance of the deltas
        assert samples["delta_variance"] == pytest.approx(
            sum((d + 0.5) ** 2 for d in (-0.8, -0.6, -0.1)) / 3
        )
        assert record.explanation == "mean of 3 samples · target said 0.2"
        assert record.metadata["raw"] == "0.2"
        assert record.error is None

    async def test_one_example_row_per_evaluator_regardless_of_samples(self) -> None:
        cells = await _cells({"ex1": ["0.2", "0.4"], "ex2": ["1.0", "1.0"]})
        assert [(c.example_id, c.kind) for c in cells] == [("ex1", "by_text"), ("ex2", "by_text")]
        coverage = _coverage_for(cells)
        assert [(e.kind, e.attempted, e.recorded) for e in coverage] == [("by_text", 2, 2)]

    async def test_a_failed_sample_is_dropped_from_the_mean(self) -> None:
        cells = await _cells({"ex1": ["0.2", "boom", "0.4"]})
        (cell,) = cells
        record = cell.record
        assert record is not None
        assert record.error is None
        assert record.target_score == pytest.approx(0.3)
        samples = record.metadata["samples"]
        assert samples["n"] == 3
        assert samples["scored"] == 2
        assert samples["delta_variance"] == pytest.approx(0.01)
        assert record.explanation == "mean of 2 samples · target said 0.2"

    async def test_error_only_when_every_sample_failed(self) -> None:
        cells = await _cells({"ex1": ["boom", "boom"]})
        (cell,) = cells
        record = cell.record
        assert record is not None
        assert record.error == "evaluator error: judge exploded"
        assert record.source_score == 0.5
        assert record.metadata["samples"]["scored"] == 0

    async def test_unmeasured_samples_do_not_count(self) -> None:
        cells = await _cells({"ex1": ["skip", "0.6"], "ex2": ["skip", "skip"]})
        assert [(c.example_id, c.record is not None) for c in cells] == [
            ("ex1", True),
            ("ex2", False),
        ]
        ex1 = cells[0].record
        assert ex1 is not None
        assert ex1.target_score == pytest.approx(0.6)
        assert ex1.metadata["samples"] == {
            "n": 2,
            "scored": 1,
            "source_scores": [1.0],
            "target_scores": [0.6],
            "deltas": pytest.approx([-0.4]),
            "delta_variance": 0.0,
        }
        coverage = _coverage_for(cells)
        assert [(e.attempted, e.recorded) for e in coverage] == [(2, 1)]


class TestEvaluateWithSamples:
    def test_scores_jsonl_has_one_row_per_example_and_evaluator(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _write_config(tmp_path)
        run_id = _scaffold_run(tmp_path)
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        # A second sample for every (example, role).
        for ex_id in ("ex1", "ex2"):
            for role, model_id, text in (
                ("source", "gemini/gemini-2.5-flash", "Hi Alex (short)"),
                ("target", "gemini/gemini-2.5-pro", "Hi Alex! How are you?"),
            ):
                append_call(
                    run_dir,
                    Call(
                        run_id=run_id,
                        prompt_id="greet",
                        example_id=ex_id,
                        model_id=model_id,
                        role=role,  # type: ignore[arg-type]
                        text=text,
                        sample_index=1,
                    ),
                )
        state = read_state(run_dir)
        write_state(
            run_dir,
            state.model_copy(update={"samples_per_example": 2, "total_evaluations": 8}),
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["evaluate", run_id])
        assert result.exit_code == 0, result.stdout

        rows = [
            json.loads(line)
            for line in (run_dir / SCORES_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # 2 examples x 2 evaluators, not 4 sample pairs x 2.
        assert len(rows) == 4
        for row in rows:
            assert row["metadata"]["samples"]["n"] == 2
            assert row["metadata"]["samples"]["scored"] == 2
        coverage = read_state(run_dir).evaluator_coverage
        assert [(c.attempted, c.recorded) for c in coverage] == [(2, 2), (2, 2)]


# ---------------------------------------------------------------------------
# trace_invariants — build, applies_to scoping, per-sample violations
# ---------------------------------------------------------------------------

_INVARIANTS = {
    "name": "payments",
    "rules": [{"id": "no-v1", "type": "forbidden", "tools": ["refund_v1"]}],
}


def _tool_pair(prompt_id: str, example_id: str, target_tool: str) -> _PairedCalls:
    def call(role: str, tool: str) -> Call:
        return Call(
            run_id="r1",
            prompt_id=prompt_id,
            example_id=example_id,
            model_id=f"m/{role}",
            role=role,  # type: ignore[arg-type]
            trace=ToolTrace(calls=[ToolCall(tool_name=tool, sequence_index=0)]),
        )

    return _PairedCalls(
        prompt_id=prompt_id,
        example_id=example_id,
        source=call("source", "lookup"),
        target=call("target", target_tool),
    )


def _imported_trace_line(prompt_id: str, role: str, tool: str) -> str:
    return json.dumps(
        {
            "run_id": "r1",
            "prompt_id": prompt_id,
            "example_id": "e1",
            "role": role,
            "events": [
                {
                    "type": "tool_call",
                    "sequence_index": 0,
                    "timestamp": "2026-06-09T12:00:00Z",
                    "metadata": {},
                    "name": tool,
                    "arguments": {},
                },
            ],
        },
    )


def _write_imported_traces(run_dir: Path, prompt_ids: list[str]) -> list[_PairedCalls]:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        _imported_trace_line(prompt_id, role, tool)
        for prompt_id in prompt_ids
        for role, tool in (("source", "lookup"), ("target", "refund_v1"))
    ]
    (run_dir / TRACES_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [_tool_pair(prompt_id, "e1", "refund_v1") for prompt_id in prompt_ids]


class TestTraceInvariantsWiring:
    def test_build_routes_on_traces(self, tmp_path: Path) -> None:
        cfg = EvaluatorsConfig(
            trace_invariants=[
                TraceInvariantsEvaluatorConfig.model_validate(_INVARIANTS),
                TraceInvariantsEvaluatorConfig.model_validate(
                    {**_INVARIANTS, "name": "imported", "traces": "imported"},
                ),
            ],
        )
        built = _build_evaluators(cfg, tmp_path, judge_client=ModelClient())
        assert isinstance(built[0], TraceInvariantsEvaluator)
        assert isinstance(built[1], ImportedTraceInvariantsEvaluator)

    def test_each_variant_lands_on_its_own_scoring_path(self, tmp_path: Path) -> None:
        cfg = EvaluatorsConfig(
            trace_invariants=[
                TraceInvariantsEvaluatorConfig.model_validate(_INVARIANTS),
                TraceInvariantsEvaluatorConfig.model_validate(
                    {**_INVARIANTS, "name": "imported", "traces": "imported"},
                ),
            ],
        )
        replayed, imported = _build_evaluators(cfg, tmp_path, judge_client=ModelClient())
        assert _is_tool_evaluator(replayed)
        assert not _is_agent_trace_evaluator(replayed)
        assert _is_agent_trace_evaluator(imported)
        assert not _is_tool_evaluator(imported)

    def test_the_config_blocking_flag_rides_on_the_evaluator(self, tmp_path: Path) -> None:
        cfg = EvaluatorsConfig(
            trace_invariants=[
                TraceInvariantsEvaluatorConfig.model_validate({**_INVARIANTS, "blocking": False}),
            ],
        )
        (built,) = _build_evaluators(cfg, tmp_path, judge_client=ModelClient())
        assert built.blocking is False  # type: ignore[attr-defined]

    async def test_a_prompt_outside_applies_to_is_never_attempted(self) -> None:
        evaluator = TraceInvariantsEvaluator(
            TraceInvariantsEvaluatorConfig.model_validate({**_INVARIANTS, "applies_to": ["pay*"]}),
        )
        cells = await _score_all(
            Console(),
            [evaluator],  # type: ignore[list-item]
            [_tool_pair("payments", "e1", "refund_v1"), _tool_pair("support", "e2", "refund_v1")],
            "r1",
            {},
            concurrency=2,
            quiet=True,
        )
        assert [(c.prompt_id, c.record is not None) for c in cells] == [("payments", True)]
        coverage = _coverage_for(cells)
        assert (coverage[0].attempted, coverage[0].unmeasured) == (1, [])

    async def test_imported_prompt_outside_applies_to_is_skipped(self, tmp_path: Path) -> None:
        evaluator = ImportedTraceInvariantsEvaluator(
            TraceInvariantsEvaluatorConfig.model_validate(
                {**_INVARIANTS, "traces": "imported", "applies_to": ["pay*"]},
            ),
        )
        run_dir = tmp_path / "r1"
        pairs = _write_imported_traces(run_dir, ["support"])

        cells = await _score_agent_traces(
            run_dir=run_dir,
            evaluators=[evaluator],  # type: ignore[list-item]
            pairs=pairs,
            run_id="r1",
        )

        assert cells == []

    async def test_imported_prompt_inside_applies_to_is_scored(self, tmp_path: Path) -> None:
        evaluator = ImportedTraceInvariantsEvaluator(
            TraceInvariantsEvaluatorConfig.model_validate(
                {**_INVARIANTS, "traces": "imported", "applies_to": ["pay*"]},
            ),
        )
        run_dir = tmp_path / "r1"
        pairs = _write_imported_traces(run_dir, ["payments", "support"])

        cells = await _score_agent_traces(
            run_dir=run_dir,
            evaluators=[evaluator],  # type: ignore[list-item]
            pairs=pairs,
            run_id="r1",
        )

        assert [(c.prompt_id, c.kind) for c in cells] == [("payments", "trace_invariants")]
        record = cells[0].record
        assert record is not None
        assert record.target_score < 1.0
        assert record.metadata["failure_categories"] == [INVARIANT_VIOLATION]

    def test_sample_reduction_keeps_every_samples_violations(self) -> None:
        """Review Focus 1: the lead sample was clean, sample 2 broke the rule."""

        def cell(index: int, target_violations: list[dict[str, Any]]) -> _ScoredCell:
            target_score = 0.0 if target_violations else 1.0
            metadata: dict[str, Any] = {
                "rules_checked": ["no-v1"],
                "source_violations": [],
                "target_violations": target_violations,
            }
            if target_violations:
                metadata["failure_categories"] = [INVARIANT_VIOLATION]
            return _ScoredCell(
                prompt_id="p",
                example_id="e1",
                evaluator_name="payments",
                kind="trace_invariants",
                record=EvalRecord(
                    run_id="r1",
                    prompt_id="p",
                    example_id="e1",
                    evaluator_name="payments",
                    kind="trace_invariants",
                    source_score=1.0,
                    target_score=target_score,
                    delta=target_score - 1.0,
                    metadata=metadata,
                ),
                sample_index=index,
            )

        broken = {
            "rule_id": "no-v1",
            "rule_type": "forbidden",
            "tool": "refund_v1",
            "round_index": 0,
            "detail": "called forbidden tool 'refund_v1'",
        }
        (reduced,) = _reduce_sample_cells([cell(0, []), cell(1, []), cell(2, [broken])])
        assert reduced.record is not None
        assert reduced.record.target_score < 1.0
        assert reduced.record.metadata["target_violations"] == [{**broken, "sample": 2}]
        assert INVARIANT_VIOLATION in reduced.record.metadata["failure_categories"]

    async def test_other_families_reduced_metadata_gains_no_violation_keys(self) -> None:
        cells = await _cells({"ex1": ["0.2", "0.4", "0.9"]})
        (cell,) = cells
        assert cell.record is not None
        assert set(cell.record.metadata) == {"raw", "samples"}
