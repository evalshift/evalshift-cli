"""Tests for ``evalshift compare``: the single-shot pipeline command.

Two flavours:

* Pure unit tests for the verdict picker (``_compose_verdict``) and
  the rendering helpers (``_bar``, ``_evaluator_family_summary``).
* One end-to-end smoke test that drives the full pipeline via
  :class:`CliRunner` against a temporary scaffold + a stubbed
  :class:`ModelClient`. We assert the rendered output mentions every
  stage glyph and that ``report.html`` is written under the run dir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.analysis.statistics import ComparisonResult
from evalshift_cli.cli.commands.analyze import AnalyzeResult
from evalshift_cli.cli.commands.analyze import run_analyze as _real_run_analyze
from evalshift_cli.cli.commands.compare import (
    _bar,
    _compose_verdict,
    _evaluator_family_summary,
)
from evalshift_cli.cli.commands.doctor import CheckResult
from evalshift_cli.cli.main import app
from evalshift_cli.config.models import (
    EvalShiftConfig,
    EvaluatorsConfig,
    LLMJudgeConfig,
    PromptDefinition,
    SemanticEvaluatorConfig,
    StructuralEvaluatorConfig,
)
from evalshift_cli.models.client import CompletionResult, ModelClient

runner = CliRunner()


# ---------------------------------------------------------------------------
# _bar: block-bar renderer
# ---------------------------------------------------------------------------


class TestBar:
    def test_empty_when_total_zero(self) -> None:
        assert _bar(0, 0) == "▱" * 10

    def test_full_when_complete(self) -> None:
        assert _bar(80, 80) == "▰" * 10

    def test_half_when_half(self) -> None:
        out = _bar(40, 80)
        assert out.count("▰") == 5
        assert out.count("▱") == 5

    def test_clamps_overflow(self) -> None:
        # If completed > total (shouldn't happen, but defensive).
        out = _bar(100, 80)
        assert out == "▰" * 10


# ---------------------------------------------------------------------------
# _evaluator_family_summary
# ---------------------------------------------------------------------------


def _cfg_with(**evaluators: Any) -> EvalShiftConfig:
    return EvalShiftConfig(
        prompts=[
            PromptDefinition(
                id="p1",
                detection="manual",
                content="hello {name}",
                variables=["name"],
            ),
        ],
        evaluators=EvaluatorsConfig(**evaluators),
    )


class TestFamilySummary:
    def test_just_structural(self) -> None:
        cfg = _cfg_with(
            structural=[StructuralEvaluatorConfig(type="length", min_chars=1)],
        )
        assert _evaluator_family_summary(cfg.evaluators) == "structural"

    def test_full_house(self) -> None:
        cfg = _cfg_with(
            structural=[StructuralEvaluatorConfig(type="length", min_chars=1)],
            semantic=SemanticEvaluatorConfig(
                embedding_model="text-embedding-3-small",
            ),
            llm_judge=[
                LLMJudgeConfig(
                    criterion_name="helpfulness",
                    criterion_prompt="Is this helpful?",
                ),
            ],
        )
        assert _evaluator_family_summary(cfg.evaluators) == "structural · semantic · judge"

    def test_empty_returns_placeholder(self) -> None:
        cfg = _cfg_with()
        assert _evaluator_family_summary(cfg.evaluators) == "(none)"

    def test_reports_the_families_the_chosen_suite_resolves_to(self) -> None:
        """The gating line must name what *this* suite is scored with."""
        cfg = EvalShiftConfig.model_validate(
            {
                "version": 1,
                "prompts": [{"id": "p1", "detection": "manual", "content": "hi"}],
                "evaluators": {"semantic": {}},
                "suites": {
                    "main_chat": {
                        "path": "a.jsonl",
                        "evaluators": {"tool_selection": [{"name": "routing"}]},
                    },
                    "briefing": {"path": "b.jsonl"},
                },
            },
        )
        assert _evaluator_family_summary(cfg.evaluators_for("main_chat")) == "semantic · tool-call"
        assert _evaluator_family_summary(cfg.evaluators_for("briefing")) == "semantic"


# ---------------------------------------------------------------------------
# _compose_verdict
# ---------------------------------------------------------------------------


def _comp(
    *,
    severity: str,
    effect_size: float = 0.3,
    delta: float = 0.05,
    p_corr: float = 0.01,
    evaluator: str = "length",
) -> ComparisonResult:
    return ComparisonResult(
        prompt_id="p1",
        evaluator_name=evaluator,
        slice_name="all",
        n=20,
        test="paired_t",
        statistic=1.5,
        p_value=0.01,
        p_value_corrected=p_corr,
        effect_size=effect_size,
        effect_size_ci_low=effect_size - 0.1,
        effect_size_ci_high=effect_size + 0.1,
        delta_avg_score=delta,
        severity=severity,  # type: ignore[arg-type]
        notes=[],
    )


class TestVerdict:
    def test_critical_regression_takes_priority(self) -> None:
        comparisons = [
            _comp(severity="improved", effect_size=0.4, delta=0.1),
            _comp(severity="critical", effect_size=-0.9, delta=-0.2),
        ]
        verdict = _compose_verdict(comparisons)
        assert "regressed" in verdict.headline.plain
        # Detail line should reflect the regression's numbers (delta = -0.200).
        assert verdict.detail is not None
        assert "-0.200" in verdict.detail.plain

    def test_improved_only_picks_largest_effect(self) -> None:
        comparisons = [
            _comp(severity="improved", effect_size=0.2, delta=0.05, evaluator="length"),
            _comp(severity="improved", effect_size=0.6, delta=0.1, evaluator="judge"),
        ]
        verdict = _compose_verdict(comparisons)
        assert "significantly better" in verdict.headline.plain
        # The detail uses the d=0.6 row.
        assert verdict.detail is not None
        assert "0.60" in verdict.detail.plain

    def test_no_change_when_only_none_severity(self) -> None:
        comparisons = [_comp(severity="none", effect_size=0.05, delta=0.0)]
        verdict = _compose_verdict(comparisons)
        assert "no significant change" in verdict.headline.plain
        assert verdict.detail is None
        assert verdict.regression_callout is None

    def test_minor_regressions_show_callout(self) -> None:
        comparisons = [
            _comp(severity="improved", effect_size=0.4, delta=0.1),
            _comp(severity="medium", effect_size=-0.4, delta=-0.05, evaluator="judge"),
            _comp(severity="low", effect_size=-0.25, delta=-0.02, evaluator="length"),
        ]
        verdict = _compose_verdict(comparisons)
        assert "significantly better" in verdict.headline.plain
        assert verdict.regression_callout is not None
        callout = verdict.regression_callout.plain
        assert "2 sub-metrics regressed" in callout
        # Worst minor was the medium one (d=-0.4).
        assert "judge" in callout


# ---------------------------------------------------------------------------
# End-to-end smoke through CliRunner
# ---------------------------------------------------------------------------


def _scaffold(tmp_path: Path) -> None:
    """Lay out a minimal valid project under ``tmp_path``."""
    (tmp_path / "evalshift.yaml").write_text(
        """
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
          concurrency: 4
        evaluators:
          structural:
            - type: length
              min_chars: 1
              max_chars: 200
        """,
        encoding="utf-8",
    )
    rows = "\n".join(
        f'{{"id": "ex{i}", "inputs": {{"name": "User{i}"}}, "tools": []}}' for i in range(4)
    )
    (tmp_path / "golden.jsonl").write_text(rows + "\n", encoding="utf-8")


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
        return CompletionResult(
            text="A short polite reply.",
            model_id=str(kwargs["model"]),
            input_tokens=5,
            output_tokens=2,
            cost_usd=0.0,
            latency_ms=10,
        )

    monkeypatch.setattr(ModelClient, "complete", fake_complete)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


PIPED_CONSOLE_COLUMNS = "80"
"""Rich's fallback width whenever it cannot measure a terminal — i.e. any pipe."""

LONG_VIEW_URL = (
    "https://www.evalshift.dev/app/acme-analytics/model-migration"
    "/runs/0f1d2c3b-4a59-4c8e-9b1f-2d3e4f5a6b7c"
)
"""A realistic hosted run URL: 102 characters, well past the 80-column fold."""


class TestEndToEnd:
    def test_all_runs_full_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 0, result.output
        # Final verdict block is printed.
        assert (
            "candidate is significantly better" in result.output
            or "no significant change" in result.output
            or "candidate regressed" in result.output
        )
        # Report wrote an HTML file.
        runs = list((tmp_path / ".evalshift" / "runs").iterdir())
        assert len(runs) == 1
        assert (runs[0] / "report.html").exists()
        assert (runs[0] / "scores.jsonl").exists()
        assert (runs[0] / "analysis.json").exists()

    def test_all_prints_warnings_under_the_pipeline_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Warnings raised mid-pipeline surface in one section below the block.

        LiteLLM's deprecation notices used to print wherever the emitting call
        happened to be — above the live region, or glued between the block and
        the verdict. They are deferred and printed once, under the pipeline,
        separated by a blank line.
        """
        _scaffold(tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.chdir(tmp_path)
        # Unique per test: the dedupe filter's ``_seen`` set is process-global.
        warning = "e2e-test: sampling params deprecated for gemini-9-nano"

        async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            logging.getLogger("LiteLLM").warning(warning)
            return CompletionResult(
                text="A short polite reply.",
                model_id=str(kwargs["model"]),
                input_tokens=5,
                output_tokens=2,
                cost_usd=0.0,
                latency_ms=10,
            )

        monkeypatch.setattr(ModelClient, "complete", fake_complete)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 0, result.output
        lines = result.output.splitlines()
        (analyze_idx,) = [i for i, line in enumerate(lines) if "analyze" in line]
        warning_lines = [i for i, line in enumerate(lines) if warning in line]
        assert len(warning_lines) == 1, result.output
        (warning_idx,) = warning_lines
        # Under the block, named by origin, set off by exactly one blank line.
        assert warning_idx == analyze_idx + 2, result.output
        assert lines[analyze_idx + 1] == ""
        assert "LiteLLM:" in lines[warning_idx]

    def test_all_advisory_inconclusive_prints_reason_and_fix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Advisory-only evaluators + a policy → inconclusive verdict. The
        # verdict block must explain why and print the actionable fix, not
        # the misleading "collect more examples".
        _scaffold(tmp_path)
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "              max_chars: 200",
                "              max_chars: 200\n              blocking: false",
            )
            + "\n        migration_policy:\n"
            + "          max_critical_regressions: 0\n",
            encoding="utf-8",
        )
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert "Migration verdict: inconclusive" in flat
        assert "advisory" in flat
        assert "Set blocking: true" in flat
        assert "Collect more examples" not in flat

    def test_all_pushes_before_gate_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        pushed: list[str] = []

        def fake_push_local_run(
            *,
            run_id: str,
            config_path: Path,
            suite_path: Path,
            runs_base: Path,
            console: Any,
        ) -> Any:
            pushed.append(run_id)

            class Result:
                view_url = "https://app.test/app/acme/project/runs/" + run_id

            return Result()

        monkeypatch.setattr(
            "evalshift_cli.cli.commands.compare.push_local_run", fake_push_local_run
        )

        def fake_run_analyze(*, run_id: str, config_path: Path, runs_base: Path) -> AnalyzeResult:
            real = _real_run_analyze(run_id=run_id, config_path=config_path, runs_base=runs_base)
            return AnalyzeResult(
                run_id=real.run_id,
                output_path=real.output_path,
                comparisons=(
                    _comp(
                        severity="critical",
                        effect_size=-1.0,
                        delta=-0.4,
                    ),
                ),
                n_records=real.n_records,
            )

        monkeypatch.setattr("evalshift_cli.cli.commands.compare.run_analyze", fake_run_analyze)

        result = runner.invoke(app, ["all", "--yes", "--push", "--gate", "critical"])

        assert result.exit_code == 1
        assert len(pushed) == 1
        assert "https://app.test/app/acme/project/runs/" in result.output

    def test_all_push_prints_the_run_url_on_one_unbroken_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``all --push`` prints the same hosted URL ``push`` does, and must not fold it.

        Rich wraps at 80 columns whenever it cannot measure a terminal, which is
        every CI pipe. ``COLUMNS`` is pinned so a wide developer terminal cannot
        mask the fold.
        """
        assert len(LONG_VIEW_URL) > int(PIPED_CONSOLE_COLUMNS)
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("COLUMNS", PIPED_CONSOLE_COLUMNS)

        def fake_push_local_run(**_: Any) -> Any:
            class Result:
                view_url = LONG_VIEW_URL

            return Result()

        monkeypatch.setattr(
            "evalshift_cli.cli.commands.compare.push_local_run", fake_push_local_run
        )

        result = runner.invoke(app, ["all", "--yes", "--push"])

        assert result.exit_code == 0, result.output
        assert f"hosted: {LONG_VIEW_URL}" in [line.strip() for line in result.output.splitlines()]

    def test_all_invalid_config_shows_failing_check_detail(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # An invalid config must name the failing check in the doctor row,
        # not hide behind a generic "config invalid".
        _scaffold(tmp_path)
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + "        bogus_field: 1\n",
            encoding="utf-8",
        )
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 1
        assert "evalshift.yaml" in result.output
        assert "config invalid" not in result.output

    def test_all_doctor_check_failure_names_failing_check(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # The doctor row must surface the failing check's name and detail,
        # not a generic "config invalid".
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)

        def fake_run_checks(cwd: Path, env: Any) -> list[CheckResult]:
            return [
                CheckResult(
                    name="evalshift.yaml",
                    status="fail",
                    detail="2 validation errors",
                ),
            ]

        monkeypatch.setattr("evalshift_cli.cli.commands.compare.run_checks", fake_run_checks)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 1
        assert "evalshift.yaml: 2 validation errors" in result.output
        assert "config invalid" not in result.output

    def test_all_aborts_on_missing_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _scaffold(tmp_path)
        # Deliberately don't set the API key; ensure it isn't leaked in.
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])
        assert result.exit_code == 1
        assert "missing API key" in result.output


# ---------------------------------------------------------------------------
# Evaluator key preflight
# ---------------------------------------------------------------------------


_JUDGE_YAML = (
    "              max_chars: 200\n"
    "          llm_judge:\n"
    "            - criterion_name: equivalence\n"
    "              criterion_prompt: which is better?\n"
    "              judge_model: gpt-4o-mini\n"
    "              blocking: {blocking}"
)


class TestEvaluatorKeyPreflight:
    def _with_judge(self, tmp_path: Path, *, blocking: bool) -> None:
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "              max_chars: 200",
                _JUDGE_YAML.format(blocking=str(blocking).lower()),
            )
            + "\n        migration_policy:\n"
            + "          max_critical_regressions: 0\n",
            encoding="utf-8",
        )

    def test_keyless_advisory_judge_warns_and_the_run_completes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        self._with_judge(tmp_path, blocking=False)
        _patch_client(monkeypatch)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert "llm_judge.equivalence skipped: no API key for gpt-4o-mini" in flat

    @pytest.mark.xfail(reason="recommendation lines land in Task 5", strict=True)
    def test_skipped_judge_is_named_in_the_recommendations(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        self._with_judge(tmp_path, blocking=False)
        _patch_client(monkeypatch)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert (
            "llm_judge.equivalence was skipped: no API key for gpt-4o-mini. "
            "Export OPENAI_API_KEY to enable it."
        ) in flat

    def test_keyless_blocking_judge_exits_before_any_model_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        self._with_judge(tmp_path, blocking=True)
        _patch_client(monkeypatch)
        calls: list[str] = []

        async def spy_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls.append(str(kwargs["model"]))
            raise AssertionError("no model call may happen before the key preflight")

        monkeypatch.setattr(ModelClient, "complete", spy_complete)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 1
        flat = " ".join(result.output.split())
        assert (
            "missing API key for gpt-4o-mini (llm_judge.equivalence); export OPENAI_API_KEY."
            in flat
        )
        assert calls == []


class TestFamilySummarySkipped:
    def test_marks_a_skipped_semantic_and_an_all_skipped_judge(self) -> None:
        evaluators = EvaluatorsConfig.model_validate(
            {
                "semantic": {"embedding_model": "openai/text-embedding-3-small"},
                "llm_judge": [
                    {"criterion_name": "e", "criterion_prompt": "p", "judge_model": "gpt-4o-mini"}
                ],
            }
        )
        from evalshift_cli.evaluators.keys import evaluator_key_gaps

        gaps = evaluator_key_gaps(evaluators, {})
        assert _evaluator_family_summary(evaluators, gaps) == "semantic (no key) · judge (no key)"
        assert _evaluator_family_summary(evaluators) == "semantic · judge"
