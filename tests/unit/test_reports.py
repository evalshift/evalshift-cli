"""Tests for the reports package + the ``evalshift report`` command."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.analysis.policy import evaluate_migration_policy
from evalshift_cli.analysis.statistics import UNMEASURED_NOTE_PREFIX
from evalshift_cli.cli.commands.analyze import ANALYSIS_FILENAME
from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME
from evalshift_cli.cli.main import app
from evalshift_cli.config.models import MigrationPolicy
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.reports.html import REPORT_HTML_FILENAME, render_html, write_html
from evalshift_cli.reports.json import (
    REPORT_JSON_FILENAME,
    TopRegression,
    build_report_payload,
    write_report_json,
)
from evalshift_cli.runner.checkpoint import append_call, write_state
from evalshift_cli.runner.models import Call, RunModels, RunState
from evalshift_cli.traces.loader import TRACES_FILENAME

runner = CliRunner()


def _scaffold_full_run(
    tmp_path: Path,
    *,
    non_deterministic_models: list[str] | None = None,
    dropped_params: dict[str, list[str]] | None = None,
    samples_per_example: int = 1,
    with_judge_scores: bool = False,
) -> tuple[Path, str]:
    """Scaffold a run dir with raw.jsonl, scores.jsonl, and analysis.json.

    ``with_judge_scores`` adds an ``llm_judge.helpfulness`` row per example
    so the report sees a judge that actually contributed verdicts.
    """
    run_id = "r_20260601_aaaaaa"
    run_dir = tmp_path / ".evalshift" / "runs" / run_id

    # Real golden.jsonl so the report builder can attach tags + inputs.
    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text(
        '{"id": "ex1", "inputs": {"input": "Greet the user named Alex."}, '
        '"tags": ["greeting"], "tools": []}\n'
        '{"id": "ex2", "inputs": {}, "tags": ["greeting", "casual"], "tools": []}\n',
        encoding="utf-8",
    )

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
            suite_path=str(suite_path),
            total_evaluations=4,
            completed_evaluations=4,
            non_deterministic_models=non_deterministic_models or [],
            dropped_params=dropped_params or {},
            samples_per_example=samples_per_example,
        ),
    )

    # raw.jsonl with two pairs (times ``samples_per_example`` samples each; a
    # later sample's text is suffixed so a consumer that shows the wrong one
    # is caught).
    for ex_id, src_text, tgt_text in (
        ("ex1", "Hello there!", "Hi"),
        ("ex2", "Greetings, friend.", "yo"),
    ):
        for sample in range(samples_per_example):
            suffix = f" [sample {sample}]" if sample else ""
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id="gemini/gemini-2.5-flash",
                    role="source",
                    text=src_text + suffix,
                    cost_usd=0.0001,
                    latency_ms=100,
                    input_tokens=20,
                    output_tokens=10,
                    sample_index=sample,
                ),
            )
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id="gemini/gemini-2.5-pro",
                    role="target",
                    text=tgt_text + suffix,
                    cost_usd=0.0002,
                    latency_ms=120,
                    input_tokens=22,
                    output_tokens=15,
                    sample_index=sample,
                ),
            )

    # scores.jsonl
    rows = [
        EvalRecord(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex1",
            evaluator_name="structural.length",
            source_score=1.0,
            target_score=0.5,
            delta=-0.5,
        ),
        EvalRecord(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex2",
            evaluator_name="structural.length",
            source_score=1.0,
            target_score=0.0,
            delta=-1.0,
        ),
    ]
    if with_judge_scores:
        rows.extend(
            EvalRecord(
                run_id=run_id,
                prompt_id="greet",
                example_id=ex_id,
                evaluator_name="llm_judge.helpfulness",
                kind="llm_judge",
                source_score=0.5,
                target_score=0.5,
                delta=0.0,
            )
            for ex_id in ("ex1", "ex2")
        )
    (run_dir / SCORES_FILENAME).write_text(
        "\n".join(r.model_dump_json() for r in rows) + "\n",
        encoding="utf-8",
    )

    # analysis.json
    analysis = {
        "run_id": run_id,
        "models": {
            "source": "gemini/gemini-2.5-flash",
            "target": "gemini/gemini-2.5-pro",
        },
        "n_examples": 2,
        "n_records": 2,
        "slices": ["all"],
        "aggregates": {},
        "comparisons": [
            {
                "prompt_id": "greet",
                "evaluator_name": "structural.length",
                "slice_name": "all",
                "n": 2,
                "test": "skipped",
                "statistic": 0.0,
                "p_value": 1.0,
                "p_value_corrected": 1.0,
                "effect_size": 0.0,
                "effect_size_ci_low": 0.0,
                "effect_size_ci_high": 0.0,
                "delta_avg_score": -0.75,
                "severity": "insufficient",
                "notes": ["n=2 < 5; no test run"],
            },
        ],
    }
    (run_dir / ANALYSIS_FILENAME).write_text(json.dumps(analysis), encoding="utf-8")
    return tmp_path, run_id


def _write_migration_decision(run_dir: Path, run_id: str, verdict: str = "fail") -> None:
    decision = {
        "run_id": run_id,
        "source_model": "gemini/gemini-2.5-flash",
        "target_model": "gemini/gemini-2.5-pro",
        "verdict": verdict,
        "overall": {
            "n_records": 2,
            "improved_rate": 0.0,
            "equivalent_rate": 0.0,
            "regression_rate": 1.0,
            "critical_regressions": 1,
            "tool_argument_drift_rate": 0.0,
            "cost_increase_rate": 0.0,
            "latency_increase_rate": 0.0,
        },
        "slices": {},
        "budget_results": [
            {
                "name": "max_overall_regression_rate",
                "observed": 0.5,
                "allowed": 0.03,
                "passed": False,
                "scope": "overall",
            },
            {
                "name": "max_critical_regressions",
                "observed": 2.0,
                "allowed": 0.0,
                "passed": False,
                "scope": "overall",
            },
            {
                "name": "min_equivalence_rate",
                "observed": 0.1667,
                "allowed": 0.95,
                "passed": False,
                "scope": "overall",
            },
        ],
        "blocking_regressions": [
            {
                "prompt_id": "greet",
                "evaluator_name": "structural.length",
                "slice_name": "all",
                "severity": "critical",
                "delta_avg_score": -0.75,
                "effect_size": -1.2,
            },
        ],
        "failure_categories": [{"category": "SEMANTIC_REGRESSION", "count": 2}],
        "recommendations": ["Do not migrate globally under the configured policy."],
    }
    (run_dir / "migration_decision.json").write_text(
        json.dumps(decision),
        encoding="utf-8",
    )


def _write_report_agent_traces(run_dir: Path, run_id: str) -> None:
    source = {
        "run_id": run_id,
        "prompt_id": "greet",
        "example_id": "ex1",
        "role": "source",
        "events": [
            {
                "type": "tool_call",
                "sequence_index": 0,
                "timestamp": "2026-06-09T12:00:00Z",
                "metadata": {},
                "name": "check_refund_policy",
                "arguments": {},
            },
            {
                "type": "tool_call",
                "sequence_index": 1,
                "timestamp": "2026-06-09T12:00:01Z",
                "metadata": {},
                "name": "issue_refund",
                "arguments": {"ticket_id": "T-1032"},
            },
        ],
    }
    target = {
        "run_id": run_id,
        "prompt_id": "greet",
        "example_id": "ex1",
        "role": "target",
        "events": [
            {
                "type": "tool_call",
                "sequence_index": 0,
                "timestamp": "2026-06-09T12:00:00Z",
                "metadata": {},
                "name": "issue_refund",
                "arguments": {"ticket_id": "T-1023"},
            },
        ],
    }
    (run_dir / TRACES_FILENAME).write_text(
        json.dumps(source) + "\n" + json.dumps(target) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# build_report_payload
# ---------------------------------------------------------------------------


class TestReportPayload:
    def test_assembles_data(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        assert payload.run_id == run_id
        assert payload.source_model == "gemini/gemini-2.5-flash"
        assert payload.n_examples == 2
        assert payload.n_calls == 4
        assert len(payload.prompt_sections) == 1
        # Top regressions should have the negative-delta records.
        section = payload.prompt_sections[0]
        assert section.prompt_id == "greet"
        assert all(tr.delta < 0 for tr in section.top_regressions)

    def test_top_regression_carries_example_input(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        by_example = {tr.example_id: tr for tr in payload.prompt_sections[0].top_regressions}
        # ex1 has a single "input" var → rendered verbatim.
        if "ex1" in by_example:
            assert by_example["ex1"].input_text == "Greet the user named Alex."
        # ex2 has no input vars → None (nothing to show).
        if "ex2" in by_example:
            assert by_example["ex2"].input_text is None

    def test_top_regression_input_truncated_at_cap(self, tmp_path: Path) -> None:
        from evalshift_cli.reports.json import INPUT_TEXT_MAX_CHARS, _render_input_text

        big = "x" * (INPUT_TEXT_MAX_CHARS + 500)
        rendered = _render_input_text({"input": big}, example_id="ex1")
        assert rendered is not None
        assert len(rendered) < len(big)
        assert "truncated" in rendered
        assert "ex1" in rendered

    def test_render_input_text_multi_var_is_json(self, tmp_path: Path) -> None:
        from evalshift_cli.reports.json import _render_input_text

        rendered = _render_input_text({"a": 1, "b": "two"}, example_id="ex1")
        assert rendered is not None
        assert '"a"' in rendered and '"b"' in rendered

    def test_attaches_migration_decision_when_present(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_migration_decision(run_dir, run_id)

        payload = build_report_payload(run_dir)

        assert payload.migration_decision is not None
        assert payload.migration_decision["verdict"] == "fail"
        assert payload.migration_decision["failure_categories"][0]["category"] == (
            "SEMANTIC_REGRESSION"
        )

    def test_economics_aggregates_per_role(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        econ = payload.prompt_sections[0].economics

        # 2 calls per role, all live (no `cached=True`), no errors.
        assert econ.source.calls == 2
        assert econ.source.live_calls == 2
        assert econ.source.cached_calls == 0
        assert econ.source.failed_calls == 0
        assert econ.source.truncated_calls == 0
        assert econ.source.total_cost_usd == pytest.approx(0.0002)
        assert econ.source.total_input_tokens == 40
        assert econ.source.total_output_tokens == 20
        assert econ.source.latency_ms_avg == pytest.approx(100.0)
        assert econ.source.latency_ms_p95 == 100.0

        assert econ.target.calls == 2
        assert econ.target.total_cost_usd == pytest.approx(0.0004)
        assert econ.target.total_input_tokens == 44
        assert econ.target.total_output_tokens == 30
        assert econ.target.latency_ms_avg == pytest.approx(120.0)

    def test_example_rows_aggregate_per_example(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        rows = payload.prompt_sections[0].example_rows

        # 2 examples in the fixture, both have a complete (source, target) pair.
        assert len(rows) == 2
        rows_by_id = {r.example_id: r for r in rows}

        # Δ latency = target − source = 120 − 100 = 20 ms.
        assert rows_by_id["ex1"].delta_latency_ms == 20
        # Δ cost = 0.0002 − 0.0001 = 0.0001.
        assert rows_by_id["ex1"].delta_cost_usd == pytest.approx(0.0001)
        # Worst delta score from the fixture's structural.length records.
        # ex1: -0.5, ex2: -1.0. So worst on ex1 is -0.5.
        assert rows_by_id["ex1"].worst_delta_score == pytest.approx(-0.5)
        assert rows_by_id["ex2"].worst_delta_score == pytest.approx(-1.0)
        # Tags came through from the suite.
        assert rows_by_id["ex1"].tags == ["greeting"]
        assert rows_by_id["ex2"].tags == ["greeting", "casual"]
        # No tool evaluators in this fixture → tool_match is None.
        assert rows_by_id["ex1"].tool_match is None
        assert rows_by_id["ex2"].tool_match is None

    def test_example_rows_sorted_worst_first(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        rows = payload.prompt_sections[0].example_rows
        # ex2 has worst_delta = -1.0; ex1 has -0.5. ex2 should come first.
        assert rows[0].example_id == "ex2"
        assert rows[1].example_id == "ex1"

    def test_example_rows_flag_tool_mismatch(self, tmp_path: Path) -> None:
        """Inject a tool_selection record with target_score < 1 and confirm
        the breakdown row flags tool_match=False."""
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        # Append a tool_selection record on ex1 where target dropped the
        # expected tool (target_score = 0).
        with (run_dir / SCORES_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(
                EvalRecord(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id="ex1",
                    evaluator_name="tool_selection.routing",
                    source_score=1.0,
                    target_score=0.0,
                    delta=-1.0,
                ).model_dump_json()
                + "\n",
            )
        # And one where target matched (target_score = 1) on ex2.
        with (run_dir / SCORES_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(
                EvalRecord(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id="ex2",
                    evaluator_name="tool_selection.routing",
                    source_score=1.0,
                    target_score=1.0,
                    delta=0.0,
                ).model_dump_json()
                + "\n",
            )
        payload = build_report_payload(
            run_dir,
            tool_evaluator_names=frozenset({"tool_selection.routing"}),
        )
        rows_by_id = {r.example_id: r for r in payload.prompt_sections[0].example_rows}
        assert rows_by_id["ex1"].tool_match is False
        assert rows_by_id["ex2"].tool_match is True

    def test_top_regression_attaches_imported_agent_trace_diff(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_report_agent_traces(run_dir, run_id)
        with (run_dir / SCORES_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(
                EvalRecord(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id="ex1",
                    evaluator_name="trace_safety",
                    source_score=1.0,
                    target_score=0.0,
                    delta=-1.0,
                    metadata={"failure_categories": ["ARGUMENT_VALUE_DRIFT"]},
                ).model_dump_json()
                + "\n",
            )

        payload = build_report_payload(run_dir)
        top = payload.prompt_sections[0].top_regressions[0]

        assert top.source_agent_trace is not None
        assert top.target_agent_trace is not None
        assert top.trace_diff is not None
        assert any(item.category == "ARGUMENT_VALUE_DRIFT" for item in top.trace_diff.items)

    def test_economics_skips_cached_in_latency(self, tmp_path: Path) -> None:
        """Cached call's latency_ms is 0 by convention; it must not pull
        the average down."""
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        # Inject one extra cached source call with latency 0.
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_cached",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text="cached",
                cost_usd=0.0,
                latency_ms=0,
                cached=True,
            ),
        )
        payload = build_report_payload(run_dir)
        econ = payload.prompt_sections[0].economics
        assert econ.source.calls == 3
        assert econ.source.cached_calls == 1
        assert econ.source.live_calls == 2
        # Avg should still be 100ms (the two live calls), not pulled to ~67ms.
        assert econ.source.latency_ms_avg == pytest.approx(100.0)

    def test_truncated_calls_counted_in_economics_and_header(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        # Inject one truncated target call (finish_reason="length").
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_trunc",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="Hel",
                finish_reason="length",
            ),
        )
        payload = build_report_payload(run_dir)
        assert payload.truncated_calls == 1
        assert payload.prompt_sections[0].economics.target.truncated_calls == 1

    def test_missing_analysis_raises(self, tmp_path: Path) -> None:
        run_id = "r_20260601_xxxxxx"
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(source="a", target="b"),
                prompt_ids=["a"],
                suite_path="x.jsonl",
                total_evaluations=0,
                completed_evaluations=0,
            ),
        )
        with pytest.raises(FileNotFoundError, match="analysis"):
            build_report_payload(run_dir)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


class TestHtmlRender:
    def test_executive_summary_is_plain_language(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        # Friendly column headers replace the raw-stat ones.
        assert "<th>Verdict</th>" in html
        assert "Score change" in html
        assert "Confidence" in html
        # The "Worst severity" jargon header is gone entirely (detailed
        # per-prompt tables keep the raw p<sub>corrected</sub> columns).
        assert "Worst severity" not in html
        # A plain-language verdict sentence renders for the scaffolded prompt,
        # which has too few pairs to judge.
        assert "Not enough data" in html
        assert "verdict-head" in html
        # Small samples (n < 10) are flagged so tiny effect sizes aren't
        # mistaken for conclusive evidence.
        assert "small-sample" in html
        # Raw statistics survive as hover tooltips for power users.
        assert "Cohen's |d|" in html
        assert "Benjamini–Hochberg" in html  # noqa: RUF001 — matches report copy

    def test_per_example_breakdown_is_plain_language(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        # Column heads name the quantity, and the caption below the table —
        # not a tooltip — says what the number is measured against.
        assert "Time Δ" in html
        assert "Cost Δ" in html
        assert "Worst score Δ" in html
        assert "target minus source" in html

    def test_evaluator_labels_are_friendly(self) -> None:
        from evalshift_cli.reports.html import _evaluator_label, _test_label

        assert _evaluator_label("semantic.cosine") == "Semantic similarity"
        assert _evaluator_label("llm_judge.equivalence") == "LLM judge: equivalence"
        # Unknown families still get a humanised label, never a KeyError.
        assert _evaluator_label("custom.my_metric") == "Custom: my metric"
        assert _test_label("wilcoxon") == "Wilcoxon signed-rank test"
        assert _test_label("mystery") == "mystery"

    def test_aggregate_table_is_plain_language(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        # Friendly headers replace the jargon ones.
        assert "Score change" in html
        assert "Effect size" in html
        assert "Confidence" in html
        # The bare "Test" and dual raw/corrected p columns are folded away.
        assert "<th>Test</th>" not in html
        assert "p<sub>raw</sub>" not in html
        # The evaluator's raw id is still shown for power users.
        assert "eval-id" in html
        # Exact p-values survive in the row tooltip.
        assert "p (corrected)" in html

    def test_slice_table_is_plain_language(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        # Friendlier section heading and headers.
        assert "Changes within specific slices" in html
        assert "Slices with significant change" not in html
        # A caption explains what a slice is.
        assert "Slices are subsets of your examples" in html

    def test_no_determinism_banner_when_sampling_is_controlled(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        # Asserting on the heading, not the CSS class: the stylesheet is
        # inlined into every report, so the class name is always present.
        assert "Sampling is not controlled" not in render_html(payload)

    def test_determinism_banner_precedes_the_numbers(self, tmp_path: Path) -> None:
        """Buried in the methodology list, this would go unread.

        It weakens every p-value in the run, so it renders above the verdict.
        """
        cwd, run_id = _scaffold_full_run(
            tmp_path, non_deterministic_models=["gemini/gemini-3.5-flash-lite"]
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        assert "Sampling is not controlled" in html
        assert "gemini/gemini-3.5-flash-lite" in html
        # Above the executive summary, so it is read before the numbers it
        # qualifies rather than after them.
        assert html.index("Sampling is not controlled") < html.index("<h2>")

    def test_determinism_banner_lists_every_affected_model(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(
            tmp_path,
            non_deterministic_models=[
                "gemini/gemini-2.5-flash",
                "gemini/gemini-3.5-flash-lite",
            ],
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        assert "gemini/gemini-2.5-flash" in html
        assert "gemini/gemini-3.5-flash-lite" in html

    def test_regression_reason_explains_why(self) -> None:
        from evalshift_cli.reports.html import _regression_reason
        from evalshift_cli.reports.json import ToolChange

        def _reason(**overrides: object) -> str:
            defaults: dict[str, object] = {
                "prompt_id": "greet",
                "example_id": "ex1",
                "evaluator_name": "custom.metric",
                "delta": -0.5,
                "source_text": "",
                "target_text": "",
            }
            return _regression_reason(TopRegression(**{**defaults, **overrides}))  # type: ignore[arg-type]

        # An evaluator's own rationale wins.
        assert _reason(
            evaluator_name="llm_judge.factuality", explanation="Source was correct."
        ) == ("Source was correct.")
        # A numeric semantic evaluator gets a synthesised, plain reason.
        assert "90% similar" in _reason(
            evaluator_name="semantic.cosine", metadata={"raw_cosine": 0.8966}
        )
        # Unknown numeric evaluator with no prose → no fabricated reason.
        assert _reason() == ""
        # The axis, not the user-chosen name, is what picks the sentence: on
        # the run that prompted this work both axes were called ``routing``,
        # which matches no family prefix and produced no reason at all.
        divergence = _reason(
            evaluator_name="routing",
            kind="tool_selection.divergence",
            tool_change=ToolChange(
                source_names=["get_recent_files"], target_names=["display_info"]
            ),
        )
        assert "get_recent_files" in divergence
        assert "display_info" in divergence
        conformance = _reason(evaluator_name="routing", kind="tool_selection.conformance")
        assert "recorded tool calls" in conformance
        assert conformance != divergence

    def test_top_regression_renders_collapsed_input(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)
        # ex1 regressions carry the example input, collapsed by default.
        assert '<details class="reg-input">' in html
        assert "Greet the user named Alex." in html

    def test_top_regressions_open_first_and_hint_expandable(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)

        opened = html.count('<details class="panel regression" open>')
        closed = html.count('<details class="panel regression">')
        # The fixture must yield several regressions for this to mean anything.
        assert opened + closed >= 2
        # Only each section's worst regression starts expanded, so the reader
        # sees one opened and learns the rest open too.
        assert opened == sum(1 for ps in payload.prompt_sections if ps.top_regressions)
        # Every row says it expands, in words, not just a caret glyph.
        assert html.count('<span class="reg-toggle"') == opened + closed

    def test_top_regression_shows_reason_and_scores(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        if payload.prompt_sections and any(ps.top_regressions for ps in payload.prompt_sections):
            assert "Why flagged" in html
            assert "source 1.00 → target 0.00" in html

    def test_latency_uses_human_units(self, tmp_path: Path) -> None:
        from evalshift_cli.reports.html import _latency

        # No live calls (all cached) → em dash, never a misleading "0 ms".
        assert _latency(0.0, 0) == "—"
        # Sub-second stays in milliseconds.
        assert _latency(847.0, 3) == "847 ms"
        # ≥1s renders as seconds with one decimal.
        assert _latency(11541.0, 3) == "11.5 s"
        assert _latency(12673.0, 6) == "12.7 s"

    def test_renders_with_inlined_css(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)
        assert html.startswith("<!DOCTYPE html>")
        # CSS must be inlined (we ship no external assets).
        assert "<style>" in html
        # Run id and key metadata reach the document.
        assert run_id in html
        assert "gemini/gemini-2.5-flash" in html
        # No external scripts.
        assert "<script" not in html.lower()
        # The header's truncated count must render (regression guard: the
        # renderer once dropped this field from the template context).
        assert "Failed / truncated" in html
        assert '<th class="num">Truncated</th>' in html

    def test_renders_migration_verdict_when_present(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_migration_decision(run_dir, run_id)
        payload = build_report_payload(run_dir)

        html = render_html(payload)

        assert "Migration verdict" in html
        # Failure categories render as plain language, never the machine label.
        assert "Meaning of the answer changed" in html
        assert "SEMANTIC_REGRESSION" not in html
        assert "Do not migrate globally" in html

        # Budget failures render as human-readable percentages with friendly
        # labels, the raw config key, and a failure-direction comparator.
        assert "Overall regression rate" in html  # friendly label
        assert "max_overall_regression_rate" in html  # raw config key kept
        assert "≤ 3.0%" in html  # rate budget: fraction -> percent + comparator
        assert "≥ 95.0%" in html  # min_* budget flips the comparator
        assert "≤ 0" in html  # count budget stays an integer, no percent
        assert "0.030" not in html  # no bare fractions anymore

    def test_a_failed_trace_rule_budget_renders_as_a_count(self, tmp_path: Path) -> None:
        """``max_invariant_violations`` is a count of examples, never a percentage."""
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_migration_decision(run_dir, run_id)
        path = run_dir / "migration_decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["budget_results"] = [
            {
                "name": "max_invariant_violations",
                "observed": 2.0,
                "allowed": 1.0,
                "passed": False,
                "scope": "overall",
                "conclusive": True,
                "denominator": 2,
            },
        ]
        path.write_text(json.dumps(decision), encoding="utf-8")

        html = render_html(build_report_payload(run_dir))

        assert "Trace-rule violations" in html
        assert '<td class="num bad">2</td>' in html  # observed: two examples
        assert "≤ 1<" in html  # allowed: one example
        assert "200.0%" not in html

    def test_sub_granular_budget_warning_reaches_the_report(self, tmp_path: Path) -> None:
        # The report describes the *persisted* decision, so the warning has to
        # survive `to_dict()` → migration_decision.json → payload → HTML. A
        # note that only ever reaches the terminal is half a fix.
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        decision = evaluate_migration_policy(
            run_id=run_id,
            source_model="gemini/gemini-2.5-flash",
            target_model="gemini/gemini-2.5-pro",
            policy=MigrationPolicy(max_tool_argument_drift=0.01),
            comparisons=[],
            records=[
                EvalRecord(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=f"ex{i}",
                    evaluator_name="routing_args",
                    kind="tool_arguments",
                    source_score=1.0,
                    target_score=1.0,
                    delta=0.0,
                )
                for i in range(10)
            ],
            calls=[],
        )
        (run_dir / "migration_decision.json").write_text(
            json.dumps(decision.to_dict()),
            encoding="utf-8",
        )
        payload = build_report_payload(run_dir)

        html = render_html(payload)

        assert (
            "The tool-argument drift budget of 1% (max_tool_argument_drift in "
            "evalshift.yaml) is below the 10% granularity of 10 tool-argument "
            "comparisons — effective tolerance is zero at this sample size."
        ) in html

    def test_all_advisory_verdict_shows_reason_and_advisory_rates(self, tmp_path: Path) -> None:
        # All-advisory run: blocking n=0 → the panel must explain WHY it is
        # inconclusive and show the advisory rates instead of misleading 0.0%.
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        decision = {
            "run_id": run_id,
            "source_model": "gemini/gemini-2.5-flash",
            "target_model": "gemini/gemini-2.5-pro",
            "verdict": "inconclusive",
            "overall": {
                "n_records": 0,
                "improved_rate": 0.0,
                "equivalent_rate": 0.0,
                "regression_rate": 0.0,
                "critical_regressions": 0,
                "tool_argument_drift_rate": 0.0,
                "cost_increase_rate": 0.0,
                "latency_increase_rate": 0.0,
            },
            "slices": {},
            "budget_results": [],
            "blocking_regressions": [],
            "failure_categories": [],
            "recommendations": ["Collect more examples before making a migration decision."],
            "reason": "no blocking evaluator records — every configured evaluator is advisory",
            "advisory": {
                "n_records": 36,
                "improved_rate": 0.5,
                "equivalent_rate": 0.333,
                "regression_rate": 0.167,
                "critical_regressions": 0,
                "tool_argument_drift_rate": 0.0,
                "cost_increase_rate": 0.0,
                "latency_increase_rate": 0.0,
            },
            "advisory_regressions": [],
        }
        (run_dir / "migration_decision.json").write_text(json.dumps(decision), encoding="utf-8")
        payload = build_report_payload(run_dir)

        html = render_html(payload)

        assert "no blocking evaluator records" in html
        # Advisory rates shown, labeled advisory; no bare 0.0% rates row.
        assert "advisory" in html.lower()
        assert "50.0%" in html
        assert "16.7%" in html
        assert "Equivalent 0.0%" not in html

    def test_renders_imported_agent_trace_timeline(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        _write_report_agent_traces(run_dir, run_id)
        with (run_dir / SCORES_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(
                EvalRecord(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id="ex1",
                    evaluator_name="trace_safety",
                    source_score=1.0,
                    target_score=0.0,
                    delta=-1.0,
                    metadata={"failure_categories": ["ARGUMENT_VALUE_DRIFT"]},
                ).model_dump_json()
                + "\n",
            )
        payload = build_report_payload(run_dir)

        html = render_html(payload)

        assert "Source agent trace" in html
        assert "Target agent trace" in html
        assert "check_refund_policy" in html
        # The category badge speaks the reader's language.
        assert "Tool arguments changed" in html
        assert "ARGUMENT_VALUE_DRIFT" not in html

    def test_writes_file(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        payload = build_report_payload(run_dir)
        out = write_html(payload, run_dir)
        assert out.name == REPORT_HTML_FILENAME
        assert out.exists()


# ---------------------------------------------------------------------------
# Judge family note — the judge graded its own relatives
# ---------------------------------------------------------------------------

_JUDGES = {"llm_judge.helpfulness": "gemini/gemini-3.1-flash-lite-preview"}


class TestJudgeFamilyNote:
    def test_payload_names_the_judge_that_shares_the_arms_family(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path, with_judge_scores=True)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id, judge_models=_JUDGES)
        # Both scaffold arms are gemini, as is the judge.
        assert payload.judge_family_overlap == [
            {
                "judge_model": "gemini/gemini-3.1-flash-lite-preview",
                "provider": "google",
                "roles": ["source", "target"],
            }
        ]

    def test_html_renders_a_one_line_note_and_json_carries_it(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path, with_judge_scores=True)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        payload = build_report_payload(run_dir, judge_models=_JUDGES)

        html = render_html(payload)
        assert "shares a model family" in html
        assert "gemini/gemini-3.1-flash-lite-preview" in html

        write_report_json(payload, run_dir)
        data = json.loads((run_dir / REPORT_JSON_FILENAME).read_text(encoding="utf-8"))
        assert data["judge_family_overlap"][0]["provider"] == "google"

    def test_no_note_when_the_judge_contributed_no_scores(self, tmp_path: Path) -> None:
        # Configured but never ran (applies_to matched nothing, or the run
        # predates the criterion): there is no verdict to caveat.
        cwd, run_id = _scaffold_full_run(tmp_path, with_judge_scores=False)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id, judge_models=_JUDGES)
        assert payload.judge_family_overlap == []
        assert "shares a model family" not in render_html(payload)

    def test_no_note_for_a_third_family_judge(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path, with_judge_scores=True)
        payload = build_report_payload(
            cwd / ".evalshift" / "runs" / run_id,
            judge_models={"llm_judge.helpfulness": "openai/gpt-4.1-mini"},
        )
        assert payload.judge_family_overlap == []
        assert "shares a model family" not in render_html(payload)

    def test_default_payload_has_no_overlap_and_json_still_has_the_key(
        self, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path, with_judge_scores=True)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        payload = build_report_payload(run_dir)
        assert payload.judge_family_overlap == []
        write_report_json(payload, run_dir)
        data = json.loads((run_dir / REPORT_JSON_FILENAME).read_text(encoding="utf-8"))
        assert data["judge_family_overlap"] == []

    def test_report_command_reads_the_judges_from_the_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path, with_judge_scores=True)
        (cwd / "evalshift.yaml").write_text(
            "prompts:\n  - {id: greet, detection: manual, content: hi}\n"
            "evaluators:\n  llm_judge:\n    - criterion_name: helpfulness\n"
            "      criterion_prompt: which helps more?\n"
            "      judge_model: gemini/gemini-3.1-flash-lite-preview\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["report", run_id, "--no-insights"])
        assert result.exit_code == 0, result.stdout
        run_dir = cwd / ".evalshift" / "runs" / run_id
        data = json.loads((run_dir / REPORT_JSON_FILENAME).read_text(encoding="utf-8"))
        assert data["judge_family_overlap"][0]["roles"] == ["source", "target"]
        assert "shares a model family" in (run_dir / REPORT_HTML_FILENAME).read_text(
            encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestReportCommand:
    def test_writes_html_and_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        # The command needs an evalshift.yaml in the cwd.
        (cwd / "evalshift.yaml").write_text(
            "prompts:\n  - {id: greet, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(cwd)
        result = runner.invoke(app, ["report", run_id])
        assert result.exit_code == 0, result.stdout
        run_dir = cwd / ".evalshift" / "runs" / run_id
        assert (run_dir / REPORT_HTML_FILENAME).exists()
        assert (run_dir / REPORT_JSON_FILENAME).exists()

    def test_missing_analysis_exits_one_with_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run_id = "r_20260601_yyyyyy"
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        write_state(
            run_dir,
            RunState(
                run_id=run_id,
                status="completed",
                config_hash="x",
                started_at=datetime(2026, 6, 1, tzinfo=UTC),
                models=RunModels(source="a", target="b"),
                prompt_ids=["a"],
                suite_path="x.jsonl",
                total_evaluations=0,
                completed_evaluations=0,
            ),
        )
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["report", run_id])
        assert result.exit_code == 1
        assert "evaluate" in result.stdout
        assert "analyze" in result.stdout


# ---------------------------------------------------------------------------
# v0.2 — trace rendering
# ---------------------------------------------------------------------------


from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace  # noqa: E402


class TestTraceRendering:
    def _scaffold_agent_run(self, tmp_path: Path) -> tuple[Path, str]:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        # Replace the calls in raw.jsonl with tool-bearing ones so the
        # trace plumbing shows up end-to-end.
        from evalshift_cli.runner.checkpoint import append_call
        from evalshift_cli.runner.models import Call

        (run_dir / "raw.jsonl").unlink()
        for ex_id in ("ex1", "ex2"):
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id="gemini/gemini-2.5-flash",
                    role="source",
                    text="",
                    cost_usd=0.0,
                    latency_ms=1,
                    trace=ToolTrace(
                        calls=[
                            ToolCall(
                                tool_name="search_db",
                                arguments={"q": "ACME"},
                                sequence_index=0,
                            ),
                        ],
                    ),
                ),
            )
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="greet",
                    example_id=ex_id,
                    model_id="gemini/gemini-2.5-pro",
                    role="target",
                    text="",
                    cost_usd=0.0,
                    latency_ms=1,
                    trace=ToolTrace(calls=[]),  # target dropped the call
                ),
            )
        return cwd, run_id

    def test_payload_attaches_traces_to_top_regressions(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_agent_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        section = payload.prompt_sections[0]
        # The fixture has negative-delta scores → top regressions populated.
        if section.top_regressions:
            tr = section.top_regressions[0]
            assert tr.source_trace is not None
            assert tr.target_trace is not None
            assert tr.source_trace.tool_names == ["search_db"]
            assert tr.target_trace.tool_names == []
        # Section flag set.
        assert section.has_tool_traces is True

    def test_html_renders_trace_diff(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_agent_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)
        # Trace structure markers in the rendered HTML.
        if any(s.has_tool_traces for s in payload.prompt_sections) and any(
            tr.source_trace for s in payload.prompt_sections for tr in s.top_regressions
        ):
            assert "Source trace" in html
            assert "Target trace" in html
            assert "search_db" in html
        # No JS still — even with the new trace section.
        assert "<script" not in html.lower()


# ---------------------------------------------------------------------------
# Multi-turn — conversation transcript + turn badge
# ---------------------------------------------------------------------------


class TestMultiTurnTranscript:
    def _scaffold_multiturn_run(self, tmp_path: Path) -> tuple[Path, str]:
        """Rewrite ex1 in the suite as a multi-turn example with history."""
        cwd, run_id = _scaffold_full_run(tmp_path)
        suite_path = cwd / "golden.jsonl"
        suite_path.write_text(
            json.dumps(
                {
                    "id": "ex1",
                    "inputs": {},
                    "tags": ["greeting"],
                    "tools": [],
                    "history": [
                        {"role": "user", "content": "What time is it?"},
                        {"role": "assistant", "content": "It's noon."},
                    ],
                    "conversation_id": "conv-1",
                    "turn_index": 2,
                },
            )
            + "\n"
            + json.dumps({"id": "ex2", "inputs": {}, "tags": ["greeting", "casual"], "tools": []})
            + "\n",
            encoding="utf-8",
        )
        return cwd, run_id

    def test_top_regression_carries_history_and_turn_index(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_multiturn_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        section = payload.prompt_sections[0]
        by_example = {tr.example_id: tr for tr in section.top_regressions}

        ex1 = by_example["ex1"]
        assert ex1.history == [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It's noon."},
        ]
        assert ex1.turn_index == 2

        ex2 = by_example["ex2"]
        assert ex2.history is None
        assert ex2.turn_index is None

    def test_example_row_carries_turn_index(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_multiturn_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        rows_by_id = {r.example_id: r for r in payload.prompt_sections[0].example_rows}

        assert rows_by_id["ex1"].turn_index == 2
        assert rows_by_id["ex2"].turn_index is None

    def test_serialised_payload_includes_history_and_turn_index(self, tmp_path: Path) -> None:
        from evalshift_cli.reports.json import _to_jsonable

        cwd, run_id = self._scaffold_multiturn_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        data = _to_jsonable(payload)
        section = data["prompt_sections"][0]

        ex1_row = next(r for r in section["example_rows"] if r["example_id"] == "ex1")
        assert ex1_row["turn_index"] == 2

        ex1_regression = next(tr for tr in section["top_regressions"] if tr["example_id"] == "ex1")
        assert ex1_regression["turn_index"] == 2
        assert ex1_regression["history"] == [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It's noon."},
        ]

    def test_html_renders_transcript_details_when_history_present(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_multiturn_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)

        assert "Conversation context (2 messages)" in html
        assert '<span class="transcript-role">[user]</span> What time is it?' in html
        assert '<span class="transcript-role">[assistant]</span>' in html
        assert "It&#39;s noon." in html

    def test_html_renders_tool_turns_in_the_transcript(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        (cwd / "golden.jsonl").write_text(
            json.dumps(
                {
                    "id": "ex1",
                    "inputs": {},
                    "tags": ["greeting"],
                    "tools": [],
                    "history": [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "c1", "name": "get_projects", "arguments": {"status": "a"}},
                            ],
                        },
                        {"role": "tool", "tool_call_id": "c1", "content": '{"projects": []}'},
                    ],
                    "turn_index": 1,
                },
            )
            + "\n"
            + json.dumps({"id": "ex2", "inputs": {}, "tags": ["greeting", "casual"], "tools": []})
            + "\n",
            encoding="utf-8",
        )
        html = render_html(build_report_payload(cwd / ".evalshift" / "runs" / run_id))

        assert '<span class="transcript-role">[assistant]</span>' in html
        assert "→ get_projects(" in html
        assert '<span class="transcript-role">[tool]</span>' in html

    def test_html_omits_transcript_details_when_no_history(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)

        assert "Conversation context" not in html

    def test_html_renders_turn_badge_in_example_rows(self, tmp_path: Path) -> None:
        cwd, run_id = self._scaffold_multiturn_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)

        assert "turn 2" in html

    def test_html_escapes_history_content(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        suite_path = cwd / "golden.jsonl"
        suite_path.write_text(
            json.dumps(
                {
                    "id": "ex1",
                    "inputs": {},
                    "tags": ["greeting"],
                    "tools": [],
                    "history": [
                        {"role": "user", "content": "<script>alert(1)</script>"},
                    ],
                    "turn_index": 1,
                },
            )
            + "\n"
            + json.dumps({"id": "ex2", "inputs": {}, "tools": []})
            + "\n",
            encoding="utf-8",
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        html = render_html(payload)

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_escapes_content_without_manual_escape_filters(self, tmp_path: Path) -> None:
        """Autoescape must cover interpolations that lack a manual |escape.

        Tags and evaluator explanations are user/model-controlled and render
        without explicit |escape filters in the template, so they only stay
        safe if the Jinja environment autoescapes ``.j2`` templates.
        """
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id

        # Malicious tag flows into the per-example chip spans.
        (cwd / "golden.jsonl").write_text(
            json.dumps(
                {"id": "ex1", "inputs": {}, "tags": ["<script>alert(1)</script>"], "tools": []}
            )
            + "\n"
            + json.dumps({"id": "ex2", "inputs": {}, "tools": []})
            + "\n",
            encoding="utf-8",
        )
        # Malicious evaluator explanation flows into the "Why flagged" line.
        rows = [
            EvalRecord(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex1",
                evaluator_name="structural.length",
                source_score=1.0,
                target_score=0.5,
                delta=-0.5,
                explanation="<script>alert(2)</script>",
            ),
            EvalRecord(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex2",
                evaluator_name="structural.length",
                source_score=1.0,
                target_score=0.0,
                delta=-1.0,
            ),
        ]
        (run_dir / SCORES_FILENAME).write_text(
            "\n".join(r.model_dump_json() for r in rows) + "\n",
            encoding="utf-8",
        )

        payload = build_report_payload(run_dir)
        html = render_html(payload)

        assert "<script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "&lt;script&gt;alert(2)&lt;/script&gt;" in html


# ---------------------------------------------------------------------------
# Multi-turn — empty-output (thinking-only) tracking
# ---------------------------------------------------------------------------


class TestEmptyOutputTracking:
    def test_role_economics_counts_empty_output_calls(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        # Empty text but tokens spent and no error → thinking-only response.
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_empty",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=956,
                finish_reason="stop",
            ),
        )
        payload = build_report_payload(run_dir)
        econ = payload.prompt_sections[0].economics
        assert econ.target.empty_output_calls == 1
        assert econ.source.empty_output_calls == 0

    def test_empty_output_not_counted_when_error_present(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_error",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=100,
                error="rate limited",
            ),
        )
        payload = build_report_payload(run_dir)
        assert payload.prompt_sections[0].economics.target.empty_output_calls == 0

    def test_empty_output_not_counted_when_zero_output_tokens(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_zero_tok",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=0,
            ),
        )
        payload = build_report_payload(run_dir)
        assert payload.prompt_sections[0].economics.target.empty_output_calls == 0

    def test_empty_output_not_counted_for_tool_trace_calls(self, tmp_path: Path) -> None:
        """A pure tool-call turn has empty text + tokens + no error, but it's
        not a "thinking-only" response — it's the primary agent use case. The
        orchestrator's tool path sets ``text=result.trace.final_text or ""``,
        so this must not be flagged as empty output.
        """
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_tool",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=200,
                finish_reason="stop",
                trace=ToolTrace(
                    calls=[
                        ToolCall(
                            tool_name="search_db",
                            arguments={"q": "ACME"},
                            sequence_index=0,
                        ),
                    ],
                ),
            ),
        )
        payload = build_report_payload(run_dir)
        assert payload.prompt_sections[0].economics.target.empty_output_calls == 0

    def test_empty_output_not_counted_when_finish_reason_not_stop(self, tmp_path: Path) -> None:
        """Aligns with the client-side warning gate in
        ``ModelClient._build_result``, which only warns about a likely
        thinking-only response when ``finish_reason == "stop"``. A ``None``
        or ``"length"`` finish reason means we can't attribute the empty
        text to that cause, so it shouldn't be counted either.
        """
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_no_finish_reason",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=200,
                finish_reason=None,
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_length",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=200,
                finish_reason="length",
            ),
        )
        payload = build_report_payload(run_dir)
        assert payload.prompt_sections[0].economics.target.empty_output_calls == 0

    def test_top_regression_flags_target_empty_output(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        # Overwrite ex1's target call with an empty-output response so the
        # existing ex1 regression (structural.length, delta -0.5) picks it up.
        raw_path = run_dir / "raw.jsonl"
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        rewritten = []
        for line in lines:
            call = Call.model_validate_json(line)
            if call.example_id == "ex1" and call.role == "target":
                call = call.model_copy(
                    update={"text": "", "output_tokens": 956, "finish_reason": "stop"},
                )
            rewritten.append(call.model_dump_json())
        raw_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

        payload = build_report_payload(run_dir)
        by_example = {tr.example_id: tr for tr in payload.prompt_sections[0].top_regressions}
        assert by_example["ex1"].target_empty_output is True
        assert by_example["ex2"].target_empty_output is False

    def test_top_regression_not_flagged_for_tool_trace_target(self, tmp_path: Path) -> None:
        """A regression whose target call is a pure tool-call turn (empty
        text, tokens spent, no error) must not be badged as empty-output —
        that's the normal shape of a correct agent response.
        """
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        raw_path = run_dir / "raw.jsonl"
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        rewritten = []
        for line in lines:
            call = Call.model_validate_json(line)
            if call.example_id == "ex1" and call.role == "target":
                call = call.model_copy(
                    update={
                        "text": "",
                        "output_tokens": 956,
                        "finish_reason": "stop",
                        "trace": ToolTrace(
                            calls=[
                                ToolCall(
                                    tool_name="search_db",
                                    arguments={"q": "ACME"},
                                    sequence_index=0,
                                ),
                            ],
                        ),
                    },
                )
            rewritten.append(call.model_dump_json())
        raw_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

        payload = build_report_payload(run_dir)
        by_example = {tr.example_id: tr for tr in payload.prompt_sections[0].top_regressions}
        assert by_example["ex1"].target_empty_output is False

    def test_serialised_payload_includes_empty_output_fields(self, tmp_path: Path) -> None:
        from evalshift_cli.reports.json import _to_jsonable

        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_empty",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=956,
                finish_reason="stop",
            ),
        )
        payload = build_report_payload(run_dir)
        data = _to_jsonable(payload)
        section = data["prompt_sections"][0]
        assert section["economics"]["target"]["empty_output_calls"] == 1
        assert "target_empty_output" in section["top_regressions"][0]

    def test_html_shows_empty_output_marker_on_regression(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        raw_path = run_dir / "raw.jsonl"
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        rewritten = []
        for line in lines:
            call = Call.model_validate_json(line)
            if call.example_id == "ex1" and call.role == "target":
                call = call.model_copy(
                    update={"text": "", "output_tokens": 956, "finish_reason": "stop"},
                )
            rewritten.append(call.model_dump_json())
        raw_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

        payload = build_report_payload(run_dir)
        html = render_html(payload)

        assert "empty output" in html.lower()
        assert "thinking-only" in html.lower() or "no visible text" in html.lower()

    def test_html_shows_empty_output_count_in_economics(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex_empty",
                model_id="gemini/gemini-2.5-pro",
                role="target",
                text="",
                output_tokens=956,
                finish_reason="stop",
            ),
        )
        payload = build_report_payload(run_dir)
        html = render_html(payload)

        assert ">Empty output</th>" in html
        assert ">1</td>" in html  # the empty_output_calls count cell


def test_unmeasured_comparison_does_not_render_as_too_few_samples() -> None:
    from evalshift_cli.reports.html import _verdict

    note = f"{UNMEASURED_NOTE_PREFIX} this evaluator scored no comparable pair"
    head, blurb = _verdict("insufficient", [note])
    assert head == "Nothing measured"
    assert "Too few samples" not in blurb
    assert _verdict("insufficient", []) == ("Not enough data", "Too few samples to judge.")
    assert _verdict("insufficient") == ("Not enough data", "Too few samples to judge.")


# ---------------------------------------------------------------------------
# S4 — the two tool-selection axes in the report
# ---------------------------------------------------------------------------


def _scaffold_two_axis_run(tmp_path: Path, *, rounds: bool = False) -> tuple[Path, str]:
    """A run where one evaluator name scored both ``tool_selection`` axes.

    ``rounds=True`` makes ``ex1`` a teacher-forced two-round replay: the
    divergence record then carries the per-round breakdown the evaluator
    writes for a multi-round trace, which is what the report renders one line
    per round from.

    Shaped exactly like the frozen ``project_insights`` fixture and just as
    hostile: every conformance row is ``0.0 / 0.0`` — both models called a
    tool the recording never made — and the divergence rows say the target
    went somewhere the source did not. The suite carries no tool data and the
    calls carry no ``ToolTrace``, so the only place the tool names exist is
    the evaluator's own record metadata. A report that cannot read them there
    shows a wall of numbers naming nothing.
    """
    run_id = "r_20260821_twoaxis"
    run_dir = tmp_path / ".evalshift" / "runs" / run_id

    suite_path = tmp_path / "golden.jsonl"
    suite_path.write_text(
        '{"id": "ex1", "inputs": {"input": "What did I work on?"}, '
        '"tags": ["captured"], "expected_no_tools": true, "tools": []}\n'
        '{"id": "ex2", "inputs": {"input": "Anything new?"}, '
        '"tags": ["captured"], "expected_no_tools": true, "tools": []}\n',
        encoding="utf-8",
    )
    write_state(
        run_dir,
        RunState(
            run_id=run_id,
            status="completed",
            config_hash="x",
            started_at=datetime(2026, 8, 21, tzinfo=UTC),
            models=RunModels(
                source="gemini/gemini-3.5-flash-lite",
                target="gemini/gemini-3.7-flash",
            ),
            prompt_ids=["replay"],
            suite_path=str(suite_path),
            total_evaluations=4,
            completed_evaluations=4,
        ),
    )

    def _two_round_trace(second: str) -> ToolTrace:
        return ToolTrace(
            calls=[
                ToolCall(
                    tool_name="get_recent_files",
                    arguments={},
                    sequence_index=0,
                    round_index=0,
                ),
                ToolCall(tool_name=second, arguments={}, sequence_index=1, round_index=1),
            ],
            round_count=2,
        )

    for example_id in ("ex1", "ex2"):
        for role, model_id in (
            ("source", "gemini/gemini-3.5-flash-lite"),
            ("target", "gemini/gemini-3.7-flash"),
        ):
            append_call(
                run_dir,
                Call(
                    run_id=run_id,
                    prompt_id="replay",
                    example_id=example_id,
                    model_id=model_id,
                    role=role,
                    text="",
                    finish_reason="tool_calls",
                    trace=(
                        _two_round_trace("archive_project" if role == "source" else "display_info")
                        if rounds and example_id == "ex1"
                        else None
                    ),
                ),
            )

    rows = [
        # Both sides missed the recorded ground truth by the same margin.
        EvalRecord(
            run_id=run_id,
            prompt_id="replay",
            example_id=example_id,
            evaluator_name="routing",
            kind="tool_selection.conformance",
            source_score=0.0,
            target_score=0.0,
            delta=0.0,
            metadata={
                "mode": "expected_no_tools",
                "source_calls": 1,
                "target_calls": 1,
                "failure_categories": ["TOOL_GROUND_TRUTH_MISS"],
            },
        )
        for example_id in ("ex1", "ex2")
    ]
    rows += [
        EvalRecord(
            run_id=run_id,
            prompt_id="replay",
            example_id="ex1",
            evaluator_name="routing",
            kind="tool_selection.divergence",
            source_score=1.0,
            target_score=0.0,
            delta=-1.0,
            metadata={
                "mode": "set",
                "source_set": ["get_recent_files"],
                "target_set": ["display_info"],
                "failure_categories": ["TOOL_SELECTION_DRIFT"],
                **(
                    {
                        "rounds": [
                            {
                                "round": 0,
                                "expected_names": [],
                                "source_names": ["get_recent_files"],
                                "target_names": ["get_recent_files"],
                                "source_score": 1.0,
                                "target_score": 1.0,
                            },
                            {
                                "round": 1,
                                "expected_names": [],
                                "source_names": ["archive_project"],
                                "target_names": ["display_info"],
                                "source_score": 1.0,
                                "target_score": 0.0,
                            },
                        ],
                    }
                    if rounds
                    else {}
                ),
            },
        ),
        EvalRecord(
            run_id=run_id,
            prompt_id="replay",
            example_id="ex2",
            evaluator_name="routing",
            kind="tool_selection.divergence",
            source_score=1.0,
            target_score=1.0,
            delta=0.0,
            metadata={
                "mode": "set",
                "source_set": ["get_projects"],
                "target_set": ["get_projects"],
            },
        ),
    ]
    (run_dir / SCORES_FILENAME).write_text(
        "\n".join(r.model_dump_json() for r in rows) + "\n",
        encoding="utf-8",
    )

    def _comparison(**overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "prompt_id": "replay",
            "evaluator_name": "routing",
            "slice_name": "all",
            "n": 2,
            "test": "skipped",
            "statistic": 0.0,
            "p_value": 1.0,
            "p_value_corrected": 1.0,
            "effect_size": 0.0,
            "effect_size_ci_low": 0.0,
            "effect_size_ci_high": 0.0,
            "delta_avg_score": 0.0,
            "severity": "none",
            "notes": [],
        }
        row.update(overrides)
        return row

    analysis = {
        "run_id": run_id,
        "models": {
            "source": "gemini/gemini-3.5-flash-lite",
            "target": "gemini/gemini-3.7-flash",
        },
        "n_examples": 2,
        "n_records": 4,
        "slices": ["all"],
        "aggregates": {},
        "comparisons": [
            _comparison(notes=["axis: tool_selection.conformance"]),
            _comparison(
                notes=["axis: tool_selection.divergence"],
                delta_avg_score=-0.5,
                effect_size=-1.4,
                severity="critical",
            ),
        ],
    }
    (run_dir / ANALYSIS_FILENAME).write_text(json.dumps(analysis), encoding="utf-8")
    return run_dir, run_id


class TestTheTwoToolAxesAreDifferentFindings:
    """A divergence regression and a conformance failure must not share a row.

    Both axes are scored under one user-chosen evaluator name (``routing``),
    so ``(prompt, evaluator, slice)`` no longer identifies a row and the
    report rendered two visually identical lines — one of them a critical
    regression, the other a ✓.
    """

    def _html(self, tmp_path: Path) -> str:
        run_dir, _ = _scaffold_two_axis_run(tmp_path)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        return render_html(payload)

    def test_each_axis_is_named_in_the_evaluator_table(self, tmp_path: Path) -> None:
        html = self._html(tmp_path)
        assert "tool_selection.divergence" in html
        assert "tool_selection.conformance" in html

    def test_the_axes_carry_different_labels(self, tmp_path: Path) -> None:
        """Two rows both reading ``Routing`` is the defect, not the fix."""
        html = self._html(tmp_path)
        assert html.count(">Routing<") == 0, "the bare family name identifies neither axis"

    def test_a_shared_ground_truth_miss_is_not_rendered_as_equivalent(self, tmp_path: Path) -> None:
        """Both models failing the same recorded ground truth is not a ✓.

        The conformance axis scored ``0.0 / 0.0`` on every pair: a zero delta
        the report used to headline as "Equivalent — No meaningful difference
        between models", one line under a critical regression on the same
        evaluator name.
        """
        html = self._html(tmp_path)
        assert "No meaningful difference between models" not in html
        assert "Ground truth missed by both" in html


class TestADivergenceFindingNamesTheTools:
    """``grep get_recent_files report.html`` returned nothing on the real run.

    The tool names are the entire finding — "the target called something
    else" is unactionable without them — and they live in the evaluator's
    record metadata, which nothing rendered.
    """

    def _html(self, tmp_path: Path) -> str:
        run_dir, _ = _scaffold_two_axis_run(tmp_path)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        return render_html(payload)

    def test_the_regression_names_both_sides_tools(self, tmp_path: Path) -> None:
        html = self._html(tmp_path)
        assert "get_recent_files" in html
        assert "display_info" in html

    def test_every_example_shows_its_tool_change_not_only_the_top_five(
        self, tmp_path: Path
    ) -> None:
        """The per-example table is the plan's table — one line per example.

        Only the five worst regressions get a card, so a suite of ten
        divergences renders four of them nowhere. ``ex2`` did not regress at
        all and still has a tool change worth showing.
        """
        run_dir, _ = _scaffold_two_axis_run(tmp_path)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        rows = {r.example_id: r for r in payload.prompt_sections[0].example_rows}
        assert rows["ex1"].tool_change is not None
        assert rows["ex1"].tool_change.source_names == ["get_recent_files"]
        assert rows["ex1"].tool_change.target_names == ["display_info"]
        assert rows["ex2"].tool_change is not None
        assert rows["ex2"].tool_change.source_names == ["get_projects"]
        assert "get_projects" in render_html(payload)

    def test_the_tool_change_reaches_report_json(self, tmp_path: Path) -> None:
        from evalshift_cli.reports.json import _to_jsonable

        run_dir, _ = _scaffold_two_axis_run(tmp_path)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        rows = _to_jsonable(payload)["prompt_sections"][0]["example_rows"]
        by_id = {row["example_id"]: row for row in rows}
        assert by_id["ex1"]["tool_change"] == {
            "source_names": ["get_recent_files"],
            "target_names": ["display_info"],
        }


class TestToolMatchIsSigned:
    """What ``tool_match`` means once one evaluator scores two axes.

    It was ``all(target_score >= 1.0)`` — a single-axis predicate that became
    an AND across two questions with two different baselines the moment
    ``tool_selection`` started emitting two rows. A conformance row both
    models missed at the same height then forced ✗ on a pair whose migration
    changed nothing.
    """

    def test_a_shared_ground_truth_miss_is_not_a_tool_mismatch(self, tmp_path: Path) -> None:
        run_dir, _ = _scaffold_two_axis_run(tmp_path)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        rows = {r.example_id: r for r in payload.prompt_sections[0].example_rows}
        # ex2: conformance 0.0/0.0 (both missed), divergence 1.0/1.0. The
        # target did everything the source did; the suite is what is broken.
        assert rows["ex2"].tool_match is True

    def test_a_target_that_lost_ground_is_a_mismatch(self, tmp_path: Path) -> None:
        run_dir, _ = _scaffold_two_axis_run(tmp_path)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        rows = {r.example_id: r for r in payload.prompt_sections[0].example_rows}
        assert rows["ex1"].tool_match is False


# ---------------------------------------------------------------------------
# The glass report shell — header, hero panels, metric strip
# ---------------------------------------------------------------------------


class TestReportShell:
    """The figures the redesigned header and hero panels derive themselves.

    Everything here is computed in the renderer rather than carried on
    ``ReportData``: ``report.json`` is the payload the server's bundle spec
    describes, and a presentational rollup has no business widening it.
    """

    def _html(self, tmp_path: Path, *, decision: bool = False) -> str:
        cwd, run_id = _scaffold_full_run(tmp_path)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        if decision:
            _write_migration_decision(run_dir, run_id)
        return render_html(build_report_payload(run_dir))

    def test_header_names_the_suite_and_the_model_hop(self, tmp_path: Path) -> None:
        html = self._html(tmp_path)
        assert "suite golden" in html
        assert "gemini/gemini-2.5-flash" in html
        assert "gemini/gemini-2.5-pro" in html
        assert "2 examples" in html
        assert "4 calls" in html

    def test_header_timestamp_is_human_readable(self) -> None:
        from evalshift_cli.reports.html import _display_timestamp

        assert _display_timestamp("2026-08-23T15:01:56.674222+00:00") == "2026-08-23 15:01:56 UTC"
        # An offset other than UTC is converted, not relabelled.
        assert _display_timestamp("2026-08-23T17:01:56+02:00") == "2026-08-23 15:01:56 UTC"
        # Naive input is taken as already-UTC rather than shifted by the host's zone.
        assert _display_timestamp("2026-08-23T15:01:56") == "2026-08-23 15:01:56 UTC"
        # A malformed stamp is shown, not swallowed.
        assert _display_timestamp("not a date") == "not a date"

    def test_suite_pill_prefers_the_suite_directory_name(self) -> None:
        from evalshift_cli.reports.html import _suite_name

        assert _suite_name("/tmp/p/.evalshift/suites/main_chat/golden.jsonl") == "main_chat"
        assert _suite_name("/tmp/golden.jsonl") == "golden"
        assert _suite_name("") == ""

    def test_metric_strip_carries_the_run_level_deltas(self, tmp_path: Path) -> None:
        # Source 2 x $0.0001 @ 100 ms, target 2 x $0.0002 @ 120 ms.
        html = self._html(tmp_path)
        assert "+100.0%" in html  # cost delta
        assert "+20.0%" in html  # latency delta
        assert "Failed / truncated" in html
        assert "Avg score" in html

    def test_run_deltas_are_none_when_the_source_side_measured_nothing(self) -> None:
        from evalshift_cli.reports.html import _pct_delta

        assert _pct_delta(0.0, 1.0) is None
        assert _pct_delta(1.0, 2.0) == pytest.approx(100.0)
        assert _pct_delta(2.0, 1.0) == pytest.approx(-50.0)

    def test_run_latency_delta_is_none_when_either_side_measured_nothing_live(self) -> None:
        # A role whose every call was served (wholly or partly) from the cache
        # has no live latency sample; its 0.0 mean is "unmeasured", so the
        # header must not read it as a -100% latency change.
        from evalshift_cli.reports.html import _run_latency_delta_pct

        def totals(src_live: float, tgt_live: float) -> dict[str, dict[str, float]]:
            return {
                "source": {"latency_ms_avg": 100.0 if src_live else 0.0, "live_calls": src_live},
                "target": {"latency_ms_avg": 150.0 if tgt_live else 0.0, "live_calls": tgt_live},
            }

        assert _run_latency_delta_pct(totals(2, 0)) is None
        assert _run_latency_delta_pct(totals(0, 2)) is None
        assert _run_latency_delta_pct(totals(2, 2)) == pytest.approx(50.0)

    def test_verdict_panel_shows_the_outcome_split(self, tmp_path: Path) -> None:
        html = self._html(tmp_path, decision=True)
        assert "Equivalent 0.0%" in html
        assert "Regressed 100.0%" in html
        assert "Top regression causes" in html

    def test_budget_tally_ignores_gates_that_measured_nothing(self) -> None:
        """A ``0/0`` budget renders as a clean row and measures nothing.

        Counting it as passed is the bug commit 4a83c40 fixed for the
        narrative; the verdict panel must not reintroduce it.
        """
        from evalshift_cli.reports.html import _budget_tally

        budgets = [
            {"name": "a", "passed": True, "conclusive": True},
            {"name": "b", "passed": True, "conclusive": False},  # blind gate
            {"name": "c", "passed": False, "conclusive": True},
        ]
        assert _budget_tally(budgets) == (1, 3, 1)


class TestReportResolvesPerSuiteEvaluators:
    """The tool-match column must key on *this suite's* tool evaluators.

    ``report`` runs long after the CLI invocation that chose the suite, so it
    re-resolves the set from the run's recorded ``suite_name``. Reading the
    top-level block instead would leave a tool-calling suite's rows looking
    like they had no tool evaluators at all.
    """

    _CONFIG = """
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        evaluators:
          structural:
            - type: length
              min_chars: 1
        suites:
          main_chat:
            path: ./golden.jsonl
            evaluators:
              tool_selection:
                - name: routing
    """

    def _captured_names(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        suite_name: str | None,
    ) -> frozenset[str]:
        from evalshift_cli.cli.commands import report as report_module
        from evalshift_cli.runner.checkpoint import read_state

        root, run_id = _scaffold_full_run(tmp_path)
        config_path = root / "evalshift.yaml"
        config_path.write_text(self._CONFIG, encoding="utf-8")
        runs_base = root / ".evalshift" / "runs"
        run_dir = runs_base / run_id
        state = read_state(run_dir)
        write_state(run_dir, state.model_copy(update={"suite_name": suite_name}))

        seen: dict[str, frozenset[str]] = {}
        real = report_module.build_report_payload

        def _spy(run_dir: Path, **kwargs: frozenset[str]) -> object:
            seen["names"] = kwargs["tool_evaluator_names"]
            return real(run_dir, **kwargs)

        monkeypatch.setattr(report_module, "build_report_payload", _spy)
        report_module.run_report(
            run_id=run_id,
            config_path=config_path,
            runs_base=runs_base,
            insights=False,
        )
        return seen["names"]

    def test_uses_the_suites_own_tool_evaluators(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        names = self._captured_names(monkeypatch, tmp_path, suite_name="main_chat")
        assert names == frozenset({"routing"})

    def test_falls_back_to_the_top_level_for_a_raw_suite_path_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        names = self._captured_names(monkeypatch, tmp_path, suite_name=None)
        assert names == frozenset()


class TestDroppedParamsBanner:
    """The report must say when the replay's constraints did not reach a model.

    ``drop_params=True`` keeps the call alive at the cost of the constraint.
    Without this banner the arm looks like a clean model swap when it is a
    model swap plus a missing constraint.
    """

    def test_absent_when_every_constraint_was_honoured(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        assert payload.dropped_params == {}
        assert "Constraints not honoured" not in render_html(payload)

    def test_payload_carries_the_run_state_record(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(
            tmp_path, dropped_params={"gemini/gemini-2.5-pro": ["response_format", "tool_choice"]}
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        assert payload.dropped_params == {
            "gemini/gemini-2.5-pro": ["response_format", "tool_choice"]
        }

    def test_report_json_emits_the_field(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(
            tmp_path, dropped_params={"gemini/gemini-2.5-pro": ["tool_choice"]}
        )
        run_dir = cwd / ".evalshift" / "runs" / run_id
        payload = build_report_payload(run_dir)
        out = write_report_json(payload, run_dir)

        assert json.loads(out.read_text())["dropped_params"] == {
            "gemini/gemini-2.5-pro": ["tool_choice"]
        }

    def test_banner_names_the_target_and_its_params(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(
            tmp_path, dropped_params={"gemini/gemini-2.5-pro": ["response_format", "tool_choice"]}
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        assert "Constraints not honoured by the target" in html
        assert "<code>response_format</code>" in html
        assert "<code>tool_choice</code>" in html
        # Above the numbers it qualifies, next to the determinism banner.
        assert html.index("Constraints not honoured") < html.index("<h2>")

    def test_banner_names_the_source_when_that_is_the_affected_arm(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(
            tmp_path, dropped_params={"gemini/gemini-2.5-flash": ["tool_choice"]}
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        assert "Constraints not honoured by <code>gemini/gemini-2.5-flash</code>" in html

    def test_banner_lists_every_affected_model(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(
            tmp_path,
            dropped_params={
                "gemini/gemini-2.5-flash": ["tool_choice"],
                "gemini/gemini-2.5-pro": ["response_format"],
            },
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        assert "Constraints not honoured by both arms" in html
        assert "<code>gemini/gemini-2.5-flash</code>" in html
        assert "<code>gemini/gemini-2.5-pro</code>" in html

    def test_banner_follows_the_determinism_banner(self, tmp_path: Path) -> None:
        """Sampling weakens every p-value; a dropped constraint weakens one arm."""
        cwd, run_id = _scaffold_full_run(
            tmp_path,
            non_deterministic_models=["gemini/gemini-2.5-pro"],
            dropped_params={"gemini/gemini-2.5-pro": ["tool_choice"]},
        )
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        html = render_html(payload)

        assert html.index("Sampling is not controlled") < html.index("Constraints not honoured")


class TestPerRoundToolChange:
    """A teacher-forced replay diverged in *a round*, and the report says which.

    One flattened arrow — ``a, b → a, c`` — cannot say whether the target
    called the wrong tool or called the right ones in the wrong order across
    the loop. The per-round rows are the finding.
    """

    def test_the_rounds_are_read_off_the_record_metadata(self) -> None:
        from evalshift_cli.reports.json import _tool_change

        change = _tool_change(
            {
                "mode": "set",
                "source_set": ["a", "b"],
                "target_set": ["a", "c"],
                "rounds": [
                    {
                        "round": 0,
                        "expected_names": ["a"],
                        "source_names": ["a"],
                        "target_names": ["a"],
                        "source_score": 1.0,
                        "target_score": 1.0,
                    },
                    {
                        "round": 1,
                        "expected_names": ["b"],
                        "source_names": ["b"],
                        "target_names": ["c"],
                        "source_score": 1.0,
                        "target_score": 0.0,
                    },
                ],
            },
        )

        assert change is not None
        assert change.rounds is not None
        assert [r.round for r in change.rounds] == [0, 1]
        assert change.rounds[0].diverged is False
        assert change.rounds[1].diverged is True
        assert change.rounds[1].expected_names == ["b"]

    def test_a_single_round_record_carries_no_rounds(self) -> None:
        from evalshift_cli.reports.json import _tool_change, _tool_change_to_dict

        change = _tool_change({"source_names": ["a"], "target_names": ["b"]})
        assert change is not None
        assert change.rounds is None
        assert _tool_change_to_dict(change) == {
            "source_names": ["a"],
            "target_names": ["b"],
        }

    def test_report_json_carries_the_rounds(self, tmp_path: Path) -> None:
        from evalshift_cli.reports.json import _to_jsonable

        run_dir, _ = _scaffold_two_axis_run(tmp_path, rounds=True)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        rows = _to_jsonable(payload)["prompt_sections"][0]["example_rows"]
        by_id = {row["example_id"]: row for row in rows}
        assert by_id["ex1"]["tool_change"]["rounds"] == [
            {
                "round": 0,
                "source_names": ["get_recent_files"],
                "target_names": ["get_recent_files"],
                "expected_names": [],
                "diverged": False,
            },
            {
                "round": 1,
                "source_names": ["archive_project"],
                "target_names": ["display_info"],
                "expected_names": [],
                "diverged": True,
            },
        ]
        # ex2 is single-round and must be untouched.
        assert by_id["ex2"]["tool_change"] == {
            "source_names": ["get_projects"],
            "target_names": ["get_projects"],
        }

    def test_the_html_renders_one_line_per_round(self, tmp_path: Path) -> None:
        run_dir, _ = _scaffold_two_axis_run(tmp_path, rounds=True)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        html = render_html(payload)
        assert "archive_project" in html
        assert "r2" in html

    def test_the_raw_trace_listing_labels_its_rounds(self, tmp_path: Path) -> None:
        run_dir, _ = _scaffold_two_axis_run(tmp_path, rounds=True)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        html = render_html(payload)
        assert "round 2" in html

    def test_the_tool_diffs_name_the_round_that_diverged(self, tmp_path: Path) -> None:
        run_dir, _ = _scaffold_two_axis_run(tmp_path, rounds=True)
        payload = build_report_payload(run_dir, tool_evaluator_names=frozenset({"routing"}))
        messages = [d.message for d in payload.prompt_sections[0].top_regressions[0].tool_diffs]
        assert messages == [
            "Round 2: Position 1: source called archive_project, target called display_info.",
        ]


class TestPerRoundToolDiffs:
    """Positions restart each round, so the diff must too."""

    def test_positions_reset_and_messages_name_the_round(self) -> None:
        from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
        from evalshift_cli.reports.json import _build_tool_diffs

        def _multi(*rounds: list[str]) -> ToolTrace:
            calls: list[ToolCall] = []
            for round_index, names in enumerate(rounds):
                for name in names:
                    calls.append(
                        ToolCall(
                            tool_name=name,
                            arguments={},
                            sequence_index=len(calls),
                            round_index=round_index,
                        ),
                    )
            return ToolTrace(calls=calls, round_count=len(rounds))

        diffs = _build_tool_diffs(_multi(["a"], ["b"]), _multi(["a"], ["c"]))

        assert [d.kind for d in diffs] == ["tool_order_or_selection"]
        assert diffs[0].message == "Round 2: Position 1: source called b, target called c."

    def test_a_single_round_diff_message_is_unchanged(self) -> None:
        from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
        from evalshift_cli.reports.json import _build_tool_diffs

        source = ToolTrace(calls=[ToolCall(tool_name="a", arguments={}, sequence_index=0)])
        target = ToolTrace(calls=[ToolCall(tool_name="b", arguments={}, sequence_index=0)])

        diffs = _build_tool_diffs(source, target)

        assert diffs[0].message == "Position 1: source called a, target called b."


class TestSamplesPerExampleInReport:
    """``samples_per_example`` (Task 7.1)."""

    def test_single_sample_run_shows_no_samples_pill(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)
        assert payload.samples_per_example == 1
        assert "samples per example" not in render_html(payload)

    def test_repeated_sampling_run_shows_the_pill_and_keeps_one_row_per_example(
        self, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path, samples_per_example=3)
        payload = build_report_payload(cwd / ".evalshift" / "runs" / run_id)

        assert payload.samples_per_example == 3
        assert payload.n_examples == 2
        assert payload.n_calls == 12
        html = render_html(payload)
        assert "3 samples per example" in html
        # Example rows show sample 0's output, once per example.
        (section,) = payload.prompt_sections
        assert len(section.example_rows) == 2
        assert "[sample 1]" not in html
        assert "[sample 2]" not in html

    def test_report_json_carries_the_sample_count(self, tmp_path: Path) -> None:
        cwd, run_id = _scaffold_full_run(tmp_path, samples_per_example=2)
        run_dir = cwd / ".evalshift" / "runs" / run_id
        payload = build_report_payload(run_dir)
        write_report_json(payload, run_dir)
        data = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert data["samples_per_example"] == 2

    def test_determinism_banner_suggests_repeated_sampling_only_when_n_is_one(
        self, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold_full_run(
            tmp_path, non_deterministic_models=["gemini/gemini-2.5-pro"]
        )
        html = render_html(build_report_payload(cwd / ".evalshift" / "runs" / run_id))
        assert "samples_per_example" in html

        again = tmp_path / "again"
        again.mkdir()
        cwd, run_id = _scaffold_full_run(
            again,
            non_deterministic_models=["gemini/gemini-2.5-pro"],
            samples_per_example=2,
        )
        html = render_html(build_report_payload(cwd / ".evalshift" / "runs" / run_id))
        assert "Sampling is not controlled" in html
        assert "samples_per_example" not in html


def test_example_row_latency_is_incomparable_when_a_side_replayed_rounds() -> None:
    from evalshift_cli.reports.json import _build_example_rows
    from evalshift_cli.runner.models import Call

    def call(role: str, example_id: str, latency: int, cached_rounds: int = 0) -> Call:
        return Call(
            run_id="r_20260601_abc123",
            prompt_id="p",
            example_id=example_id,
            model_id="m",
            role=role,  # type: ignore[arg-type]
            latency_ms=latency,
            cached_rounds=cached_rounds,
        )

    rows = _build_example_rows(
        prompt_id="p",
        calls=[
            call("source", "live", 100),
            call("target", "live", 150),
            call("source", "mixed", 100, cached_rounds=1),
            call("target", "mixed", 150),
        ],
        records=[],
        tags_by_example_id={},
        tool_evaluator_names=frozenset(),
        examples_by_id={},
    )
    by_id = {r.example_id: r for r in rows}
    assert by_id["live"].latency_comparable
    assert by_id["live"].delta_latency_ms == 50
    assert not by_id["mixed"].latency_comparable
    assert by_id["mixed"].delta_latency_ms == 0


def _run_dir_with_scores(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    """A completed run (``_scaffold_full_run``) plus extra ``scores.jsonl`` rows."""
    cwd, run_id = _scaffold_full_run(tmp_path)
    run_dir = cwd / ".evalshift" / "runs" / run_id
    with (run_dir / SCORES_FILENAME).open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(EvalRecord.model_validate(record).model_dump_json() + "\n")
    return run_dir


class TestInvariantViolationsPanel:
    def _record(self, example_id: str, *, source_broke: bool) -> dict[str, Any]:
        violation = {
            "rule_id": "no-v1",
            "rule_type": "forbidden",
            "tool": "refund_v1",
            "round_index": 0,
            "detail": "called forbidden tool 'refund_v1'",
        }
        return {
            "run_id": "r1",
            "prompt_id": "p",
            "example_id": example_id,
            "evaluator_name": "payments",
            "kind": "trace_invariants",
            "source_score": 0.0 if source_broke else 1.0,
            "target_score": 0.0,
            "delta": 0.0 if source_broke else -1.0,
            "metadata": {
                "owner": "@payments",
                "rules_checked": ["no-v1"],
                "source_violations": [violation] if source_broke else [],
                "target_violations": [violation],
                "failure_categories": ["INVARIANT_VIOLATION"],
            },
        }

    def test_payload_lists_every_target_violation_even_at_zero_delta(self, tmp_path: Path) -> None:
        run_dir = _run_dir_with_scores(
            tmp_path,
            [self._record("e1", source_broke=True), self._record("e2", source_broke=False)],
        )
        report = build_report_payload(run_dir)
        rows = [
            (v.example_id, v.rule_id, v.source_also, v.owner) for v in report.invariant_violations
        ]
        assert rows == [("e1", "no-v1", True, "@payments"), ("e2", "no-v1", False, "@payments")]

    def test_html_renders_the_panel_with_owner_and_detail(self, tmp_path: Path) -> None:
        run_dir = _run_dir_with_scores(tmp_path, [self._record("e1", source_broke=False)])
        html = render_html(build_report_payload(run_dir))
        assert "Trace rules broken" in html
        assert "no-v1" in html and "@payments" in html
        assert "called forbidden tool" in html

    def test_no_panel_without_trace_rules(self, tmp_path: Path) -> None:
        run_dir = _run_dir_with_scores(tmp_path, [])
        assert "Trace rules broken" not in render_html(build_report_payload(run_dir))

    def test_report_json_carries_the_violations(self, tmp_path: Path) -> None:
        run_dir = _run_dir_with_scores(tmp_path, [self._record("e1", source_broke=False)])
        write_report_json(build_report_payload(run_dir), run_dir)
        data = json.loads((run_dir / REPORT_JSON_FILENAME).read_text(encoding="utf-8"))
        (row,) = data["invariant_violations"]
        assert row["rule_id"] == "no-v1"
        assert row["owner"] == "@payments"
        assert row["source_also"] is False
        assert row["sample"] is None

    def test_report_json_key_is_empty_without_trace_rules(self, tmp_path: Path) -> None:
        run_dir = _run_dir_with_scores(tmp_path, [])
        write_report_json(build_report_payload(run_dir), run_dir)
        data = json.loads((run_dir / REPORT_JSON_FILENAME).read_text(encoding="utf-8"))
        assert data["invariant_violations"] == []

    def test_a_reduced_row_lists_each_samples_violation_once(self, tmp_path: Path) -> None:
        # samples_per_example > 1: evaluate merges every scored sample's
        # violations onto the reduced row, each tagged with its ``sample``.
        record = self._record("e1", source_broke=False)
        (violation,) = record["metadata"]["target_violations"]
        record["metadata"]["target_violations"] = [
            {**violation, "sample": 2},
            {**violation, "sample": 0},
        ]
        run_dir = _run_dir_with_scores(tmp_path, [record])
        report = build_report_payload(run_dir)
        assert [(v.rule_id, v.sample) for v in report.invariant_violations] == [
            ("no-v1", 0),
            ("no-v1", 2),
        ]
        html = render_html(report)
        assert "sample 1" in html and "sample 3" in html

    def test_an_errored_row_lists_nothing(self, tmp_path: Path) -> None:
        record = self._record("e1", source_broke=False)
        record["error"] = "trace missing"
        run_dir = _run_dir_with_scores(tmp_path, [record])
        assert build_report_payload(run_dir).invariant_violations == []

    def test_rows_carry_the_records_blocking_flag(self, tmp_path: Path) -> None:
        advisory = {**self._record("e2", source_broke=False), "blocking": False}
        run_dir = _run_dir_with_scores(tmp_path, [self._record("e1", source_broke=False), advisory])
        report = build_report_payload(run_dir)
        assert [(v.example_id, v.blocking) for v in report.invariant_violations] == [
            ("e1", True),
            ("e2", False),
        ]
        write_report_json(report, run_dir)
        data = json.loads((run_dir / REPORT_JSON_FILENAME).read_text(encoding="utf-8"))
        assert [row["blocking"] for row in data["invariant_violations"]] == [True, False]

    def test_html_tags_an_advisory_violation_without_bad_styling(self, tmp_path: Path) -> None:
        advisory = {**self._record("e1", source_broke=False), "blocking": False}
        html = render_html(build_report_payload(_run_dir_with_scores(tmp_path, [advisory])))
        tag = re.search(r'<span class="([^"]*advisory-tag[^"]*)"[^>]*>advisory</span>', html)
        assert tag is not None
        assert "badge-bad" not in tag.group(1)

    def test_html_leaves_a_blocking_violation_untagged(self, tmp_path: Path) -> None:
        run_dir = _run_dir_with_scores(tmp_path, [self._record("e1", source_broke=False)])
        html = render_html(build_report_payload(run_dir))
        assert re.search(r'class="[^"]*advisory-tag', html) is None

    def test_family_label_reads_as_words(self) -> None:
        from evalshift_cli.reports.html import _kind_label

        assert _kind_label("trace_invariants") == "Trace rules"


def _append_comparisons(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Add aggregate comparisons to the scaffold's ``analysis.json``."""
    path = run_dir / ANALYSIS_FILENAME
    analysis = json.loads(path.read_text(encoding="utf-8"))
    analysis["comparisons"].extend(rows)
    path.write_text(json.dumps(analysis), encoding="utf-8")


class TestAdvisoryEvaluatorsInSummary:
    """A ``blocking: false`` evaluator never speaks for a prompt that has a gate."""

    @staticmethod
    def _comparison(evaluator_name: str, delta: float, **overrides: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "prompt_id": "greet",
            "evaluator_name": evaluator_name,
            "kind": "trace_invariants",
            "slice_name": "all",
            "n": 2,
            "test": "wilcoxon",
            "statistic": 0.0,
            "p_value": 1.0,
            "p_value_corrected": 1.0,
            "effect_size": 0.3 if delta > 0 else -0.3 if delta < 0 else 0.0,
            "effect_size_ci_low": -0.6,
            "effect_size_ci_high": 0.6,
            "delta_avg_score": delta,
            "severity": "none",
            "notes": [],
        }
        row.update(overrides)
        return row

    @staticmethod
    def _record(
        evaluator_name: str, delta: float, *, blocking: bool, prompt_id: str = "greet"
    ) -> dict[str, Any]:
        return {
            "run_id": "r_20260601_aaaaaa",
            "prompt_id": prompt_id,
            "example_id": "ex1",
            "evaluator_name": evaluator_name,
            "kind": "trace_invariants",
            "source_score": 1.0 if delta <= 0 else 0.0,
            "target_score": 1.0 if delta >= 0 else 0.0,
            "delta": delta,
            "blocking": blocking,
        }

    def _run(
        self, tmp_path: Path, records: list[dict[str, Any]], comparisons: list[dict[str, Any]]
    ) -> Path:
        run_dir = _run_dir_with_scores(tmp_path, records)
        _append_comparisons(run_dir, comparisons)
        return run_dir

    def _mixed_run(self, tmp_path: Path) -> Path:
        # The advisory row is listed first, so it won every severity tie.
        return self._run(
            tmp_path,
            [
                self._record("notes_watch", 1.0, blocking=False),
                self._record("payments", -1.0, blocking=True),
            ],
            [self._comparison("notes_watch", 0.5), self._comparison("payments", -0.5)],
        )

    def test_executive_summary_prefers_a_blocking_evaluator_over_an_advisory_one(
        self, tmp_path: Path
    ) -> None:
        rows = build_report_payload(self._mixed_run(tmp_path)).executive_summary
        assert [(r["prompt_id"], r["evaluator_name"]) for r in rows] == [("greet", "payments")]

    def test_executive_summary_breaks_a_severity_tie_toward_the_worst_effect(
        self, tmp_path: Path
    ) -> None:
        run_dir = self._run(
            tmp_path,
            [
                self._record("a_improved", 1.0, blocking=True),
                self._record("b_regressed", -1.0, blocking=True),
            ],
            [self._comparison("a_improved", 0.5), self._comparison("b_regressed", -0.5)],
        )
        rows = build_report_payload(run_dir).executive_summary
        assert [r["evaluator_name"] for r in rows] == ["b_regressed"]

    def test_executive_summary_keeps_a_prompt_whose_only_evaluator_is_advisory(
        self, tmp_path: Path
    ) -> None:
        run_dir = self._run(
            tmp_path,
            [self._record("notes_watch", 1.0, blocking=False, prompt_id="side")],
            [self._comparison("notes_watch", 0.5, prompt_id="side")],
        )
        rows = build_report_payload(run_dir).executive_summary
        assert [(r["prompt_id"], r["evaluator_name"]) for r in rows] == [
            ("greet", "structural.length"),
            ("side", "notes_watch"),
        ]

    def test_header_average_follows_the_blocking_row(self, tmp_path: Path) -> None:
        html = render_html(build_report_payload(self._mixed_run(tmp_path)))
        shown = re.search(
            r'Avg score Δ</span>\s*<span class="strip-value[^"]*">([^<]+)</span>', html
        )
        assert shown is not None
        assert shown.group(1) == "-0.500"

    def test_evaluator_table_tags_an_advisory_evaluator(self, tmp_path: Path) -> None:
        html = render_html(build_report_payload(self._mixed_run(tmp_path)))
        tagged = r'<code class="eval-id">{}</code>\s*<span class="badge advisory-tag"'
        assert re.search(tagged.format("notes_watch"), html) is not None
        assert re.search(tagged.format("payments"), html) is None

    def test_evaluator_table_says_what_a_trace_rule_row_measures(self, tmp_path: Path) -> None:
        run_dir = self._run(
            tmp_path,
            [self._record("payments", 0.0, blocking=True)],
            [self._comparison("payments", 0.0)],
        )
        html = render_html(build_report_payload(run_dir))
        assert "each side judged against the entry" in html
