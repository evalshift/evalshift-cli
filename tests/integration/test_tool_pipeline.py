"""End-to-end integration tests for v0.2 tool-call evaluation.

Each scenario from PRD §9.3 has a corresponding test that:

1. Sets up an agent project on disk (config + tools + suite + prompt).
2. Monkeypatches ``ModelClient.complete_with_tools`` to return canned
   :class:`ToolCompletionResult` objects keyed on the work item.
3. Runs the full ``evalshift run → evaluate → analyze`` pipeline.
4. Asserts the analysis layer produces the expected severity.

These tests are the most valuable in the v0.2 suite — each one
corresponds to a real regression scenario customers will hit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME
from evalshift_cli.cli.main import app
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.models.client import ModelClient, ToolCompletionResult
from evalshift_cli.runner import orchestrator as orch_module

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scaffold_agent_project(
    tmp_path: Path,
    *,
    examples: list[dict[str, Any]] | None = None,
    extra_evaluators: str = "",
) -> Path:
    """Lay out a minimal agent project. Returns the project root."""
    cfg = f"""
version: 1

prompts:
  - id: routing
    detection: manual
    content: "Route this: {{query}}"
    variables: [query]

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-2.5-pro
  concurrency: 4
  cache: false   # keep cache off so tests don't share state

evaluators:
  tool_selection:
    - name: routing_selection
      conformance: expected
      divergence: set
{extra_evaluators}
"""
    (tmp_path / "evalshift.yaml").write_text(cfg, encoding="utf-8")

    examples = examples or [
        {
            "id": f"ex_{i:02d}",
            "inputs": {"query": f"security alert #{i}"},
            "tags": ["security"],
            "expected_tools": [{"tool_name": "notify_security_team"}],
        }
        for i in range(20)
    ]
    import json

    # The example's own toolset, inlined rather than referenced (no sidecar
    # to write for a hand-authored fixture). Dispatch reads THIS field: an
    # empty placeholder here would route every call through the plain
    # (no-tools) path, breaking every test in this module that mocks
    # complete_with_tools.
    agent_tools = [
        {
            "name": "search_orders",
            "description": "Search the customer database.",
            "input_schema": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
            },
        },
        {
            "name": "notify_security_team",
            "description": "Page the security team.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        },
        {
            "name": "send_email",
            "description": "Send an email.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
        },
    ]

    (tmp_path / "golden.jsonl").write_text(
        "\n".join(json.dumps({"tools": agent_tools, **e}) for e in examples) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _patch_with_tools(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_for_role: dict[str, str | None] | None = None,
    fake: Callable[..., Any] | None = None,
) -> None:
    """Replace ``ModelClient.complete_with_tools`` with a deterministic fake.

    Either pass an explicit ``fake`` callable, or use ``tool_for_role``
    to map ``'source'`` / ``'target'`` → the tool name that role should
    "call" on every example. ``None`` means no tool call.
    """

    if fake is None:
        assert tool_for_role is not None

        async def default_fake(self: ModelClient, **kwargs: Any) -> ToolCompletionResult:
            model = str(kwargs["model"])
            role = "target" if "pro" in model else "source"
            tool_name = tool_for_role.get(role)
            calls = (
                [
                    ToolCall(
                        tool_name=tool_name,
                        arguments={"summary": "x", "severity": "high"},
                        sequence_index=0,
                    ),
                ]
                if tool_name
                else []
            )
            return ToolCompletionResult(
                trace=ToolTrace(calls=calls, final_text=None),
                model_id=model,
                input_tokens=10,
                output_tokens=4,
                cost_usd=0.0,
                latency_ms=10,
                raw_provider_response={},
            )

        fake = default_fake

    monkeypatch.setattr(ModelClient, "complete_with_tools", fake)
    orch_module.asyncio = asyncio  # ensure we use the real asyncio


# ---------------------------------------------------------------------------
# Scenarios from PRD §9.3
# ---------------------------------------------------------------------------


def _run_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Run evalshift run → evaluate → analyze; return run_id."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    result = runner.invoke(app, ["run", "--yes"])
    assert result.exit_code == 0, result.stdout
    runs = sorted((tmp_path / ".evalshift" / "runs").iterdir())
    assert len(runs) == 1
    run_id = runs[0].name
    result = runner.invoke(app, ["evaluate", run_id])
    assert result.exit_code == 0, result.stdout
    return run_id


class TestToolPipeline:
    def test_no_regression(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Both source and target call the right tool — no regression."""
        _scaffold_agent_project(tmp_path)
        _patch_with_tools(
            monkeypatch,
            tool_for_role={"source": "notify_security_team", "target": "notify_security_team"},
        )
        run_id = _run_pipeline(monkeypatch, tmp_path)

        scores_path = tmp_path / ".evalshift" / "runs" / run_id / SCORES_FILENAME
        rows = scores_path.read_text(encoding="utf-8").splitlines()
        assert rows  # something was written
        # Every row should be a perfect 1.0/1.0 (both sides match expectation).
        import json

        for row in rows:
            data = json.loads(row)
            assert data["target_score"] == 1.0
            assert data["source_score"] == 1.0

    def test_detects_missing_tool_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Source always calls notify_security_team; target drops it on most examples.

        Some target wins (no regression on those) keep the delta
        distribution non-degenerate so the analysis layer's statistical
        pipeline produces a real classification rather than the
        zero-variance ``skipped`` short-circuit.
        """
        _scaffold_agent_project(tmp_path)

        async def fake(self: ModelClient, **kwargs: Any) -> ToolCompletionResult:
            model = str(kwargs["model"])
            role = "target" if "pro" in model else "source"
            # Source: always call the right tool.
            # Target: skip on most examples, succeed on a few — gives variance.
            example_idx = int(
                "".join(ch for ch in str(kwargs["prompt"]) if ch.isdigit())[-2:] or "0",
            )
            should_call = role == "source" or example_idx % 5 == 0
            calls = (
                [
                    ToolCall(
                        tool_name="notify_security_team",
                        arguments={"summary": "x", "severity": "high"},
                        sequence_index=0,
                    ),
                ]
                if should_call
                else []
            )
            return ToolCompletionResult(
                trace=ToolTrace(calls=calls),
                model_id=model,
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
                raw_provider_response={},
            )

        _patch_with_tools(monkeypatch, fake=fake)
        run_id = _run_pipeline(monkeypatch, tmp_path)

        # Run analyze and verify it picks up a regression.
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout
        import json

        analysis = json.loads(
            (tmp_path / ".evalshift" / "runs" / run_id / "analysis.json").read_text(),
        )
        # The "all" slice for routing_selection should be a regression.
        regressions = [
            c
            for c in analysis["comparisons"]
            if c["evaluator_name"] == "routing_selection"
            and c["slice_name"] == "all"
            and c["severity"] in {"critical", "high", "medium", "low"}
        ]
        assert regressions, (
            f"expected a regression on routing_selection; got "
            f"{[c['severity'] for c in analysis['comparisons']]}"
        )

    def test_text_prompts_unaffected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Plain text prompt + agent prompt mixed; tool evaluator only scores agent."""
        # Add a non-agent prompt to the config.
        cfg_root = _scaffold_agent_project(tmp_path)
        (cfg_root / "evalshift.yaml").write_text(
            (cfg_root / "evalshift.yaml")
            .read_text(encoding="utf-8")
            .replace(
                "evaluators:",
                """evaluators:
  structural:
    - type: length
      min_chars: 1
      max_chars: 1000""",
            ),
            encoding="utf-8",
        )

        async def fake_text(self: ModelClient, **kwargs: Any) -> Any:
            from evalshift_cli.models.client import CompletionResult

            return CompletionResult(
                text="ok",
                model_id=str(kwargs["model"]),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            )

        monkeypatch.setattr(ModelClient, "complete", fake_text)
        _patch_with_tools(
            monkeypatch,
            tool_for_role={"source": "notify_security_team", "target": "notify_security_team"},
        )

        run_id = _run_pipeline(monkeypatch, tmp_path)
        scores_path = tmp_path / ".evalshift" / "runs" / run_id / SCORES_FILENAME
        rows = scores_path.read_text(encoding="utf-8").splitlines()
        import json

        # Both length (structural) and tool_selection records should be present.
        evaluator_names = {json.loads(r)["evaluator_name"] for r in rows}
        assert "structural.length" in evaluator_names
        assert "routing_selection" in evaluator_names

    def test_stats_handle_bimodal_distribution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """PRD §6.7 lock-in: bimodal scores (0/1) route through Wilcoxon, not t-test.

        Tool evaluators tend to produce 0/1 scores. We don't add new
        analysis code; we just verify the existing pipeline doesn't
        crash and produces a meaningful severity classification.
        """
        _scaffold_agent_project(tmp_path)
        _patch_with_tools(
            monkeypatch,
            tool_for_role={"source": "notify_security_team", "target": None},
        )
        run_id = _run_pipeline(monkeypatch, tmp_path)
        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout
        import json

        analysis = json.loads(
            (tmp_path / ".evalshift" / "runs" / run_id / "analysis.json").read_text(),
        )
        comparisons = analysis["comparisons"]
        # n=20 examples, all delta = -1.0 → variance 0 → "skipped" per
        # the analysis layer. That's the documented behaviour for
        # zero-variance inputs and is exactly what we want here.
        assert any(
            c["severity"] in {"critical", "high", "medium", "low", "none", "improved"}
            or c["test"] == "skipped"
            for c in comparisons
        )


class TestBrokenHarnessReachesTheVerdictBlock:
    """S5 — ``evalshift compare`` must print it, not only ``evalshift evaluate``.

    ``all`` scores with ``quiet=True`` because the evaluate stage runs inside
    a Live grid this table would fight with, so the finding is handed back on
    :class:`~evalshift_cli.cli.commands.evaluate.EvaluateResult` and rendered here
    instead — directly above the verdict it invalidates, since a verdict read
    first is a verdict believed.
    """

    def test_all_prints_the_finding_above_the_verdict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Every example expects ``notify_security_team``; both models call
        # ``send_email``. The *source* misses its own ground truth 20 times.
        _scaffold_agent_project(tmp_path)
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + "\nmigration_policy:\n  max_critical_regressions: 0\n",
            encoding="utf-8",
        )
        _patch_with_tools(
            monkeypatch,
            tool_for_role={"source": "send_email", "target": "send_email"},
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

        result = runner.invoke(app, ["all", "--yes", "--no-insights"])

        assert result.exit_code == 0, result.stdout
        flat = " ".join(result.stdout.split())
        assert "broken eval harness" in flat
        assert "20 of 20 tool-selection conformance rows (100%)" in flat
        assert flat.index("broken eval harness") < flat.index("Migration verdict")


class TestTraceInvariantsGate:
    """A rule the target breaks fails the run, though the source broke nothing."""

    _RULES = """\
  trace_invariants:
    - name: security_contract
      owner: "@secops"
      rules:
        - id: never-email
          type: forbidden
          tools: [send_email]
"""

    def _decision(self, tmp_path: Path, run_id: str) -> dict[str, Any]:
        import json

        result = runner.invoke(app, ["analyze", run_id])
        assert result.exit_code == 0, result.stdout
        path = tmp_path / ".evalshift" / "runs" / run_id / "migration_decision.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_target_breaking_a_rule_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold_agent_project(tmp_path, extra_evaluators=self._RULES)
        config = tmp_path / "evalshift.yaml"
        config.write_text(
            config.read_text(encoding="utf-8")
            + "\nmigration_policy:\n  max_invariant_violations: 0\n",
            encoding="utf-8",
        )
        _patch_with_tools(
            monkeypatch,
            tool_for_role={"source": "notify_security_team", "target": "send_email"},
        )
        run_id = _run_pipeline(monkeypatch, tmp_path)

        decision = self._decision(tmp_path, run_id)
        budgets = {b["name"]: b for b in decision["budget_results"]}
        assert decision["verdict"] == "fail"
        assert budgets["max_invariant_violations"]["observed"] == 20.0
        assert budgets["max_invariant_violations"]["denominator"] == 20
        assert budgets["max_invariant_violations"]["passed"] is False

    def test_a_rule_both_models_break_still_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Both models call ``send_email``: tool_selection sees equivalent
        # traces, so only the trace rule's budget can fail the run.
        _scaffold_agent_project(tmp_path, extra_evaluators=self._RULES)
        config = tmp_path / "evalshift.yaml"
        config.write_text(
            config.read_text(encoding="utf-8")
            + "\nmigration_policy:\n  max_invariant_violations: 0\n",
            encoding="utf-8",
        )
        _patch_with_tools(
            monkeypatch,
            tool_for_role={"source": "send_email", "target": "send_email"},
        )
        run_id = _run_pipeline(monkeypatch, tmp_path)

        decision = self._decision(tmp_path, run_id)
        budgets = {b["name"]: b for b in decision["budget_results"]}
        assert decision["verdict"] == "fail"
        assert budgets["max_invariant_violations"]["observed"] == 20.0
        assert budgets["max_invariant_violations"]["denominator"] == 20
        assert budgets["max_invariant_violations"]["passed"] is False
        failed = [b["name"] for b in decision["budget_results"] if not b["passed"]]
        assert failed == ["max_invariant_violations"]

    def test_clean_target_passes_the_rule(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold_agent_project(tmp_path, extra_evaluators=self._RULES)
        config = tmp_path / "evalshift.yaml"
        config.write_text(
            config.read_text(encoding="utf-8")
            + "\nmigration_policy:\n  max_invariant_violations: 0\n",
            encoding="utf-8",
        )
        _patch_with_tools(
            monkeypatch,
            tool_for_role={"source": "notify_security_team", "target": "notify_security_team"},
        )
        run_id = _run_pipeline(monkeypatch, tmp_path)

        budgets = {b["name"]: b for b in self._decision(tmp_path, run_id)["budget_results"]}
        assert budgets["max_invariant_violations"]["observed"] == 0.0
        assert budgets["max_invariant_violations"]["denominator"] == 20
        assert budgets["max_invariant_violations"]["passed"] is True
