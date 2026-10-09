"""Tests for the ``evalshift run`` CLI command.

Higher-level than ``test_orchestrator.py``: we drive the actual Typer
command via :class:`CliRunner` and assert exit codes plus key bits of
the rendered output. The model client is monkeypatched so no real LLM
calls happen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.main import app
from evalshift_cli.models.client import CompletionResult, ModelClient

runner = CliRunner()


def _scaffold(tmp_path: Path, n_examples: int = 2) -> Path:
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
        """,
        encoding="utf-8",
    )
    rows = "\n".join(
        f'{{"id": "ex{i}", "inputs": {{"name": "User{i}"}}, "tools": []}}'
        for i in range(n_examples)
    )
    (tmp_path / "golden.jsonl").write_text(rows + "\n", encoding="utf-8")
    return tmp_path


def _set_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the ``run`` precheck without hitting any real provider.

    Tests that drive ``evalshift run`` past model resolution need at
    least one acceptable env var set per provider used in the scaffold
    (Gemini-only by default). The values are sentinel strings — no
    network call is ever made because :func:`_patch_client` stubs the
    LLM hop.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
        return CompletionResult(
            text="ok",
            model_id=str(kwargs["model"]),
            input_tokens=5,
            output_tokens=2,
            cost_usd=0.0,
            latency_ms=10,
        )

    monkeypatch.setattr(ModelClient, "complete", fake_complete)
    _set_api_keys(monkeypatch)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRunHappy:
    def test_run_completes_with_default_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 0, result.stdout
        assert "completed" in result.stdout.lower() or "calls:" in result.stdout
        # The run dir should exist under .evalshift/runs/.
        runs = list((tmp_path / ".evalshift" / "runs").iterdir())
        assert len(runs) == 1

    def test_from_to_flags_override_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "run",
                "--from",
                "gemini-2.5-pro",
                "--to",
                "gemini-2.5-flash",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.stdout

    def test_summary_includes_run_id(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert "r_" in result.stdout
        assert "evalshift run" in result.stdout


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestRunErrors:
    def test_missing_config_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "Invalid config" in result.stdout

    def test_missing_suite_exits_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Valid config, but no golden.jsonl in the cwd.
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n"
            "defaults:\n  source_model: gemini-2.5-flash\n"
            "  target_model: gemini-2.5-pro\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "Invalid suite" in result.stdout

    def test_missing_models_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Valid config + suite, but no defaults.source_model and no --from.
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        (tmp_path / "golden.jsonl").write_text(
            '{"id": "ex1", "inputs": {}, "tools": []}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "missing model selection" in result.stdout

    def test_incompatible_suite_exits_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Prompt requires `name`; suite provides only `tone`.
        (tmp_path / "evalshift.yaml").write_text(
            """
            version: 1
            prompts:
              - id: greet
                detection: manual
                content: "Hi {name}"
                variables: [name]
            defaults:
              source_model: gemini-2.5-flash
              target_model: gemini-2.5-pro
            """,
            encoding="utf-8",
        )
        (tmp_path / "golden.jsonl").write_text(
            '{"id": "ex1", "inputs": {"tone": "formal"}, "tools": []}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        _set_api_keys(monkeypatch)
        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "Suite incompatible" in result.stdout
        assert "name" in result.stdout


# ---------------------------------------------------------------------------
# CI behaviour: EVALSHIFT_NONINTERACTIVE
# ---------------------------------------------------------------------------


class TestRunNoninteractive:
    """`EVALSHIFT_NONINTERACTIVE=1` should imply `--yes` so CI never blocks
    on the cost prompt. We verify the flag propagation by stubbing
    ``run_orchestrator`` and capturing the kwargs it receives."""

    def test_env_var_implies_yes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _scaffold(tmp_path)
        monkeypatch.chdir(tmp_path)
        _set_api_keys(monkeypatch)
        monkeypatch.setenv("EVALSHIFT_NONINTERACTIVE", "1")

        captured: dict[str, Any] = {}

        async def fake_orchestrator(**kwargs: Any) -> Any:
            captured.update(kwargs)
            raise RuntimeError("stop here — we only care about kwargs")

        monkeypatch.setattr(
            "evalshift_cli.cli.commands.run.run_orchestrator",
            fake_orchestrator,
        )
        runner.invoke(app, ["run"])
        assert captured.get("yes") is True

    def test_no_env_no_yes_passes_yes_false(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        monkeypatch.chdir(tmp_path)
        _set_api_keys(monkeypatch)
        monkeypatch.delenv("EVALSHIFT_NONINTERACTIVE", raising=False)

        captured: dict[str, Any] = {}

        async def fake_orchestrator(**kwargs: Any) -> Any:
            captured.update(kwargs)
            raise RuntimeError("stop here")

        monkeypatch.setattr(
            "evalshift_cli.cli.commands.run.run_orchestrator",
            fake_orchestrator,
        )
        runner.invoke(app, ["run"])
        assert captured.get("yes") is False


# ---------------------------------------------------------------------------
# API key precheck
# ---------------------------------------------------------------------------


class TestRunApiKeyPrecheck:
    """`evalshift run` should refuse to launch the orchestrator when the
    provider API keys for the requested models aren't set. This catches
    the common foot-gun of running the suite with no key exported and
    getting a wall of LiteLLM banner errors instead of one clean
    message."""

    def _clear_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_no_keys_exits_one_with_clear_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        monkeypatch.chdir(tmp_path)
        self._clear_keys(monkeypatch)

        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 1
        assert "missing API key" in result.stdout
        # The Gemini-only scaffold should mention at least one acceptable key.
        assert "GEMINI_API_KEY" in result.stdout or "GOOGLE_API_KEY" in result.stdout

    def test_either_google_env_var_satisfies_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        monkeypatch.chdir(tmp_path)
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "x")  # legacy alias still accepted
        _patch_client(monkeypatch)
        # _patch_client also sets GEMINI_API_KEY; clear it so we know
        # GOOGLE_API_KEY alone is what carried us through.
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        result = runner.invoke(app, ["run", "--yes"])
        assert result.exit_code == 0, result.stdout

    def test_deepseek_arm_without_key_is_caught_before_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        monkeypatch.chdir(tmp_path)
        self._clear_keys(monkeypatch)

        result = runner.invoke(
            app, ["run", "--from", "deepseek-flash", "--to", "deepseek-v4-pro", "--yes"]
        )
        assert result.exit_code == 1
        assert "missing API key" in result.stdout
        assert "DEEPSEEK_API_KEY" in result.stdout


# ---------------------------------------------------------------------------
# Evaluator key preflight
# ---------------------------------------------------------------------------


class TestEvaluatorKeyPreflight:
    def test_keyless_blocking_judge_exits_before_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "          concurrency: 4",
                "          concurrency: 4\n"
                "        evaluators:\n"
                "          llm_judge:\n"
                "            - criterion_name: equivalence\n"
                "              criterion_prompt: which is better?\n"
                "              judge_model: gpt-4o-mini",
            ),
            encoding="utf-8",
        )
        _patch_client(monkeypatch)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["run", "--yes"])

        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.stdout
        assert not (tmp_path / ".evalshift" / "runs").exists()
