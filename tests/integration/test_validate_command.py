"""Integration tests for ``evalshift validate``.

Each test runs the command inside a fixture project directory under
``tests/integration/fixtures/`` and asserts the exit code + key fragments
of the rendered output. No LLM calls happen here — Phase 2 is purely
local validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.main import app

runner = CliRunner()

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _invoke_in_fixture(
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
) -> tuple[int, str]:
    """Run ``evalshift validate`` inside the named fixture directory."""
    monkeypatch.chdir(FIXTURES_DIR / fixture_name)
    result = runner.invoke(app, ["validate"])
    return result.exit_code, result.stdout


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidateHappy:
    def test_ok_fixture_exits_zero_with_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, output = _invoke_in_fixture(monkeypatch, "validate_ok")
        assert code == 0, output
        assert "Loaded 1 prompt" in output
        assert "2 examples" in output
        assert "compatible" in output

    def test_validate_is_hidden_from_help(self) -> None:
        # `validate` is registered with `hidden=True`; it should run, but
        # not appear in the top-level --help text.
        result = runner.invoke(app, ["--help"])
        assert "validate" not in result.stdout


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestValidateFailures:
    def test_missing_template_variable_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, output = _invoke_in_fixture(monkeypatch, "validate_missing_var")
        assert code == 1, output
        # Error must point at the offending prompt + example + variable.
        assert "greet" in output
        assert "ex2" in output
        assert "tone" in output

    def test_non_literal_prompt_exits_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, output = _invoke_in_fixture(monkeypatch, "validate_non_literal")
        assert code == 1, output
        # Error should mention the offending value form.
        assert "f-string" in output or "non_literal" in output.lower()

    def test_missing_config_exits_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)  # empty dir — no evalshift.yaml
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 1
        assert "Invalid config" in result.stdout

    @pytest.mark.parametrize(
        ("block", "removed"),
        [
            ("thresholds:\n  pass_rate_min: 0.9\n", "`thresholds` was removed"),
            ("slices:\n  - {name: refunds, filter: refunds}\n", "`slices` was removed"),
        ],
        ids=["thresholds", "slices"],
    )
    def test_a_removed_key_is_named_not_called_a_typo(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        block: str,
        removed: str,
    ) -> None:
        """Both retired keys fail the same way: the panel names the removal."""
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n" + block,
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        # Wide enough that Rich does not wrap the message inside the panel.
        result = runner.invoke(app, ["validate"], env={"COLUMNS": "400"})
        assert result.exit_code == 1
        assert "Invalid config" in result.stdout
        assert removed in result.stdout
        assert "Extra inputs are not permitted" not in result.stdout

    def test_missing_suite_exits_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Valid config but no suite file in the cwd.
        (tmp_path / "evalshift.yaml").write_text(
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 1
        assert "Invalid suite" in result.stdout


# ---------------------------------------------------------------------------
# CI pin drift (advisory — never changes the exit code)
# ---------------------------------------------------------------------------


class TestValidateCiPin:
    def test_warns_after_the_success_line_when_ci_pins_an_older_cli(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import shutil

        shutil.copytree(FIXTURES_DIR / "validate_ok", tmp_path, dirs_exist_ok=True)
        workflow = tmp_path / ".github" / "workflows" / "evalshift.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text(
            "on: push\njobs:\n  evalshift:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: evalshift/evalshift-action@v0\n"
            '        with:\n          evalshift-version: "0.0.1"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("evalshift_cli.cli.commands.validate.__version__", "1.2.3")
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, result.stdout
        assert result.stdout.index("compatible") < result.stdout.index(
            "CI installs evalshift 0.0.1"
        )


# ---------------------------------------------------------------------------
# Judge family (advisory — never changes the exit code)
# ---------------------------------------------------------------------------


class TestValidateJudgeFamily:
    def _project(self, tmp_path: Path, *, judge: str) -> Path:
        import shutil

        shutil.copytree(FIXTURES_DIR / "validate_ok", tmp_path, dirs_exist_ok=True)
        cfg = tmp_path / "evalshift.yaml"
        cfg.write_text(
            cfg.read_text(encoding="utf-8")
            + "defaults:\n  source_model: anthropic/claude-sonnet-4-5\n"
            "  target_model: gemini/gemini-2.5-pro\n"
            "evaluators:\n  llm_judge:\n    - criterion_name: c\n"
            f"      criterion_prompt: which is better?\n      judge_model: {judge}\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_warns_after_the_success_line_when_the_judge_shares_a_family(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(self._project(tmp_path, judge="gemini/gemini-3.1-flash-lite-preview"))
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, result.stdout
        assert "self-preference" in result.stdout
        assert "target" in result.stdout
        assert result.stdout.index("compatible") < result.stdout.index("self-preference")

    def test_silent_for_a_third_family_judge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(self._project(tmp_path, judge="openai/gpt-4.1-mini"))
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, result.stdout
        assert "self-preference" not in result.stdout


# ---------------------------------------------------------------------------
# Stringified history (advisory — never changes the exit code)
# ---------------------------------------------------------------------------


class TestValidateStringifiedHistory:
    def _project(self, tmp_path: Path, *, history_content: str) -> Path:
        import json
        import shutil

        shutil.copytree(FIXTURES_DIR / "validate_ok", tmp_path, dirs_exist_ok=True)
        row = {
            "id": "ex_conv",
            "inputs": {"name": "Kim", "tone": "formal"},
            "tools": [],
            "history": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": history_content},
            ],
        }
        with (tmp_path / "golden.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return tmp_path

    def test_warns_after_the_success_line_on_stringified_blocks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stringified = str([{"type": "text", "text": "hello"}])
        monkeypatch.chdir(self._project(tmp_path, history_content=stringified))
        result = runner.invoke(app, ["validate"])
        out = " ".join(result.stdout.split())
        assert result.exit_code == 0, result.stdout
        assert (
            "1 example(s) carry message blocks an earlier capture sync stringified "
            "(e.g. ex_conv); run evalshift capture sync --repair."
        ) in out
        assert out.index("compatible") < out.index("stringified")

    def test_silent_for_ordinary_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(self._project(tmp_path, history_content="hello"))
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, result.stdout
        assert "stringified" not in result.stdout
