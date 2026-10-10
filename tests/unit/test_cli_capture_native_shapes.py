"""``capture sync`` on captures the real SDK wrappers recorded (Claude, OpenAI Chat).

The trees under ``fixtures/captures/native_shapes/<scenario>/`` are ready-made ``--base``
directories written by ``generate.py`` beside them, which drives the real ``wrap_anthropic`` /
``wrap_openai`` with scripted responses. Regenerate them with the command in that script's
docstring when the SDK's recording changes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.main import app
from evalshift_cli.suite.loader import load_jsonl
from evalshift_cli.suite.models import SuiteExample

runner = CliRunner()
NATIVE = Path(__file__).parent / "fixtures" / "captures" / "native_shapes"


def _sync(scenario: str, base: Path) -> Any:
    shutil.copytree(NATIVE / scenario, base, dirs_exist_ok=True)
    return runner.invoke(
        app,
        [
            "capture",
            "sync",
            "--offline",
            "--print",
            "--config",
            str(base / "evalshift.yaml"),
            "--base",
            str(base),
        ],
    )


def _turns(base: Path, suite: str) -> dict[int, SuiteExample]:
    examples = load_jsonl(base / "suites" / suite / "golden.jsonl").examples
    return {ex.turn_index: ex for ex in examples if ex.turn_index is not None}


def test_claude_turns_replay_tool_calls_results_and_the_cached_system_prompt(
    tmp_path: Path,
) -> None:
    result = _sync("claude_agent", tmp_path)
    assert result.exit_code == 0, result.output
    turns = _turns(tmp_path, "claude_agent")

    first = turns[0]
    assert first.inputs == {"input": "What is evalshift?"}
    assert first.history is not None
    assert [(m.role, m.content) for m in first.history] == [("system", "You are a support agent.")]

    second = turns[1]
    assert second.inputs == {"input": "And what does it cost?"}
    history = second.history
    assert history is not None
    assert [m.role for m in history] == ["system", "user", "assistant", "tool", "assistant"]
    assert history[2].content == "Let me look."
    assert history[2].tool_calls is not None
    assert [(c.id, c.name, c.arguments) for c in history[2].tool_calls] == [
        ("toolu_1", "search", {"q": "evalshift"})
    ]
    assert (history[3].tool_call_id, history[3].content) == ("toolu_1", "results for evalshift")
    assert history[4].content == "EvalShift is a regression tool."

    golden = (tmp_path / "suites" / "claude_agent" / "golden.jsonl").read_text(encoding="utf-8")
    assert "'type'" not in golden  # no Python-repr block lists
    assert "sig_abc" not in golden  # thinking and its signature are dropped


@pytest.mark.parametrize("scenario", ["openai_dump", "openai_lean"])
def test_openai_follow_up_turn_recovers_inputs_and_the_tool_round(
    tmp_path: Path, scenario: str
) -> None:
    result = _sync(scenario, tmp_path)
    assert result.exit_code == 0, result.output
    second = _turns(tmp_path, scenario)[1]
    assert second.inputs == {"input": "And what does it cost?"}
    history = second.history
    assert history is not None
    assert [m.role for m in history] == ["system", "user", "assistant", "tool", "assistant"]
    assert history[2].tool_calls is not None
    assert [(c.id, c.name, c.arguments) for c in history[2].tool_calls] == [
        ("call_1", "search", {"q": "evalshift"})
    ]
    assert history[3].tool_call_id == "call_1"


def test_a_claude_turn_with_an_image_is_skipped_with_its_own_reason(tmp_path: Path) -> None:
    result = _sync("claude_image", tmp_path)
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "carries a `image` block" in out
    assert "skipped 1 capture(s) with non-text content a text replay can't carry" in out
    assert "--allow-errored to promote anyway" not in out
    assert not (tmp_path / "suites" / "claude_image" / "golden.jsonl").exists()
