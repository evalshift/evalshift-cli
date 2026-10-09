"""Tests for :mod:`evalshift_cli.config.loader`.

We assert two things across all paths:

1. The right :class:`ConfigError.kind` is reported.
2. The error message contains a useful pointer (file path, line number, or
   field name) so an engineer can find and fix the problem fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from evalshift_cli.config.loader import (
    ConfigError,
    ConfigErrorDetail,
    _format_loc,
    load_config,
)
from evalshift_cli.config.models import EvalShiftConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, body: str, name: str = "evalshift.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadConfigHappy:
    def test_minimal_manual_prompt(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            version: 1
            prompts:
              - id: cs
                detection: manual
                content: "Summarize: {conversation}"
                variables: [conversation]
            """,
        )
        cfg = load_config(path)
        assert isinstance(cfg, EvalShiftConfig)
        assert len(cfg.prompts) == 1
        assert cfg.prompts[0].id == "cs"
        assert cfg.prompts[0].variables == ["conversation"]

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "prompts:\n  - {id: a, detection: manual, content: hi}\n",
        )
        cfg = load_config(str(path))
        assert cfg.prompts[0].id == "a"

    def test_full_config_round_trip(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            version: 1
            prompts:
              - id: cs
                detection: python_string
                path: src/prompts.py
                variable: MY_PROMPT
                variables: [conversation]
            defaults:
              concurrency: 5
              max_cost_usd: 25.0
            evaluators:
              structural:
                - type: regex
                  pattern: "^summary"
              semantic:
                embedding_model: text-embedding-3-small
              llm_judge:
                - criterion_name: factuality
                  criterion_prompt: Which output preserves more factual detail?
            """,
        )
        cfg = load_config(path)
        assert cfg.defaults.concurrency == 5
        assert cfg.defaults.max_cost_usd == 25.0
        assert cfg.evaluators.semantic is not None
        assert len(cfg.evaluators.llm_judge) == 1


# ---------------------------------------------------------------------------
# Missing / wrong-type files
# ---------------------------------------------------------------------------


class TestLoadConfigMissing:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as info:
            load_config(tmp_path / "does-not-exist.yaml")
        assert info.value.kind == "missing"
        assert "file not found" in info.value.summary

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as info:
            load_config(tmp_path)
        assert info.value.kind == "not_a_file"

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "")
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert info.value.kind == "not_a_mapping"
        assert "empty" in info.value.summary

    def test_top_level_list_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "- a\n- b\n")
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert info.value.kind == "not_a_mapping"
        assert "list" in info.value.summary


# ---------------------------------------------------------------------------
# YAML parse errors
# ---------------------------------------------------------------------------


class TestLoadConfigYAMLParse:
    def test_malformed_yaml_reports_line(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            prompts:
              - id: a
                detection: manual
                content: "unterminated
            """,
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert info.value.kind == "yaml_parse"
        assert info.value.details
        # The reported location should mention a line number.
        assert "line " in info.value.details[0].location


# ---------------------------------------------------------------------------
# Schema violations
# ---------------------------------------------------------------------------


class TestLoadConfigSchema:
    def test_missing_prompts_field(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "version: 1\n")
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert info.value.kind == "schema"
        # Field name should appear in one of the details.
        assert any(d.location == "prompts" for d in info.value.details)

    def test_wrong_detection_value(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            prompts:
              - id: a
                detection: not_a_real_mode
                content: hi
            """,
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert info.value.kind == "schema"
        assert any("detection" in d.location for d in info.value.details)

    def test_python_string_missing_path_reports_field(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            prompts:
              - id: a
                detection: python_string
                variable: MY_PROMPT
            """,
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert any("'path' and 'variable' are required" in d.message for d in info.value.details)

    def test_extra_top_level_key_reported(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            prompts:
              - id: a
                detection: manual
                content: hi
            mystery_section: 42
            """,
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        assert any("mystery_section" in d.location for d in info.value.details)

    def test_removed_thresholds_key_explains_the_removal(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            prompts:
              - id: a
                detection: manual
                content: hi
            thresholds:
              pass_rate_min: 0.9
            """,
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        rendered = info.value.format_plain()
        assert "`thresholds` was removed" in rendered
        assert "migration_policy is the single source of truth" in rendered

    def test_removed_slices_key_explains_the_removal(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            prompts:
              - id: a
                detection: manual
                content: hi
            slices:
              - name: refunds
                filter: refunds
            """,
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        rendered = info.value.format_plain()
        assert "`slices` was removed" in rendered
        assert "migration_policy.slices" in rendered

    def test_multiple_errors_collected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            version: 99
            prompts:
              - id: ""
                detection: manual
            """,
        )
        with pytest.raises(ConfigError) as info:
            load_config(path)
        # We expect at least: bad version, empty id, missing content.
        assert len(info.value.details) >= 2


# ---------------------------------------------------------------------------
# ConfigError formatting
# ---------------------------------------------------------------------------


class TestConfigErrorFormatting:
    def test_format_plain_includes_path_and_details(self, tmp_path: Path) -> None:
        err = ConfigError(
            path=tmp_path / "evalshift.yaml",
            kind="schema",
            summary="2 schema problems found",
            details=[
                ConfigErrorDetail("prompts.0.content", "Field required"),
                ConfigErrorDetail("defaults.concurrency", "must be > 0"),
            ],
        )
        text = err.format_plain()
        assert "evalshift.yaml" in text
        assert "prompts.0.content" in text
        assert "defaults.concurrency" in text
        assert "Field required" in text

    def test_format_rich_renders_to_string(self, tmp_path: Path) -> None:
        err = ConfigError(
            path=tmp_path / "evalshift.yaml",
            kind="schema",
            summary="1 schema problem found",
            details=[ConfigErrorDetail("prompts", "Field required")],
        )
        console = Console(record=True, width=100)
        console.print(err.format_rich())
        rendered = console.export_text()
        assert "Invalid config" in rendered
        assert "prompts" in rendered

    def test_format_rich_titles_a_missing_key_as_such(self, tmp_path: Path) -> None:
        # The config is valid; only the environment lacks a key.
        err = ConfigError(
            path=tmp_path / "evalshift.yaml",
            kind="missing_key",
            summary="missing API key for a blocking evaluator",
            details=[ConfigErrorDetail("evaluators.llm_judge", "export OPENAI_API_KEY")],
        )
        console = Console(record=True, width=100)
        console.print(err.format_rich())
        rendered = console.export_text()
        assert "Missing API key" in rendered
        assert "Invalid config" not in rendered

    def test_str_uses_plain_format(self, tmp_path: Path) -> None:
        err = ConfigError(
            path=tmp_path / "x.yaml",
            kind="missing",
            summary="file not found: x.yaml",
        )
        assert "file not found" in str(err)


# ---------------------------------------------------------------------------
# _format_loc
# ---------------------------------------------------------------------------


class TestFormatLoc:
    @pytest.mark.parametrize(
        ("loc", "expected"),
        [
            ((), "<root>"),
            (("prompts",), "prompts"),
            (("prompts", 0), "prompts[0]"),
            (("prompts", 0, "content"), "prompts[0].content"),
            (("defaults", "concurrency"), "defaults.concurrency"),
        ],
    )
    def test_format(self, loc: tuple[int | str, ...], expected: str) -> None:
        assert _format_loc(loc) == expected


_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


class TestLoadConfigCheckedInExamples:
    """Every ``examples/**/evalshift.yaml`` shipped in this repo must load.

    They are the configs readers copy first. Removing a config key breaks any
    of them that still sets it, and only two are exercised end to end by other
    tests -- the sibling ``golden.jsonl`` check lives in ``test_suite_loader``.
    """

    def test_every_checked_in_evalshift_yaml_loads(self) -> None:
        configs = sorted(_EXAMPLES_DIR.glob("**/evalshift.yaml"))
        # Guard the guard: an empty glob means the path is broken.
        assert len(configs) >= 4, f"expected at least 4 example configs, found {configs}"

        failures: list[str] = []
        for path in configs:
            try:
                load_config(path)
            except ConfigError as exc:
                failures.append(f"{path.relative_to(_EXAMPLES_DIR.parent)}: {exc.format_plain()}")

        assert not failures, "example config(s) failed to load:\n" + "\n".join(failures)
