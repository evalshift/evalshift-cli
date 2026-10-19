"""Tests for the shared suite helpers in ``cli.commands._suites``."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from evalshift_cli.captures.toolset import EMPTY_TOOLSET_FINGERPRINT
from evalshift_cli.cli.commands._suites import (
    SUITE_FILENAME,
    AmbiguousSuiteError,
    UnknownSuiteNameError,
    derive_suite_evaluators,
    derive_suite_slug,
    format_invocation,
    parse_suites_region,
    render_suites_region,
    render_suites_yaml,
    resolve_suite_path,
    selected_suite_name,
    suite_entry_payload,
)
from evalshift_cli.config.models import EvalShiftConfig
from evalshift_cli.evaluators.tool_models import ToolSpec
from evalshift_cli.suite.models import ExpectedToolCall, SuiteExample


def _config(**extra: Any) -> EvalShiftConfig:
    return EvalShiftConfig.model_validate(
        {
            "version": 1,
            "prompts": [{"id": "p", "detection": "manual", "content": "Hi {q}"}],
            **extra,
        },
    )


class TestResolveSuitePath:
    _CFG_PATH = Path("/proj/evalshift.yaml")

    def _resolve(self, cfg: EvalShiftConfig, **kw: Any) -> Path:
        return resolve_suite_path(
            suite_path=kw.get("suite_path"),
            suite_name=kw.get("suite_name"),
            cfg=cfg,
            config_path=self._CFG_PATH,
        )

    def test_explicit_suite_path_wins(self) -> None:
        cfg = _config(suites={"a": {"path": "a.jsonl"}, "b": {"path": "b.jsonl"}})
        got = self._resolve(cfg, suite_path=Path("custom.jsonl"))
        assert got == Path("custom.jsonl")

    def test_suite_name_resolves_relative_to_config(self) -> None:
        cfg = _config(suites={"promoted": {"path": ".evalshift/suites/x/golden.jsonl"}})
        got = self._resolve(cfg, suite_name="promoted")
        assert got == Path("/proj/.evalshift/suites/x/golden.jsonl")

    def test_unknown_suite_name_raises(self) -> None:
        cfg = _config(suites={"promoted": {"path": "x.jsonl"}})
        with pytest.raises(UnknownSuiteNameError):
            self._resolve(cfg, suite_name="missing")

    def test_single_wired_suite_auto_selected(self) -> None:
        # Bare `evalshift compare` after capture sync: exactly one suite, no flag.
        cfg = _config(suites={"only": {"path": ".evalshift/suites/only/golden.jsonl"}})
        got = self._resolve(cfg)
        assert got == Path("/proj/.evalshift/suites/only/golden.jsonl")

    def test_multiple_wired_suites_are_ambiguous(self) -> None:
        cfg = _config(suites={"a": {"path": "a.jsonl"}, "b": {"path": "b.jsonl"}})
        with pytest.raises(AmbiguousSuiteError) as excinfo:
            self._resolve(cfg)
        # Error names the choices and the fix.
        assert "a" in str(excinfo.value)
        assert "--suite-name" in str(excinfo.value)

    def test_no_suites_falls_back_to_golden_file(self) -> None:
        cfg = _config()
        got = self._resolve(cfg)
        assert got == Path(SUITE_FILENAME)


class TestSelectedSuiteName:
    """The ``suites:`` key :func:`resolve_suite_path` resolves to, if any."""

    def _name(self, cfg: EvalShiftConfig, **kw: Any) -> str | None:
        return selected_suite_name(
            suite_path=kw.get("suite_path"), suite_name=kw.get("suite_name"), cfg=cfg
        )

    def test_explicit_suite_path_names_no_suite(self) -> None:
        cfg = _config(suites={"only": {"path": "a.jsonl"}})
        assert self._name(cfg, suite_path=Path("a.jsonl"), suite_name="only") is None

    def test_suite_name_is_returned_as_given(self) -> None:
        cfg = _config(suites={"a": {"path": "a.jsonl"}, "b": {"path": "b.jsonl"}})
        assert self._name(cfg, suite_name="b") == "b"

    def test_single_wired_suite_is_auto_selected(self) -> None:
        cfg = _config(suites={"only": {"path": "a.jsonl"}})
        assert self._name(cfg) == "only"

    def test_several_or_no_wired_suites_select_nothing(self) -> None:
        assert self._name(_config()) is None
        assert self._name(_config(suites={"a": {"path": "a"}, "b": {"path": "b"}})) is None


class TestDeriveSuiteSlug:
    def test_prefers_explicit_suite_name(self) -> None:
        slug = derive_suite_slug(
            suite_name="note_search",
            suite_path=Path(".evalshift/suites/note_search/golden.jsonl"),
        )
        assert slug == "note_search"

    def test_uses_parent_dir_for_golden_layout(self) -> None:
        slug = derive_suite_slug(
            suite_name=None,
            suite_path=Path(".evalshift/suites/note_search/golden.jsonl"),
        )
        assert slug == "note_search"

    def test_falls_back_to_file_stem(self) -> None:
        slug = derive_suite_slug(
            suite_name=None,
            suite_path=Path("datasets/smoke.jsonl"),
        )
        assert slug == "smoke"


# ---------------------------------------------------------------------------
# derive_suite_evaluators
# ---------------------------------------------------------------------------


def _example(idx: int, **kw: Any) -> SuiteExample:
    """A minimal suite row; ``kw`` carries the toolset/ground-truth under test."""
    return SuiteExample(id=f"ex{idx}", inputs={"q": "hi"}, **kw)


def _tool_row(idx: int, **kw: Any) -> SuiteExample:
    """A row whose agent was offered one tool."""
    return _example(idx, tools=[ToolSpec(name="search_orders")], **kw)


class TestDeriveSuiteEvaluators:
    def test_no_tools_offered_yields_no_block(self) -> None:
        """A prose suite gets no tool evaluators — an empty denominator is not a verdict."""
        assert derive_suite_evaluators([_example(1, tools=[]), _example(2, tools=[])]) is None

    def test_empty_toolset_ref_yields_no_block(self) -> None:
        """The ``toolset_ref`` spelling of "no tools offered" reads the same."""
        rows = [_example(1, toolset_ref=EMPTY_TOOLSET_FINGERPRINT)]
        assert derive_suite_evaluators(rows) is None

    def test_no_rows_yields_no_block(self) -> None:
        assert derive_suite_evaluators([]) is None

    def test_any_toolset_row_yields_tool_selection(self) -> None:
        override = derive_suite_evaluators([_example(1, tools=[]), _tool_row(2)])

        assert override is not None
        assert override.tool_selection is not None
        (selection,) = override.tool_selection
        assert selection.name == "routing"
        assert selection.conformance == "expected"
        # `set`, not `off`: reordered identical calls are the same behaviour,
        # but two models failing the ground truth differently is not.
        assert selection.divergence == "set"

    def test_toolset_without_recorded_arguments_yields_no_tool_arguments(self) -> None:
        rows = [_tool_row(1, expected_tools=[ExpectedToolCall(tool_name="search_orders")])]
        override = derive_suite_evaluators(rows)

        assert override is not None
        assert override.tool_arguments is None

    def test_recorded_arguments_yield_tool_arguments(self) -> None:
        rows = [
            _tool_row(1, expected_tools=[ExpectedToolCall(tool_name="search_orders")]),
            _tool_row(
                2,
                expected_tools=[
                    ExpectedToolCall(tool_name="search_orders", arguments={"customer_id": "c42"}),
                ],
            ),
        ]
        override = derive_suite_evaluators(rows)

        assert override is not None
        assert override.tool_arguments is not None
        (arguments,) = override.tool_arguments
        assert arguments.name == "routing_args"
        assert arguments.against == "expected"
        # No `strategies:` — `default_strategy: auto` already grades free text.
        assert arguments.strategies == {}

    def test_declares_nothing_about_the_families_it_does_not_generate(self) -> None:
        """Undeclared families inherit the top-level block (see ``evaluators_for``)."""
        override = derive_suite_evaluators([_tool_row(1)])

        assert override is not None
        assert override.semantic is None
        assert override.llm_judge is None
        assert override.structural is None

    def test_names_are_stable_across_calls(self) -> None:
        """Reports key on evaluator names across runs, so regeneration must not rename."""
        rows = [
            _tool_row(
                1,
                expected_tools=[
                    ExpectedToolCall(tool_name="search_orders", arguments={"customer_id": "c42"}),
                ],
            ),
        ]
        first = derive_suite_evaluators(rows)
        second = derive_suite_evaluators(rows)

        assert first is not None
        assert second is not None
        assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# suites: region rendering
# ---------------------------------------------------------------------------


class TestSuiteEntryPayload:
    def test_omits_evaluators_when_none(self) -> None:
        payload = suite_entry_payload(path="suites/alpha/golden.jsonl", evaluators=None)
        assert payload == {"source": "captured", "path": "suites/alpha/golden.jsonl"}

    def test_carries_only_the_generated_keys(self) -> None:
        """Defaults stay unwritten so a future default change still reaches the config."""
        override = derive_suite_evaluators([_tool_row(1)])
        payload = suite_entry_payload(path="p.jsonl", evaluators=override)

        assert payload["evaluators"] == {
            "tool_selection": [
                {"name": "routing", "conformance": "expected", "divergence": "set"},
            ],
        }


class TestRenderSuitesYaml:
    def test_empty_mapping(self) -> None:
        assert render_suites_yaml({}).strip() == "suites: {}"

    def test_indents_sequences_under_their_key(self) -> None:
        rendered = render_suites_yaml(
            {
                "alpha": suite_entry_payload(
                    path="p.jsonl", evaluators=derive_suite_evaluators([_tool_row(1)])
                )
            },
        )

        assert rendered.splitlines() == [
            "suites:",
            "  alpha:",
            "    source: captured",
            "    path: p.jsonl",
            "    evaluators:",
            "      tool_selection:",
            "        - name: routing",
            "          conformance: expected",
            "          divergence: set",
        ]

    def test_sorted_by_suite_name(self) -> None:
        rendered = render_suites_yaml(
            {
                "beta": suite_entry_payload(path="b.jsonl", evaluators=None),
                "alpha": suite_entry_payload(path="a.jsonl", evaluators=None),
            },
        )
        assert rendered.index("alpha") < rendered.index("beta")

    def test_round_trips_through_the_region_parser(self) -> None:
        entries = {
            "alpha": suite_entry_payload(
                path="a.jsonl",
                evaluators=derive_suite_evaluators([_tool_row(1)]),
            ),
            "beta": suite_entry_payload(path="b.jsonl", evaluators=None),
        }
        rendered = render_suites_yaml(entries)

        assert parse_suites_region(render_suites_region(rendered)) == entries
        # ...and re-rendering what was parsed is byte-identical (idempotency).
        assert render_suites_yaml(parse_suites_region(render_suites_region(rendered))) == rendered


class TestParseSuitesRegion:
    def test_missing_markers_yield_no_entries(self) -> None:
        assert parse_suites_region("version: 1\n") == {}

    def test_empty_region_yields_no_entries(self) -> None:
        assert parse_suites_region(render_suites_region("suites: {}")) == {}

    def test_ignores_config_outside_the_region(self) -> None:
        """Only the managed region is ours to carry forward."""
        text = "suites:\n  hand_written:\n    path: h.jsonl\n\n" + render_suites_region(
            "suites:\n  alpha:\n    source: captured\n    path: a.jsonl"
        )
        assert parse_suites_region(text) == {
            "alpha": {"source": "captured", "path": "a.jsonl"},
        }

    def test_unparseable_region_yields_no_entries(self) -> None:
        assert parse_suites_region(render_suites_region("suites: [oops")) == {}


class TestFormatInvocation:
    def test_rebuilds_the_command_line_under_the_evalshift_name(self) -> None:
        got = format_invocation(["/usr/bin/evalshift", "compare", "--yes", "--push"])
        assert got == "evalshift compare --yes --push"

    def test_drops_an_existing_suite_selection_flag(self) -> None:
        # The hint appends --suite-name, so the stale one must not survive.
        got = format_invocation(["evalshift", "compare", "--suite-name", "gone", "--yes"])
        assert got == "evalshift compare --yes"

    def test_drops_the_equals_spelling_too(self) -> None:
        got = format_invocation(["evalshift", "compare", "--suite-name=gone", "--yes"])
        assert got == "evalshift compare --yes"

    def test_drops_an_explicit_suite_path(self) -> None:
        got = format_invocation(["evalshift", "run", "--suite", "a.jsonl"])
        assert got == "evalshift run"

    def test_quotes_arguments_that_need_it(self) -> None:
        got = format_invocation(["evalshift", "compare", "--to", "my model"])
        assert got == "evalshift compare --to 'my model'"

    def test_empty_argv_falls_back_to_the_bare_name(self) -> None:
        assert format_invocation([]) == "evalshift"

    def test_the_legacy_all_name_survives_the_rebuild(self) -> None:
        # Someone still typing `all` must get hints that work as typed.
        got = format_invocation(["evalshift", "all", "--yes"])
        assert got == "evalshift all --yes"


class TestAmbiguousSuiteErrorMessage:
    def _error(self, invocation: str = "evalshift compare --yes --push") -> AmbiguousSuiteError:
        return AmbiguousSuiteError(
            suite_names=["support_agent", "billing_agent"],
            invocation=invocation,
        )

    def test_lists_one_runnable_command_per_suite(self) -> None:
        text = self._error().format_plain()
        assert "evalshift compare --yes --push --suite-name billing_agent" in text
        assert "evalshift compare --yes --push --suite-name support_agent" in text

    def test_commands_are_listed_in_name_order(self) -> None:
        text = self._error().format_plain()
        assert text.index("billing_agent") < text.index("support_agent")

    def test_the_legacy_all_name_gets_the_not_all_suites_note(self) -> None:
        text = self._error("evalshift all --yes").format_plain()
        assert "does not run all suites" in text

    def test_compare_needs_no_such_note(self) -> None:
        # The note exists to correct `all`; `compare` never made the promise.
        text = self._error().format_plain()
        assert "does not run all suites" not in text

    def test_the_note_is_omitted_for_other_commands(self) -> None:
        text = self._error("evalshift run").format_plain()
        assert "does not run all suites" not in text
        assert "evalshift run --suite-name billing_agent" in text

    def test_renders_rich_without_raising(self) -> None:
        console = Console(file=io.StringIO(), width=100)
        console.print(self._error().format_rich())
        out = console.file.getvalue()  # type: ignore[attr-defined]
        assert "--suite-name billing_agent" in out


class TestUnknownSuiteNameErrorMessage:
    def test_names_the_typo_and_offers_the_real_commands(self) -> None:
        exc = UnknownSuiteNameError(
            name="suport_agent",
            known=["support_agent"],
            invocation="evalshift compare --suite-name suport_agent",
        )
        text = exc.format_plain()
        assert "suport_agent" in text
        assert "evalshift compare --suite-name support_agent" in text

    def test_renders_rich_without_raising(self) -> None:
        exc = UnknownSuiteNameError(
            name="nope",
            known=["support_agent"],
            invocation="evalshift bundle r_1",
        )
        console = Console(file=io.StringIO(), width=100)
        console.print(exc.format_rich())
        out = console.file.getvalue()  # type: ignore[attr-defined]
        assert "evalshift bundle r_1 --suite-name support_agent" in out

    def test_points_at_the_config_when_no_suites_are_wired(self) -> None:
        exc = UnknownSuiteNameError(name="x", known=[], invocation="evalshift compare")
        text = exc.format_plain()
        assert "evalshift.yaml" in text
