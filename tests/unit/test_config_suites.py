"""Tests for the ``suites:`` config section (capture-lifecycle bridge).

A ``suites`` entry names a promoted capture suite so ``evalshift run
--suite-name <name>`` can resolve its path without the user retyping it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evalshift_cli.captures.promote import PromoteOptions, build_example_from_capture
from evalshift_cli.captures.reader import iter_captures, toolset_path
from evalshift_cli.cli.commands._suites import (
    SUITES_MARKER_BEGIN,
    SUITES_MARKER_END,
    resolve_suite_path,
)
from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config
from evalshift_cli.config.loader import load_config
from evalshift_cli.config.models import (
    EvalShiftConfig,
    EvaluatorsConfig,
    SuiteEvaluatorsOverride,
    SuiteSource,
)
from evalshift_cli.suite.loader import load_jsonl


def _base_config(**extra: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "prompts": [{"id": "p", "detection": "manual", "content": "Hello {query}"}],
        **extra,
    }


def test_suites_defaults_to_empty() -> None:
    cfg = EvalShiftConfig.model_validate(_base_config())
    assert cfg.suites == {}


def test_suites_parses_captured_source() -> None:
    cfg = EvalShiftConfig.model_validate(
        _base_config(
            suites={
                "promoted": {
                    "source": "captured",
                    "path": ".evalshift/suites/support_agent/golden.jsonl",
                },
            },
        ),
    )
    entry = cfg.suites["promoted"]
    assert isinstance(entry, SuiteSource)
    assert entry.source == "captured"
    assert entry.path == ".evalshift/suites/support_agent/golden.jsonl"


def test_suites_rejects_unknown_source() -> None:
    with pytest.raises(ValidationError):
        EvalShiftConfig.model_validate(
            _base_config(suites={"x": {"source": "magic", "path": "p.jsonl"}}),
        )


def test_suites_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        EvalShiftConfig.model_validate(
            _base_config(suites={"x": {"source": "jsonl", "path": "p.jsonl", "bogus": 1}}),
        )


# ---------------------------------------------------------------------------
# Per-suite evaluator overrides (`suites.<name>.evaluators`)
# ---------------------------------------------------------------------------


_TOP_LEVEL_EVALUATORS: dict[str, Any] = {
    "semantic": {"min_similarity": 0.8},
    "llm_judge": [{"criterion_name": "helpful", "criterion_prompt": "Which is better?"}],
    "tool_selection": [{"name": "global_routing"}],
    "tool_arguments": [{"name": "global_args"}],
}


def _config_with_suite_evaluators(**suite_extra: Any) -> EvalShiftConfig:
    return EvalShiftConfig.model_validate(
        _base_config(
            evaluators=_TOP_LEVEL_EVALUATORS,
            suites={
                "main_chat": {"source": "captured", "path": "a.jsonl", **suite_extra},
                "briefing": {"source": "captured", "path": "b.jsonl"},
            },
        ),
    )


def test_suite_evaluators_defaults_to_none_and_managed_true() -> None:
    cfg = EvalShiftConfig.model_validate(
        _base_config(suites={"x": {"source": "jsonl", "path": "p.jsonl"}}),
    )
    assert cfg.suites["x"].evaluators is None
    assert cfg.suites["x"].managed is True


def test_suite_evaluators_parses_full_block() -> None:
    cfg = _config_with_suite_evaluators(
        managed=False,
        evaluators={
            "tool_selection": [{"name": "routing", "divergence": "set"}],
            "tool_arguments": [{"name": "routing_args", "against": "expected"}],
        },
    )
    override = cfg.suites["main_chat"].evaluators
    assert override is not None
    assert isinstance(override, SuiteEvaluatorsOverride)
    assert override.tool_selection is not None
    assert override.tool_selection[0].name == "routing"
    assert override.tool_arguments is not None
    assert override.tool_arguments[0].against == "expected"
    assert cfg.suites["main_chat"].managed is False


def test_suite_evaluators_rejects_typo_in_nested_block() -> None:
    with pytest.raises(ValidationError):
        _config_with_suite_evaluators(
            evaluators={"tool_selection": [{"name": "routing", "divergance": "set"}]},
        )


def test_suite_evaluators_rejects_unknown_family() -> None:
    with pytest.raises(ValidationError):
        _config_with_suite_evaluators(evaluators={"tool_arguemnts": []})


def test_evaluators_for_none_returns_top_level_identity() -> None:
    cfg = _config_with_suite_evaluators()
    assert cfg.evaluators_for(None) is cfg.evaluators


def test_evaluators_for_suite_without_block_returns_top_level_identity() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={"tool_selection": [{"name": "routing"}]},
    )
    assert cfg.evaluators_for("briefing") is cfg.evaluators


def test_evaluators_for_unknown_suite_returns_top_level() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={"tool_selection": [{"name": "routing"}]},
    )
    assert cfg.evaluators_for("not-a-suite") is cfg.evaluators


def test_evaluators_for_replaces_declared_family_and_inherits_rest() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={"tool_arguments": [{"name": "routing_args", "against": "expected"}]},
    )
    resolved = cfg.evaluators_for("main_chat")

    # Declared family fully replaced — the top-level `global_args` is gone.
    assert [e.name for e in resolved.tool_arguments] == ["routing_args"]
    # Undeclared families inherited verbatim.
    assert resolved.semantic is not None
    assert resolved.semantic.min_similarity == 0.8
    assert [e.criterion_name for e in resolved.llm_judge] == ["helpful"]
    assert [e.name for e in resolved.tool_selection] == ["global_routing"]
    # The top-level config is untouched by resolution.
    assert [e.name for e in cfg.evaluators.tool_arguments] == ["global_args"]


def test_evaluators_for_declared_empty_family_removes_it() -> None:
    cfg = _config_with_suite_evaluators(evaluators={"tool_selection": []})
    resolved = cfg.evaluators_for("main_chat")
    assert resolved.tool_selection == []
    # Still inherits everything it did not declare.
    assert [e.name for e in resolved.tool_arguments] == ["global_args"]


def test_evaluators_for_explicit_null_removes_semantic() -> None:
    cfg = _config_with_suite_evaluators(evaluators={"semantic": None})
    assert cfg.evaluators_for("main_chat").semantic is None
    assert cfg.evaluators.semantic is not None


def test_evaluators_for_reflects_suite_tool_evaluator_names() -> None:
    cfg = _config_with_suite_evaluators(
        evaluators={
            "tool_selection": [{"name": "routing"}],
            "tool_arguments": [{"name": "routing_args", "against": "expected"}],
        },
    )
    assert cfg.evaluators_for("main_chat").tool_evaluator_names == frozenset(
        {"routing", "routing_args"},
    )
    assert cfg.evaluators_for("briefing").tool_evaluator_names == frozenset(
        {"global_routing", "global_args"},
    )


def test_suite_override_covers_every_evaluator_family() -> None:
    """The override must stay field-for-field aligned with the top-level model."""
    assert set(SuiteEvaluatorsOverride.model_fields) == set(EvaluatorsConfig.model_fields)


# ---------------------------------------------------------------------------
# The checked-in capture-first example
# ---------------------------------------------------------------------------

# Repo root: tests/unit/test_config_suites.py -> tests/unit -> tests -> <root>.
# A direct path, never an rglob from the repo root, for the same reason
# test_suite_loader.py spells it out: .claude/worktrees/ can hold a stale full
# copy of the repo that must never be treated as source of truth.
_CAPTURE_FIRST = Path(__file__).resolve().parents[2] / "examples" / "capture-first"


class TestCheckedInCaptureFirstExample:
    """``examples/capture-first/`` must stay exactly what ``capture sync`` writes.

    It is the only example wired through the managed ``suites:`` block, and the
    only one whose committed artefacts (captures, toolset sidecar, promoted
    cases, ``golden.jsonl``) were produced by the CLI rather than typed. Nothing
    else in the suite would notice a promotion change quietly invalidating them,
    because ``examples/`` is documentation: users copy it, CI does not run it.
    """

    def test_config_wires_the_promoted_suite(self) -> None:
        cfg = load_config(_CAPTURE_FIRST / "evalshift.yaml")
        entry = cfg.suites["oncall_triage"]
        assert entry.source == "captured"
        resolved = resolve_suite_path(
            suite_path=None,
            suite_name="oncall_triage",
            cfg=cfg,
            config_path=_CAPTURE_FIRST / "evalshift.yaml",
        )
        assert resolved.is_file(), f"{resolved} is not committed"

    def test_config_is_a_fresh_init_without_its_captures_block(self) -> None:
        """Outside the managed region, the config is what the README's ``init`` writes.

        ``init --provider gemini --no-wire-agents`` (profile ``model-upgrade``), minus the
        ``captures:`` block the README has the reader delete: the demo captures are older
        than ``max_age`` and recorded on a placeholder model, so a bounded sync would
        promote none of them.
        """

        def outside_region(text: str) -> str:
            head, _, rest = text.partition(SUITES_MARKER_BEGIN)
            _, _, tail = rest.partition(SUITES_MARKER_END)
            return head + tail

        committed = (_CAPTURE_FIRST / "evalshift.yaml").read_text(encoding="utf-8")
        assert SUITES_MARKER_BEGIN in committed
        assert SUITES_MARKER_END in committed
        fresh = render_minimal_config(profile="model-upgrade", provider="gemini", env={})
        expected = fresh.replace(CAPTURES_SCAFFOLD + "\n", "")
        assert expected != fresh
        assert outside_region(committed) == outside_region(expected)

    def test_derived_evaluators_are_wired_for_the_suite(self) -> None:
        """The captures call tools, so sync must have derived both tool families."""
        cfg = load_config(_CAPTURE_FIRST / "evalshift.yaml")
        resolved = cfg.evaluators_for("oncall_triage")
        assert [e.name for e in resolved.tool_selection] == ["routing"]
        assert [e.name for e in resolved.tool_arguments] == ["routing_args"]

    def test_every_row_resolves_its_committed_toolset_sidecar(self) -> None:
        base = _CAPTURE_FIRST / ".evalshift"
        suite = load_jsonl(base / "suites" / "oncall_triage" / "golden.jsonl")
        assert suite.examples
        for example in suite.examples:
            assert example.toolset_ref is not None
            assert toolset_path(example.toolset_ref, base=base).is_file()

    def test_committed_suite_is_what_promotion_still_produces(self) -> None:
        """Re-promote the committed captures and diff against the committed rows.

        Guards the example against a promotion change it cannot notice on its
        own: the captures are the input, ``golden.jsonl`` is the output, and a
        drift between them means the walkthrough in the README no longer
        describes what the CLI does.
        """
        base = _CAPTURE_FIRST / ".evalshift"
        golden = load_jsonl(base / "suites" / "oncall_triage" / "golden.jsonl")
        committed = {e.id: e for e in golden.examples}
        records = iter_captures(base=base)
        assert len(records) == len(committed) > 0

        for record in records:
            built = build_example_from_capture(record.envelope, PromoteOptions(), base=base)
            assert built.blocked is None, built.blocked
            assert built.example == committed[record.envelope.capture_id]
