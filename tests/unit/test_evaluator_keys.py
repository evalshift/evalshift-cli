"""Tests for :mod:`evalshift_cli.evaluators.keys`."""

from __future__ import annotations

from typing import Any

from evalshift_cli.config.models import EvalShiftConfig, EvaluatorsConfig
from evalshift_cli.evaluators.keys import (
    AUTO_FALLBACK_NOTE,
    EXACT_FALLBACK_NOTE,
    EvaluatorKeyGap,
    all_skipped_error,
    evaluator_key_gaps,
    forced_gate_hint,
    gap_detail,
    has_evaluators,
    missing_key_error,
    skip_recommendation,
    skip_warning,
    without_gaps,
)

_JUDGE: dict[str, Any] = {
    "criterion_name": "equivalence",
    "criterion_prompt": "which is better?",
    "judge_model": "gpt-4o-mini",
    "blocking": False,
}


def _evaluators(**families: Any) -> EvaluatorsConfig:
    return EvaluatorsConfig.model_validate(families)


class TestEvaluatorKeyGaps:
    def test_no_gaps_when_every_key_is_set(self) -> None:
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small"}, llm_judge=[_JUDGE]
        )
        assert evaluator_key_gaps(evaluators, {"OPENAI_API_KEY": "k"}) == []

    def test_keyless_semantic_and_judge_are_both_reported(self) -> None:
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small", "blocking": False},
            llm_judge=[_JUDGE],
        )
        assert evaluator_key_gaps(evaluators, {}) == [
            EvaluatorKeyGap(
                evaluator_name="semantic.cosine",
                kind="semantic",
                label="semantic",
                model="openai/text-embedding-3-small",
                env_vars=("OPENAI_API_KEY",),
                blocking=False,
            ),
            EvaluatorKeyGap(
                evaluator_name="llm_judge.equivalence",
                kind="llm_judge",
                label="llm_judge.equivalence",
                model="gpt-4o-mini",
                env_vars=("OPENAI_API_KEY",),
                blocking=False,
            ),
        ]

    def test_bare_default_embedding_id_is_checked(self) -> None:
        # The config default `text-embedding-3-small` used to resolve to
        # provider "other", so its key was never checked.
        gaps = evaluator_key_gaps(_evaluators(semantic={}), {})
        assert [g.env_vars for g in gaps] == [("OPENAI_API_KEY",)]

    def test_gemini_judge_accepts_google_alias(self) -> None:
        judge = {**_JUDGE, "judge_model": "gemini-2.5-flash"}
        assert evaluator_key_gaps(_evaluators(llm_judge=[judge]), {"GOOGLE_API_KEY": "k"}) == []

    def test_blocking_flag_is_carried(self) -> None:
        judge = {**_JUDGE, "blocking": True}
        (gap,) = evaluator_key_gaps(_evaluators(llm_judge=[judge]), {})
        assert gap.blocking is True

    def test_gaps_follow_the_resolved_suite_set(self) -> None:
        cfg = EvalShiftConfig.model_validate(
            {
                "prompts": [{"id": "a", "detection": "manual", "content": "hi"}],
                "evaluators": {"llm_judge": [_JUDGE]},
                "suites": {"tools": {"path": "s.jsonl", "evaluators": {"llm_judge": []}}},
            }
        )
        assert evaluator_key_gaps(cfg.evaluators_for("tools"), {}) == []
        assert len(evaluator_key_gaps(cfg.evaluators_for(None), {})) == 1


class TestSemanticToolArgumentsInteraction:
    def _gap(self, **tool_arguments: Any) -> EvaluatorKeyGap:
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small", "blocking": False},
            tool_arguments=[{"name": "args", **tool_arguments}],
        )
        (gap,) = evaluator_key_gaps(evaluators, {})
        return gap

    def test_blocking_explicit_semantic_strategy_makes_the_gap_blocking(self) -> None:
        gap = self._gap(strategies={"query": "semantic"}, blocking=True)
        assert gap.blocking is True
        assert gap.note == EXACT_FALLBACK_NOTE

    def test_blocking_semantic_default_strategy_makes_the_gap_blocking(self) -> None:
        gap = self._gap(default_strategy="semantic", blocking=True)
        assert gap.blocking is True

    def test_advisory_explicit_semantic_strategy_stays_advisory_with_a_note(self) -> None:
        gap = self._gap(strategies={"query": "semantic"}, blocking=False)
        assert gap.blocking is False
        assert gap.note == EXACT_FALLBACK_NOTE

    def test_auto_strategy_stays_advisory_and_notes_difflib(self) -> None:
        gap = self._gap(blocking=True)  # default_strategy is auto
        assert gap.blocking is False
        assert gap.note == AUTO_FALLBACK_NOTE

    def test_forced_gate_names_the_tool_arguments_entry(self) -> None:
        gap = self._gap(strategies={"query": "semantic"}, blocking=True)
        assert gap.blocked_by == "args"

    def test_advisory_tool_arguments_names_no_entry(self) -> None:
        gap = self._gap(strategies={"query": "semantic"}, blocking=False)
        assert gap.blocked_by == ""

    def test_own_blocking_flag_names_no_entry(self) -> None:
        # semantic is a gate on its own account; the tool_arguments entry is
        # not why it blocks, so "set blocking: false" is still the right hint.
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small", "blocking": True},
            tool_arguments=[{"name": "args", "default_strategy": "semantic", "blocking": True}],
        )
        (gap,) = evaluator_key_gaps(evaluators, {})
        assert gap.blocking is True
        assert gap.blocked_by == ""


class TestWithoutGaps:
    def test_drops_only_the_named_evaluators(self) -> None:
        other = {**_JUDGE, "criterion_name": "tone", "judge_model": "claude-sonnet-5"}
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small"},
            llm_judge=[_JUDGE, other],
            structural=[{"type": "length", "min_chars": 1}],
        )
        gaps = evaluator_key_gaps(evaluators, {"ANTHROPIC_API_KEY": "k"})
        kept = without_gaps(evaluators, gaps)
        assert kept.semantic is None
        assert [j.criterion_name for j in kept.llm_judge] == ["tone"]
        assert len(kept.structural) == 1

    def test_no_gaps_returns_the_same_instance(self) -> None:
        evaluators = _evaluators(llm_judge=[_JUDGE])
        assert without_gaps(evaluators, []) is evaluators


class TestHasEvaluators:
    def test_empty_set_has_none(self) -> None:
        assert has_evaluators(_evaluators()) is False

    def test_dropping_every_keyless_entry_can_leave_none(self) -> None:
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small", "blocking": False},
            llm_judge=[_JUDGE],
        )
        assert has_evaluators(evaluators) is True
        assert has_evaluators(without_gaps(evaluators, evaluator_key_gaps(evaluators, {}))) is False

    def test_any_other_family_counts(self) -> None:
        assert has_evaluators(_evaluators(tool_selection=[{"name": "sel"}])) is True


class TestMessages:
    def test_skip_warning(self) -> None:
        assert skip_warning(
            label="semantic", model="openai/text-embedding-3-small", env_vars=("OPENAI_API_KEY",)
        ) == (
            "[yellow]⚠[/yellow] semantic skipped: no API key for "
            "openai/text-embedding-3-small — export OPENAI_API_KEY to enable it."
        )

    def test_skip_warning_appends_the_note_on_its_own_line(self) -> None:
        text = skip_warning(label="semantic", model="m", env_vars=("K",), note="n.")
        assert text.endswith("\n  n.")

    def test_missing_key_error(self) -> None:
        assert missing_key_error(
            label="llm_judge.equivalence", model="gpt-4o-mini", env_vars=("OPENAI_API_KEY",)
        ) == (
            "[red]✗[/red] missing API key for [bold]gpt-4o-mini[/bold] "
            "(llm_judge.equivalence); export OPENAI_API_KEY."
        )

    def test_skip_recommendation_joins_aliases(self) -> None:
        assert skip_recommendation(
            label="llm_judge.equivalence",
            model="gemini-2.5-flash",
            env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        ) == (
            "llm_judge.equivalence was skipped: no API key for gemini-2.5-flash. "
            "Export GEMINI_API_KEY or GOOGLE_API_KEY to enable it."
        )

    def test_all_skipped_error(self) -> None:
        assert all_skipped_error() == (
            "[red]✗[/red] every configured evaluator would be skipped: none of their "
            "models has an API key, so nothing would score this run. Export a key above, "
            "or add an evaluator that needs none."
        )


def _gap(**overrides: Any) -> EvaluatorKeyGap:
    fields: dict[str, Any] = {
        "evaluator_name": "semantic.cosine",
        "kind": "semantic",
        "label": "semantic",
        "model": "openai/text-embedding-3-small",
        "env_vars": ("OPENAI_API_KEY",),
        "blocking": True,
    }
    return EvaluatorKeyGap(**{**fields, **overrides})


class TestGapDetail:
    def test_blocking(self) -> None:
        assert gap_detail(_gap()) == (
            "semantic uses openai/text-embedding-3-small; export OPENAI_API_KEY "
            "(or set blocking: false to run without it)"
        )

    def test_advisory(self) -> None:
        assert gap_detail(_gap(blocking=False)) == (
            "semantic uses openai/text-embedding-3-small; export OPENAI_API_KEY "
            "(skipped until then)"
        )

    def test_forced_gate_names_the_entry_not_the_flag(self) -> None:
        gap = _gap(blocked_by="args")
        assert forced_gate_hint(gap) == (
            "blocking because tool_arguments `args` uses the semantic strategy — "
            "export OPENAI_API_KEY or change that strategy"
        )
        assert gap_detail(gap) == (
            f"semantic uses openai/text-embedding-3-small; {forced_gate_hint(gap)}"
        )
        assert "blocking: false" not in gap_detail(gap)
