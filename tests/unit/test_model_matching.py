"""Matching a capture's recorded model id against ``defaults.source_model``."""

from __future__ import annotations

import pytest

from evalshift_cli.models.matching import describe_models, model_filter_applies, same_model


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("claude-sonnet-5-20260301", "claude-sonnet-5"),
        ("anthropic/claude-sonnet-5", "claude-sonnet-5"),
        ("google/gemini-3.8-flash", "gemini-3.8-flash"),
        ("gpt-5-2026-08-07", "openai/gpt-5"),
        ("Claude-Sonnet-5", "claude-sonnet-5"),
    ],
)
def test_same_model_matches_prefix_case_and_snapshot_variants(a: str, b: str) -> None:
    assert same_model(a, b)
    assert same_model(b, a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("claude-sonnet-5", "claude-sonnet-5-5"),
        ("gpt-5", "gpt-5-mini"),
        ("gpt-4o", "gpt-5"),
        ("gpt-4o-mini", "gpt-5-mini"),
        ("", "claude-sonnet-5"),
        ("m", "claude-sonnet-5"),
    ],
)
def test_same_model_keeps_distinct_models_apart(a: str, b: str) -> None:
    assert not same_model(a, b)


def test_filter_applies_when_any_recorded_model_matches() -> None:
    assert model_filter_applies(["claude-sonnet-5-20260301", "gpt-5"], "claude-sonnet-5")


def test_filter_is_off_without_a_source_model() -> None:
    assert not model_filter_applies(["claude-sonnet-5"], None)


def test_filter_is_off_when_nothing_recorded_a_model() -> None:
    assert not model_filter_applies(["", ""], "claude-sonnet-5")


def test_guardrail_filter_is_off_when_no_recorded_model_matches() -> None:
    assert not model_filter_applies(["claude-sonnet-5"] * 3, "claude-sonnet-5-5")


def test_describe_models_counts_the_most_recorded_first() -> None:
    recorded = ["a"] * 3 + ["b"] + ["c"] * 2 + ["d"] + [""]
    assert describe_models(recorded) == "a ×3, c ×2, b ×1, +1 more"
