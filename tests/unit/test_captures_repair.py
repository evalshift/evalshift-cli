"""Finding promoted cases an earlier sync built from stringified blocks, and rebuilding them."""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.repair import (
    is_repair_candidate,
    repaired_case,
    select_repairs,
    stringified_content,
    stringified_warnings,
)
from evalshift_cli.suite.models import SuiteExample

TOOLSET_REF = "sha256:" + "ab" * 32
#: What str() of a Claude block list looks like -- the bug's signature.
REPR_BLOCKS = str(
    [{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t", "name": "s", "input": {}}]
)


def _example(
    *,
    id_: str = "ex",
    inputs: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> SuiteExample:
    return SuiteExample.model_validate(
        {
            "id": id_,
            "inputs": {"input": "q"} if inputs is None else inputs,
            "history": history,
            "toolset_ref": TOOLSET_REF,
            **extra,
        }
    )


def _case(
    name: str,
    *,
    capture: str | None = None,
    example: SuiteExample | None = None,
    suite: str = "alpha",
    conversation_id: str | None = None,
) -> PromotedCase:
    return PromotedCase(
        name=name,
        suite=suite,
        from_capture=capture or name,
        example=example or _example(id_=name),
        conversation_id=conversation_id,
    )


def _broken(name: str, **kwargs: Any) -> PromotedCase:
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": REPR_BLOCKS}]
    return _case(name, example=_example(id_=name, history=history), **kwargs)


# --- the signature --------------------------------------------------------------------------


def test_signature_found_in_history() -> None:
    ex = _example(
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": REPR_BLOCKS}]
    )
    assert stringified_content(ex) == "history message 2"


def test_signature_found_in_inputs_after_leading_whitespace() -> None:
    assert stringified_content(_example(inputs={"input": "  " + REPR_BLOCKS})) == "inputs['input']"


@pytest.mark.parametrize(
    "text",
    [
        json.dumps([{"type": "order", "id": 1}]),  # a JSON tool result: opens with [{"
        str([{"id": 1}, {"id": 2}]),  # a repr'd list of untyped dicts
        "[{'type': 'text'",  # truncated, does not evaluate
        "[note] see above",
        "",
        "[]",
        "[{'type': __import__('os')}]",  # not a literal: never evaluated
    ],
)
def test_signature_ignores_lookalikes(text: str) -> None:
    ex = _example(history=[{"role": "user", "content": text}])
    assert stringified_content(ex) is None


def test_signature_ignores_an_invalid_escape_silently() -> None:
    # str() of a block list always doubles a backslash, so a lone ``\d`` -- an invalid escape
    # that literal_eval only warns about -- can't be one; it must not leak a SyntaxWarning.
    text = "[{'type': 'regex', 'pattern': '\\d+'}]"
    ex = _example(history=[{"role": "user", "content": text}])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert stringified_content(ex) is None
    assert caught == []


def test_repair_candidates_are_stringified_or_input_less() -> None:
    assert is_repair_candidate(_example(history=[{"role": "assistant", "content": REPR_BLOCKS}]))
    assert is_repair_candidate(_example(inputs={}))
    assert not is_repair_candidate(_example())


# --- selection ------------------------------------------------------------------------------


def test_select_repairs_collects_candidates_and_their_conversation_siblings() -> None:
    broken = _broken("cap_t1", conversation_id="conv")
    sibling = _case("cap_t0", conversation_id="conv")
    elsewhere = _case("cap_x", conversation_id="other")
    other_suite = _case("cap_y", suite="beta", conversation_id="conv")
    empty = _case("cap_empty", example=_example(id_="cap_empty", inputs={}))
    selection = select_repairs([broken, sibling, elsewhere, other_suite, empty])
    assert [c.case.name for c in selection.candidates] == ["cap_t1", "cap_empty"]
    assert [c.stringified for c in selection.candidates] == ["history message 2", None]
    assert selection.refetch_ids == {"cap_t1", "cap_t0", "cap_empty"}
    assert selection.cases_by_capture() == {"cap_t1": [broken], "cap_empty": [empty]}


def test_stringified_warnings_count_per_suite_and_skip_input_less_cases() -> None:
    selection = select_repairs(
        [
            _broken("a"),
            _broken("b"),
            _broken("c", suite="beta"),
            _case("d", example=_example(id_="d", inputs={})),
        ]
    )
    assert stringified_warnings(selection) == [
        "alpha: 2 promoted case(s) carry message blocks an earlier sync stringified (e.g. a); "
        "run evalshift capture sync --repair to rebuild them.",
        "beta: 1 promoted case(s) carry message blocks an earlier sync stringified (e.g. c); "
        "run evalshift capture sync --repair to rebuild them.",
    ]


# --- rebuilding -----------------------------------------------------------------------------


def test_repaired_case_replaces_only_inputs_and_history() -> None:
    stored = _case(
        "billing_q",
        capture="cap_1",
        example=_example(
            id_="billing_q",
            inputs={"input": REPR_BLOCKS},
            history=[{"role": "assistant", "content": REPR_BLOCKS}],
            tags=["captured", "billing"],
            expected_tools=[{"tool_name": "search", "arguments": {"q": "x"}}],
        ),
    )
    rebuilt = _example(
        id_="cap_1",
        inputs={"input": "And the price?"},
        history=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "toolu_1", "name": "search", "arguments": {}}],
            },
            {"role": "tool", "content": "res", "tool_call_id": "toolu_1"},
        ],
    )
    fixed = repaired_case(stored, rebuilt, promoted_at="2026-10-10T00:00:00+00:00")
    assert fixed is not None
    assert fixed.example.inputs == {"input": "And the price?"}
    assert fixed.example.history == rebuilt.history
    assert fixed.name == "billing_q"
    assert fixed.example.id == "billing_q"
    assert fixed.example.tags == ["captured", "billing"]
    assert fixed.example.expected_tools == stored.example.expected_tools
    assert fixed.promoted_at == "2026-10-10T00:00:00+00:00"
    assert fixed.from_capture == "cap_1"


def test_repaired_case_rebuilds_history_when_inputs_already_match() -> None:
    stored = _case(
        "c",
        example=_example(
            id_="c",
            inputs={"input": "And the price?"},
            history=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": REPR_BLOCKS},
            ],
        ),
    )
    rebuilt = _example(
        id_="c",
        inputs={"input": "And the price?"},
        history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    )
    fixed = repaired_case(stored, rebuilt, promoted_at="t")
    assert fixed is not None
    assert fixed.example.history == rebuilt.history
    assert fixed.example.inputs == {"input": "And the price?"}


def test_repaired_case_is_none_when_nothing_changed() -> None:
    stored = _case("c", example=_example(id_="c", inputs={}))
    assert repaired_case(stored, _example(id_="other", inputs={}), promoted_at="t") is None
