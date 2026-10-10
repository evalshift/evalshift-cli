# Native Message Shapes + `capture sync --repair` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `capture sync` / `capture promote` read every SDK wrapper's native message shape (Claude blocks, OpenAI Chat, OpenAI Responses, LangChain roles) into real `tool_calls` / `tool` history with no stringified blocks, block turns carrying non-text content, and let users find and rebuild suites an earlier sync broke — including from a fresh clone whose captures live only in S3/GCS/Azure.

**Architecture:** One pure shape-keyed translator (`captures/native_messages.py`) runs at the top of promotion's existing messages-list recovery, so everything downstream keeps seeing the canonical shape the Gemini adapter already writes. A second pure module (`captures/repair.py`) detects the bug's signature in promoted cases and rebuilds only their `inputs` / `history`; `capture sync` wires it in behind `--repair`, and `fetch_captures` gains a `force_ids` escape from its promoted-id and `--since` skips.

**Tech Stack:** Python 3.11+, pydantic v2, Typer + Rich, pytest + `typer.testing.CliRunner`, uv. Fixture generation only: the sibling `evalshift-sdk` checkout with `anthropic` and `openai`.

**Spec:** `docs/superpowers/specs/2026-10-09-native-message-shapes-design.md` — read it first; this plan argues from it.

## Global Constraints

- Python floor 3.11 (`ruff.toml` `target-version = "py311"`): no 3.12+ syntax (no `type X = …`, no PEP 695 generics).
- Every module starts with `from __future__ import annotations`; public functions/classes get Google-style docstrings; `mypy --strict src/evalshift_cli` stays clean.
- No new runtime or dev dependencies. No change to the suite schema: `ChatMessage`, `HistoryToolCall`, `SuiteExample`, `PromotedCase` stay as they are.
- Message content is never `str()`-ed. `_history_message` raises `TypeError` on non-string content.
- New blocked reason literal: `"unreplayable_content"`. Unreplayable line formats, verbatim: ``recorded message N (<recorded role>) carries a `<type>` block``, ``recorded message N carries a tool call with no name``, ``recorded item N carries a `<type>` item``.
- User-facing strings, verbatim:
  - blocked: ``capture '<id>' carries content a text replay can't carry — <first line>[ (+N more)]. A replay without it is not ground truth; re-capture without it or hand-write the case.``
  - sync summary clause: `skipped N capture(s) with non-text content a text replay can't carry (--allow-errored does not apply)`
  - sync warning: `<suite>: N promoted case(s) carry message blocks an earlier sync stringified (e.g. <case>); run evalshift capture sync --repair to rebuild them.`
  - sync summary clauses: `repaired N case(s)` (whenever `--repair` is set), `left N case(s) for --repair`
  - `no promoted cases need repair.`
  - `can't repair N case(s) — their capture is no longer local or in the store (<names>): delete the case file or re-capture.`
  - `can't repair <case>: <blocked message>`
  - validate advisory: `N example(s) carry message blocks an earlier capture sync stringified (e.g. <id>); run evalshift capture sync --repair.`
  - bad arguments warning: ``recorded message N: tool call `<name>` (<id>) had arguments that are not a JSON object; replayed with {}``
- Gate ("green" means all four, as CI): `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest` (coverage floor 90).
- Commit with `uv run --with pre-commit git commit …` — this repo's shared git hook points at a deleted venv. Never `--no-verify`. Conventional Commits, ending with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `docs/superpowers/` is gitignored: add files there with `git add -f`.
- Docs travel with behaviour (CLAUDE.md): `DOCS.md`, `llms-full.txt`, the matching `docs/` pages and `CHANGELOG.md` `## [Unreleased]` change in Task 8.

## Review Focus

- A tool result that is a JSON array of typed objects (`[{"type": "order", "id": 1}]`) must not be flagged as stringified — the signature requires Python repr's `[{'` opening (Task 4 test `test_signature_ignores_lookalikes`).
- Running `capture sync --repair` twice: the second run finds nothing and rewrites no byte (Task 6 test `test_repair_twice_is_a_no_op`).
- A candidate whose rebuild is blocked (its toolset sidecar is gone) is named `can't repair`, left untouched, and sync still exits 0 (Task 6 test `test_repair_leaves_a_case_whose_rebuild_is_blocked`).
- A corrupt promoted case file in an unrelated suite warns and does not stop `--repair` from fixing the others (Task 6 test `test_repair_survives_a_corrupt_case_file_elsewhere`).
- Claude parallel tool calls answered in one user message become `tool` messages in block order, keyed by id, with any text after them (Task 1 test `test_parallel_tool_results_keep_block_order_and_precede_shared_text`).

---

### Task 1: The shape translator

**Files:**
- Create: `src/evalshift_cli/captures/native_messages.py`
- Test: `tests/unit/test_native_messages.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `NormalizedMessages(messages: list[dict[str, Any]], unreplayable: list[str], warnings: list[str])` (frozen dataclass) and `normalize_messages(raw: Any) -> NormalizedMessages | None`. Output messages are `{"role": str, "content": str}` plus optional `"tool_calls": [{"id": str | None, "name": str, "arguments": dict}]` (assistant) and optional `"tool_call_id": str` (tool).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_native_messages.py`:

```python
"""Tests for translating recorded provider-native messages into the replay shape."""

from __future__ import annotations

from typing import Any

import pytest

from evalshift_cli.captures.native_messages import NormalizedMessages, normalize_messages


def _norm(raw: Any) -> NormalizedMessages:
    result = normalize_messages(raw)
    assert result is not None
    return result


#: A Claude follow-up turn as wrap_anthropic records it (system folded in as a message).
CLAUDE_FOLLOW_UP: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "search", "signature": "sig"},
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "evalshift"}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "results"}],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "A regression tool."}]},
    {"role": "user", "content": "And the price?"},
]

#: An OpenAI Chat history built from message.model_dump().
OPENAI_CHAT: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "q1"},
    {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "x"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "res"},
    {"role": "assistant", "content": "a1", "refusal": None, "tool_calls": None},
    {"role": "user", "content": "q2"},
]

#: An OpenAI Responses input list.
RESPONSES: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
    {"type": "reasoning", "id": "rs_1", "summary": []},
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Checking."}],
    },
    {"type": "function_call", "call_id": "fc_1", "name": "weather", "arguments": '{"city": "Paris"}'},
    {"type": "function_call", "call_id": "fc_2", "name": "time", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "fc_1", "output": "sunny"},
    {
        "type": "function_call_output",
        "call_id": "fc_2",
        "output": [{"type": "input_text", "text": "noon"}],
    },
    {"role": "user", "content": "thanks"},
]

#: The LangChain callback over ChatAnthropic: message.type roles, Claude blocks in content.
LANGCHAIN: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "human", "content": "hi"},
    {
        "role": "ai",
        "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
        ],
    },
    {"role": "tool", "content": "res"},
    {"role": "human", "content": "thanks"},
]

#: Already the replay shape (what wrap_genai writes, minus its extra keys).
CANONICAL: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "q1"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": None, "name": "convert", "arguments": {"miles": 3.5}}],
    },
    {"role": "tool", "content": '{"km": 5.6}'},
    {"role": "tool", "content": "x", "tool_call_id": "c9"},
    {"role": "assistant", "content": "5.6 km"},
    {"role": "user", "content": "q2"},
]


# --- what counts as a messages list ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        [],
        ["opaque", "payload"],
        [{"content": "no role"}],
        [{"role": "user", "content": "hi"}, "stray"],
        [{"type": "custom_thing"}],
        "not a list",
        None,
    ],
)
def test_non_messages_input_returns_none(raw: Any) -> None:
    assert normalize_messages(raw) is None


# --- roles ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recorded", "canonical"),
    [("human", "user"), ("ai", "assistant"), ("model", "assistant"), ("developer", "system")],
)
def test_role_aliases(recorded: str, canonical: str) -> None:
    assert _norm([{"role": recorded, "content": "x"}]).messages == [
        {"role": canonical, "content": "x"}
    ]


def test_legacy_function_role_passes_through_unaliased() -> None:
    # Promotion drops it with its "unrecognised role" warning: the legacy function_call it
    # answers is not carried, so as a tool message it could only replay unpaired.
    assert _norm([{"role": "function", "content": "out"}]).messages == [
        {"role": "function", "content": "out"}
    ]


# --- content --------------------------------------------------------------------------------


def test_absent_and_none_content_become_empty_strings() -> None:
    assert _norm([{"role": "user"}, {"role": "assistant", "content": None}]).messages == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]


def test_text_blocks_of_every_kind_join_with_newlines() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "input_text", "text": "b"},
                "c",
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "d"},
                {"type": "refusal", "refusal": "e"},
            ],
        },
    ]
    assert _norm(raw).messages == [
        {"role": "user", "content": "a\nb\nc"},
        {"role": "assistant", "content": "d\ne"},
    ]


def test_a_single_block_dict_is_read_as_a_one_block_list() -> None:
    assert _norm([{"role": "user", "content": {"type": "text", "text": "hi"}}]).messages == [
        {"role": "user", "content": "hi"}
    ]


def test_a_prompt_cached_system_prompt_becomes_its_text() -> None:
    raw = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a support agent.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "hi"},
    ]
    assert _norm(raw).messages[0] == {"role": "system", "content": "You are a support agent."}


def test_thinking_is_dropped_without_a_warning() -> None:
    raw = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "answer"},
            ],
        }
    ]
    result = _norm(raw)
    assert result.messages == [{"role": "assistant", "content": "answer"}]
    assert result.warnings == []
    assert result.unreplayable == []


# --- Claude ---------------------------------------------------------------------------------


def test_claude_tool_round_becomes_tool_calls_and_a_tool_message() -> None:
    assert _norm(CLAUDE_FOLLOW_UP).messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is evalshift?"},
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [{"id": "toolu_1", "name": "search", "arguments": {"q": "evalshift"}}],
        },
        {"role": "tool", "content": "results", "tool_call_id": "toolu_1"},
        {"role": "assistant", "content": "A regression tool."},
        {"role": "user", "content": "And the price?"},
    ]


def test_parallel_tool_results_keep_block_order_and_precede_shared_text() -> None:
    raw = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_a", "name": "a", "input": {}},
                {"type": "tool_use", "id": "toolu_b", "name": "b", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_a",
                    "content": [{"type": "text", "text": "A"}],
                },
                {"type": "tool_result", "tool_use_id": "toolu_b"},
                {"type": "text", "text": "also, hurry"},
            ],
        },
    ]
    assert _norm(raw).messages[1:] == [
        {"role": "tool", "content": "A", "tool_call_id": "toolu_a"},
        {"role": "tool", "content": "", "tool_call_id": "toolu_b"},
        {"role": "user", "content": "also, hurry"},
    ]


def test_tool_result_is_error_is_not_carried() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": True}
            ],
        }
    ]
    assert _norm(raw).messages == [{"role": "tool", "content": "boom", "tool_call_id": "t"}]


# --- unreplayable content -------------------------------------------------------------------


@pytest.mark.parametrize(
    "block_type",
    [
        "image",
        "document",
        "image_url",
        "input_image",
        "input_file",
        "input_audio",
        "file",
        "server_tool_use",
        "web_search_tool_result",
        "brand_new_block",
    ],
)
def test_non_text_blocks_are_unreplayable_and_skipped(block_type: str) -> None:
    raw = [
        {
            "role": "user",
            "content": [{"type": block_type}, {"type": "text", "text": "what is this?"}],
        }
    ]
    result = _norm(raw)
    assert result.unreplayable == [f"recorded message 1 (user) carries a `{block_type}` block"]
    assert result.messages == [{"role": "user", "content": "what is this?"}]


def test_an_image_inside_a_tool_result_is_unreplayable() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": [{"type": "image", "source": {}}],
                }
            ],
        }
    ]
    assert _norm(raw).unreplayable == ["recorded message 1 (user) carries a `image` block"]


def test_a_tool_use_with_no_name_is_unreplayable_not_a_crash() -> None:
    raw = [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "input": {}}]}]
    result = _norm(raw)
    assert result.unreplayable == ["recorded message 1 carries a tool call with no name"]
    assert result.messages == [{"role": "assistant", "content": ""}]


def test_an_unknown_responses_item_is_unreplayable() -> None:
    raw = [{"role": "user", "content": "search the web"}, {"type": "web_search_call", "id": "ws_1"}]
    assert _norm(raw).unreplayable == ["recorded item 2 carries a `web_search_call` item"]


# --- OpenAI Chat ----------------------------------------------------------------------------


def test_openai_nested_tool_calls_are_flattened_and_arguments_parsed() -> None:
    raw = [
        {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "x"}'},
                },
                {"id": "call_2", "type": "function", "function": {"name": "noop", "arguments": ""}},
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "dicty", "arguments": {"k": 1}},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res"},
    ]
    result = _norm(raw)
    assert result.messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "search", "arguments": {"q": "x"}},
                {"id": "call_2", "name": "noop", "arguments": {}},
                {"id": "call_3", "name": "dicty", "arguments": {"k": 1}},
            ],
        },
        {"role": "tool", "content": "res", "tool_call_id": "call_1"},
    ]
    assert result.warnings == []


@pytest.mark.parametrize("arguments", ['{"q": ', "[1, 2]", '"just a string"'])
def test_arguments_that_are_not_a_json_object_replay_as_empty_with_a_warning(
    arguments: str,
) -> None:
    raw = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": arguments},
                }
            ],
        }
    ]
    result = _norm(raw)
    assert result.messages[0]["tool_calls"] == [{"id": "call_1", "name": "search", "arguments": {}}]
    assert result.warnings == [
        "recorded message 1: tool call `search` (call_1) had arguments that are not a JSON "
        "object; replayed with {}"
    ]


def test_an_openai_refusal_becomes_the_assistant_content() -> None:
    raw = [{"role": "assistant", "content": None, "refusal": "I can't help with that."}]
    assert _norm(raw).messages == [{"role": "assistant", "content": "I can't help with that."}]


def test_openai_chat_history() -> None:
    assert _norm(OPENAI_CHAT).messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "name": "search", "arguments": {"q": "x"}}],
        },
        {"role": "tool", "content": "res", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


# --- OpenAI Responses -----------------------------------------------------------------------


def test_responses_items_become_messages_tool_calls_and_tool_messages() -> None:
    assert _norm(RESPONSES).messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {"id": "fc_1", "name": "weather", "arguments": {"city": "Paris"}},
                {"id": "fc_2", "name": "time", "arguments": {}},
            ],
        },
        {"role": "tool", "content": "sunny", "tool_call_id": "fc_1"},
        {"role": "tool", "content": "noon", "tool_call_id": "fc_2"},
        {"role": "user", "content": "thanks"},
    ]


def test_a_function_call_after_a_non_assistant_message_opens_an_assistant_turn() -> None:
    raw = [
        {"role": "user", "content": "go"},
        {"type": "function_call", "call_id": "fc_1", "name": "run", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "fc_1", "output": {"ok": True}},
    ]
    assert _norm(raw).messages == [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "fc_1", "name": "run", "arguments": {}}],
        },
        {"role": "tool", "content": '{"ok": true}', "tool_call_id": "fc_1"},
    ]


# --- LangChain ------------------------------------------------------------------------------


def test_langchain_roles_with_claude_blocks() -> None:
    assert _norm(LANGCHAIN).messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "looking",
            "tool_calls": [{"id": "toolu_1", "name": "search", "arguments": {"q": "x"}}],
        },
        {"role": "tool", "content": "res"},
        {"role": "user", "content": "thanks"},
    ]


# --- invariants -----------------------------------------------------------------------------


def test_canonical_input_passes_through_unchanged() -> None:
    result = _norm(CANONICAL)
    assert result.messages == CANONICAL
    assert result.unreplayable == []
    assert result.warnings == []


@pytest.mark.parametrize(
    "raw",
    [CANONICAL, CLAUDE_FOLLOW_UP, OPENAI_CHAT, RESPONSES, LANGCHAIN],
    ids=["canonical", "claude", "openai_chat", "responses", "langchain"],
)
def test_normalizing_is_idempotent_and_content_is_always_a_string(
    raw: list[dict[str, Any]],
) -> None:
    once = _norm(raw).messages
    assert all(isinstance(message["content"], str) for message in once)
    assert _norm(once).messages == once
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_native_messages.py -q -p no:cacheprovider --no-cov`
Expected: collection error, `ModuleNotFoundError: No module named 'evalshift_cli.captures.native_messages'`.

- [ ] **Step 3: Write the module**

Create `src/evalshift_cli/captures/native_messages.py`:

```python
"""Translate a recorded messages list from a provider's own shape into the replay shape.

The SDK's adapters record ``model_call.input`` the way each provider takes it: Claude's
block-list ``content`` (``text`` / ``thinking`` / ``tool_use`` / ``tool_result`` blocks),
OpenAI Chat's nested ``tool_calls[].function`` with JSON-string arguments, OpenAI Responses
items (``function_call`` / ``function_call_output`` carry no ``role``), and LangChain's
``human`` / ``ai`` roles. Only ``wrap_genai`` writes the shape promotion replays -- ``content``
a string, ``tool_calls: [{id, name, arguments}]`` on assistant turns, ``tool_call_id`` on tool
turns. :func:`normalize_messages` is the one place the others are translated, keyed on each
element's *shape* and never on the model id or adapter: both lie behind gateways (LiteLLM,
OpenRouter), LangChain, and hand-written ``record_model_call``. Already-canonical input passes
through unchanged.

Content a text replay cannot carry -- an image, a document, a server-tool result, any block or
item type this module does not know -- is reported in ``unreplayable`` (promotion then blocks
the case) and skipped, so the rest still normalizes. Thinking and reasoning are dropped
silently: the Claude API ignores earlier turns' thinking itself, and a signature means nothing
to a different candidate model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: Recorded role -> replay role. Anything else passes through for promotion's "unrecognised
#: role" warning -- the legacy OpenAI ``function`` role included: the ``function_call`` it
#: answers is not carried, so as a ``tool`` message it could only replay unpaired.
_ROLE_ALIASES: dict[str, str] = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "system": "system",
    "developer": "system",
    "tool": "tool",
}
_TEXT_BLOCKS = frozenset({"text", "input_text", "output_text"})
_DROPPED_BLOCKS = frozenset({"thinking", "redacted_thinking"})
_KNOWN_ITEMS = frozenset({"message", "function_call", "function_call_output", "reasoning"})


@dataclass(frozen=True, slots=True)
class NormalizedMessages:
    """A recorded messages list in the replay shape.

    Attributes:
        messages: ``{"role", "content": str}`` dicts, plus ``tool_calls`` (each
            ``{"id": str | None, "name": str, "arguments": dict}``) on assistant messages
            and ``tool_call_id`` on tool messages when one was recorded.
        unreplayable: One line per element a text replay cannot carry, e.g.
            ``"recorded message 3 (user) carries a `image` block"``. Non-empty means the
            case must not be promoted.
        warnings: Non-fatal notes for the operator.
    """

    messages: list[dict[str, Any]]
    unreplayable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_messages(raw: Any) -> NormalizedMessages | None:
    """Translate ``raw`` into the replay shape, or return ``None`` when it is not a messages list.

    ``raw`` is a messages list when it is a non-empty list of dicts that each carry a string
    ``role`` or a string ``type``, and at least one carries a ``role`` or a known Responses item
    type (``message`` / ``function_call`` / ``function_call_output`` / ``reasoning``).

    Args:
        raw: A recorded ``model_call.input``.

    Returns:
        The translated messages with any unreplayable elements and warnings, or ``None``.
    """
    if not _is_messages_list(raw):
        return None
    builder = _Builder()
    for position, item in enumerate(raw, start=1):
        builder.add(position, item)
    return NormalizedMessages(builder.messages, builder.unreplayable, builder.warnings)


def _is_messages_list(raw: Any) -> bool:
    if not isinstance(raw, list) or not raw:
        return False
    if not all(
        isinstance(item, dict)
        and (isinstance(item.get("role"), str) or isinstance(item.get("type"), str))
        for item in raw
    ):
        return False
    return any(
        isinstance(item.get("role"), str) or item.get("type") in _KNOWN_ITEMS for item in raw
    )


class _Builder:
    """Accumulates the normalized messages of one :func:`normalize_messages` call."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.unreplayable: list[str] = []
        self.warnings: list[str] = []

    def add(self, position: int, item: dict[str, Any]) -> None:
        kind = item.get("type")
        role = item.get("role")
        if kind == "function_call":
            self._function_call(position, item)
        elif kind == "function_call_output":
            self._function_call_output(position, item)
        elif kind == "reasoning":
            return
        elif isinstance(role, str) and kind in (None, "message"):
            self._message(position, role, item)
        else:
            self.unreplayable.append(f"recorded item {position} carries a `{kind}` item")

    def _message(self, position: int, recorded_role: str, item: dict[str, Any]) -> None:
        role = _ROLE_ALIASES.get(recorded_role, recorded_role)
        texts, calls, results = self._content(position, recorded_role, item.get("content"))
        # Claude puts tool results in a user turn; they replay as tool messages ahead of it.
        self.messages.extend(results)
        if role == "assistant":
            calls = [*self._tool_calls(position, item.get("tool_calls")), *calls]
        if results and not texts and not calls:
            return  # a user turn made only of tool results
        content = "\n".join(texts)
        refusal = item.get("refusal")
        if role == "assistant" and not content and isinstance(refusal, str):
            content = refusal
        message: dict[str, Any] = {"role": role, "content": content}
        if calls:
            message["tool_calls"] = calls
        tool_call_id = item.get("tool_call_id")
        if role == "tool" and tool_call_id is not None and tool_call_id != "":
            message["tool_call_id"] = str(tool_call_id)
        self.messages.append(message)

    def _content(
        self, position: int, recorded_role: str, content: Any
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """Split recorded ``content`` into text pieces, ``tool_use`` calls and tool messages."""
        if content is None:
            return [], [], []
        if isinstance(content, str):
            return [content], [], []
        blocks = [content] if isinstance(content, dict) else content
        if not isinstance(blocks, list):
            return [json.dumps(content)], [], []
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        where = f"recorded message {position} ({recorded_role})"
        for block in blocks:
            if isinstance(block, str):
                texts.append(block)
                continue
            if not isinstance(block, dict):
                self.unreplayable.append(f"{where} carries a {type(block).__name__} element")
                continue
            kind = block.get("type")
            if kind in _TEXT_BLOCKS:
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
            elif kind == "refusal":
                refusal = block.get("refusal")
                if isinstance(refusal, str):
                    texts.append(refusal)
            elif kind in _DROPPED_BLOCKS:
                continue
            elif kind == "tool_use":
                call = self._call(position, block.get("id"), block.get("name"), block.get("input"))
                if call is not None:
                    calls.append(call)
            elif kind == "tool_result":
                results.append(self._tool_result(position, recorded_role, block))
            else:
                self.unreplayable.append(f"{where} carries a `{kind}` block")
        return texts, calls, results

    def _tool_result(
        self, position: int, recorded_role: str, block: dict[str, Any]
    ) -> dict[str, Any]:
        texts, _calls, _results = self._content(position, recorded_role, block.get("content"))
        message: dict[str, Any] = {"role": "tool", "content": "\n".join(texts)}
        tool_use_id = block.get("tool_use_id")
        if tool_use_id is not None and tool_use_id != "":
            message["tool_call_id"] = str(tool_use_id)
        return message

    def _tool_calls(self, position: int, raw: Any) -> list[dict[str, Any]]:
        """A message's own ``tool_calls``: canonical entries, or OpenAI's nested ``function``."""
        if not isinstance(raw, list):
            return []
        calls: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                self.unreplayable.append(
                    f"recorded message {position} carries a tool call that is not an object"
                )
                continue
            function = entry.get("function")
            source = function if isinstance(function, dict) else entry
            call = self._call(
                position, entry.get("id"), source.get("name"), source.get("arguments")
            )
            if call is not None:
                calls.append(call)
        return calls

    def _call(
        self, position: int, call_id: Any, name: Any, arguments: Any
    ) -> dict[str, Any] | None:
        if not isinstance(name, str) or not name:
            self.unreplayable.append(f"recorded message {position} carries a tool call with no name")
            return None
        return {
            "id": None if call_id is None or call_id == "" else str(call_id),
            "name": name,
            "arguments": self._arguments(position, name, call_id, arguments),
        }

    def _arguments(self, position: int, name: str, call_id: Any, arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            if not arguments.strip():
                return {}
            try:
                decoded = json.loads(arguments)
            except ValueError:
                decoded = None
            if isinstance(decoded, dict):
                return decoded
        self.warnings.append(
            f"recorded message {position}: tool call `{name}` ({call_id}) had arguments that "
            "are not a JSON object; replayed with {}"
        )
        return {}

    def _function_call(self, position: int, item: dict[str, Any]) -> None:
        call = self._call(position, item.get("call_id"), item.get("name"), item.get("arguments"))
        if call is None:
            return
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "assistant":
            last.setdefault("tool_calls", []).append(call)
        else:
            self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})

    def _function_call_output(self, position: int, item: dict[str, Any]) -> None:
        output = item.get("output")
        if isinstance(output, dict):
            content = json.dumps(output)
        else:
            texts, _calls, _results = self._content(position, "tool", output)
            content = "\n".join(texts)
        message: dict[str, Any] = {"role": "tool", "content": content}
        call_id = item.get("call_id")
        if call_id is not None and call_id != "":
            message["tool_call_id"] = str(call_id)
        self.messages.append(message)


__all__ = ["NormalizedMessages", "normalize_messages"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_native_messages.py -q -p no:cacheprovider --no-cov`
Expected: all pass.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check src/evalshift_cli/captures/native_messages.py tests/unit/test_native_messages.py && uv run ruff format src/evalshift_cli/captures/native_messages.py tests/unit/test_native_messages.py && uv run mypy --strict src/evalshift_cli`
Expected: no errors (format may rewrap lines; that is fine).

- [ ] **Step 6: Commit**

```bash
git add src/evalshift_cli/captures/native_messages.py tests/unit/test_native_messages.py
uv run --with pre-commit git commit -m "feat(captures): translate provider-native message shapes into the replay shape

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Promotion reads native shapes and blocks unreplayable content

**Files:**
- Modify: `src/evalshift_cli/captures/promote.py` (imports; `BlockedReason`; `BuiltExample`; `RecoveredInput`; `build_example_from_capture` ~line 300-360; `_looks_like_messages_list` ~903 removed; `_recover_inputs` ~911; `_history_message` ~956; `_recover_inputs_from_messages` ~1014)
- Modify: `src/evalshift_cli/cli/commands/capture.py` (`capture_sync` counters + summary)
- Test: `tests/unit/test_captures_promote.py`, `tests/unit/test_cli_capture.py`

**Interfaces:**
- Consumes: `normalize_messages` (Task 1).
- Produces: `BlockedReason = Literal["errored", "no_toolset", "multi_toolset", "unreplayable_content"]` exported from `captures/promote.py`; `RecoveredInput.unreplayable: list[str]`; `BuiltExample.blocked_reason: BlockedReason | None`.

- [ ] **Step 1: Write the failing promotion tests**

In `tests/unit/test_captures_promote.py` (it already imports `pytest`, `build_example_from_capture`, `PromoteOptions` and defines `_envelope`, `_messages_call`, `_final`, `_error`), add `_history_message` to the existing `from evalshift_cli.captures.promote import (...)` block and a top-level `from evalshift_cli.suite.models import HistoryToolCall`, then append:

```python
# ---------------------------------------------------------------------------
# provider-native message shapes
# ---------------------------------------------------------------------------

_CLAUDE_FOLLOW_UP: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "search", "signature": "sig_abc"},
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "evalshift"}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "results"}],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "A regression tool."}]},
    {"role": "user", "content": "And the price?"},
]

_IMAGE_TURN: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
            {"type": "text", "text": "What is in this screenshot?"},
        ],
    }
]


def test_claude_blocks_become_tool_calls_and_tool_messages_in_history() -> None:
    events = [_messages_call(0, messages=_CLAUDE_FOLLOW_UP), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is None
    assert built.example.inputs == {"input": "And the price?"}
    history = built.example.history
    assert history is not None
    assert [m.role for m in history] == ["system", "user", "assistant", "tool", "assistant"]
    assert history[2].content == "Let me look."
    assert history[2].tool_calls == [
        HistoryToolCall(id="toolu_1", name="search", arguments={"q": "evalshift"})
    ]
    assert history[3].tool_call_id == "toolu_1"
    assert history[3].content == "results"
    assert not any("'type'" in m.content or "sig_abc" in m.content for m in history)


def test_openai_chat_history_with_null_content_no_longer_fails() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res"},
        {"role": "assistant", "content": "a1", "refusal": None, "tool_calls": None},
        {"role": "user", "content": "q2"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"input": "q2"}
    history = built.example.history
    assert history is not None
    assert history[2].tool_calls == [HistoryToolCall(id="call_1", name="search", arguments={"q": "x"})]
    assert history[3].tool_call_id == "call_1"


def test_an_assistant_message_without_a_content_key_still_recovers_inputs() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "s", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "res"},
        {"role": "user", "content": "q2"},
    ]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"input": "q2"}
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["user", "assistant", "tool"]


def test_langchain_roles_recover_inputs() -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "human", "content": "hi"}]
    events = [_messages_call(0, messages=messages), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.example.inputs == {"input": "hi"}
    assert built.example.history is not None
    assert [m.role for m in built.example.history] == ["system"]


def test_non_text_content_blocks_the_capture() -> None:
    events = [_messages_call(0, messages=_IMAGE_TURN), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked_reason == "unreplayable_content"
    assert built.blocked == (
        "capture 'cap_abc' carries content a text replay can't carry — recorded message 1 "
        "(user) carries a `image` block. A replay without it is not ground truth; re-capture "
        "without it or hand-write the case."
    )


def test_unreplayable_content_counts_the_rest() -> None:
    turn = [
        {"role": "user", "content": [{"type": "image"}, {"type": "document"}, {"type": "file"}]},
    ]
    events = [_messages_call(0, messages=turn), _final("done", 1)]
    built = build_example_from_capture(_envelope(events=events), PromoteOptions())
    assert built.blocked is not None
    assert "carries a `image` block (+2 more)." in built.blocked


def test_unreplayable_content_wins_over_errored_even_with_allow_errored() -> None:
    events = [_messages_call(0, messages=_IMAGE_TURN), _error("boom", 1)]
    built = build_example_from_capture(
        _envelope(events=events), PromoteOptions(allow_errored=True)
    )
    assert built.blocked_reason == "unreplayable_content"


def test_history_message_refuses_non_string_content() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        _history_message({"role": "assistant", "content": [{"type": "text", "text": "x"}]}, "assistant")
```

- [ ] **Step 2: Write the failing CLI tests**

Append to `tests/unit/test_cli_capture.py` (it defines `_write_capture`, `_messages_events`, `_invoke`):

```python
_IMAGE_TURN: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
            {"type": "text", "text": "What is in this screenshot?"},
        ],
    }
]


def test_promote_unreplayable_refusal_does_not_suggest_allow_errored(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_img", suite="alpha", events=_messages_events(_IMAGE_TURN))
    result = _invoke(["promote", "cap_img"], tmp_path)
    out = " ".join(result.stdout.split())
    assert result.exit_code == 1
    assert "carries a `image` block" in out
    assert "--allow-errored" not in out
    assert not (tmp_path / "suites" / "alpha" / "cap_img.json").exists()


def test_sync_counts_unreplayable_captures_under_their_own_reason(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_img", suite="alpha", events=_messages_events(_IMAGE_TURN))
    _write_capture(tmp_path, capture_id="cap_ok", suite="alpha")
    result = _invoke(["sync", "--print", "--config", str(tmp_path / "evalshift.yaml")], tmp_path)
    out = " ".join(result.stdout.split())
    assert result.exit_code == 0, result.stdout
    assert (
        "skipped 1 capture(s) with non-text content a text replay can't carry "
        "(--allow-errored does not apply)"
    ) in out
    assert "errored capture(s)" not in out
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_captures_promote.py tests/unit/test_cli_capture.py -q -p no:cacheprovider --no-cov -k "native or claude_blocks or openai_chat_history or content_key or langchain_roles or unreplayable or non_text or history_message_refuses"`
Expected: FAIL — the Claude test sees repr strings, the OpenAI test raises `ValidationError` (`HistoryToolCall name`), the content-key/LangChain tests get `inputs == {}`, the image tests are not blocked, `_history_message` does not raise.

- [ ] **Step 4: Add the blocked-reason alias and `RecoveredInput.unreplayable` in `promote.py`**

Imports: change `from dataclasses import dataclass, field` to `from dataclasses import dataclass, field, replace` and add `from evalshift_cli.captures.native_messages import normalize_messages` to the first-party block.

After `_CONTEXT_LOSS_MIN_GAP_TOKENS = 300` add:

```python
#: Why :func:`build_example_from_capture` refused a capture -- see :class:`BuiltExample`.
BlockedReason = Literal["errored", "no_toolset", "multi_toolset", "unreplayable_content"]
```

In `BuiltExample`, change the field to `blocked_reason: BlockedReason | None = None`, and in its docstring, after the sentence ending "…cannot yet be promoted as a single case).", insert:

```
            ``"unreplayable_content"`` covers a capture whose recorded messages carry
            content a text replay can't carry (an image, a document, a server-tool
            result, an unknown block type) — a replay without it is not ground truth.
```

and change "Neither of the latter two is rescued by ``--allow-errored``" to "None of the latter three is rescued by ``--allow-errored``".

In `RecoveredInput`, add the field and document it:

```python
    inputs: dict[str, Any]
    history: list[ChatMessage] | None = None
    unreplayable: list[str] = field(default_factory=list)
```

```
        unreplayable: One line per recorded element a text replay can't carry
            (see :func:`~evalshift_cli.captures.native_messages.normalize_messages`).
            Non-empty blocks the capture.
```

Add `"BlockedReason"` to `__all__`.

- [ ] **Step 5: Route messages lists through the translator**

Delete `_looks_like_messages_list` entirely. In `_recover_inputs`, replace

```python
    if _looks_like_messages_list(value):
        return _recover_inputs_from_messages(value, input_var=input_var, warnings=warnings)
```

with

```python
    if isinstance(value, list):
        normalized = normalize_messages(value)
        if normalized is not None:
            warnings.extend(normalized.warnings)
            recovered = _recover_inputs_from_messages(
                normalized.messages, input_var=input_var, warnings=warnings
            )
            return replace(recovered, unreplayable=list(normalized.unreplayable))
```

and replace the docstring's messages-list bullet (the one starting "A messages list (``[{"role": ..., "content": ...}, ...]``) is split:") with:

```
    * A messages list in any wrapped provider's shape is first translated by
      :func:`~evalshift_cli.captures.native_messages.normalize_messages`, then
      split: the last ``user`` message becomes the current-turn ``inputs``, and
      everything before it becomes ``history``. Unrecognised roles are dropped
      with an aggregated warning; content a text replay can't carry lands in
      ``unreplayable``.
```

- [ ] **Step 6: Make string content an invariant**

Replace `_history_message` and add `_text_content` right after it:

```python
def _history_message(
    msg: dict[str, Any],
    role: Literal["system", "user", "assistant", "tool"],
) -> ChatMessage:
    """Build one history entry from a normalized message dict.

    ``msg`` has been through :func:`~evalshift_cli.captures.native_messages.normalize_messages`,
    so its ``content`` is a string and its ``tool_calls`` are ``{id, name, arguments}``.
    Recorded ``tool_calls`` are carried through on ``assistant`` turns and
    ``tool_call_id`` on ``tool`` turns, so a replay sees the same agent loop
    production saw.

    Raises:
        TypeError: when ``content`` is not a string. Never ``str()`` it: that is how
            a Python repr of a block list once reached promoted suites.
    """
    raw_calls = msg.get("tool_calls") if role == "assistant" else None
    tool_calls: list[HistoryToolCall] | None = None
    if isinstance(raw_calls, list) and raw_calls:
        tool_calls = [
            HistoryToolCall(
                id=(str(c["id"]) if isinstance(c, dict) and c.get("id") is not None else None),
                name=str(c.get("name", "")) if isinstance(c, dict) else "",
                arguments=dict(c.get("arguments") or {}) if isinstance(c, dict) else {},
            )
            for c in raw_calls
        ]
    tool_call_id = str(msg["tool_call_id"]) if role == "tool" and msg.get("tool_call_id") else None
    return ChatMessage(
        role=role,
        content=_text_content(msg),
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def _text_content(msg: dict[str, Any]) -> str:
    """A normalized message's ``content`` -- a string, or a ``TypeError`` naming what it was."""
    content = msg.get("content", "")
    if not isinstance(content, str):
        raise TypeError(
            f"message content must be a string once normalized, got {type(content).__name__}"
        )
    return content
```

In `_recover_inputs_from_messages`, change the final `inputs={input_var: str(current.get("content", ""))},` to `inputs={input_var: _text_content(current)},`.

- [ ] **Step 7: Block unreplayable captures**

In `build_example_from_capture`, change `blocked_reason: Literal["errored", "no_toolset", "multi_toolset"] | None = None` to `blocked_reason: BlockedReason | None = None`, and between the `multi_toolset` branch and `elif errors:` insert:

```python
    elif recovered.unreplayable:
        # Unconditional, like the toolset refusals: --allow-errored cannot put back the
        # image (or document, or server-tool result) a text replay leaves out.
        more = len(recovered.unreplayable) - 1
        extra = f" (+{more} more)" if more else ""
        blocked = (
            f"capture {envelope.capture_id!r} carries content a text replay can't carry — "
            f"{recovered.unreplayable[0]}{extra}. A replay without it is not ground truth; "
            "re-capture without it or hand-write the case."
        )
        blocked_reason = "unreplayable_content"
```

- [ ] **Step 8: Count the new reason in `capture sync`**

In `src/evalshift_cli/cli/commands/capture.py`, `capture_sync`: add `skipped_unreplayable = 0` next to `skipped_multi_toolset = 0`; in the blocked branch insert before `else:`

```python
                elif built.blocked_reason == "unreplayable_content":
                    skipped_unreplayable += 1
```

and after the `skipped_multi_toolset` summary clause add:

```python
    if skipped_unreplayable:
        summary += (
            f", skipped {skipped_unreplayable} capture(s) with non-text content a text replay "
            "can't carry (--allow-errored does not apply)"
        )
```

(`capture promote` already offers `--allow-errored` only for `blocked_reason == "errored"`; no change there.)

- [ ] **Step 9: Run the promotion and CLI suites**

Run: `uv run pytest tests/unit/test_native_messages.py tests/unit/test_captures_promote.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_remote.py -q -p no:cacheprovider --no-cov`
Expected: all pass, including the pre-existing messages-list tests (`test_build_messages_list_unrecognised_role_dropped_with_aggregated_warning` still drops `function`; `test_build_non_messages_list_falls_back_to_existing_behavior` still falls back). If a pre-existing test relied on a list item without `content` being rejected, read it and decide with the spec — do not weaken the new behaviour.

- [ ] **Step 10: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format . && uv run mypy --strict src/evalshift_cli`

```bash
git add src/evalshift_cli/captures/promote.py src/evalshift_cli/cli/commands/capture.py tests/unit/test_captures_promote.py tests/unit/test_cli_capture.py
uv run --with pre-commit git commit -m "fix(captures): promote Claude, OpenAI and LangChain histories without stringifying blocks

Messages lists go through normalize_messages before history recovery, so
tool_use/tool_result blocks become tool_calls and tool messages, OpenAI
content: null no longer aborts sync, and turns carrying non-text content are
blocked as unreplayable_content.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fixture captures from the real SDK wrappers, end to end through `capture sync`

**Files:**
- Create: `tests/unit/fixtures/captures/native_shapes/generate.py`
- Create (generated): `tests/unit/fixtures/captures/native_shapes/{claude_agent,claude_image,openai_dump,openai_lean}/captures/<scenario>/*.json` and `…/toolsets/*.json`
- Test: `tests/unit/test_cli_capture_native_shapes.py`

**Interfaces:**
- Consumes: the CLI behaviour from Task 2.
- Produces: four ready-made `--base` trees, one per scenario; suite name == scenario name; conversation captures carry `turn_index` 0 and 1.

- [ ] **Step 1: Write the generator**

Create `tests/unit/fixtures/captures/native_shapes/generate.py`:

```python
"""Regenerate the native-shape capture fixtures from the real SDK wrappers.

Run from the evalshift-cli repo root, with an evalshift-sdk checkout beside it:

    EVALSHIFT_CAPTURE=1 EVALSHIFT_DIR="$(mktemp -d)" \
        uv run --project ../evalshift-sdk python tests/unit/fixtures/captures/native_shapes/generate.py

Each scenario drives ``wrap_anthropic`` / ``wrap_openai`` over a fake client that returns
scripted responses, so what lands on disk is exactly what the SDK records for a real agent
loop -- Claude's block-list content, OpenAI's nested ``tool_calls`` -- with no network. Every
scenario becomes a ready-made ``--base`` tree: ``<scenario>/captures/<scenario>/*.json`` plus
``<scenario>/toolsets/``.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from anthropic.types import Message
from evalshift import capture
from evalshift.adapters.anthropic import wrap_anthropic
from evalshift.adapters.openai import wrap_openai
from openai.types.chat import ChatCompletion

HERE = Path(__file__).parent
SCENARIOS = ("claude_agent", "claude_image", "openai_dump", "openai_lean")
QUESTIONS = ("What is evalshift?", "And what does it cost?")
CLAUDE_TOOLS = [
    {
        "name": "search",
        "description": "Search the docs",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the docs",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]
CACHED_SYSTEM = [
    {"type": "text", "text": "You are a support agent.", "cache_control": {"type": "ephemeral"}}
]


@capture.tool
def search(q: str) -> str:
    return f"results for {q}"


def _scripted(responses: list[Any]) -> Callable[..., Any]:
    it: Iterator[Any] = iter(responses)
    return lambda **_: next(it)


def _claude(content: list[dict[str, Any]], stop: str) -> Message:
    return Message.model_validate(
        {
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


def claude_agent() -> None:
    """Two turns; turn 1 thinks, then calls search; the system prompt is prompt-cached."""
    responses = [
        _claude(
            [
                {"type": "thinking", "thinking": "I should search.", "signature": "sig_abc"},
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "evalshift"}},
            ],
            "tool_use",
        ),
        _claude([{"type": "text", "text": "EvalShift is a regression tool."}], "end_turn"),
        _claude(
            [{"type": "tool_use", "id": "toolu_2", "name": "search", "input": {"q": "pricing"}}],
            "tool_use",
        ),
        _claude([{"type": "text", "text": "Pricing starts at $0."}], "end_turn"),
    ]
    client = wrap_anthropic(SimpleNamespace(messages=SimpleNamespace(create=_scripted(responses))))
    messages: list[dict[str, Any]] = []
    for turn, question in enumerate(QUESTIONS):
        with capture.agent_session(
            suite="claude_agent",
            redact=False,
            tools=CLAUDE_TOOLS,
            conversation_id="conv-claude",
            turn_index=turn,
            agent_input=question,
        ):
            messages.append({"role": "user", "content": question})
            while True:
                response = client.messages.create(
                    model="claude-opus-5",
                    max_tokens=1024,
                    system=CACHED_SYSTEM,
                    tools=CLAUDE_TOOLS,
                    messages=messages,
                )
                # The idiomatic loop: the response's content blocks go back in as-is.
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": search(**block.input),
                            }
                            for block in response.content
                            if block.type == "tool_use"
                        ],
                    }
                )


def claude_image() -> None:
    """One turn whose user message carries a screenshot."""
    responses = [_claude([{"type": "text", "text": "A login form."}], "end_turn")]
    client = wrap_anthropic(SimpleNamespace(messages=SimpleNamespace(create=_scripted(responses))))
    question = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
        },
        {"type": "text", "text": "What is in this screenshot?"},
    ]
    with capture.agent_session(
        suite="claude_image", redact=False, tools=CLAUDE_TOOLS, agent_input="screenshot"
    ):
        client.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            system="You are a support agent.",
            tools=CLAUDE_TOOLS,
            messages=[{"role": "user", "content": question}],
        )


def _completion(message: dict[str, Any], finish: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl_fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5",
            "choices": [
                {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


def _function_call(call_id: str, q: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "search", "arguments": json.dumps({"q": q})},
    }


def openai_agent(suite: str, *, exclude_none: bool) -> None:
    """Two turns; each assistant reply goes back in via message.model_dump()."""
    responses = [
        _completion({"content": None, "tool_calls": [_function_call("call_1", "evalshift")]}, "tool_calls"),
        _completion({"content": "EvalShift is a regression tool."}, "stop"),
        _completion({"content": None, "tool_calls": [_function_call("call_2", "pricing")]}, "tool_calls"),
        _completion({"content": "Pricing starts at $0."}, "stop"),
    ]
    client = wrap_openai(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_scripted(responses))))
    )
    messages: list[Any] = [{"role": "system", "content": "You are a support agent."}]
    for turn, question in enumerate(QUESTIONS):
        with capture.agent_session(
            suite=suite,
            redact=False,
            tools=OPENAI_TOOLS,
            conversation_id=f"conv-{suite}",
            turn_index=turn,
            agent_input=question,
        ):
            messages.append({"role": "user", "content": question})
            while True:
                reply = (
                    client.chat.completions.create(model="gpt-5", tools=OPENAI_TOOLS, messages=messages)
                    .choices[0]
                    .message
                )
                messages.append(reply.model_dump(exclude_none=exclude_none))
                if not reply.tool_calls:
                    break
                for call in reply.tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": search(**json.loads(call.function.arguments)),
                        }
                    )


def main() -> None:
    written = Path(os.environ["EVALSHIFT_DIR"])
    claude_agent()
    claude_image()
    openai_agent("openai_dump", exclude_none=False)
    openai_agent("openai_lean", exclude_none=True)
    for scenario in SCENARIOS:
        target = HERE / scenario
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(written / "captures" / scenario, target / "captures" / scenario)
        shutil.copytree(written / "toolsets", target / "toolsets")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixtures**

Run (from the worktree root):

```bash
EVALSHIFT_CAPTURE=1 EVALSHIFT_DIR="$(mktemp -d)" \
  uv run --project ../evalshift-sdk python tests/unit/fixtures/captures/native_shapes/generate.py
```

Expected: four `wrote …/native_shapes/<scenario>` lines. If `anthropic` or `openai` is missing from the SDK environment, add `--with anthropic --with openai` after `--project ../evalshift-sdk`.

Then inspect one, e.g. `python3 -m json.tool tests/unit/fixtures/captures/native_shapes/claude_agent/captures/claude_agent/*.json | head -80`: the turn-1 capture's first `model_call.input` must hold assistant `content` block lists with `tool_use` / `thinking`, and a user `tool_result` block — the raw shape. If it does not, stop: the SDK's recording changed and the spec's premise needs re-checking.

- [ ] **Step 3: Write the end-to-end tests**

Create `tests/unit/test_cli_capture_native_shapes.py`:

```python
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
    assert [(m.role, m.content) for m in first.history] == [
        ("system", "You are a support agent.")
    ]

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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_cli_capture_native_shapes.py -q -p no:cacheprovider --no-cov`
Expected: all pass (Task 2 already made them pass).

Then prove they guard the bug — run them against the pre-fix `promote.py` and restore it (do not use `git stash`: the stash stack is shared with other worktrees):

```bash
cp src/evalshift_cli/captures/promote.py "$TMPDIR/promote.fixed.py"
git show HEAD~1:src/evalshift_cli/captures/promote.py > src/evalshift_cli/captures/promote.py
uv run pytest tests/unit/test_cli_capture_native_shapes.py -q -p no:cacheprovider --no-cov  # expect the Claude + OpenAI tests to FAIL
cp "$TMPDIR/promote.fixed.py" src/evalshift_cli/captures/promote.py
git diff --exit-code src/evalshift_cli/captures/promote.py  # restored: no diff
```

(`HEAD~1` is Task 1's commit here, whose `promote.py` is the original. If `$TMPDIR` is unset, use any scratch directory.)

- [ ] **Step 5: Commit**

```bash
git add tests/unit/fixtures/captures/native_shapes tests/unit/test_cli_capture_native_shapes.py
uv run --with pre-commit git commit -m "test(captures): sync real wrap_anthropic and wrap_openai captures end to end

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

If `end-of-file-fixer` rewrites a fixture (adds a trailing newline), `git add` the fixtures again and re-run the same commit command.

---

### Task 4: Detect and rebuild stringified cases (pure)

**Files:**
- Create: `src/evalshift_cli/captures/repair.py`
- Test: `tests/unit/test_captures_repair.py`

**Interfaces:**
- Consumes: `PromotedCase` (`captures/models.py`), `SuiteExample` / `ChatMessage` (`suite/models.py`).
- Produces:
  - `stringified_content(example: SuiteExample) -> str | None` — `"history message N"` or `"inputs['<key>']"`.
  - `is_repair_candidate(example: SuiteExample) -> bool`
  - `RepairCandidate(case: PromotedCase, stringified: str | None)` (frozen dataclass)
  - `RepairSelection(candidates: tuple[RepairCandidate, ...] = (), refetch_ids: frozenset[str] = frozenset())` with `cases_by_capture() -> dict[str, list[PromotedCase]]`
  - `select_repairs(cases: Iterable[PromotedCase]) -> RepairSelection`
  - `stringified_warnings(selection: RepairSelection) -> list[str]`
  - `repaired_case(case: PromotedCase, rebuilt: SuiteExample, *, promoted_at: str) -> PromotedCase | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_captures_repair.py`:

```python
"""Finding promoted cases an earlier sync built from stringified blocks, and rebuilding them."""

from __future__ import annotations

import json
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
    ex = _example(history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": REPR_BLOCKS}])
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


def test_repaired_case_is_none_when_nothing_changed() -> None:
    stored = _case("c", example=_example(id_="c", inputs={}))
    assert repaired_case(stored, _example(id_="other", inputs={}), promoted_at="t") is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_captures_repair.py -q -p no:cacheprovider --no-cov`
Expected: `ModuleNotFoundError: No module named 'evalshift_cli.captures.repair'`.

- [ ] **Step 3: Write the module**

Create `src/evalshift_cli/captures/repair.py`:

```python
"""Find promoted cases an earlier ``capture sync`` built from stringified message blocks.

Before :mod:`~evalshift_cli.captures.native_messages` existed, promotion passed Claude's
block-list ``content`` through ``str()``, so a promoted history held Python-repr strings such
as ``"[{'id': 'toolu_1', ..., 'type': 'tool_use'}]"``, and OpenAI / Responses / LangChain
captures could promote with empty ``inputs``. This module finds those cases and rebuilds them;
``capture sync --repair`` drives it. Pure: no I/O.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.suite.models import SuiteExample

#: How Python's repr opens a list of dicts with string keys. JSON opens with ``[{"``,
#: so a JSON tool result is never mistaken for a stringified block list.
_REPR_OPENING = "[{'"


def stringified_content(example: SuiteExample) -> str | None:
    """Where ``example`` holds a stringified block list, or ``None``.

    Args:
        example: A promoted case's example.

    Returns:
        ``"inputs['<key>']"`` or ``"history message N"`` (1-based) for the first match.
    """
    for key, value in example.inputs.items():
        if isinstance(value, str) and _is_stringified_blocks(value):
            return f"inputs[{key!r}]"
    for position, message in enumerate(example.history or [], start=1):
        if _is_stringified_blocks(message.content):
            return f"history message {position}"
    return None


def is_repair_candidate(example: SuiteExample) -> bool:
    """Whether ``capture sync --repair`` should rebuild ``example`` from its capture.

    Stringified blocks, or empty ``inputs`` -- what the OpenAI ``exclude_none``, Responses and
    LangChain failures wrote. Empty inputs are not part of the *warning*: a genuinely
    input-less case would nag on every sync, and ``validate`` already fails a case that is
    missing a prompt's template variables.
    """
    return example.inputs == {} or stringified_content(example) is not None


def _is_stringified_blocks(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith(_REPR_OPENING):
        return False
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and isinstance(item.get("type"), str) for item in value)
    )


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    """A promoted case ``--repair`` rebuilds.

    Attributes:
        case: The case as stored.
        stringified: Where its stringified blocks are (see :func:`stringified_content`), or
            ``None`` for a candidate selected only for its empty ``inputs``.
    """

    case: PromotedCase
    stringified: str | None


@dataclass(frozen=True, slots=True)
class RepairSelection:
    """What ``capture sync`` found among the promoted cases.

    Attributes:
        candidates: Cases to rebuild, in the order they were read.
        refetch_ids: Capture ids ``--repair`` needs local: every candidate's, plus every
            promoted case sharing a ``(suite, conversation_id)`` with one -- a turn whose
            history is reconstructed from sibling turns needs them.
    """

    candidates: tuple[RepairCandidate, ...] = ()
    refetch_ids: frozenset[str] = frozenset()

    def cases_by_capture(self) -> dict[str, list[PromotedCase]]:
        """Candidate cases keyed by the capture they were promoted from."""
        by_capture: dict[str, list[PromotedCase]] = {}
        for candidate in self.candidates:
            by_capture.setdefault(candidate.case.from_capture, []).append(candidate.case)
        return by_capture


def select_repairs(cases: Iterable[PromotedCase]) -> RepairSelection:
    """Pick the repair candidates among ``cases`` and the captures rebuilding them needs.

    Args:
        cases: Every promoted case in scope (all suites, or the ``--suite`` one).

    Returns:
        The selection; empty when nothing needs repair.
    """
    all_cases = list(cases)
    candidates = tuple(
        RepairCandidate(case=case, stringified=stringified_content(case.example))
        for case in all_cases
        if is_repair_candidate(case.example)
    )
    conversations = {
        (c.case.suite, c.case.conversation_id)
        for c in candidates
        if c.case.conversation_id is not None
    }
    refetch = {c.case.from_capture for c in candidates}
    refetch.update(
        case.from_capture
        for case in all_cases
        if case.conversation_id is not None and (case.suite, case.conversation_id) in conversations
    )
    return RepairSelection(candidates=candidates, refetch_ids=frozenset(refetch))


def stringified_warnings(selection: RepairSelection) -> list[str]:
    """One warning per suite holding cases with the stringified-blocks signature."""
    by_suite: dict[str, list[str]] = {}
    for candidate in selection.candidates:
        if candidate.stringified is not None:
            by_suite.setdefault(candidate.case.suite, []).append(candidate.case.name)
    return [
        f"{suite}: {len(names)} promoted case(s) carry message blocks an earlier sync "
        f"stringified (e.g. {names[0]}); run evalshift capture sync --repair to rebuild them."
        for suite, names in sorted(by_suite.items())
    ]


def repaired_case(
    case: PromotedCase, rebuilt: SuiteExample, *, promoted_at: str
) -> PromotedCase | None:
    """``case`` with its ``inputs`` and ``history`` taken from ``rebuilt``, or ``None``.

    Everything else on the case -- its name and example id, tags, ``expected_*``, match
    strategies, generation config, any hand edits -- is kept: the stringification bug only
    ever corrupted those two fields.

    Args:
        case: The stored case.
        rebuilt: The example promotion builds from the case's capture today.
        promoted_at: Timestamp for the rewritten case.

    Returns:
        The rewritten case, or ``None`` when ``inputs`` and ``history`` already match.
    """
    current = case.example
    if rebuilt.inputs == current.inputs and rebuilt.history == current.history:
        return None
    payload = current.model_dump()
    payload["inputs"] = rebuilt.inputs
    payload["history"] = (
        None if rebuilt.history is None else [m.model_dump() for m in rebuilt.history]
    )
    example = SuiteExample.model_validate(payload)
    return case.model_copy(update={"example": example, "promoted_at": promoted_at})


__all__ = [
    "RepairCandidate",
    "RepairSelection",
    "is_repair_candidate",
    "repaired_case",
    "select_repairs",
    "stringified_content",
    "stringified_warnings",
]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_captures_repair.py -q -p no:cacheprovider --no-cov`
Expected: all pass. If `SuiteExample.model_validate(current.model_dump())` rejects its own dump (a field serializer/alias mismatch), fix `repaired_case` with `current.model_dump(mode="json")` / `by_alias=True` as the error indicates — the round-trip test is the guard.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format . && uv run mypy --strict src/evalshift_cli`

```bash
git add src/evalshift_cli/captures/repair.py tests/unit/test_captures_repair.py
uv run --with pre-commit git commit -m "feat(captures): detect cases an earlier sync built from stringified blocks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `fetch_captures(force_ids=…)`

**Files:**
- Modify: `src/evalshift_cli/captures/remote.py` (`fetch_captures`, ~line 245-328)
- Test: `tests/unit/test_captures_remote.py`

**Interfaces:**
- Produces: `fetch_captures(store, *, base=None, suite=None, since=None, skip_ids=frozenset(), force_ids: Collection[str] = frozenset(), workers=8) -> FetchSummary`. Ids in `force_ids` bypass the promoted skip and `since`; already-local, size-cap and malformed-key rules still apply.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_captures_remote.py`:

```python
def test_force_ids_bypass_the_promoted_skip_and_since(tmp_path: Path) -> None:
    now = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    store = populated_store("cap_forced", "cap_old", "cap_promoted")
    store.modified["captures/alpha/cap_forced.json"] = now - timedelta(days=30)
    store.modified["captures/alpha/cap_old.json"] = now - timedelta(days=30)
    summary = fetch_captures(
        store,
        base=tmp_path,
        skip_ids={"cap_forced", "cap_promoted"},
        force_ids={"cap_forced"},
        since=now - timedelta(days=1),
    )
    assert _local_captures(tmp_path) == {"alpha/cap_forced.json"}
    assert summary.skipped_promoted == 1  # cap_promoted
    assert summary.skipped_old == 1  # cap_old


def test_force_ids_still_skip_captures_already_local(tmp_path: Path) -> None:
    store = populated_store("cap_1")
    fetch_captures(store, base=tmp_path)
    store.gets.clear()
    summary = fetch_captures(store, base=tmp_path, force_ids={"cap_1"})
    assert summary.skipped_existing == 1
    assert store.gets == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_captures_remote.py -q -p no:cacheprovider --no-cov -k force_ids`
Expected: `TypeError: fetch_captures() got an unexpected keyword argument 'force_ids'`.

- [ ] **Step 3: Implement**

In `fetch_captures`, add the parameter after `skip_ids`:

```python
    skip_ids: Collection[str] = frozenset(),
    force_ids: Collection[str] = frozenset(),
```

extend the docstring after the sentence about `skip_ids`:

```
    Ids in ``force_ids`` are fetched even when promoted or older than ``since`` -- the
    captures ``capture sync --repair`` needs to rebuild promoted cases. They still respect the
    already-local, size-cap and malformed-key rules.
```

add `force = set(force_ids)` next to `skip = set(skip_ids)`, and replace

```python
        if name[: -len(".json")] in skip:
            summary.skipped_promoted += 1
            continue
        if since is not None and info.last_modified is not None and info.last_modified < since:
            summary.skipped_old += 1
            continue
```

with

```python
        capture_id = name[: -len(".json")]
        forced = capture_id in force
        if not forced and capture_id in skip:
            summary.skipped_promoted += 1
            continue
        if (
            not forced
            and since is not None
            and info.last_modified is not None
            and info.last_modified < since
        ):
            summary.skipped_old += 1
            continue
```

- [ ] **Step 4: Run the remote tests**

Run: `uv run pytest tests/unit/test_captures_remote.py tests/unit/test_cli_capture_remote.py -q -p no:cacheprovider --no-cov`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/captures/remote.py tests/unit/test_captures_remote.py
uv run --with pre-commit git commit -m "feat(captures): let fetch_captures force named captures past the promoted and --since skips

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `capture sync --repair` and the warning

**Files:**
- Modify: `src/evalshift_cli/cli/commands/capture.py` (imports; `_fetch_into_base`; `_maybe_fetch`; new `_repair_selection`, `_warn_unrepairable`; `capture_sync`)
- Modify: `tests/unit/fake_remote_store.py` (`capture_payload` gains `conversation_id` / `turn_index`)
- Test: `tests/unit/test_cli_capture_repair.py`

**Interfaces:**
- Consumes: `select_repairs`, `RepairSelection`, `stringified_warnings`, `repaired_case` (Task 4); `fetch_captures(force_ids=…)` (Task 5); `write_promoted_case`, `iter_promoted_cases`, `promoted_suite_dir` (`captures/promote.py`); `suites_root` (`captures/reader.py`).
- Produces: `capture sync --repair` (bool flag, default off).

- [ ] **Step 1: Let the fake store's captures carry conversation identity**

In `tests/unit/fake_remote_store.py`, change `capture_payload`'s signature and envelope:

```python
def capture_payload(
    capture_id: str,
    suite: str = "alpha",
    *,
    model_input: Any = "where is order 12345?",
    created_at: str = "2026-06-16T12:00:00+00:00",
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> bytes:
```

and add to the `envelope` dict, after `"created_at": created_at,`:

```python
        "conversation_id": conversation_id,
        "turn_index": turn_index,
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_cli_capture_repair.py`:

```python
"""``capture sync --repair`` and the warning that points at it.

The broken cases here are what CLI 1.3.0 promoted from a Claude follow-up turn: every block
list went through ``str()``. The store fixture plays S3; ``--base`` plays a fresh clone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.remote import RemoteStore
from evalshift_cli.captures.store_uri import StoreURI
from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import render_minimal_config
from evalshift_cli.cli.main import app
from evalshift_cli.suite.models import SuiteExample
from tests.unit.fake_remote_store import (
    TOOLSET_HEX,
    TOOLSET_KEY,
    TOOLSET_PAYLOAD,
    TOOLSET_REF,
    FakeRemoteStore,
    capture_payload,
)

runner = CliRunner()

CLAUDE_INPUT: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "evalshift"}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "results"}],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "A regression tool."}]},
    {"role": "user", "content": "And the price?"},
]
BROKEN_HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "What is evalshift?"},
    {"role": "assistant", "content": str(CLAUDE_INPUT[2]["content"])},
    {"role": "user", "content": str(CLAUDE_INPUT[3]["content"])},
    {"role": "assistant", "content": str(CLAUDE_INPUT[4]["content"])},
]


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeRemoteStore:
    fake = FakeRemoteStore({TOOLSET_KEY: TOOLSET_PAYLOAD})

    def _open(parsed: StoreURI) -> RemoteStore:
        return fake

    monkeypatch.setattr(capture_module, "open_store", _open)
    return fake


def _config(base: Path) -> Path:
    path = base / "evalshift.yaml"
    path.write_text(
        render_minimal_config(profile="model-upgrade") + "\ncaptures:\n  store: s3://b/p\n",
        encoding="utf-8",
    )
    return path


def _sync(base: Path, *flags: str) -> Any:
    return runner.invoke(
        app,
        ["capture", "sync", *flags, "--config", str(_config(base)), "--base", str(base)],
    )


def _out(result: Any) -> str:
    return " ".join(result.output.split())


def _put(
    store: FakeRemoteStore,
    capture_id: str,
    *,
    model_input: Any = None,
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> None:
    store.objects[f"captures/alpha/{capture_id}.json"] = capture_payload(
        capture_id,
        "alpha",
        model_input=CLAUDE_INPUT if model_input is None else model_input,
        conversation_id=conversation_id,
        turn_index=turn_index,
    )


def _write_local_capture(base: Path, capture_id: str) -> None:
    captures = base / "captures" / "alpha"
    captures.mkdir(parents=True, exist_ok=True)
    (captures / f"{capture_id}.json").write_bytes(
        capture_payload(capture_id, "alpha", model_input=CLAUDE_INPUT)
    )
    toolsets = base / "toolsets"
    toolsets.mkdir(parents=True, exist_ok=True)
    (toolsets / f"{TOOLSET_HEX}.json").write_bytes(TOOLSET_PAYLOAD)


def _write_case(
    base: Path,
    capture_id: str,
    *,
    name: str | None = None,
    suite: str = "alpha",
    inputs: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    tags: list[str] | None = None,
    expected_tools: list[dict[str, Any]] | None = None,
    conversation_id: str | None = None,
) -> Path:
    case_name = name or capture_id
    example = SuiteExample.model_validate(
        {
            "id": case_name,
            "inputs": {"input": "And the price?"} if inputs is None else inputs,
            "tags": tags or ["captured"],
            "history": BROKEN_HISTORY if history is None else history,
            "expected_tools": expected_tools,
            "toolset_ref": TOOLSET_REF,
            "conversation_id": conversation_id,
        }
    )
    case = PromotedCase(
        name=case_name,
        suite=suite,
        from_capture=capture_id,
        example=example,
        conversation_id=conversation_id,
    )
    path = base / "suites" / suite / f"{case_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(case.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _stored(path: Path) -> PromotedCase:
    return PromotedCase.model_validate_json(path.read_text(encoding="utf-8"))


# --- without --repair ------------------------------------------------------------------------


def test_a_fresh_clone_sync_warns_without_fetching_the_broken_cases(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _put(store, "cap_1")
    result = _sync(tmp_path)
    assert result.exit_code == 0, result.output
    assert (
        "alpha: 1 promoted case(s) carry message blocks an earlier sync stringified "
        "(e.g. cap_1); run evalshift capture sync --repair to rebuild them."
    ) in _out(result)
    assert "captures/alpha/cap_1.json" not in store.gets
    assert case.read_bytes() == before


@pytest.mark.parametrize("flags", [(), ("--force",)])
def test_sync_leaves_candidates_for_repair_even_with_force(
    tmp_path: Path, store: FakeRemoteStore, flags: tuple[str, ...]
) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _write_local_capture(tmp_path, "cap_1")
    result = _sync(tmp_path, "--offline", *flags)
    assert result.exit_code == 0, result.output
    assert "left 1 case(s) for --repair" in _out(result)
    assert case.read_bytes() == before


# --- --repair ---------------------------------------------------------------------------------


def test_repair_from_a_fresh_clone_refetches_and_rebuilds_only_the_broken_case(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    broken = _write_case(
        tmp_path,
        "cap_1",
        tags=["captured", "billing"],
        expected_tools=[{"tool_name": "search", "arguments": {"q": "evalshift"}}],
    )
    healthy = _write_case(tmp_path, "cap_2", inputs={"input": "hand edited"}, history=[])
    healthy_before = healthy.read_bytes()
    _put(store, "cap_1")
    _put(store, "cap_2", model_input="where is order cap_2?")

    result = _sync(tmp_path, "--repair")

    assert result.exit_code == 0, result.output
    assert "repaired 1 case(s)" in _out(result)
    assert "captures/alpha/cap_1.json" in store.gets
    assert "captures/alpha/cap_2.json" not in store.gets
    fixed = _stored(broken).example
    assert fixed.history is not None
    assert [m.role for m in fixed.history] == ["system", "user", "assistant", "tool", "assistant"]
    assert fixed.history[2].tool_calls is not None
    assert fixed.history[2].tool_calls[0].id == "toolu_1"
    assert fixed.history[3].tool_call_id == "toolu_1"
    assert fixed.tags == ["captured", "billing"]
    assert fixed.expected_tools is not None
    assert fixed.expected_tools[0].tool_name == "search"
    assert healthy.read_bytes() == healthy_before
    golden = (tmp_path / "suites" / "alpha" / "golden.jsonl").read_text(encoding="utf-8")
    assert "'type'" not in golden


def test_repair_keeps_a_custom_name_and_writes_no_second_case(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    _write_case(tmp_path, "cap_1", name="billing_q")
    _put(store, "cap_1")
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    suite_dir = tmp_path / "suites" / "alpha"
    assert sorted(p.name for p in suite_dir.iterdir()) == ["billing_q.json", "golden.jsonl"]
    fixed = _stored(suite_dir / "billing_q.json")
    assert fixed.example.id == "billing_q"
    assert fixed.example.history is not None
    assert fixed.example.history[3].role == "tool"


def test_repair_fetches_conversation_siblings_without_rewriting_them(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    sibling = _write_case(
        tmp_path,
        "cap_t0",
        inputs={"input": "What is evalshift?"},
        history=[],
        conversation_id="conv",
    )
    sibling_before = sibling.read_bytes()
    _write_case(tmp_path, "cap_t1", conversation_id="conv")
    _put(store, "cap_t0", model_input="What is evalshift?", conversation_id="conv", turn_index=0)
    _put(store, "cap_t1", conversation_id="conv", turn_index=1)

    result = _sync(tmp_path, "--repair")

    assert result.exit_code == 0, result.output
    assert {"captures/alpha/cap_t0.json", "captures/alpha/cap_t1.json"} <= set(store.gets)
    assert sibling.read_bytes() == sibling_before
    assert "repaired 1 case(s)" in _out(result)


def test_repair_names_cases_whose_capture_is_gone(tmp_path: Path, store: FakeRemoteStore) -> None:
    case = _write_case(tmp_path, "cap_gone")
    before = case.read_bytes()
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    assert (
        "can't repair 1 case(s) — their capture is no longer local or in the store "
        "(cap_gone): delete the case file or re-capture."
    ) in _out(result)
    assert case.read_bytes() == before


def test_repair_is_not_hidden_by_since(tmp_path: Path, store: FakeRemoteStore) -> None:
    _write_case(tmp_path, "cap_1")
    _put(store, "cap_1")
    store.modified["captures/alpha/cap_1.json"] = datetime(2020, 1, 1, tzinfo=UTC)
    result = _sync(tmp_path, "--repair", "--since", "1h")
    assert result.exit_code == 0, result.output
    assert "repaired 1 case(s)" in _out(result)


def test_repair_offline_uses_local_captures_only(tmp_path: Path, store: FakeRemoteStore) -> None:
    _write_case(tmp_path, "cap_1")
    _write_local_capture(tmp_path, "cap_1")
    result = _sync(tmp_path, "--repair", "--offline")
    assert result.exit_code == 0, result.output
    assert store.gets == []
    assert "repaired 1 case(s)" in _out(result)


def test_repair_twice_is_a_no_op(tmp_path: Path, store: FakeRemoteStore) -> None:
    case = _write_case(tmp_path, "cap_1")
    _put(store, "cap_1")
    assert _sync(tmp_path, "--repair").exit_code == 0
    after_first = case.read_bytes()
    second = _sync(tmp_path, "--repair")
    assert second.exit_code == 0, second.output
    assert "no promoted cases need repair." in _out(second)
    assert case.read_bytes() == after_first


def test_repair_leaves_a_case_whose_rebuild_is_blocked(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    case = _write_case(tmp_path, "cap_1")
    before = case.read_bytes()
    _put(store, "cap_1")
    del store.objects[TOOLSET_KEY]  # no sidecar anywhere: the rebuild is refused (no_toolset)
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    assert "can't repair cap_1:" in _out(result)
    assert case.read_bytes() == before


def test_repair_survives_a_corrupt_case_file_elsewhere(
    tmp_path: Path, store: FakeRemoteStore
) -> None:
    broken = _write_case(tmp_path, "cap_1")
    bad = tmp_path / "suites" / "beta" / "bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    _put(store, "cap_1")
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    out = _out(result)
    assert "to check for repairs: ValidationError" in out
    assert "repaired 1 case(s)" in out
    history = _stored(broken).example.history
    assert history is not None
    assert history[3].role == "tool"


def test_repair_reports_when_there_is_nothing_to_do(tmp_path: Path, store: FakeRemoteStore) -> None:
    result = _sync(tmp_path, "--repair")
    assert result.exit_code == 0, result.output
    assert "no promoted cases need repair." in _out(result)
```

(`test_repair_survives_a_corrupt_case_file_elsewhere`: the selection warns about `beta/bad.json` and moves on; the fetch step's promoted-id read also warns and treats nothing as promoted, which only widens the fetch; `alpha` is still repaired and only `alpha`'s `golden.jsonl` is rebuilt.)

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_cli_capture_repair.py -q -p no:cacheprovider --no-cov`
Expected: FAIL — `No such option: --repair`, and the warning test finds no warning.

- [ ] **Step 4: Plumb `force_ids` through the fetch helpers**

In `capture.py`, imports: add `suites_root` to the `evalshift_cli.captures.reader` import list, and add

```python
from evalshift_cli.captures.repair import (
    RepairSelection,
    repaired_case,
    select_repairs,
    stringified_warnings,
)
```

Replace `_fetch_into_base`'s signature and fetch call:

```python
def _fetch_into_base(
    store: RemoteStore,
    *,
    base: Path | None,
    suite: str | None,
    since: datetime | None,
    console: Console,
    force_ids: frozenset[str] = frozenset(),
) -> FetchSummary:
    """Mirror ``store`` into ``base`` and print the one-line summary; exit 1 on failure.

    ``force_ids`` are fetched even when promoted or older than ``since`` -- the captures
    ``capture sync --repair`` needs to rebuild promoted cases.
    """
    skip_ids = _promoted_capture_ids_or_warn(base=base, console=console)
    try:
        summary = fetch_captures(
            store, base=base, suite=suite, since=since, skip_ids=skip_ids, force_ids=force_ids
        )
```

(the rest of the body is unchanged). In `_maybe_fetch`, add the parameter `force_ids: frozenset[str] = frozenset(),` after `console: Console,` and change its last line to

```python
    _fetch_into_base(
        store, base=base, suite=suite, since=since_dt, console=console, force_ids=force_ids
    )
```

- [ ] **Step 5: Add the selection and unrepairable helpers**

After `_promoted_content_owners`, add:

```python
def _repair_selection(
    *, base: Path | None, suite: str | None, console: Console
) -> RepairSelection:
    """The promoted cases ``capture sync`` treats as repair candidates (``captures/repair.py``).

    Never raises: a suite holding an unreadable case file is warned about and skipped, the
    same degradation :func:`_promoted_content_owners` applies.
    """
    root = suites_root(base)
    if suite is not None:
        suite_dirs = [promoted_suite_dir(suite, base=base)]
    elif root.is_dir():
        suite_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        suite_dirs = []
    cases: list[PromotedCase] = []
    for suite_dir in suite_dirs:
        if not suite_dir.is_dir():
            continue
        try:
            cases.extend(iter_promoted_cases(suite_dir))
        except ValueError as exc:  # pydantic ValidationError subclasses ValueError
            console.print(
                f"[yellow]⚠[/yellow] could not read promoted cases in "
                f"[cyan]{escape(str(suite_dir))}[/cyan] to check for repairs: "
                f"{type(exc).__name__}",
            )
    return select_repairs(cases)


def _warn_unrepairable(
    selection: RepairSelection, records: list[CaptureRecord], *, console: Console
) -> None:
    """Name the repair candidates whose capture is neither local nor fetched from the store."""
    available = {r.envelope.capture_id for r in records}
    missing = [c.case.name for c in selection.candidates if c.case.from_capture not in available]
    if not missing:
        return
    shown = ", ".join(missing[:3]) + (", …" if len(missing) > 3 else "")
    console.print(
        f"[yellow]⚠[/yellow] can't repair {len(missing)} case(s) — their capture is no longer "
        f"local or in the store ({escape(shown)}): delete the case file or re-capture.",
    )
```

- [ ] **Step 6: Add the flag and wire it into `capture_sync`**

Add the option after `keep_duplicates` (before `since`):

```python
    repair: Annotated[
        bool,
        typer.Option(
            "--repair",
            help=(
                "Rebuild the inputs and history of promoted cases an earlier sync built from "
                "stringified message blocks (or left with empty inputs), re-fetching their "
                "captures from captures.store even though they are promoted. Every other "
                "field of those cases is kept; other promoted cases are untouched."
            ),
        ),
    ] = False,
```

Replace the body's opening (from `_check_rounds(rounds)` through the `if not records:` block) with:

```python
    _check_rounds(rounds)
    selection = _repair_selection(base=base, suite=suite, console=console)
    _maybe_fetch(
        config_path=config_path,
        offline=offline,
        since=since,
        suite=suite,
        base=base,
        console=console,
        force_ids=selection.refetch_ids if repair else frozenset(),
    )
    if not repair:
        for line in stringified_warnings(selection):
            console.print(f"[yellow]⚠[/yellow] {escape(line)}")
    elif not selection.candidates:
        console.print("[dim]no promoted cases need repair.[/dim]")
    records = iter_captures(suite=suite, base=base, on_error=_warn_unreadable(console))
    if repair:
        _warn_unrepairable(selection, records, console=console)
    if not records:
        where = f" for suite {suite!r}" if suite else ""
        console.print(f"[dim]no captures found{where} under .evalshift/captures/.[/dim]")
        return
```

Next to the other counters (`skipped_unreplayable = 0` …) add:

```python
    repaired = 0
    left_for_repair = 0
    backed_by = selection.cases_by_capture()
```

In the inner loop, right after `envelope = record.envelope` and **before** `if built.blocked is not None:`, insert:

```python
            backed = backed_by.get(envelope.capture_id)
            if backed is not None:
                # A capture behind a repair candidate is only ever written by --repair, and
                # then only the case's inputs and history -- never as a new case (a custom
                # `--as` name would otherwise get a second copy next to the broken one).
                if not repair:
                    left_for_repair += len(backed)
                    continue
                for warning in built.warnings:
                    console.print(f"[yellow]⚠[/yellow] {envelope.capture_id}: {warning}")
                if built.blocked is not None:
                    for case in backed:
                        console.print(
                            f"[yellow]⚠[/yellow] can't repair [cyan]{escape(case.name)}[/cyan]: "
                            f"{escape(built.blocked)}",
                        )
                    continue
                for case in backed:
                    fixed = repaired_case(case, built.example, promoted_at=_now_iso())
                    if fixed is not None:
                        write_promoted_case(fixed, base=base, force=True)
                        repaired += 1
                continue
```

After the `skipped_unreplayable` summary clause add:

```python
    if repair:
        summary += f", repaired {repaired} case(s)"
    if left_for_repair:
        summary += f", left {left_for_repair} case(s) for --repair"
```

- [ ] **Step 7: Run the repair tests and the neighbouring suites**

Run: `uv run pytest tests/unit/test_cli_capture_repair.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_remote.py tests/unit/test_captures_remote.py tests/unit/test_cli_capture_native_shapes.py -q -p no:cacheprovider --no-cov`
Expected: all pass. `test_sync_does_not_refetch_promoted_then_cleaned_captures` (existing) must still pass: healthy promoted captures are still never re-downloaded.

- [ ] **Step 8: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format . && uv run mypy --strict src/evalshift_cli`

```bash
git add src/evalshift_cli/cli/commands/capture.py tests/unit/fake_remote_store.py tests/unit/test_cli_capture_repair.py
uv run --with pre-commit git commit -m "feat(capture): sync --repair rebuilds cases an earlier sync stringified

capture sync now warns, per suite, when promoted cases carry stringified
message blocks, and leaves them alone (--force included). --repair re-fetches
exactly their captures (and conversation siblings) past the promoted-id and
--since skips, and replaces only each case's inputs and history.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `validate` advisory

**Files:**
- Modify: `src/evalshift_cli/cli/commands/validate.py`
- Test: `tests/integration/test_validate_command.py`

**Interfaces:**
- Consumes: `stringified_content` (Task 4).

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_validate_command.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/integration/test_validate_command.py -q -p no:cacheprovider --no-cov -k Stringified`
Expected: the warning test FAILS (no advisory); the silent test passes.

- [ ] **Step 3: Implement**

In `validate.py`, add `from evalshift_cli.captures.repair import stringified_content` to the imports, and right after the CI-pin advisory block (`if finding is not None: console.print(...)`) add:

```python
    # Advisory: cases an earlier `capture sync` built from stringified message blocks.
    flagged = [ex.id for ex in loaded_suite.examples if stringified_content(ex) is not None]
    if flagged:
        console.print(
            f"[yellow]⚠[/yellow] {len(flagged)} example(s) carry message blocks an earlier "
            f"capture sync stringified (e.g. {escape(flagged[0])}); run evalshift capture "
            "sync --repair.",
        )
```

- [ ] **Step 4: Run the validate tests**

Run: `uv run pytest tests/integration/test_validate_command.py -q -p no:cacheprovider --no-cov`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/cli/commands/validate.py tests/integration/test_validate_command.py
uv run --with pre-commit git commit -m "feat(validate): warn when a suite holds history an earlier sync stringified

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docs, changelog, full gate

**Files:**
- Modify: `docs/conversations.md`, `DOCS.md`, `llms-full.txt`, `docs/sdk.md`, `docs/configuration.md`, `CHANGELOG.md`
- Test: `tests/unit/test_docs_currency.py` (must stay green; no edits expected)

- [ ] **Step 1: `docs/conversations.md`**

1. Replace the bullet

```
* If a turn's own capture recorded a full messages list (the convention
  above), its `history` is recovered **verbatim** from that list: everything
  before the last `user` message becomes `history`, and the last `user`
  message becomes the current turn's `inputs`.
```

with

```
* If a turn's own capture recorded a full messages list (the convention
  above), its `history` is recovered from that list: everything before the
  last `user` message becomes `history`, and the last `user` message becomes
  the current turn's `inputs`. The list is first translated from the
  provider's own message shape into the replay shape (see
  [Provider message shapes](#provider-message-shapes)), so its text and tool
  calls are kept exactly, in the form `run` replays.
```

2. In the paragraph after it, replace `but they are not equivalent: verbatim recovery\nreproduces exactly what the model saw, while` with `but they are not equivalent: recovery from the turn's own messages list\nreproduces the text and tool calls the model saw, while` (rewrap the paragraph to ~78 columns).

3. Insert this section immediately before `## Teacher-forced replay semantics`:

````markdown
## Provider message shapes

The SDK's client wrappers record each model call's messages the way that
provider takes them, and `capture sync` / `capture promote` translate them into
the replay shape before splitting off `history`:

| Recorded by | What promotion reads |
| --- | --- |
| `wrap_anthropic` (Claude Messages) | `text` blocks become the message text; `tool_use` blocks become the assistant's `tool_calls`; each `tool_result` block becomes a `tool` message keyed by its `tool_use_id`; a `system` prompt sent as blocks (prompt caching) becomes its text. |
| `wrap_openai`, Chat Completions | `tool_calls[].function` becomes `{id, name, arguments}` with the JSON-string arguments parsed; `content: null` or no `content` key is an empty string; `developer` is read as `system`; a `refusal` becomes the assistant's text. |
| `wrap_openai`, Responses | `message` items become messages; `function_call` items become the assistant's `tool_calls`; `function_call_output` items become `tool` messages keyed by `call_id`. |
| `wrap_genai` (Gemini) | Already the replay shape. |
| LangChain callback | `human` / `ai` are read as `user` / `assistant`; block-list content (Claude through LangChain) is read as above. |

Thinking (`thinking`, `redacted_thinking`, Responses `reasoning`) is dropped:
the Claude API ignores earlier turns' thinking itself, and a signature means
nothing to a different candidate model.

A turn whose recorded messages carry anything a text replay can't carry — an
image, a document, audio, a server-tool result such as `web_search_tool_result`,
or a block type promotion doesn't know — is **not promoted**. `capture sync`
skips it, naming the block, and counts it as `non-text content a text replay
can't carry`; `capture promote` refuses it. `--allow-errored` does not apply: a
replay without the image is not ground truth.

### Repairing suites promoted before this

Earlier CLI releases passed Claude's blocks through `str()`, so a promoted
follow-up turn's history held Python-repr strings such as
`"[{'id': 'toolu_1', … 'type': 'tool_use'}]"`, and OpenAI, Responses and
LangChain turns could come out with empty `inputs`. `capture sync` and
`validate` now warn when a suite still holds such cases:

```text
⚠ support: 4 promoted case(s) carry message blocks an earlier sync stringified (e.g. cap_…); run evalshift capture sync --repair to rebuild them.
```

`evalshift capture sync --repair` rebuilds them from their captures. It replaces
only each affected case's `inputs` and `history` — the case's name, tags,
`expected_*` and any other hand edits stay — and leaves every other promoted
case alone. With `captures.store` set it re-downloads exactly the captures it
needs, even though they are promoted and even past `--since`, plus the other
turns of the same conversation when a history has to be reconstructed. A case
whose capture is gone from both `.evalshift/captures/` and the store is named
(`can't repair N case(s)`): delete it or re-capture. Without `--repair`, sync —
`--force` included — leaves these cases untouched (`left N case(s) for
--repair`). Commit the regenerated `golden.jsonl` afterwards; CI reads the
committed suite.
````

4. In `## Limitations`, change `get verbatim/reconstructed history spanning` to `get recovered/reconstructed history spanning`, and append:

```
* **Responses conversations chained server-side lose their history.** An
  OpenAI Responses call that continues a conversation through
  `previous_response_id` or `conversation=` sends only the new input, so that
  is all the capture holds; the earlier turns can't be recovered.
* **LangChain over OpenAI loses tool rounds.** The SDK's LangChain callback
  records each message's role and content but not `tool_calls` or
  `tool_call_id`, and an OpenAI-backed `AIMessage` keeps its calls only
  there, so its tool results replay unpaired. LangChain over Claude keeps
  them: the `tool_use` blocks are in the content.
```

- [ ] **Step 2: `DOCS.md`**

1. The `Rules and limitations` bullet that starts `- Captures recorded with the SDK's messages-list convention recover \`history\` **verbatim**;` — replace its first clause so the bullet reads:

```
- Captures recorded with the SDK's messages-list convention recover `history` from that list, translated from the provider's own message shape (Claude blocks, OpenAI Chat and Responses, LangChain roles) into the replay shape with text and tool calls kept; thinking is dropped, and a turn carrying an image or other non-text content is skipped. Captures that only recorded a bare string get history **reconstructed** from sibling turn captures (an approximation: assistant replies come from final outputs, intermediate tool exchanges are absent). Prefer the messages-list convention in your instrumentation. Suites promoted before this translation existed are flagged by `capture sync` and `validate`; `capture sync --repair` rebuilds them — see [Provider message shapes](docs/conversations.md#provider-message-shapes).
```

2. Step 7 of the `capture sync` list (`7. Skips captures whose turn recorded an \`error\` event — …`): append at the end of that item: ` Also unconditionally, a capture whose recorded messages carry content a text replay can't carry — an image, a document, audio, a server-tool result, an unknown block type — is skipped and counted on its own (\`capture promote\` refuses it): a replay without it is not ground truth.`

3. The `**\`evalshift capture sync\`**` flag line: append ` · \`--repair\`` after `` `--offline` ``.

- [ ] **Step 3: `llms-full.txt`**

1. In the `evalshift capture sync` usage block, change `[--write/--print (default write)] [--keep-duplicates] [--since 30m|24h|7d|ISO] [--offline]` to `[--write/--print (default write)] [--keep-duplicates] [--since 30m|24h|7d|ISO] [--offline]\n    [--repair]`, and add after that command's description paragraph:

```
  --repair rebuilds the inputs + history (only) of promoted cases an earlier sync built from
  stringified message blocks (or left with empty inputs), re-fetching their captures from
  captures.store past the promoted-id and --since skips (plus same-conversation siblings).
  Without --repair, sync (--force included) leaves those cases alone and warns per suite;
  `validate` warns too. A candidate whose capture is gone: "can't repair N case(s)".
```

2. Replace

```
- capture sync recovers history VERBATIM when the capture recorded a full messages list, else
```

with

```
- capture sync recovers history from the capture's own messages list (translated from Claude
  blocks / OpenAI Chat / OpenAI Responses / LangChain roles into the replay shape: tool_use ->
  tool_calls, tool_result -> tool messages, thinking dropped) when one was recorded, else
```

3. After the `- Promotion ALSO blocks unconditionally (--allow-errored does not help) when the first …` bullet, add:

```
- Promotion ALSO blocks unconditionally when the recorded messages carry content a text
  replay can't carry (image, document, audio, server-tool result, unknown block or Responses
  item type): blocked_reason "unreplayable_content"; sync counts it separately; promote exits 1
  without suggesting --allow-errored.
```

- [ ] **Step 4: `docs/sdk.md` and `docs/configuration.md`**

`docs/sdk.md`: replace `` can: `capture sync` recovers conversation history verbatim from a messages list, `` with `` can: `capture sync` recovers conversation history, tool calls included, from a messages list in any wrapped provider's shape, ``; and after `` `--keep-duplicates` disables dedup. `` add `` `--repair` rebuilds cases an earlier sync built from stringified message blocks (see [Multi-turn conversations](conversations.md#repairing-suites-promoted-before-this)). ``

`docs/configuration.md`: in the `capture sync` paragraph, after `captures can't slip a duplicate past it.` insert `` Captures whose messages carry an image or other non-text content are skipped too (see [Provider message shapes](conversations.md#provider-message-shapes)), and `--repair` rebuilds cases an earlier sync built from stringified message blocks. ``

- [ ] **Step 5: `CHANGELOG.md`**

Under `## [Unreleased]` add:

```markdown
### Fixed

- `capture sync` and `capture promote` now read every wrapped provider's own
  message shape. Claude follow-up turns no longer get Python-repr strings
  (`"[{'id': 'toolu_1', … 'type': 'tool_use'}]"`) in their history: `tool_use`
  blocks become the assistant's `tool_calls`, `tool_result` blocks become `tool`
  messages, thinking is dropped, and a prompt-cached `system` prompt or a
  text-block user turn becomes plain text. OpenAI Chat histories with
  `content: null` no longer abort the whole sync with a traceback, and OpenAI
  Responses and LangChain captures recover their `inputs` instead of promoting
  them empty.
- A turn whose recorded messages carry an image, a document, audio, a
  server-tool result or an unknown block type is skipped (`capture promote`
  refuses it) instead of being promoted with that content stringified or lost;
  `--allow-errored` does not apply.

### Added

- `capture sync --repair` rebuilds the `inputs` and `history` of cases an
  earlier sync built from stringified message blocks, keeping every other
  field, and re-downloads exactly the captures it needs from `captures.store`
  even though they are promoted. `capture sync` and `validate` warn when a
  suite still holds such cases; without `--repair`, sync leaves them untouched.
```

- [ ] **Step 6: Full gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest`
Expected: all four green, coverage ≥ 90, `tests/unit/test_docs_currency.py` included. Then `uv run --with pre-commit pre-commit run --all-files` — all hooks pass.

- [ ] **Step 7: Live check against the original repro**

Re-run the investigation's repro scripts (session scratchpad `repro/agent.py`, `repro-openai/agent.py`) or, equivalently, the Task 3 generator into a scratch `EVALSHIFT_DIR`, then `uv run evalshift capture sync --offline --print --base <that dir>` with this branch's CLI, and read the second turn's history in `golden.jsonl`: real `tool_calls`, a `tool` message, no `'type'` strings. Paste the relevant lines into the PR description.

- [ ] **Step 8: Commit**

```bash
git add docs/conversations.md DOCS.md llms-full.txt docs/sdk.md docs/configuration.md CHANGELOG.md
uv run --with pre-commit git commit -m "docs: provider message shapes, unreplayable turns and capture sync --repair

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
