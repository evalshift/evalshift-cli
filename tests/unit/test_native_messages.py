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
    {
        "type": "function_call",
        "call_id": "fc_1",
        "name": "weather",
        "arguments": '{"city": "Paris"}',
    },
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
