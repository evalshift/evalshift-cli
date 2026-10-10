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
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"q": "evalshift"},
                },
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
        _completion(
            {"content": None, "tool_calls": [_function_call("call_1", "evalshift")]}, "tool_calls"
        ),
        _completion({"content": "EvalShift is a regression tool."}, "stop"),
        _completion(
            {"content": None, "tool_calls": [_function_call("call_2", "pricing")]}, "tool_calls"
        ),
        _completion({"content": "Pricing starts at $0."}, "stop"),
    ]
    client = wrap_openai(
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_scripted(responses)))
        )
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
                    client.chat.completions.create(
                        model="gpt-5", tools=OPENAI_TOOLS, messages=messages
                    )
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
