"""Tests for :mod:`evalshift_cli.models.client`.

We test the client by monkeypatching ``litellm.acompletion`` and
``litellm.completion_cost``. There are no live API calls in this suite —
that lives in ``scripts/smoke_live.py`` which the user runs by hand.

What we care about here:

* Successful calls produce a :class:`CompletionResult` with the right
  text, tokens, and cost.
* Provider exceptions are mapped into :class:`RateLimitError` /
  :class:`AuthError` / :class:`ModelError`.
* Transient errors are retried (and the retry sleeps respect the policy
  cap), permanent errors (auth) are not.
* The canonical model id is what gets dispatched even when an alias is
  passed in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import litellm
import pytest

from evalshift_cli.evaluators.tool_models import ToolSpec
from evalshift_cli.models import client as client_module
from evalshift_cli.models.client import (
    AuthError,
    CompletionResult,
    ModelClient,
    ModelError,
    RateLimitError,
    RetryPolicy,
    ToolCompletionResult,
    serialize_tools,
)
from evalshift_cli.models.registry import resolve_model

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(
        self,
        text: str,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        finish_reason: str | None = None,
    ) -> None:
        self.choices = [_FakeChoice(text, finish_reason)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


def _patch_acompletion(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[..., Any],
) -> dict[str, Any]:
    """Replace ``litellm.acompletion`` with ``handler`` and capture call kwargs."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        result = handler(**kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result

    monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        client_module.litellm,
        "completion_cost",
        lambda completion_response=None, **_: 0.00042,
    )
    # Make backoff sleeps a no-op so retry tests run instantly.
    monkeypatch.setattr(client_module.asyncio, "sleep", _noop_sleep)
    return captured


async def _noop_sleep(_seconds: float) -> None:
    return None


# ---------------------------------------------------------------------------
# Tool-aware fakes
# ---------------------------------------------------------------------------


def _patch_tools_acompletion(
    monkeypatch: pytest.MonkeyPatch,
    response_dict: dict[str, Any] | Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Replace ``litellm.acompletion`` so it returns a dict-shaped response.

    The tool-aware code path coerces SDK objects to dicts via
    ``response.model_dump()``; for the test we just hand back a dict
    directly. Returns the captured kwargs so tests can assert what was
    sent to LiteLLM.
    """
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        if callable(response_dict):
            return response_dict(**kwargs)
        return response_dict

    monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        client_module.litellm,
        "completion_cost",
        lambda completion_response=None, **_: 0.0,
    )
    monkeypatch.setattr(client_module.asyncio, "sleep", _noop_sleep)
    return captured


_OPENAI_SINGLE_RESPONSE: dict[str, Any] = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "search_db",
                            "arguments": '{"query": "ACME"}',
                        },
                    },
                ],
            },
        },
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_delay_zero_attempts_under_cap(self) -> None:
        policy = RetryPolicy(max_attempts=3, base_seconds=2.0, cap_seconds=10.0)
        # First attempt's upper bound should be `base * 2^0 = 2`.
        for _ in range(50):
            d = policy.delay(1)
            assert 0 <= d <= 2.0

    def test_delay_caps_at_cap_seconds(self) -> None:
        policy = RetryPolicy(max_attempts=10, base_seconds=2.0, cap_seconds=4.0)
        # By attempt 5 the exponential blows past cap=4; delay must respect it.
        for _ in range(50):
            d = policy.delay(5)
            assert 0 <= d <= 4.0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestCompleteHappyPath:
    async def test_returns_completion_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse("Hello Alex!", prompt_tokens=11, completion_tokens=6),
        )
        client = ModelClient()
        result = await client.complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
        )
        assert isinstance(result, CompletionResult)
        assert result.text == "Hello Alex!"
        assert result.input_tokens == 11
        assert result.output_tokens == 6
        assert result.cost_usd == pytest.approx(0.00042)
        assert result.latency_ms >= 0
        # Canonical id must reach LiteLLM.
        assert captured["kwargs"]["model"] == "gemini/gemini-2.5-flash"

    async def test_alias_resolved_to_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        client = ModelClient()
        result = await client.complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.model_id == "gemini/gemini-2.5-flash"
        assert captured["kwargs"]["model"] == "gemini/gemini-2.5-flash"

    async def test_default_temperature_and_max_tokens_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
        )
        assert captured["kwargs"]["temperature"] == 0.0
        assert captured["kwargs"]["max_tokens"] == 4096

    async def test_drop_params_enabled_for_reasoning_model_compat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Reasoning-tier models (e.g. gpt-5.6-*) reject temperature != 1.
        # ``drop_params`` tells LiteLLM to silently drop unsupported params
        # instead of erroring, so such models stay usable as judges.
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert captured["kwargs"]["drop_params"] is True

    async def test_finish_reason_stop_is_captured_not_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok", finish_reason="stop"))
        result = await ModelClient().complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.finish_reason == "stop"
        assert result.truncated is False

    async def test_finish_reason_length_marks_truncated_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch, lambda **_: _FakeResponse("partial", finish_reason="length")
        )
        with caplog.at_level("WARNING"):
            result = await ModelClient().complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.finish_reason == "length"
        assert result.truncated is True
        assert any("truncated" in r.message for r in caplog.records)

    async def test_finish_reason_absent_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A response object whose choice lacks finish_reason → None, not a crash.
        _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        result = await ModelClient().complete(model="gemini-2.5-flash", prompt="Hi")
        assert result.finish_reason is None
        assert result.truncated is False

    async def test_explicit_max_tokens_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
            max_tokens=256,
        )
        assert captured["kwargs"]["max_tokens"] == 256

    async def test_explicit_temperature_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
            temperature=0.7,
            max_tokens=64,
        )
        assert captured["kwargs"]["temperature"] == 0.7
        assert captured["kwargs"]["max_tokens"] == 64

    async def test_extra_kwargs_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(
            model="gemini/gemini-2.5-flash",
            prompt="Hi",
            extra={"response_format": {"type": "json_object"}},
        )
        assert captured["kwargs"]["response_format"] == {"type": "json_object"}

    async def test_unknown_cost_does_not_fail_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))

        def boom(**_kwargs: Any) -> float:
            raise RuntimeError("no pricing for this model")

        monkeypatch.setattr(client_module.litellm, "completion_cost", boom)
        result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class _ProviderRateLimitError(Exception):
    pass


class _ProviderAuthError(Exception):
    pass


# EvalShift's error mapper looks at the *runtime* class name, so we
# rename these stand-ins to match what real providers raise without
# importing each vendor SDK into the test suite.
_ProviderRateLimitError.__name__ = "RateLimitError"
_ProviderAuthError.__name__ = "AuthenticationError"


class TestErrorMapping:
    async def test_rate_limit_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise _ProviderRateLimitError("slow down")
            return _FakeResponse("ok-after-retry")

        _patch_acompletion(monkeypatch, handler)
        result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.text == "ok-after-retry"
        assert attempts["n"] == 2

    async def test_rate_limit_exhausts_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(**_kwargs: Any) -> Any:
            raise _ProviderRateLimitError("slow down")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(RateLimitError):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")

    async def test_auth_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            raise _ProviderAuthError("bad key")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=5))
        with pytest.raises(AuthError):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        # Auth errors are deterministic — burning retries on them wastes time.
        assert attempts["n"] == 1

    async def test_unknown_error_becomes_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(**_kwargs: Any) -> Any:
            raise ValueError("what is happening")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError, match="what is happening"):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")


# ---------------------------------------------------------------------------
# Response extraction edge cases
# ---------------------------------------------------------------------------


class TestResponseExtraction:
    async def test_response_with_dict_message_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Some providers (and the LiteLLM mock harness) return choices
        # whose `.message` is a dict rather than an object.
        class _DictMessageChoice:
            def __init__(self, content: str) -> None:
                self.message = {"content": content}

        class _DictMessageResponse:
            def __init__(self, content: str) -> None:
                self.choices = [_DictMessageChoice(content)]
                self.usage = _FakeUsage(1, 1)

        _patch_acompletion(monkeypatch, lambda **_: _DictMessageResponse("dict msg"))
        result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.text == "dict msg"

    async def test_malformed_response_raises_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Empty:
            def __init__(self) -> None:
                self.choices: list[Any] = []

        _patch_acompletion(monkeypatch, lambda **_: _Empty())
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError, match="could not extract text"):
            await client.complete(model="gemini/gemini-2.5-flash", prompt="Hi")


# ---------------------------------------------------------------------------
# complete_with_tools
# ---------------------------------------------------------------------------


_DEMO_TOOL = ToolSpec(
    name="search_db",
    description="Search the customer DB",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


class TestCompleteWithTools:
    async def test_returns_tool_completion_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert isinstance(result, ToolCompletionResult)
        assert result.model_id == "openai/gpt-4o"
        assert result.trace.call_count == 1
        assert result.trace.calls[0].tool_name == "search_db"
        assert result.trace.calls[0].arguments == {"query": "ACME"}
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        # Captured kwargs include the tool payload in OpenAI shape.
        kwargs = captured["kwargs"]
        assert "tools" in kwargs
        assert kwargs["tools"][0]["type"] == "function"
        assert kwargs["tools"][0]["function"]["name"] == "search_db"

    async def test_tool_finish_reason_length_marks_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = {
            **_OPENAI_SINGLE_RESPONSE,
            "choices": [dict(_OPENAI_SINGLE_RESPONSE["choices"][0])],
        }
        response["choices"][0]["finish_reason"] = "length"
        _patch_tools_acompletion(monkeypatch, response)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.finish_reason == "length"
        assert result.truncated is True

    async def test_real_litellm_dispatches_a_tool_call(self) -> None:
        # No monkeypatch: the request goes through the installed LiteLLM, which
        # returns `mock_response` without contacting the provider. LiteLLM
        # 1.106.x imports its MCP proxy module, which needs the `mcp` package
        # we don't install, on every call that carries tools, so this fails
        # there -- the reason pyproject caps litellm below 1.106.
        result = await ModelClient(retry_policy=RetryPolicy(max_attempts=1)).complete_with_tools(
            model="claude-sonnet-5",
            prompt="hi",
            tools=[_DEMO_TOOL],
            extra={"mock_response": "no tool needed"},
        )
        assert result.model_id == "anthropic/claude-sonnet-5"
        assert result.trace.call_count == 0

    async def test_tool_max_tokens_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
            max_tokens=321,
        )
        assert captured["kwargs"]["max_tokens"] == 321

    async def test_anthropic_serialises_tool_in_anthropic_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Anthropic-shape response (content list with tool_use block).
        anthropic_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_x",
                                "name": "search_db",
                                "input": {"query": "ACME"},
                            },
                        ],
                    },
                },
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        captured = _patch_tools_acompletion(monkeypatch, anthropic_response)
        result = await ModelClient().complete_with_tools(
            model="claude-4.5-sonnet",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.trace.call_count == 1
        # Tool payload should be the Anthropic shape (no `function` wrapper).
        sent_tool = captured["kwargs"]["tools"][0]
        assert "function" not in sent_tool
        assert sent_tool["name"] == "search_db"
        assert "input_schema" in sent_tool

    async def test_alias_resolves_to_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Use a Gemini-shape response (delegates to OpenAI parser).
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_with_tools(
            model="gemini-2.5-flash",  # alias
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.model_id == "gemini/gemini-2.5-flash"
        assert captured["kwargs"]["model"] == "gemini/gemini-2.5-flash"

    async def test_explicit_temperature_and_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
            temperature=0.7,
            max_tokens=64,
        )
        assert captured["kwargs"]["temperature"] == 0.7
        assert captured["kwargs"]["max_tokens"] == 64

    async def test_extra_kwargs_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
            extra={"tool_choice": "auto"},
        )
        assert captured["kwargs"]["tool_choice"] == "auto"

    async def test_text_only_response_yields_empty_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text_response = {
            "choices": [
                {"message": {"role": "assistant", "content": "Sure thing!"}},
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        _patch_tools_acompletion(monkeypatch, text_response)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.trace.call_count == 0
        assert result.trace.final_text == "Sure thing!"

    async def test_parse_failure_wrapped_as_model_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bad shape that won't pass the parser.
        _patch_tools_acompletion(monkeypatch, {"unexpected": "shape"})
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=1))
        with pytest.raises(ModelError, match="failed to parse tool response"):
            await client.complete_with_tools(model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL])

    async def test_rate_limit_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] < 2:

                class _RLError(Exception):
                    pass

                _RLError.__name__ = "RateLimitError"
                raise _RLError("slow down")
            return _OPENAI_SINGLE_RESPONSE

        _patch_tools_acompletion(monkeypatch, handler)
        result = await ModelClient().complete_with_tools(
            model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL]
        )
        assert result.trace.call_count == 1
        assert attempts["n"] == 2

    async def test_auth_error_short_circuits_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1

            class _AuthenticationError(Exception):
                pass

            _AuthenticationError.__name__ = "AuthenticationError"
            raise _AuthenticationError("bad key")

        _patch_tools_acompletion(monkeypatch, handler)
        with pytest.raises(AuthError):
            await ModelClient().complete_with_tools(model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL])
        assert attempts["n"] == 1

    async def test_deepseek_bare_id_dispatches_openai_shaped_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_with_tools(
            model="deepseek-flash",
            prompt="hi",
            tools=[_DEMO_TOOL],
        )
        assert result.model_id == "deepseek/deepseek-flash"
        assert result.trace.calls[0].tool_name == "search_db"
        kwargs = captured["kwargs"]
        assert kwargs["model"] == "deepseek/deepseek-flash"
        assert kwargs["tools"][0]["type"] == "function"


# ---------------------------------------------------------------------------
# complete_messages / complete_messages_with_tools
# ---------------------------------------------------------------------------


_HISTORY_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "Sunny."},
    {"role": "user", "content": "And tomorrow?"},
]


class TestCompleteMessages:
    async def test_passes_messages_through_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("Sunny too."))
        result = await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=_HISTORY_MESSAGES,
        )
        assert isinstance(result, CompletionResult)
        assert result.text == "Sunny too."
        # The exact list object must reach litellm unmodified.
        assert captured["kwargs"]["messages"] is _HISTORY_MESSAGES

    async def test_delegation_equivalence_with_complete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_a = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="x")
        kwargs_a = captured_a["kwargs"]

        captured_b = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "x"}],
        )
        kwargs_b = captured_b["kwargs"]

        assert kwargs_a == kwargs_b

    async def test_temperature_and_max_tokens_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=_HISTORY_MESSAGES,
            temperature=0.7,
            max_tokens=64,
        )
        assert captured["kwargs"]["temperature"] == 0.7
        assert captured["kwargs"]["max_tokens"] == 64

    async def test_extra_kwargs_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash",
            messages=_HISTORY_MESSAGES,
            extra={"response_format": {"type": "json_object"}},
        )
        assert captured["kwargs"]["response_format"] == {"type": "json_object"}

    async def test_retries_on_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise _ProviderRateLimitError("slow down")
            return _FakeResponse("ok-after-retry")

        _patch_acompletion(monkeypatch, handler)
        result = await ModelClient().complete_messages(
            model="gemini/gemini-2.5-flash", messages=_HISTORY_MESSAGES
        )
        assert result.text == "ok-after-retry"
        assert attempts["n"] == 2


class TestCompleteMessagesWithTools:
    async def test_tools_serialisation_matches_complete_with_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_a = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_with_tools(model="gpt-4o", prompt="hi", tools=[_DEMO_TOOL])
        tools_a = captured_a["kwargs"]["tools"]

        captured_b = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
        )
        tools_b = captured_b["kwargs"]["tools"]

        assert tools_a == tools_b

    async def test_passes_messages_through_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        result = await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=_HISTORY_MESSAGES,
            tools=[_DEMO_TOOL],
        )
        assert isinstance(result, ToolCompletionResult)
        assert captured["kwargs"]["messages"] is _HISTORY_MESSAGES

    async def test_anthropic_serialises_tool_in_anthropic_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        anthropic_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_x",
                                "name": "search_db",
                                "input": {"query": "ACME"},
                            },
                        ],
                    },
                },
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        captured = _patch_tools_acompletion(monkeypatch, anthropic_response)
        result = await ModelClient().complete_messages_with_tools(
            model="claude-4.5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
        )
        assert result.trace.call_count == 1
        sent_tool = captured["kwargs"]["tools"][0]
        assert "function" not in sent_tool
        assert sent_tool["name"] == "search_db"

    async def test_max_tokens_and_extra_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
            max_tokens=321,
            extra={"tool_choice": "auto"},
        )
        assert captured["kwargs"]["max_tokens"] == 321
        assert captured["kwargs"]["tool_choice"] == "auto"

    async def test_auth_error_short_circuits_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"n": 0}

        def handler(**_kwargs: Any) -> Any:
            attempts["n"] += 1

            class _AuthenticationError(Exception):
                pass

            _AuthenticationError.__name__ = "AuthenticationError"
            raise _AuthenticationError("bad key")

        _patch_tools_acompletion(monkeypatch, handler)
        with pytest.raises(AuthError):
            await ModelClient().complete_messages_with_tools(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                tools=[_DEMO_TOOL],
            )
        assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# Empty-output warning
# ---------------------------------------------------------------------------


class TestEmptyOutputWarning:
    async def test_warns_on_empty_text_with_output_tokens_and_stop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=42, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            result = await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert result.text == ""
        assert any(
            "returned no text" in r.message and "42 output tokens" in r.message
            for r in caplog.records
        )

    async def test_no_warning_when_zero_output_tokens(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=0, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert not any("returned no text" in r.message for r in caplog.records)

    async def test_no_warning_when_finish_reason_not_stop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=42, finish_reason="length"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert not any("returned no text" in r.message for r in caplog.records)

    async def test_no_warning_when_text_non_empty(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "hello", prompt_tokens=10, completion_tokens=42, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert not any("returned no text" in r.message for r in caplog.records)

    async def test_warning_names_the_model(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_acompletion(
            monkeypatch,
            lambda **_: _FakeResponse(
                "", prompt_tokens=10, completion_tokens=5, finish_reason="stop"
            ),
        )
        with caplog.at_level("WARNING"):
            await ModelClient().complete(model="gemini/gemini-2.5-flash", prompt="Hi")
        assert any(
            "gemini/gemini-2.5-flash" in r.message and "returned no text" in r.message
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Temperature value rejection (reasoning-tier models)
# ---------------------------------------------------------------------------

# Name-based to match _map_exception's idiom: production raises
# litellm.BadRequestError; the client only inspects the type NAME.
_BadRequestError = type("BadRequestError", (Exception,), {})

_TEMP_400_MSG = (
    "OpenAIException - Unsupported value: 'temperature' does not support 0.0 "
    "with this model. Only the default (1) value is supported."
)


class TestTemperatureValueRejection:
    async def test_adapts_resends_without_temperature_and_records_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            return _FakeResponse("adapted ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with caplog.at_level("WARNING"):
            result = await client.complete(model="gpt-4o", prompt="hi")

        assert result.text == "adapted ok"
        assert len(calls) == 2
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]
        assert client.temperature_rejected_models == frozenset({"openai/gpt-4o"})
        warnings = [r for r in caplog.records if "temperature" in r.getMessage()]
        assert len(warnings) == 1

    async def test_adaptation_does_not_consume_a_retry_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # max_attempts=2. Sequence: temp-400 (adaptation), transient timeout
        # (attempt 1), success (attempt 2). Only passes if the adaptation
        # left the full retry budget intact.
        calls: list[dict[str, Any]] = []
        timeout_error = type("Timeout", (Exception,), {})

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            if len(calls) == 2:
                raise timeout_error("transient")
            return _FakeResponse("ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        result = await client.complete(model="gpt-4o", prompt="hi")
        assert result.text == "ok"
        assert len(calls) == 3

    async def test_later_calls_omit_temperature_preemptively(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def handler(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise _BadRequestError(_TEMP_400_MSG)
            return _FakeResponse("ok")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient()
        await client.complete(model="gpt-4o", prompt="first")
        await client.complete(model="gpt-4o", prompt="second")
        # First call: two dispatches (reject + adapted); second call: one.
        assert len(calls) == 3
        assert "temperature" not in calls[2]

    async def test_non_temperature_400_keeps_existing_retry_then_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(**kwargs: Any) -> Any:
            raise _BadRequestError("Unsupported value: 'tool_choice'")

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(ModelError):
            await client.complete(model="gpt-4o", prompt="hi")
        assert client.temperature_rejected_models == frozenset()

    async def test_temperature_400_without_temperature_in_kwargs_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A model that keeps 400ing about temperature even after the
        # parameter is gone must surface as ModelError, not loop: after the
        # one adaptation kwargs no longer carry temperature, so the
        # predicate's kwargs leg fails and the normal path takes over.
        def handler(**kwargs: Any) -> Any:
            raise _BadRequestError(_TEMP_400_MSG)

        _patch_acompletion(monkeypatch, handler)
        client = ModelClient(retry_policy=RetryPolicy(max_attempts=2))
        with pytest.raises(ModelError):
            await client.complete(model="gpt-4o", prompt="hi")
        # The adaptation itself still fired once and recorded the model.
        assert client.temperature_rejected_models == frozenset({"openai/gpt-4o"})


# ---------------------------------------------------------------------------
# Replay-time tool constraints: tool_choice / parallel_tool_calls / strict
# ---------------------------------------------------------------------------


_STRICT_TOOL = ToolSpec(
    name="search_db",
    description="Search the customer DB",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    strict=True,
)


class TestToolConstraintsReachLiteLLM:
    """The CLI hands LiteLLM OpenAI-style constraints and lets it translate."""

    async def test_tool_choice_and_parallel_flag_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
            extra={"tool_choice": "required", "parallel_tool_calls": False},
        )
        assert captured["kwargs"]["tool_choice"] == "required"
        assert captured["kwargs"]["parallel_tool_calls"] is False

    async def test_named_tool_choice_is_forwarded_verbatim_for_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        choice = {"type": "function", "function": {"name": "search_db"}}
        await ModelClient().complete_messages_with_tools(
            model="claude-4.5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
            extra={"tool_choice": choice},
        )
        assert captured["kwargs"]["tool_choice"] == choice

    async def test_strict_tool_reaches_the_wire_in_both_shapes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_STRICT_TOOL],
        )
        assert captured["kwargs"]["tools"][0]["function"]["strict"] is True

        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="claude-4.5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_STRICT_TOOL],
        )
        assert captured["kwargs"]["tools"][0]["strict"] is True

    async def test_gemini_gaps_dispatch_without_a_second_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The two constraints Gemini silently loses are a run-start record now.

        ``detect_dropped_params`` names them per model in ``state.json``, warns
        once, banners the report, and can fail the verdict; a per-dispatch line
        here would only repeat that, worse. The call itself is unchanged —
        LiteLLM accepts both, and Gemini's own body is where they vanish.
        """
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.models.client"):
            await ModelClient().complete_messages_with_tools(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                tools=[_STRICT_TOOL],
                extra={"tool_choice": "auto", "parallel_tool_calls": False},
            )
        assert caplog.text == ""
        assert captured["kwargs"]["tool_choice"] == "auto"
        assert captured["kwargs"]["parallel_tool_calls"] is False
        assert captured["kwargs"]["tools"][0]["function"]["strict"] is True

    async def test_no_warning_for_expressible_constraints(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.models.client"):
            await ModelClient().complete_messages_with_tools(
                model="claude-4.5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
                tools=[_STRICT_TOOL],
                extra={"tool_choice": "auto", "parallel_tool_calls": False},
            )
        assert caplog.text == ""

    async def test_toolless_path_drops_tool_choice_with_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        with caplog.at_level(logging.WARNING, logger="evalshift_cli.models.client"):
            await ModelClient().complete_messages(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                extra={
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                    "response_format": {"type": "json_object"},
                },
            )
        assert "tool_choice" not in captured["kwargs"]
        assert "parallel_tool_calls" not in captured["kwargs"]
        assert captured["kwargs"]["response_format"] == {"type": "json_object"}
        assert "no tools" in caplog.text

    async def test_caller_extra_is_not_mutated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        extra = {"tool_choice": "required"}
        await ModelClient().complete_messages(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            extra=extra,
        )
        assert extra == {"tool_choice": "required"}


class TestLiteLLMTranslatesToolChoice:
    """Pin what LiteLLM itself does with the OpenAI-style values we send.

    This is the justification for *not* hand-translating in the CLI: LiteLLM's
    provider configs already map OpenAI-style ``tool_choice`` /
    ``parallel_tool_calls`` onto Anthropic's ``tool_choice`` object and
    Gemini's ``toolConfig``. If a LiteLLM upgrade changes that, this test
    fails and the CLI's pass-through decision gets revisited.
    """

    @staticmethod
    def _mapped(model: str, provider: str, tools: list[dict[str, Any]], **kwargs: Any) -> Any:
        from litellm.utils import get_optional_params

        return get_optional_params(
            model=model,
            custom_llm_provider=provider,
            tools=tools,
            drop_params=True,
            **kwargs,
        )

    def test_anthropic_named_choice_becomes_a_tool_object(self) -> None:
        mapped = self._mapped(
            "claude-sonnet-4-5",
            "anthropic",
            [_STRICT_TOOL.to_anthropic()],
            tool_choice={"type": "function", "function": {"name": "search_db"}},
            parallel_tool_calls=False,
        )
        assert mapped["tool_choice"] == {
            "type": "tool",
            "name": "search_db",
            "disable_parallel_tool_use": True,
        }
        # Anthropic-native tools (our wire shape for this provider) pass
        # through untouched, `strict` included.
        assert mapped["tools"][0]["strict"] is True

    def test_anthropic_required_becomes_any(self) -> None:
        mapped = self._mapped(
            "claude-sonnet-4-5",
            "anthropic",
            [_DEMO_TOOL.to_anthropic()],
            tool_choice="required",
        )
        assert mapped["tool_choice"] == {"type": "any"}

    def test_gemini_choice_becomes_tool_config(self) -> None:
        mapped = self._mapped(
            "gemini-2.5-flash",
            "gemini",
            [_DEMO_TOOL.to_openai()],
            tool_choice={"type": "function", "function": {"name": "search_db"}},
        )
        assert mapped["tool_choice"] == {
            "functionCallingConfig": {"mode": "ANY", "allowed_function_names": ["search_db"]}
        }

    def test_gemini_drops_strict_from_function_declarations(self) -> None:
        """Documents the un-expressible pair (gemini, strict)."""
        mapped = self._mapped("gemini-2.5-flash", "gemini", [_STRICT_TOOL.to_openai()])
        declaration = mapped["tools"][0]["function_declarations"][0]
        assert "strict" not in declaration

    def test_openai_passes_both_through_unchanged(self) -> None:
        mapped = self._mapped(
            "gpt-4o",
            "openai",
            [_STRICT_TOOL.to_openai()],
            tool_choice="required",
            parallel_tool_calls=False,
        )
        assert mapped["tool_choice"] == "required"
        assert mapped["parallel_tool_calls"] is False
        assert mapped["tools"][0]["function"]["strict"] is True


# ---------------------------------------------------------------------------
# DeepSeek thinking mode
# ---------------------------------------------------------------------------

_REPLAYED_ROUND: list[dict[str, Any]] = [
    {"role": "user", "content": "find ACME"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_r0_0",
                "type": "function",
                "function": {"name": "search_db", "arguments": '{"query": "ACME"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_r0_0", "content": '{"rows": []}'},
]


class TestDeepSeekReasoningBackfill:
    async def test_replayed_assistant_turn_carries_placeholder_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # DeepSeek 400s a tools request whose earlier assistant turns lack
        # reasoning_content; a teacher-forced round never has DeepSeek's own.
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: True)
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="deepseek-flash",
            messages=[dict(m) for m in _REPLAYED_ROUND],
            tools=[_DEMO_TOOL],
        )
        sent = captured["kwargs"]["messages"]
        assert sent[1]["reasoning_content"] == " "
        assert "reasoning_content" not in sent[0]

    async def test_text_only_history_turn_carries_placeholder_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without tools DeepSeek ignores reasoning_content, but the backfill is
        # applied to every DeepSeek thinking request with messages — including
        # replayed chat history and judge calls.
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: True)
        captured = _patch_acompletion(monkeypatch, lambda **_: _FakeResponse("ok"))
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "what is your refund policy?"},
        ]
        snapshot = [dict(m) for m in history]
        await ModelClient().complete_messages(model="deepseek-flash", messages=history)
        sent = captured["kwargs"]["messages"]
        assert sent[1]["reasoning_content"] == " "
        assert sent[1]["content"] == "Hello! How can I help?"
        assert "reasoning_content" not in sent[0]
        assert "reasoning_content" not in sent[2]
        assert history == snapshot

    async def test_other_providers_are_sent_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "supports_reasoning", lambda **_: True)
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[dict(m) for m in _REPLAYED_ROUND],
            tools=[_DEMO_TOOL],
        )
        assert captured["kwargs"]["messages"] == _REPLAYED_ROUND


_SECOND_TOOL = ToolSpec(
    name="add_note",
    description="Attach a note",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    strict=True,
)


class TestSerializeTools:
    """The one place a toolset becomes the ``tools`` array on the wire.

    The run cache keys on this exact list, so it must be what
    ``complete_messages_with_tools`` sends, in the caller's order.
    """

    def test_openai_shape_in_the_given_order(self) -> None:
        payload = serialize_tools(resolve_model("gpt-4o").id, [_SECOND_TOOL, _DEMO_TOOL])
        assert payload == [_SECOND_TOOL.to_openai(), _DEMO_TOOL.to_openai()]

    def test_anthropic_shape_for_anthropic_models(self) -> None:
        payload = serialize_tools(resolve_model("claude-4.5-sonnet").id, [_DEMO_TOOL, _SECOND_TOOL])
        assert payload == [_DEMO_TOOL.to_anthropic(), _SECOND_TOOL.to_anthropic()]

    async def test_the_wire_carries_exactly_the_serialized_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_SECOND_TOOL, _DEMO_TOOL],
        )
        assert captured["kwargs"]["tools"] == serialize_tools(
            resolve_model("gpt-4o").id, [_SECOND_TOOL, _DEMO_TOOL]
        )

    async def test_dispatch_goes_through_serialize_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Shared source, not a parallel copy: patching the helper changes the wire.
        sentinel = [{"name": "sentinel"}]
        monkeypatch.setattr(client_module, "serialize_tools", lambda _c, _t: sentinel)
        captured = _patch_tools_acompletion(monkeypatch, _OPENAI_SINGLE_RESPONSE)
        await ModelClient().complete_messages_with_tools(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[_DEMO_TOOL],
        )
        assert captured["kwargs"]["tools"] == sentinel
