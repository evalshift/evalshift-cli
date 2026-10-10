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
        isinstance(item.get("role"), str)
        or (isinstance(item.get("type"), str) and item.get("type") in _KNOWN_ITEMS)
        for item in raw
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
        elif calls:
            # tool_calls are assistant-only: a tool_use on any other turn has nowhere to replay.
            self.unreplayable.extend(
                f"recorded message {position} ({recorded_role}) carries a `tool_use` block"
                for _call in calls
            )
            calls = []
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
            if not isinstance(kind, str):  # unhashable kinds would break the set tests below
                self.unreplayable.append(f"{where} carries a `{kind}` block")
            elif kind in _TEXT_BLOCKS:
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
            self.unreplayable.append(
                f"recorded message {position} carries a tool call with no name"
            )
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
