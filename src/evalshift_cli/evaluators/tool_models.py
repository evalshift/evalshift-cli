"""Provider-agnostic data models for tool calls and traces.

These models normalise the shape of tool-use responses across Anthropic,
OpenAI, and Gemini so the rest of the v0.2 pipeline (parser, evaluators,
report) speaks one language regardless of provider.

The wire format mirrors Anthropic's tool spec (``name`` / ``description`` /
``input_schema``, plus an optional ``strict``), which OpenAI's adapter accepts
as input. We provide
``ToolSpec.to_anthropic()`` / ``.to_openai()`` for outbound serialisation
and ``ToolSpec.from_dict()`` for inbound deserialisation that accepts
either shape.

This module is dependency-free beyond pydantic so the evaluator package
stays cheap to import.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    """Forbid extras + validate on assignment.

    Mirrors the same-named bases in :mod:`evalshift_cli.config.models`,
    :mod:`evalshift_cli.runner.models`, and :mod:`evalshift_cli.suite.models`.
    Kept parallel rather than extracted into a shared location because
    the v0.1 → v0.2 cut deliberately avoids mid-flight refactors of
    cross-cutting types.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ToolSpec(_StrictModel):
    """A tool definition the model can call.

    Wire format mirrors Anthropic's tool spec, which OpenAI's adapter
    accepts as input via ``to_openai()``. Use :meth:`from_dict` to
    deserialise either shape.
    """

    name: str = Field(min_length=1, description="Tool name; must match the model's understanding.")
    description: str = Field(
        default="",
        description=(
            "Human-readable description; the model uses this to decide when to call. May be "
            "empty: a recorded tool with no description fingerprints fine on the SDK side "
            "(evalshift.capture.toolset.normalize_tools), and the CLI is not the source of "
            "truth about what production offered -- it does not get to reject that."
        ),
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing the tool's arguments.",
    )
    strict: bool = Field(
        default=False,
        description=(
            "Whether the provider must constrain generated arguments to ``input_schema`` "
            "exactly (OpenAI strict function calling / Anthropic strict tools). Recorded by "
            "the SDK only when production set it, and only ever ``true`` on the wire -- a "
            "tool without the key is not strict."
        ),
    )

    def to_anthropic(self) -> dict[str, Any]:
        """Serialise in Anthropic / LiteLLM-Anthropic-path shape.

        ``strict`` is emitted as a top-level key, and only when set: this is
        also the canonical shape the SDK fingerprints toolsets in
        (:func:`evalshift_cli.captures.toolset.fingerprint_tools`), so a
        non-strict tool must serialise byte-identically to before.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
        if self.strict:
            payload["strict"] = True
        return payload

    def to_openai(self) -> dict[str, Any]:
        """Serialise in OpenAI function-calling shape.

        ``strict`` lives inside the ``function`` object here, and is emitted
        only when set.
        """
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
        if self.strict:
            function["strict"] = True
        return {"type": "function", "function": function}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Accept either Anthropic or OpenAI shape and produce a :class:`ToolSpec`.

        Args:
            payload: A single tool dict in either provider shape.

        Returns:
            The normalised :class:`ToolSpec`.

        Raises:
            ValueError: If the payload doesn't look like either shape.
        """
        if "function" in payload:
            fn = payload["function"]
            if not isinstance(fn, dict):
                raise ValueError("OpenAI-shape tool payload missing 'function' object")
            return cls(
                name=fn["name"],
                description=fn.get("description", ""),
                input_schema=fn.get("parameters", {}),
                # OpenAI carries strictness inside the function object.
                strict=fn.get("strict") is True,
            )
        if "name" not in payload:
            raise ValueError("tool payload missing 'name' field")
        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            input_schema=payload.get("input_schema", payload.get("parameters", {})),
            # Canonical / Anthropic shape carries it at the top level. Only a
            # literal `true` counts: the SDK writes the key exclusively when
            # it is true, so anything else is not a strictness assertion.
            strict=payload.get("strict") is True,
        )


class ToolCall(_StrictModel):
    """A single tool invocation parsed from a model response.

    Attributes:
        tool_name: The name of the tool the model invoked.
        arguments: The arguments the model passed. ``{"_parse_error": True}``
            indicates the underlying provider returned a malformed JSON
            argument string; downstream evaluators handle this.
        call_id: The provider-assigned id for this call (when available).
        parent_call_id: For chained / nested calls. ``None`` for top-level
            calls. v0.2 treats two top-level calls (``parent_call_id is None``)
            as parallel.
        sequence_index: 0-indexed position within the trace. Keeps counting
            across rounds, so it is unique per trace.
        round_index: Which model response this call came from, for a
            teacher-forced multi-round replay (see :attr:`ToolTrace.round_count`).
            ``0`` for every single-shot call, and for every trace written
            before the field existed.
    """

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    parent_call_id: str | None = None
    sequence_index: int = Field(ge=0)
    round_index: int = Field(default=0, ge=0)


class ToolTrace(_StrictModel):
    """The full sequence of tool calls in one model response.

    Or, for a teacher-forced multi-round replay, in several: ``round_count``
    responses were made for one example, each call is tagged with the
    ``round_index`` it came from, and :meth:`rounds` splits the trace back into
    one single-round trace per response so per-response scoring runs unchanged.
    ``final_text`` / ``raised_refusal`` describe the *last* response -- the
    answer a user would have seen.

    A trace can be:

    * **Empty** (``calls=[]``) — the model produced text only.
    * **Tool-only** (``calls=[...]``, ``final_text=None``) — the model
      issued tool calls without a final text answer.
    * **Mixed** (``calls=[...]``, ``final_text="..."``) — both happened.
    * **Refusal** (``raised_refusal=True``) — the model declined to act.

    Attributes:
        calls: Tool invocations in the order the model emitted them.
        final_text: Free-form text the model produced after / instead of
            tool calls. ``None`` if the response was tool-only.
        raised_refusal: True if the response indicates refusal.
        refusal_text: Optional text accompanying a refusal.
        round_count: How many model responses this trace spans. ``1`` for a
            single-shot call and for every trace written before the field
            existed; a teacher-forced replay sets it to the number of rounds
            it made. Every call's ``round_index`` is below it.
    """

    calls: list[ToolCall] = Field(default_factory=list)
    final_text: str | None = None
    raised_refusal: bool = False
    refusal_text: str | None = None
    round_count: int = Field(default=1, ge=1)

    @property
    def call_count(self) -> int:
        """Number of tool calls in this trace."""
        return len(self.calls)

    @property
    def ends_in_tool_calls(self) -> bool:
        """True iff the last response called a tool, so it awaits results.

        Text beside such a call is narration ("Let me check!"), never the
        answer a user would read: that answer comes after the results.
        """
        last = self.round_count - 1
        return any(c.round_index == last for c in self.calls)

    @property
    def tool_names(self) -> list[str]:
        """Tool names in the order they were called."""
        return [c.tool_name for c in self.calls]

    @property
    def tool_name_set(self) -> set[str]:
        """Distinct tool names called."""
        return {c.tool_name for c in self.calls}

    def has_parallel_calls(self) -> bool:
        """True iff some single response contains two or more top-level calls.

        v0.2 treats "top-level" as ``parent_call_id is None``. Provider
        adapters may refine this once we have richer parent/child signals
        from the underlying SDKs. Parallelism is a property of one response:
        two rounds of one call each are sequential, not a fan-out, so a
        multi-round trace is checked round by round.
        """
        by_round: dict[int, int] = {}
        for call in self.calls:
            if call.parent_call_id is None:
                by_round[call.round_index] = by_round.get(call.round_index, 0) + 1
        return any(n >= 2 for n in by_round.values())

    def round(self, index: int) -> ToolTrace:
        """The calls of one response, as a single-round trace.

        ``sequence_index`` is renumbered from 0 and ``round_index`` reset, so
        the result is exactly what a single-shot call would have produced.
        ``final_text`` and the refusal flags belong to the last round only.

        Raises:
            IndexError: If ``index`` is not below :attr:`round_count`.
        """
        if not 0 <= index < self.round_count:
            raise IndexError(f"round {index} out of range for a {self.round_count}-round trace")
        calls = [
            c.model_copy(update={"sequence_index": position, "round_index": 0})
            for position, c in enumerate(c for c in self.calls if c.round_index == index)
        ]
        last = index == self.round_count - 1
        return ToolTrace(
            calls=calls,
            final_text=self.final_text if last else None,
            raised_refusal=self.raised_refusal if last else False,
            refusal_text=self.refusal_text if last else None,
        )

    def rounds(self) -> list[ToolTrace]:
        """Every response as its own single-round trace, in order.

        A single-round trace returns ``[self]`` unchanged, so callers that
        loop over rounds see exactly the trace they were given today.
        """
        if self.round_count == 1:
            return [self]
        return [self.round(k) for k in range(self.round_count)]

    def calls_by_tool(self, tool_name: str) -> list[ToolCall]:
        """Return every call for the given ``tool_name`` (in trace order)."""
        return [c for c in self.calls if c.tool_name == tool_name]

    @model_validator(mode="after")
    def _check_sequence_indices_unique(self) -> Self:
        """Reject traces where two calls share a ``sequence_index``.

        Provider adapters always assign sequential indices; a duplicate
        indicates a parser bug or hand-crafted invalid data.
        """
        indices = [c.sequence_index for c in self.calls]
        if len(indices) != len(set(indices)):
            duplicates = sorted({i for i in indices if indices.count(i) > 1})
            raise ValueError(f"duplicate sequence_index in trace: {duplicates}")
        return self

    @model_validator(mode="after")
    def _check_round_indices_in_range(self) -> Self:
        """Reject a call tagged with a round the trace says it never made."""
        out_of_range = sorted(
            {c.round_index for c in self.calls if c.round_index >= self.round_count}
        )
        if out_of_range:
            raise ValueError(
                f"round_index {out_of_range} out of range for round_count={self.round_count}",
            )
        return self


__all__ = ["ToolCall", "ToolSpec", "ToolTrace"]
