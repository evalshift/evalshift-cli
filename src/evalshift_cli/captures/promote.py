"""Promote a capture into a golden suite case.

Promotion turns a recorded agent run into a self-contained golden test case.
The capture supplies the **ground truth** — what tools the agent called and
what it ultimately answered — which becomes the ``expected_tools`` /
``expected`` an evaluator scores a candidate model against.

Two halves:

* :func:`build_example_from_capture` — the pure, side-effect-free mapping from
  a :class:`CaptureEnvelope` to a :class:`SuiteExample`. Easy to unit-test.
* :func:`write_promoted_case` / :func:`rebuild_golden_jsonl` — the disk side.
  Each promoted case is written as a canonical :class:`PromotedCase` file, and
  ``golden.jsonl`` is *regenerated* from every case in the suite dir so the
  dir form and the run-facing JSONL can never drift.

Known limitation: a capture stores a one-way ``input_hash``, not the raw agent
input. ``inputs`` recovery is therefore best-effort — see
:func:`_recover_inputs`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.captures.native_messages import normalize_messages
from evalshift_cli.captures.reader import CaptureRecord, suites_root, toolset_path
from evalshift_cli.suite.models import (
    ChatMessage,
    ExpectedToolCall,
    HistoryToolCall,
    SuiteExample,
    ToolResultFixture,
)
from evalshift_cli.suite.tags import CAPTURED_TAG
from evalshift_cli.traces.models import (
    ErrorEvent,
    FinalOutputEvent,
    ModelCallEvent,
    RequestedToolCall,
    ToolCallEvent,
    ToolResultEvent,
)
from evalshift_cli.utils.cost import estimate_call_cost

# Context-loss heuristic: warn when the capture's recorded ``input_tokens``
# imply far more context was actually sent to the model than we managed to
# recover into ``inputs``/``history`` — a strong signal the SDK caller only
# passed the current-turn string to ``capture.model_call(input=...)`` instead
# of the full messages list, so a replay would silently drop the system
# prompt / conversation history.
_CONTEXT_LOSS_RATIO = 4
_CONTEXT_LOSS_MIN_GAP_TOKENS = 300

#: Why :func:`build_example_from_capture` refused a capture -- see :class:`BuiltExample`.
BlockedReason = Literal[
    "errored", "no_toolset", "multi_toolset", "unreplayable_content", "invalid_case"
]


@dataclass(frozen=True, slots=True)
class PromoteOptions:
    """Knobs controlling how a capture maps to a golden case.

    Attributes:
        name: Case id (and dir-form filename stem). Defaults to the capture id.
        input_var: Template-variable name used when the recorded model input is
            a bare string (``inputs = {input_var: <string>}``).
        tags: Extra tags appended after the automatic ``["captured", <suite>]``.
        strict_args: Use ``match_strategy="exact"`` for expected tools (default
            ``subset``).
        names_only: Drop recorded arguments — match tool *names* only.
        tool_count: Also pin ``expected_tool_count`` to the recorded count.
        allow_errored: Promote a capture whose turn recorded an ``error``
            event anyway. The case is still never given
            ``expected_no_tools=True`` — a failed turn is not evidence that
            calling nothing was correct.
        rounds: How many recorded agent rounds the case is replayed over.
            ``first`` (default) writes no fixtures: replay is single-shot,
            scored against round 1, the only round it can reproduce.
            ``all`` additionally carries the recorded tool results as
            ``tool_result_fixtures``, so the runner replays each round
            teacher-forced — the candidate sees the recorded rounds before
            it and is scored against that round's own ground truth.
            ``expected_tools`` is round 1 either way; ``all`` no longer
            flattens the rounds into it.
        tool_properties: Declared argument property names per tool, resolved
            from the recorded toolset of the capture(s) being promoted (see
            ``cli.commands.capture._declared_tool_properties``). Used only to
            recognise (and undo) a capture that recorded a wrapper function's
            parameters instead of the arguments the model passed — see
            :func:`_unwrap_recorded_arguments`. ``None`` disables the check:
            without the schema there is no evidence, and guessing at a
            recorded shape is worse than leaving it alone.
    """

    name: str | None = None
    input_var: str = "input"
    tags: tuple[str, ...] = ()
    strict_args: bool = False
    names_only: bool = False
    tool_count: bool = False
    allow_errored: bool = False
    rounds: Literal["first", "all"] = "first"
    tool_properties: Mapping[str, frozenset[str]] | None = None


@dataclass(frozen=True, slots=True)
class BuiltExample:
    """A promoted :class:`SuiteExample` plus any best-effort warnings.

    Attributes:
        example: The mapped case.
        warnings: Non-fatal notes for the operator.
        blocked: When set, the reason this capture should not be promoted.
            Callers must skip it unless the operator opted in explicitly —
            a capture whose turn failed records what the agent *couldn't*
            do, which is the opposite of ground truth.
        blocked_reason: The *kind* of refusal ``blocked`` describes, for
            callers that report an aggregate summary and need to route each
            refusal to an accurate counter/message rather than lump every
            refusal under one (see ``cli/commands/capture.py``'s
            ``capture sync`` summary). ``"errored"`` is the one reason
            ``--allow-errored`` can rescue; ``"no_toolset"`` covers both a
            capture missing ``toolset_ref`` entirely and one whose
            ``toolset_ref`` has no resolvable sidecar; ``"multi_toolset"``
            covers a capture whose ``model_call`` events recorded more than
            one distinct ``toolset_ref`` (the SDK stamps one per call, so a
            runtime toolset switch is legitimate — but ``SuiteExample``
            carries exactly one toolset for its whole trace, so such a
            capture cannot yet be promoted as a single case).
            ``"unreplayable_content"`` covers a capture whose recorded messages carry
            content a text replay can't carry (an image, a document, a server-tool
            result, an unknown block type) — a replay without it is not ground truth.
            ``"invalid_case"`` is the safety net: the mapped case failed
            ``SuiteExample``'s own validation in a way no rule above anticipated, so
            ``example`` is an inert placeholder and ``blocked`` names pydantic's first error.
            None of the latter four is rescued by ``--allow-errored``: re-capturing fixes
            the first, and the others reproduce identically unless the app changes what
            it records. ``None`` when ``blocked`` is ``None``.
        promotion_source: Which yardstick the tool-call ground truth came
            from — ``"requested"`` when the capture recorded the calls the
            model itself asked for, ``"executed"`` when it fell back to the
            ``tool_call`` events the app recorded. Copied onto the
            :class:`~evalshift_cli.captures.models.PromotedCase` so reports
            can say which one a row measures.
        cost_usd: What the recorded run cost, summed over its ``model_call``
            events — see :func:`_capture_cost` for the rule. Copied onto the
            :class:`~evalshift_cli.captures.models.PromotedCase`, never onto
            the example.
        cost_source: ``"recorded"`` when the figure is entirely what the
            app's instrumentation set, ``"estimated"`` when any part of it
            was priced here from litellm's table, ``None`` when it is 0.
    """

    example: SuiteExample
    warnings: list[str] = field(default_factory=list)
    blocked: str | None = None
    blocked_reason: BlockedReason | None = None
    promotion_source: Literal["requested", "executed"] = "executed"
    cost_usd: float = 0.0
    cost_source: Literal["recorded", "estimated"] | None = None


@dataclass(frozen=True, slots=True)
class RecoveredInput:
    """Result of recovering ``inputs`` (+ optional ``history``) from a trace.

    Attributes:
        inputs: Template-variable values for the current turn.
        history: The conversation prefix recorded before the current turn,
            when the capture's first ``model_call.input`` was a full messages
            list. ``None`` when the recorded input was a bare dict/string (no
            history to recover).
        unreplayable: One line per recorded element a text replay can't carry
            (see :func:`~evalshift_cli.captures.native_messages.normalize_messages`).
            Non-empty blocks the capture.
    """

    inputs: dict[str, Any]
    history: list[ChatMessage] | None = None
    unreplayable: list[str] = field(default_factory=list)


def build_example_from_capture(
    envelope: CaptureEnvelope,
    opts: PromoteOptions,
    *,
    base: Path | None = None,
) -> BuiltExample:
    """Map a capture to a :class:`SuiteExample`.

    Almost pure: the only disk access is a single ``Path.is_file()`` check
    (see :func:`~evalshift_cli.captures.reader.toolset_path`) confirming the
    recorded ``toolset_ref``'s sidecar actually exists, so a capture whose
    sidecar was deleted (or never written) is refused here rather than
    promoting cleanly and crashing later, the first time something tries to
    resolve it. It does not read the sidecar's contents — promotion carries
    the *reference*, not the tool bodies.

    Args:
        envelope: The capture to map.
        opts: Promotion knobs.
        base: Capture base dir the sidecar existence check resolves
            against. ``None`` resolves via
            :func:`~evalshift_cli.captures.reader.capture_base`.
    """
    events = envelope.trace.events
    warnings: list[str] = []

    recovered = _recover_inputs(events, input_var=opts.input_var, warnings=warnings)
    loss = _context_loss_warning(events, recovered.inputs, recovered.history)
    if loss is not None:
        warnings.append(loss)
    tags = _build_tags(envelope.suite, opts.tags)
    expected = _recover_expected(events, warnings)

    match_strategy: Literal["exact", "subset", "contains_per_field"] = (
        "exact" if opts.strict_args else "subset"
    )
    unwrapped_keys: set[str] = set()

    def _from_executed(call: ToolCallEvent) -> ExpectedToolCall:
        arguments: dict[str, Any] | None = None
        if not opts.names_only:
            arguments, wrapper = _unwrap_recorded_arguments(
                call.name,
                dict(call.arguments),
                opts.tool_properties,
            )
            if wrapper is not None:
                unwrapped_keys.add(wrapper)
        return ExpectedToolCall(
            tool_name=call.name,
            arguments=arguments,
            match_strategy=match_strategy,
            # Stated at the write site, not left to the field default: these
            # arguments are a transcript of what the source model did, which
            # nobody has yet confirmed is what it *should* have done. A human
            # who checks a row flips it to `reviewed`.
            provenance="captured",
        )

    def _from_requested(call: RequestedToolCall) -> ExpectedToolCall:
        # No unwrapping here, deliberately: _unwrap_recorded_arguments exists
        # because a decorated Python function's signature stood between the
        # model and the recording. Nothing stands between the model and its
        # own requested call, so the same {"tool_args": {...}} shape here IS
        # what the model produced, and rewriting it would corrupt the one
        # yardstick a candidate model can actually be held to.
        return ExpectedToolCall(
            tool_name=call.name,
            arguments=None if opts.names_only else dict(call.arguments),
            match_strategy=match_strategy,
            provenance="captured",
        )

    event_rounds = _event_rounds(events)
    executed_rounds = _tool_rounds(events)
    requested_rounds = _requested_tool_rounds(events, warnings)
    promotion_source: Literal["requested", "executed"] = (
        "executed" if requested_rounds is None else "requested"
    )
    # `result_rounds[k]` holds the tool_result events recorded in the same
    # round as `tool_rounds[k]`, so it must be filtered by whichever grouping
    # produced tool_rounds -- the tool-emitting rounds for executed calls, the
    # requesting model_calls for requested ones.
    call_ids: list[list[str | None]]
    if requested_rounds is None:
        tool_rounds = [[_from_executed(c) for c in r] for r in executed_rounds]
        call_ids = [[c.call_id for c in r] for r in executed_rounds]
        result_rounds = [r.results for r in event_rounds if r.calls]
    else:
        disagreement = _requested_vs_executed_warning(requested_rounds, executed_rounds)
        if disagreement is not None:
            warnings.append(disagreement)
        tool_rounds = [[_from_requested(c) for c in r] for r in requested_rounds]
        call_ids = [[c.call_id for c in r] for r in requested_rounds]
        result_rounds = [
            r.results
            for r in event_rounds
            if r.model_call is not None and r.model_call.requested_tool_calls
        ]

    # No flattening under --rounds all any more: expected_tools is always
    # round 1. The later rounds travel as expected_tool_rounds plus the
    # fixtures below and are scored round by round, which is a yardstick a
    # replay can actually meet -- the flattened list never was.
    scoped_calls = tool_rounds[0] if tool_rounds else []

    first_model_call = next((e for e in events if isinstance(e, ModelCallEvent)), None)
    toolset_ref = first_model_call.toolset_ref if first_model_call is not None else None
    tools_offered = first_model_call.tools_offered if first_model_call is not None else None

    # Every distinct toolset_ref the capture's model_call events recorded, in
    # first-seen (round) order. The SDK stamps toolset_ref per call, so more
    # than one here means the agent switched toolsets mid-run -- legitimate,
    # not corruption (see the module docstring's "several distinct
    # toolset_ref values" note) -- but SuiteExample has only one toolset
    # field for the whole example (I1 of the final review).
    distinct_toolset_refs = list(
        dict.fromkeys(
            e.toolset_ref
            for e in events
            if isinstance(e, ModelCallEvent) and e.toolset_ref is not None
        ),
    )

    errors = _error_events(events)
    blocked: str | None = None
    blocked_reason: BlockedReason | None = None
    if toolset_ref is None:
        # Unconditional -- unlike an errored turn, --allow-errored cannot
        # rescue this: there is no recorded toolset to promote at all, only
        # re-capturing with an SDK that stamps toolset_ref produces one.
        blocked = (
            f"capture {envelope.capture_id!r} has no toolset_ref on its first model call — "
            "the SDK did not record which tools were offered (or failed to write the "
            "sidecar). Re-capture with a current evalshift-sdk to promote it."
        )
        blocked_reason = "no_toolset"
    elif not (sidecar_path := toolset_path(toolset_ref, base=base)).is_file():
        # Also unconditional, same reasoning as the missing-ref case above:
        # the SDK *did* record a toolset_ref, but its sidecar isn't on disk
        # (deleted by `capture clean`, moved, or never written) -- there is
        # no toolset to carry, existence-only, deliberately not a full
        # `load_toolset` resolution (promotion never needs the tool bodies,
        # and resolving every capture's sidecar here would cost every
        # existing promote fixture a real file on disk for no benefit).
        blocked = (
            f"capture {envelope.capture_id!r} references toolset_ref {toolset_ref!r}, but its "
            f"sidecar is missing ({sidecar_path}) — it may have been deleted, moved by "
            "`capture clean`, or never written. Re-capture with a current evalshift-sdk to "
            "promote it."
        )
        blocked_reason = "no_toolset"
    elif len(distinct_toolset_refs) > 1:
        # Unconditional too, and independent of opts.rounds: expected_tool_rounds
        # (populated below) always retains every round regardless of --rounds, so
        # even the default --rounds first would carry a later round's calls
        # against a toolset the example's single toolset_ref doesn't name.
        refs_list = ", ".join(repr(r) for r in distinct_toolset_refs)
        blocked = (
            f"capture {envelope.capture_id!r} recorded {len(distinct_toolset_refs)} distinct "
            f"toolsets across its model_call events ({refs_list}) — a promoted example carries "
            "exactly one toolset for its whole trace, so a capture whose agent switched "
            "toolsets mid-run cannot yet be promoted as a single case. Promote or hand-edit "
            "each toolset's rounds separately."
        )
        blocked_reason = "multi_toolset"
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
    elif errors:
        detail = errors[0].message.strip().replace("\n", " ")[:160]
        note = (
            f"capture recorded {len(errors)} error event(s) — the turn failed before the agent "
            f"finished, so it is not ground truth. First error: {detail}"
        )
        if opts.allow_errored:
            warnings.append(note)
        else:
            blocked = note
            blocked_reason = "errored"

    failed_results = _failed_tool_results(events)
    if failed_results:
        names = ", ".join(sorted(set(failed_results)))
        warnings.append(
            f"{len(failed_results)} recorded tool result(s) failed ({names}) — this turn is "
            "promoted as ground truth including the failing call(s). Review whether that is "
            "the behaviour you want a candidate model to reproduce.",
        )

    if _has_no_scoreable_ground_truth(events):
        warnings.append(
            "turn has no scoreable ground truth (empty recorded output, no tool calls); "
            "case will only exercise structural checks",
        )

    if opts.rounds == "first" and len(tool_rounds) > 1:
        dropped = sum(len(r) for r in tool_rounds[1:])
        warnings.append(
            f"capture spans {len(tool_rounds)} agent round(s); expected_tools was scoped to "
            f"round 1 ({len(tool_rounds[0])} call(s)) and {dropped} later call(s) were moved to "
            "expected_tool_rounds. A single-shot replay cannot reach round 2 — promote with "
            "--rounds all to replay the later rounds teacher-forced, feeding the recorded tool "
            "results back and scoring each round against its own ground truth.",
        )

    # Teacher-forced replay: carry the recorded results of the ground-truth
    # calls so the runner can put the candidate in the same context the
    # recorded agent was in at the start of each round.
    tool_result_fixtures: list[list[ToolResultFixture]] | None = None
    covered_rounds = 0
    if opts.rounds == "all" and tool_rounds:
        fixture_rounds, unpaired = _replay_fixtures(tool_rounds, call_ids, result_rounds)
        covered_rounds = len(fixture_rounds)
        tool_result_fixtures = fixture_rounds or None
        if unpaired is not None:
            warnings.append(
                f"round {covered_rounds + 1} has {unpaired} tool call(s) with no recorded "
                + (
                    "result; replay stays single-shot"
                    if covered_rounds == 0
                    else f"result; replay will cover rounds 1..{covered_rounds}"
                ),
            )

    # Under --rounds all the pinned count spans every round the replay
    # reaches: the covered ones plus the round after them, which is what
    # tool_trace_structure compares the candidate's total call count to.
    counted_calls = (
        sum(len(r) for r in tool_rounds[: covered_rounds + 1])
        if opts.rounds == "all"
        else len(scoped_calls)
    )

    if scoped_calls:
        expected_tools: list[ExpectedToolCall] | None = [c.model_copy() for c in scoped_calls]
        expected_no_tools = False
    else:
        expected_tools = None
        # expected_no_tools asserts "tools were offered and none were
        # called" -- it must never fire when nothing was offered (there is
        # nothing a candidate model could have failed to call, and
        # _score_no_tools would award 1.0/1.0 to an unmeasurable row), and a
        # turn that errored produced no tool calls because it never ran, not
        # because calling nothing was right.
        expected_no_tools = bool(tools_offered) and not errors
    expected_tool_rounds = tool_rounds if tool_rounds else None
    if unwrapped_keys:
        keys = ", ".join(sorted(unwrapped_keys))
        warnings.append(
            f"recorded tool arguments were wrapped in {{{keys}}} — the capture recorded the "
            "wrapper function's parameters, not the arguments the model passed. The declared "
            "tool schema confirms the inner shape, so expected_tools was unwrapped to match "
            "what a model can actually produce.",
        )

    generation_config: dict[str, Any] | None = None
    if first_model_call is not None:
        raw = first_model_call.metadata.get("generation_config")
        if isinstance(raw, dict) and raw:
            generation_config = raw

    # SuiteExample requires exactly one of toolset_ref/tools. A blocked
    # capture with no toolset_ref at all (see the first blocking condition
    # above) has no real toolset to carry, but .example must still construct
    # -- callers gate on .blocked, not on catching a ValidationError -- so it
    # gets the inert tools=[] placeholder every other "we don't know" site in
    # this codebase uses (see cli/commands/evaluate.py's _score_one_tool).
    # SuiteExample now rejects tools=[] paired with tool-call ground truth
    # (I2): scoped_calls/tool_rounds above were computed from the trace
    # regardless of whether a toolset_ref exists to dispatch them against
    # (e.g. a tool_call recorded with no preceding model_call at all still
    # opens a defensive round -- see _tool_rounds), so the placeholder must
    # drop any ground truth here too to stay genuinely inert. A blocked
    # capture's .example is never scored, so nothing of substance is lost.
    no_real_toolset = toolset_ref is None
    try:
        example = SuiteExample(
            id=opts.name or envelope.capture_id,
            inputs=recovered.inputs,
            tags=tags,
            expected=expected,
            expected_tools=None if no_real_toolset else expected_tools,
            expected_tool_rounds=None if no_real_toolset else expected_tool_rounds,
            tool_result_fixtures=None if no_real_toolset else tool_result_fixtures,
            expected_tool_count=(
                None if no_real_toolset else (counted_calls if opts.tool_count else None)
            ),
            expected_no_tools=expected_no_tools,
            history=recovered.history,
            conversation_id=envelope.conversation_id,
            turn_index=envelope.turn_index,
            generation_config=generation_config,
            toolset_ref=toolset_ref,
            tools=[] if no_real_toolset else None,
        )
    except ValidationError as exc:
        # The safety net under every rule above: a capture none of them anticipated must
        # skip with a reason, never abort the whole sync with a raw ValidationError. The
        # placeholder is the inert one the no-toolset refusal builds.
        blocked = (
            f"capture {envelope.capture_id!r} could not be turned into a valid case — "
            f"{exc.errors()[0]['msg']}. Hand-write the case or report this capture."
        )
        blocked_reason = "invalid_case"
        example = SuiteExample(id=opts.name or envelope.capture_id, inputs={}, tags=tags, tools=[])
    cost_usd, cost_source = _capture_cost(events)
    return BuiltExample(
        example=example,
        warnings=warnings,
        blocked=blocked,
        blocked_reason=blocked_reason,
        promotion_source=promotion_source,
        cost_usd=cost_usd,
        cost_source=cost_source,
    )


def _capture_cost(
    events: list[Any],
) -> tuple[float, Literal["recorded", "estimated"] | None]:
    """Total cost of the capture's ``model_call`` events, and where it came from.

    The SDK never prices anything — ``cost_usd`` on a ``model_call`` is 0.0
    unless the app's own instrumentation set it, and the provider client
    wrappers record tokens but leave cost at 0 by design. Pricing is the
    CLI's job, here, from litellm's price table.

    Per event: a non-zero recorded ``cost_usd`` is kept exactly as recorded,
    never re-estimated; otherwise, when the event recorded any tokens, it is
    priced by its *own* ``model_id`` via
    :func:`~evalshift_cli.utils.cost.estimate_call_cost`, which yields 0 for
    a model litellm does not price (local / self-hosted — normal, so no
    warning). The per-event figures are summed. The tag is ``"estimated"``
    when any event contributed a non-zero estimate, else ``"recorded"`` when
    any contributed a recorded cost, else ``None`` (nothing priced, total
    0.0). So a run mixing one recorded and one estimated call is
    ``estimated`` — the figure is only as trustworthy as its least certain
    part — while a recorded call alongside an unpriced one stays
    ``recorded``, because everything in the sum was.
    """
    total = 0.0
    any_recorded = False
    any_estimated = False
    for event in events:
        if not isinstance(event, ModelCallEvent):
            continue
        if event.cost_usd > 0:
            total += event.cost_usd
            any_recorded = True
            continue
        estimate = estimate_call_cost(event.model_id, event.input_tokens, event.output_tokens)
        if estimate > 0:
            total += estimate
            any_estimated = True
    if any_estimated:
        return total, "estimated"
    if any_recorded:
        return total, "recorded"
    return 0.0, None


def _unwrap_recorded_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tool_properties: Mapping[str, frozenset[str]] | None,
) -> tuple[dict[str, Any], str | None]:
    """Undo a single-dict wrapper around recorded tool arguments.

    A capture SDK that decorates a Python function records that *function's*
    parameters. When the function takes one dict — ``def archive_project(
    tool_args: dict)`` — every call is recorded as
    ``{"tool_args": {"project_name": ...}}``, while the model only ever saw the
    declared schema's flat properties. No model can produce the recorded shape,
    so ground truth in it scores 0 against every candidate.

    Unwrapping requires the declared schema to *confirm* the shape on both
    sides: the wrapper key must not be a declared property, and the inner keys
    must all be declared ones. Without that evidence the recording is returned
    untouched — a wrong guess here silently rewrites ground truth.

    Args:
        tool_name: Name of the tool the arguments belong to.
        arguments: The arguments exactly as recorded.
        tool_properties: Declared property names per tool, or ``None``.

    Returns:
        ``(arguments, wrapper_key)`` — the arguments to use, and the wrapper
        key that was removed, or ``None`` when nothing was unwrapped.
    """
    if tool_properties is None or len(arguments) != 1:
        return arguments, None
    declared = tool_properties.get(tool_name)
    if not declared:
        return arguments, None

    (key,), (value,) = tuple(arguments.keys()), tuple(arguments.values())
    if key in declared or not isinstance(value, dict):
        return arguments, None
    # An empty wrapper is the no-argument call — vacuously consistent with the
    # schema, and the shape this bug most often takes.
    if value and not set(value).issubset(declared):
        return arguments, None
    return dict(value), key


def _error_events(events: list[Any]) -> list[ErrorEvent]:
    """Error events recorded during the turn, in sequence order."""
    return [e for e in events if isinstance(e, ErrorEvent)]


def _failed_tool_results(events: list[Any]) -> list[str]:
    """Names of tools whose recorded result signals failure.

    Recognises an explicit ``error`` on the event and the common
    ``{"success": false}`` convention in a dict result. Anything else is
    opaque to us and is left alone — a tool's result schema is the app's,
    not ours.
    """
    failed: list[str] = []
    for event in events:
        if not isinstance(event, ToolResultEvent):
            continue
        if event.error or (isinstance(event.result, dict) and event.result.get("success") is False):
            failed.append(event.name)
    return failed


def _has_no_scoreable_ground_truth(events: list[Any]) -> bool:
    """True when a turn recorded no final output, no non-empty model output, no tool calls.

    A model that *requested* tools counts, even when the app executed none:
    the requested list is what promotion scores against, so the case is
    measurable regardless of what the app went on to run.
    """
    if any(isinstance(e, ToolCallEvent) for e in events):
        return False
    if any(isinstance(e, ModelCallEvent) and e.requested_tool_calls for e in events):
        return False
    if any(isinstance(e, FinalOutputEvent) and e.text for e in events):
        return False
    return not any(isinstance(e, ModelCallEvent) and _nonempty_text(e.output) for e in events)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


@dataclass(slots=True)
class _EventRound:
    """One agent round's events — the ``model_call`` that opened it, then what ran.

    A capture's events are sequence-ordered: a ``model_call``, then the tool
    calls that model call emitted and the results they returned, then the next
    ``model_call``. Splitting on ``model_call`` therefore recovers the rounds,
    and — the reason this exists rather than just :func:`_tool_rounds` — keeps
    each round's ``tool_result`` events with the calls they answered, which is
    the boundary the fixture pairing must not cross.

    Attributes:
        model_call: The event that opened the round, or ``None`` for the
            defensive leading round a trace whose first tool call precedes
            any ``model_call`` needs.
        calls: The round's ``tool_call`` events, in order.
        results: The round's ``tool_result`` events, in order.
    """

    model_call: ModelCallEvent | None
    calls: list[ToolCallEvent] = field(default_factory=list)
    results: list[ToolResultEvent] = field(default_factory=list)


def _event_rounds(events: list[Any]) -> list[_EventRound]:
    """Split a capture's events into rounds at each ``model_call``.

    Unlike :func:`_tool_rounds` nothing is filtered: every ``model_call``
    yields a round, including the trailing text-only one, so a caller can
    line the result of this up with either grouping (executed calls or
    model-requested calls) by applying that grouping's own filter.

    Args:
        events: The capture's trace events, in sequence order.

    Returns:
        One :class:`_EventRound` per ``model_call``, in order.
    """
    rounds: list[_EventRound] = []
    for event in events:
        if isinstance(event, ModelCallEvent):
            rounds.append(_EventRound(model_call=event))
        elif isinstance(event, ToolCallEvent | ToolResultEvent):
            if not rounds:
                # Defensive: a trace whose first tool call precedes any
                # model_call still gets a round to live in.
                rounds.append(_EventRound(model_call=None))
            if isinstance(event, ToolCallEvent):
                rounds[-1].calls.append(event)
            else:
                rounds[-1].results.append(event)
    return rounds


def _tool_rounds(events: list[Any]) -> list[list[ToolCallEvent]]:
    """Group tool calls into agent rounds, splitting at each ``model_call``.

    A capture's events are sequence-ordered: a ``model_call``, then the tool
    calls that model call emitted, then the next ``model_call``, and so on.
    Each ``model_call`` therefore opens a round. Rounds that emitted no tool
    call (typically the final, text-producing round) are dropped, so
    ``rounds[0]`` is always the first round that actually called something.

    Args:
        events: The capture's trace events, in sequence order.

    Returns:
        One list of :class:`ToolCallEvent` per tool-emitting round, in order.
        Empty when the capture called no tools at all.
    """
    return [r.calls for r in _event_rounds(events) if r.calls]


def _pair_round_results(
    calls: Sequence[tuple[str, str | None]],
    results: Sequence[ToolResultEvent],
) -> list[ToolResultEvent | None]:
    """Match each of one round's tool calls to the result event that answered it.

    Two passes, ids before names: every call carrying a ``call_id`` first
    claims the result recorded under that id, so a name-only fallback can
    never steal a result that provably belongs to another call. Each
    remaining call then takes the first still-unclaimed result with the same
    tool name — which is all a capture from an SDK (or a provider) that
    reported no call ids leaves to go on.

    Only results recorded *in the same round* are offered here: a later
    round's result answering an earlier round's call would fabricate a
    context the recorded agent never saw.

    Args:
        calls: ``(tool_name, call_id)`` for each of the round's calls, in
            order.
        results: The round's ``tool_result`` events, in order.

    Returns:
        One entry per call, positionally aligned: the result that answered
        it, or ``None`` when nothing in the round did.
    """
    paired: list[ToolResultEvent | None] = [None] * len(calls)
    claimed: set[int] = set()

    by_id: dict[str, int] = {}
    for pos, result in enumerate(results):
        if result.call_id is not None:
            by_id.setdefault(result.call_id, pos)
    for i, (_, call_id) in enumerate(calls):
        if call_id is None:
            continue
        match = by_id.get(call_id)
        if match is not None and match not in claimed:
            paired[i] = results[match]
            claimed.add(match)

    for i, (name, _) in enumerate(calls):
        if paired[i] is not None:
            continue
        for pos, result in enumerate(results):
            if pos not in claimed and result.name == name:
                paired[i] = result
                claimed.add(pos)
                break
    return paired


def _replay_fixtures(
    tool_rounds: Sequence[Sequence[ExpectedToolCall]],
    call_ids: Sequence[Sequence[str | None]],
    result_rounds: Sequence[Sequence[ToolResultEvent]],
) -> tuple[list[list[ToolResultFixture]], int | None]:
    """Recorded results for as many *leading* rounds as are fully covered.

    Coverage stops at the first round with any unpaired call: a teacher-forced
    replay feeds rounds ``0..k-1`` back verbatim before asking for round ``k``,
    so a round missing even one result cannot be replayed, and neither can
    anything after it.

    Args:
        tool_rounds: The ground-truth calls, one list per round.
        call_ids: The id each of those calls was recorded under (``None``
            where the capture recorded none), positionally aligned with
            ``tool_rounds``.
        result_rounds: The ``tool_result`` events of each round, aligned with
            ``tool_rounds``.

    Returns:
        ``(fixtures, unpaired)``. ``fixtures`` covers rounds ``0..m-1``
        positionally; ``unpaired`` is how many calls in round ``m`` had no
        recorded result, or ``None`` when every round is covered.
    """
    fixtures: list[list[ToolResultFixture]] = []
    for k, round_ in enumerate(tool_rounds):
        paired = _pair_round_results(
            [(call.tool_name, call_ids[k][i]) for i, call in enumerate(round_)],
            result_rounds[k] if k < len(result_rounds) else [],
        )
        unpaired = sum(1 for p in paired if p is None)
        if unpaired:
            return fixtures, unpaired
        # `unpaired == 0` means the filter drops nothing -- it is there to
        # narrow the Optional, not to shorten the list.
        fixtures.append(
            [
                ToolResultFixture(tool_name=call.tool_name, result=p.result, error=p.error)
                for call, p in zip(round_, paired, strict=True)
                if p is not None
            ],
        )
    return fixtures, None


def _requested_tool_rounds(
    events: list[Any],
    warnings: list[str],
) -> list[list[RequestedToolCall]] | None:
    """Group the calls the *model itself requested* into rounds, or ``None`` to fall back.

    Preferred over :func:`_tool_rounds` whenever the capture carries it. The
    executed ``tool_call`` events have already passed through the app — its
    filtering, re-ordering, retries, and (see
    :func:`_unwrap_recorded_arguments`) its own function signatures — so they
    are evidence of what the *app* did, while a golden case has to state what a
    *model* should produce. The requested list is exactly that.

    Rounds need no reconstruction here: each ``model_call`` carries its own
    response's requested calls, so each ``model_call`` simply *is* a round.
    Rounds that requested nothing (typically the final, text-producing one)
    are dropped, exactly as :func:`_tool_rounds` drops tool-less rounds.

    Args:
        events: The capture's trace events, in sequence order.
        warnings: Collector appended to when the capture is internally
            inconsistent (see the return contract).

    Returns:
        One list of :class:`RequestedToolCall` per requesting round, or
        ``None`` to mean "use the executed calls instead". ``None`` covers two
        cases: no ``model_call`` recorded the field at all (a capture written
        before the SDK had it — silent, this is the normal legacy path), and a
        capture where only *some* did.

        The second is a recording gap, not a version skew: the SDK stores
        ``None`` for an omitted ``record_model_call(requested_tool_calls=...)``
        argument, so a capture goes mixed as soon as one call in the run leaves
        it off — typically the final, text-only call, where the app has no
        tool calls to pass and omits the argument instead of passing ``[]``.
        Since ``None`` there means "not recorded" and not "requested nothing",
        the round cannot be read either way, so rather than promote half the
        trace against one yardstick and half against another the whole capture
        falls back to the executed calls, with the fix named in ``warnings``.
    """
    model_calls = [e for e in events if isinstance(e, ModelCallEvent)]
    with_field = [e for e in model_calls if e.requested_tool_calls is not None]
    if not with_field:
        return None
    if len(with_field) != len(model_calls):
        warnings.append(
            f"{len(with_field)} of {len(model_calls)} model_call event(s) recorded "
            "requested_tool_calls — pass it on every model call in the run, using [] for a "
            "round in which the model requested no tools (None means 'not recorded', not "
            "'nothing requested'). Ground truth fell back to the executed tool calls for the "
            "whole capture.",
        )
        return None
    return [list(e.requested_tool_calls or []) for e in model_calls if e.requested_tool_calls]


def _call_signature(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    """A comparable ``(name, canonical arguments)`` key for one tool call."""
    return name, json.dumps(arguments, sort_keys=True, default=str)


def _requested_vs_executed_warning(
    requested: list[list[RequestedToolCall]],
    executed: list[list[ToolCallEvent]],
) -> str | None:
    """Warning when what the model asked for differs from what the app ran, else ``None``.

    Compared as a multiset of ``(tool_name, arguments)`` per round, so a
    re-ordering *within* one round is not a difference (nothing downstream
    scores intra-round order) while a different tool, a different argument
    value, or a different round grouping is.

    The difference is reported, never resolved: the requested calls win by
    construction (they are the yardstick a candidate model can meet), and the
    warning exists so an operator can tell a legitimate app-side filter from a
    capture that lost calls.
    """
    if [
        Counter(_call_signature(c.name, dict(c.arguments)) for c in round_) for round_ in requested
    ] == [
        Counter(_call_signature(c.name, dict(c.arguments)) for c in round_) for round_ in executed
    ]:
        return None

    requested_names = [c.name for round_ in requested for c in round_]
    executed_names = [c.name for round_ in executed for c in round_]
    if Counter(requested_names) == Counter(executed_names):
        detail = (
            f"the same tool(s) ({_join_names(requested_names)}), but with different arguments "
            "or split across different rounds"
        )
    else:
        detail = (
            f"the model requested {_join_names(requested_names)}; the app executed "
            f"{_join_names(executed_names)}"
        )
    return (
        f"model-requested and executed tool calls disagree — {detail}. Ground truth follows the "
        "requested calls, which is what a candidate model can actually reproduce. A difference "
        "is normal when the app filters, retries, or rewrites the calls it runs; check it is "
        "not a gap in what was captured."
    )


def _join_names(names: list[str]) -> str:
    return ", ".join(names) if names else "no tools"


def _coerce_history_role(role: Any) -> Literal["system", "user", "assistant", "tool"] | None:
    """Map a recorded message role to a supported history role, or None to drop it."""
    if role == "model":
        return "assistant"
    if role == "system":
        return "system"
    if role == "user":
        return "user"
    if role == "assistant":
        return "assistant"
    if role == "tool":
        return "tool"
    return None


def _recover_inputs(
    events: list[Any],
    *,
    input_var: str,
    warnings: list[str],
) -> RecoveredInput:
    """Best-effort recovery of template-variable ``inputs`` (+ ``history``) from the trace.

    The capture only stores a one-way ``input_hash``, not the raw args, so we
    fall back to the first ``model_call`` input:

    * A dict is used directly as ``inputs`` (no history).
    * A messages list in any wrapped provider's shape is first translated by
      :func:`~evalshift_cli.captures.native_messages.normalize_messages`, then
      split: the last ``user`` message becomes the current-turn ``inputs``, and
      everything before it becomes ``history``. Unrecognised roles are dropped
      with an aggregated warning; content a text replay can't carry lands in
      ``unreplayable``. A replay carries one leading system prompt: the run of
      system (or ``developer``) messages opening the history merges into it, and
      one after the first turn is dropped with an aggregated warning.
    * A non-empty string is wrapped as ``{input_var: string}`` (no history).
    * Anything else yields empty ``inputs`` with a warning.
    """
    first_model_call = next((e for e in events if isinstance(e, ModelCallEvent)), None)
    if first_model_call is None:
        warnings.append(
            "no model_call event found — inputs left empty; edit the case before running.",
        )
        return RecoveredInput(inputs={})

    value = first_model_call.input

    recovered = recover_from_messages(value, input_var=input_var, warnings=warnings)
    if recovered is not None:
        return recovered

    if isinstance(value, dict):
        return RecoveredInput(inputs=dict(value))
    if isinstance(value, str) and value:
        return RecoveredInput(inputs={input_var: value})

    warnings.append(
        "could not recover structured inputs from the capture "
        f"(model_call input was {type(value).__name__}); "
        "inputs left empty; edit the case before running.",
    )
    return RecoveredInput(inputs={})


def recover_from_messages(
    raw: Any, *, input_var: str, warnings: list[str]
) -> RecoveredInput | None:
    """Recover ``inputs`` and ``history`` from a recorded messages list, as promotion does.

    The messages-list half of :func:`_recover_inputs`, public so ``captures/repair.py`` can
    rebuild a case's content key offline through the very same path.

    Args:
        raw: A recorded ``model_call.input``.
        input_var: The ``inputs`` key the current turn's text goes under.
        warnings: Non-fatal notes are appended here.

    Returns:
        The recovered input, its ``unreplayable`` lines included, or ``None`` when ``raw`` is
        not a messages list
        (see :func:`~evalshift_cli.captures.native_messages.normalize_messages`).
    """
    normalized = normalize_messages(raw)
    if normalized is None:
        return None
    warnings.extend(normalized.warnings)
    recovered = _recover_inputs_from_messages(
        normalized.messages, input_var=input_var, warnings=warnings
    )
    return replace(recovered, unreplayable=list(normalized.unreplayable))


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


def _assistant_with_call_ids(
    msg: dict[str, Any],
    position: int,
) -> tuple[dict[str, Any], list[str]]:
    """The assistant message with every tool call carrying an id, plus those ids.

    A call recorded without an id gets ``call_p<position>_<index>`` so the
    ``tool`` result that answers it can be keyed to *that* id in the golden
    file — the pairing has to be visible in the artefact, not re-derived at
    dispatch. An assistant turn without tool calls returns no ids: a plain
    reply closes the agent loop, so a later id-less result answers nothing.
    """
    raw_calls = msg.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return msg, []
    calls: list[Any] = []
    ids: list[str] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict):
            calls.append(call)
            continue
        if call.get("id") is None:
            call = {**call, "id": f"call_p{position}_{index}"}
        calls.append(call)
        ids.append(str(call["id"]))
    return {**msg, "tool_calls": calls}, ids


def _recover_inputs_from_messages(
    messages: list[dict[str, Any]],
    *,
    input_var: str,
    warnings: list[str],
) -> RecoveredInput:
    last_user_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        None,
    )
    if last_user_idx is None:
        warnings.append(
            "could not recover structured inputs from the capture "
            "(messages list had no user message); "
            "inputs left empty; edit the case before running.",
        )
        return RecoveredInput(inputs={})

    current = messages[last_user_idx]
    prefix = messages[:last_user_idx]

    history: list[ChatMessage] = []
    dropped_roles: set[str] = set()
    dropped_count = 0
    late_system = 0
    paired_by_order = 0
    unpaired_tool_results = 0
    # Call ids of the latest assistant turn that no tool result has answered
    # yet, in emission order. Providers that carry no ids on the wire (Gemini)
    # pair results with calls by that order, so an id-less result answers the
    # first still-open call rather than getting an id nothing can match.
    open_call_ids: list[str] = []
    for position, msg in enumerate(prefix):
        role = _coerce_history_role(msg.get("role"))
        if role is None:
            dropped_count += 1
            dropped_roles.add(str(msg.get("role")))
            continue
        if role == "system" and history:
            # A replay carries one system prompt, first (SuiteExample enforces it). The leading
            # run -- e.g. Responses `instructions=` folded in front of a `developer` input --
            # merges into it; a system message after the first turn has nowhere to go.
            if len(history) == 1 and history[0].role == "system":
                merged = f"{history[0].content}\n{_text_content(msg)}"
                history[0] = ChatMessage(role="system", content=merged)
            else:
                late_system += 1
            continue
        if role == "assistant":
            msg, open_call_ids = _assistant_with_call_ids(msg, position)
        elif role == "user":
            open_call_ids = []
        elif role == "tool":
            recorded_id = msg.get("tool_call_id")
            if recorded_id:
                if str(recorded_id) in open_call_ids:
                    open_call_ids.remove(str(recorded_id))
            elif open_call_ids:
                msg = {**msg, "tool_call_id": open_call_ids.pop(0)}
                paired_by_order += 1
            else:
                # Nothing in the prefix this result can answer. Synthesise an
                # id so the strict model validates; the gap is worth naming,
                # not a reason to delete the result from the replayed context.
                msg = {**msg, "tool_call_id": f"_pos{position}"}
                unpaired_tool_results += 1
        history.append(_history_message(msg, role))

    if dropped_count:
        roles = ", ".join(sorted(dropped_roles))
        warnings.append(
            f"dropped {dropped_count} history message(s) with unrecognised role(s) {{{roles}}}",
        )
    if late_system:
        warnings.append(
            f"dropped {late_system} system message(s) after the first turn — a replay carries "
            "one leading system prompt",
        )
    if paired_by_order:
        warnings.append(
            f"{paired_by_order} tool result(s) in history had no tool_call_id; "
            "each was paired, in order, with the preceding assistant turn's open tool call.",
        )
    if unpaired_tool_results:
        warnings.append(
            f"{unpaired_tool_results} tool result(s) in history had no tool_call_id and no "
            "preceding tool call to answer; synthetic ids were assigned. "
            "Record the provider call id for exact pairing.",
        )

    return RecoveredInput(
        inputs={input_var: _text_content(current)},
        history=history,
    )


def _context_loss_warning(
    events: list[Any],
    inputs: dict[str, Any],
    history: list[ChatMessage] | None,
) -> str | None:
    """Warning when recorded ``input_tokens`` imply far more context than recovered, else None.

    Returned (not appended) so conversation grouping can re-evaluate the check
    after reconstruction supplies a history prefix the per-capture pass
    couldn't see.
    """
    first_model_call = next((e for e in events if isinstance(e, ModelCallEvent)), None)
    if first_model_call is None or first_model_call.input_tokens == 0:
        return None

    input_tokens = first_model_call.input_tokens
    recovered_chars = len(json.dumps(inputs, default=str)) + sum(
        len(m.content) for m in history or []
    )
    est_tokens = max(1, recovered_chars // 4)

    if (
        input_tokens >= _CONTEXT_LOSS_RATIO * est_tokens
        and input_tokens - est_tokens > _CONTEXT_LOSS_MIN_GAP_TOKENS
    ):
        return (
            f"capture recorded {input_tokens} input tokens but only ~{est_tokens} were "
            "recovered — the system prompt / conversation history was probably not "
            "captured. Pass the full messages list to capture.model_call(input=...)."
        )
    return None


def _build_tags(suite: str, extra: tuple[str, ...]) -> list[str]:
    tags = [CAPTURED_TAG, suite]
    for tag in extra:
        if tag not in tags:
            tags.append(tag)
    return tags


def _recover_expected(events: list[Any], warnings: list[str]) -> dict[str, Any] | None:
    """Recover the turn's user-visible reply as text ground truth.

    A ``FinalOutputEvent`` is the explicit signal and always wins. The manual
    SDK API does not emit one (only the LangChain adapter does), so fall back
    to the last ``model_call`` whose ``output`` is non-empty text — on an agent
    turn that is the reply the user saw, after the tool round-trips. ``output``
    is typed ``Any``; anything that is not a ``str`` is left alone rather than
    stringified into fake ground truth.
    """
    finals = [e for e in events if isinstance(e, FinalOutputEvent)]
    if finals:
        return {"final_output": finals[-1].text}
    for event in reversed(events):
        if isinstance(event, ModelCallEvent) and isinstance(event.output, str):
            text = event.output.strip()
            if text:
                return {"final_output": text}
    warnings.append(
        "no text ground truth: this capture has no final_output event and no "
        "model_call with text output, so example.expected is null. Text "
        "evaluators compare the two candidate models to each other, never to "
        "what production actually said.",
    )
    return None


# ---------------------------------------------------------------------------
# Conversation grouping
# ---------------------------------------------------------------------------


def duplicate_turn_warnings(envelopes: list[CaptureEnvelope]) -> list[str]:
    """Warn when two captures claim the same conversation turn.

    A retried turn produces two captures with identical
    ``(conversation_id, turn_index)`` — typically one that errored and one
    that succeeded. Both get promoted, both appear in the suite, and the
    reconstruction order between them is arbitrary. Surface it so the
    operator can drop one.

    Args:
        envelopes: Every capture being synced, in any order.

    Returns:
        One warning per colliding ``(conversation_id, turn_index)`` pair.
    """
    seen: dict[tuple[str, int], list[str]] = {}
    for env in envelopes:
        if env.conversation_id is None or env.turn_index is None:
            continue
        seen.setdefault((env.conversation_id, env.turn_index), []).append(env.capture_id)

    out: list[str] = []
    for (conversation_id, turn_index), ids in sorted(seen.items()):
        if len(ids) < 2:
            continue
        joined = ", ".join(sorted(ids))
        out.append(
            f"conversation {conversation_id} turn {turn_index} is claimed by "
            f"{len(ids)} captures ({joined}) — usually a retried turn. Both will be promoted "
            "and their reconstruction order is arbitrary; drop the one you don't want.",
        )
    return out


def _turn_sort_key(record: CaptureRecord) -> tuple[int, int, str, str]:
    """Sort key ordering group members by turn_index (None last), then created_at, then id.

    The leading int flags a ``None`` turn_index so those members always sort
    after every explicitly-indexed member, regardless of their numeric value.
    """
    turn_index = record.envelope.turn_index
    has_index = 0 if turn_index is not None else 1
    return (
        has_index,
        turn_index if turn_index is not None else 0,
        record.envelope.created_at,
        record.envelope.capture_id,
    )


def _turn_current_text(built: BuiltExample, opts: PromoteOptions) -> str:
    """Best-effort text for the current turn, for reconstructing later turns' history."""
    value = built.example.inputs.get(opts.input_var)
    if isinstance(value, str):
        return value
    return json.dumps(built.example.inputs, default=str)


def _turn_assistant_text(record: CaptureRecord) -> str | None:
    """The text a reconstructed history entry should use for a prior turn's reply.

    Prefers the last :class:`FinalOutputEvent`; falls back to the last
    non-empty :class:`ModelCallEvent` output. ``None`` when neither exists.
    """
    events = record.envelope.trace.events
    finals = [e for e in events if isinstance(e, FinalOutputEvent)]
    if finals:
        return finals[-1].text
    model_calls = [e for e in events if isinstance(e, ModelCallEvent) and _nonempty_text(e.output)]
    if model_calls:
        return str(model_calls[-1].output)
    return None


def _earliest_prior_system_message(
    prior: list[tuple[CaptureRecord, BuiltExample]],
    *,
    warnings: list[str],
) -> ChatMessage | None:
    """The earliest prior turn's recovered system message, if any turn has one.

    A prior turn carries a system message when its own capture recorded a full
    messages list (or its history was itself seeded by an earlier turn). The
    earliest turn's wins; a single aggregated warning is emitted when later
    turns recorded a different system prompt.
    """
    found: ChatMessage | None = None
    found_capture_id: str | None = None
    mismatched: list[str] = []
    for prior_record, prior_built in prior:
        hist = prior_built.example.history
        if not hist or hist[0].role != "system":
            continue
        if found is None:
            found = hist[0]
            found_capture_id = prior_record.envelope.capture_id
        elif hist[0].content != found.content:
            mismatched.append(prior_record.envelope.capture_id)
    if mismatched:
        warnings.append(
            f"prior turns recorded different system prompts — using the earliest turn's "
            f"({found_capture_id!r}); differing: {', '.join(repr(c) for c in mismatched)}",
        )
    return found


def _reconstruct_history(
    prior: list[tuple[CaptureRecord, BuiltExample]],
    opts: PromoteOptions,
    *,
    warnings: list[str],
) -> list[ChatMessage]:
    """Build a conversation prefix from earlier turns in a group.

    The prefix is seeded with the earliest prior turn's recovered system
    message, when one exists (see :func:`_earliest_prior_system_message`).
    Each prior turn then contributes a ``user`` message (its recovered
    current-turn text) and an ``assistant`` message (its final/model-call
    output). A prior turn with no recoverable assistant text contributes only
    the ``user`` message, with a warning.
    """
    history: list[ChatMessage] = []
    system = _earliest_prior_system_message(prior, warnings=warnings)
    if system is not None:
        history.append(system)
    for prior_record, prior_built in prior:
        history.append(ChatMessage(role="user", content=_turn_current_text(prior_built, opts)))
        assistant_text = _turn_assistant_text(prior_record)
        if assistant_text is None:
            warnings.append(
                f"conversation {prior_record.envelope.conversation_id!r} turn "
                f"{prior_record.envelope.capture_id!r} has no recoverable assistant reply — "
                "reconstructed history is missing its assistant message for that turn",
            )
            continue
        history.append(ChatMessage(role="assistant", content=assistant_text))
    return history


def build_conversation_examples(
    records: list[CaptureRecord], opts: PromoteOptions, *, base: Path | None = None
) -> list[tuple[CaptureRecord, BuiltExample]]:
    """Build one :class:`SuiteExample` per capture, grouping multi-turn conversations.

    Records sharing a non-``None`` ``envelope.conversation_id`` are grouped and
    ordered by ``turn_index`` (missing indices sort last, tie-broken by
    ``created_at`` then ``capture_id``). Each turn's history is either kept
    verbatim (when the capture itself recorded a full messages list — the
    source of truth) or reconstructed from the group's prior turns' recovered
    text and replies.

    Records with ``conversation_id is None`` are built independently via
    :func:`build_example_from_capture`, unaffected by grouping.

    Args:
        records: Captures to build, in any order.
        opts: Promotion knobs, shared across every record.
        base: Capture base dir threaded to :func:`build_example_from_capture`
            for its toolset-sidecar existence check. ``None`` resolves via
            :func:`~evalshift_cli.captures.reader.capture_base`.

    Returns:
        ``(record, built)`` pairs, one per input record — order follows group
        turn order for conversation members, and input order for standalone
        records interleaved with the groups' first appearance.
    """
    groups: dict[str, list[CaptureRecord]] = {}
    order: list[str] = []
    standalone: list[CaptureRecord] = []

    for record in records:
        conv_id = record.envelope.conversation_id
        if conv_id is None:
            standalone.append(record)
            continue
        if conv_id not in groups:
            groups[conv_id] = []
            order.append(conv_id)
        groups[conv_id].append(record)

    results: list[tuple[CaptureRecord, BuiltExample]] = []

    for record in standalone:
        results.append((record, build_example_from_capture(record.envelope, opts, base=base)))

    for conv_id in order:
        members = sorted(groups[conv_id], key=_turn_sort_key)
        built_so_far: list[tuple[CaptureRecord, BuiltExample]] = []
        for ordinal, record in enumerate(members):
            built = build_example_from_capture(record.envelope, opts, base=base)
            example = built.example
            warnings = list(built.warnings)

            if example.history is None:
                history = _reconstruct_history(built_so_far, opts, warnings=warnings)
                # The per-capture context-loss check ran without this prefix;
                # re-evaluate it against what reconstruction just supplied.
                events = record.envelope.trace.events
                stale = _context_loss_warning(events, example.inputs, None)
                if stale is not None and stale in warnings:
                    warnings.remove(stale)
                fresh = _context_loss_warning(events, example.inputs, history)
                if fresh is not None:
                    warnings.append(fresh)
                example = example.model_copy(update={"history": history})

            turn_index = (
                record.envelope.turn_index if record.envelope.turn_index is not None else ordinal
            )
            example = example.model_copy(
                update={"conversation_id": conv_id, "turn_index": turn_index},
            )

            final_built = BuiltExample(
                example=example,
                warnings=warnings,
                blocked=built.blocked,
                blocked_reason=built.blocked_reason,
                promotion_source=built.promotion_source,
                cost_usd=built.cost_usd,
                cost_source=built.cost_source,
            )
            # A blocked turn is never promoted, so it must not seed later
            # turns' history either: a turn that died before the agent acted
            # contributes a user message with no reply, and a retried turn
            # would inject the same user text twice.
            if final_built.blocked is None:
                built_so_far.append((record, final_built))
            results.append((record, final_built))

    return results


# ---------------------------------------------------------------------------
# Disk side
# ---------------------------------------------------------------------------

GOLDEN_FILENAME = "golden.jsonl"


def _safe_segment(value: str) -> str:
    """Sanitise a user-supplied path segment (suite / case name).

    Strips path separators and parent refs so a crafted ``--suite`` or
    ``--as`` can't escape the suites directory.
    """
    cleaned = value.replace("/", "_").replace("\\", "_").replace("..", "_").strip()
    return cleaned or "_"


def promoted_suite_dir(suite: str, *, base: Path | None = None) -> Path:
    """Return the promoted-suite directory ``<base>/suites/<suite>`` for ``suite``.

    Path segments are sanitised the same way :func:`write_promoted_case` writes
    them, so callers (e.g. ``capture sync``) can locate a suite's dir without
    re-deriving the layout.
    """
    return suites_root(base) / _safe_segment(suite)


def write_promoted_case(
    case: PromotedCase,
    *,
    base: Path | None = None,
    force: bool = False,
) -> Path:
    """Write ``case`` to ``<base>/suites/<suite>/<name>.json``.

    Raises:
        FileExistsError: if the target exists and ``force`` is False.
    """
    suite_dir = promoted_suite_dir(case.suite, base=base)
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / f"{_safe_segment(case.name)}.json"
    if path.exists() and not force:
        raise FileExistsError(path)
    path.write_text(case.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def example_content_key(example: SuiteExample) -> str:
    """A stable key over the *replayed content* of ``example`` (inputs + history).

    Two examples sharing this key replay identically, so promoting both would
    duplicate the case, double ``n``, and inflate every downstream p-value and
    effect size. Deliberately **not** the capture's ``input_hash``: the SDK
    salts that with ``conversation_id`` (and derives it from the agent's bound
    arguments), so identical replay content routinely carries distinct hashes.
    """
    return json.dumps(
        {
            "inputs": example.inputs,
            "history": [m.model_dump() for m in example.history or []],
        },
        sort_keys=True,
        default=str,
    )


def iter_promoted_cases(suite_dir: Path) -> list[PromotedCase]:
    """Load every promoted case file in ``suite_dir`` (``golden.jsonl`` excluded).

    Raises:
        pydantic.ValidationError: if a case file is not a valid ``PromotedCase``.
    """
    cases: list[PromotedCase] = []
    for path in sorted(suite_dir.glob("*.json")):
        if path.name == GOLDEN_FILENAME:
            continue
        cases.append(PromotedCase.model_validate_json(path.read_text(encoding="utf-8")))
    return cases


def rebuild_golden_jsonl(suite_dir: Path) -> Path:
    """Regenerate ``<suite_dir>/golden.jsonl`` from every promoted case file.

    The JSONL is the run-facing index: one ``SuiteExample`` per line, sorted by
    id for determinism. Regenerated (never appended) so it can't drift from the
    canonical dir-form case files.
    """
    cases = iter_promoted_cases(suite_dir)
    cases.sort(key=lambda c: c.example.id)
    golden_path = suite_dir / GOLDEN_FILENAME
    golden_path.write_text(
        "".join(c.example.model_dump_json() + "\n" for c in cases),
        encoding="utf-8",
    )
    return golden_path


__all__ = [
    "GOLDEN_FILENAME",
    "BlockedReason",
    "BuiltExample",
    "PromoteOptions",
    "RecoveredInput",
    "build_conversation_examples",
    "build_example_from_capture",
    "duplicate_turn_warnings",
    "example_content_key",
    "iter_promoted_cases",
    "promoted_suite_dir",
    "rebuild_golden_jsonl",
    "recover_from_messages",
    "write_promoted_case",
]
