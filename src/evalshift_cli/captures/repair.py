"""Find promoted cases an earlier ``capture sync`` built from stringified message blocks.

Before :mod:`~evalshift_cli.captures.native_messages` existed, promotion passed Claude's
block-list ``content`` through ``str()``, so a promoted history held Python-repr strings such
as ``"[{'id': 'toolu_1', ..., 'type': 'tool_use'}]"``, and OpenAI / Responses / LangChain
captures could promote with empty ``inputs``. This module finds those cases and rebuilds them;
``capture sync --repair`` drives it. Pure: no I/O.
"""

from __future__ import annotations

import ast
import warnings
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
        # An invalid escape (a lone ``\d``) only warns -- DeprecationWarning on 3.11,
        # SyntaxWarning from 3.12 -- and str() of a block list never writes one, so treat it as
        # an error: the parser re-raises it as SyntaxError, which means "not stringified".
        with warnings.catch_warnings():
            warnings.simplefilter("error")
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

    When the stored and the rebuilt ``inputs`` each have exactly one key, the stored key is
    kept with the rebuilt value: a messages-list capture always rebuilds under this sync's
    ``--input-var``, and the case may have been promoted (or hand-edited) under another name.
    Any other shape -- stored ``{}``, several keys on either side -- takes ``rebuilt.inputs``.

    Args:
        case: The stored case.
        rebuilt: The example promotion builds from the case's capture today.
        promoted_at: Timestamp for the rewritten case.

    Returns:
        The rewritten case, or ``None`` when its ``inputs`` (after keeping the stored key)
        and ``history`` already match.
    """
    current = case.example
    inputs = rebuilt.inputs
    if len(current.inputs) == 1 and len(rebuilt.inputs) == 1:
        (stored_key,) = current.inputs
        (rebuilt_value,) = rebuilt.inputs.values()
        inputs = {stored_key: rebuilt_value}
    if inputs == current.inputs and rebuilt.history == current.history:
        return None
    payload = current.model_dump()
    payload["inputs"] = inputs
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
