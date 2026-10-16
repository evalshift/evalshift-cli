"""What makes a promoted case stale -- shared by ``capture sync`` and the freshness checks.

See ``docs/superpowers/specs/2026-10-16-bounded-capture-suites-design.md`` §3.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.models.matching import same_model
from evalshift_cli.suite.models import SuiteExample

StaleReason = Literal["age", "model"]
"""Why a case is stale: older than ``captures.max_age``, or recorded on another model."""

#: The route of a case that expects no tool call.
NO_TOOLS_ROUTE = "∅"


def parse_instant(text: str) -> datetime | None:
    """An ISO 8601 timestamp as an aware UTC datetime; naive means UTC; ``None`` if unparseable."""
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def case_instant(case: PromotedCase) -> datetime | None:
    """When ``case``'s capture was recorded.

    ``captured_at``, else ``promoted_at`` -- an upper bound on the capture time, so a legacy
    case looks younger than it is and is retired later, never earlier.
    """
    return parse_instant(case.captured_at) or parse_instant(case.promoted_at)


def is_reviewed(example: SuiteExample) -> bool:
    """Whether a human marked any of ``example``'s tool expectations ``provenance: reviewed``."""
    calls = list(example.expected_tools or [])
    for round_calls in example.expected_tool_rounds or []:
        calls.extend(round_calls)
    return any(call.provenance == "reviewed" for call in calls)


def examples_route(examples: Iterable[SuiteExample]) -> str:
    """The sorted, comma-joined round-1 tool names across ``examples``; ``∅`` for none."""
    names = sorted({call.tool_name for e in examples for call in e.expected_tools or []})
    return ",".join(names) if names else NO_TOOLS_ROUTE


@dataclass(frozen=True, slots=True)
class StalenessRules:
    """The limits one sync (or one freshness check) measures cases against.

    Attributes:
        now: The single instant ages are measured from.
        max_age: ``captures.max_age``; ``None`` turns age staleness off.
        source_model: The model a case must have been recorded on; ``None`` turns model
            staleness off (no ``defaults.source_model``, or the guardrail tripped).
    """

    now: datetime
    max_age: timedelta | None = None
    source_model: str | None = None

    def stale_reason(self, instant: datetime | None, model: str) -> StaleReason | None:
        """``model`` before ``age``; ``None`` when fresh or when the fact is unknown."""
        if self.source_model and model and not same_model(model, self.source_model):
            return "model"
        if self.max_age is not None and instant is not None and instant < self.now - self.max_age:
            return "age"
        return None
