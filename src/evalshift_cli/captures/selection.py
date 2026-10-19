"""Which captures enter a bounded suite and which promoted cases leave it.

Pure: no I/O, no clock. ``capture sync`` builds the inputs from disk and applies the plan.
See ``docs/superpowers/specs/2026-10-16-bounded-capture-suites-design.md`` §5.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from evalshift_cli.captures.staleness import StaleReason

RetireReason = Literal["age", "model", "over_cap"]
"""Why a case left: too old, recorded on another model, or trimmed to ``max_cases``."""


@dataclass(frozen=True, slots=True)
class KeptCase:
    """A promoted case already in the suite, as the planner sees it.

    Attributes:
        name: The case name (its file stem) -- what a retirement deletes.
        instant: When its capture was recorded (``staleness.case_instant``), if known.
        stale: Why it is stale, or ``None`` when fresh.
        reviewed: A human marked one of its expectations ``reviewed``: never retired.
        route: Its tool route (``staleness.examples_route``).
        content_key: ``promote.example_content_key`` of its example.
    """

    name: str
    instant: datetime | None
    stale: StaleReason | None
    reviewed: bool
    route: str
    content_key: str


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    """Eligible captures that enter a suite together: one conversation, or one lone capture.

    Attributes:
        key: Stable identity used for the deterministic order.
        capture_ids: Its turns' case names (the capture ids), in turn order.
        route: The union route of its turns.
        content_keys: Each turn's ``example_content_key``, in turn order.
    """

    key: str
    capture_ids: tuple[str, ...]
    route: str
    content_keys: tuple[str, ...]

    @property
    def size(self) -> int:
        """How many cases the group adds."""
        return len(self.capture_ids)

    def prefix(self, turns: int) -> CandidateGroup:
        """The group's first ``turns`` turns: a conversation cut short, its history intact."""
        return CandidateGroup(
            key=self.key,
            capture_ids=self.capture_ids[:turns],
            route=self.route,
            content_keys=self.content_keys[:turns],
        )


@dataclass(frozen=True, slots=True)
class SuitePlan:
    """What one sync does to one suite.

    Attributes:
        add: Groups to promote, in the order they were picked.
        retire: ``(case name, reason)`` for every case to delete.
        reviewed_over_cap: Reviewed cases beyond ``max_cases``, kept anyway.
    """

    add: tuple[CandidateGroup, ...] = ()
    retire: tuple[tuple[str, RetireReason], ...] = ()
    reviewed_over_cap: int = 0

    @property
    def added_cases(self) -> int:
        """How many cases ``add`` promotes."""
        return sum(group.size for group in self.add)


def _hash_order(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _when(case: KeptCase) -> float:
    """Sort key, oldest first; an unknown instant counts as oldest of all."""
    return case.instant.timestamp() if case.instant is not None else float("-inf")


def plan_suite(
    kept: Sequence[KeptCase],
    candidates: Sequence[CandidateGroup],
    *,
    max_cases: int,
) -> SuitePlan:
    """Plan one suite: fill free slots route-balanced; retire stale cases only to make room.

    1. Reviewed cases are pinned; every other case is fresh or stale.
    2. Free slots (``max_cases`` minus pinned and fresh) are filled one group at a time from
       the route least represented so far (ties by route name), each route's groups in
       ``sha256(key)`` order. A group is added whole, or skipped when it does not fit. Once no
       whole group fits, the slots left take the leading turns of the next group in that
       order, so a long conversation cannot leave a suite nearly empty.
    3. A stale case whose content an added turn replays is retired outright, so no duplicate
       content survives. Other stale cases leave only while the suite is over ``max_cases``:
       recorded-on-another-model first, then oldest.
    4. Still over (a legacy or over-cap suite): fresh cases are trimmed from the most
       represented route, oldest first. Reviewed cases never leave.

    Same inputs, same plan, on any machine.
    """
    pinned = [c for c in kept if c.reviewed]
    fresh = [c for c in kept if not c.reviewed and c.stale is None]
    stale = [c for c in kept if not c.reviewed and c.stale is not None]

    counts: Counter[str] = Counter(c.route for c in (*pinned, *fresh))
    queues: dict[str, list[CandidateGroup]] = {}
    for group in sorted(candidates, key=lambda g: _hash_order(g.key)):
        queues.setdefault(group.route, []).append(group)

    slots = max(0, max_cases - len(pinned) - len(fresh))
    added: list[CandidateGroup] = []
    while slots > 0:
        fitting = [r for r, queue in queues.items() if any(g.size <= slots for g in queue)]
        if not fitting:
            break
        route = min(fitting, key=lambda r: (counts[r], r))
        queue = queues[route]
        group = queue.pop(next(i for i, g in enumerate(queue) if g.size <= slots))
        added.append(group)
        counts[route] += group.size
        slots -= group.size
    waiting = [r for r, queue in queues.items() if queue]
    if slots > 0 and waiting:
        route = min(waiting, key=lambda r: (counts[r], r))
        added.append(queues[route].pop(0).prefix(slots))
        counts[route] += slots

    retire: list[tuple[str, RetireReason]] = []
    replayed = frozenset(key for group in added for key in group.content_keys)
    remaining: list[KeptCase] = []
    for case in stale:
        if case.stale is not None and case.content_key in replayed:
            retire.append((case.name, case.stale))
        else:
            remaining.append(case)

    excess = len(pinned) + len(fresh) + sum(g.size for g in added) + len(remaining) - max_cases
    for case in sorted(remaining, key=lambda c: (c.stale != "model", _when(c), c.name)):
        if excess <= 0:
            break
        if case.stale is not None:
            retire.append((case.name, case.stale))
            excess -= 1

    pool = sorted(fresh, key=lambda c: (_when(c), c.name))
    while excess > 0 and pool:
        route = min({c.route for c in pool}, key=lambda r: (-counts[r], r))
        victim = next(c for c in pool if c.route == route)
        pool.remove(victim)
        counts[route] -= 1
        retire.append((victim.name, "over_cap"))
        excess -= 1

    return SuitePlan(
        add=tuple(added),
        retire=tuple(retire),
        reviewed_over_cap=max(0, len(pinned) - max_cases),
    )
