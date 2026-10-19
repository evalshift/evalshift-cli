"""``plan_suite``: which captures enter a bounded suite and which cases leave (spec §5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from evalshift_cli.captures.selection import CandidateGroup, KeptCase, SuitePlan, plan_suite
from evalshift_cli.captures.staleness import StaleReason

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _kept(
    name: str,
    *,
    days: int = 0,
    stale: StaleReason | None = None,
    reviewed: bool = False,
    route: str = "a",
    key: str | None = None,
) -> KeptCase:
    return KeptCase(
        name=name,
        instant=T0 + timedelta(days=days),
        stale=stale,
        reviewed=reviewed,
        route=route,
        content_key=key or f"k-{name}",
    )


def _group(
    key: str, *, route: str = "a", size: int = 1, content: set[str] | None = None
) -> CandidateGroup:
    ids = tuple(f"{key}-t{i}" for i in range(size)) if size > 1 else (key,)
    return CandidateGroup(
        key=key,
        capture_ids=ids,
        route=route,
        content_keys=frozenset(content or {f"k-{i}" for i in ids}),
    )


def _added(plan: SuitePlan) -> list[str]:
    return [g.key for g in plan.add]


def test_fills_free_slots_up_to_the_cap() -> None:
    plan = plan_suite([], [_group(f"c{i}") for i in range(5)], max_cases=3)
    assert plan.added_cases == 3
    assert plan.retire == ()


def test_the_plan_is_deterministic_whatever_the_input_order() -> None:
    groups = [_group(f"c{i}", route="ab"[i % 2]) for i in range(7)]
    assert plan_suite([], groups, max_cases=4) == plan_suite([], groups[::-1], max_cases=4)


def test_the_least_represented_route_fills_first() -> None:
    kept = [_kept(f"f{i}") for i in range(3)]
    groups = [_group(f"a{i}") for i in range(5)] + [_group(f"b{i}", route="b") for i in range(2)]
    plan = plan_suite(kept, groups, max_cases=6)
    assert [g.route for g in plan.add] == ["b", "b", "a"]


def test_a_group_is_added_whole_or_not_at_all() -> None:
    plan = plan_suite([], [_group("conv", size=3), _group("solo", route="b")], max_cases=2)
    assert _added(plan) == ["solo"]
    assert plan.added_cases == 1


def test_stale_cases_stay_when_nothing_replaces_them() -> None:
    kept = [_kept(f"s{i}", stale="age") for i in range(3)]
    assert plan_suite(kept, [], max_cases=3) == SuitePlan()


def test_stale_cases_leave_only_to_make_room_oldest_first() -> None:
    kept = [
        _kept("s1", days=0, stale="age"),
        _kept("s2", days=1, stale="age"),
        _kept("s3", days=2, stale="age"),
    ]
    plan = plan_suite(kept, [_group("new")], max_cases=3)
    assert _added(plan) == ["new"]
    assert plan.retire == (("s1", "age"),)


def test_another_model_leaves_before_old_age() -> None:
    kept = [_kept("old", days=0, stale="age"), _kept("mod", days=5, stale="model")]
    plan = plan_suite(kept, [_group("new")], max_cases=2)
    assert plan.retire == (("mod", "model"),)


def test_a_stale_twin_is_replaced_even_under_the_cap() -> None:
    kept = [_kept("twin", stale="age", key="k-q")]
    plan = plan_suite(kept, [_group("new", content={"k-q"})], max_cases=10)
    assert _added(plan) == ["new"]
    assert plan.retire == (("twin", "age"),)


def test_reviewed_cases_are_never_retired() -> None:
    kept = [_kept("r1", stale="age", reviewed=True), _kept("r2", stale="model", reviewed=True)]
    plan = plan_suite(kept, [_group(f"c{i}") for i in range(3)], max_cases=1)
    assert plan.add == ()
    assert plan.retire == ()
    assert plan.reviewed_over_cap == 1


def test_an_over_cap_suite_is_trimmed_from_its_largest_route_oldest_first() -> None:
    kept = [
        _kept("a1", days=0),
        _kept("a2", days=1),
        _kept("a3", days=2),
        _kept("b1", days=0, route="b"),
    ]
    plan = plan_suite(kept, [], max_cases=2)
    assert plan.retire == (("a1", "over_cap"), ("a2", "over_cap"))


def test_planning_twice_changes_nothing() -> None:
    kept = [_kept(f"f{i}") for i in range(3)]
    assert plan_suite(kept, [_group("c1"), _group("c2")], max_cases=3) == SuitePlan()


@pytest.mark.parametrize("max_cases", [1, 2, 3, 5, 8])
def test_a_suite_never_ends_over_the_cap_unless_reviewed_cases_force_it(max_cases: int) -> None:
    kept = [
        _kept("r", reviewed=True),
        _kept("f1"),
        _kept("f2", route="b"),
        _kept("s1", stale="age"),
        _kept("s2", stale="model", route="b"),
    ]
    groups = [_group("c1"), _group("c2", route="c"), _group("conv", size=2, route="b")]
    plan = plan_suite(kept, groups, max_cases=max_cases)
    names = [name for name, _ in plan.retire]
    assert len(names) == len(set(names))
    assert "r" not in names
    assert len(kept) - len(plan.retire) + plan.added_cases <= max_cases
