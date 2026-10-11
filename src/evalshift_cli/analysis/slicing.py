"""Group evaluation records into slices for analysis.

A *slice* is a named subset of suite examples: one per distinct example
tag, named after the tag. The implicit ``"all"`` slice always exists and
contains every example. There is nothing to configure — ``evalshift.yaml``
has no slice definitions; per-slice budgets are keyed by tag under
``migration_policy.slices``.

The output of :func:`build_slices` is a mapping from slice name to a
list of ``(prompt_id, evaluator_name, kind, example_id, source_score,
target_score, delta)`` tuples that the statistics layer can pair up and
test. ``kind`` is part of the grouping identity: one evaluator can score
several axes — ``tool_selection`` scores conformance against ground truth
*and* divergence from the source — and they are different measurements
against different baselines. Testing them in one comparison averages a
regression against a non-regression, which is the bug this whole change
exists to remove.

A pair an evaluator declined to score has no record, so it cannot be
sliced from ``scores.jsonl``. :func:`build_unmeasured` slices the run's
:class:`EvaluatorCoverage` instead, mapping each unmeasured pair to the
same slices its record would have landed in — which is what keeps the
statistics layer's not-applicable count and its ``insufficient`` verdict
alive now that the rows themselves are gone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from evalshift_cli.analysis.tolerance import within_tolerance
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.runner.models import EvaluatorCoverage
from evalshift_cli.suite.models import Suite, SuiteExample
from evalshift_cli.suite.tags import PROVENANCE_TAGS

ALL_SLICE: str = "all"

_Signature = tuple[tuple[str, str, str, str], ...]

#: The grouping identity of one comparison: prompt, evaluator name, and the
#: evaluator's axis slug. Named because :mod:`evalshift_cli.analysis.statistics`
#: has to group on exactly the same triple.
ComparisonKey = tuple[str, str, str]

#: ``{slice_name: {ComparisonKey: count}}`` — how many pairs an evaluator's
#: axis was handed in a slice and produced no row for.
UnmeasuredCounts = dict[str, dict[ComparisonKey, int]]


@dataclass(frozen=True, slots=True)
class SlicedScore:
    """One paired score belonging to a slice."""

    prompt_id: str
    evaluator_name: str
    example_id: str
    source_score: float
    target_score: float
    delta: float
    #: The record's evaluator-type slug. Defaulted because a record written
    #: before slugs existed carries none, and because it is only ever a
    #: grouping key — an empty one groups with its own kind, as it should.
    kind: str = ""
    #: The row's drop is inside its evaluator's own threshold — a semantic row
    #: above ``min_similarity`` — so the paired test reads it as no change,
    #: exactly as the policy budgets count it (see
    #: :mod:`evalshift_cli.analysis.tolerance`). ``delta`` stays raw for the
    #: aggregates. Defaulted for the same reason as ``kind``.
    within_tolerance: bool = False


@dataclass(frozen=True, slots=True)
class SliceAggregate:
    """Per-slice aggregate stats (computed without significance testing)."""

    name: str
    n: int
    source_avg_score: float
    target_avg_score: float
    delta_avg_score: float
    delta_min_score: float
    delta_max_score: float
    source_score_stdev: float
    target_score_stdev: float


def build_slices(
    *,
    records: list[EvalRecord],
    suite: Suite,
    coverage: Sequence[EvaluatorCoverage] = (),
) -> dict[str, list[SlicedScore]]:
    """Group evaluation records into slices keyed by slice name.

    Args:
        records: Every :class:`EvalRecord` from ``scores.jsonl``.
        suite: The loaded suite, used to look up an example's ``tags``.
        coverage: The run's per-evaluator coverage. Only its unmeasured
            pairs are read, and only to *seed* slice names: a slice whose
            every row was a non-measurement still has to exist here, or it
            would silently drop out of the analysis instead of being
            reported as unmeasured.

    Returns:
        ``{slice_name: [SlicedScore, ...]}`` always containing at
        least the implicit ``"all"`` slice.
    """
    by_id = {ex.id: ex for ex in suite.examples}
    out: dict[str, list[SlicedScore]] = defaultdict(list)

    for rec in records:
        # Skip records that errored at the evaluator layer — they don't
        # carry meaningful scores.
        if rec.error is not None:
            continue
        sliced = SlicedScore(
            prompt_id=rec.prompt_id,
            evaluator_name=rec.evaluator_name,
            example_id=rec.example_id,
            source_score=rec.source_score,
            target_score=rec.target_score,
            delta=rec.delta,
            kind=rec.kind,
            within_tolerance=within_tolerance(rec),
        )
        for slice_name in _slices_of(rec.example_id, by_id):
            out[slice_name].append(sliced)

    for entry in coverage:
        for pair in entry.unmeasured:
            for slice_name in _slices_of(pair.example_id, by_id):
                out.setdefault(slice_name, [])

    return dict(out)


def build_unmeasured(
    *,
    coverage: Sequence[EvaluatorCoverage],
    suite: Suite,
) -> UnmeasuredCounts:
    """Count, per slice and evaluator, the pairs that produced no row.

    Deleting a fabricated score also deletes the evidence that anything was
    attempted. This puts that evidence back on the same axes the statistics
    layer already works in, so ``"8 of 10 rows not applicable"`` still has a
    numerator *and* a denominator once the eight rows are gone.

    Args:
        coverage: The run's per-evaluator coverage, from ``state.json``.
        suite: The loaded suite, used to look up an example's ``tags``.

    Returns:
        ``{slice_name: {ComparisonKey: count}}``, empty when every
        evaluator measured everything it was handed. Coverage is booked per
        axis, so an evaluator whose conformance axis measured nothing and
        whose divergence axis measured everything reports exactly that,
        rather than one blended number over a doubled denominator.
    """
    by_id = {ex.id: ex for ex in suite.examples}
    out: UnmeasuredCounts = defaultdict(lambda: defaultdict(int))
    for entry in coverage:
        for pair in entry.unmeasured:
            key = (pair.prompt_id, entry.evaluator_name, entry.kind)
            for slice_name in _slices_of(pair.example_id, by_id):
                out[slice_name][key] += 1
    return {name: dict(counts) for name, counts in out.items()}


def _slices_of(example_id: str, by_id: dict[str, SuiteExample]) -> list[str]:
    """Every slice an example belongs to, ``"all"`` first, then one per tag.

    A tag *is* its slice's name. An example the suite no longer carries lands
    in ``"all"`` only — the same fate a record for it already had.
    """
    names = [ALL_SLICE]
    example = by_id.get(example_id)
    if example is not None:
        names.extend(example.tags)
    return names


def dedupe_slices(
    sliced: dict[str, list[SlicedScore]],
    *,
    preferred: frozenset[str] = frozenset(),
) -> tuple[dict[str, list[SlicedScore]], dict[str, str]]:
    """Drop slices that carry no information the surviving slices don't.

    Tags routinely overlap perfectly — a suite built entirely by ``capture
    promote`` tags every example ``["captured", <suite>]``, so both tags
    describe the same set of examples, and often the same set as the
    implicit ``"all"`` slice. Keeping the duplicates would restate the same
    numbers in the report *and* inflate the Benjamini-Hochberg correction in
    :func:`evalshift_cli.analysis.statistics.analyze`, which corrects across
    every comparison it is handed regardless of whether they are
    independent.

    Two slices are identical when they hold the same ``(prompt_id,
    evaluator_name, example_id)`` triples. Among identical slices:

    * ``"all"`` always survives — it is the overall scope.
    * Names in ``preferred`` always survive. Deduplication must never turn a
      budget the user wrote by hand into a no-op.
    * Otherwise a provenance tag (see :mod:`evalshift_cli.suite.tags`) loses to
      an ordinary one, and alphabetical order breaks what's left.

    Args:
        sliced: Slice mapping as returned by :func:`build_slices`.
        preferred: Slice names the user named explicitly in
            ``evalshift.yaml`` — in practice the keys of
            ``migration_policy.slices``.

    Returns:
        ``(kept, collapsed)`` where ``kept`` is ``sliced`` minus the
        redundant entries and ``collapsed`` maps each dropped slice name to
        the name that now stands in for it.
    """
    all_signature = _signature(sliced[ALL_SLICE]) if ALL_SLICE in sliced else None

    by_signature: dict[_Signature, list[str]] = defaultdict(list)
    collapsed: dict[str, str] = {}
    for name, scores in sliced.items():
        if name == ALL_SLICE:
            continue
        signature = _signature(scores)
        if signature == all_signature and name not in preferred:
            collapsed[name] = ALL_SLICE
            continue
        by_signature[signature].append(name)

    for names in by_signature.values():
        if len(names) < 2:
            continue
        survivors = [n for n in names if n in preferred]
        winner = min(survivors or names, key=lambda n: _rank(n, preferred))
        for name in names:
            if name != winner and name not in preferred:
                collapsed[name] = winner

    kept = {name: scores for name, scores in sliced.items() if name not in collapsed}
    return kept, collapsed


def _signature(scores: list[SlicedScore]) -> _Signature:
    """Identify a slice by its membership, preserving multiplicity."""
    return tuple(
        sorted((s.prompt_id, s.evaluator_name, s.kind, s.example_id) for s in scores),
    )


def _rank(name: str, preferred: frozenset[str]) -> tuple[bool, bool, str]:
    """Sort key picking the most meaningful name among identical slices."""
    return (name not in preferred, name in PROVENANCE_TAGS, name)


def aggregates(
    sliced: list[SlicedScore],
    name: str,
) -> SliceAggregate:
    """Compute simple aggregates over a list of paired scores.

    Every score here is a real measurement — a pair an evaluator declined
    to score has no row at all, so there is nothing to filter out.
    """
    n = len(sliced)
    if n == 0:
        return SliceAggregate(
            name=name,
            n=0,
            source_avg_score=0.0,
            target_avg_score=0.0,
            delta_avg_score=0.0,
            delta_min_score=0.0,
            delta_max_score=0.0,
            source_score_stdev=0.0,
            target_score_stdev=0.0,
        )
    source = [s.source_score for s in sliced]
    target = [s.target_score for s in sliced]
    deltas = [s.delta for s in sliced]
    return SliceAggregate(
        name=name,
        n=n,
        source_avg_score=_mean(source),
        target_avg_score=_mean(target),
        delta_avg_score=_mean(deltas),
        delta_min_score=min(deltas),
        delta_max_score=max(deltas),
        source_score_stdev=_std(source),
        target_score_stdev=_std(target),
    )


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return float((sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)


__all__ = [
    "ALL_SLICE",
    "ComparisonKey",
    "SliceAggregate",
    "SlicedScore",
    "UnmeasuredCounts",
    "aggregates",
    "build_slices",
    "build_unmeasured",
    "dedupe_slices",
]
