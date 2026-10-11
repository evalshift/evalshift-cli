"""Tests for slicing + statistics."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
from scipy import stats

from evalshift_cli.analysis.slicing import (
    ALL_SLICE,
    SliceAggregate,
    SlicedScore,
    aggregates,
    build_slices,
    build_unmeasured,
    dedupe_slices,
)
from evalshift_cli.analysis.statistics import (
    ADVISORY_NOTE_PREFIX,
    AXIS_NOTE_PREFIX,
    UNMEASURED_NOTE_PREFIX,
    WITHIN_TOLERANCE_REASON,
    ComparisonResult,
    _benjamini_hochberg,
    _classify_severity,
    _cohens_d_paired,
    analyze,
)
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION
from evalshift_cli.runner.models import EvaluatorCoverage, UnmeasuredPair
from evalshift_cli.suite.models import Suite
from tests.unit.suite_examples import suite_example

# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def _record(
    *,
    prompt: str = "p",
    evaluator: str = "e",
    example: str,
    delta: float,
    error: str | None = None,
) -> EvalRecord:
    src = max(0.0, -delta) if delta < 0 else 0.0
    tgt = src + delta
    src = max(0.0, min(1.0, src))
    tgt = max(0.0, min(1.0, tgt))
    return EvalRecord(
        run_id="r1",
        prompt_id=prompt,
        example_id=example,
        evaluator_name=evaluator,
        source_score=src,
        target_score=tgt,
        delta=tgt - src,
        error=error,
    )


class TestBuildSlices:
    def test_all_slice_always_present(self) -> None:
        suite = Suite(
            examples=[suite_example(id="ex1", tags=["formal"])],
        )
        records = [_record(example="ex1", delta=0.1)]
        out = build_slices(records=records, suite=suite)
        assert ALL_SLICE in out
        assert out[ALL_SLICE][0].example_id == "ex1"

    def test_examples_appear_in_each_tag_slice(self) -> None:
        suite = Suite(
            examples=[
                suite_example(id="a", tags=["formal", "english"]),
                suite_example(id="b", tags=["casual"]),
            ],
        )
        records = [
            _record(example="a", delta=0.0),
            _record(example="b", delta=0.0),
        ]
        out = build_slices(records=records, suite=suite)
        assert {s.example_id for s in out["formal"]} == {"a"}
        assert {s.example_id for s in out["english"]} == {"a"}
        assert {s.example_id for s in out["casual"]} == {"b"}
        assert {s.example_id for s in out[ALL_SLICE]} == {"a", "b"}

    def test_errored_records_skipped(self) -> None:
        suite = Suite(examples=[suite_example(id="a", tags=["t"])])
        records = [_record(example="a", delta=0.0, error="boom")]
        out = build_slices(records=records, suite=suite)
        assert out == {}

    def test_unknown_example_id_only_lands_in_all(self) -> None:
        suite = Suite(examples=[suite_example(id="a", tags=["t"])])
        records = [_record(example="ghost", delta=0.0)]
        out = build_slices(records=records, suite=suite)
        assert "t" not in out
        assert len(out[ALL_SLICE]) == 1

    def test_coverage_seeds_a_slice_whose_every_row_is_absent(self) -> None:
        """A slice with no rows left must still exist, or it drops out silently."""
        suite = Suite(examples=[suite_example(id="a", tags=["t"])])
        out = build_slices(records=[], suite=suite, coverage=[_coverage("a")])
        assert out == {ALL_SLICE: [], "t": []}


def _coverage(*example_ids: str, evaluator: str = "e", prompt: str = "p") -> EvaluatorCoverage:
    """Coverage for one evaluator that scored nothing on ``example_ids``."""
    return EvaluatorCoverage(
        evaluator_name=evaluator,
        kind="semantic",
        attempted=len(example_ids),
        recorded=0,
        unmeasured=[UnmeasuredPair(prompt_id=prompt, example_id=e) for e in example_ids],
    )


class TestBuildUnmeasured:
    """Absent rows are re-sliced from coverage, on the same axes real rows use."""

    def test_counts_land_in_every_slice_the_example_belongs_to(self) -> None:
        suite = Suite(examples=[suite_example(id="a", tags=["formal", "english"])])
        out = build_unmeasured(coverage=[_coverage("a")], suite=suite)
        assert out == {
            ALL_SLICE: {("p", "e", "semantic"): 1},
            "formal": {("p", "e", "semantic"): 1},
            "english": {("p", "e", "semantic"): 1},
        }

    def test_counts_accumulate_per_prompt_and_evaluator(self) -> None:
        suite = Suite(examples=[suite_example(id=x, tags=["t"]) for x in ("a", "b")])
        out = build_unmeasured(
            coverage=[_coverage("a", "b"), _coverage("a", evaluator="other")],
            suite=suite,
        )
        assert out[ALL_SLICE] == {("p", "e", "semantic"): 2, ("p", "other", "semantic"): 1}

    def test_an_unknown_example_only_lands_in_all(self) -> None:
        suite = Suite(examples=[suite_example(id="a", tags=["t"])])
        out = build_unmeasured(coverage=[_coverage("ghost")], suite=suite)
        assert out == {ALL_SLICE: {("p", "e", "semantic"): 1}}

    def test_a_fully_measured_run_produces_nothing(self) -> None:
        suite = Suite(examples=[suite_example(id="a", tags=["t"])])
        measured = EvaluatorCoverage(evaluator_name="e", attempted=1, recorded=1)
        assert build_unmeasured(coverage=[measured], suite=suite) == {}


class TestDedupeSlices:
    def _sliced(self, **by_name: list[str]) -> dict[str, list[SlicedScore]]:
        """Build a slice mapping from ``{slice_name: [example_id, ...]}``."""
        return {
            name: [
                SlicedScore(
                    prompt_id="p",
                    evaluator_name="e",
                    example_id=ex,
                    source_score=0.5,
                    target_score=0.5,
                    delta=0.0,
                )
                for ex in ids
            ]
            for name, ids in by_name.items()
        }

    def test_identical_partial_slices_collapse(self) -> None:
        sliced = self._sliced(all=["a", "b"], alpha=["a"], beta=["a"])
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE, "alpha"}
        assert collapsed == {"beta": "alpha"}

    def test_slice_identical_to_all_is_dropped(self) -> None:
        sliced = self._sliced(all=["a", "b"], everything=["a", "b"])
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE}
        assert collapsed == {"everything": ALL_SLICE}

    def test_all_never_dropped(self) -> None:
        sliced = self._sliced(all=["a"])
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE}
        assert collapsed == {}

    def test_preferred_survives_identity_with_all(self) -> None:
        sliced = self._sliced(all=["a", "b"], budgeted=["a", "b"])
        kept, collapsed = dedupe_slices(sliced, preferred=frozenset({"budgeted"}))
        assert set(kept) == {ALL_SLICE, "budgeted"}
        assert collapsed == {}

    def test_preferred_wins_over_non_preferred_peer(self) -> None:
        sliced = self._sliced(all=["a", "b"], aaa=["a"], zzz=["a"])
        kept, collapsed = dedupe_slices(sliced, preferred=frozenset({"zzz"}))
        assert set(kept) == {ALL_SLICE, "zzz"}
        assert collapsed == {"aaa": "zzz"}

    def test_two_preferred_peers_both_survive(self) -> None:
        sliced = self._sliced(all=["a", "b"], one=["a"], two=["a"], three=["a"])
        kept, collapsed = dedupe_slices(sliced, preferred=frozenset({"one", "two"}))
        assert set(kept) == {ALL_SLICE, "one", "two"}
        assert collapsed == {"three": "one"}

    def test_provenance_tag_loses_to_ordinary_peer(self) -> None:
        sliced = self._sliced(all=["a", "b"], captured=["a"], zebra=["a"])
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE, "zebra"}
        assert collapsed == {"captured": "zebra"}

    def test_alphabetical_breaks_remaining_ties(self) -> None:
        sliced = self._sliced(all=["a", "b"], zebra=["a"], antelope=["a"])
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE, "antelope"}
        assert collapsed == {"zebra": "antelope"}

    def test_three_way_group_keeps_one(self) -> None:
        sliced = self._sliced(all=["a", "b"], x=["a"], y=["a"], z=["a"])
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE, "x"}
        assert collapsed == {"y": "x", "z": "x"}

    def test_distinct_slices_untouched(self) -> None:
        sliced = self._sliced(all=["a", "b"], left=["a"], right=["b"])
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE, "left", "right"}
        assert collapsed == {}

    def test_membership_not_size_decides_identity(self) -> None:
        sliced = self._sliced(all=["a", "b"], left=["a"], right=["b"])
        kept, _ = dedupe_slices(sliced)
        assert len(kept["left"]) == len(kept["right"]) == 1
        assert kept["left"][0].example_id == "a"

    def test_differing_evaluator_makes_slices_distinct(self) -> None:
        sliced = self._sliced(all=["a"], one=["a"])
        sliced["two"] = [
            SlicedScore(
                prompt_id="p",
                evaluator_name="other",
                example_id="a",
                source_score=0.5,
                target_score=0.5,
                delta=0.0,
            ),
        ]
        kept, collapsed = dedupe_slices(sliced)
        assert "two" in kept
        assert collapsed == {"one": ALL_SLICE}

    def test_capture_promoted_suite_collapses_to_all(self) -> None:
        """The real-world case: every example carries ``captured`` + suite tag."""
        ids = [f"ex{i}" for i in range(5)]
        sliced = self._sliced(all=ids, captured=ids, project_insights=ids)
        kept, collapsed = dedupe_slices(sliced)
        assert set(kept) == {ALL_SLICE}
        assert collapsed == {"captured": ALL_SLICE, "project_insights": ALL_SLICE}

    def test_dedupe_shrinks_the_correction_family(self) -> None:
        ids = [f"ex{i}" for i in range(6)]
        sliced = self._sliced(all=ids, captured=ids, suite=ids)
        before = analyze(sliced_by_slice=sliced)
        kept, _ = dedupe_slices(sliced)
        after = analyze(sliced_by_slice=kept)
        assert len(before) == 3
        assert len(after) == 1
        # Uniformly duplicated families cancel exactly, so the adjusted
        # p-value is unchanged here — the win is one finding, not three.
        assert after[0].p_value_corrected == pytest.approx(before[0].p_value_corrected)


class TestAggregates:
    def test_empty_slice(self) -> None:
        agg = aggregates([], "all")
        assert agg.n == 0
        assert agg.delta_avg_score == 0.0

    def test_simple_aggregate(self) -> None:
        ss = [
            SlicedScore("p", "e", "1", 1.0, 0.5, -0.5),
            SlicedScore("p", "e", "2", 1.0, 1.0, 0.0),
            SlicedScore("p", "e", "3", 0.5, 0.0, -0.5),
        ]
        agg = aggregates(ss, "all")
        assert agg.n == 3
        assert agg.delta_avg_score == pytest.approx((-0.5 + 0.0 - 0.5) / 3)
        assert agg.delta_min_score == -0.5
        assert agg.delta_max_score == 0.0


# ---------------------------------------------------------------------------
# Statistics — helpers
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_empty(self) -> None:
        assert _benjamini_hochberg([]) == []

    def test_all_significant(self) -> None:
        # p = [0.001, 0.002, 0.003, 0.004]; BH adjusted should still be < 0.05
        adj = _benjamini_hochberg([0.001, 0.002, 0.003, 0.004])
        assert all(a <= 0.05 for a in adj)

    def test_input_order_preserved(self) -> None:
        # Even though BH sorts internally, output must be in input order.
        raw = [0.5, 0.01, 0.04]
        adj = _benjamini_hochberg(raw)
        # Smallest raw p (0.01) should produce the smallest adjusted.
        assert adj[1] < adj[0]
        assert adj[1] < adj[2]

    def test_clamped_to_one(self) -> None:
        # A high p should not exceed 1 after correction.
        adj = _benjamini_hochberg([0.99] * 5)
        assert all(0.0 <= a <= 1.0 for a in adj)

    def test_matches_known_result(self) -> None:
        # statsmodels.stats.multitest.multipletests([0.01, 0.04, 0.20], method='fdr_bh')
        # produces adjusted p-values [0.03, 0.06, 0.20] — confirm we match.
        adj = _benjamini_hochberg([0.01, 0.04, 0.20])
        assert adj[0] == pytest.approx(0.03, abs=1e-6)
        assert adj[1] == pytest.approx(0.06, abs=1e-6)
        assert adj[2] == pytest.approx(0.20, abs=1e-6)

    def test_duplicated_p_values_bias_the_family_downward(self) -> None:
        """Why identical slices must be collapsed before correction.

        A duplicate raises both the family size and the rank its copies
        reach. Since ``(n + k) / (r + k) < n / r``, every adjusted p-value
        comes out *smaller* — anti-conservative, so a duplicated regression
        can be classified more severe than the evidence supports.
        """
        distinct = _benjamini_hochberg([0.001, 0.04])
        duplicated = _benjamini_hochberg([0.001, 0.001, 0.04])
        assert distinct[0] == pytest.approx(0.002)
        assert duplicated[0] == pytest.approx(0.0015)
        assert duplicated[0] < distinct[0]

    def test_uniformly_duplicated_family_is_unchanged(self) -> None:
        # Every test duplicated the same number of times: the ratios cancel.
        once = _benjamini_hochberg([0.001, 0.04])
        thrice = _benjamini_hochberg([0.001] * 3 + [0.04] * 3)
        assert thrice[0] == pytest.approx(once[0])
        assert thrice[-1] == pytest.approx(once[-1])


class TestCohensD:
    def test_zero_variance(self) -> None:
        assert _cohens_d_paired(np.array([0.0, 0.0, 0.0])) == 0.0

    def test_positive_effect(self) -> None:
        d = _cohens_d_paired(np.array([0.5, 0.6, 0.4, 0.5]))
        assert d > 0


class TestClassifySeverity:
    def test_insufficient(self) -> None:
        assert _classify_severity(corrected_p=0.001, d=-1.0, mean_delta=-0.5, n=3) == "insufficient"

    def test_no_significance(self) -> None:
        assert _classify_severity(corrected_p=0.5, d=-2.0, mean_delta=-0.5, n=20) == "none"

    def test_improved(self) -> None:
        assert _classify_severity(corrected_p=0.001, d=0.5, mean_delta=0.4, n=20) == "improved"

    def test_critical(self) -> None:
        assert _classify_severity(corrected_p=0.001, d=-1.0, mean_delta=-0.4, n=50) == "critical"

    def test_high(self) -> None:
        assert _classify_severity(corrected_p=0.02, d=-0.6, mean_delta=-0.2, n=30) == "high"

    def test_medium(self) -> None:
        assert _classify_severity(corrected_p=0.04, d=-0.3, mean_delta=-0.1, n=30) == "medium"

    def test_low(self) -> None:
        assert _classify_severity(corrected_p=0.04, d=-0.1, mean_delta=-0.05, n=30) == "low"


# ---------------------------------------------------------------------------
# Statistics — full pipeline
# ---------------------------------------------------------------------------


def _ss(n: int, delta_avg_score: float, *, prompt: str = "p", ev: str = "e") -> list[SlicedScore]:
    """Build n SlicedScores with a controlled mean delta."""
    rng = np.random.default_rng(123)
    deltas = rng.normal(loc=delta_avg_score, scale=0.1, size=n)
    return [
        SlicedScore(
            prompt_id=prompt,
            evaluator_name=ev,
            example_id=f"ex{i}",
            source_score=0.5,
            target_score=0.5 + d,
            delta=float(d),
        )
        for i, d in enumerate(deltas)
    ]


class TestAnalyze:
    def test_clear_regression_marked_significant(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(40, -0.3)})
        # Single comparison so BH = raw.
        c = results[0]
        assert c.severity in ("critical", "high")
        assert c.test == "paired_t" or c.test == "wilcoxon"
        assert c.delta_avg_score < 0
        assert c.effect_size < 0
        assert c.p_value_corrected < 0.01

    def test_clear_improvement_marked_improved(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(40, +0.3)})
        c = results[0]
        assert c.severity == "improved"
        assert c.delta_avg_score > 0

    def test_no_real_effect_returns_none(self) -> None:
        # Mean delta near 0 and small spread → not significant.
        results = analyze(sliced_by_slice={"all": _ss(40, 0.0)})
        c = results[0]
        assert c.severity == "none"

    def test_small_sample_skipped(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(3, -0.5)})
        c = results[0]
        assert c.severity == "insufficient"
        assert c.test == "skipped"

    def test_zero_variance_skipped(self) -> None:
        identical = [SlicedScore("p", "e", f"ex{i}", 0.5, 0.4, -0.1) for i in range(10)]
        results = analyze(sliced_by_slice={"all": identical})
        # Every delta is the same; no inference possible.
        c = results[0]
        assert c.test == "skipped"
        assert "zero variance" in " ".join(c.notes)

    def test_results_sorted_by_severity(self) -> None:
        results = analyze(
            sliced_by_slice={
                "all": _ss(40, -0.5, prompt="bad"),
                "extra": _ss(40, +0.3, prompt="good"),
            },
        )
        # Critical/high should come before "improved" / "none".
        first_sev = results[0].severity
        last_sev = results[-1].severity
        order_first = {"critical": 0, "high": 1}.get(first_sev, 99)
        order_last = {"improved": 4, "none": 5, "insufficient": 6}.get(last_sev, 99)
        assert order_first < order_last

    def test_bh_correction_changes_p_with_many_tests(self) -> None:
        # Use the helper directly with a spread of raw p-values; BH must
        # raise at least some of them for the multi-comparison case.
        from evalshift_cli.analysis.statistics import _benjamini_hochberg

        raw_ps = [
            0.001,
            0.002,
            0.01,
            0.04,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.85,
            0.9,
            0.92,
            0.94,
            0.96,
            0.97,
            0.98,
            0.99,
        ]
        adj = _benjamini_hochberg(raw_ps)
        increased = [a for a, r in zip(adj, raw_ps, strict=True) if a > r]
        assert len(increased) > 0


class TestComparisonResultShape:
    def test_carries_ci_bounds(self) -> None:
        results = analyze(sliced_by_slice={"all": _ss(20, -0.3)})
        c = results[0]
        assert isinstance(c, ComparisonResult)
        assert c.effect_size_ci_low <= c.effect_size <= c.effect_size_ci_high or (
            c.effect_size_ci_low > c.effect_size_ci_high  # bootstrap may flip
        )


class TestSliceAggregateShape:
    def test_zero_n_aggregate(self) -> None:
        agg = SliceAggregate(
            name="empty",
            n=0,
            source_avg_score=0,
            target_avg_score=0,
            delta_avg_score=0,
            delta_min_score=0,
            delta_max_score=0,
            source_score_stdev=0,
            target_score_stdev=0,
        )
        assert agg.n == 0


class TestEachAxisIsItsOwnComparison:
    """One evaluator can score several axes; they must not be pooled.

    ``tool_selection`` writes a conformance row and a divergence row per
    example under one user-chosen name. They are different measurements
    against different baselines — conformance grades each side against
    ground truth, divergence grades the target against the source — so
    testing them as one sample averages a regression against a
    non-regression and reproduces the bug the split exists to fix.
    """

    CONFORMANCE = "tool_selection.conformance"
    DIVERGENCE = "tool_selection.divergence"

    def _rows(self) -> list[SlicedScore]:
        # Both models miss the ground truth (0.0/0.0, delta 0) and the target
        # does something else entirely (1.0/0.0, delta -1) — the personalButler
        # run's exact shape.
        return [
            SlicedScore("p", "routing", f"ex{i}", 0.0, 0.0, 0.0, self.CONFORMANCE) for i in range(6)
        ] + [
            SlicedScore("p", "routing", f"ex{i}", 1.0, 0.0, -1.0, self.DIVERGENCE) for i in range(6)
        ]

    def test_two_axes_produce_two_comparisons(self) -> None:
        results = analyze(sliced_by_slice={ALL_SLICE: self._rows()})

        assert len(results) == 2
        assert {r.n for r in results} == {6}  # not one pooled comparison of 12
        assert sorted(r.delta_avg_score for r in results) == [-1.0, 0.0]
        assert {r.evaluator_name for r in results} == {"routing"}

    def test_each_comparison_names_its_axis(self) -> None:
        """The rows share prompt, evaluator and slice, so the note is the
        only thing that tells a reader which is which. It rides in ``notes``
        because the bundle's Comparison schema forbids new fields."""
        results = analyze(sliced_by_slice={ALL_SLICE: self._rows()})

        axes = {
            note.removeprefix(AXIS_NOTE_PREFIX).strip()
            for r in results
            for note in r.notes
            if note.startswith(AXIS_NOTE_PREFIX)
        }
        assert axes == {self.CONFORMANCE, self.DIVERGENCE}

    def test_a_single_axis_evaluator_is_not_annotated(self) -> None:
        rows = [SlicedScore("p", "semantic.cosine", f"ex{i}", 1.0, 0.4, -0.6) for i in range(6)]

        (result,) = analyze(sliced_by_slice={ALL_SLICE: rows})

        assert not any(note.startswith(AXIS_NOTE_PREFIX) for note in result.notes)

    def test_each_comparison_carries_its_axis_kind(self) -> None:
        """``kind`` is a real field now, not only a note.

        The server keys ``run_comparisons`` on
        ``(run_id, prompt_id, evaluator_name, kind, slice_name)`` — without the
        field, the two axis rows of one evaluator collide on the server's
        unique constraint and finalize fails.
        """
        results = analyze(sliced_by_slice={ALL_SLICE: self._rows()})

        assert {r.kind for r in results} == {self.CONFORMANCE, self.DIVERGENCE}

    def test_kind_survives_the_bh_correction_rebuild(self) -> None:
        """Zero-variance rows skip the test and keep their object; rows with
        real variance are rebuilt after Benjamini-Hochberg, and the rebuild
        must not drop the axis."""
        varied = [
            SlicedScore(
                "p", "routing", f"ex{i}", 1.0, 1.0 - (i % 3) * 0.3, -(i % 3) * 0.3, self.CONFORMANCE
            )
            for i in range(9)
        ]

        (result,) = analyze(sliced_by_slice={ALL_SLICE: varied})

        assert result.test != "skipped"
        assert result.kind == self.CONFORMANCE

    def test_an_unmeasured_axis_gets_its_own_comparison_too(self) -> None:
        """Coverage is booked per axis, so its keys carry the slug as well."""
        results = analyze(
            sliced_by_slice={ALL_SLICE: self._rows()[:6]},
            unmeasured_by_slice={ALL_SLICE: {("p", "routing", self.DIVERGENCE): 6}},
        )

        by_axis = {
            next(
                note.removeprefix(AXIS_NOTE_PREFIX).strip()
                for note in r.notes
                if note.startswith(AXIS_NOTE_PREFIX)
            ): r
            for r in results
        }
        assert by_axis[self.CONFORMANCE].n == 6
        assert by_axis[self.DIVERGENCE].n == 0
        assert by_axis[self.DIVERGENCE].severity == "insufficient"


class TestUnmeasuredPairsAreNotMeasurements:
    """A pair an evaluator declined to score writes no row — and still counts.

    Its absence has to be reconstructed from the run's
    :class:`EvaluatorCoverage`, or an evaluator that measured nothing would
    simply disappear from the analysis instead of reporting
    ``severity: insufficient``, and a run where nothing was compared would
    read as a clean pass.
    """

    KEY = ("p", "llm_judge.equivalence", "")

    def test_an_evaluator_with_no_rows_at_all_is_insufficient_not_absent(self) -> None:
        (result,) = analyze(sliced_by_slice={}, unmeasured_by_slice={"all": {self.KEY: 9}})
        assert result.evaluator_name == "llm_judge.equivalence"
        assert result.severity == "insufficient"
        assert result.test == "skipped"
        assert result.n == 0
        assert any(note.startswith(UNMEASURED_NOTE_PREFIX) for note in result.notes)

    def test_unmeasured_pairs_are_excluded_from_the_test_but_not_the_note(self) -> None:
        measured = [
            SlicedScore("p", "llm_judge.equivalence", f"ex{i}", 1.0, 0.4, -0.6) for i in range(6)
        ]
        (result,) = analyze(
            sliced_by_slice={"all": measured},
            unmeasured_by_slice={"all": {self.KEY: 4}},
        )
        assert result.n == 6
        assert result.delta_avg_score == pytest.approx(-0.6)
        assert any("4 of 10 rows not applicable" in note for note in result.notes)

    def test_a_real_test_runs_on_the_measured_rows_only(self) -> None:
        # Non-zero variance, so the comparison reaches ttest_rel/wilcoxon. The
        # score arrays fed to the test must be the measured rows and nothing
        # else — the unmeasured four contribute a count, never a number.
        measured = [
            SlicedScore(
                "p", "llm_judge.equivalence", f"ex{i}", 1.0, 0.6 - i * 0.05, -0.4 - i * 0.05
            )
            for i in range(6)
        ]
        (result,) = analyze(
            sliced_by_slice={"all": measured},
            unmeasured_by_slice={"all": {self.KEY: 4}},
        )
        assert result.n == 6
        assert result.test in {"paired_t", "wilcoxon"}
        assert result.delta_avg_score < 0.0

    def test_a_measured_evaluator_needs_no_coverage_to_be_analysed(self) -> None:
        """The common case: nothing unmeasured, no note, no behaviour change."""
        measured = [SlicedScore("p", "e", f"ex{i}", 1.0, 0.4, -0.6) for i in range(6)]
        (result,) = analyze(sliced_by_slice={"all": measured})
        assert result.n == 6
        assert not any("not applicable" in note for note in result.notes)

    def test_an_advisory_axis_that_scored_nothing_says_so_in_its_notes(self) -> None:
        """Zero rows in ``scores.jsonl`` leave the policy layer nothing to
        read the config ``blocking`` flag from; this note is what keeps a
        ``blocking: false`` evaluator from being named as a blind gate."""
        (result,) = analyze(
            sliced_by_slice={},
            unmeasured_by_slice={"all": {self.KEY: 9}},
            advisory_axes={("llm_judge.equivalence", "")},
        )
        assert any(note.startswith(ADVISORY_NOTE_PREFIX) for note in result.notes)

    def test_a_gating_axis_never_carries_the_advisory_note(self) -> None:
        (result,) = analyze(sliced_by_slice={}, unmeasured_by_slice={"all": {self.KEY: 9}})
        assert not any(note.startswith(ADVISORY_NOTE_PREFIX) for note in result.notes)

    def test_a_measured_advisory_axis_needs_no_note(self) -> None:
        """Rows exist, and every row carries ``blocking`` — the note would
        restate what the policy layer already reads from the records."""
        measured = [
            SlicedScore("p", "llm_judge.equivalence", f"ex{i}", 1.0, 0.4, -0.6) for i in range(6)
        ]
        (result,) = analyze(
            sliced_by_slice={"all": measured},
            advisory_axes={("llm_judge.equivalence", "")},
        )
        assert not any(note.startswith(ADVISORY_NOTE_PREFIX) for note in result.notes)

    def test_unmeasured_note_survives_the_bundle_comparison_schema(self) -> None:
        # The bundle's Comparison object is additionalProperties: false, so the
        # unmeasured signal must ride in `notes` — never in a new field. `kind`
        # is in the server schema (it is part of the row's identity there).
        (result,) = analyze(sliced_by_slice={}, unmeasured_by_slice={"all": {self.KEY: 9}})
        allowed = {
            "prompt_id",
            "evaluator_name",
            "kind",
            "slice_name",
            "n",
            "test",
            "statistic",
            "p_value",
            "p_value_corrected",
            "effect_size",
            "effect_size_ci_low",
            "effect_size_ci_high",
            "delta_avg_score",
            "severity",
            "notes",
        }
        assert set(asdict(result)) == allowed


# ---------------------------------------------------------------------------
# Semantic drift inside min_similarity — tested as no change
# ---------------------------------------------------------------------------

#: The nine semantic.cosine target scores of run r_20261011_triage_e14cda,
#: every one above min_similarity 0.9 — the run that failed as "critical".
_TRIAGE_TARGETS = [0.9734, 0.9608, 0.9212, 0.9531, 0.9321, 0.9492, 0.9043, 0.9800, 0.9542]


def _semantic_record(example: str, target: float, *, flagged: bool) -> EvalRecord:
    return EvalRecord(
        run_id="r1",
        prompt_id="replay",
        example_id=example,
        evaluator_name="semantic.cosine",
        kind="semantic",
        source_score=1.0,
        target_score=target,
        delta=target - 1.0,
        metadata={"failure_categories": [SEMANTIC_REGRESSION]} if flagged else {},
    )


def _semantic_sliced(targets: list[float], flagged: list[bool]) -> dict[str, list[SlicedScore]]:
    records = [
        _semantic_record(f"ex{i}", t, flagged=f)
        for i, (t, f) in enumerate(zip(targets, flagged, strict=True))
    ]
    suite = Suite(examples=[suite_example(id=r.example_id) for r in records])
    return build_slices(records=records, suite=suite)


class TestWithinTolerance:
    def test_build_slices_marks_only_unflagged_semantic_rows(self) -> None:
        records = [
            _semantic_record("ex1", 0.95, flagged=False),
            _semantic_record("ex2", 0.70, flagged=True),
            _record(example="ex3", delta=-0.2),
        ]
        suite = Suite(examples=[suite_example(id=f"ex{i}") for i in (1, 2, 3)])
        sliced = build_slices(records=records, suite=suite)[ALL_SLICE]
        assert [s.within_tolerance for s in sliced] == [True, False, False]

    def test_drift_inside_min_similarity_is_not_a_regression(self) -> None:
        sliced = _semantic_sliced(_TRIAGE_TARGETS, [False] * 9)
        (c,) = analyze(sliced_by_slice=sliced)
        assert c.severity == "none"
        assert c.test == "skipped"
        # Descriptive mean stays the raw drift: the report still says -5%.
        assert c.delta_avg_score == pytest.approx(
            sum(t - 1.0 for t in _TRIAGE_TARGETS) / 9,
        )
        assert (
            "9 of 9 rows within the evaluator's own threshold (min_similarity) — "
            f"{WITHIN_TOLERANCE_REASON}"
        ) in c.notes

    def test_drift_below_min_similarity_still_regresses(self) -> None:
        targets = [0.70, 0.72, 0.75, 0.71, 0.74, 0.73, 0.76, 0.69, 0.72]
        (c,) = analyze(sliced_by_slice=_semantic_sliced(targets, [True] * 9))
        assert c.severity == "critical"
        assert not any(WITHIN_TOLERANCE_REASON in note for note in c.notes)

    def test_one_flagged_row_among_forgiven_ones_runs_and_stays_quiet(self) -> None:
        targets = [*_TRIAGE_TARGETS[:8], 0.60]
        flagged = [False] * 8 + [True]
        (c,) = analyze(sliced_by_slice=_semantic_sliced(targets, flagged))
        assert c.test == "wilcoxon"
        assert c.severity == "none"
        assert any(note.startswith("8 of 9 rows within") for note in c.notes)
        # The effect size reads the tolerance-adjusted deltas, not the raw
        # drift: on the raw ones d is about -0.78, which reads as a regression.
        tested = np.array([0.0] * 8 + [0.60 - 1.0])
        assert c.effect_size == pytest.approx(
            float(np.mean(tested) / np.std(tested, ddof=1)),
        )

    def test_the_t_path_sizes_its_effect_and_ci_on_adjusted_deltas(self) -> None:
        """Forgiven rows reach the paired t too, and its CI is built from that d."""
        flagged_deltas = [
            -0.30, -0.26, -0.24, -0.22, -0.21, -0.20, -0.19, -0.18,
            -0.17, -0.16, -0.15, -0.14, -0.12, -0.10, -0.08, -0.06,
        ]  # fmt: skip
        forgiven_targets = [0.96, 0.95]
        targets = [1.0 + d for d in flagged_deltas] + forgiven_targets
        flagged = [True] * len(flagged_deltas) + [False] * len(forgiven_targets)
        (c,) = analyze(sliced_by_slice=_semantic_sliced(targets, flagged))
        assert c.test == "paired_t"
        tested = np.array([*flagged_deltas, 0.0, 0.0])
        d = float(np.mean(tested) / np.std(tested, ddof=1))
        assert c.effect_size == pytest.approx(d)
        se = float(np.sqrt(1.0 / tested.size + d * d / (2.0 * tested.size)))
        assert c.effect_size_ci_low == pytest.approx(d - 1.96 * se)
        assert c.effect_size_ci_high == pytest.approx(d + 1.96 * se)
        # The descriptive mean is still the raw drift.
        assert c.delta_avg_score == pytest.approx(float(np.mean([t - 1.0 for t in targets])))

    def test_non_semantic_statistics_are_unchanged(self) -> None:
        pairs = _ss(40, -0.3)
        (c,) = analyze(sliced_by_slice={"all": pairs})
        assert c.test == "paired_t"
        expected = stats.ttest_rel(
            [p.target_score for p in pairs],
            [p.source_score for p in pairs],
        )
        assert c.statistic == pytest.approx(float(expected.statistic))
        assert c.p_value == pytest.approx(float(expected.pvalue))

    def test_aggregates_keep_the_raw_drift(self) -> None:
        sliced = _semantic_sliced(_TRIAGE_TARGETS, [False] * 9)
        agg = aggregates(sliced[ALL_SLICE], ALL_SLICE)
        assert agg.delta_min_score == pytest.approx(min(_TRIAGE_TARGETS) - 1.0)
