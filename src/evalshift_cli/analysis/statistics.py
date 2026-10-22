"""Statistical analysis of paired source/target scores.

Implements the contract from PDF §5.5 step-by-step:

1. Pair scores per slice; require ``n>=5`` to test, flag ``n<20``.
2. Test normality via Shapiro-Wilk on the deltas; route to a paired
   t-test (normal) or Wilcoxon signed-rank (non-normal).
3. Compute Cohen's d for paired samples (``mean(deltas)/std(deltas)``)
   over the tolerance-adjusted deltas.
4. Compute a 95% CI on the effect size — analytical for the t-test,
   bootstrap for Wilcoxon.
5. After every comparison in the run is collected, apply the
   Benjamini-Hochberg correction (FDR=0.05) across all p-values to
   control false-discovery in the multi-test setting.
6. Classify severity from the *corrected* p-value + effect size.

Why these choices:

* **Paired tests** because each example is run on both models — the
  measurements are inherently paired and ignoring that throws away
  power.
* **Shapiro-Wilk gate** because the t-test is sensitive to non-normal
  deltas at small N; falling back to Wilcoxon (which only assumes
  symmetric distribution) is the standard MVP-grade safety net.
* **Cohen's d** as the effect size because it's interpretable in units
  of standard deviation and the field has an intuition for it
  (``|d|>0.8`` = "large").
* **Benjamini-Hochberg** instead of Bonferroni because BH controls FDR
  at the same level with substantially more power, especially when
  some comparisons are real (which is exactly the case in a real
  migration eval).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from evalshift_cli.analysis.slicing import ComparisonKey, SlicedScore, UnmeasuredCounts

Severity = Literal[
    "critical",
    "high",
    "medium",
    "low",
    "improved",
    "none",
    "insufficient",
]

TestKind = Literal["paired_t", "wilcoxon", "skipped"]

MIN_N_FOR_TEST: int = 5
MIN_N_RELIABLE: int = 20
NORMALITY_ALPHA: float = 0.05
FDR_ALPHA: float = 0.05
BOOTSTRAP_RESAMPLES: int = 2000
DEFAULT_RNG_SEED: int = 0

#: Prefix of the note written when a comparison has no measurements at all.
#: The policy and report layers select on it. It lives in ``notes`` rather
#: than a new ``ComparisonResult`` field because the bundle's ``Comparison``
#: object is ``additionalProperties: false`` — a new field would fail upload
#: validation.
UNMEASURED_NOTE_PREFIX = "nothing measured:"

#: Prefix of the note naming which of an evaluator's axes a comparison
#: covers. Written only when one evaluator contributed more than one axis to
#: the same slice, because that is the only time the ``(prompt, evaluator,
#: slice)`` triple a report renders no longer identifies the row.
AXIS_NOTE_PREFIX = "axis:"

#: Prefix of the note marking a comparison whose axis is advisory
#: (``blocking: false`` in ``evalshift.yaml``). Written only on a comparison
#: with no measurements at all: an axis with rows carries the flag on every
#: row of ``scores.jsonl``, but zero rows leave the policy layer nothing to
#: read it from, and without this note a silent advisory evaluator was named
#: as a blind *gate* by ``unmeasured_gating_evaluators``. Rides in ``notes``
#: for the same schema reason as :data:`UNMEASURED_NOTE_PREFIX`.
ADVISORY_NOTE_PREFIX = "advisory:"

#: Suffix of the note counting the pairs an evaluator was handed and did not
#: score. Those pairs have no row in ``scores.jsonl``; they are reconstructed
#: from the run's ``EvaluatorCoverage`` — see
#: :func:`evalshift_cli.analysis.slicing.build_unmeasured`.
NOT_APPLICABLE_REASON = "the evaluator measured nothing on them"

#: Suffix of the note counting rows whose drop sat inside the evaluator's own
#: threshold (a semantic row above ``min_similarity``). Those rows enter the
#: test as a delta of zero — see :mod:`evalshift_cli.analysis.tolerance` —
#: and the note is what tells a reader why a visible average drift tested as
#: no change.
WITHIN_TOLERANCE_REASON = "tested as no change"

#: The note on a comparison whose tested deltas are all the same value, so no
#: test ran: its p-value (1.0) and effect size (0.0) are placeholders.
ZERO_VARIANCE_NOTE = "zero variance — no test"

#: Prefixes of the note on a zero-variance comparison whose pairs all moved by
#: the same non-zero amount. No test can run on it, so it is severity ``none``
#: with a placeholder ``effect_size`` of 0 -- but every pair agrees, and a
#: report reading that ``none`` as "no meaningful difference" put a green tick
#: beside "-100%". The direction is in the prefix, so a reader never parses the
#: figure. In ``notes`` for the same schema reason as
#: :data:`UNMEASURED_NOTE_PREFIX`.
UNIFORM_DROP_NOTE_PREFIX = "uniform drop:"
UNIFORM_GAIN_NOTE_PREFIX = "uniform gain:"

#: Cohen's |d| below which an effect is negligible (Cohen's conventions). A
#: ``none`` comparison means only that the corrected p-value missed
#: :data:`FDR_ALPHA`; it reads as *equivalent* only below this.
NEGLIGIBLE_EFFECT: float = 0.2


def uniform_change(notes: Sequence[str]) -> Literal["drop", "gain"] | None:
    """The direction a zero-variance comparison's pairs all moved in, or ``None``."""
    for note in notes:
        if note.startswith(UNIFORM_DROP_NOTE_PREFIX):
            return "drop"
        if note.startswith(UNIFORM_GAIN_NOTE_PREFIX):
            return "gain"
    return None


def unconfirmed_change(severity: str, effect_size: float) -> bool:
    """A ``none`` comparison whose effect is too large to read as equivalent.

    ``none`` means the corrected p-value missed alpha, whatever the effect: a
    medium effect (d -0.51 at n = 21) that missed it is unconfirmed, not absent.
    """
    return severity == "none" and abs(effect_size) >= NEGLIGIBLE_EFFECT


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """One (prompt_id, evaluator_name, kind, slice_name) statistical verdict.

    ``kind`` is the evaluator's axis, for an evaluator that scores more than
    one; it is part of what :func:`analyze` groups on and part of the bundle's
    ``Comparison`` identity — the server keys ``run_comparisons`` on
    ``(run_id, prompt_id, evaluator_name, kind, slice_name)``, so two axis
    rows of one evaluator would collide without it. It defaults to ``""``
    because ``analysis.json`` files written before the field existed load
    without it. For multi-axis evaluators the axis additionally rides in
    ``notes`` under :data:`AXIS_NOTE_PREFIX`, which is what the reports
    render.
    """

    prompt_id: str
    evaluator_name: str
    slice_name: str
    n: int
    test: TestKind
    statistic: float
    p_value: float
    p_value_corrected: float
    effect_size: float
    effect_size_ci_low: float
    effect_size_ci_high: float
    delta_avg_score: float
    severity: Severity
    notes: list[str]
    # Defaulted (and therefore last): pre-existing ``analysis.json`` files
    # load without it.
    kind: str = ""


def analyze(
    *,
    sliced_by_slice: dict[str, list[SlicedScore]],
    unmeasured_by_slice: UnmeasuredCounts | None = None,
    advisory_axes: Collection[tuple[str, str]] | None = None,
    rng: np.random.Generator | None = None,
) -> list[ComparisonResult]:
    """Run every comparison and apply BH correction across the whole set.

    Args:
        sliced_by_slice: Measured pairs, grouped by slice.
        unmeasured_by_slice: Pairs an evaluator was handed and produced no
            row for, from :func:`evalshift_cli.analysis.slicing.build_unmeasured`.
            A ``(prompt, evaluator, kind)`` that appears *only* here still
            gets a comparison — one with ``n=0`` and ``severity: "insufficient"``.
            That verdict is the sole remaining barrier between a run where
            nothing was measured and a confident pass, so an evaluator must
            never drop out of the analysis just because all of its rows are
            absent.
        advisory_axes: ``(evaluator_name, kind)`` axes configured
            ``blocking: false``, read from the run's ``EvaluatorCoverage``.
            An axis listed here that measured nothing gets an
            :data:`ADVISORY_NOTE_PREFIX` note on its synthesized comparison —
            with zero rows, that note is the only carrier of the flag left
            for the policy layer to read.
        rng: Seeded generator for the bootstrap CI.

    Returns a list of :class:`ComparisonResult` ordered by descending
    severity then absolute effect size — i.e. the worst regressions
    bubble to the top.
    """
    rng_inst = rng or np.random.default_rng(DEFAULT_RNG_SEED)
    unmeasured_by_slice = unmeasured_by_slice or {}
    advisory = frozenset(advisory_axes or ())

    slice_names = [
        *sliced_by_slice,
        *(name for name in unmeasured_by_slice if name not in sliced_by_slice),
    ]

    # Group within each slice by (prompt_id, evaluator_name, kind) so we
    # run one test per (prompt, evaluator, axis, slice). Keying on the name
    # alone pooled every axis an evaluator scores into a single sample.
    raw: list[ComparisonResult] = []
    for slice_name in slice_names:
        by_key: dict[ComparisonKey, list[SlicedScore]] = {}
        for s in sliced_by_slice.get(slice_name, []):
            by_key.setdefault((s.prompt_id, s.evaluator_name, s.kind), []).append(s)
        unmeasured = unmeasured_by_slice.get(slice_name, {})
        keys = [*by_key, *(key for key in unmeasured if key not in by_key)]
        multi_axis = _multi_axis_names(keys)
        for key in keys:
            prompt_id, evaluator_name, kind = key
            raw.append(
                _one_comparison(
                    prompt_id=prompt_id,
                    evaluator_name=evaluator_name,
                    kind=kind,
                    slice_name=slice_name,
                    pairs=by_key.get(key, []),
                    n_unmeasured=unmeasured.get(key, 0),
                    name_axis=(prompt_id, evaluator_name) in multi_axis,
                    advisory=(evaluator_name, kind) in advisory,
                    rng=rng_inst,
                ),
            )

    # Apply Benjamini-Hochberg across every comparison that actually
    # ran a test (skipped comparisons keep p_value = 1.0 and don't
    # participate in the correction).
    testable_idx = [i for i, c in enumerate(raw) if c.test != "skipped"]
    p_raw = [raw[i].p_value for i in testable_idx]
    p_corrected_seq = _benjamini_hochberg(p_raw)
    corrected_lookup: dict[int, float] = dict(
        zip(testable_idx, p_corrected_seq, strict=True),
    )

    finalised: list[ComparisonResult] = []
    for i, c in enumerate(raw):
        if c.test == "skipped":
            finalised.append(c)
            continue
        cp = corrected_lookup[i]
        sev = _classify_severity(
            corrected_p=cp, d=c.effect_size, mean_delta=c.delta_avg_score, n=c.n
        )
        finalised.append(
            ComparisonResult(
                prompt_id=c.prompt_id,
                evaluator_name=c.evaluator_name,
                kind=c.kind,
                slice_name=c.slice_name,
                n=c.n,
                test=c.test,
                statistic=c.statistic,
                p_value=c.p_value,
                p_value_corrected=cp,
                effect_size=c.effect_size,
                effect_size_ci_low=c.effect_size_ci_low,
                effect_size_ci_high=c.effect_size_ci_high,
                delta_avg_score=c.delta_avg_score,
                severity=sev,
                notes=c.notes,
            ),
        )

    finalised.sort(key=_severity_sort_key)
    return finalised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _multi_axis_names(keys: list[ComparisonKey]) -> set[tuple[str, str]]:
    """``(prompt_id, evaluator_name)`` pairs that contributed several axes.

    Only those need their axis spelled out in ``notes``: everywhere else the
    ``(prompt, evaluator, slice)`` triple already identifies the row, and a
    note on every comparison in the run would be noise.
    """
    axes: dict[tuple[str, str], set[str]] = {}
    for prompt_id, evaluator_name, kind in keys:
        axes.setdefault((prompt_id, evaluator_name), set()).add(kind)
    return {name for name, kinds in axes.items() if len(kinds) > 1}


def _one_comparison(
    *,
    prompt_id: str,
    evaluator_name: str,
    kind: str,
    slice_name: str,
    pairs: list[SlicedScore],
    n_unmeasured: int,
    name_axis: bool,
    advisory: bool,
    rng: np.random.Generator,
) -> ComparisonResult:
    """Compute one comparison, deferring severity until BH is done.

    ``pairs`` holds only real measurements; ``n_unmeasured`` counts the pairs
    this evaluator was handed in this slice and wrote no row for, so the
    two together restore the denominator the absent rows took with them.
    ``kind`` is the axis these rows came from. It is not carried onto the
    result — see :class:`ComparisonResult` — so when ``name_axis`` says the
    evaluator scored more than one, it is written into ``notes`` instead;
    without that, two rows a report labels identically would be
    indistinguishable. ``advisory`` says the axis is ``blocking: false`` in
    config; it is only worth writing down when there are no rows left to say
    it (``n == 0``) — a measured axis carries the flag on every record.
    """
    n = len(pairs)
    deltas = np.array([p.delta for p in pairs], dtype=float)
    # What the test sees: a drop inside the evaluator's own threshold is no
    # change, exactly as the policy budgets count it. ``deltas`` stays raw
    # for ``delta_avg_score``, which describes the drift rather than judging
    # it. Only a semantic row is ever forgiven, and a semantic delta is never
    # positive (the source is pinned at 1.0), so zeroing forgiven rows moves
    # the tested mean from the raw one toward zero — never past it, so the
    # two means never disagree in sign.
    tested = np.array([0.0 if p.within_tolerance else p.delta for p in pairs], dtype=float)
    forgiven = sum(1 for p in pairs if p.within_tolerance)
    notes: list[str] = []
    if name_axis:
        notes.append(f"{AXIS_NOTE_PREFIX} {kind}")
    if n_unmeasured:
        notes.append(
            f"{n_unmeasured} of {n + n_unmeasured} rows not applicable — {NOT_APPLICABLE_REASON}",
        )
    if forgiven:
        notes.append(
            f"{forgiven} of {n} rows within the evaluator's own threshold "
            f"(min_similarity) — {WITHIN_TOLERANCE_REASON}",
        )

    if n == 0:
        # Every pair was a non-measurement. This is emphatically not
        # "equivalent": the evaluator never compared anything. Report it as
        # insufficient so `_verdict_for` can refuse to call it a pass.
        empty_notes = [
            *notes,
            f"{UNMEASURED_NOTE_PREFIX} this evaluator scored no comparable pair",
        ]
        if advisory:
            empty_notes.append(
                f"{ADVISORY_NOTE_PREFIX} blocking is false in evalshift.yaml — "
                "this axis reports and never gates",
            )
        return ComparisonResult(
            prompt_id=prompt_id,
            evaluator_name=evaluator_name,
            kind=kind,
            slice_name=slice_name,
            n=0,
            test="skipped",
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            effect_size=0.0,
            effect_size_ci_low=0.0,
            effect_size_ci_high=0.0,
            delta_avg_score=0.0,
            severity="insufficient",
            notes=empty_notes,
        )

    if n < MIN_N_FOR_TEST:
        return ComparisonResult(
            prompt_id=prompt_id,
            evaluator_name=evaluator_name,
            kind=kind,
            slice_name=slice_name,
            n=n,
            test="skipped",
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            effect_size=0.0,
            effect_size_ci_low=0.0,
            effect_size_ci_high=0.0,
            delta_avg_score=float(np.mean(deltas)),
            severity="insufficient",
            notes=[*notes, f"n={n} < {MIN_N_FOR_TEST}; no test run"],
        )
    if n < MIN_N_RELIABLE:
        notes.append(f"n={n} < {MIN_N_RELIABLE}; results uncertain")

    # Variance ≈ 0 (every delta identical, modulo float noise) → no
    # inference possible. Use a small tolerance to dodge catastrophic
    # cancellation that scipy already warns about.
    if float(np.std(tested, ddof=1)) < 1e-9:
        # Every tested delta is the same value. When it is not zero, every
        # pair moved by it: name that, since d below is only a placeholder.
        # (All pairs forgiven within tolerance test as 0 and stay unnamed, and
        # so does a shift too small to print: "moved by -0.000" says nothing.)
        shift = float(np.mean(tested))
        uniform: list[str] = []
        if round(shift, 3) != 0:
            prefix = UNIFORM_DROP_NOTE_PREFIX if shift < 0 else UNIFORM_GAIN_NOTE_PREFIX
            uniform.append(
                f"{prefix} every pair moved by {shift:+.3f}; zero variance leaves nothing "
                "to test, but no pair disagrees",
            )
        return ComparisonResult(
            prompt_id=prompt_id,
            evaluator_name=evaluator_name,
            kind=kind,
            slice_name=slice_name,
            n=n,
            test="skipped",
            statistic=0.0,
            p_value=1.0,
            p_value_corrected=1.0,
            effect_size=0.0,
            effect_size_ci_low=0.0,
            effect_size_ci_high=0.0,
            delta_avg_score=float(np.mean(deltas)),
            severity="none",
            notes=[*notes, ZERO_VARIANCE_NOTE, *uniform],
        )

    # Pick the test based on a normality screen.
    use_t = _is_normal(tested)

    if use_t:
        # One-sample t on the paired deltas is ``ttest_rel(target, source)``
        # exactly; it is written this way so a forgiven row can be zero.
        result = stats.ttest_1samp(tested, 0.0)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        d = _cohens_d_paired(tested)
        ci_low, ci_high = _t_test_d_ci(d=d, n=n)
        test_kind: TestKind = "paired_t"
    else:
        result = stats.wilcoxon(tested, zero_method="wilcox", correction=False)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        d = _cohens_d_paired(tested)
        ci_low, ci_high = _bootstrap_d_ci(tested, rng=rng)
        test_kind = "wilcoxon"
        notes.append("non-normal deltas; used Wilcoxon")

    return ComparisonResult(
        prompt_id=prompt_id,
        evaluator_name=evaluator_name,
        kind=kind,
        slice_name=slice_name,
        n=n,
        test=test_kind,
        statistic=statistic,
        p_value=p_value,
        p_value_corrected=p_value,  # placeholder; overwritten by BH
        effect_size=d,
        effect_size_ci_low=ci_low,
        effect_size_ci_high=ci_high,
        delta_avg_score=float(np.mean(deltas)),
        severity="none",  # placeholder; classified after BH
        notes=notes,
    )


def _is_normal(deltas: np.ndarray) -> bool:
    """Shapiro-Wilk normality screen.

    Skips for n>5000 where Shapiro is unreliable / overly powerful;
    in that regime a t-test is robust enough by central-limit-theorem
    grounds.
    """
    if deltas.size > 5000:
        return True
    try:
        result = stats.shapiro(deltas)
    except Exception:
        return False
    return float(result.pvalue) > NORMALITY_ALPHA


def _cohens_d_paired(deltas: np.ndarray) -> float:
    sd = float(np.std(deltas, ddof=1))
    if sd < 1e-9:
        return 0.0
    return float(np.mean(deltas) / sd)


def _t_test_d_ci(*, d: float, n: int) -> tuple[float, float]:
    """Analytical 95% CI on Cohen's d (paired) using the SE approximation.

    Standard error: sqrt(1/n + d^2/(2n)). Times 1.96 for 95%.
    """
    se = float(np.sqrt(1.0 / n + (d * d) / (2.0 * n)))
    return d - 1.96 * se, d + 1.96 * se


def _bootstrap_d_ci(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI on Cohen's d (paired)."""
    n = deltas.size
    sample_ds = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        d = deltas[idx]
        sd = float(np.std(d, ddof=1))
        sample_ds[i] = float(np.mean(d)) / sd if sd != 0 else 0.0
    return float(np.percentile(sample_ds, 2.5)), float(np.percentile(sample_ds, 97.5))


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Apply the BH FDR correction; returns adjusted p-values in input order.

    Mirrors ``statsmodels.stats.multitest.multipletests(..., method='fdr_bh')``
    semantics: each adjusted p-value is the smallest q such that the
    BH cutoff would still call the corresponding raw p significant.
    """
    n = len(p_values)
    if n == 0:
        return []
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    adjusted_ranked = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity (right-to-left running min) and clamp to 1.
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adjusted_ranked
    return [float(x) for x in out]


def _classify_severity(
    *,
    corrected_p: float,
    d: float,
    mean_delta: float,
    n: int,
) -> Severity:
    """Map (corrected p, effect size, direction, n) to severity per PDF §5.5."""
    if n < MIN_N_FOR_TEST:
        return "insufficient"
    if corrected_p >= FDR_ALPHA:
        return "none"
    abs_d = abs(d)
    if mean_delta > 0:
        return "improved"
    # mean_delta < 0 → regression
    if corrected_p < 0.01 and abs_d > 0.8:
        return "critical"
    if abs_d > 0.5:
        return "high"
    if abs_d > 0.2:
        return "medium"
    return "low"


_SEVERITY_RANK: dict[Severity, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "improved": 4,
    "none": 5,
    "insufficient": 6,
}


def _severity_sort_key(c: ComparisonResult) -> tuple[int, float, str, str, str]:
    return (
        _SEVERITY_RANK[c.severity],
        -abs(c.effect_size),
        c.prompt_id,
        c.evaluator_name,
        c.slice_name,
    )


__all__ = [
    "ADVISORY_NOTE_PREFIX",
    "AXIS_NOTE_PREFIX",
    "BOOTSTRAP_RESAMPLES",
    "FDR_ALPHA",
    "MIN_N_FOR_TEST",
    "MIN_N_RELIABLE",
    "NEGLIGIBLE_EFFECT",
    "NORMALITY_ALPHA",
    "NOT_APPLICABLE_REASON",
    "UNIFORM_DROP_NOTE_PREFIX",
    "UNIFORM_GAIN_NOTE_PREFIX",
    "UNMEASURED_NOTE_PREFIX",
    "WITHIN_TOLERANCE_REASON",
    "ZERO_VARIANCE_NOTE",
    "ComparisonResult",
    "Severity",
    "TestKind",
    "analyze",
    "unconfirmed_change",
    "uniform_change",
]
