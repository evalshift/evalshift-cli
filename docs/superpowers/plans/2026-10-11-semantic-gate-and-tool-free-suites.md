# Semantic Gate + Tool-Free Suites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A run whose semantic drift stays inside `min_similarity` and whose suite offers no tools gets the verdict its budgets earned (`pass` for `r_20261011_triage_e14cda`), and the report never claims a budget breach that did not happen.

**Architecture:** One shared predicate (`analysis/tolerance.py::within_tolerance`) defines "drop the evaluator's own threshold forgives". The policy budgets already used that rule; the slicer now stamps it on each `SlicedScore` so the paired test runs on tolerance-adjusted deltas while every descriptive number stays raw. Separately, `evaluate` stops attempting tool evaluators on pairs that were offered no tools — the same no-cell treatment an `applies_to` miss gets — and warns once with the real cause. Three wording fixes (`reason`, fallback recommendation, critical count) ride with it.

**Tech Stack:** Python 3.11+, numpy, scipy, pydantic v2, Typer + Rich, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-11-semantic-gate-and-tool-free-suites-design.md` — read it first; this plan argues from it.

## Global Constraints

- Python floor 3.11: no 3.12+ syntax. Every module starts with `from __future__ import annotations`; public functions get Google-style docstrings; `mypy --strict src/evalshift_cli` stays clean.
- No config, suite, `state.json`, `scores.jsonl` or bundle schema change. `ComparisonResult` gets **no new field** (the bundle's `Comparison` is `additionalProperties: false`); new information rides in `notes`.
- `delta_avg_score`, slice `aggregates`, per-example rows: **raw** deltas, unchanged. Only the test statistic, p-value, effect size, its CI, and therefore severity see tolerance-adjusted deltas.
- User-facing strings, verbatim:
  - comparison note: `"{k} of {n} rows within the evaluator's own threshold (min_similarity) — tested as no change"`
  - severity reason, per blocking comparison joined with `"; "`: `"{evaluator} regressed at {severity} severity on prompt {prompt} (effect size {d:+.2f}, adjusted p {p:.4f})"`, then `". A critical or high regression from a blocking evaluator fails the run on its own, whatever the budgets say."`
  - tool-free warning: `"[yellow]⚠[/yellow] {kind} evaluator '{name}' checked nothing: no example in {suite 'x' | this run} offered the models any tools, so there were no tool calls to check. It did not gate this run."`
  - fallback recommendation (fail): budgets → `"Do not migrate under the configured policy — the {labels} {budget was|budgets were} breached. Address the blocking evaluators, then re-run."`; severity → `"Do not migrate under the configured policy — {evaluators} regressed at blocking severity although no budget was breached. Review {its|their} regressed examples, then re-run."`; neither → `"Do not migrate under the configured policy — see the verdict's reason, address it, then re-run."`
  - verdict summary: `"… with {b} blocking {regression|regressions} ({c} critical) across {n} examples."`
- Gate ("green" = all four, as CI): `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest` (coverage floor 90). `make ci` is the exact CI mirror.
- Commit with `uv run --with pre-commit git commit …` (the shared hook points at a deleted venv). Never `--no-verify`. Conventional Commits, ending with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. `docs/superpowers/` is gitignored: add files there with `git add -f`.
- Always `git -C /home/lukas/repos/evalshift/evalshift-cli` (= `$WT`); never run git from a sibling repo's cwd. Before each commit, confirm the branch is `fix/semantic-gate-tool-free-suites`, not `main`.

## Review Focus

- A semantic comparison where most rows are forgiven and one is flagged (`[0]*8 + [-0.4]`) must not crash scipy or report `critical` — Task 2 `test_one_flagged_row_among_forgiven_ones_runs_and_stays_quiet`.
- Rows checkpointed before `EvalRecord.kind` existed (`kind=""`, name `semantic.cosine`) and a semantic evaluator the user renamed (`kind="semantic"`, name `meaning`) are both forgiven — Task 1 `test_legacy_rows_fall_back_to_the_name_prefix`, `test_a_renamed_semantic_evaluator_is_selected_by_kind`.
- A tool-free pair whose call **errored** keeps its error row, and a pair where only **one** side has a trace stays unmeasured — Task 5 `test_an_errored_tool_free_pair_keeps_its_error_row`, `test_a_one_sided_trace_is_still_unmeasured`.
- A run that both breaches a budget and has a critical comparison: `reason` stays `None` (the hero's budget count explains it) and the recommendation names the budget, not the severity — Task 3 `test_a_budget_breach_keeps_the_budget_line`, Task 4 `test_a_budget_fail_names_the_budget`.
- An `applies_to` that matches nothing on a tool-free run still gets the `applies_to` wording, not the tool-free one — Task 5 `test_an_applies_to_miss_keeps_its_own_wording_on_a_tool_free_run`.

---

### Task 0: Branch (done 2026-10-11)

**Files:** none.

Work happens in the **main checkout** on branch `fix/semantic-gate-tool-free-suites`, cut from
`origin/main` at `1045843` — no worktree. The spec and this plan are committed on it.
`$WT` below always means `/home/lukas/repos/evalshift/evalshift-cli`.

- [x] **Step 1: Create the branch from an up-to-date `main`**

```bash
git -C /home/lukas/repos/evalshift/evalshift-cli fetch origin
git -C /home/lukas/repos/evalshift/evalshift-cli switch -c fix/semantic-gate-tool-free-suites
```

- [x] **Step 2: Commit the spec and plan**

```bash
WT=/home/lukas/repos/evalshift/evalshift-cli
git -C $WT add -f docs/superpowers/specs/2026-10-11-semantic-gate-and-tool-free-suites-design.md docs/superpowers/plans/2026-10-11-semantic-gate-and-tool-free-suites.md
cd $WT && uv run --with pre-commit git commit -m "docs(plan): semantic gate and tool-free suites"
```

- [ ] **Step 3: Baseline the suite before Task 1**

```bash
WT=/home/lukas/repos/evalshift/evalshift-cli
git -C $WT branch --show-current   # must print fix/semantic-gate-tool-free-suites
cd $WT && uv sync --extra dev && uv run pytest -q -x -p no:cacheprovider 2>&1 | tail -3
```
Expected: all pass (≈3000 tests).

---

### Task 1: One definition of "within tolerance"

**Files:**
- Create: `src/evalshift_cli/analysis/tolerance.py`
- Modify: `src/evalshift_cli/analysis/policy.py:24-29` (imports), `:38`, `:59-62`, `:664-690`, `:733`
- Test: `tests/unit/test_tolerance.py`

**Interfaces:**
- Consumes: `EvalRecord` (`evaluators/base.py`), `SEMANTIC_REGRESSION` (`evaluators/failures.py`).
- Produces: `SEMANTIC_KIND: str = "semantic"`, `is_semantic(record: EvalRecord) -> bool`, `is_semantic_regression(record: EvalRecord) -> bool`, `within_tolerance(record: EvalRecord) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tolerance.py`:

```python
"""Tests for the shared within-tolerance predicate."""

from __future__ import annotations

from evalshift_cli.analysis.tolerance import (
    is_semantic,
    is_semantic_regression,
    within_tolerance,
)
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION


def _record(
    *,
    kind: str = "semantic",
    name: str = "semantic.cosine",
    target: float = 0.95,
    flagged: bool = False,
) -> EvalRecord:
    return EvalRecord(
        run_id="r1",
        prompt_id="p",
        example_id="ex1",
        evaluator_name=name,
        kind=kind,
        source_score=1.0,
        target_score=target,
        delta=target - 1.0,
        metadata={"failure_categories": [SEMANTIC_REGRESSION]} if flagged else {},
    )


def test_unflagged_semantic_drift_is_within_tolerance() -> None:
    assert within_tolerance(_record(target=0.904))


def test_flagged_semantic_drift_is_not() -> None:
    record = _record(target=0.75, flagged=True)
    assert is_semantic_regression(record)
    assert not within_tolerance(record)


def test_other_kinds_are_never_within_tolerance() -> None:
    assert not within_tolerance(_record(kind="llm_judge", name="llm_judge.equivalence"))
    assert not within_tolerance(_record(kind="", name="structural.length"))


def test_legacy_rows_fall_back_to_the_name_prefix() -> None:
    legacy = _record(kind="", name="semantic.cosine")
    assert is_semantic(legacy)
    assert within_tolerance(legacy)


def test_a_renamed_semantic_evaluator_is_selected_by_kind() -> None:
    assert within_tolerance(_record(kind="semantic", name="meaning"))


def test_a_malformed_category_list_is_not_a_regression() -> None:
    record = _record().model_copy(update={"metadata": {"failure_categories": "SEMANTIC"}})
    assert not is_semantic_regression(record)
    assert within_tolerance(record)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_tolerance.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalshift_cli.analysis.tolerance'`.

- [ ] **Step 3: Create the module**

Create `src/evalshift_cli/analysis/tolerance.py`:

```python
"""Which score drops an evaluator's own threshold already forgives.

``semantic`` pins the source at 1.0, so every target that words anything
differently scores below it: a negative delta is the evaluator's *normal*
output, not evidence of a regression. ``min_similarity`` is where the user
drew the line, and the evaluator records which side of it a row fell on as a
``SEMANTIC_REGRESSION`` failure category at scoring time.

The policy budgets and the paired significance test must read that one line.
When only the budgets did, nine rows at 0.90-0.98 passed every budget and still
failed the run as a "critical" regression — nine small, consistent drops have
almost no variance, so the test was certain of a difference nobody had asked
it to find.
"""

from __future__ import annotations

from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION

#: The ``EvalRecord.kind`` slug the semantic evaluator stamps.
SEMANTIC_KIND = "semantic"

#: Records checkpointed before ``EvalRecord.kind`` existed carry only the name
#: prefix the evaluator was registered under.
_LEGACY_SEMANTIC_PREFIX = "semantic."


def is_semantic(record: EvalRecord) -> bool:
    """Whether ``record`` came from a ``semantic`` evaluator.

    Selected by kind, so a user who renames the evaluator keeps the rule;
    falls back to the legacy name prefix only for rows written without one.
    """
    if record.kind:
        return record.kind == SEMANTIC_KIND
    return record.evaluator_name.startswith(_LEGACY_SEMANTIC_PREFIX)


def is_semantic_regression(record: EvalRecord) -> bool:
    """Whether a semantic record breached ``min_similarity``.

    Reads the ``SEMANTIC_REGRESSION`` flag the evaluator wrote at scoring time
    rather than re-deriving it from ``delta``: the threshold lives in config,
    and the evaluator is the one place that knew it.
    """
    categories = record.metadata.get("failure_categories", [])
    return isinstance(categories, list) and SEMANTIC_REGRESSION in categories


def within_tolerance(record: EvalRecord) -> bool:
    """Whether ``record``'s drop is inside its evaluator's own threshold.

    True only for a semantic row that stayed above ``min_similarity``. Such a
    row counts as *equivalent* in every policy rate and as *no change* in the
    paired test; every other row is taken at its delta.
    """
    return is_semantic(record) and not is_semantic_regression(record)


__all__ = [
    "SEMANTIC_KIND",
    "is_semantic",
    "is_semantic_regression",
    "within_tolerance",
]
```

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/unit/test_tolerance.py -q`
Expected: 6 passed.

- [ ] **Step 5: Make the policy read the shared predicate**

In `src/evalshift_cli/analysis/policy.py`:

1. Imports (lines 16-31): add `from evalshift_cli.analysis.tolerance import is_semantic, is_semantic_regression, within_tolerance` after the `statistics` import, and drop `SEMANTIC_REGRESSION,` from the `evaluators.failures` import list (nothing else in the module uses it — confirm with `grep -n SEMANTIC_REGRESSION src/evalshift_cli/analysis/policy.py`).
2. Delete line 38 `_SEMANTIC_KIND = "semantic"`.
3. In `_LEGACY_KIND_PREFIXES` (lines 59-62) delete the `_SEMANTIC_KIND: "semantic.",` entry (the legacy prefix now lives in `tolerance.py`).
4. Replace lines 664-690 (`_is_semantic_regression`, `_is_regression`, `_is_equivalent`) with:

```python
def _is_regression(r: EvalRecord) -> bool:
    """Whether a record counts as a regression for policy budgets."""
    if is_semantic(r):
        return is_semantic_regression(r)
    return r.delta < 0


def _is_equivalent(r: EvalRecord) -> bool:
    """Whether a record counts as source/target equivalent.

    Semantic drift within ``min_similarity`` is treated as equivalent (not a
    regression), keeping the improved/equivalent/regressed partition intact.
    The same rule decides what the paired test sees — see
    :mod:`evalshift_cli.analysis.tolerance`.
    """
    if is_semantic(r):
        return within_tolerance(r)
    return r.delta == 0
```

5. In `is_shared_ground_truth_miss`'s docstring (line ~733) change ``:func:`_is_semantic_regression` `` to ``:func:`~evalshift_cli.analysis.tolerance.is_semantic_regression` ``.

- [ ] **Step 6: Run the policy tests and the type check**

Run: `uv run pytest tests/unit/test_policy.py tests/unit/test_tolerance.py -q && uv run mypy --strict src/evalshift_cli/analysis`
Expected: all pass (in particular `test_unflagged_semantic_drift_is_equivalent_not_regression`, `test_flagged_semantic_drift_counts_as_regression`, `test_semantic_regression_is_selected_by_kind`); mypy `Success`.

- [ ] **Step 7: Commit**

```bash
git -C $WT add src/evalshift_cli/analysis/tolerance.py src/evalshift_cli/analysis/policy.py tests/unit/test_tolerance.py
cd $WT && uv run --with pre-commit git commit -m "refactor(analysis): share the semantic within-tolerance rule

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The paired test runs on tolerance-adjusted deltas

**Files:**
- Modify: `src/evalshift_cli/analysis/slicing.py` (`SlicedScore`, `build_slices`)
- Modify: `src/evalshift_cli/analysis/statistics.py` (`WITHIN_TOLERANCE_REASON`, `_one_comparison`, `__all__`)
- Test: `tests/unit/test_analysis.py`, `tests/unit/test_policy.py`

**Interfaces:**
- Consumes: `within_tolerance(record) -> bool` (Task 1).
- Produces: `SlicedScore.within_tolerance: bool = False` (last field, defaulted); `statistics.WITHIN_TOLERANCE_REASON = "tested as no change"`.

- [ ] **Step 1: Write the failing statistics tests**

Add to the imports of `tests/unit/test_analysis.py`:

```python
from scipy import stats

from evalshift_cli.analysis.statistics import WITHIN_TOLERANCE_REASON
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION
```

(merge `WITHIN_TOLERANCE_REASON` into the existing `evalshift_cli.analysis.statistics` import block rather than adding a second one). Append at the end of the file:

```python
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
        assert c.test in ("paired_t", "wilcoxon")
        assert c.severity == "none"
        assert any(note.startswith("8 of 9 rows within") for note in c.notes)

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
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_analysis.py -q -k "WithinTolerance"`
Expected: FAIL — `ImportError: cannot import name 'WITHIN_TOLERANCE_REASON'`.

- [ ] **Step 3: Stamp the flag in the slicer**

In `src/evalshift_cli/analysis/slicing.py`, add the import `from evalshift_cli.analysis.tolerance import within_tolerance` after the `EvalRecord` import, and add the field at the end of `SlicedScore` (after `kind`):

```python
    #: The row's drop is inside its evaluator's own threshold — a semantic row
    #: above ``min_similarity`` — so the paired test reads it as no change,
    #: exactly as the policy budgets count it (see
    #: :mod:`evalshift_cli.analysis.tolerance`). ``delta`` stays raw for the
    #: aggregates. Defaulted for the same reason as ``kind``.
    within_tolerance: bool = False
```

In `build_slices`, add the keyword to the `SlicedScore(...)` construction:

```python
            kind=rec.kind,
            within_tolerance=within_tolerance(rec),
        )
```

- [ ] **Step 4: Test the adjusted deltas in `_one_comparison`**

In `src/evalshift_cli/analysis/statistics.py`:

1. After `NOT_APPLICABLE_REASON` (line 89) add:

```python
#: Suffix of the note counting rows whose drop sat inside the evaluator's own
#: threshold (a semantic row above ``min_similarity``). Those rows enter the
#: test as a delta of zero — see :mod:`evalshift_cli.analysis.tolerance` —
#: and the note is what tells a reader why a visible average drift tested as
#: no change.
WITHIN_TOLERANCE_REASON = "tested as no change"
```

and add `"WITHIN_TOLERANCE_REASON",` to `__all__` (alphabetical, after `"UNMEASURED_NOTE_PREFIX",`).

2. In `_one_comparison`, replace the opening (lines 279-287) with:

```python
    n = len(pairs)
    deltas = np.array([p.delta for p in pairs], dtype=float)
    # What the test sees: a drop inside the evaluator's own threshold is no
    # change, exactly as the policy budgets count it. ``deltas`` stays raw
    # for ``delta_avg_score``, which describes the drift rather than judging
    # it — the sign of the two means always agrees, because a forgiven row is
    # a non-positive delta set to zero.
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
```

3. Zero-variance check (line 344): `if float(np.std(tested, ddof=1)) < 1e-9:` — the body keeps `delta_avg_score=float(np.mean(deltas))`.

4. Replace the test block (lines 363-383) with:

```python
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
```

The final `return ComparisonResult(...)` keeps `delta_avg_score=float(np.mean(deltas))` (raw).

5. Update the module docstring's step 3 to `Compute Cohen's d for paired samples (mean(deltas)/std(deltas)) over the tolerance-adjusted deltas.`

- [ ] **Step 5: Run the statistics tests**

Run: `uv run pytest tests/unit/test_analysis.py -q`
Expected: all pass, including the six new `TestWithinTolerance` tests and the existing `test_zero_variance_skipped` (positional `SlicedScore` still works — the new field is last and defaulted).

- [ ] **Step 6: Write the pipeline regression test for the triage run**

In `tests/unit/test_policy.py`, add `from evalshift_cli.analysis.slicing import build_slices`, add `analyze` to the `evalshift_cli.analysis.statistics` import, add `SEMANTIC_REGRESSION` to the `evalshift_cli.evaluators.failures` import, and add `from evalshift_cli.suite.models import Suite` and `from tests.unit.suite_examples import suite_example`. Append:

```python
class TestTriageRunRegression:
    """r_20261011_triage_e14cda: every budget held and the run still failed."""

    _TARGETS = [0.9734, 0.9608, 0.9212, 0.9531, 0.9321, 0.9492, 0.9043, 0.9800, 0.9542]

    def _records(self, *, flagged: bool) -> list[EvalRecord]:
        semantic = [
            EvalRecord(
                run_id="r1",
                prompt_id="replay",
                example_id=f"ex{i}",
                evaluator_name="semantic.cosine",
                kind="semantic",
                source_score=1.0,
                target_score=t,
                delta=t - 1.0,
                metadata={"failure_categories": [SEMANTIC_REGRESSION]} if flagged else {},
            )
            for i, t in enumerate(self._TARGETS)
        ]
        judge = [
            EvalRecord(
                run_id="r1",
                prompt_id="replay",
                example_id=f"ex{i}",
                evaluator_name="llm_judge.equivalence",
                kind="llm_judge",
                source_score=0.0 if i < 7 else 0.5,
                target_score=1.0 if i < 7 else 0.5,
                delta=1.0 if i < 7 else 0.0,
                blocking=False,
            )
            for i in range(9)
        ]
        return [*semantic, *judge]

    def _decide(self, records: list[EvalRecord]) -> MigrationDecision:
        suite = Suite(examples=[suite_example(id=f"ex{i}") for i in range(9)])
        comparisons = analyze(sliced_by_slice=build_slices(records=records, suite=suite))
        return evaluate_migration_policy(
            run_id="r1",
            source_model="anthropic/claude-sonnet-5",
            target_model="anthropic/claude-sonnet-5-5",
            policy=MigrationPolicy(
                max_overall_regression_rate=0.3,
                max_critical_regressions=1,
                min_equivalence_rate=0.75,
                max_tool_argument_drift=0.2,
                max_tool_divergence=0.2,
                max_cost_increase=0.5,
                max_latency_increase=0.3,
            ),
            comparisons=comparisons,
            records=records,
            calls=[],
        )

    def test_drift_inside_min_similarity_passes(self) -> None:
        decision = self._decide(self._records(flagged=False))
        assert decision.verdict == "pass"
        assert decision.blocking_regressions == []
        assert decision.recommendations[0] == "Safe to migrate under the configured policy."

    def test_the_same_drop_below_min_similarity_still_fails(self) -> None:
        decision = self._decide(self._records(flagged=True))
        assert decision.verdict == "fail"
```

- [ ] **Step 7: Run it — it passes with Task 2's code already in**

Run: `uv run pytest tests/unit/test_policy.py -q -k TriageRunRegression`
Expected: 2 passed. (To see it fail as a check, `git -C $WT stash` the two `src/` files, rerun → `test_drift_inside_min_similarity_passes` fails with `'fail' == 'pass'`; `git -C $WT stash pop`.)

- [ ] **Step 8: Full gate, then commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest -q`
Expected: green. If any existing test pinned a semantic comparison's `statistic`/`p_value` from raw within-tolerance rows, it was asserting the bug: update its expectation to the tolerance-adjusted value and say so in the commit body.

```bash
git -C $WT add src/evalshift_cli/analysis/slicing.py src/evalshift_cli/analysis/statistics.py tests/unit/test_analysis.py tests/unit/test_policy.py
cd $WT && uv run --with pre-commit git commit -m "fix(analysis): test semantic drift against min_similarity, not raw deltas

semantic pins the source at 1.0, so any rewording is a negative delta. The
paired test ran on those raw deltas and called nine consistent 0.90-0.98
rows 'critical', failing a run whose every budget passed. Rows inside
min_similarity now enter the test as no change, matching the budgets;
delta_avg_score and the aggregates stay raw.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: A severity-driven fail says why

**Files:**
- Modify: `src/evalshift_cli/analysis/policy.py` (`evaluate_migration_policy` lines 537-541; new `_severity_reason` after `_invariant_reason`)
- Test: `tests/unit/test_policy.py`

**Interfaces:**
- Consumes: `_BLOCKING_SEVERITIES`, `BudgetResult`, `ComparisonResult` (existing).
- Produces: `_severity_reason(comparisons: list[ComparisonResult], budgets: list[BudgetResult]) -> str | None`; `MigrationDecision.reason` populated on a severity-only fail.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_policy.py`:

```python
class TestSeverityDrivenFailReason:
    def _clean_records(self) -> list[EvalRecord]:
        return [_record(example_id=f"ex{i}", delta=0.0) for i in range(30)]

    def test_a_critical_comparison_with_every_budget_held_names_itself(self) -> None:
        decision = _decide(
            self._clean_records(),
            comparisons=[
                _comparison(
                    severity="critical",
                    evaluator_name="semantic.cosine",
                    prompt_id="replay",
                ),
            ],
        )
        assert decision.verdict == "fail"
        assert decision.reason is not None
        assert decision.reason.startswith(
            "semantic.cosine regressed at critical severity on prompt replay "
            "(effect size -0.80, adjusted p 0.0100)",
        )
        assert decision.reason.endswith(
            "A critical or high regression from a blocking evaluator fails the run "
            "on its own, whatever the budgets say.",
        )

    def test_every_blocking_comparison_is_named(self) -> None:
        decision = _decide(
            self._clean_records(),
            comparisons=[
                _comparison(severity="critical", evaluator_name="a.one"),
                _comparison(severity="high", evaluator_name="b.two"),
                _comparison(severity="medium", evaluator_name="c.three"),
            ],
        )
        assert decision.reason is not None
        assert "a.one regressed at critical severity" in decision.reason
        assert "b.two regressed at high severity" in decision.reason
        assert "c.three" not in decision.reason

    def test_a_budget_breach_keeps_the_budget_line(self) -> None:
        decision = _decide(
            [_record(example_id=f"ex{i}", delta=-0.5) for i in range(30)],
            comparisons=[_comparison(severity="critical")],
        )
        assert decision.verdict == "fail"
        assert decision.reason is None

    def test_a_pass_has_no_reason(self) -> None:
        decision = _decide(self._clean_records(), comparisons=[_comparison(severity="none")])
        assert decision.verdict == "pass"
        assert decision.reason is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_policy.py -q -k SeverityDrivenFailReason`
Expected: `test_a_critical_comparison_with_every_budget_held_names_itself` and `test_every_blocking_comparison_is_named` FAIL (`assert None is not None`); the other two pass.

- [ ] **Step 3: Implement**

In `evaluate_migration_policy`, replace lines 537-541:

```python
        reason = (
            _inconclusive_reason(gating_budgets, n=overall.n_records)
            if verdict == "inconclusive"
            else None
        )
```

with:

```python
        if verdict == "inconclusive":
            reason = _inconclusive_reason(gating_budgets, n=overall.n_records)
        elif verdict == "fail":
            # A budget breach is named by the budget table and the hero's
            # "N of M" line; a fail on severity alone has neither, and read
            # "7 of 7 budgets within policy" under FAIL.
            reason = _severity_reason(overall_comparisons, gating_budgets)
        else:
            reason = None
```

Add after `_invariant_reason`:

```python
def _severity_reason(
    comparisons: list[ComparisonResult],
    budgets: list[BudgetResult],
) -> str | None:
    """Why a run failed on comparison severity alone, naming every blocking row.

    ``None`` when a budget's confirmed breach is (also) why it failed — the
    budget rows already say so — or when no comparison reached a blocking
    severity. The invariant and dropped-params reasons are applied after this
    and keep their precedence.
    """
    if any(not b.passed and b.conclusive for b in budgets):
        return None
    blocking = [c for c in comparisons if c.severity in _BLOCKING_SEVERITIES]
    if not blocking:
        return None
    named = "; ".join(
        f"{c.evaluator_name} regressed at {c.severity} severity on prompt {c.prompt_id} "
        f"(effect size {c.effect_size:+.2f}, adjusted p {c.p_value_corrected:.4f})"
        for c in blocking
    )
    return (
        f"{named}. A critical or high regression from a blocking evaluator fails the run "
        "on its own, whatever the budgets say."
    )
```

- [ ] **Step 4: Run the policy tests**

Run: `uv run pytest tests/unit/test_policy.py -q`
Expected: all pass. If an existing test asserted `decision.reason is None` on a severity-only fail, it pinned the defect: change it to assert the new sentence.

- [ ] **Step 5: Commit**

```bash
git -C $WT add src/evalshift_cli/analysis/policy.py tests/unit/test_policy.py
cd $WT && uv run --with pre-commit git commit -m "fix(policy): give a severity-driven fail a reason

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The templated narrative names the real cause

**Files:**
- Modify: `src/evalshift_cli/insights/facts.py` (`Facts`, `build_facts`)
- Modify: `src/evalshift_cli/insights/templates.py` (`_RECOMMENDATIONS`, `fallback_insight`, `_verdict_summary`, new `_recommendation`)
- Test: `tests/unit/test_insight_facts.py`

**Interfaces:**
- Consumes: `MigrationDecision.blocking_regressions[*].severity`, `BudgetResult.passed/conclusive`, `BUDGET_LABELS` (already imported in `facts.py`).
- Produces: `Facts.failed_budgets: list[str]` (display labels, default `[]`); `rendered["critical_regressions"]` = count of critical blocking regressions.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_insight_facts.py` (`decision`, `budgets` and `BlockingRegression` are already importable there — add `budgets`/`decision` to the `tests.unit.insights_factories` import if missing):

```python
def _critical_semantic() -> BlockingRegression:
    return BlockingRegression(
        prompt_id="replay",
        evaluator_name="semantic.cosine",
        slice_name="all",
        severity="critical",
        delta_avg_score=-0.05,
        effect_size=-2.15,
    )


def test_a_severity_fail_never_claims_a_budget_breach(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = decision(verdict="fail", blocking=[_critical_semantic()])
    recommendation = fallback_insight(build_facts(**sample_run)).recommendation
    assert recommendation == (
        "Do not migrate under the configured policy — semantic.cosine regressed at "
        "blocking severity although no budget was breached. Review its regressed "
        "examples, then re-run."
    )


def test_a_budget_fail_names_the_budget(sample_run: dict[str, Any]) -> None:
    breached = BudgetResult(
        name="max_overall_regression_rate",
        observed=0.5,
        allowed=0.03,
        passed=False,
        conclusive=True,
        ci_low=0.3,
        ci_high=0.7,
    )
    sample_run["decision"] = decision(
        verdict="fail",
        blocking=[_critical_semantic()],
        budget_results=[breached, *budgets()[1:]],
    )
    facts = build_facts(**sample_run)
    assert facts.failed_budgets == ["Overall regression rate"]
    assert fallback_insight(facts).recommendation.startswith(
        "Do not migrate under the configured policy — the overall regression rate "
        "budget was breached.",
    )


def test_a_fail_with_no_named_cause_points_at_the_reason(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = decision(verdict="fail")
    assert "see the verdict's reason" in fallback_insight(build_facts(**sample_run)).recommendation


def test_the_critical_count_reads_the_blocking_regressions(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = decision(verdict="fail", blocking=[_critical_semantic()])
    facts = build_facts(**sample_run)
    assert facts.rendered["critical_regressions"] == "1"
    assert "1 blocking regression (1 critical)" in fallback_insight(facts).verdict_summary
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_insight_facts.py -q`
Expected: the four new tests FAIL (`AttributeError: 'Facts' object has no attribute 'failed_budgets'`, wrong recommendation text, `"0" == "1"`).

- [ ] **Step 3: Implement the facts**

In `src/evalshift_cli/insights/facts.py`:

1. Add to `Facts` after `coverage_basis`:

```python
    #: Display names of the budgets whose breach the sample confirmed. Empty on
    #: a fail that came from comparison severity alone — which is how the
    #: templated advice tells the two causes apart instead of claiming a
    #: breach the budget table contradicts.
    failed_budgets: list[str] = field(default_factory=list)
```

2. In `build_facts`, after `coverage_basis = (...)`:

```python
    failed_budgets = [
        BUDGET_LABELS.get(b.name, b.name)
        for b in decision.budget_results
        if not b.passed and b.conclusive
    ]
```

3. Replace the `"critical_regressions"` entry in `rendered`:

```python
        # Counted from the blocking comparisons themselves. The policy metric
        # of the same name reads a per-record severity no evaluator writes,
        # so it is always 0 and once sat beside a critical blocking item.
        "critical_regressions": str(
            sum(1 for r in decision.blocking_regressions if r.severity == "critical"),
        ),
```

4. Pass `failed_budgets=failed_budgets,` in the `Facts(...)` return.

- [ ] **Step 4: Implement the template**

In `src/evalshift_cli/insights/templates.py`:

1. Delete the `"fail": (...)` entry from `_RECOMMENDATIONS`.
2. In `fallback_insight`, replace the `recommendation=_clamp(_RECOMMENDATIONS.get(...))` argument with `recommendation=_clamp(_recommendation(facts)),`.
3. Add after `fallback_insight`:

```python
def _recommendation(facts: Facts) -> str:
    """The verdict's advice, naming what actually failed the run.

    Keyed on the verdict alone, a ``fail`` once said "at least one regression
    budget was breached" over a run whose every budget held: the fail came
    from a blocking evaluator's severity. The facts carry the cause, so the
    sentence names it.
    """
    if facts.verdict != "fail":
        return _RECOMMENDATIONS.get(facts.verdict, _RECOMMENDATIONS["inconclusive"])
    if facts.failed_budgets:
        labels = ", ".join(label[:1].lower() + label[1:] for label in facts.failed_budgets)
        noun = "budget was" if len(facts.failed_budgets) == 1 else "budgets were"
        return (
            f"Do not migrate under the configured policy — the {labels} {noun} breached. "
            "Address the blocking evaluators, then re-run."
        )
    if facts.blocking_evaluators:
        whose = "its" if len(facts.blocking_evaluators) == 1 else "their"
        return (
            "Do not migrate under the configured policy — "
            f"{', '.join(facts.blocking_evaluators)} regressed at blocking severity "
            f"although no budget was breached. Review {whose} regressed examples, then re-run."
        )
    return (
        "Do not migrate under the configured policy — see the verdict's reason, "
        "address it, then re-run."
    )
```

4. In `_verdict_summary`, replace the first `parts` sentence's middle with:

```python
    blocking = rendered["blocking_regressions"]
    noun = "regression" if blocking == "1" else "regressions"
    parts = [
        f"{rendered['verdict']} under the configured policy. "
        f"{rendered['budgets_passed']} of {rendered['budgets_total']} budgets passed, "
        f"with {blocking} blocking {noun} ({rendered['critical_regressions']} critical) "
        f"across {rendered['n_examples']} examples.",
    ]
```

- [ ] **Step 5: Run the insights tests**

Run: `uv run pytest tests/unit/test_insight_facts.py tests/unit/test_insight_generator.py tests/unit/test_insights_stage.py -q`
Expected: all pass. A test that pinned the old `"… blocking and … critical regressions"` sentence or the old fail recommendation was asserting the defect — update it to the new wording.

- [ ] **Step 6: Commit**

```bash
git -C $WT add src/evalshift_cli/insights/facts.py src/evalshift_cli/insights/templates.py tests/unit/test_insight_facts.py
cd $WT && uv run --with pre-commit git commit -m "fix(insights): name the real cause of a fail and count critical honestly

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Tool evaluators skip pairs that offered no tools

**Files:**
- Modify: `src/evalshift_cli/cli/commands/evaluate.py` (`run_evaluate` lines 291-293; `_score_all` line 907; new `_offers_no_tools`, `_attempts`, `_tool_free_evaluators`, `_tool_free_warning`; comments at 1262-1264 and 1321-1324)
- Test: `tests/unit/test_evaluate_command.py`

**Interfaces:**
- Consumes: `_applies`, `_is_tool_evaluator`, `_evaluator_kind`, `_PairedCalls`, `escape` (all existing in the module).
- Produces: `_offers_no_tools(pair: _PairedCalls) -> bool`, `_attempts(evaluator: Evaluator, pair: _PairedCalls) -> bool`, `_tool_free_evaluators(evaluators: list[Evaluator], pairs: list[_PairedCalls]) -> list[Evaluator]`, `_tool_free_warning(evaluator: Evaluator, suite_name: str | None) -> str`.

- [ ] **Step 1: Let the run scaffold carry traces**

In `tests/unit/test_evaluate_command.py`, give `_scaffold_run` a `with_traces: bool = False` keyword (after `suite_name`) and pass it into each `Call(...)`:

```python
                    text=text,
                    trace=(
                        ToolTrace(calls=[ToolCall(tool_name="lookup", sequence_index=0)])
                        if with_traces
                        else None
                    ),
```

Then change `test_partial_scoping_stays_silent` to call `_scaffold_run(tmp_path, with_traces=True)` — it asserts two attempted pairs, which after this task requires pairs that were offered tools.

- [ ] **Step 2: Write the failing tests**

Add `from dataclasses import dataclass, replace` (extend the existing `dataclasses` import). Append:

```python
# ---------------------------------------------------------------------------
# Tool evaluators on pairs that offered no tools -- not applicable, not unmeasured
# ---------------------------------------------------------------------------


class TestToolEvaluatorsOnToolFreePairs:
    def _invariants(self) -> TraceInvariantsEvaluator:
        return TraceInvariantsEvaluator(TraceInvariantsEvaluatorConfig.model_validate(_INVARIANTS))

    async def _cells(self, evaluator: Any, pairs: list[_PairedCalls]) -> list[_ScoredCell]:
        return await _score_all(Console(), [evaluator], pairs, "r1", {}, concurrency=2, quiet=True)

    async def test_a_pair_that_offered_no_tools_is_never_attempted(self) -> None:
        cells = await self._cells(
            self._invariants(),
            [_pair_for("ex1"), _tool_pair("greet", "ex2", "lookup")],
        )
        assert [(c.example_id, c.record is not None) for c in cells] == [("ex2", True)]
        coverage = _coverage_for(cells)
        assert (coverage[0].attempted, coverage[0].unmeasured) == (1, [])

    async def test_a_wholly_tool_free_run_books_no_coverage(self) -> None:
        cells = await self._cells(self._invariants(), [_pair_for("ex1"), _pair_for("ex2")])
        assert cells == []
        assert _coverage_for(cells) == []

    async def test_text_evaluators_still_score_tool_free_pairs(self) -> None:
        cells = await self._cells(_SometimesUnmeasuredEvaluator(), [_pair_for("ex1")])
        assert [c.record is not None for c in cells] == [True]

    async def test_a_one_sided_trace_is_still_unmeasured(self) -> None:
        pair = _tool_pair("greet", "ex1", "lookup")
        one_sided = replace(pair, target=pair.target.model_copy(update={"trace": None}))
        cells = await self._cells(self._invariants(), [one_sided])
        coverage = _coverage_for(cells)
        assert [p.example_id for p in coverage[0].unmeasured] == ["ex1"]

    async def test_an_errored_tool_free_pair_keeps_its_error_row(self) -> None:
        pair = _pair_for("ex1")
        errored = replace(pair, source=pair.source.model_copy(update={"error": "boom"}))
        cells = await self._cells(self._invariants(), [errored])
        assert len(cells) == 1
        assert cells[0].record is not None
        assert (cells[0].record.error or "").startswith("upstream call failed")
```

And inside `class TestTraceInvariantsThatCheckedNothing`, add:

```python
    def test_a_tool_free_suite_names_the_real_cause(self, tmp_path: Path) -> None:
        _write_invariants_config(tmp_path, applies_to=["*"])
        _scaffold_run(tmp_path, suite_name="triage")

        output = " ".join(_evaluate_output(tmp_path).split())

        assert (
            "trace_invariants evaluator 'payments_contract' checked nothing: no example in "
            "suite 'triage' offered the models any tools, so there were no tool calls to "
            "check. It did not gate this run."
        ) in output
        assert "applies_to" not in output
        state = read_state(tmp_path / ".evalshift" / "runs" / "r_20260601_aaaaaa")
        assert state.evaluator_coverage == []

    def test_an_applies_to_miss_keeps_its_own_wording_on_a_tool_free_run(
        self, tmp_path: Path
    ) -> None:
        _write_invariants_config(tmp_path, applies_to=["checkout-*"])
        _scaffold_run(tmp_path)

        output = _evaluate_output(tmp_path)

        assert "['checkout-*']" in output
        assert "offered the models any tools" not in output
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/unit/test_evaluate_command.py -q -k "ToolFreePairs or ThatCheckedNothing"`
Expected: `test_a_pair_that_offered_no_tools_is_never_attempted` and `test_a_wholly_tool_free_run_books_no_coverage` FAIL (a recordless cell exists for the tool-free pair); `test_a_tool_free_suite_names_the_real_cause` FAILS (old `applies_to` wording, coverage non-empty). The rest pass already.

- [ ] **Step 4: Implement**

In `src/evalshift_cli/cli/commands/evaluate.py`, add after `_applies` (line 711):

```python
def _offers_no_tools(pair: _PairedCalls) -> bool:
    """Whether neither side of ``pair`` was offered a tool.

    The orchestrator dispatches with tools exactly when the example's resolved
    toolset is non-empty, and that path always stores a :class:`ToolTrace` on
    the call. Two error-free calls with no trace between them are therefore a
    plain text exchange, and a tool evaluator has nothing it could check — the
    same as an ``applies_to`` miss. An errored call proves nothing about the
    toolset and keeps the error path; one trace without the other stays
    unmeasured (see :func:`_score_one_tool`).
    """
    return (
        pair.source.error is None
        and pair.target.error is None
        and pair.source.trace is None
        and pair.target.trace is None
    )


def _attempts(evaluator: Evaluator, pair: _PairedCalls) -> bool:
    """Whether ``evaluator`` is asked to score ``pair`` at all.

    ``False`` means no cell: no row, and no coverage attempt that would later
    read as "unmeasured" and demote the verdict. Booking a tool-free suite as
    unmeasured is what turned a clean ``pass`` into ``conditional_pass`` for a
    top-level ``trace_invariants`` entry written for a different suite.
    """
    if not _applies(evaluator, pair.prompt_id):
        return False
    return not (_is_tool_evaluator(evaluator) and _offers_no_tools(pair))


def _tool_free_evaluators(
    evaluators: list[Evaluator],
    pairs: list[_PairedCalls],
) -> list[Evaluator]:
    """Tool evaluators whose every in-scope pair offered no tools.

    Computed from the pairs rather than from coverage: an evaluator with no
    coverage entry may also be an ``applies_to`` miss, which has its own
    warning and its own fix.

    Args:
        evaluators: Every evaluator built for the run.
        pairs: Every (source, target) pair in the run.

    Returns:
        The ``score_pair`` evaluators that checked nothing because no pair
        they were scoped to was offered a tool, in configured order.
    """
    out: list[Evaluator] = []
    for evaluator in evaluators:
        if not _is_tool_evaluator(evaluator):
            continue
        scoped = [pair for pair in pairs if _applies(evaluator, pair.prompt_id)]
        if scoped and all(_offers_no_tools(pair) for pair in scoped):
            out.append(evaluator)
    return out


def _tool_free_warning(evaluator: Evaluator, suite_name: str | None) -> str:
    """The console line for a tool evaluator that a tool-free run left nothing to check."""
    where = f"suite {suite_name!r}" if suite_name else "this run"
    kind = _evaluator_kind(evaluator) or "tool"
    return (
        f"[yellow]⚠[/yellow] {escape(kind)} evaluator {escape(repr(evaluator.name))} "
        f"checked nothing: no example in {escape(where)} offered the models any tools, "
        "so there were no tool calls to check. It did not gate this run."
    )
```

In `_score_all`, replace the work-list filter `if _applies(evaluator, pair.prompt_id)` with `if _attempts(evaluator, pair)`.

In `run_evaluate`, replace lines 291-293 with:

```python
    if not quiet:
        tool_free = _tool_free_evaluators(evaluators, pairs)
        for evaluator in tool_free:
            console.print(_tool_free_warning(evaluator, state.suite_name))
        for evaluator in _unchecked_trace_invariants(evaluators, coverage):
            if evaluator not in tool_free:
                console.print(_unchecked_trace_invariants_warning(evaluator))
```

Update the two stale comments:
- `_score_one` (lines 1262-1264): `# v0.2 — tool evaluators consume the (source, target) ToolTrace pair. A pair offered no tools never reaches here (see _attempts).`
- `_score_one_tool` (lines 1321-1324): `# Only one side carries a trace — a pair offered no tools at all never gets here (see _attempts). There is nothing to pair, so nothing to report; coverage books it unmeasured.`

- [ ] **Step 5: Run the evaluate tests**

Run: `uv run pytest tests/unit/test_evaluate_command.py -q`
Expected: all pass. Any other test that fed a traceless pair to a tool evaluator through `_score_all`/`run_evaluate` and asserted an unmeasured coverage entry was pinning the old behaviour: give it `with_traces=True` / `_tool_pair` if it is about something else, or update it to assert no cell if it is about tool-free pairs.

- [ ] **Step 6: Full gate, then commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest -q`
Expected: green.

```bash
git -C $WT add src/evalshift_cli/cli/commands/evaluate.py tests/unit/test_evaluate_command.py
cd $WT && uv run --with pre-commit git commit -m "fix(evaluate): tool evaluators do not apply to pairs offered no tools

A top-level trace_invariants entry is inherited by tool-free suites, where
every pair was booked unmeasured and the blind gate demoted a clean run to
conditional_pass with a note about 'tool calls and no prose'. Such pairs now
get no cell, like an applies_to miss, and evaluate names the real cause.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Docs and changelog

**Files:**
- Modify: `docs/methodology.md`, `docs/configuration.md`, `docs/agents.md`, `DOCS.md`, `llms-full.txt`, `CHANGELOG.md`

- [ ] **Step 1: `docs/methodology.md`**

After the severity table (before `## How policy metrics find their rows`) insert:

```markdown
### Drift inside `min_similarity` is tested as no change

`semantic` pins the source at 1.0, so every target that words anything
differently scores below it: a negative delta is the evaluator's normal output,
not evidence of a regression. Its rows therefore enter the paired test
**tolerance-adjusted** — a row within `min_similarity` (no `SEMANTIC_REGRESSION`
flag) contributes a delta of `0`, a flagged row its full drop. That is the line
the policy budgets already draw, so the two layers cannot disagree: nine rows at
0.90–0.98 once passed every budget and still failed the run as a `critical`
regression, because nine small, consistent drops have almost no variance.
`delta_avg_score` stays the raw mean — it describes the drift; severity judges
it — and the comparison carries the note `k of n rows within the evaluator's own
threshold (min_similarity) — tested as no change`. A comparison where only a few
rows are flagged may not reach significance on its own; the rate budgets
(`max_overall_regression_rate`, `min_equivalence_rate`) decide it.

### Which severities fail the run

A `critical` or `high` comparison from a blocking evaluator fails the run on its
own, even when every budget holds; `medium` and `low` downgrade a `pass` to
`conditional_pass`. When severity alone failed the run, the decision's `reason`
names each such comparison, so the report's verdict line explains the FAIL.
```

And in the materiality paragraph (line ~153-155) replace `The same reasoning is why semantic drift is counted against the evaluator's own `min_similarity` rather than against `delta < 0`.` with `The same reasoning is why semantic drift is counted against the evaluator's own `min_similarity` rather than against `delta < 0` — by the budgets and by the paired test alike (see [Drift inside `min_similarity` is tested as no change](#drift-inside-min_similarity-is-tested-as-no-change)).`

- [ ] **Step 2: `docs/configuration.md`**

Replace the paragraph at lines ~241-245 (`Semantic-evaluator drift counts toward …`) with:

```markdown
Semantic-evaluator drift counts toward `max_overall_regression_rate` and
`min_equivalence_rate` — and toward the paired test that sets each comparison's
severity — only when it breaches
[`evaluators.semantic.min_similarity`](#evaluatorssemantic). A near-identical
output (cosine ~0.98) that stays within `min_similarity` is treated as
*equivalent*, not a regression, by the budgets and the test alike — so the
policy gate, the severity and the report agree.
```

At lines ~955-957 replace `or hands the tool-free ones a tool evaluator with an empty denominator, which reads as an *inconclusive* gate rather than as "not applicable here". So a suite` with `or hands the tool-free ones tool evaluators with nothing to check — they skip every pair that was offered no tools (no row, no coverage entry, and `evaluate` warns that they checked nothing). So a suite`.

At line ~1138 replace `inherits the top level, and no tool evaluator scores an empty denominator.` with `inherits the top level; a tool evaluator it inherits skips its tool-free pairs.`

- [ ] **Step 3: `docs/agents.md`**

Lines ~119-121: replace `a tool evaluator pointed at a tool-free suite scores an empty denominator, which the policy reads as an *inconclusive* gate rather than as "not applicable here".` with `a tool evaluator pointed at a tool-free suite has nothing to check: it skips every pair that was offered no tools — no row, no coverage entry, no effect on the verdict — and `evaluate` warns that it checked nothing.`

- [ ] **Step 4: `DOCS.md`**

1. Line 381: same replacement as configuration.md ~955 (`or hands the tool-free ones a tool evaluator with an empty denominator, which the policy reads as an *inconclusive* gate rather than as "not applicable here".` → `or hands the tool-free ones tool evaluators with nothing to check (they skip every pair that was offered no tools, and `evaluate` warns that they checked nothing).`).
2. Line 514 (semantic section): replace `Within it, drift counts as *equivalent* for policy purposes.` with `Within it, drift counts as *equivalent* for policy purposes and as *no change* in the paired test that sets severity — see [methodology](docs/methodology.md#drift-inside-min_similarity-is-tested-as-no-change).`
3. Line 704 (sample guards), two insertions in the same paragraph:
   - after `Zero-variance deltas → skipped, severity `none`.` insert: ` A semantic row within `min_similarity` enters the test as a delta of 0 (noted `k of n rows within the evaluator's own threshold (min_similarity) — tested as no change`); `delta_avg_score` stays the raw mean.`
   - at the end of the paragraph (after `…the **Not enough data** the rest of `insufficient` gets.`) append: ` A tool evaluator is never handed a pair that was offered no tools at all — that pair is "not applicable", like an `applies_to` miss, not "nothing measured".`
4. Line 759: append ` When severity alone fails the run, `reason` names each blocking comparison.`
5. Line 762: replace with `- Semantic drift that stays above `min_similarity` counts as equivalent, not regression — in the budgets and in the paired test.`

- [ ] **Step 5: `llms-full.txt`**

1. Lines 409-410: `(suite inherits the top level; a tool evaluator on a tool-free suite would score an empty denominator = inconclusive gate)` → `(suite inherits the top level; an inherited tool evaluator skips pairs offered no tools: no cell, no coverage, evaluate warns "checked nothing")`.
2. Line 618: `min_similarity: float = 0.9     # below -> SEMANTIC_REGRESSION flag` → `min_similarity: float = 0.9     # below -> SEMANTIC_REGRESSION flag; above -> delta tested as 0`.
3. Line 749: `Replayed pairs with no tool trace get no row.` → `Replayed pairs with no tool trace get no row; pairs offered no tools at all get no cell (not attempted).`
4. After line 905 (`… Never "none" - unmeasured is not equivalent.`) add: `   Pairs offered no tools are not handed to tool evaluators at all (no cell, no coverage).` and a new item: `0b. Tolerance: a semantic row without SEMANTIC_REGRESSION (within min_similarity) enters the test as delta 0 (SlicedScore.within_tolerance); note "k of n rows within the evaluator's own threshold (min_similarity) - tested as no change". delta_avg_score and aggregates stay raw.`
5. Line 1126: `- Semantic drift above min_similarity counts as equivalent, not regression.` → `- Semantic drift above min_similarity counts as equivalent, not regression -- in budgets and in the paired test (severity).`; and line 1108, `- fail: any conclusive budget failure OR any blocking critical/high comparison.` → `- fail: any conclusive budget failure OR any blocking critical/high comparison (when severity alone fails the run, reason names each such comparison).`
6. Lines 1478-1479: `a tool-free suite that # inherits a tool evaluator scores an empty denominator = an inconclusive gate.` → `a tool-free suite that # inherits a tool evaluator gives it nothing to check (it skips those pairs and evaluate warns).` — keep the comment-line breaks.

- [ ] **Step 6: `CHANGELOG.md`**

Under `## [Unreleased]` add:

```markdown
### Fixed

- A `semantic` evaluator set to `blocking: true` no longer fails runs whose every row
  stays within `min_similarity`. The source is pinned at 1.0, so any rewording is a
  small negative delta, and the paired significance test ran on those raw deltas:
  nine rows at 0.90–0.98 tested as a `critical` regression and failed a run whose every
  budget passed. Rows within `min_similarity` now enter the test as no change — the
  rule the budgets already used — and the comparison notes how many. `delta_avg_score`
  and the aggregates still show the raw drift. Re-analysing an older run can lower its
  semantic severities.
- A fail caused by comparison severity alone now carries a `reason` naming the
  comparison, so the report no longer shows "7 of 7 budgets within policy" under FAIL.
  The templated recommendation names the actual cause instead of always claiming a
  budget breach, and the summary's critical count reads the blocking comparisons.
- Tool evaluators (including a top-level `trace_invariants` inherited by every suite) no
  longer attempt pairs that were offered no tools. On a tool-free suite they used to
  book every pair as unmeasured, demote a clean `pass` to `conditional_pass`, and blame
  "tool calls and no prose". They now skip those pairs, and `evaluate` warns that the
  evaluator checked nothing because no example offered tools.
```

- [ ] **Step 7: Check the doc tripwires, then commit**

Run: `uv run pytest -q -k "docs or llms or currency" && uv run pre-commit run --all-files`
Expected: green (the repo's docs-currency tests — added 2026-09-21 — pass).

```bash
git -C $WT add docs/methodology.md docs/configuration.md docs/agents.md DOCS.md llms-full.txt CHANGELOG.md
cd $WT && uv run --with pre-commit git commit -m "docs: semantic tolerance in the paired test; tool-free suites skip tool evaluators

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Verify on the real run, then PR

**Files:** none changed (a backup in the scratchpad).

- [ ] **Step 1: Back up the run, then re-score it with the branch build**

```bash
cp -r /home/lukas/repos/testApp/.evalshift/runs/r_20261011_triage_e14cda "$SCRATCH/r_20261011_triage_e14cda.bak"
cd /home/lukas/repos/testApp
uv run --project $WT evalshift evaluate r_20261011_triage_e14cda
uv run --project $WT evalshift analyze r_20261011_triage_e14cda
uv run --project $WT evalshift report r_20261011_triage_e14cda
```
`$SCRATCH` is the session scratchpad. The global `evalshift` on PATH is the PyPI 1.4.1 wheel, so it **must** be `uv run --project $WT`. `evaluate` re-embeds through the cache; if it refuses for a missing judge/embedder key, ask the user to run the three commands with `!` in their shell.

Expected:
- `evaluate` prints `⚠ trace_invariants evaluator 'agent_contract' checked nothing: no example in suite 'triage' offered the models any tools, …`.
- `analysis.json`: the `semantic.cosine` comparison has `severity: "none"`, `delta_avg_score ≈ -0.0524`, a note `9 of 9 rows within the evaluator's own threshold (min_similarity) — tested as no change`; no `agent_contract` comparison.
- `migration_decision.json`: `verdict: "pass"`, `blocking_regressions: []`, `recommendations[0] == "Safe to migrate under the configured policy."`.
- `report.html` hero: **PASS**, "7 of 7 budgets within policy".

- [ ] **Step 2: Full CI mirror**

Run: `cd $WT && make ci`
Expected: ruff, format, mypy, pytest all green; coverage ≥ 90.

- [ ] **Step 3: Code review**

Use superpowers:requesting-code-review on the branch diff `origin/main...fix/semantic-gate-tool-free-suites`; apply findings with superpowers:receiving-code-review.

- [ ] **Step 4: Push and open the PR** (only after the user approves pushing)

```bash
cd $WT && uv run --with pre-commit git push -u origin fix/semantic-gate-tool-free-suites
gh -R evalshift/evalshift-cli pr create --base main --head fix/semantic-gate-tool-free-suites \
  --title "fix: semantic drift within min_similarity no longer fails runs; tool evaluators skip tool-free suites" \
  --body-file "$SCRATCH/pr-body.md"
```
The PR body names the triage run, the before/after verdict (FAIL → PASS), the accepted consequence (a few flagged rows among many forgiven ones leave the decision to the rate budgets), the follow-ups from the spec, and ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Releasing it as 1.4.2 is a separate `chore(release): 1.4.2` PR after merge, per `CONTRIBUTING.md`.
