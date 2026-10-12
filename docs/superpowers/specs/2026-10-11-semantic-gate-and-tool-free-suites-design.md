# Semantic drift inside `min_similarity` must not fail the run; tool evaluators skip tool-free suites

**Date:** 2026-10-11 · **Status:** approved approaches A + D, for maintainer review · **Repos:** evalshift-cli (behaviour + docs); evalshift-server, evalshift-client, evalshift-action, evalshift-sdk (nothing)

## Problem

Run `r_20261011_triage_e14cda` (testApp, suite `triage`, `anthropic/claude-sonnet-5` →
`anthropic/claude-sonnet-5-5`, CLI 1.4.1) is **FAIL** although nothing in it regressed:

- every budget passes — 100% equivalent, 0% regressed, 7 of 7 budgets within policy;
- every `semantic.cosine` row is above `min_similarity: 0.9` (lowest 0.904), so none carries
  `SEMANTIC_REGRESSION`;
- yet `blocking_regressions` lists `semantic.cosine` at severity `critical` (paired t, adjusted
  p 0.0004, d −2.15), and `_verdict_for` (`analysis/policy.py:1292`) fails the run on any
  `critical`/`high` gating comparison.

**Root cause.** `semantic` pins `source_score = 1.0` (`evaluators/semantic.py:118,156`), so every
target that words anything differently has a negative delta. The paired test
(`analysis/statistics.py::_one_comparison`) runs on those raw deltas; nine small, consistent
drops (−0.02…−0.10) have tiny variance, so it calls them `critical` for *any* two models whose
text differs. `min_similarity` is applied only by the policy's per-row predicates
(`_is_regression` / `_is_equivalent`, `policy.py:675–690`) — the statistics layer never sees it.
`docs/methodology.md:155` already claims semantic drift "is counted against the evaluator's own
`min_similarity` rather than against `delta < 0`"; only half the pipeline does that. Not a
regression from a recent commit: before 1.4.1's format normalization every row sat at ~0.75,
so the rate budgets failed the run first and hid it.

**The report then contradicts itself:**

1. The hero prints "7 of 7 budgets within policy" under **FAIL** — a severity-driven fail leaves
   `decision.reason = None` (`policy.py:537–541`).
2. The templated recommendation says "at least one regression budget was breached"
   (`insights/templates.py:31–34`) — it is keyed on the verdict alone.
3. The summary says "1 blocking and 0 critical regressions" while the one blocking item *is*
   critical — `critical_regressions` counts a `metadata["severity"]` key no evaluator writes
   (`policy.py:819`).

**Second, independent defect.** The top-level `trace_invariants` entry `agent_contract`
(written for the planner suite's tools) is inherited by every suite without its own family
(`config/models.py:869–905`; the docs recommend top level, `DOCS.md:298`). On `triage` no example
offers tools, so no `Call.trace` exists (`runner/orchestrator.py:1219`), `_score_one_tool`
returns `[]` (`cli/commands/evaluate.py:1321`), and every pair is booked *unmeasured*. A blocking
evaluator that measured nothing demotes `pass` → `conditional_pass` (`policy.py:528–532`) and
emits a note written for semantic's both-empty case ("both models answered with tool calls and no
prose") — false here. So fixing the semantic gate alone still leaves this run at
`conditional_pass`. `DOCS.md:381` documents the current behaviour as "an empty denominator, which
the policy reads as an *inconclusive* gate rather than as 'not applicable here'".

## Decision

### A. The paired test sees the same "regression" the budgets do

- One shared definition, in a new `analysis/tolerance.py`: `within_tolerance(record)` is true for
  a `semantic` record that does **not** carry `SEMANTIC_REGRESSION`. The policy's
  `_is_regression` / `_is_equivalent` and the slicer both read it.
- `SlicedScore` gains `within_tolerance: bool = False`, set by `build_slices`.
- `_one_comparison` tests the **tolerance-adjusted** deltas — `0.0` for a within-tolerance row,
  the raw delta otherwise — for the zero-variance check, the normality screen, the test, Cohen's
  d and its CI. The paired t-test becomes `stats.ttest_1samp(tested, 0.0)`, which equals
  `ttest_rel(target, source)` exactly on raw deltas (verified: identical statistic and p), so no
  other evaluator's numbers move.
- `delta_avg_score` stays the **raw** mean delta: it is descriptive, the report headlines it, and
  "drifted −5% on average, within tolerance" is the honest reading. Its sign always matches the
  tested mean for semantic (both ≤ 0), so severity classification is unaffected.
- A comparison where any row was forgiven gets a note:
  `"<k> of <n> rows within the evaluator's own threshold (min_similarity) — tested as no change"`.
- Slice aggregates (`aggregates()`, `analysis.json.aggregates`) stay raw. Per-example rows,
  `scores.jsonl` and the bundle are unchanged.

Effect: this run's nine rows test as nine zeros → zero variance → severity `none` → no blocking
regression. A run whose every row sits below `min_similarity` (the pre-1.4.1 run) tests exactly
as today → `critical`. **Accepted consequence:** a mix where only a few rows are flagged (3 of 9
at 0.75) no longer reaches significance on its own; the rate budgets decide it (here 33% > 30% →
breach, `inconclusive` at n=9). That is the policy's own small-sample rule, and it is the
behaviour `methodology.md` already describes.

### Report honesty (rides with A)

- **`reason` on a severity-driven fail.** When the overall verdict is `fail`, no budget was
  conclusively breached and at least one overall gating comparison is `critical`/`high`, set
  `decision.reason` to:
  `"<evaluator> regressed at <severity> severity on prompt <prompt> (effect size <d:+.2f>, adjusted p <p:.4f>)[; …]. A critical or high regression from a blocking evaluator fails the run on its own, whatever the budgets say."`
  The hero renders `reason` already (`report.html.j2:142`). Invariant and dropped-params reasons
  keep their current precedence (they run after and override/append as today).
- **Cause-keyed fallback recommendation.** `Facts` gains `failed_budgets` (display labels of
  conclusively breached overall budgets). For `fail`: budgets breached → name them; else blocking
  evaluators → name them as regressed at blocking severity; else a generic line pointing at the
  verdict's reason. Never claim a budget breach that did not happen.
- **Honest critical count in the narrative.** `rendered["critical_regressions"]` counts
  `decision.blocking_regressions` with severity `critical`; the template sentence becomes
  `"with <b> blocking regressions (<c> critical) across <n> examples"`.

### D. Tool evaluators do not apply to a pair that offered no tools

- A pair "offered no tools" when neither call errored and neither carries a `trace`. The
  orchestrator dispatches with tools iff the example's resolved toolset is non-empty and always
  stores a trace on that path, so for error-free calls this is exact. Errored pairs keep today's
  path (error rows).
- `_score_all` builds no work item for a `score_pair` evaluator on such a pair — the same
  treatment `_applies` already gives an `applies_to` miss: no cell, no row, no coverage entry, no
  `n=0` comparison, no demotion, no misleading note. One-sided `trace is None` stays unmeasured
  exactly as today.
- `evaluate` warns once per tool evaluator whose every in-scope pair offered no tools:
  `"⚠ <kind> evaluator '<name>' checked nothing: no example in suite '<suite>' offered the models any tools, so there were no tool calls to check. It did not gate this run."`
  The existing `applies_to`-matched-nothing warning keeps its wording and only fires when the
  cause really is `applies_to`.

With A + D this run is **pass**: all budgets pass, semantic `none`, judge advisory,
`agent_contract` not attempted.

## Not in scope (recorded follow-ups)

- `max_critical_regressions` is inert: `_metrics` counts `metadata["severity"]`, which nothing
  writes. Fixing it changes what the hosted gate reads (`evalshift-server
  app/policy/service.py:366`), so it gets its own cross-repo change.
- `severity_floor` metadata written by `tool_selection` / `tool_trace_structure` is read by
  nothing.
- `tool_arguments` against `source` has the same pinned-source shape (`tool_argument_drift_floor`
  applies only to the rate). `within_tolerance` is the hook to extend later.
- With `samples_per_example > 1`, a reduced semantic row keeps the *lead* sample's
  `failure_categories` (`evaluate.py::_reduce_one_group`) while its scores are means, so its
  `SEMANTIC_REGRESSION` flag can disagree with its mean score. Pre-existing, and it already
  affects the budgets the same way; A makes the test inherit it rather than fix it.
- The "both models answered with tool calls and no prose" wording in
  `_recommendations.nothing_measured` is still a guess for other unmeasured causes; after D it no
  longer fires for tool-free suites.

## Docs that change

`docs/methodology.md` (severity section: tolerance-adjusted test; which severities fail the run),
`docs/configuration.md` (semantic/policy paragraph ~l.241; per-suite evaluators ~l.955),
`docs/agents.md` (~l.118), `DOCS.md` (l.381, 514, 704, 759–762), `llms-full.txt` (l.409, 618,
749, 903–905, 1116–1127, 1479), `CHANGELOG.md` `## [Unreleased]`.

## Compatibility

No config, schema, bundle or `state.json` change. `analysis.json` severities for semantic
comparisons drop where rows sit within tolerance; old runs re-analysed with the new CLI get the
new answer (intended). A tool-free suite's `state.json` no longer lists tool evaluators in
`evaluator_coverage`. The `--gate critical,high` CLI gate follows the new severities.
