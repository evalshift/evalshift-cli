# Methodology

How EvalShift decides whether a model swap caused a regression.

## The shape of the problem

For each (prompt × example) pair, EvalShift produces two outputs:
`source_output` (from the old model) and `target_output` (from the
new one). Each evaluator scores both halves on a 0–1 scale, and the
quantity we care about is the per-example **delta**:

```
delta_i = target_score_i - source_score_i
```

Negative deltas mean the target lost ground; positive deltas mean it
gained. A "regression" is a slice where the deltas are systematically
negative across many examples.

## Slice deduplication

Slices come from example `tags`, and tags overlap. A suite built entirely
by `evalshift capture promote` tags every example `["captured", <suite>]`,
so both tags describe exactly the same examples — and if the run covers the
whole suite, the same examples as the implicit `all` slice too.

Duplicate slices are not free. They restate the same numbers in the report as
if they were independent findings, produce duplicate slice verdicts in the
migration decision, and — less visibly — corrupt the Benjamini–Hochberg
correction below, whose family is supposed to be the set of *distinct*
hypotheses.

The corruption is anti-conservative, not conservative. Duplicating a p-value
`k` times raises both the family size `n` and the rank the copies reach, and
`(n + k) / (r + k) < n / r`, so every adjusted p-value in the family comes out
*smaller* than it should. Results look more significant than they are, and
severity — which keys off the adjusted p-value — can be classified a step too
high. (When every test in the family is duplicated the same number of times,
the ratios cancel and nothing moves; the damage shows up when duplicated and
non-duplicated comparisons share a family.)

So before any test runs, slices holding exactly the same
`(prompt, evaluator, example)` triples are collapsed to one. Among
identical slices:

* `all` always survives — it is the overall scope.
* Any slice named under `migration_policy.slices` always survives, so
  deduplication can never quietly turn a budget you wrote into a no-op.
* Otherwise a provenance tag (`captured`) loses to an ordinary tag, and
  alphabetical order breaks the remaining ties.

Whatever was dropped is reported: a line on the terminal
(`slices: dropped captured, project_insights (identical to all)`) and a
`collapsed_slices` map in `analysis.json`. A suite whose tags all cover
every example ends up with no slice section at all — correctly, because it
has no subsets to compare.

## Per-comparison testing

For each `(prompt, evaluator, slice)` triple we run one paired test
on the deltas:

1. **Sample-size guards.**
   * `n < 5`: no test runs; severity is `insufficient`.
   * `5 ≤ n < 20`: test runs but the result carries an "uncertain"
     note in the report.

2. **Normality screen.** Shapiro-Wilk on the deltas. If the screen
   passes (`p > 0.05`), we run a paired t-test. Otherwise we fall
   back to a Wilcoxon signed-rank test, which only assumes
   symmetric distributions. (For `n > 5000`, Shapiro-Wilk is
   over-powered, so we skip the screen and use the t-test by CLT
   grounds.)

3. **Effect size.** Cohen's d for paired samples:
   ```
   d = mean(deltas) / std(deltas, ddof=1)
   ```
   Sign of `d` tracks the sign of `mean(deltas)` (negative = regression).

4. **95% CI on the effect size.** Analytical (`d ± 1.96 × SE`)
   when the t-test ran; percentile bootstrap (2000 resamples) when
   Wilcoxon ran.

## Multi-test correction

The whole point of slicing is to find regressions in *some* subset
of the data, but every additional comparison inflates the family-wise
false-discovery rate. We control it with **Benjamini–Hochberg** at
FDR α = 0.05 across every `(prompt × evaluator × slice)` p-value
in the run.

We pick BH over Bonferroni because BH controls the same error rate
with substantially more power, especially when several comparisons
are real (the typical case in a real eval suite).

The implementation matches `statsmodels.stats.multitest.multipletests(method='fdr_bh')`
exactly — verified against a known-result test in the suite.

## Severity classification

After BH correction, each comparison gets a severity tag based on the
*adjusted* p-value and the absolute effect size:

| Severity        | Rule                                                  |
| --------------- | ----------------------------------------------------- |
| `critical`      | adjusted p < 0.01 AND \|d\| > 0.8 AND mean(delta) < 0 |
| `high`          | adjusted p < 0.05 AND \|d\| > 0.5 AND mean(delta) < 0 |
| `medium`        | adjusted p < 0.05 AND \|d\| > 0.2 AND mean(delta) < 0 |
| `low`           | adjusted p < 0.05 AND mean(delta) < 0                 |
| `improved`      | adjusted p < 0.05 AND mean(delta) > 0                 |
| `none`          | adjusted p ≥ 0.05                                     |
| `insufficient`  | n < 5                                                 |

## How policy metrics find their rows

`migration_policy` budgets are computed over subsets of the scored records —
`max_tool_argument_drift` only looks at tool-argument rows,
`max_tool_divergence` only at `tool_selection.divergence` rows,
`max_invariant_violations` only at `trace_invariants` rows, and the
semantic regression rule only applies to semantic ones. That selection is made on each
record's **evaluator kind** (`tool_arguments`, `tool_selection.conformance`,
`tool_selection.divergence`, `trace_invariants`, `semantic`, `llm_judge`,
`structural`, …), which is
a property of the evaluator's type, not of what you called it. An evaluator that
measures more than one thing gets a slug per measurement — `tool_selection`
scores ground-truth conformance and target-vs-source divergence, and they are
separate comparisons with separate baselines, so they must never share a
selector.

So renaming an evaluator in `evalshift.yaml` is safe:

```yaml
tool_arguments:
  - name: routing_args      # any name you like — the budget still binds
```

Records written before `kind` existed fall back to the old
`<kind>.`-prefixed-name selection, so older runs re-analyse identically.

### Rate budgets over continuous scores need a materiality floor

`max_tool_argument_drift` is a *rate*: the share of tool calls whose arguments
drifted. But the underlying evaluator returns a continuous score, so "drifted"
has to be defined before the rate means anything. Counting every negative delta
defines it as "differs at all", which weighs a `0.98` exactly like a `0.0` — and
two different models never produce byte-identical arguments, so the rate
saturates and the shipped `0.01` budget is unreachable by construction.

`migration_policy.tool_argument_drift_floor` (default `0.9`) is that definition:
a call counts as drifted only when its target score falls below the floor. The
rate then measures what its name promises — the fraction of tool calls whose
arguments are *materially* wrong — and the budget becomes something you can act
on. The same reasoning is why semantic drift is counted against the evaluator's
own `min_similarity` rather than against `delta < 0`.

### A zero delta both models earned by failing is not equivalence

The conformance axis grades each side *absolutely* against the ground truth the
suite recorded, so both sides can miss it at the same height: `0.0 / 0.0` on an
example both models answered with a tool call the recording never made. The
delta is zero, and a zero delta is what "equivalent" means everywhere else in
this module — which is how a real run reported `equivalent_rate: 1.0` over a
suite where nine pairs in ten called entirely different tools.

The fix is not a different equivalence rule but a different *denominator*.
Ground truth captured from the source model that the source model then fails is
evidence about the harness — the wrong toolset attached, the wrong prompt, a
suite promoted from a different agent — and carries nothing about the
migration. Conformance rows the evaluator flagged `TOOL_GROUND_TRUTH_MISS`
whose delta is exactly zero are therefore excluded from every policy rate and
from `n_records`.

Only the shared-height case goes. A conformance row the target lost ground on
(`0.8 / 0.3`) is a real regression the migration caused, and one it improved on
(`0.2 / 0.6`) is a real improvement; both stay. Divergence rows are never
excluded — that axis has no ground truth to miss.

Excluded is not invisible: the rows keep their `TOOL_GROUND_TRUTH_MISS` count
in `failure_categories`, and `analyze` adds a recommendations line naming how
many left the rates and why. A run whose *every* blocking row was a shared miss
reports `inconclusive` with that as its reason, rather than the "every
evaluator is advisory" reason, which would be false — and its recommendation is
**Fix the eval harness before collecting more examples**, because more pairs
from the same setup are more excluded rows and the denominator stays empty
however many are added.

### A source model that fails its own ground truth

The exclusion above is deliberately narrow — it needs `delta == 0` and both
sides tagged — because its job is to protect the rates, and a row carrying real
signal must not be thrown away with the rest. The *diagnosis* is wider than
that, so `evaluate` asks its own question, over the **source side alone**:
on what fraction of the conformance rows did the source model fail?

The two are not the same count. A suite the source fails at `0.0` and the
target happens to satisfy at `1.0` is a positive delta with no
`TOOL_GROUND_TRUTH_MISS` tag — invisible to the exclusion rule, and exactly as
misconfigured. Grading the source alone is what catches it: the expectations
were recorded *from* the source model, so the source is the one side that
should always meet them.

At **half** the conformance rows or more, `evaluate` reports a broken eval
harness at `doctor` volume — a red row naming the rate and the likely causes —
because half is where the ground truth has stopped describing the source model
at all, and a coin-flip rate is not sampling noise. Under half the rows stay a
finding rather than an accusation.

The check needs at least **four** conformance rows. A 100% failure rate over
three has a 95% Wilson lower bound of `0.44` — under half — so a three-example
smoke suite cannot support the sentence the check would print; at four rows the
bound is `0.51` and it can. Below that the check is silent rather than guessing.
The verdict itself does not move: the conformance rows are already out of the
rates, and the divergence axis may still have measured a real migration. What
moves is the order the run is read in — the finding prints immediately above the
verdict it invalidates.

### A rate budget finer than one row is a zero-tolerance budget

A rate counted over `n` rows can only land on multiples of `1/n`. On a ten-row
starter suite the achievable tool-argument drift rates are `{0.0, 0.1, 0.2, …}`,
so a budget of `0.01` is not "1% tolerance" — the first representable non-zero
value already breaches it. The budget is silently equivalent to `0.0`, and it
lands hardest on new users, whose suites are smallest.

The gate is not wrong and no default changes: `0` drift still passes cleanly and
a real breach is still a real breach. What was missing was anyone saying so.
`analyze` now adds a line to the decision's recommendations — rendered in the
terminal and in the HTML report — whenever a rate **ceiling** sits below one
step of its own denominator:

```
The tool-argument drift budget of 1% (max_tool_argument_drift in
evalshift.yaml) is below the 10% granularity of 10 tool-argument
comparisons — effective tolerance is zero at this sample size.
```

It applies to every rate ceiling with a row denominator, each judged on its own:
`max_overall_regression_rate` over the scope's scored records,
`max_tool_argument_drift` over its tool-argument rows, and
`max_tool_divergence` over its tool-divergence rows. Slices are checked
against their own row counts, which are coarser still, and named in the warning.

Three cases stay silent. A budget of exactly `0.0` is a deliberate
zero-tolerance choice, not a mistake. A denominator of `0` measured nothing at
all, which `BudgetResult.conclusive` already reports. And `min_equivalence_rate`
is excluded because it is a *floor*: below one row's granularity it collapses to
maximally lax — only a 0% rate could fail it — so "effective tolerance is zero"
would be the opposite of the truth. The count budgets
(`max_critical_regressions`, `max_invariant_violations`) are not rates, and the
cost/latency ratios have no row denominator, so `1/n` describes none of them.

Widening the budget is not the fix, and neither is a confidence interval: at 1
material drift in 10 the 95% Wilson lower bound is `0.0179`, still above `0.01`,
so the breach is statistically confirmed. The fix is more rows — or a budget
written at a size the suite can express.

### A cost ratio computed from zeroes is not a measurement

`max_cost_increase` and `max_latency_increase` are ratios of two averages, so
unlike the rate budgets they have no row denominator — their denominator is the
*source average*, and that can itself be zero. The ratio falls back to `0.00`
in two different situations, and `0.00` clears every budget:

- **No error-free call on one side.** Every target call errored, or the run has
  no calls at all. Reported as `conclusive: false`.
- **Both sides average zero.** The models are not priced (LiteLLM returns
  `cost_usd: 0.0` for an id it has no pricing for), or every call reported
  `latency_ms: 0` for the same reason.

Only the first was caught originally: the call lists in the second are
non-empty, so a pairing check says "measured" and the row renders
`observed 0.00, allowed 0.20, passed, conclusive` — a confident statement that
the target does not cost more, about a cost nobody ever priced. Both are now
`conclusive: false`.

`Call.cost_usd` is `0.0` both when a model is unpriced and when it is genuinely
free, and no other field in the record separates them, so this deliberately
marks a genuinely-free pair as unmeasured too. That trade is one-sided: a false
confident pass on an unpriced migration is the failure mode worth avoiding, and
it is not recoverable by the reader, while "could not tell" is — price the
models, or read the raw calls.

Because the flag alone cannot distinguish the two situations for a reader,
`analyze` adds a line to the recommendations for the second one:

```
The cost increase budget could not be measured: all 4 error-free calls
across both models recorded a cost of 0, so its observed 0.00 is a default,
not a measurement.
```

The first situation stays silent: an empty `raw.jsonl` beside a
`conclusive: false` already explains itself, the same don't-double-report rule
the granularity warnings follow. The note is emitted once per run rather than
once per scope, since slices are evaluated against the same run-level calls.

Nothing else moves. The row still observes `0.00`, which clears its budget, so
the verdict is unchanged; `conclusive` is the only field that differs.

### Every budget reports the sample behind it

`observed: 0.0` is written by three different situations — nothing was ever
counted, something was counted and the answer really is zero, or the value is a
fallback for a quantity this run could not observe. `conclusive` separates the
last from the first two, but it does not say *how much* was counted. Each
`BudgetResult` therefore also carries a `denominator`:

| Budget | Counted over |
| --- | --- |
| `max_overall_regression_rate` | the scope's scored records |
| `min_equivalence_rate` | the same records — non-regression is the exact complement |
| `max_critical_regressions` | the same records |
| `max_tool_argument_drift` | the scope's `tool_arguments` rows only |
| `max_tool_divergence` | the scope's `tool_selection.divergence` rows only |
| `max_invariant_violations` | the distinct examples (`prompt_id`, `example_id`) with a blocking `trace_invariants` row in the scope — one per example however many entries scored it; the row is emitted only when there is at least one |
| `max_cost_increase` / `max_latency_increase` | the error-free calls both averages were taken over, across both roles |

The first three are counted over *measurements*, not over examples. An
evaluator that scores several axes contributes a row per axis per example —
`tool_selection` contributes two — and that is correct: conformance and
divergence ask different questions against different baselines, and a
regression on either is a regression. The two per-axis budgets below them keep
their own rows, so neither moves when the other axis is switched on or off.

Slices report their own counts, not the run's. `0` means the budget was
evaluated against an empty sample, so `observed` is a default and `passed` is
vacuous — the same fact `conclusive: false` already carries, now with the count
that produced it. It is one number per budget, computed once and used both to
fill this field and to judge the `1/n` granularity warning above, so a budget
can never be reported against a sample it was not judged on.

`denominator` and `conclusive` answer different questions, and the cost budgets
are where they visibly disagree: a run whose calls all report `cost_usd: 0`
counted every one of them and still measured nothing, so it reports a positive
denominator beside `conclusive: false`. Reading either field as a stand-in for
the other loses that.

The field exists for the hosted gate. `GET /runs/{run_id}/policy-check`
re-decides an uploaded run against the project's *current* policy and never saw
the underlying records, so without a sample size it cannot tell a clean run from
an empty one. Bundles written before the field omit it entirely, and a missing
`denominator` means **unknown, not zero** — the server falls back to
`conclusive` for those. This CLI always knows its own denominators, so every
budget it emits carries an integer.

### Every proportion budget gets an interval, and both engines agree on which

A budget whose `observed` is a *count of records over a count of records* is a
binomial proportion, and only those get a confidence interval. There are four:

| Budget | Proportion of |
| --- | --- |
| `max_overall_regression_rate` | regressed rows over the scope's scored rows |
| `min_equivalence_rate` | the exact complement of the above, over the same rows |
| `max_tool_argument_drift` | materially drifted rows over the scope's `tool_arguments` rows |
| `max_tool_divergence` | diverged rows over the scope's `tool_selection.divergence` rows |

`max_critical_regressions` and `max_invariant_violations` are raw counts and
the cost/latency budgets are ratios of two averages; none describes a
proportion, so none is given an interval and all report `ci_low`/`ci_high` as
`null`.

Each of the four carries a 95% Wilson score interval — Wilson rather than the
normal approximation because these samples are small and these rates sit near 0
or 1, precisely where the textbook interval produces bounds outside `[0, 1]` and
claims certainty it has not got. The rule the interval feeds is deliberately
**asymmetric**:

- **Breached, and the favourable bound clears the budget** → the sample confirms
  the breach. `conclusive: true`, and the verdict is `fail`.
- **Breached, but the interval still spans the budget** → the sample cannot tell.
  `conclusive: false`, and the verdict is `inconclusive`, not `fail`.
- **Held** → `conclusive: true`, whatever the interval does. A wide interval
  around a clean observation is not evidence of harm, and letting it downgrade a
  passing run would punish exactly the small suites this tool asks people to
  start with.

The drift budget was the last one to get an interval. It had none while the
bundle reported no drift denominator; once it did, the hosted gate — which has
always counted drift as a proportion — began confirming breaches the CLI still
called outright failures, so one run could read `fail` locally and
`inconclusive` hosted off the CLI's own numbers. Both engines now compute the
interval over the same three budgets — regression rate, equivalence rate and
tool-argument drift; the CLI's fourth, `max_tool_divergence`, has no hosted
interval yet — from the same two-sided 95% quantile
(`1.959963984540054`, not the rounded `1.96`), and apply the same asymmetric
rule, so a local verdict and a hosted one no longer disagree about whether a
breach was confirmed.

The consequence, accepted deliberately: on a thin sample a drift breach whose
lower bound does not clear the budget is no longer a confident local failure. It
is reported `inconclusive` — which is what it always was statistically — and the
`1/n` granularity warning above still fires to say the sample is too coarse to
express the budget.

### A broken trace rule is a finding, not a statistic

`max_invariant_violations` counts the distinct examples on which the target
broke a blocking [`trace_invariants`](configuration.md#evaluatorstrace_invariants)
rule — any `(prompt_id, example_id)` with a blocking `trace_invariants` row whose
target score is below `1.0`, which includes a row averaged over repeated samples
where one sample broke a rule. Each entry writes its own row, so an example that
broke rules in two entries is still one example, over a denominator of distinct
examples rather than rows. It
is judged against the rules alone, so a rule the source broke on the same
example still counts.

Every other budget estimates a rate the next run could land on either side of;
this one reports an assertion the team wrote down and the target falsified.
There is nothing to be uncertain about, so the budget is **conclusive by
construction**: no interval, no `1/n` warning, and a breach is a `fail` that
overrides `inconclusive` and `conditional_pass` — including the
"every comparison is `insufficient`" path a small suite otherwise takes, and in
a slice exactly as overall. The decision's `reason` names the worst scope and
its count. When the source broke the same rule on the same example,
`recommendations` adds a line naming the rule: either both models break it or
it no longer matches the toolset, and only its owner can say which. That line
is advice, never an exemption.

The row exists only in scopes that scored at least one blocking
`trace_invariants` row. A project with no trace rules, or only advisory ones,
sees no new budget row, and a slice that holds no in-scope example has no row
to breach.

### Non-applicable measurements are absent, not scored

A turn where **both** models produced only tool calls has no text to compare.
`semantic` and `llm_judge` return nothing at all instead of calling their
provider — an embedding endpoint 400s on empty input, and a judge shown two
empty strings returns a meaningless tie. No row is written, so a
non-applicable measurement cannot masquerade as agreement between the models
in `scores.jsonl`, the report, or the hosted bundle.

Absence still has to be *counted*, or an evaluator that measured 2 of 10 pairs
would report a rate over a denominator of 2 with the other 8 invisible. The
`evaluate` stage records, per evaluator, how many pairs it attempted and how
many produced a row, into `state.json` under `evaluator_coverage`; the
analysis layer reads it back to keep the "K of N rows not applicable" note and
to keep an evaluator that measured *nothing* in the analysis as
`severity: insufficient` rather than letting it vanish.

One empty side is *not* skipped: a target that went silent where the source
answered is a real regression, and the evaluator still scores it. The two
families score it differently, because only one of them can measure it.
`llm_judge` sends the pair to the judge as usual — a chat model can compare
prose against silence. `semantic` cannot: the embedding endpoint 400s on the
empty side, so it assigns similarity **0.0 by definition** without calling the
provider at all. That score passes through the normal `min_similarity` gate
(so a `min_similarity: 0.0` config still un-flags it), and the record carries
`empty_side: "source" | "target"` in its metadata plus a written explanation,
so the report says "produced no text" rather than deriving a misleading
"reworded the content (0% similar)" sentence from the raw cosine. Both
directions get the same treatment: a target that answered where the source was
silent is the same total divergence, just mirrored.

A skipped row is then excluded everywhere downstream: from the paired test and
the slice aggregates, and from the policy metrics that feed `equivalent_rate`.
A comparison with no applicable rows left is reported `insufficient` with a
`nothing measured:` note, never `none` — and a run whose blocking evaluators
all fell into that state cannot return `pass`. It is downgraded to
`conditional_pass` (or stays `inconclusive`) with those evaluators named in
`recommendations`, because an evaluator that compared nothing is unknown, not
equivalent. Advisory (`blocking: false`) evaluators are exempt: their silence
gates nothing by design, so it neither demotes the verdict nor lands in
`recommendations`. Zero rows leave nothing in `scores.jsonl` carrying the
config flag, so `evaluator_coverage` records it per axis and the synthesized
comparison carries an `advisory:` note — riding in `notes`, like the
`nothing measured:` marker, so the hosted verdict reads the same set. In `report.html` that row reads **Nothing measured** rather than the
**Not enough data** headline the rest of `insufficient` carries — the sample was
absent, not small.

## What we always report

Every comparison row in `analysis.json` and `report.html` carries:

* raw p-value
* BH-adjusted p-value
* Cohen's d and its 95% CI
* sample size
* test used (`paired_t`, `wilcoxon`, or `skipped`)
* severity classification

You will never see a naked "regression detected" without the numbers
backing it up.

## Limitations to be aware of

* **Score scales must be comparable across (source, target).** Most
  evaluators are inherently per-output (structural, judge), so
  this is fine. The semantic evaluator deliberately frames source
  as 1.0 to make deltas interpretable, but that means the absolute
  similarity is the signal — interpret with care.

* **Wilcoxon's assumption is symmetry.** If your delta distribution
  is asymmetric (e.g. small wins, occasional huge losses), Wilcoxon's
  null can be off. The test still flags real shifts but the p-values
  may be approximate.

* **BH controls expected FDR, not per-test type-I error.** A few
  false positives are expected when many tests run. Use the effect
  sizes and CIs to triage.

* **Per-call sampling variance.** By default EvalShift makes one call
  per (prompt, example) per model, at `temperature=0` (the registry
  default). If a model samples freely anyway (the report says so in a
  banner), set `defaults.samples_per_example` above 1. Every example is
  then sent to each model N times; sample *i* of the source is scored
  against sample *i* of the target; and the example's row in
  `scores.jsonl` is the **mean** of its samples' `source_score`,
  `target_score` and `delta`, with the per-sample values and the
  population `delta_variance` recorded under `metadata.samples`. The
  paired tests above run over those per-example rows, so `n` stays the
  number of examples: repeated sampling reduces the noise in each
  delta, it does not manufacture power. Read a large `delta_variance`
  next to a small mean delta as "this example is noisy", not "this
  example regressed".

* **Some models no longer honour `temperature`.** Every paired test
  here assumes the only difference between the two arms is the model. That
  assumption depends on sampling being pinned, which EvalShift does by
  sending `temperature=0` on every call. Google has announced the removal of
  `temperature`, `top_p`, and `top_k` for Gemini 3+, and other providers may
  follow. When a model stops accepting the parameter, LiteLLM drops it and
  sampling reverts to the provider default — silently, because the call still
  succeeds. Each arm is checked against LiteLLM at run start, so this case is
  caught before the first call.

  A second failure mode surfaces at call time instead: reasoning-tier models
  (for example `gpt-5.6-terra`) advertise `temperature` but reject every value
  except their own default with a 400, and `drop_params` does not cover them —
  LiteLLM special-cases only o-series names. The first rejection makes
  EvalShift resend that call without `temperature` and stop sending the
  parameter to that model for the rest of the process, judge models included.
  One call per affected model fails and is resent adapted, so no measurement is
  lost, but sampling on that arm is the provider default rather than pinned.

  Either path records the affected model in `state.json` under
  `non_deterministic_models` — from the run-start probe, or merged in when the
  run phase (and, for judge models, scoring) finishes. The report then carries
  a banner above the verdict and a matching methodology note. Treat those runs
  as measuring *model change plus sampling noise*: differences are still
  real when large, but non-significant results are much weaker evidence than
  usual, and the honest fix is more examples or repeated trials rather than a
  tighter p-value.

  EvalShift does **not** compensate by moving sampling guidance into the
  system prompt. Doing so would change the prompt under test, and change it
  for one arm only — confounding the very comparison it was meant to
  protect.

* **A replay can lose constraints other than sampling.** A promoted capture
  may pin `response_format`, `tool_choice`, `parallel_tool_calls`, `top_p` or
  a completion cap, and `drop_params=True` lets a model that never accepted
  one of them answer anyway — minus the constraint. EvalShift probes both arms
  at run start for exactly the parameters the suite recorded (provider
  spellings mapped to their OpenAI names first; `temperature` belongs to the
  bullet above) and writes the shortfall to `state.json` under
  `dropped_params`, with one `WARNING` per (model, parameter) and a
  **Constraints not honoured** banner beside the sampling one. Uncertainty
  from LiteLLM records nothing, on the same reasoning as the sampling probe.
  The probe has one blind spot, covered by a hard-coded table: a parameter
  LiteLLM reports as supported and then drops while building the provider's
  request body. Two Gemini cases are known — `parallel_tool_calls`, and a
  tool's `strict` flag, recorded as **`tools.strict`** because it is a field
  on the `tools` array rather than a generation parameter — and they land in
  the same `dropped_params` record.
  Read such an arm as *model change plus a missing constraint*: a difference
  there is not attributable to the model alone, so either accept the caveat,
  pick a target that supports the constraint, or set
  `migration_policy.fail_on_dropped_params: true` to make the run fail rather
  than explain itself.

* **Truncated outputs are excluded; empty outputs are not.** A call cut
  off at the `max_tokens` cap (`finish_reason == "length"`) is dropped
  from the paired statistics — a cut-off output is a measurement artefact,
  not a model verdict, so scoring it would manufacture a false regression.
  An **empty-output** call (no visible text but `output_tokens > 0`, no
  error, and a normal `"stop"` finish reason — typically a thinking-only
  response with nothing surfaced) is the opposite case: the model genuinely
  produced nothing usable, so its score stands and counts toward the
  regression. Tool-call responses are never counted as empty output — an
  agent that only calls a tool legitimately has no visible text, so that's
  excluded, along with any call whose finish reason isn't `"stop"` (e.g. a
  truncated call already handled above). The report flags the remaining
  calls (`empty_output_calls` per role, and a per-regression marker) so you
  can tell "the model failed" apart from "the run was cut short", but
  neither one is silently dropped from what you see.
