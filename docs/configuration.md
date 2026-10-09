# Configuration reference

Every EvalShift run is driven by a single `evalshift.yaml` file. This
page documents every field — types, defaults, and what they do.

`evalshift init` writes a minimal, capture-first `evalshift.yaml` —
a passthrough `replay` prompt, default models for the provider picked with
`--provider gemini|openai|anthropic|deepseek` (default `gemini`), evaluators, and an empty
managed `suites:` block you fill in with `evalshift capture sync`. Below is
the canonical reference.

## Top-level shape

```yaml
version: 1                # optional (default 1); must be 1 when set
project: org/project      # optional, required for hosted push unless passed by flag
migration_policy: {...}   # optional local migration verdict policy
prompts: [...]            # required, at least one
defaults: {...}           # optional
evaluators: {...}         # optional (but at least one is needed for `evaluate`)
suites: {...}             # optional named suites for `run --suite-name`, each with
                          # its own optional `evaluators:` block
retention: {...}          # optional run-history pruning policy
captures: {...}           # optional, see below
```

Unknown keys are rejected (`extra: forbid` everywhere) so typos fail
fast instead of silently dropping.

## Config version policy

`version: 1` changes when a config that is still *valid* would be read with the
wrong meaning by the wrong CLI — a field renamed, or redefined to mean
something else. Additive fields (a new evaluator option, a new top-level block)
do *not* bump it; they ride on the CLI version instead.

Nor does removing a field, as long as a config that still sets it **fails to
load and says why**. The literal exists to catch silent misreadings, and an
error naming the removed key is the opposite of silent — it cannot be mistaken
for a config that still works. `thresholds` left in 1.1.0 that way, and the
top-level `slices` list after it; `version` stayed `1` both times. Bumping it
would have forced an edit on every config, including the majority that never
set the key.

Because unknown keys are rejected, that puts one rule on you: the CLI that
*reads* a config must be at least as new as the CLI that *wrote* it. In
practice that means the `evalshift-version` your CI workflow installs must be
at least the version you run `capture sync` and `init` with locally. The CI
pin check is the mechanism that enforces it: `capture sync`, `init`, `doctor`,
and `validate` warn when a workflow under `.github/workflows/` pins an older
CLI, or none at all, and print the exact line to set. They also warn when
every pin is newer than the local CLI. That does not break the rule — CI
reads with the newer CLI — but local runs then disagree with CI, and the fix
printed is `pip install -U evalshift`. See
[Pin drift](github-action.md#pin-drift).

## `project`

This field is used only by hosted commands. Local `run`, `evaluate`,
`analyze`, and `report` do not require it.

| Field        | Type   | Required | Description |
| ------------ | ------ | -------- | ----------- |
| `project`    | string | no       | Hosted project slug in `org-slug/project-slug` form. `evalshift push` also accepts `--project`, which overrides the config value. |

Example:

```yaml
project: acme/model-migration
```

`evalshift init` scaffolds `project:` **commented out**, with the slug shape and
a placeholder — uncomment it once you have a hosted project. It is left unset
because a local run never needs it and nothing uploads without an explicit
`push` / `compare --push`, so a guessed slug would be wrong in the one place it
matters.

### `thresholds` was removed

`evalshift.yaml` used to accept a free-form `thresholds:` block next to
`project:`. It was never read by anything that gates: the migration verdict
comes from [`migration_policy`](#migration_policy), and the hosted gate reads
the resolved policy the bundle carries. Nothing replaced it.

A config that still sets the key now **fails to load**:

```text
`thresholds` was removed: it was free-form and gated nothing. Delete it from evalshift.yaml; migration_policy is the single source of truth for gating.
```

The fix is to delete the block. If you were using it to express a gate, encode
that as a `migration_policy` budget instead.

### Top-level `slices` was removed

`evalshift.yaml` also used to accept a top-level `slices:` list (`name`,
`filter`, `applies_to`). It was validated and recorded in the run bundle, but
analysis never read it: slices come from example `tags` (see
[Slices](#slices)), so none of its fields renamed, filtered or scoped anything.

A config that still sets the key now **fails to load**:

```text
`slices` was removed: it never had any effect. Slices come from example `tags` automatically (one per distinct tag, plus `all`). Delete it from evalshift.yaml; per-slice budgets go under migration_policy.slices, keyed by tag.
```

The fix is to delete the block; a run reports the same slices without it. For
per-slice budgets, use [`migration_policy.slices`](#migration_policy), keyed
by tag.

Hosted baselines are unaffected for any config that never set the key (or set
`slices: []`): the bundle's evaluator config still carries an empty `slices`
list, so `eval_config_hash` does not move. Deleting a *non-empty* block does
change that hash: runs pushed afterwards are not comparable to baselines pushed
before, until the base branch pushes a run with the edited config.

## `migration_policy`

Optional local regression budget used by `evalshift analyze`,
`evalshift compare`, and the HTML report to produce a migration verdict.
Ratio fields use decimal fractions: `0.03` means 3%.

```yaml
migration_policy:
  max_overall_regression_rate: 0.30
  max_critical_regressions: 1
  min_equivalence_rate: 0.75
  max_tool_argument_drift: 0.20
  max_tool_divergence: 0.20  # share of pairs where the target routed elsewhere
  max_invariant_violations: 0  # examples where the target broke a blocking trace rule
  tool_argument_drift_floor: 0.9   # below this a call counts as drifted
  max_cost_increase: 0.30    # target may cost up to 30% more
  max_latency_increase: 0.30 # …and be up to 30% slower
```

Those values are the ones `evalshift init` writes, and they are also the
`MigrationPolicy` field defaults a config that omits the block inherits — the
two differences are `tool_argument_drift_floor` and `fail_on_dropped_params`,
which init leaves out of the scaffold and every config therefore inherits at
`0.9` and `false`. They are a
first-migration starting point, deliberately loose enough that a fresh suite
reports its regressions instead of failing on a couple of reworded tool
arguments. Tighten them as the suite grows and the migration nears merge; the
other `--profile` presets (`cost-reduction`, `quantization`, `provider-switch`,
`local-model`) scaffold tighter numbers already.

`max_overall_regression_rate`, `min_equivalence_rate`,
`max_tool_argument_drift` and `max_tool_divergence` are true rates bounded
`0.0`–`1.0`.
`min_equivalence_rate` is a floor on the **non-regression** rate: a record that
is equivalent *or improved* counts as passing, so a target that beats the source
still meets the `0.75` default (the downside is bounded separately by
`max_overall_regression_rate`). The two *increase*
budgets — `max_cost_increase` and `max_latency_increase` — may exceed `1.0`
(e.g. `2.0` = the target may be 200% more expensive / slower before the policy
fails), since migrating a small model to a much larger one is legitimately far
costlier and slower. They are capped at `10.0` (1000%) only to reject obvious
typos (e.g. writing `200` instead of `2.0`).

Rates are counted over whole rows, so pick values your suite can actually
express. With 10 tool-argument rows the achievable drift rates are
`{0.0, 0.1, 0.2, …}` and `max_tool_argument_drift: 0.01` means "any drift at
all fails" — the budget is below one row's worth. `analyze` says so rather than
enforcing it silently:

```
The tool-argument drift budget of 1% (max_tool_argument_drift in
evalshift.yaml) is below the 10% granularity of 10 tool-argument
comparisons — effective tolerance is zero at this sample size.
```

The same check covers `max_overall_regression_rate` and
`max_tool_divergence`, and each slice is judged on its own row count. Capture more examples, or write a budget the suite can
represent; `0.0` is silent, since an explicit zero-tolerance budget is a choice.

The two *increase* budgets have no row denominator, but they can go unmeasured
in their own way: if neither model is priced (LiteLLM has no cost table entry
for the id), every `cost_usd` is `0.0` and the ratio has nothing to divide.
`analyze` reports that budget as `conclusive: false` rather than as a clean
pass, and says why:

```
The cost increase budget could not be measured: all 4 error-free calls
across both models recorded a cost of 0, so its observed 0.00 is a default,
not a measurement.
```

`max_latency_increase` behaves the same when every call reports `latency_ms: 0`
— typically an unpriced or unmetered model. Price the models, or re-run
live, if you need those budgets to bind.

`max_tool_divergence` is the divergence axis's own budget: the share of
`tool_selection.divergence` rows on which the target called different tools
than the source, counted over *those* rows only. It has deliberately no
materiality floor of its own — an argument score slides continuously (a
reworded query is not a wrong call), whereas a divergence score below `1.0`
means the target called a tool the source did not, or skipped one it did.
On a teacher-forced multi-round replay the row's score is the mean over the
replayed rounds, so an example counts as diverged if **any** round diverged.

`max_invariant_violations` is a **count**, not a rate: the number of examples
on which the target broke at least one rule of a blocking
[`trace_invariants`](#evaluatorstrace_invariants) evaluator, judged against the
rules alone — a rule the source broke too still counts against the target. The
default `0` fails on the first broken rule. Examples are counted distinctly by
`(prompt_id, example_id)`: every `trace_invariants` entry writes its own row, and
an example that broke rules in two entries counts once, over a denominator of
the distinct examples with a blocking row. A row averaged over
`samples_per_example` repeats counts when any one sample broke a rule. Because
a rule is an assertion the team wrote down rather than a statistic, a breach is
always conclusive: no interval softens it, it carries no `1/n` granularity
warning, and it turns `inconclusive` and `conditional_pass` into `fail` even
when the suite is too small for the paired tests. The same holds per slice: a
slice whose own `max_invariant_violations` breaches reads `fail` however few
comparisons it has. The budget's row appears only in scopes that scored at least
one blocking `trace_invariants` row, so a project with no trace rules (or only
advisory ones) sees no new budget row. It is overridable per slice like the
others.

A `tool_selection.conformance` row where **both** models missed the recorded
ground truth by the same margin is excluded from every policy rate: its zero
delta is a shared failure, not evidence the migration is safe. Such rows are
still reported, as a `TOOL_GROUND_TRUTH_MISS` count and as a recommendations
line naming how many were excluded — see [methodology](methodology.md). And
when the **source** model missed conformance on half or more of at least four
rows, `evalshift evaluate` reports a broken eval harness in red before any of
these budgets are read.

`tool_argument_drift_floor` is not a budget — it is the materiality threshold
`max_tool_argument_drift` is counted at. Argument scoring is continuous, so a
reworded search query or an omitted optional filter lands below `1.0` without
being wrong; counting every non-identical call would weigh a `0.98` the same as
a `0.0` and burn the drift budget on calls that were never wrong. A call counts as
drifted only when its target argument score falls **below** the floor. The `0.9`
default mirrors [`evaluators.semantic.min_similarity`](#evaluatorssemantic) —
the same "close enough" line on the same kind of score — and deliberately sits
above the `0.5` an omitted optional argument earns, because omitting a filter
changes which rows the tool returns. Lower it to `0.4` if you want presence
differences ignored.

Semantic-evaluator drift counts toward `max_overall_regression_rate` and
`min_equivalence_rate` only when it breaches
[`evaluators.semantic.min_similarity`](#evaluatorssemantic). A near-identical
output (cosine ~0.98) that stays within `min_similarity` is treated as
*equivalent*, not a regression — so the policy gate and the report agree.

Per-slice overrides live under `migration_policy.slices` — a map of
slice name (an example tag: every distinct tag is a slice) to a partial
policy block; unset fields inherit the top level. A slice budget gates
the run exactly like a top-level one: a conclusively breached slice
budget **fails** the run, an unconfirmed breach makes it `inconclusive`,
and `recommendations` names which slice budget blocked. A slice that
fails on comparison *severity* rather than a budget still only
downgrades an overall `pass` to `conditional_pass`.

```yaml
migration_policy:
  max_overall_regression_rate: 0.03
  slices:
    security:
      max_overall_regression_rate: 0.0   # zero tolerance on this slice
```

Two behaviours to know:

* **Only blocking evaluators gate quality.** Advisory
  (`blocking: false`) records are summarised separately and never flip
  the verdict. If *every* configured evaluator is advisory (the
  fresh-`init` state), the verdict is `inconclusive` — unless
  `max_cost_increase` or `max_latency_increase` is breached, which
  still `fail`s: those are computed from the run's calls, not from
  evaluator records. The recommendations then say whether each advisory
  judge has enough pairs to promote (see
  [`evaluators.llm_judge`](#evaluatorsllm_judge)).
* **The four rate budgets are Wilson-CI-aware.**
  `max_overall_regression_rate`, `min_equivalence_rate`,
  `max_tool_argument_drift` and `max_tool_divergence` are each a
  proportion of records, so each
  carries a 95% Wilson interval. A breach only *fails* when the interval
  confirms it; if the suite is too small to be sure, the verdict is
  `inconclusive`, not `fail`. A budget the observation *held* stays
  conclusive however wide its interval.
  Cost and latency budgets are exact, but report
  `conclusive: false` when neither model priced its calls (both averages
  zero is a default, not a measurement). Record-derived budgets report
  `conclusive: false` on a scope that scored zero records — their
  `0/0` default looks clean but measures nothing.

* **`max_invariant_violations` is conclusive by construction.** It counts
  distinct examples, so it has no interval and no "too small to be sure": a breach
  fails the run (or the slice) outright, overriding `inconclusive` and
  `conditional_pass`, and the verdict's `reason` names the scope and the
  count. Only blocking `trace_invariants` evaluators count toward it. When the
  source broke the same rule on the same example, `recommendations` adds a
  line saying so — not an exemption, but a hint that either both models break
  the rule or the rule no longer matches the toolset, which its owner should
  review.

* **`fail_on_dropped_params` gates constraints, not scores.** Default
  `false`. A promoted capture can pin generation parameters
  (`response_format`, `tool_choice`, `parallel_tool_calls`, `top_p`, a
  completion cap), and EvalShift replays them — but `drop_params` means a
  model that never accepted one still answers, minus the constraint. The run
  probes both arms at start and records the shortfall in `state.json` under
  `dropped_params`; the report shows a **Constraints not honoured** banner
  either way. Recorded alongside the probe's answer are the constraints LiteLLM
  claims to support and then never sends — on Gemini, `parallel_tool_calls` and
  a tool's `strict` flag, the latter under the pseudo-parameter name
  **`tools.strict`** since it lives on the `tools` array rather than in the
  generation config. Setting this to `true` makes a non-empty `dropped_params` fail
  the verdict outright, whatever the scores said, with a reason naming each
  model and parameter. Turn it on when the constraint *is* the contract — a
  suite of captures that pinned `response_format` measures nothing useful
  against a target that will not produce structured output. It is a top-level
  field only: a model either accepts a parameter or does not, which no subset
  of examples can vary, so `slices` has no equivalent. Runs recorded before
  this field existed carry no `dropped_params` and are never failed by it.

Verdicts are `pass`, `conditional_pass`, `fail`, or `inconclusive`.
When configured, `analyze` writes `migration_decision.json` next to
`analysis.json`; `report` renders it as the top-level migration verdict.
CI gating, on `analyze` and `compare`:

- `--policy-gate` exits 1 when the verdict is `fail` or `conditional_pass`,
  and also when no `migration_policy` is configured. `inconclusive` exits 0.
- `--gate critical,high` exits 1 when any comparison has one of the listed
  severities (allowed: `critical`, `high`, `medium`, `low`).
- When `$GITHUB_STEP_SUMMARY` is set, `analyze` appends a markdown results
  table to the GitHub Actions job summary.

`evalshift.yaml`'s `migration_policy` is the single source of truth for these
budgets. `analyze` stamps the resolved policy it computed the verdict under —
every top-level budget with its default applied, plus `slices` — onto
`migration_decision.json` as `policy`. `bundle` and `push` do not read that
file: `bundle` re-resolves the verdict from `evalshift.yaml` at bundle time,
so the bundle's `decision.policy` is the policy the config held *when the
bundle was built* — edit `migration_policy` between `analyze` and `bundle` and
the bundle carries the new numbers, not the ones `analyze` last wrote. `push
<run-id>` builds a bundle only when none exists, so after an edit re-run
`evalshift bundle <run-id>` before pushing. Either
way, the hosted gate checks a pull request against exactly the budgets the
bundle's own verdict used. `evalshift.yaml` is the source of truth for that
policy; the web app's project policy view is becoming a read-only display of
the snapshot each run pushed. `policy` is `null` when no `migration_policy` is
configured, and on a `migration_decision.json` written before this field
existed. See [What `push` sends](hosted.md#what-push-sends-block-by-block) for
the upload contract and the notices `push` prints around a policy-less run.
Because every budget is resolved, the snapshot carries `max_invariant_violations`
(at its default `0`) even for a config that never mentions trace rules.

## `prompts`

`prompts` is the **template axis** and [`suites`](#suites) the **dataset
axis**: a run renders every prompt template with every example of one
suite, so both are always present, and `prompts` is required even for a
capture-first project. There the single `replay` prompt that `init` writes
(`content: "{input}"`) is a passthrough — a promoted capture's example is
`{"input": "<full rendered prompt>"}`, and echoing it back verbatim is what
makes captured inputs replayable against a second model.

A list of prompt definitions. Each entry has:

| Field         | Type    | Required                          | Description |
| ------------- | ------- | --------------------------------- | ----------- |
| `id`          | string  | yes                               | Stable identifier surfaced in reports. Must be unique within the file. |
| `detection`   | enum    | yes                               | `manual` or `python_string`. |
| `content`     | string  | when `detection: manual`          | Inline prompt body. Forbidden when `detection: python_string`. |
| `path`        | string  | when `detection: python_string`   | Relative or absolute path to a `.py` file. Resolved against the directory containing `evalshift.yaml`. |
| `variable`    | string  | when `detection: python_string`   | Module-level variable name holding the prompt string. |
| `variables`   | list    | optional                          | Names of `{template}` placeholders the prompt expects. Used by the pre-flight compatibility check. |
| `max_tokens`  | int     | optional (`> 0`)                  | Per-prompt override of `defaults.max_tokens`. Raise it for prompts whose models emit long JSON / tool arguments that would otherwise be truncated. |

### Two prompt-detection modes

* `manual` — write the prompt body inline:
  ```yaml
  - id: greet
    detection: manual
    content: "Hello {name}"
    variables: [name]
  ```
* `python_string` — point at an existing module-level string in your
  codebase:
  ```yaml
  - id: greet
    detection: python_string
    path: src/prompts/greet.py
    variable: GREET_PROMPT
    variables: [name]
  ```
  EvalShift AST-walks the file and extracts the string literal. **It
  does not run user code.** F-strings, concatenations, `.format()`
  calls, and other dynamic forms are explicitly rejected.

## `defaults`

| Field           | Type   | Default                       | Description |
| --------------- | ------ | ----------------------------- | ----------- |
| `source_model`  | string | (none)                        | Default `--from` model id (or alias). |
| `target_model`  | string | (none)                        | Default `--to` model id (or alias). |
| `judge_model`   | string | `gemini-3.1-flash-lite-preview` | Default LLM-as-judge model. |
| `insights_model`| string | (none)                        | Model that writes the run-insights narrative rendered in `report.html` and uploaded with the bundle. Falls back to `judge_model` when unset — writing analytical prose is a harder task than a pairwise A/B verdict, so it is worth tuning separately. See [Run insights](#run-insights). |
| `concurrency`   | int    | 10 (1 ≤ x ≤ 64)               | Max in-flight LLM calls during `evalshift run` **and** `evalshift evaluate` (the embedding and judge calls made while scoring). |
| `cache`         | bool   | `true`                        | Read/write the local SQLite cache at `~/.evalshift/cache.db`. Covers run-stage completions — for examples that offer tools, one entry per replayed round — plus `semantic` embeddings and `llm_judge` verdicts. See [Response cache](https://github.com/evalshift/evalshift-cli/blob/main/DOCS.md#response-cache) for the key. |
| `max_cost_usd`  | float  | 50.0                          | Soft ceiling reserved for future enforcement. The pre-flight cost prompt currently triggers above $10 (skip with `--yes`). |
| `max_tokens`    | int    | 4096 (`> 0`)                  | Completion length cap sent to every model call. Raise it if outputs are being truncated (the provider returns `finish_reason == "length"`); a `prompts[].max_tokens` entry overrides it per prompt. Truncated calls are detected, surfaced in the report, and **excluded from the regression statistics** so a cut-off output can't manufacture a false regression. |
| `samples_per_example` | int | 1 (1 ≤ x ≤ 20)           | How many times each `(prompt, example)` is sent to **each** model. Above 1, every sample is its own live call (the cache keys on the sample index), sample *i* of the source is scored against sample *i* of the target, and the example's row in `scores.jsonl` becomes the **mean over samples** with the per-sample scores and the within-example `delta_variance` under `metadata.samples`. The paired tests still run over examples, not samples, so this reduces noise without inflating `n`. Cost and the call count multiply by it; only worth turning on for a model that samples non-deterministically (see the report banner). See [Methodology](methodology.md#limitations-to-be-aware-of). |

### Run insights

`evalshift report` (and therefore `compare`) writes a plain-language
explanation of the run — a summary each for the verdict, the advisory
signal and the economics, plus behavioural findings and a
recommendation. It is rendered at the top of `report.html` and uploaded
with the bundle.

```yaml
defaults:
  insights_model: gemini-3.1-flash-lite-preview   # optional
```

The prose is machine-written, but the figures in it are not: every
number is computed first and handed to the model pre-rendered as a
display string to copy verbatim, and the output is rejected and
regenerated if it contains a numeric token that was not supplied. Two
bad generations fall back to deterministic templated prose.

- One model call per run (a second only on a rejected generation),
  cached in `insights.json` and keyed on the run's `config_hash` plus
  the model id — re-running `report` or `push` costs nothing.
- Skip it with `--insights/--no-insights` on `report` and `compare`. It is
  also skipped when no API key is configured for the chosen model, and
  when the run has no usable `evalshift.yaml`.
- A generation failure never fails the run.
- The worst 8 regressions' inputs and both models' outputs are sent to
  `insights_model` (truncated to 2000 characters each) — the same
  exposure an `llm_judge` criterion already has. Use `--no-insights`
  if that is not acceptable for your suite.

## `evaluators`

Eight sub-keys, all optional — `structural`, `semantic`, `tool_selection`,
`tool_arguments`, `tool_trace_structure`, `agent_trace`, `trace_invariants` and
`llm_judge`, each
documented below. **At least one evaluator must be configured for
`evalshift evaluate` to do anything.**

### `blocking` (every evaluator)

Every evaluator entry accepts `blocking: bool` (default `true`).

| Value   | Behaviour |
| ------- | --------- |
| `true`  | Regressions from this evaluator count toward `migration_policy` budgets and can fail the migration verdict. |
| `false` | *Advisory*: the evaluator still scores every pair and its results appear in the report (under advisory metrics/regressions), but it never gates the verdict. |

The `init` scaffold marks `semantic` and `llm_judge` advisory: at the
small suite sizes fresh captures start with, embedding drift and judge
noise would otherwise dominate the verdict. Flip them to `blocking: true`
once your suite is large enough that you trust their calls. Deterministic
evaluators (structural, tool-call) default to blocking.

Note the asymmetry: the **library default** for every evaluator, `semantic`
and `llm_judge` included, is `blocking: true`, so a hand-written config
that omits the key gates on them while an `init`-generated one does not.
That is deliberate — flipping the library default would silently turn a
failing migration into a passing one for every existing config that relies
on the omitted key, and a gate loosened under a minor release is worse than
the asymmetry. It stays until a `version: 2` schema. Write `blocking: false`
explicitly when you want init's behaviour in a hand-written file (see the
[FAQ](faq.md#why-does-a-hand-written-config-block-on-semantic-when-init-does-not)).

### `evaluators.structural`

A list. Each entry has a `type` and the fields that type needs.

| `type`        | Required fields                          | Behaviour |
| ------------- | ---------------------------------------- | --------- |
| `json_schema` | `schema_path` (string)                   | Each output is parsed as JSON; score 1.0 if it validates against the schema, 0.0 otherwise. |
| `regex`       | `pattern` (string)                       | Score 1.0 if the regex matches anywhere in the output, 0.0 otherwise. |
| `length`      | `min_chars` and/or `max_chars` (int)     | Score 1.0 inside the bounds, distance-decayed outside. |

Optional `applies_to: ["prompt-id-glob", ...]` (default `["*"]`) for
future per-prompt scoping.

### `evaluators.semantic`

A single object (not a list).

| Field             | Type   | Default                  | Description |
| ----------------- | ------ | ------------------------ | ----------- |
| `embedding_model` | string | `text-embedding-3-small` | LiteLLM-compatible embedding model id. The bare default resolves to OpenAI (`OPENAI_API_KEY`). Use a Gemini one (e.g. `gemini/gemini-embedding-001`) if you don't have an OpenAI key. |
| `min_similarity`  | float  | `0.9`                    | Cosine similarity (0–1) below which the target is flagged as a semantic regression. Minor rewording/formatting typically scores ~0.98, so the default 0.9 avoids false flags; set to `1.0` to flag any deviation from byte-identical. Also governs whether semantic drift counts toward the [`migration_policy`](#migration_policy) regression/equivalence gates. |

The semantic evaluator scores the **target's similarity to the source**:
target_score = cosine(source, target), source_score = 1.0. A
negative `delta` means the target drifted from the source's meaning.
`blocking` defaults to `true` in the library but `init` writes `false` —
see [`blocking`](#blocking-every-evaluator) for why.

`init` writes this block active for every provider. Gemini and OpenAI
projects use their own embedding model. Anthropic and DeepSeek have no
embeddings endpoint, so their scaffold borrows one, checking the keys you
already have: `openai/text-embedding-3-small` when `OPENAI_API_KEY` is set,
else `gemini/gemini-embedding-001` when `GEMINI_API_KEY` or `GOOGLE_API_KEY`
is, else `openai/text-embedding-3-small` anyway. A comment above the block
names the key it needs, and `init --ci` wires that key into the workflow as
a second, optional secret.

**No API key for `embedding_model`.** `compare` and `run` check it before
the first model call. An advisory block (`blocking: false`, as scaffolded)
is skipped — no embedding calls — with
`⚠ semantic skipped: no API key for openai/text-embedding-3-small — export OPENAI_API_KEY to enable it.`;
the skip is recorded in `state.json` (`skipped_evaluators`) and repeated in
the verdict's recommendations. A blocking one stops the run with exit 1
instead. `evaluate` applies the same rule.

**Interaction with `tool_arguments`.** [`tool_arguments`](#evaluatorstool_arguments)
borrows this block's embedder, so skipping it changes how arguments score:

| `tool_arguments` uses | Without the embedder | A keyless `semantic` is treated as |
| --- | --- | --- |
| the `semantic` strategy (in `strategies` or as `default_strategy`) on a **blocking** entry | that gate would fall back to `exact` | **blocking** — the run stops |
| the `semantic` strategy on an advisory entry | `exact` | this block's own `blocking`; the warning says so |
| `auto` (the default) | free text graded by `difflib` ratio | this block's own `blocking`; the warning says so |

### `evaluators.tool_selection`

A list. Each entry has:

| Field             | Type   | Default       | Description |
| ----------------- | ------ | ------------- | ----------- |
| `name`            | string | (required)    | Identifier surfaced in reports.  |
| `conformance`     | enum   | `expected`    | Ground-truth axis: `expected` / `expected_set` / `off`. `expected_set` is `expected` made order-insensitive: multiset recall of `example.expected_tools` names. Use it when the expected calls are a parallel fan-out whose order carries no meaning. |
| `divergence`      | enum   | `set`         | Target-vs-source axis: `set` (Jaccard on tool names) / `exact` (sequence equality) / `first` (first call only) / `off`. |
| `applies_to`      | list   | `["*"]`       | Glob list of prompt ids (accepted, not yet enforced — only [`trace_invariants`](#evaluatorstrace_invariants) enforces it). |
| `severity_floor`  | enum   | `null`        | If set, surfaces in metadata so the analysis layer can floor severity. |

The two axes are independent and each writes **its own record**, under
`kind: tool_selection.conformance` and `kind: tool_selection.divergence`.
They answer different questions:

* **conformance** grades *each side* against the example's ground truth, so
  both can fail at once and the delta stays 0 — the migration did not cause a
  failure both models share. An example carrying `expected_no_tools` is graded
  against *that*, under either strategy; an example with no ground truth at all
  is not measured and writes no row. When **both** sides miss, the record is
  tagged `TOOL_GROUND_TRUTH_MISS` — ground truth captured from the source model
  that the source model then fails means the harness is misconfigured (wrong
  toolset attached, wrong prompt, suite promoted from a different agent), not
  that the migration regressed.
* **divergence** grades the target against the source, which is its own
  baseline at 1.0, so behaving differently is a negative delta — a regression.
  It needs no ground truth, which is the point: it is what catches two models
  failing the same expectation in two different ways.

`divergence` defaults to `set` rather than `exact` so that reordered identical
calls do not read as drift. Setting both axes to `off` is a config error — the
evaluator would measure nothing. There is no `mode` field: it was removed, not
deprecated, and a config still carrying one fails validation.

### `evaluators.tool_arguments`

| Field                   | Type   | Default | Description |
| ----------------------- | ------ | ------- | ----------- |
| `name`                  | string | (required) | Identifier. |
| `applies_to`            | list   | `["*"]` | Glob list of prompt ids (accepted, not yet enforced — only [`trace_invariants`](#evaluatorstrace_invariants) enforces it). |
| `against`               | enum   | `source` | What arguments are compared to: `source` (drift from the source model) or `expected` (correctness against `expected_tools[].arguments`, scored on both sides). |
| `strategies`            | dict   | `{}`    | Per-field strategy overrides (`exact`/`subset`/`numeric`/`semantic`/`auto`). |
| `default_strategy`      | enum   | `auto`  | Strategy for fields `strategies` does not name. `auto` is the ladder below; `exact` restores byte-equality scoring. |
| `numeric_tolerance`     | float  | `0.05`  | Relative-error tolerance for `numeric`. |
| `optional_fields_scored`| string | `lenient` | How a field present on one side only is scored: `lenient` = 0.5, `strict` = 0.0. |
| `use_llm_judge_fallback`| bool   | `false` | Reserved; not yet implemented. |

`strategies` keys are **field names matched across every tool**, so pick names
that mean the same thing throughout your toolset. The `semantic` strategy needs
a configured [`evaluators.semantic`](#evaluatorssemantic) to borrow an embedding
model (and its cache) from; without one it degrades to `exact`. A `semantic`
block whose model has no API key is skipped when advisory — unless a blocking
entry here uses the `semantic` strategy, in which case the missing key stops
the run — see [`evaluators.semantic`](#evaluatorssemantic).

#### The `auto` strategy ladder

Every field `strategies` does not name is scored by `default_strategy`, which
defaults to `auto`. `auto` is a ladder, cheapest rung first:

1. **Normalized exact.** Two strings that compare equal after normalization —
   case, surrounding whitespace, repeated internal whitespace — score `1.0`.
   No schema lookup, no API call. `"Find people"` vs `"find  people"` is a
   capitalization difference, not a wrong value.
2. **Schema dispatch.** The field is looked up in the toolset the example
   carries (`toolset_ref` sidecar or inline `tools`) and its declared type
   picks the strategy: identifiers (`*_id`, `*_ids`), `enum` values, booleans
   and `date` / `date-time` / `uuid` / `email` formats are scored `exact` —
   a reworded timestamp is wrong, not "similar"; `number` / `integer` go to
   `numeric`; `object` / `array` to `subset`.
3. **Graded similarity.** Whatever the schema did not decide — free text, and
   anything the example has no schema for — is graded rather than failed:
   `semantic` when an `evaluators.semantic` block lent an embedding model,
   `difflib` sequence ratio when it did not, so partial credit survives with
   no embedding model configured. Numbers still go through `numeric`, dicts
   and lists through `subset`, everything else through `exact`.

Set `default_strategy: exact` for byte-equality scoring, where a
capitalization difference is a wrong value. A per-field entry in `strategies`
always wins over `default_strategy`.

`against: source` (the default) answers *"did the arguments change?"* — the
source's own score is 1.0 by construction, so a source model that passed a value
that does not exist still scores perfectly. `against: expected` scores **both**
models against `expected_tools[].arguments`, which is what a capture-first suite
wants. Each expectation's `match_strategy` decides which keys are compared:
`exact` takes the union of expected and actual keys, `subset` and
`contains_per_field` (what `capture promote` writes) compare the recorded keys
only. An example with no expected arguments is skipped at a neutral 1.0/1.0.

`optional_fields_scored` governs *presence*, never values. A tool parameter that
is `"required": []` in your schema can legitimately be passed by one model and
omitted by the other — `lenient` scores that 0.5 rather than treating it as a
total failure, which is what turned every tool with optional parameters into a
false regression at `blocking: true`. A field both sides passed with different
values still scores by its strategy, so a wrong value is still 0.0.

Under `against: expected`, a ground-truth field that **neither** model produced
is dropped from that call's denominator on both sides and disclosed as
`unmeasured_fields` in the record's per-call metadata (`scores.jsonl`). It is a
stale expectation, not a model defect: scored, it would cap the call below 1.0
for good, since no model change could ever lift it. A field only one side
omitted is unaffected — that is `optional_fields_scored`' business. A call whose
expectation consists entirely of such fields scores 1.0/1.0, consistent with the
neutral score for an expectation with nothing comparable in it.

`ARGUMENT_VALUE_DRIFT` is stamped on a record only when the target scored
**below the source** — a regression. Both sides failing the same expectation by
the same margin is a fact about your ground truth, and stamping it counted one
migration defect twice. The `migration_policy.max_tool_argument_drift` budget is
unaffected: it counts calls whose *target* score fell below
`tool_argument_drift_floor`, not failure-category labels.

### `evaluators.tool_trace_structure`

| Field                 | Type   | Default | Description |
| --------------------- | ------ | ------- | ----------- |
| `name`                | string | (required) | Identifier. |
| `applies_to`          | list   | `["*"]` | Glob list of prompt ids (accepted, not yet enforced — only [`trace_invariants`](#evaluatorstrace_invariants) enforces it). |
| `check_call_count`    | bool   | `true`  | Score the number of tool calls. |
| `check_parallelism`   | bool   | `true`  | Score parallel-vs-sequential alignment. |
| `check_refusals`      | bool   | `true`  | Score refusal alignment; mismatches force `severity_floor: high`. |
| `call_count_tolerance`| int    | `1`     | `+/- N` calls considered equivalent. |

### `evaluators.agent_trace`

A list. These evaluators consume traces imported with
`evalshift traces import`; they do not run your agent and do not replace
the normal `raw.jsonl` model-call artifact.

| Field                        | Type   | Default | Description |
| ---------------------------- | ------ | ------- | ----------- |
| `name`                       | string | required | Identifier surfaced in scores and reports. |
| `applies_to`                 | list   | `["*"]` | Glob list of prompt ids (accepted, not yet enforced — only [`trace_invariants`](#evaluatorstrace_invariants) enforces it). |
| `check_tool_order`           | bool   | `true`  | Compare source and target tool-call order. |
| `check_arguments`            | bool   | `true`  | Compare arguments for same-name matched tool calls. |
| `check_missing_verification` | bool   | `true`  | Check dangerous tools have an earlier verification tool. |
| `verification_tools`         | list   | `[]`    | Tool names treated as verification steps. |
| `dangerous_tools`            | list   | `[]`    | Tool names that should not appear without verification or as extras. |

Example:

```yaml
evaluators:
  agent_trace:
    - name: trace_safety
      verification_tools: ["check_refund_policy"]
      dangerous_tools: ["issue_refund"]
```

### `evaluators.trace_invariants`

A list. Each entry is a set of hand-written rules that every in-scope tool-call
trace must satisfy, checked on **both** sides of every pair and judged against
the rules, not against the source: a rule the source also broke still counts
against the target. See [Evaluators → Trace invariants](evaluators.md#trace-invariants)
for when to reach for it.

Write it in the top-level `evaluators:` block (outside the `capture sync`
managed region) or in a suite's own `evaluators:` override. `capture sync`
keeps a managed suite entry's `trace_invariants` when it regenerates that
entry — see [`managed`](#managed).

```yaml
evaluators:
  trace_invariants:
    - name: payments_contract
      owner: "@payments-team"     # optional; echoed onto every record and the report
      applies_to: ["checkout-*"]  # prompt-id globs; default ["*"]
      blocking: true              # default; false = reported, never gates
      traces: replayed            # default; `imported` checks `evalshift traces import` traces
      rules:
        - id: auth-before-charge
          type: order
          before: authenticate
          after: charge_card
        - id: no-legacy-refund
          type: forbidden
          tools: [refund_v1]
        - id: one-charge
          type: call_count
          tool: charge_card
          max_calls: 1              # min_calls, max_calls or both; both equal = exact
        - id: must-book
          type: required
          tools: [create_booking]
        - id: sane-amount
          type: arguments
          tool: charge_card
          json_schema:
            type: object
            required: [amount]
            properties:
              amount: {type: number, minimum: 0.01, maximum: 5000}

migration_policy:
  max_invariant_violations: 0     # default; examples where the target broke any blocking rule
```

| Field        | Type   | Default      | Description |
| ------------ | ------ | ------------ | ----------- |
| `name`       | string | (required)   | Identifier surfaced in scores and reports. |
| `applies_to` | list   | `["*"]`      | Glob list of prompt ids. **Enforced here**, unlike on the other families: a prompt outside these globs is never checked, gets no row and is never counted. If the globs match no prompt in a run (with `traces: imported`, no imported trace pair), `evaluate` prints a warning naming the entry and its `applies_to`, because none of its rules were checked (`compare` scores quietly and does not print it); matching at least one prompt stays silent. |
| `blocking`   | bool   | `true`       | `true` counts target violations toward [`max_invariant_violations`](#migration_policy); `false` still checks and reports them, but never gates. |
| `traces`     | string | `replayed`   | `replayed` checks the tool calls `evalshift run` replayed (a pair whose replay recorded no tool trace gets no row). `imported` checks the traces brought in with [`evalshift traces import`](traces.md); `evaluate` fails on a run with no `traces.jsonl`, naming `trace_invariants (traces: imported)` in the error. |
| `owner`      | string | `null`       | Free text (a team or a handle). Echoed onto every record and into the report, so a violation names who to ask. Nothing enforces it. |
| `rules`      | list   | (required)   | At least one rule. Every rule has an `id` (unique within the entry; it names violations in reports) and a `type`. |

The unit of checking is a **response**: one model turn, plus the tool calls
already in its **context** before it answered. On a replayed trace there is one
response per replayed round, and round *k*'s context is the tool calls on the
example's `history` assistant turns plus the *recorded* calls of the rounds the
teacher-forced replay fed back before it; the model's own earlier rounds are
never context, because the replay never feeds them back. An imported trace is
one response with an empty context — every call in it was the agent's own.
Context calls can never *be* violations, since the model under test did not
make them, but they can satisfy or count toward a rule:

| `type`       | Fields | Violated when | Context role |
| ------------ | ------ | ------------- | ------------ |
| `forbidden`  | `tools` (list) | the model calls any tool in `tools` (one violation per call) | none |
| `required`   | `tools` (list) | some tool in `tools` is called in no response by the model | none: a call only in the context does not satisfy it. Use `call_count` `min_calls` when the conversation may already contain the call |
| `order`      | `before`, `after` (two different tools) | the model calls `after` and no `before` precedes it, either in the context or earlier in the same response | a context `before` satisfies it |
| `call_count` | `tool`, `min_calls` and/or `max_calls` (at least one; `max_calls` ≥ `min_calls`) | `max_calls`: a model call to `tool` takes the running count past it, counting context calls first (one violation per excess call). `min_calls`: at the end of the last response, context calls plus the model's calls to `tool` number fewer than it (one violation). Both set and equal is exact cardinality | context calls count, toward both bounds |
| `arguments`  | `tool`, `json_schema` | a model call to `tool` has arguments that fail `json_schema` (Draft 2020-12, itself validated when the config loads) | none |

"Earlier in the same response" means emitted earlier; imported traces are
ordered by `sequence_index`, so the two modes agree.

**Scoring.** Each side scores the share of rules it broke zero times —
`(rules − rules broken) / rules` — so a rule broken three times costs the same
as one broken once, and `delta = target − source`. The record's kind is
`trace_invariants`; its metadata carries `rules_checked`, `source_violations`
and `target_violations` (each a list of `{rule_id, rule_type, tool,
round_index, detail}`), `owner` when set, and the failure category
`INVARIANT_VIOLATION` when the target broke anything. The report's **Top
regression causes** panel (and `failure_categories` in
`migration_decision.json`) counts that category the way the budget does —
distinct examples on which the target broke a *blocking* rule — so an advisory
violation, or a second entry broken on the same example, never inflates it
past the figure the verdict used. With `samples_per_example` above 1 the
example's row keeps every scored sample's violations, each tagged with its
`sample` ordinal. The HTML report's **Trace rules broken** panel and
`report.json`'s `invariant_violations` list every rule the target broke,
whatever the delta — including those of advisory (`blocking: false`) entries,
which never count toward the budget — with the owner and whether the source
broke it too. Each row carries the entry's `blocking` flag, and the panel tags
`blocking: false` rows **advisory**. The **Overall, by evaluator** table tags
every advisory (`blocking: false`) evaluator's row the same way, and every
`trace_invariants` row there also says what it measures: each side judged
against the entry's rules, not against the other side. The
per-prompt **Executive summary** — and the header's mean score Δ, which
averages its rows — is drawn from blocking evaluators only (worst severity
first, ties to the most negative effect size), falling back to advisory ones
on a prompt that has nothing else, so an advisory rule the target satisfied by
calling no tools cannot read as the prompt's improvement above a failed
verdict.

Four things to know:

- **`required` and `min_calls` are not vacuous.** `order`, `arguments` and a
  `call_count` with only `max_calls` say nothing about a trace that never calls
  the tool. `required` and `call_count` with `min_calls` fail on every in-scope
  trace that lacks the calls. A single-shot replay sees only the first
  response, so scope them with `applies_to`, or use them with
  `capture sync --rounds all` or with imported traces.
- **`required` counts the model's calls; `call_count` counts the
  conversation's.** `required` asks whether the model under test made the
  call. `call_count` asks whether the conversation contains it, so a recorded
  history that already called `charge_card` satisfies `min_calls: 1` and
  leaves `max_calls: 1` no room.
- **Teacher-forced rounds see recorded context.** Round *k* is judged with the
  recorded rounds before it as context, not the model's own earlier rounds.
  That applies to `min_calls` too: it is judged on the last response's context
  plus that response's own calls.
- **Malformed arguments fail any schema that requires a field.** Arguments the
  model emitted as unparseable JSON are recorded as
  `{"_raw": …, "_parse_error": true}`, which fails any `json_schema` that
  requires a field.

### `evaluators.llm_judge`

A list of pairwise judges. Each entry has:

| Field              | Type   | Required | Description |
| ------------------ | ------ | -------- | ----------- |
| `criterion_name`   | string | yes      | Short id surfaced in reports. |
| `criterion_prompt` | string | yes      | Free-form criterion the judge applies (e.g. "which output preserves more factual detail?"). |
| `judge_model`      | string | optional | Model used as the judge (built-in default `gemini-3.1-flash-lite-preview`). Prefer a judge from a third model family so it isn't grading its own relatives. |

**Judge family.** LLM judges tend to prefer output from their own relatives
(self-preference bias), and nothing in the scoring can remove that. When a
`judge_model` resolves to the same provider as `defaults.source_model` or
`defaults.target_model`, `evalshift doctor` prints a warn-level `judge family`
row and `evalshift validate` a matching `⚠` line — never a failure, because
`init` deliberately scaffolds a same-provider judge so a first run needs one
API key. The report repeats the note above the verdict whenever a judge that
actually contributed `llm_judge` rows shares a family with an arm, and
`report.json` carries it as `judge_family_overlap`. "Family" is the provider
the model id resolves to (`anthropic`, `deepseek`, `google`, `openai`); ids the registry
cannot place never match.

The judge sees both outputs (with random A/B order to defang positional
bias) and produces strict-JSON `{"winner": "A"|"B"|"tie", "reason":
"..."}`. Target wins → `(0.0, 1.0)`; tie → `(0.5, 0.5)`; source wins →
`(1.0, 0.0)`. Malformed responses degrade to `(0.5, 0.5)` with the
error preserved. `blocking` defaults to `true` in the library but `init`
writes `false` — see [`blocking`](#blocking-every-evaluator) for why.

**When to promote it.** While nothing gates the verdict (the fresh-`init`
state), the recommendations name each advisory judge and read its readiness
from its own smallest per-prompt `n`: at 20 pairs on every prompt it says
the judge is ready for `blocking: true`; below, it names the thinnest prompt
and its count. Nothing is printed once any evaluator gates.

**No API key for `judge_model`.** Checked before the first model call, as
for [`semantic`](#evaluatorssemantic): an advisory judge is skipped with a
`⚠ llm_judge.<criterion_name> skipped: …` line naming the env var to export,
and a blocking one stops `compare` / `run` with exit 1.

## Slices

Slices come from the suite; there is nothing to configure. Every distinct
example `tag` becomes a slice under its own name, alongside the implicit
`"all"` slice, and each is analysed separately. Per-slice budgets go under
[`migration_policy.slices`](#migration_policy), keyed by the tag. There is no
top-level `slices:` key — it was
[removed](#top-level-slices-was-removed).

`overall` is reserved and cannot be used as an example tag or as a
`migration_policy.slices` key. It names the run-level scope in the run bundle,
so a slice by that name would shadow the whole-run numbers wherever the two are
rendered together. Both spellings are rejected when the config or suite loads.

Slices with identical membership are collapsed to one before analysis, so
duplicate tags cannot inflate the Benjamini–Hochberg correction. `all` and any
slice named under `migration_policy.slices` always survive; see
[methodology.md](methodology.md#slice-deduplication) for the full rule.

## Suite (`golden.jsonl`) shape

The suite is JSON Lines — one example per non-blank line. Each row:

| Field      | Type    | Required | Description |
| ---------- | ------- | -------- | ----------- |
| `id`       | string  | yes      | Unique within the suite. |
| `inputs`   | object  | yes      | Mapping of template-variable name to value. |
| `tags`     | list    | optional | Slice tags. |
| `expected` | object  | optional | Reference output (unused by most evaluators). |

Also present (v0.2, tool-call ground truth — see `docs/agents.md`):
`expected_tools`, `expected_tool_count`, `expected_no_tools`,
`expected_parallel`, plus `expected_tool_rounds` (v0.3 — the whole recorded
agent loop, one list per tool-emitting model call; `expected_tools` is
`expected_tool_rounds[0]`) and `tool_result_fixtures` (the recorded results of
those calls, one inner list per covered round, positionally aligned with
`expected_tool_rounds`; each entry is `{tool_name, result, error}`). Written by
`capture promote` / `sync --rounds all`; when present the runner replays the
example teacher-forced, one round per covered round plus the answer round, and
the tool evaluators score per round — see
[Agent rounds](agents.md#agent-rounds-and-what-a-replay-can-reproduce). `null`
(the default, and every suite written before the field existed) means
single-shot replay. Fixtures must line up: they require `expected_tool_rounds`,
cannot cover more rounds than it has, and each covered round must have one
result per expected call with matching `tool_name`, else the suite fails to
load.

Each `expected_tools` entry carries `provenance`: `captured` (the default, and
what `capture promote` / `capture sync` write) means its arguments were
transcribed verbatim from the source model's own recorded call — nobody has
checked that they are *right*. Set it to `reviewed` once a human has confirmed
the row. Scoring is identical either way; the flag only controls whether the run
discloses that its `against: expected` ground truth is source-derived (see
[Agent migrations](agents.md#scoring-arguments-against-ground-truth)).

**`toolset_ref` (string) or `tools` (list) — required, exactly one of the
two.** Every model call records the toolset it was offered, so every suite
example must carry it too: `toolset_ref` points at a `<base>/toolsets/<hex>.json`
sidecar (what `capture promote`/`sync` write); `tools` inlines the toolset
directly (what a hand-authored suite uses — `[]` is a valid "no tools offered"
value). Neither present, or both, fails to load. See `docs/agents.md`.

Multi-turn conversation fields (see `docs/conversations.md` for the full
walkthrough — all optional, additive, single-turn suites parse unchanged):

| Field             | Type                                 | Required | Description |
| ----------------- | ------------------------------------- | -------- | ----------- |
| `history`          | list of `{role, content, tool_calls?, tool_call_id?}` or `null`  | optional | Conversation prefix replayed verbatim before the current turn (teacher-forced). `role` is `system`, `user`, `assistant`, or `tool`; `tool_calls` only on `assistant`; `tool_call_id` required on `tool` and forbidden elsewhere; at most one `system` message, and it must come first if present. `null` (the default) means single-turn — no message-mode dispatch. |
| `conversation_id`  | string or `null`                     | optional | Id of the recorded conversation this turn came from. Provenance only. |
| `turn_index`       | integer (`>= 0`) or `null`           | optional | Zero-based position of this turn within its conversation. Shown as a `turn N` badge in the HTML report. |
| `generation_config` | object or `null`                    | optional | Generation settings recorded by the SDK on the capture's first model call (`temperature`, `response_mime_type`, `response_schema`, `tool_choice`, `parallel_tool_calls`, `tool_config`, ...). Written by `capture promote`/`sync`; the runner translates it at dispatch — `temperature` overrides the model default, `response_mime_type: application/json` (plus an optional `response_schema`) becomes a LiteLLM `response_format`, and `tool_choice` / `tool_config` / `parallel_tool_calls` become an OpenAI-style `tool_choice` + `parallel_tool_calls` that LiteLLM maps per provider — all on both the source and target calls. See [Agents → Tool-choice constraints are replayed too](agents.md#tool-choice-constraints-are-replayed-too). Delete the field to disable the override; keys the runner cannot translate are ignored with a warning. |

Unknown keys are rejected (typos fail fast).

## `suites`

A map of named suites so `evalshift run --suite-name <name>` can resolve a
suite path without retyping it. Optional — omit it and `run` uses `--suite`
(or the default `golden.jsonl`). Suites are the dataset axis of a run, the
inputs that get rendered through every entry under [`prompts`](#prompts),
which is why a capture-first config still carries a (passthrough) prompt.

```yaml
suites:
  support_agent:
    source: captured
    path: .evalshift/suites/support_agent/golden.jsonl
```

| Field        | Type   | Required | Description |
| ------------ | ------ | -------- | ----------- |
| `source`     | string | optional | `captured` (built by `capture promote`) or `jsonl` (hand-authored). Advisory provenance; both resolve to `path`. Default `captured`. |
| `path`       | string | yes      | Path to the suite JSONL, **relative to the config file's directory**. |
| `evaluators` | block  | optional | Evaluators this suite is scored with, replacing the top-level `evaluators:` family by family. Omitted (the default) scores the suite with the top-level block unchanged. |
| `managed`    | bool   | optional | Whether `capture sync` owns this entry. Default `true`. |

Resolution precedence for `run` and `compare`: an explicit `--suite <path>`
wins, then `--suite-name <name>` (looked up here). With neither flag, a config
that wires exactly one suite uses it; one that wires two or more is an error
that lists a ready-to-run command per suite; `./golden.jsonl` is the fallback
only when `suites:` is empty.

### Per-suite `evaluators`

A project's suites are rarely homogeneous: one calls tools, six answer in prose.
One top-level `evaluators:` block either leaves the tool-calling suite unmeasured
or hands the tool-free ones a tool evaluator with an empty denominator, which
reads as an *inconclusive* gate rather than as "not applicable here". So a suite
can carry its own block:

```yaml
suites:
  main_chat:
    source: captured
    path: .evalshift/suites/main_chat/golden.jsonl
    evaluators:
      tool_selection:
        - name: routing
          conformance: expected
          divergence: set
      tool_arguments:
        - name: routing_args
          against: expected
  briefing:
    source: captured
    path: .evalshift/suites/briefing/golden.jsonl   # inherits the top level as-is
```

The block takes the same families as the top-level `evaluators:` — `structural`,
`semantic`, `llm_judge`, `tool_selection`, `tool_arguments`,
`tool_trace_structure`, `agent_trace`, `trace_invariants` — and resolution is
**family-level replacement**:

- A family the suite does **not** mention is inherited from the top level.
- A family it **does** mention replaces the top-level one wholesale. There is no
  deep merge and no per-evaluator-name merge: the suite's list is the suite's list.
- Writing the family as `[]` or `null` is how a suite **removes** a family it
  would otherwise inherit. (Absent and `null` are different instructions here.)

So in the example above `main_chat` is scored with its own two tool evaluators
plus whatever `semantic` / `llm_judge` / `structural` the top level declares,
and `briefing` is scored with the top level alone. The same resolution feeds
`evaluate`, the HTML report and the hosted bundle, so what was scored is what is
reported. A run launched with a raw `--suite <path>` (no `--suite-name`), or with
a name that has no `suites:` entry, resolves to the top-level block.

### `managed`

`capture sync` regenerates a managed suite's entry — `path` and
`evaluators` — from what that suite's captures contain, so hand edits inside the
marker-delimited region are overwritten. The one exception is
`evaluators.trace_invariants`: no capture can derive a team's rules, so sync
copies a managed entry's existing `trace_invariants` block into the entry it
regenerates, verbatim. The marker comment above the region still says hand edits
are overwritten; that text cannot change (sync finds the region by matching it
exactly), so this paragraph is where the exception is written down. Every other
hand edit is overwritten. Set `managed: false` to freeze an entry:

```yaml
suites:
  main_chat:
    managed: false
    source: captured
    path: .evalshift/suites/main_chat/golden.jsonl
    evaluators:
      tool_arguments:
        - name: routing_args
          against: expected
          strategies:
            amount_usd: numeric
```

Sync then leaves the entry alone and prints the block it *would* have written,
so you can diff your edits against the current derivation. Syncing one suite
never touches another's entry either way.

## `retention`

Bounds how much run history accumulates under `.evalshift/runs/`. Every `run` / `compare` invocation
writes a fresh `r_<date>_<suite>_<hex>/` directory, so without a cap they pile up indefinitely.
After each **completed** run the orchestrator prunes old directories automatically; an in-progress
run and the run that just finished are never touched.

```yaml
retention:
  max_runs_per_suite: 20   # keep the 20 newest runs per suite; 0 disables count pruning
  run_ttl_days: 30         # (optional) also delete runs older than 30 days
```

| Field                | Type      | Default | Description |
| -------------------- | --------- | ------- | ----------- |
| `max_runs_per_suite` | int (≥ 0) | 20      | Keep at most this many run directories **per suite** (grouped by the suite slug in the run id), evicting the oldest by mtime. `0` disables count-based pruning. |
| `run_ttl_days`       | int (≥ 1) | (none)  | Also evict run directories older than this many days. Omit to disable age-based pruning. |

Pruning is grouped per suite, so a rarely-run suite isn't evicted just because another suite is
busy. The two rules combine (a run is deleted if **either** applies). `EVALSHIFT_MAX_RUNS` overrides
`max_runs_per_suite` from the environment (`0` / `none` / `unlimited` disables count pruning), which
is handy in CI where you don't want to keep any history.

Clean up on demand with `evalshift runs clean` — it applies the same rules with explicit overrides:

```shell
evalshift runs clean --dry-run            # preview what would be deleted
evalshift runs clean --keep 5             # keep the 5 newest per suite
evalshift runs clean --older-than 14      # delete runs older than 14 days
evalshift runs clean --suite main_chat    # restrict to one suite
```

`--keep` beats `EVALSHIFT_MAX_RUNS`, which beats the config value. `runs clean` works even without a
valid `evalshift.yaml` (it falls back to the defaults), so it's always available for disk cleanup.

## `captures`

Where the capture SDK's recordings live when they are not on this machine. Omit the block
entirely for the default: captures are read from `.evalshift/captures/` (or `EVALSHIFT_DIR`)
on local disk and nothing here is needed.

Production agents on Fargate, Lambda or Kubernetes lose their disk when they stop, so the
SDK can ship captures to an object store you own instead (`EVALSHIFT_SINK=<uri>` in the
agent's environment — see [Capture SDK](sdk.md)). Name the same store here and the CLI
pulls new captures down before it promotes them, with no flags to remember:

```yaml
captures:
  store: s3://acme-evals/support-agent
```

| Field   | Type          | Default | Description |
| ------- | ------------- | ------- | ----------- |
| `store` | string (URI)  | (none)  | Object store the SDK writes to. `capture sync` and `capture list` fetch new captures and toolset sidecars from it into the local directory first; `capture fetch` does only that step. |

CLI versions before this release reject the `captures:` key (the config is strict), so bump
the CLI — and a pinned GitHub Action's `evalshift-version` — before adding it.

Accepted URI forms — the same grammar the SDK uses:

| Form | Store | Install | Credentials |
| --- | --- | --- | --- |
| `s3://<bucket>/<prefix>` | Amazon S3; MinIO, Cloudflare R2, Backblaze B2 via boto3's `AWS_ENDPOINT_URL` | `pip install "evalshift[s3]"` | IAM role in CI (OIDC), `aws sso login` locally |
| `gs://<bucket>/<prefix>` | Google Cloud Storage | `pip install "evalshift[gcs]"` | Workload Identity in CI, `gcloud auth application-default login` locally |
| `az://<account>/<container>/<prefix>` | Azure Blob Storage | `pip install "evalshift[azure]"` | Managed Identity / federated credential in CI, `az login` locally |

`<prefix>` is optional. Credentials never go in the URI: a value containing `@` or `?` fails
to load, naming the accepted forms. A rejected value is never echoed back (it may be a pasted
key or connection string); the error names at most its scheme. Under the prefix the layout is exactly the local one —
`captures/<suite>/cap_<hex>.json` and `toolsets/<hex>.json` — so the bucket is a mirror of
`.evalshift/`, not a different format.

What the fetch does: it lists the bucket, downloads only objects missing locally (captures
are immutable and toolsets content-addressed, so "exists" means "current"), skips captures
already promoted into a suite (so `capture clean` never causes a re-download), and writes
each file atomically. `--since 24h` (or `7d`, `30m`, an ISO date) limits a fetch to recently
written captures; `--offline` skips it and works with what is already local. Retention in the bucket
is yours to set with a lifecycle rule; the SDK's `max_captures` does not apply there.

## Capture lifecycle

The companion `evalshift-sdk` package records real agent runs to
`.evalshift/captures/<suite>/<capture_id>.json` (set `EVALSHIFT_DIR` to relocate
the base). The CLI consumes them via the `capture` commands:

```shell
evalshift capture list                          # see what the SDK recorded
evalshift capture fetch                         # (captures.store only) mirror new captures locally
evalshift capture sync --input-var query        # promote every capture + wire suites:
evalshift run --suite-name support_agent --yes  # score a candidate model against it
evalshift capture clean                         # prune already-promoted captures
```

With [`captures.store`](#captures) configured, `list` and `sync` run the same fetch first
and print one summary line (`fetched 12 capture(s) and 1 toolset(s) from s3://…`); pass
`--offline` to skip it. `promote`, `diff` and `clean` operate on the local mirror only.

`capture clean [<suite>]` deletes promoted captures by default (`--promoted`);
`--all` deletes every capture, promoted or not. It never touches promoted
suites, then offers to sweep toolset sidecars no surviving capture or suite
references. Each deletion asks first; `--yes`/`-y` skips both prompts.

`evalshift capture sync` is the one-shot path: it promotes **every** capture
under `.evalshift/captures/` into golden suites at
`.evalshift/suites/<suite>/golden.jsonl` **and** injects the resulting
`suites:` block into `evalshift.yaml` between the managed marker comments.

Each suite's entry is generated from what that suite's own rows contain, tool
evaluators included, so nothing has to be wired by hand:

- No row was offered a toolset → no `evaluators:` block at all; the suite
  inherits the top level, and no tool evaluator scores an empty denominator.
- Any row was offered a toolset → `tool_selection: [{name: routing, conformance:
  expected, divergence: set}]`.
- Any row recorded tool-call arguments → `tool_arguments: [{name: routing_args,
  against: expected}]`. No `strategies:` block: the default `auto` strategy
  already grades free text by meaning rather than by bytes.

The generated names (`routing`, `routing_args`) are stable by contract — reports
key on evaluator names across runs, so regenerating a suite must not rename what
it already wired. `structural` is deliberately not derived: nothing in a capture
says what shape an answer must have. Sync regenerates only the suites it just
promoted and carries every other entry in the region forward verbatim, and
`managed: false` freezes an entry entirely (see [`suites`](#suites)). A
regenerated entry keeps its hand-written `evaluators.trace_invariants` (see
[`managed`](#managed)).
Captures with no recorded events are skipped, and captures whose replayed
content duplicates an already-promoted case (or an earlier capture in the
same run) are skipped too — duplicate examples inflate *n* and corrupt the
paired statistics (`--keep-duplicates` opts out). The dedup set is seeded
from the cases already in the suite dir, so re-syncing after recording more
captures can't slip a duplicate past it. Useful flags: `--input-var`
(default `input`), `--suite <name>` to filter to one suite, `--tag`,
`--names-only`, `--tool-count`, `--strict-args`, `--rounds first|all`
(`first`, the default, scores single-shot replay against round 1; `all`
carries the recorded tool results so `run` replays every round teacher-forced
— see [Agent rounds](agents.md#agent-rounds-and-what-a-replay-can-reproduce)),
`--force`/`-f` to overwrite
existing suite files, and `--print` to preview the wiring without writing
(`--write` is the default). After syncing, run the whole pipeline against a
named suite with `evalshift compare --suite-name <suite>` (it mirrors
`evalshift run --suite-name`).

To promote a single capture instead of all of them, use
`evalshift capture promote`:

```shell
evalshift capture promote cap_abc --as case1 \
    --input-var query                           # → .evalshift/suites/<suite>/case1.json (+ golden.jsonl)
```

`promote` derives a golden case from the recorded run: the captured tool calls
become `expected_tools`, the final output becomes `expected`, and (best-effort)
the first model input becomes `inputs`. Because a capture stores only a one-way
`input_hash`, structured/opaque inputs can't always be recovered — use
`--input-var` for single-string prompts, or edit the generated case file.
