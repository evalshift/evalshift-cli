# Evaluators

An evaluator scores the (source_output, target_output) pair for one
example. Every evaluator returns a `PairedScore` with both halves and
a `delta = target_score - source_score`. Negative deltas mean the
target regressed; positive deltas mean it improved.

Every evaluator config also accepts `blocking: true|false` (default
`true`): **blocking** evaluators feed the migration-policy verdict and
CI gates; **advisory** (`blocking: false`) evaluators are computed and
reported separately but can never fail a run on their own. The `init`
scaffold ships semantic and judge as advisory deliberately — at small
suite sizes their noise would gate the verdict. See
[Configuration](configuration.md#blocking-every-evaluator).

When an evaluator's own measurement breaks (judge call fails, embedding
call fails), the record is stored as **errored and excluded from the
statistics** — not silently scored neutral. Upstream *model-call*
failures and truncated calls are recorded the same way: an errored row
(a 0.5/0.5 placeholder with `error` set), kept in `scores.jsonl` for
inspection and excluded from slicing, the paired tests and the policy
rates. The run always completes.

EvalShift ships four families (plus `agent_trace` for
[imported external traces](traces.md), and `trace_invariants` for
[hand-written trace rules](#trace-invariants)):

## Structural (deterministic, free)

Fast checks that depend only on the output shape — no API calls.

* **`json_schema`** — does the output parse as JSON and validate
  against a schema you provide? 1.0 / 0.0.
* **`regex`** — does the output match a pattern? 1.0 / 0.0.
* **`length`** — is the output within `[min_chars, max_chars]`?
  1.0 inside, distance-decayed outside.

Use these whenever you can — they're free, deterministic, and cover
most real regressions.

## Semantic (cheap)

* **`semantic.cosine`** — embed both outputs with a configurable
  embedding model, compute cosine similarity, then frame the result
  as a target-preservation score: source = 1.0, target = similarity.
  Delta < 0 means the target drifted in meaning from the source.

**What "expected" means here.** The yardstick is the *source model's*
output, not a reference answer: the suite's `expected` field is never
read by the text evaluators. So the score measures drift, not
correctness — a target that answers correctly in different words reads
as drift, and a target that repeats the source's mistake reads as
equivalent. That is why `evalshift init` ships semantic as advisory
(`blocking: false`, see the [FAQ](faq.md#why-does-a-hand-written-config-block-on-semantic-when-init-does-not))
and why correctness belongs to an `llm_judge` criterion that names the
property you care about.

Use when:
* You don't have a clean structural check.
* You want to detect "wandered off" outputs that still look fine
  syntactically.

On an agent turn where both models answered with tool calls and no
prose there is nothing to embed, so the evaluator writes no record at
all instead of erroring or inventing a score. Only one side empty still
scores — a target that went silent is a real regression. Because the
empty side can't be embedded (providers 400 on empty input), the pair
scores 0.0 similarity by definition with no embedding call, gated by
`min_similarity` as usual, with `empty_side` metadata naming which side
was silent. Either direction scores the same way.

Don't use when:
* The target is intentionally meant to differ from the source (e.g.
  you're migrating from a verbose model to a terse one). The
  similarity will look low and you'll get a confusing "regression"
  signal.

## LLM-as-judge (most expensive)

* **`llm_judge.<criterion>`** — ask a strong model "which output
  better satisfies this criterion?" with random A/B ordering to
  reduce positional bias. Verdict maps cleanly to (source, target)
  scores.

Use when you can articulate the difference you care about as a
sentence ("which output preserves more factual detail?"). Multiple
`llm_judge` entries are allowed — each becomes its own evaluator.
Tool-only turns (both outputs empty) are skipped without spending a
judge call.

Pick the judge from a **third model family**. A judge prefers output
that reads like its own (self-preference bias), and A/B randomisation
does nothing against that. `doctor` and `validate` warn when a
`judge_model` shares a provider with the source or target, and the
report notes it above the verdict when such a judge contributed rows —
advisory only; the `init` scaffold ships a same-provider judge on
purpose so a first run needs one API key.

## Tool-call evaluators (agent migrations)

For a dispatched example whose own toolset (`toolset_ref` or inline `tools`
— see [Agent migrations](agents.md#suite-ground-truth)) is non-empty,
EvalShift parses each model's response into a provider-agnostic `ToolTrace`
and scores three orthogonal dimensions:

* **`tool_selection`** — *which* tools fire? Two independent axes,
  one record each, because a migration asks both questions and the
  answers differ:
  * `conformance` — did each side match the suite's ground truth?
    `expected` (default; matches `example.expected_tools`
    order-preserving), `expected_set` (same, order-insensitive), `off`.
    Each side is graded absolutely, so both can fail at once and the
    delta stays 0 — the migration did not cause a failure both models
    share. When both miss, the record is tagged
    `TOOL_GROUND_TRUTH_MISS`: ground truth captured from the source
    model that the source model then fails means a broken harness.
  * `divergence` — did the target do what the source did? `set`
    (default; Jaccard on the tool-name sets), `exact` (sequence
    equality), `first` (first call only), `off`. Source is its own
    baseline at 1.0, so drift is a negative delta — a regression.
  `set` is the divergence default rather than `exact` so reordered
  identical calls do not read as drift. Configure `severity_floor: high`
  so a regression here can never be downgraded.

  The two axes render as **separate rows** in `report.html`, each
  labelled with its slug and with what it compares — they answer
  different questions against different baselines, and averaging or
  confusing them restates the bug they exist to catch. An axis on which
  every pair was a `TOOL_GROUND_TRUTH_MISS` is headlined **Ground truth
  missed by both**, never "Equivalent": the delta really is zero, but
  that is a fact about your suite, not about the migration.

  On a teacher-forced multi-round replay (a suite promoted with
  `--rounds all`, see [Agent rounds](agents.md#agent-rounds-and-what-a-replay-can-reproduce))
  both axes score **per round** — conformance against
  `expected_tool_rounds[k]`, divergence target-vs-source within round *k* —
  and the record's scores are the mean over the replayed rounds, with each
  round's names and scores under `metadata.rounds`. A round with no ground
  truth in which neither side called anything does not enter the mean.
  Single-shot pairs score exactly as before and carry no `rounds` key.

  When the **source** model misses conformance on half or more of at
  least four rows, `evalshift evaluate` says so in red, at `doctor`
  volume, naming the rate: the expectations were captured *from* the
  source model, so a source that fails them means the run measured your
  harness and no verdict beside it describes the target model. See
  [methodology](methodology.md#a-source-model-that-fails-its-own-ground-truth).
* **`tool_arguments`** — *what* did the model pass? Greedy match by
  `(tool_name, sequence_index)`, then per-field strategies. Use when arg
  drift matters (e.g. the model still calls `issue_refund` but the amount
  is wrong). On a multi-round replay calls are paired **within** a round
  (a right call in the wrong round is a miss) and the score is the mean
  over rounds that had something to score.

  Fields you do not name in `strategies` are scored by `default_strategy`,
  which defaults to **`auto`**: a ladder that tries normalized string
  equality first (case and whitespace differences are not wrong values),
  then dispatches on the field's declared type in the toolset the example
  carries (identifiers, enums, booleans and `date-time`/`uuid`/`email`
  formats → `exact`; numbers → `numeric`; objects and arrays → `subset`),
  and grades whatever is left — free text — by embedding similarity, or by
  `difflib` ratio when no `evaluators.semantic` block lent it a model. The
  point is that free-text arguments get partial credit by default: under
  the old `exact` default a reworded search query scored 0.0 and read as a
  regression. Set `default_strategy: exact` to restore byte equality;
  per-field `strategies` entries always win.

  A regression here is stamped `ARGUMENT_VALUE_DRIFT` only when the target
  scored **below the source**. Under `against: expected`, both models
  missing the recorded ground truth by the same margin leaves the delta at
  0 and carries no drift label: that is a fact about your suite, not a
  migration defect, and the same finding is already reported as a
  ground-truth problem. A ground-truth field *neither* side produced is
  dropped from the denominator on both sides (disclosed as
  `unmeasured_fields` in the record's per-call metadata) — a stale
  expectation would otherwise cap the call below 1.0 forever.
* **`tool_trace_structure`** — *how* did it sequence them? Sub-scores:
  call count, parallelism, refusal alignment, expected count.
  Refusal mismatches force `severity_floor: high`. Use to catch
  call-count explosions or sudden parallel/serial flips. On a multi-round
  replay the call count is over the whole trace and parallelism is
  compared round by round (fanning out is a property of one response);
  `details.rounds_replayed` records how many rounds each side made.

The seven agent-migration failure modes each map to one of these
three: dropped tool / wrong tool → `tool_selection`; arg drift /
sequence reorder → `tool_arguments`; parallel↔serial flip / loop
divergence / refusal regression → `tool_trace_structure`.

Enable `tool_selection` and `tool_arguments` together to catch both
dropped-tool and argument-drift regressions in one run. Turn on
`tool_trace_structure` once you want call-count, parallelism, and
refusal changes scored separately.

You rarely wire the first two by hand: `evalshift capture sync` writes
`tool_selection` and `tool_arguments` into each suite's own `suites:` entry
based on what that suite's captures contain, so a tool-free suite gets no
tool evaluators at all. See
[Agent migrations](agents.md#configuration) and
[Configuration → per-suite evaluators](configuration.md#per-suite-evaluators).

### Trace invariants

Every other tool evaluator treats "the source did it" as correct:
`tool_selection` divergence, `tool_arguments` against the source and
`agent_trace`'s `dangerous_tools` all measure how far the target moved from
the source, so a wrong call both models make scores as agreement.
`trace_invariants` judges **both** sides against rules your team wrote —
`forbidden` tools, `required` tools, an `order` between two tools, a
`call_count` (min, max or exact per tool) and `arguments` that must satisfy a
JSON Schema. A rule the target breaks counts against it whatever the source
did, and `migration_policy.max_invariant_violations` (default `0`) fails the
run on it, even on a suite too small for statistics.

Write rules only for invariants where a silent miss is expensive: auth before
a write, a deprecated endpoint that must stay unused, at most one charge,
argument bounds. They are a short contract next to the config, not a second
copy of the suite; production traces stay the broad corpus that the other
evaluators compare.

Keep the split clean. Captured `expected_tools` are the *recording*, and
`conformance: expected` grades their order as recorded — including order that
was incidental. An ordering that *matters* belongs in an `order` rule, where it
is stated once, owned, and checked on every in-scope trace, not in the
recording. Configuration, rule semantics and scoring:
[Configuration → `evaluators.trace_invariants`](configuration.md#evaluatorstrace_invariants).

## Mixing evaluators

You can configure several at once. They all run on every (prompt,
example) pair, and the analysis layer treats each as a separate
comparison (so BH correction adjusts for the multiple-test count
correctly).

A typical migration uses:
* 1–2 structural evaluators (cheap baseline checks)
* 1 `semantic.cosine` (catches semantic drift)
* 1 `llm_judge` per criterion the team cares about

## Cost considerations

Per (prompt, example) pair, each evaluator means:

| Evaluator              | Cost                                    |
| ---------------------- | --------------------------------------- |
| structural.*           | $0 (no calls)                           |
| semantic               | 2 embedding calls                       |
| llm_judge              | 1 judge model completion                |
| tool_selection         | $0 (compares parsed traces only)        |
| tool_trace_structure   | $0 (compares parsed traces only)        |
| trace_invariants       | $0 (checks parsed traces only)          |
| tool_arguments         | $0 without an `evaluators.semantic` block (free text uses `difflib`). With one, embedding calls per free-text or `semantic`-strategy field (cached) |

A 100-example suite with 1 prompt and 4 evaluators (2 structural +
1 semantic + 1 judge) is:

* Run: 200 model calls (100 × 2 models)
* Evaluate: 200 embedding calls + 100 judge calls

LiteLLM's pricing data drives the pre-flight estimate; the local
SQLite cache absorbs identical re-runs, examples that offer tools and
evaluate-stage embedding and judge calls included. Evaluate dispatches its calls under
`defaults.concurrency`, same as the run stage.
