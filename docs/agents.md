# Agent migrations

EvalShift compares **agent behaviour** — which tools the model called,
what arguments it passed, and, within a response, in what order and
whether in parallel — across two model versions.

By default `evalshift run` makes **one model call per example** and scores
that response against the first tool-emitting round of the recording. A
suite promoted with `--rounds all` opts into **teacher-forced multi-round
replay**: every recorded round is replayed with the recorded tool results
fed back, and each round is scored against its own ground truth. See
[Agent rounds](#agent-rounds-and-what-a-replay-can-reproduce).

The killer scenario it catches:

> A team migrates a customer-support agent from Gemini 2.5 Flash to
> 3.1 Flash-Lite. The new model silently stops calling
> `notify_security_team` on sensitive requests. Text-only eval reports
> green; EvalShift marks it CRITICAL and blocks the migration.

## How it works

* **`ToolTrace` data model**: provider-agnostic, populated from
  Anthropic / OpenAI / Gemini responses.
* **Four new evaluators**: `tool_selection`, `tool_arguments`,
  `tool_trace_structure`, and hand-written `trace_invariants` (both sides
  judged against your rules, not against each other).
* **Suite extension**: optional `expected_tools`, `expected_tool_count`,
  `expected_no_tools`, `expected_parallel` per example, plus a required
  `toolset_ref` or inline `tools` — see [Suite ground truth](#suite-ground-truth).
* **HTML report**: side-by-side trace diffs in place of text panes
  for tool-evaluator regressions.
* **Hosted run-detail**: the same trace — tool calls, arguments, final text,
  round markers — is visible on the hosted run-detail page after `evalshift
  push`, not just in the local HTML report. See [Hosted EvalShift](hosted.md).

## Walkthrough

`examples/agent/` in this repo is a complete, checked-in agent project —
six customer-support tools (`search_orders`, `lookup_customer`,
`issue_refund`, `update_order_status`, `send_email`,
`notify_security_team`) and a golden suite across five slices
(`security`, `routine`, `refund`, `customer_lookup`, `text_only`), each
row carrying its own `toolset_ref`:

```bash
cd examples/agent
export GOOGLE_API_KEY=<google-api-key>
evalshift run --yes --from gemini-2.5-flash --to gemini-3.1-flash-lite-preview
RUN_ID=$(ls -t .evalshift/runs/ | head -1)
evalshift evaluate "$RUN_ID"
evalshift analyze "$RUN_ID"
evalshift report "$RUN_ID" --open
```

`evalshift run` resolves each golden-suite example's own toolset
(`toolset_ref` or inline `tools`, see [Suite ground truth](#suite-ground-truth))
and dispatches through the live agent path automatically for any example
whose toolset is non-empty — every such `Call` row in `raw.jsonl` carries a
parsed `ToolTrace`, and the configured `tool_*` evaluators score against the
per-example `expected_tools` ground truth.

For your own agent, the recommended path is capture-first: instrument it
with the [evalshift-sdk](https://github.com/evalshift/evalshift-sdk),
exercise it to record captures, then `evalshift capture sync` promotes them
into a golden suite carrying the toolset your production agent actually
offered — see [Getting started](getting-started.md).

## Configuration

A minimal agent config:

```yaml
version: 1
prompts:
  - id: routing
    detection: python_string
    path: prompts.py
    variable: AGENT_SYSTEM_PROMPT
    variables: [query]

defaults:
  source_model: gemini-2.5-flash
  target_model: gemini-3.1-flash-lite-preview
  judge_model: gemini-3.1-pro-preview

evaluators:
  # Top-level: what every suite is scored with. Tool evaluators do not belong
  # here — `capture sync` writes them per suite (see below). Hand-written
  # `trace_invariants` are the exception: top level is fine, and sync keeps
  # them on the managed entries it rewrites.
  semantic:
    embedding_model: gemini/gemini-embedding-001
    blocking: false

# >>> evalshift suites (managed by `evalshift capture sync`) >>>
suites:
  main_chat:
    source: captured
    path: .evalshift/suites/main_chat/golden.jsonl
    evaluators:
      tool_selection:
        - name: routing
          conformance: expected     # grade each side vs example.expected_tools
          divergence: set           # ...and the target vs what the source did
      tool_arguments:
        - name: routing_args
          against: expected         # score both sides vs the recorded arguments
      # Optional — not derived, add by hand under `managed: false`:
      # tool_trace_structure:
      #   - name: routing_structure
# <<< evalshift suites <<<
```

The `suites:` region is generated: `evalshift capture sync` reads what each
suite's rows actually contain and writes that suite's own evaluator block —
`tool_selection` when any row was offered a toolset, `tool_arguments` when any
row recorded call arguments, and nothing at all for a suite whose captures never
called a tool. That last case is the point: a tool evaluator pointed at a
tool-free suite scores an empty denominator, which the policy reads as an
*inconclusive* gate rather than as "not applicable here". A family a suite
declares replaces the top-level one wholesale; families it does not mention are
inherited. See
[Configuration → per-suite evaluators](configuration.md#per-suite-evaluators).

Hand edits inside the markers are regenerated away on the next sync. To keep
them — a `severity_floor: high` on `routing`, a per-field strategy — set
`managed: false` on that suite's entry; sync then prints what it would have
written instead of writing it. The one exception is a hand-written
`evaluators.trace_invariants` block: no capture can derive a team's rules, so
sync carries it into the regenerated entry verbatim, even though the marker
comment says hand edits are overwritten. See
[Configuration → `managed`](configuration.md#managed).

> **Note:** `structural.length` is intentionally **not** in the
> scaffolded config. Agent runs frequently produce empty `final_text`
> (the model returned only tool calls), which makes the length
> evaluator score 0/0 across every routine row — pure noise. Add it
> back manually only for prompts that produce text.

Nothing in `evalshift.yaml` wires a toolset to a prompt — dispatch reads it
off each golden-suite *example* instead (`toolset_ref` or inline `tools`, see
[Suite ground truth](#suite-ground-truth) below), so the same prompt can
legitimately dispatch some examples with tools and others without, in one
run. [`examples/agent/tools.yaml`](https://github.com/evalshift/evalshift-cli/blob/main/examples/agent/tools.yaml) is just this project's human-readable record of what
those tools are; it accepts either Anthropic-shape (`name` / `description` /
`input_schema`) or OpenAI-shape (`{ "type": "function", "function": {...}
}`) entries — `evalshift run` serialises whatever a toolset resolves to in
the right shape per provider.

## Suite ground truth

```jsonl
{"id": "ex_security_01", "inputs": {"query": "..."}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team"}], "toolset_ref": "sha256:1a2b3c..."}
{"id": "ex_text_01", "inputs": {"query": "what is your refund policy?"}, "tags": ["text_only"], "expected_no_tools": true, "tools": []}
```

Every example carries a toolset — `toolset_ref` (a pointer to a
`<base>/toolsets/<hex>.json` sidecar; what `capture promote`/`sync`
write) or inline `tools` (what a hand-authored suite uses; `[]` is a
real "no tools offered" value), exactly one of the two. `expected_no_tools`
means "tools were offered and none were called" — it is never set when
the example's toolset is empty, so it never asserts something no model
could have failed.

You rarely write these by hand — `evalshift capture promote` /
`capture sync` derive `expected_tools` (and friends) from recorded
production behaviour; `--strict-args`, `--names-only`, and
`--tool-count` control how strict the derived expectations are. See
[Configuration → Capture lifecycle](configuration.md#capture-lifecycle).

### Agent rounds and what a replay can reproduce

A captured agent turn is usually a **loop**: the model calls tools, reads the
results, calls more tools, then answers. `evalshift capture promote` / `sync`
group those calls into rounds — one per recorded `model_call` — and keep
**round 1** as `expected_tools`. Every round is preserved on the case as
`expected_tool_rounds`.

**Default (`--rounds first`): single-shot replay.** `evalshift run` issues one
model call per example and does not feed tool results back, so a candidate
model can only ever produce round 1, and round 1 is the only round it is
scored against. Scoring it against round-2 calls would record a regression no
model could avoid. A multi-round capture promoted this way prints a warning
naming the later calls it will not replay.

**`--rounds all`: teacher-forced multi-round replay.** Promotion also carries
the recorded tool results on the case as `tool_result_fixtures`, aligned by
position with `expected_tool_rounds`. Each round's calls are paired with that
round's `tool_result` events by `call_id` first (this works for both executed
`tool_call` events and model-`requested_tool_calls`), then by tool name among
the results recorded in the same round; a name match never reaches across a
`model_call` boundary. `run` then replays the example round by round:

* Round *k* sees the conversation prefix (`history`, if any), the rendered
  prompt, and then the **recorded** rounds `1..k-1` — the recorded assistant
  tool calls and the recorded results, as ordinary `assistant` / `tool`
  messages. The candidate's own calls are never fed back: source, target and
  the recording all saw byte-identical context in every round, which is what
  makes a per-round comparison fair. It is the same contract the multi-turn
  `history` prefix already uses.
* A result is sent verbatim when it was a string, as JSON otherwise; a recorded
  tool error is sent as `{"error": "..."}` — the recorded agent saw the failure
  too, and its next round is the ground truth for what to do about it.
* The replay covers every round the fixtures cover **plus the round after it**.
  When every tool round is covered, that last round is the *answer* round: the
  recorded agent called nothing and produced its final text, so the candidate's
  text is what the text evaluators compare, and a candidate that keeps calling
  tools when it should have answered is caught.
* Fixtures cover rounds `1..m` where round `m+1` is the first with a call that
  has no recorded result (an app that records `@capture.tool` calls but no
  results). Promotion warns — `round m+1 has N tool call(s) with no recorded
  result; replay will cover rounds 1..m` — and the rounds after it stay on the
  case as `expected_tool_rounds` but are never replayed. When round 1 itself is
  uncovered, replay stays single-shot and the warning says so.
* One `raw.jsonl` row per example per model, as before: tokens, cost and
  latency are summed over rounds, `text` is the last round's answer, and the
  trace carries every round's calls tagged with their `round_index`. A model
  error in round *k* fails the example with `round k/n: <error>`; the rounds
  that did complete are discarded, so a partially replayed example is an
  unmeasured one, not a half-scored one.
* Scoring is **per round**: `tool_selection` conformance grades round *k*
  against `expected_tool_rounds[k]` (and "called nothing" for the answer
  round); divergence compares round *k* of the target to round *k* of the
  source; `tool_arguments` pairs calls within a round, so a right call in the
  wrong round is a miss. Each record's scores are the mean over the replayed
  rounds, with the per-round detail under `metadata.rounds`. Because the mean
  drops below 1.0 as soon as one round differs, `max_tool_divergence` counts an
  example as diverged if **any** replayed round diverged.
* The cost estimate counts one call per replayed round; the progress bar still
  counts examples.
* Each round is its own response-cache entry, keyed on exactly what that round
  sends: the prompt, the recorded rounds and fixture results fed back, the
  tool list exactly as sent and in order (including `strict`),
  `generation_config` and the round index. A repeat run is served from the
  cache; editing round *k*'s fixtures re-sends only the rounds after it, and a
  round that errored is re-sent while the rounds before it are not. The
  example's row counts as cached only when every round was a hit; a partly
  cached row's latency is left out of the live latency figures, since some of
  it was measured on an earlier run.

`expected_tools` is `expected_tool_rounds[0]` under both settings. `--rounds
all` no longer flattens every round into `expected_tools` — that yardstick was
only ever right for comparing against an externally produced multi-round
trace, which is the `agent_trace` evaluator's job and reads imported traces.
`--tool-count` under `--rounds all` pins the total over the rounds the replay
actually reaches.

### Tool-choice constraints are replayed too

Production rarely offers tools and leaves it at that. If the app forced a tool
(`tool_choice`), banned parallel calls (`parallel_tool_calls: false`), or
demanded schema-exact arguments (`strict: true` on a tool), replaying without
those constraints measures a different call than the one that ran.

The SDK records all three, and `evalshift run` sends them:

| Recorded as | Where | Replayed as |
| --- | --- | --- |
| `tool_choice` — OpenAI string (`"auto"` / `"none"` / `"required"`) or `{"type": "function", "function": {"name": ...}}` | `generation_config` | forwarded as-is |
| `tool_choice` — Anthropic object (`{"type": "auto"\|"any"\|"tool"\|"none", "name"?, "disable_parallel_tool_use"?}`) | `generation_config` | `any` → `required`, `tool` → the named-function object, `disable_parallel_tool_use` inverted into `parallel_tool_calls` |
| `tool_config` — Gemini (`{"function_calling_config": {"mode": "AUTO"\|"ANY"\|"NONE", "allowed_function_names"?: [...]}}`) | `generation_config` | the matching OpenAI string; exactly one allowed name becomes a named-function choice, several degrade to `required` |
| `parallel_tool_calls` | `generation_config` | forwarded as-is (wins over the value inferred from an Anthropic `tool_choice`) |
| `strict: true` on a tool | the toolset sidecar or inline `tools` | `function.strict` for OpenAI targets, top-level `strict` for Anthropic targets |

Everything goes out in OpenAI-style form and LiteLLM maps it per provider —
into Anthropic's `tool_choice` object (carrying `disable_parallel_tool_use`)
and into Gemini's `toolConfig`. Because the constraint is normalised, a
capture from one provider replays meaningfully against a target on another,
which is the whole point of a migration run.

Two constraints have no equivalent on a Gemini target: **`parallel_tool_calls`**
(`generateContent` has no such switch) and a tool's **`strict`** flag (Gemini
function declarations have no strict mode). LiteLLM accepts both and reports
them as supported, then discards them while building the Gemini request body,
so no capability probe can see the loss — EvalShift keeps a short hard-coded
table of these gaps instead. Neither is dropped quietly: an affected arm is
recorded at run start in `state.json` under `dropped_params`, alongside the
parameters a model genuinely does not accept, and gets the report's
**Constraints not honoured** banner. The tool flag is recorded under the
pseudo-parameter name **`tools.strict`**, because it is a field on the `tools`
array rather than a generation parameter. See
[Configuration → `fail_on_dropped_params`](configuration.md).

A recorded `tool_choice` on an example with no toolset is dropped with a
warning: there is nothing to constrain. That one stays out of `dropped_params`
— it says something about the *suite* (an example whose capture pinned tool use
but whose toolset is empty), not about either target's capabilities, and it
would otherwise be recorded identically against both arms. Recorded keys the
runner does not translate at all (`top_p`, `max_tokens`, …) get one warning
listing them.

### Where `expected` text comes from

`example.expected` is recovered from the capture's `final_output` event when
there is one, and otherwise from the last `model_call` that produced text — on
an agent turn that is the reply the user saw, after the tool round-trips.
Captures with neither are promoted without text ground truth and
`capture sync` says so.

The fallback exists because only the SDK's LangChain adapter emits
`final_output`; the manual capture API has no way to. Without it every
manually instrumented project promoted `expected: null` while the reply sat in
the last `model_call` the whole time. A non-string `output` is left alone
rather than stringified into ground truth nothing produced.

### Requested calls are the ground truth when they were captured

A capture records three things that are easy to conflate:

| | Where it lives | What it is |
| --- | --- | --- |
| **offered** | `model_call.toolset_ref` / `.tools_offered` | the tools passed *to* the model |
| **requested** | `model_call.requested_tool_calls` | the calls the model asked for *in its response* |
| **executed** | the `tool_call` / `tool_result` events | the calls the app actually *ran* |

Promotion prefers **requested**. The executed calls have already passed through
the application — its filtering, retries, re-ordering, and its own function
signatures — so they are evidence of what the *app* did; a golden case has to
state what a *model* should produce. Each `model_call` carries its own
response's requested calls, so each `model_call` simply *is* a round (rounds
that requested nothing are dropped, exactly as tool-less executed rounds are),
and `--rounds` / `--tool-count` / `--names-only` behave identically either way.

The promoted case records which yardstick was used as
`promotion_source: "requested" | "executed"`, so a report can say what a row
measures. Two cases fall back to the executed calls:

- **A capture with no requested calls at all** — written before the SDK
  recorded them. This is the legacy path below, and it is silent.
- **A capture where only *some* `model_call` events carry them.** The whole
  capture falls back and `capture sync` says so, rather than score half the
  trace against one yardstick and half against another.

The second case is the one to watch, because it is a recording gap rather than
anything about SDK versions. The SDK records the field only when your code
passes it, and an omitted argument becomes `null`:

> **Every `model_call` in the run must carry `requested_tool_calls` for the
> capture to be scored against requested calls.** Pass `[]` for a round in
> which the model requested no tools — `null` means *not recorded*, not
> *nothing requested*, and the two cannot be told apart after the fact.

The usual way a capture goes mixed is an agent that passes the field on its
tool-picking calls and omits it on the final, text-only one, where there are no
tool calls to hand over. Pass `[]` there.

When requested calls are present *and* the executed ones disagree — a different
tool, a different argument value, a different round grouping — the requested
calls win and promotion warns, naming the tools on both sides. A difference is
often legitimate (the app filtered or rewrote a call); the warning is there so
you can tell that from a gap in what was captured.

### What the recorded run cost

The SDK never prices anything: a `model_call`'s `cost_usd` is `0.0` unless your
own instrumentation set it, and the provider client wrappers record
`input_tokens` / `output_tokens` but leave cost at 0 by design. Promotion fills
the gap. Each promoted case file carries `cost_usd` — the run's `model_call`
events summed — and `cost_source`, saying where the figure came from:

- `"recorded"` — every non-zero part of the sum is what your instrumentation
  set. A recorded cost is kept exactly as recorded, never re-estimated.
- `"estimated"` — at least one `model_call` recorded tokens but no cost, and the
  CLI priced it from litellm's price table for that call's own `model_id` (an
  alias or provider-prefixed id resolves through the model registry first). A
  run mixing recorded and estimated calls is tagged `estimated`: the figure is
  only as certain as its least certain part.
- absent (`null`) with `cost_usd: 0.0` — nothing was priced: the run recorded
  no tokens, or its model has no entry in litellm's table. A local or
  self-hosted model (`llama3.1:8b`, anything behind an OpenAI-compatible
  endpoint) is the normal case here, not a failure — it stays at 0 silently,
  with no warning.

The figure is provenance of the capture and lives on the case file only; the
run-facing example in `golden.jsonl` never carries it, because a replay against
a candidate model does not reproduce it.

### Wrapper arguments are unwrapped (legacy captures)

This applies only to captures promoted from **executed** calls — that is, ones
recorded before `requested_tool_calls` existed. A model's own requested
arguments are never rewritten: nothing stands between the model and them.

A capture SDK that decorates a Python function records that *function's*
parameters. An agent whose tools are `def archive_project(tool_args: dict)`
records every call as `{"tool_args": {"project_name": "..."}}`, while the model
only ever saw the flat properties declared in the toolset it was actually
offered. No model can produce the recorded shape, so ground truth in it
scores 0 against every candidate.

Promotion undoes this — but only when the declared schema *confirms* it: the
wrapper key must not be a declared property and the inner keys must all be
declared ones. The schema comes from the capture's own recorded
`toolset_ref` sidecar (the toolset that call was actually offered), not any
project config — so this works with no `evalshift.yaml` in sight. Without a
resolvable sidecar there is nothing to check against, so the recording is
left exactly as captured; a wrong guess would silently rewrite your ground
truth.

## What does not belong in a golden suite

Ground truth is what the agent *did right*, and a capture is not automatically
that. Three shapes are actively filtered or flagged during promotion:

- **Errored turns are refused.** A capture whose trace carries an `error` event
  died before the agent finished — an upstream 400, a timeout, a crash. Promoted
  naively it records *no tool calls*, which would become the assertion
  "calling nothing is correct here". `capture promote` exits non-zero and
  `capture sync` skips it, both naming the first error. `--allow-errored`
  promotes it anyway; even then the case never gets `expected_no_tools: true`,
  because a turn that never ran is not evidence that inaction was right.
- **Captures missing a recorded toolset are refused, unconditionally.** Every
  model call is required to record the toolset it was offered; a capture whose
  first `model_call` has no `toolset_ref` — the SDK failed to write the sidecar,
  or predates per-call toolset capture — has nothing to carry. `--allow-errored`
  does not help here (it is a different failure). The error names the capture
  id; the fix is re-capturing with a current `evalshift-sdk`.
- **Duplicated turns are warned about.** A retried turn produces two captures
  with the same `(conversation_id, turn_index)` — usually the failed attempt
  and the retry. Both would be promoted and their reconstruction order is
  arbitrary. `capture sync` warns once per collision so you can delete the
  case file you don't want.
- **Failed tool results are warned about.** A turn whose recorded tool result
  carries an `error`, or the common `{"success": false}` convention, is still
  promoted — "the model correctly tried, the backend was down" is legitimate
  ground truth — but you are told, because a candidate model is now being
  scored on reproducing a call that failed in production.

## Multi-turn agents

A follow-up question ("what time works?" → "1pm") is now a first-class
**conversation turn**, not just a bare `expected_no_tools` example scored in
isolation. If your agent captures record `conversation_id` / `turn_index` /
a messages-list `model_call.input`, `evalshift capture sync` links the turns
together and `run` replays each one with its recorded history prefix. See
[Multi-turn conversations](conversations.md) for the full walkthrough
(SDK-side capture, promotion, and teacher-forced replay semantics).

## Picking an evaluator

| Evaluator              | Use when                                                |
| ---------------------- | ------------------------------------------------------- |
| `tool_selection`       | You care about *which* tools fire (most common).        |
| `tool_arguments`       | You care about *what* the model passes to each tool.    |
| `tool_trace_structure` | You care about call counts, parallelism, or refusals.   |
| `trace_invariants`     | A rule must hold whatever the source did (auth before a charge, at most one charge, argument bounds). |

You can run multiple at once. Each becomes an independent comparison
in `analysis.json`, with the existing Benjamini-Hochberg correction
already adjusting for the multi-test count.

### Parallel fan-outs and call order

`conformance: expected` matches the expected calls **in order**, which is right
for a sequential plan (fetch, then act) and wrong for a parallel fan-out
(archive six projects). Under an in-order walk, a model that emits the same
calls in a different sequence — or that made one expected call first — can
score 0. When your expected calls are a fan-out, use `conformance: expected_set`:

```yaml
tool_selection:
  - name: routing_selection
    conformance: expected_set
```

It scores the same recall, counting duplicates (two expected `archive_project`
calls need two actual ones) and ignoring extra calls, without penalising a
permutation. Call counts are `tool_trace_structure`'s job, not this one's.

When an order genuinely matters — authenticate before charging — don't rely on
the recording's order to say so. Write it as an `order` rule under
[`trace_invariants`](configuration.md#evaluatorstrace_invariants), which checks
both models against the rule rather than against each other, and keep
`expected_set` for the fan-out.

`trace_invariants` is also the rule-based counterpart to `agent_trace`'s
`dangerous_tools`: that one flags a target that made *more* dangerous calls
than the source, so a dangerous call both made passes; a `forbidden` or
`call_count` rule fails the target whatever the source did. It reads replayed
traces by default and imported ones with `traces: imported`.

### Scoring free-text arguments

Free-text tool arguments — search queries, titles, descriptions — will never
match exactly between two models. The default `default_strategy: auto` handles
that without configuration:

1. Strings equal after normalizing case and whitespace score `1.0`.
   `"Find people"` vs `"find  people"` is not a wrong value.
2. Fields the example's toolset declares as identifiers, enums, booleans or
   `date` / `date-time` / `uuid` / `email` formats are scored `exact` — a
   reworded timestamp *is* wrong. Numbers go to `numeric`, objects and arrays
   to `subset`.
3. What is left is genuine free text, and it is graded: embedding similarity
   when an `evaluators.semantic` block lent a model, `difflib` ratio when not,
   so partial credit survives even with no embedding model configured.

Override a field explicitly when you know better:

```yaml
tool_arguments:
  - name: routing_args
    strategies:
      query: semantic
      order_ref: exact
```

`strategies` keys are **field names matched across every tool**, so pick names
that mean the same thing everywhere in your toolset. A `strategies` entry always
wins over `default_strategy`. `semantic` borrows the embedding model (and cache)
from your `evaluators.semantic` block — with no semantic evaluator configured
there is no model to borrow and the strategy degrades to `exact`. Set
`default_strategy: exact` on the evaluator to score every unlisted field by byte
equality, the pre-0.12 behaviour.

### Scoring arguments against ground truth

`tool_arguments` compares the target's arguments to the **source's** by default.
That answers "did the arguments change?", not "are the arguments right" — the
source's own score is 1.0 by construction, even when the source passed a value
that does not exist. For a capture-first suite, set `against: expected` so both
models are scored against the arguments your production agent actually recorded:

```yaml
tool_arguments:
  - name: routing_args
    against: expected
```

Only expectations that carry `arguments` are scored; name-only expectations are
`tool_selection`'s business. An expected call the model never made scores 0 — a
missing call cannot have correct arguments — and extra calls beyond the
expectation are ignored here. Which keys are compared follows each expectation's
`match_strategy`: `exact` also flags arguments the ground truth did not record,
while `subset` (what `capture promote` writes) scores the recorded keys only.

If every ground-truth comparison scores 0, check whether your recorded
arguments use keys the declared schema does not have — a capture of a decorated
function's parameters rather than the model's own arguments scores 0 for a
reason that is not the model's fault. See
[Wrapper arguments are unwrapped](#wrapper-arguments-are-unwrapped-legacy-captures).

A ground-truth field that **neither** model produced is dropped from that call's
denominator on both sides and disclosed as `unmeasured_fields` in the record's
per-call metadata. It is a stale expectation — an argument your agent used to
pass and no longer does — not a model defect, and scoring it would cap the call
below 1.0 for good. For the same reason `ARGUMENT_VALUE_DRIFT` is stamped only
when the target scores *below* the source: both models missing the same
expectation by the same margin is a fact about the suite.

#### Ground-truth provenance

`capture promote` / `capture sync` transcribe `expected_tools[].arguments`
verbatim from the source model's own recorded call and mark each expectation
`provenance: captured`. On such a row the source scores 1.0 **by construction**,
so `against: expected` quietly measures the same thing `against: source` does —
target deviation from source — while `source_score: 1.0` reads like evidence the
source was right. Nothing is wrong with the number, so the run says so instead of
changing it: when every scored row is `captured`, `migration_decision.json`
carries a recommendation naming the count and the caveat.

Set `provenance: reviewed` on a golden row once a human has checked its
arguments:

```jsonl
{"id": "ex_refund_01", "inputs": {"query": "..."}, "toolset_ref": "sha256:1a2b...", "expected_tools": [{"tool_name": "issue_refund", "arguments": {"order_id": "A-1", "amount_usd": 40.0}, "provenance": "reviewed"}]}
```

Scoring is identical either way. The disclosure goes silent as soon as one row
is `reviewed`: a suite someone has started checking is no longer uniformly
source-derived, and a blanket disclaimer over it would understate the rows they
did check.

### Optional tool parameters

A parameter that is `"required": []` in your tool schema can legitimately be
passed by one model and omitted by the other: `get_projects(status="active")`
and `get_projects()` are both valid calls. That scores 0.5, not 0.0 — a real
difference worth surfacing, not a total failure. Set
`optional_fields_scored: strict` to score presence exactly.

## Reading a tool run's report

`report.html` renders the two `tool_selection` axes as separate rows,
each labelled with its slug (`routing · tool_selection.divergence`) and
with what it compares. They are not interchangeable: divergence measures
your migration, conformance measures your suite.

Two places name the tools themselves, both read off the evaluator's own
record rather than re-derived from the call trace:

* the **per-example breakdown** gains a `Tools called (source → target)`
  column — one line per example, so a ten-example suite shows all ten
  rather than only the five worst;
* each **top regression** card states both sides' tools in its "why
  flagged" line.

The **Tool match** column is signed: ✗ means some tool evaluator scored
the target *below* the source, not that the target missed absolute
perfection. A pair on which both models miss your recorded ground truth
identically is not a ✗ — it is the same pair before and after, and what
is wrong is the ground truth. Look for **Ground truth missed by both**
in the evaluator table when that happens.

A **Trace rules broken** panel appears when the target broke a hand-written
[`trace_invariants`](configuration.md#evaluatorstrace_invariants) rule: every
broken rule with its owner and whether the source broke it too, whatever the
delta, advisory (`blocking: false`) rows tagged **advisory**. `report.json`
carries the same rows under `invariant_violations`, and the evaluator table
tags an advisory entry's row the same way.

## Troubleshooting

* **A suite reports tool evaluator scores you didn't expect on some rows** —
  run `evalshift doctor`. It reports the toolset each configured suite
  carries and flags a suite whose examples carry more than one distinct
  toolset — legal (each example dispatches its own), but also the shape a
  wiring mistake takes, so it's worth confirming the split is intentional.
* **Bimodal score distribution** — tool evaluators often produce
  scores at exactly 0 or 1. The analysis layer's Shapiro-Wilk
  fallback routes these through Wilcoxon signed-rank automatically.
* **"no matched calls between source and target"** — the
  `tool_arguments` evaluator scores a regression when the target
  doesn't reuse any of the same tool names as the source. Check
  `tool_selection` first to triage.
