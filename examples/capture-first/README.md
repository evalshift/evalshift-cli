# Capture-first example

The flow the README and `evalshift init` recommend, checked in end to end:
instrument an agent with the [evalshift-sdk](https://github.com/evalshift/evalshift-sdk),
exercise it, then let `evalshift capture sync` turn what it recorded into a
golden suite and wire that suite into `evalshift.yaml`.

This is the only example that uses the managed **`suites:`** block. The other
three (`simple/`, `agent/`, `agent-traces/`) hand-author a `golden.jsonl` next
to their config and let the CLI pick it up by filename; here the suite is
derived from recorded agent behaviour, and nothing under `suites:` — the path
or the evaluators — was typed by hand.

## What's here

| Path | Where it came from |
| --- | --- |
| `agent.py` | Hand-written. A deterministic on-call triage agent instrumented with `@capture.agent`, `@capture.tool` and `record_model_call`. No LLM is called. |
| `evalshift.yaml` | `evalshift init --provider gemini --no-wire-agents`, its `captures:` block deleted (see step 2), then `evalshift capture sync`. Otherwise unedited — the only other difference from a fresh `init` is the filled-in managed region at the bottom. |
| `.evalshift/captures/oncall_triage/cap_*.json` | Written by the SDK when `agent.py` ran with `EVALSHIFT_CAPTURE=1`. Three alerts, three captures. |
| `.evalshift/toolsets/56503f6d….json` | Written by the SDK. One content-addressed sidecar holding the four tool schemas the router model was offered; every capture and every promoted case points at it by hash. |
| `.evalshift/suites/oncall_triage/cap_*.json` | Written by `evalshift capture sync`. One auditable promoted case per capture, carrying its provenance: the capture it came from, when it was recorded (`captured_at`) and on which model (`source_model`). |
| `.evalshift/suites/oncall_triage/golden.jsonl` | Written by `evalshift capture sync`, regenerated from the case files above. This is what `run` reads. |
| `.gitignore` | Hand-written, and explained below. |

## Reproduce it from scratch

Nothing here is precious — delete `.evalshift/` and `evalshift.yaml` and rebuild
both. Only step 1 needs the SDK; steps 2–3 are the CLI.

### 1. Record captures

The CLI depends on the SDK (`evalshift-sdk`, import name `evalshift`), so the
environment that has the `evalshift` binary runs the agent too:

```bash
cd examples/capture-first
EVALSHIFT_CAPTURE=1 python agent.py
```

Capture is off unless `EVALSHIFT_CAPTURE` is set, which is what makes it safe to
leave the instrumentation in production code. One capture file lands per agent
invocation under `.evalshift/captures/oncall_triage/`.

Three details in `agent.py` are what make a capture *promotable*:

* `record_model_call(input=<the fully rendered prompt string>)` — promotion
  recovers the case's `inputs` from the first model call's input. A bare string
  becomes `{"input": "<string>"}`, which is exactly what the passthrough
  `replay` prompt `init` scaffolds renders back.
* `tools=` on every model call — the toolset the model was offered. It has no
  default at any SDK entry point, and `capture sync` refuses to promote a
  capture whose model call recorded no toolset (`no usable recorded toolset`).
* `@capture.tool` — each recorded call becomes an `expected_tools` entry with
  `provenance: captured`, the ground truth a candidate model is scored against.
  These captures predate the SDK's `requested_tool_calls`, so every promoted
  case says `promotion_source: executed`; an agent that also passes
  `record_model_call(requested_tool_calls=...)` (SDK 0.4.0+) is scored against
  what the model *asked for* instead.

### 2. Scaffold the config

```bash
evalshift init --provider gemini --no-wire-agents
```

This writes `evalshift.yaml` and nothing else — no example data, no suite.
(`--no-wire-agents` skips the `EVALSHIFT.md` coding-agent guide `init` writes by
default; a real project wants it. `--ci` additionally scaffolds the GitHub
Actions workflow referenced at the bottom of this file.) Note what `init` *does*
write: a single passthrough prompt

```yaml
prompts:
  - id: replay
    detection: manual
    content: "{input}"
    variables: [input]
```

and an empty, marker-delimited `suites: {}` region at the bottom of the file.
The prompt is the whole reason `prompts:` is still required in a capture-first
project: a promoted case carries the *rendered* prompt, so replaying it means
echoing that string back verbatim.

`init` also writes a `captures:` block, just above that region:

```yaml
captures:
  max_age: 14d
  max_cases: 200
  # store: s3://<bucket>/<prefix>
```

**Delete it here**, with the comment above it. The block keeps each captured
suite a capped sample of recent captures recorded on `defaults.source_model`,
but the captures checked in here are demo recordings: they are older than
`max_age`, and were recorded on a placeholder model id (`demo/triage-router`),
not on `defaults.source_model`. With the block in place, `sync` promotes none of
them:

```
⚠ oncall_triage: none of 3 case(s) and capture(s) were recorded on gemini-3.1-flash-lite-preview (seen: demo/triage-router ×3); not filtering by model — check defaults.source_model
✓ promoted 0 capture(s) into 0 suite(s).
  oncall_triage: 0 cases — kept 0, retired 0, added 0 of 0 eligible; not eligible: 3 older than 14d
```

A real project, recording fresh captures on the model it runs, keeps the block —
see [Bounded suites](../../docs/configuration.md#bounded-suites).

### 3. Promote the captures

```bash
evalshift capture sync
```

```
✓ promoted 3 capture(s) into 1 suite(s), wired generation config for 3 case(s).
✓ wired 1 suite(s) into evalshift.yaml
run: evalshift compare --suite-name oncall_triage --to <candidate>
```

With no `captures:` limits, `sync` promotes every capture, rebuilds
`golden.jsonl`, and rewrites the managed region with an entry per suite:

```yaml
# >>> evalshift suites (managed by `evalshift capture sync`) >>>
suites:
  oncall_triage:
    source: captured
    path: .evalshift/suites/oncall_triage/golden.jsonl
    evaluators:
      tool_selection:
        - name: routing
          conformance: expected
          divergence: set
      tool_arguments:
        - name: routing_args
          against: expected
# <<< evalshift suites <<<
```

The per-suite `evaluators:` block is **derived, not scaffolded**: `sync` reads
what the suite's rows actually contain. These captures offered a non-empty
toolset and recorded tool arguments, so the suite gets `tool_selection` and
`tool_arguments`; a suite promoted from a tool-free agent would get neither
block, and would inherit the top-level `evaluators:` untouched instead of being
scored against an empty tool denominator. Hand edits inside the markers are
overwritten on the next sync — pin an entry with `managed: false` to freeze it
(sync then prints what it would have written).

Re-running `sync` is idempotent: already-promoted captures are skipped unless
you pass `--force`.

## What the three cases cover

All three share one toolset and one prompt; they differ in the ground truth
promotion recovered.

| Case | Recorded behaviour | Ground truth in `golden.jsonl` |
| --- | --- | --- |
| `checkout-api` 5xx spike | health check → open incident → page on-call | `expected_tools` with all three calls and their arguments |
| slow `batch-reporting` job | health check → log search, no page | `expected_tools` with two calls |
| ownership question | answered in text, called nothing | `expected_no_tools: true` |

The third row is the one worth stopping on. "Tools were offered and correctly
*none* were called" is its own ground truth: a candidate model that starts
paging a human over a single failed webhook retry has regressed exactly as
surely as one that stops paging on the 5xx spike, and only a suite that records
the negative case can see it.

Every case also carries `generation_config: {temperature: 0.0}` — recorded at
capture time and replayed — and `expected.final_output`, the reply production
actually gave. `expected` is provenance, not a target: no text evaluator scores
against it (they compare the two candidate models to each other), and it is
carried so a hosted push can show what production said next to what the two
models said.

## Check it

```bash
cd examples/capture-first
evalshift doctor
evalshift validate --suite .evalshift/suites/oncall_triage/golden.jsonl
```

`doctor` reports the toolset each configured suite carries and flags a suite
whose rows disagree about it. Missing provider keys are soft warnings — it
exits 0 without them. `validate` still takes a suite *path* rather than a
`--suite-name`, so a capture-first project has to point it at the promoted file.

## Run it

`run` calls real models, so it needs a provider key. Nothing above this line
does — the captures, the suite and the config were all produced offline.

```bash
cd examples/capture-first
export GEMINI_API_KEY=<gemini-api-key>

evalshift compare --suite-name oncall_triage --to gemini-3.1-pro-preview
```

`--suite-name` looks the suite up in the `suites:` block, which is where its
path *and* its evaluator overrides come from. Step by step instead of `compare`:

```bash
evalshift run --yes --suite-name oncall_triage --to gemini-3.1-pro-preview
RUN_ID=$(ls -t .evalshift/runs/ | head -1)
evalshift evaluate "$RUN_ID"
evalshift analyze "$RUN_ID"
evalshift report "$RUN_ID" --open
```

Three examples is far below the point where the paired statistics say anything:
below n=5 no test is run at all and the comparison is annotated `n=3 < 5; no
test run`, and results stay flagged uncertain below n=20. A suite this small
demonstrates the wiring, not a verdict. A real capture-first suite grows by
leaving the SDK instrumentation in place and re-running `capture sync`; sync
de-duplicates on replayed content, seeded from the cases already on disk, so
repeated syncs add cases without inflating *n*. With the `captures:` limits
`init` scaffolds, each sync also keeps the suite a capped, fresh sample instead
of letting it grow without end.

## Why `.evalshift/` is committed here

The repo root gitignores `.evalshift/` wholesale — it is generated runtime data.
This example has to commit part of it, because the captures, the sidecar and the
promoted suite *are* the example. `.gitignore` in this directory un-ignores the
directory, re-ignores everything inside it, then re-admits exactly `captures/`,
`suites/` and `toolsets/`. Runs, checkpoints and the response cache stay ignored.

A real project commits the same way **minus `captures/`** — raw captures hold
production payloads, and only the reviewed, promoted suite belongs in git:

```gitignore
.evalshift/*
!.evalshift/suites/
!.evalshift/toolsets/
```

That is the version `evalshift init --ci` documents at the top of the workflow
it scaffolds, and committing those two directories is what lets the
GitHub Action discover a suite per `.evalshift/suites/*/golden.jsonl` and gate
merges on it.
