# Agent Traces

EvalShift can compare externally recorded agent timelines. The traces come from
your own runtime — EvalShift never runs your agent — and are imported into a
completed local run, after which `evaluate`, `analyze`, and `report` work as
usual. Imported traces stay local: `evaluate` scores them and the local report
shows them, but `push` uploads only the replay's own tool-call trace (see
[Hosted](#hosted)).

## Import

Run the normal model calls first so EvalShift has a run id and source/target
call pairs:

```bash
evalshift run --yes
evalshift traces import <run-id> \
  --source source-traces.jsonl \
  --target target-traces.jsonl
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

`--source` and `--target` are both required. Add `--strict` to fail the import
when any completed run pair lacks a trace pair; without it those pairs are only
counted in the printed `missing pairs` line.

```bash
evalshift traces import <run-id> \
  --source source-traces.jsonl \
  --target target-traces.jsonl \
  --strict
```

The import command writes:

```text
.evalshift/runs/<run-id>/traces.jsonl
```

## JSONL Shape

Each line is one trace for one `(prompt_id, example_id, role)`:

```json
{"run_id":"r_20260609_trace1","prompt_id":"support_agent","example_id":"refund_017","role":"source","events":[{"type":"tool_call","sequence_index":0,"timestamp":"2026-06-09T12:00:00Z","metadata":{},"name":"check_refund_policy","arguments":{}},{"type":"tool_call","sequence_index":1,"timestamp":"2026-06-09T12:00:01Z","metadata":{},"name":"issue_refund","arguments":{"ticket_id":"T-1032"}}]}
```

The line itself carries `run_id`, `prompt_id`, `example_id` (all non-empty),
`role` (`source` or `target`), and `events` (defaults to `[]`).

Every event has `type`, `sequence_index` (integer, ≥ 0), `timestamp`, and
`metadata` (defaults to `{}`). Per type, on top of those:

| `type` | Required | Optional (default) |
| --- | --- | --- |
| `model_call` | `model_id` (non-empty) | `input`, `output` (`null`), `input_tokens`, `output_tokens`, `latency_ms` (`0`), `cost_usd` (`0.0`), `toolset_ref`, `tools_offered`, `requested_tool_calls` (`null`) |
| `tool_call` | `name` (non-empty) | `arguments` (`{}`), `call_id`, `parent_call_id` (`null`) |
| `tool_result` | `name` (non-empty) | `call_id`, `result`, `error` (`null`) |
| `retrieval` | `source` (non-empty) | `query` (`""`), `documents` (`[]`) |
| `guardrail` | `name` (non-empty), `verdict` (`pass` / `fail` / `warn` / `skipped`) | `reason` (`null`) |
| `final_output` | — | `text` (`""`) |
| `error` | `message` (non-empty) | `category` (`null`) |

`timestamp` must carry a UTC offset — `2026-06-09T12:00:00Z` or
`2026-06-09T14:00:00+02:00`. An offset timestamp is converted to UTC on import;
a naive one (no offset at all) is rejected with the file and line that carried
it. Assuming UTC for a naive value would silently relabel a trace recorded in
another zone, and the run bundle would then carry that as fact — the hosted
bundle contract requires UTC, so the ambiguity has to be resolved by the person
who has the trace, not by the CLI.

Numeric fields cannot be negative. Trace models are strict, like the rest of the
config contract: an unknown key anywhere in the line is an error, not a
warning.

A `model_call` distinguishes three things that are easy to conflate:

- **offered** — `toolset_ref` (content-addressed pointer to the toolset sidecar)
  and `tools_offered` (the display-only tool-name list): what was passed *to*
  the model;
- **requested** — `requested_tool_calls`, a list of
  `{"name": ..., "arguments": {...}, "call_id": ...}` (`arguments` defaults to
  `{}`, `call_id` to `null`): what the model asked to call *in its response*;
- **executed** — the `tool_call` / `tool_result` events: what the application
  actually ran.

Requested and executed legitimately differ — an app may filter, re-order, or
wrap the calls it runs. `requested_tool_calls` is `null` on a trace recorded
before the field existed, which is not the same as `[]` ("the model requested
no tools").

EvalShift sorts events by `sequence_index` and rejects duplicate indices. A
`tool_result` with a `call_id` must match a `tool_call` that carried the same
`call_id` earlier in the trace.

## Evaluator

```yaml
evaluators:
  agent_trace:
    - name: trace_safety
      check_tool_order: true
      check_arguments: true
      check_missing_verification: true
      verification_tools: ["check_refund_policy"]
      dangerous_tools: ["issue_refund"]
```

The evaluator emits normal `scores.jsonl` records with failure categories such
as `TOOL_ORDER_DRIFT`, `ARGUMENT_VALUE_DRIFT`, `DANGEROUS_ACTION_DRIFT`, and
`MISSING_VERIFICATION_STEP`.

`agent_trace` compares the target with the source: `dangerous_tools` flags a
target that made more dangerous calls than the source did, so a dangerous call
both sides made goes unflagged. To hold both sides to rules instead, add a
[`trace_invariants`](configuration.md#evaluatorstrace_invariants) entry with
`traces: imported`:

```yaml
evaluators:
  trace_invariants:
    - name: refund_contract
      traces: imported
      rules:
        - id: policy-before-refund
          type: order
          before: check_refund_policy
          after: issue_refund
        - id: one-refund
          type: call_count
          tool: issue_refund
          max_calls: 1
```

An imported trace is checked as one response with no prior context: every call
in it, in `sequence_index` order, was the agent's own. A rule the target breaks
counts toward `migration_policy.max_invariant_violations` whatever the source
did, when the entry is blocking (the default); `blocking: false` entries are
reported but never gate. The entry's `applies_to` scopes imported pairs by
`prompt_id`, and `evaluate` fails on a run with no `traces.jsonl`, naming
`trace_invariants (traces: imported)` in the error.

Debug commands become trace-aware:

```bash
evalshift diff case <run-id> <example-id>
evalshift inspect case <run-id> <example-id>
evalshift replay case <run-id> <example-id> --model target --trace
```

## Hosted

Imported traces are not uploaded. They stay in
`.evalshift/runs/<run-id>/traces.jsonl`, where `evaluate` (the `agent_trace`
evaluator and `trace_invariants` entries with `traces: imported`), the local
report and `diff case` / `inspect case` / `replay case` read them.

`evalshift bundle` / `evalshift push` carry only the replay's own tool-call
trace — one stream per model side with the tool calls (names, arguments, call
ids), round markers, any final text and refusal messages, capped at 256 KB per
side. The full bundle contract is in
[Hosted EvalShift](hosted.md#bundle-and-push).
