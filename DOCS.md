# EvalShift CLI Documentation

EvalShift is a **local-first CLI for safe LLM model migrations**. You point it at a golden suite of examples, it runs the same prompts on two models — your current production model and the candidate you want to migrate to — scores both sides with structural, semantic, LLM-as-judge, and tool-call evaluators, then runs paired statistics over the deltas and tells you, with confidence intervals and multiple-comparison correction, what actually regressed.

```
evalshift-sdk    captures real agent behavior in production
      ↓
evalshift CLI    replays it against a candidate model — scores, stats, report
      ↓
hosted (opt-in)  run history, diffs, PR gates
```

The suite is the crux, so the capture SDK is the recommended way to build one: it records real production runs to disk and `evalshift capture sync` promotes them into golden suites. Hand-written suites are fully supported — see [The golden suite](#the-golden-suite).

- **Package name:** `evalshift` · **CLI entry point:** `evalshift` · **version:** 1.2.2
- **Python:** >= 3.11 · **License:** Apache-2.0 · **Status:** stable
- **Local-first.** Runs, scores, stats, and reports all happen on your machine under `.evalshift/`. The only network calls are the model API calls you asked for — and, if you opt in, pushes to the hosted service and reads from a capture bucket you own (`captures.store`).
- **Four pieces:** CLI (this doc), SDK, GitHub Action, hosted server — each with its own machine-readable reference for AI tools. See [Ecosystem and AI-tool references](#ecosystem-and-ai-tool-references).

---

## Table of contents

1. [Installation](#installation)
2. [Ecosystem and AI-tool references](#ecosystem-and-ai-tool-references)
3. [Quickstart](#quickstart)
4. [Project setup](#project-setup)
5. [Capturing from production](#capturing-from-production)
6. [How it works](#how-it-works)
7. [Configuration](#configuration)
8. [The golden suite](#the-golden-suite)
9. [Prompts](#prompts)
10. [Evaluators](#evaluators)
11. [Agent evaluation](#agent-evaluation)
12. [Multi-turn conversations](#multi-turn-conversations)
13. [External agent traces](#external-agent-traces)
14. [Statistical methodology](#statistical-methodology)
15. [Migration policy and CI gating](#migration-policy-and-ci-gating)
16. [Run insights](#run-insights)
17. [Hosted EvalShift](#hosted-evalshift)
18. [GitHub Action](#github-action)
19. [Command reference](#command-reference)
20. [Compatibility and stability](#compatibility-and-stability)
21. [Environment variables](#environment-variables)
22. [Troubleshooting / FAQ](#troubleshooting--faq)
23. [Further reading](#further-reading)

---

## Installation

```bash
pip install evalshift
# or
uv pip install evalshift
```

From source:

```bash
git clone https://github.com/evalshift/evalshift-cli
cd evalshift-cli
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Requires Python 3.11+. Verify with `evalshift --version`.

> **Co-install note:** the CLI (package `evalshift`, import package `evalshift_cli`) depends on the capture SDK (package `evalshift-sdk`, import name `evalshift`), so both live in one environment and `pip install evalshift` brings the SDK with it. A production agent that only records captures installs `evalshift-sdk` alone. `python -m evalshift_cli` is the module form of the `evalshift` binary.

API keys go in the environment, never in config:

```bash
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Only the providers your configured models use need a key. `evalshift doctor` shows which keys are visible.

---

## Ecosystem and AI-tool references

EvalShift is four pieces. Each is released and documented independently; each owns exactly one dense, single-file reference written for LLMs.

| Piece | Distribution | What it does | Reference for humans | Reference for AI tools |
| --- | --- | --- | --- | --- |
| **CLI** | PyPI `evalshift` (import `evalshift_cli`) | Runs the suite on two models, scores, analyses, reports, bundles, pushes. | this document | <https://www.evalshift.dev/cli-llms-full.txt> |
| **SDK** | PyPI `evalshift-sdk` (import `evalshift`) | In-process capture: records your agent's model/tool calls to `.evalshift/captures/`. | [docs/sdk.md](docs/sdk.md), [SDK repo](https://github.com/evalshift/evalshift-sdk) | <https://www.evalshift.dev/sdk-llms-full.txt> |
| **GitHub Action** | `evalshift/evalshift-action@v0` | Runs the pipeline on PRs, pushes the run, maintains one PR comment, sets the `evalshift/regression` status. | [docs/github-action.md](docs/github-action.md), [action repo](https://github.com/evalshift/evalshift-action) | <https://www.evalshift.dev/ci-llms-full.txt> |
| **Hosted server** | service — API `https://api.evalshift.dev`, web app `https://evalshift.dev` | Stores pushed run bundles, diffs runs across branches, serves the web app, drives PR comments and gating. | [docs/hosted.md](docs/hosted.md) | covered by the CLI reference (`push`/`bundle` contract) |

Data flow is one-directional: **SDK captures → CLI runs and bundles → server stores and diffs → web app displays.** The SDK and CLI never call each other — the interface is files under `.evalshift/captures/`. The CLI (import `evalshift_cli`) depends on the SDK (import `evalshift`), so one environment holds both.

The CLI reference is generated from [llms-full.txt](llms-full.txt) at this repo's root — edit that file when CLI behaviour changes. The SDK and Action references are owned by their own repos; the copies served from `evalshift.dev` are synced from there.

### Pointing your coding agent at the right one

`evalshift init` does this for you (`--wire-agents`, on by default): it writes `EVALSHIFT.md` and adds a managed pointer block to any existing `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`, or `.github/copilot-instructions.md` — creating `AGENTS.md` if none of them exist — listing all three URLs with an instruction to fetch the relevant one before writing EvalShift code or config. It is idempotent — re-running updates the block in place. Disable with `--no-wire-agents`.

Doing it by hand, the mapping is:

| The agent's task touches | Fetch |
| --- | --- |
| CLI commands, `evalshift.yaml`, evaluators, suites, reports, bundles, push | `cli-llms-full.txt` |
| Instrumenting an application to record captures | `sdk-llms-full.txt` |
| CI workflows, PR gating, the action's inputs/outputs | `ci-llms-full.txt` |

---

## Quickstart

Point EvalShift at a real project. `evalshift init` writes a capture-first config, the [evalshift-sdk](https://github.com/evalshift/evalshift-sdk) records what your agent actually does, and `capture sync` turns those recordings into a golden suite:

```bash
evalshift init
# instrument the agent with evalshift-sdk, then run it with EVALSHIFT_CAPTURE=1
evalshift capture sync
evalshift compare --suite-name <suite> --to <candidate-model>
```

`evalshift compare` drives the full pipeline — `doctor → run → evaluate → analyze → report` — over one suite under one live progress display, then writes `report.html` (add `--open` to open it in your browser): a single-file, offline-capable HTML report with per-prompt/per-slice comparisons, severity badges, effect sizes with 95% CIs, and a migration-policy verdict panel. `run`/`compare` estimate worst-case cost up front and prompt for confirmation above $10 (skip with `--yes`).

See [Project setup](#project-setup) and [Capturing from production](#capturing-from-production). No captures to work from? Write `golden.jsonl` by hand — see [The golden suite](#the-golden-suite).

---

## Project setup

`evalshift init` is the real-project entry point. Writes **only** a minimal, capture-first `evalshift.yaml`: a passthrough `replay` prompt (`content: "{input}"`), advisory LLM-judge and (for Gemini and OpenAI) semantic evaluators, an empty managed `suites:` block for `capture sync` to fill, and a migration policy. The intended flow: instrument your agent with the evalshift-sdk → record captures → `evalshift capture sync` → run against the promoted suite.

`init` options:

- `--provider gemini|openai|anthropic|deepseek` — which provider's model ids the scaffold uses (prompted on a TTY; defaults to `gemini` otherwise). Gemini and OpenAI scaffolds include an embedding-based semantic evaluator; the Anthropic and DeepSeek scaffolds comment it out (no embedding endpoint).
- `--profile` — pre-tuned migration-policy budgets:

| Profile | regression ≤ | critical ≤ | equivalence ≥ | arg drift ≤ | tool divergence ≤ | trace-rule violations ≤ | cost Δ ≤ | latency Δ ≤ |
|---|---|---|---|---|---|---|---|---|
| `model-upgrade` (default) | 30% | 1 | 75% | 20% | 20% | 0 | +30% | +30% |
| `cost-reduction` | 2% | 0 | 97% | 1% | 2% | 0 | +5% | +30% |
| `local-model` | 5% | 0 | 90% | 2% | 5% | 0 | +0% | +50% |
| `quantization` | 2% | 0 | 97% | 0.5% | 2% | 0 | +0% | +20% |
| `provider-switch` | 3% | 0 | 95% | 1% | 3% | 0 | +20% | +40% |

The `model-upgrade` row is the loose first-migration starting point — the `migration_policy` field defaults (see [Migration policy and CI gating](#migration-policy-and-ci-gating)); the other four are tighter presets to move to as the suite grows.

- `--ci` — also scaffold `.github/workflows/evalshift.yml` (see [GitHub Action](#github-action)).
- `--wire-agents` (default on) — write `EVALSHIFT.md`, a guide for AI coding agents, and point existing agent files (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`, `.github/copilot-instructions.md`) at it, creating `AGENTS.md` if none exist. Both the guide and the pointer blocks link the three hosted llms.txt references — [cli-llms-full.txt](https://evalshift.dev/cli-llms-full.txt), [sdk-llms-full.txt](https://evalshift.dev/sdk-llms-full.txt), and [ci-llms-full.txt](https://evalshift.dev/ci-llms-full.txt) (GitHub Action). Idempotent; disable with `--no-wire-agents`.
- `--force` / `--directory` — overwrite protection and target dir (refuses to clobber an existing `evalshift.yaml` without `--force`).

---

## Capturing from production

The evalshift-sdk (`pip install evalshift-sdk` — already a dependency of the CLI, or installed alone in a production agent) records real agent runs as JSON capture files under `.evalshift/captures/<suite>/cap_<hex>.json` when `EVALSHIFT_CAPTURE=1` is set. Instrument with `@capture.agent` / `@capture.tool` / `record_model_call`, the LangChain `EvalShiftCallbackHandler`, or (SDK 0.4.0+) a provider client wrapper — `wrap_openai` / `wrap_anthropic` / `wrap_genai` — which records the model id, offered tools, `requested_tool_calls`, usage, latency and the tool-use generation settings (`tool_choice`, `parallel_tool_calls`, `strict`) from every call, streams included, with no per-call code; a hand-written `record_model_call` should pass `requested_tool_calls=extract_requested_tool_calls(response)` so promotion can use the model's requested calls (step 3 below). The CLI turns those captures into golden suites:

```bash
evalshift capture list                 # table of recorded captures (--json for machine-readable)
evalshift capture fetch                # (captures.store only) mirror new captures into .evalshift/
evalshift capture sync                 # promote ALL captures → suites + wire config
evalshift capture promote cap_ab12 --as refund_case_1   # promote one
evalshift capture diff cap_ab12 cap_cd34               # compare two tool traces
evalshift capture clean                # delete promoted capture files + sweep orphaned toolsets
```

**Captures that live in a bucket.** Production hosts on Fargate, Lambda or Kubernetes lose their disk when they stop, so the SDK can ship captures to an object store you own instead (`EVALSHIFT_SINK=s3://<bucket>/<prefix>`, `gs://…` or `az://<account>/<container>/…` in the agent's environment, SDK 0.5.0+). Name the same store once in `evalshift.yaml`:

```yaml
captures:
  store: s3://acme-evals/support-agent
```

and `capture sync` and `capture list` fetch what is new before they read the local directory — captures are immutable and toolsets content-addressed, so only objects missing locally are downloaded, captures already promoted are skipped (so `capture clean` never causes a re-download), and each file is written atomically. Bucket keys are untrusted: one not shaped like the SDK's layout — a suite directory containing `:`, `/`, `\`, NUL or `..`, or starting or ending with `.` or a space, included — is counted as malformed and never written, and no key can place a file outside `.evalshift/`. One summary line reports it (`fetched 12 capture(s) and 1 toolset(s) from s3://…`; with `capture list --json` it goes to stderr, so stdout stays pure JSON). `--since 24h` limits a fetch to recently written captures, `--offline` skips it, and `capture fetch` does only that step. `promote`, `diff` and `clean` stay local. Install the matching extra — `evalshift[s3]`, `[gcs]` or `[azure]` (azure-storage-blob plus azure-identity) — and authenticate with the provider's normal chain (an OIDC role in CI, `aws sso login` / `gcloud auth application-default login` / `az login` locally); credentials never go in the URI, and a value containing `@` or `?` fails to load without being echoed back. A missing extra exits 1 naming the `pip install "evalshift[<extra>]"` to run; a list or download failure exits 1, so `sync` promotes nothing from a partial fetch. If `evalshift.yaml` cannot be read or does not validate, `sync` and `list` skip the fetch and work on the local directory as before, warning only when the file mentions `captures`. Full field reference: [`captures`](docs/configuration.md#captures).

**`capture sync`** is the workhorse. Per suite it:

1. Groups captures (by `conversation_id` for multi-turn, ordered by `turn_index`).
2. Builds one suite example per capture/turn: first model input → `inputs` (bare strings land under the `--input-var` name, default `input`), recorded tool calls → `expected_tools`, final output → `expected`, messages list → `history`, first `model_call`'s `toolset_ref` → the example's `toolset_ref`. Whether tools were offered (not just whether any were *called*) governs `expected_no_tools`: it is set only when the capture's toolset was non-empty and no tool was called — a text-only reply from a call that was never offered a tool proves nothing, so it stays unasserted. `expected` prefers a `final_output` event and otherwise falls back to the **last `model_call` with non-empty text output** — the reply the user saw, after the tool round-trips — because only the SDK's LangChain adapter emits `final_output`, so a manually instrumented project would otherwise promote `expected: null` on every case. A non-string `output` is left alone rather than stringified, and a capture with neither source of text is promoted without ground truth and warns. Tool calls are grouped into **agent rounds** (split at each recorded `model_call`): every round lands in `expected_tool_rounds`, and `expected_tools` is always **round 1**. Under the default `--rounds first` that is the only round replayed — `run` makes one call per example and does not feed tool results back — and a multi-round capture prints a warning naming the calls it will not replay. `--rounds all` also carries the recorded tool results on the case as `tool_result_fixtures` (each round's calls paired with that round's `tool_result` events by `call_id`, then by name within the round), and `run` then replays the example **teacher-forced**: round *k* sees the prompt plus the *recorded* rounds `1..k-1` as assistant tool calls and `tool` results (the candidate's own calls are never fed back), for every covered round plus the answer round after it; the tool evaluators score each round against its own ground truth and report the mean, so `max_tool_divergence` counts an example as diverged if any round diverged. Fixtures stop at the first round with an unpaired call (promotion warns: `round m+1 has N tool call(s) with no recorded result; replay will cover rounds 1..m`); if round 1 itself is uncovered, replay stays single-shot. `--rounds all` no longer flattens every round into `expected_tools`. See [Agent rounds](docs/agents.md#agent-rounds-and-what-a-replay-can-reproduce). Recorded arguments wrapped in a single undeclared key — the signature of a capture recording a decorated function's parameters rather than what the model passed, e.g. `{"tool_args": {"project_name": ...}}` — are **unwrapped**, but only when the capture's own recorded toolset schema confirms it (wrapper key not declared, inner keys all declared); with no resolvable sidecar to check against, the recording is left untouched. Unwrapping applies only to the executed-call path below.
3. Prefers the model's **requested** tool calls as ground truth when the capture recorded them (`model_call.requested_tool_calls`). Executed `tool_call` events have already passed through the app — its filtering, retries, re-ordering, and its own function signatures — so they show what the *app* did, while a golden case must state what a *model* should produce. Each `model_call` is then a round (rounds that requested nothing are dropped, as tool-less executed rounds are) and requested arguments are carried verbatim, never unwrapped. The case records which yardstick it used as `promotion_source: "requested" | "executed"`. **Every `model_call` in the run must carry the field** for the capture to be scored against requested calls: pass `[]` for a round in which the model requested no tools — `null` means *not recorded*, not *nothing requested*, and after the fact the two cannot be told apart. Fallback to the executed calls is silent when no `model_call` carries the field (a capture predating it) and warns when only *some* do, in which case the whole capture falls back rather than mixing yardsticks. That mixed case is a recording gap, not a version skew — the SDK records `null` for an omitted `record_model_call(requested_tool_calls=...)` argument, and the usual cause is an agent that passes the field on its tool-picking calls but omits it on the final, text-only one. When requested and executed calls disagree — different tool, argument, or round grouping — requested wins and promotion warns, naming the tools on both sides.
4. Carries the recorded generation settings onto the promoted example: the first `model_call`'s `metadata["generation_config"]` is copied verbatim to the example's `generation_config`, so the replay dispatches under the temperature, structured-output, and tool-choice settings the production call actually used. `capture sync` appends `wired generation config for N case(s)` to its summary line when any promoted capture carried one.
5. Records what the captured run cost on the case file as `cost_usd` (its `model_call` events summed) plus `cost_source`. The SDK never prices anything — a `model_call`'s `cost_usd` is `0.0` unless the app's own instrumentation set it, and the provider client wrappers record tokens but no cost by design — so each event with tokens and no cost is priced at promotion from litellm's price table for its own `model_id` (aliases and provider prefixes resolve through the registry first). `cost_source` is `"recorded"` when every non-zero part came from the capture (a recorded cost is never re-estimated), `"estimated"` when any part was priced here, and absent with `cost_usd: 0.0` when nothing was priced — no tokens, or a model litellm does not price. An unpriced local / self-hosted model is the normal case, not a failure: it stays at 0 silently, with no warning. The example in `golden.jsonl` never carries the figure; it is provenance of the capture, not something a replay reproduces.
6. Skips captures whose replayed content duplicates an already-promoted case or an earlier capture in the same run — duplicate examples inflate *n* and corrupt the paired statistics (`--keep-duplicates` opts out). Dedup is seeded from the cases already in the suite dir, so it holds across repeated syncs.
7. Skips captures whose turn recorded an `error` event — a turn that died before the agent acted is not ground truth, and promoting it would assert `expected_no_tools: true` on a question that needed a tool. `--allow-errored` promotes it anyway (still never asserting `expected_no_tools`). `capture promote` exits non-zero on the same condition. Separately and unconditionally — `--allow-errored` does not help — a capture whose first `model_call` has no `toolset_ref` is refused: the SDK did not record what tools were offered, so there is nothing to carry, and re-capturing with a current `evalshift-sdk` is the only fix.
8. Warns when two captures claim the same `(conversation_id, turn_index)` (a retried turn), and when a promoted turn contains a failed tool result (`error`, or `{"success": false}`). Both stay warnings — see [Agent evals → What does not belong in a golden suite](docs/agents.md).
9. Writes `.evalshift/suites/<suite>/golden.jsonl` and rewrites the managed `suites:` block in `evalshift.yaml` (between the `>>> evalshift suites` markers).
10. After the write (or after printing the block for you to paste), checks the CI pin: if a workflow under `.github/workflows/` uses `evalshift/evalshift-action` with an `evalshift-version` older than this CLI, with no pin at all, or with pins that are all newer than this CLI, it prints a warning naming the workflow and job plus the fix — the exact `evalshift-version: "<this version>"` line to set, or, for a newer pin, `pip install -U evalshift` locally. Advisory only — sync never edits a workflow and the exit code is unchanged. See [Pin drift](#pin-drift).

Strictness knobs for the derived tool expectations: `--strict-args` (exact argument matches), `--names-only` (ignore arguments), `--tool-count` (also pin the call count, scoped the same way as `expected_tools`), `--rounds {first,all}` (which agent rounds become ground truth, default `first`). `--tag` attaches extra slice tags; `--print` previews the `suites:` block without writing.

Then evaluate a candidate against real recorded behaviour:

```bash
evalshift compare --suite-name <suite> --to <candidate-model>
```

`capture promote` (single capture) recovers history only from that capture's own messages list — promoting a mid-conversation capture warns and points you at `sync`, which does cross-capture reconstruction.

`capture clean` deletes capture files (promoted-only by default; `--all` for every capture) and never touches promoted suites — but it does sweep `<base>/toolsets/`: any toolset sidecar left referenced by neither a surviving capture (in any suite) nor a promoted suite example is deleted and reported. A sidecar a promoted `golden.jsonl` still uses is refcounted across both `<base>/captures/` and `<base>/suites/`, so it is never touched, even with `--all`.

Captures are read from `.evalshift/captures/` under the current directory (or `EVALSHIFT_DIR`); `capture` subcommands accept a hidden `--base` override.

---

## How it works

### The pipeline

The CLI is a five-stage pipeline. Each stage writes one artefact under `.evalshift/runs/<run-id>/` and the next stage reads it; every stage is independently re-runnable.

```
init          →   doctor   →   run          →   evaluate       →   analyze           →   report
(scaffold)        (checks)     raw.jsonl        scores.jsonl       analysis.json         report.html
                               state.json                          migration_decision     report.json
                                                                   .json (if policy)
```

- **`doctor`** validates local config and shows which provider keys are visible. Exit 1 only when an existing `evalshift.yaml` fails validation — the row gives the problem count and points at `evalshift validate`, which prints each problem — or when its `captures.store` needs a client extra that is not installed; missing keys are soft warnings. Its second row, `evalshift-sdk`, reports the SDK version the `evalshift` import name resolves to in this environment (`warn` when the SDK is missing or shadowed by an older CLI's leftover files; never a failure). It also reports the toolset each configured suite carries (or the flat `golden.jsonl`) and flags a suite whose examples carry more than one distinct toolset — legal (each example dispatches its own), but also the shape a wiring mistake takes. When a workflow under `.github/workflows/` uses the GitHub Action it adds a `ci pin` row: `ok` (`pinned to <v>`) when CI installs this CLI version, `warn` when the pin is older, absent, or newer than the local CLI (see [Pin drift](#pin-drift)). When the config sets `captures.store`, a `captures.store` row reports it: `fail` (exit 1) naming the `pip install "evalshift[<extra>]"` when the client extra is missing, `warn` when the bucket cannot be listed (no cloud credentials on this machine, say), `ok` (`<uri> reachable`) otherwise; no row without a store. When the config wires an `llm_judge` evaluator and names both `defaults.source_model` and `target_model`, a `judge family` row warns for every `judge_model` that resolves to the same provider as an arm (self-preference bias; `ok` "from a third family" otherwise, no row when either arm is unset) — advisory, never a failure; `validate` prints the same line and the report repeats it above the verdict (see [`evaluators.llm_judge`](docs/configuration.md#evaluatorsllm_judge)).
- **`run`** parses prompts, validates every example against every prompt, estimates cost, then dispatches `(prompt × example × {source, target})` calls through an async orchestrator under a concurrency semaphore. Responses are cached — per call, and per replayed round for tool-calling examples (see [Response cache](#response-cache)); progress is checkpointed every 50 completions.
- **`evaluate`** scores each (source, target) pair with the configured evaluators, one `EvalRecord` per pair × evaluator. Scoring runs under the same `defaults.concurrency` semaphore as `run`, and the embedding/judge calls it makes go through the same response cache.
- **`analyze`** runs paired statistics per `(prompt, evaluator, slice)`, applies Benjamini–Hochberg FDR correction, classifies severities, and — when a `migration_policy` is configured — computes a pass/fail verdict.
- **`report`** renders the single-file HTML report (no external assets; works offline and attaches cleanly to a PR or email), and writes the machine-written [run insights](#run-insights) narrative unless `--no-insights` is passed. The page opens on a verdict / advisory-signal / economics panel row and a six-cell run strip (examples, calls, failed-or-truncated, spend, latency Δ, mean score Δ), then the executive summary (one row per prompt, drawn from blocking evaluators — advisory ones only on a prompt nothing else scored; the strip's mean score Δ averages those rows), the narrative, the **Trace rules broken** panel when the target broke a hand-written trace rule, one section per prompt, and the methodology. Every figure on it is derived from the run's own artefacts; the deltas in the header are the run-level rollup of the per-prompt economics. Top regressions are collapsed cards — expand one for the trace diff, the tool diffs and the conversation context. The report is dark-only.
- **`compare`** chains everything end to end over one suite and adds `--gate`, `--policy-gate`, `--push`, `--open`. Warnings raised while the pipeline runs (LiteLLM deprecation notices, insights-retry notes) are deferred and printed as one `⚠` section directly under the pipeline block — also when a stage fails, since a held-back warning may explain the failure. Errors are never deferred.

### Artefacts

Run ids look like `r_20260722_golden_a1b2c3` (`r_<date>_<suite-slug>_<hex>`). Under `.evalshift/runs/<run-id>/`:

| File | Written by | Contents |
|---|---|---|
| `state.json` | run | Run status, models, config hash, progress counters, `non_deterministic_models`, `dropped_params`, and — added by `evaluate` — `evaluator_coverage`: attempted vs recorded per axis, the pairs that produced no row, and the axis's `blocking` flag (atomic write) |
| `raw.jsonl` | run | One line per (prompt, example, role, sample): rendered prompt, output, tokens, cost, latency, tool trace, error. A teacher-forced multi-round replay is one row (tokens, cost and latency summed over its rounds) |
| `scores.jsonl` | evaluate | One line per (pair × evaluator): source/target scores, delta, explanation |
| `analysis.json` | analyze | Per-comparison statistics, severities, notes |
| `migration_decision.json` | analyze | Policy verdict + per-budget detail (only when `migration_policy` set) |
| `report.json` | report | The report payload, including `invariant_violations`: one row per [trace rule](#trace-invariants) the target broke |
| `report.html` | report | Single-file HTML report |
| `insights.json` | report | Cached [run insights](#run-insights) narrative (optional; skipped with `--no-insights` or no API key) |
| `traces.jsonl` | traces import | Bring-your-own agent traces (optional) |
| `run_bundle.json.gz` | bundle / push | Hosted upload bundle (optional) |

### Checkpointing and resume

`state.json` records a `config_hash` (SHA-256 over the canonicalised config plus the suite path). `run --resume` picks up the most recent in-progress run, verifies the hash still matches (aborts if the config or the suite **path** changed — suite *contents* are not hashed, so start a fresh run after editing examples), and skips every `(prompt, example, role, sample)` already present in `raw.jsonl`. Calls that errored are counted as done — they are not retried automatically.

### Response cache

Live run-stage responses are cached in SQLite at `~/.evalshift/cache.db`, keyed by SHA-256 over canonical JSON of `(model, prompt, inputs, temperature, max_tokens[, history])` — plus `generation_config`, the tools array as sent, the round index and the sample index when each is set — with a 7-day TTL. Re-running an identical evaluation costs no run-stage calls, agent suites included. An example with a non-empty toolset gets one entry per replayed round, keyed on the canonical model, the prompt and inputs, **the exact message list that round sends** (history, current turn, and the recorded rounds and fixture results teacher forcing feeds back), the tool list exactly as sent, in order (tool names, descriptions, schemas and `strict` — reordering the tools is a miss), `generation_config` (so `tool_choice` and `parallel_tool_calls`), the effective temperature and `max_tokens`, the round index and the sample index — editing a fixture re-sends only the rounds after it. A hit restores the parsed tool trace (calls with their ids and arguments, final text, refusal), tokens, cost, latency and finish reason, so the `raw.jsonl` row matches the live one apart from `cached` and `cached_rounds`; a multi-round example counts as cached only when every round was a hit. A row with any round served from the cache (`cached_rounds > 0`) carries latency recorded on an earlier run, so it is left out of the report's live latency figures and its latency delta is marked not comparable. Errors are never cached (the next run re-sends a failed round; the rounds before it come from the cache), truncated responses are cached and stay flagged, and a hit records the original call's cost and latency. A `cache.db` written by an earlier version is upgraded in place on open and keeps its entries; several `evalshift` processes may open it at once. Disable per-project with `defaults.cache: false`; wipe with `evalshift cache clear`.

The cache covers the evaluate stage too: `semantic` embeddings are keyed by `(embedding model, text)`, and `llm_judge` verdicts by `(judge model, criterion, source output, target output)`. The judge key uses a canonical A/B ordering, so the per-call orientation randomization doesn't halve the hit rate — the orientation that was actually used is recorded with the verdict and replayed on a hit, leaving `metadata.target_was_a` faithful.

### Run retention

Run history is pruned automatically after every completed `run`/`compare`, per suite: keep the newest `retention.max_runs_per_suite` (default 20), optionally evict runs older than `retention.run_ttl_days`. In-progress runs and the run just finished are never pruned. `EVALSHIFT_MAX_RUNS` overrides the count; `evalshift runs clean` prunes on demand with `--keep` / `--older-than` / `--suite` / `--dry-run`.

---

## Configuration

Everything lives in one file, `evalshift.yaml`, validated by strict Pydantic models with **`extra: "forbid"` everywhere** — a typo'd key fails loudly at load time instead of being silently ignored.

A representative config (what `evalshift init --provider gemini` writes, trimmed):

```yaml
version: 1

# project: your-org/your-project    # hosted only; scaffolded commented out

prompts:
  - id: replay
    detection: manual
    content: "{input}"
    variables: [input]

defaults:
  source_model: gemini-3.1-flash-lite-preview
  # target_model: gemini-3.1-pro-preview   # or pass --to per run
  concurrency: 4
  max_cost_usd: 50.0
  max_tokens: 4096

evaluators:
  semantic:
    embedding_model: gemini/gemini-embedding-001
    min_similarity: 0.9
    blocking: false            # advisory: reports drift, never gates
  llm_judge:
    - criterion_name: equivalence
      criterion_prompt: >
        Which output is more complete and correct? ... Answer "tie" when both
        are equivalent in substance and differ only in wording.
      judge_model: gemini-3.1-pro-preview
      blocking: false

# Tool evaluators are not written here: `capture sync` derives them per suite,
# inside the managed region at the end of the file, from what each suite's own
# captures contain. Hand-written `trace_invariants` are the exception: top
# level is fine, and sync keeps them on the managed entries it rewrites.

migration_policy:
  max_overall_regression_rate: 0.30
  max_critical_regressions: 1
  min_equivalence_rate: 0.75
  max_tool_argument_drift: 0.20
  max_tool_divergence: 0.20
  max_invariant_violations: 0
  max_cost_increase: 0.30
  max_latency_increase: 0.30

# The managed region goes last: it is the only part of the file a command
# rewrites, and the only part that grows without bound (one entry per suite,
# each with its own evaluator block).
# >>> evalshift suites (managed by `evalshift capture sync`) >>>
suites: {}
# <<< evalshift suites <<<
```

### Top-level fields

| Field | Type / default | Meaning |
|---|---|---|
| `version` | literal `1`, default `1` (optional) | Config schema version |
| `project` | `str \| None` | Hosted project slug, `org/project` (regex `^[a-z0-9-]+/[a-z0-9-]+$`) |
| `prompts` | list, required, ≥1 | Prompt definitions (unique ids enforced) |
| `defaults` | block | Run defaults, below |
| `evaluators` | block | Evaluator configs, see [Evaluators](#evaluators) |
| `migration_policy` | block \| absent | Regression budgets, see [Migration policy](#migration-policy-and-ci-gating) |
| `suites` | map | Named suites (`{name: {source: captured\|jsonl, path: ..., evaluators: ..., managed: true}}`); the block between the `>>> evalshift suites` markers is managed by `capture sync`. See [Per-suite evaluators](#per-suite-evaluators) |
| `retention` | block | `max_runs_per_suite` (default 20, `0` disables), `run_ttl_days` (default off) |
| `captures` | block | `store`: `s3://`, `gs://` or `az://` URI of the object store the SDK ships captures to; `sync` and `list` mirror it locally first. Omit for local disk only. See [Capturing from production](#capturing-from-production) |

`thresholds` was a top-level field until it was removed. It was free-form and gated nothing — `migration_policy` is the single source of truth for gating — so rather than being silently ignored it is now rejected by name, and a config that still carries it fails to load:

```text
`thresholds` was removed: it was free-form and gated nothing. Delete it from evalshift.yaml; migration_policy is the single source of truth for gating.
```

Nothing replaced it. Delete the block; express any gate you meant by it as a `migration_policy` budget.

A top-level `slices` list was removed the same way. It was validated and recorded in the run bundle, but analysis never read it — slices come from example tags (see [Slices](#slices)) — so a config that still carries it fails to load too:

```text
`slices` was removed: it never had any effect. Slices come from example `tags` automatically (one per distinct tag, plus `all`). Delete it from evalshift.yaml; per-slice budgets go under migration_policy.slices, keyed by tag.
```

Delete the block; a run reports the same slices without it. Hosted baselines are unaffected for any config that never set it (or set `slices: []`): the bundle's evaluator config still carries an empty `slices` list, so `eval_config_hash` does not move. Deleting a *non-empty* block does change that hash: runs pushed afterwards are not comparable to baselines pushed before, until the base branch pushes a run with the edited config.

### `defaults`

| Field | Default | Meaning |
|---|---|---|
| `source_model` | `None` | Baseline model id or alias (CLI `--from` overrides) |
| `target_model` | `None` | Candidate model id or alias (CLI `--to` overrides) |
| `judge_model` | `gemini-3.1-flash-lite-preview` | Declared default judge model. Note: each `llm_judge` entry carries its own `judge_model` field with the same built-in default — set it per criterion |
| `insights_model` | `None` | Model that writes the [run insights](#run-insights) narrative. Falls back to `judge_model` when unset |
| `concurrency` | `10` (1–64) | Max in-flight model calls, during both `run` and `evaluate` |
| `cache` | `true` | Use the SQLite response cache |
| `max_cost_usd` | `50.0` | Soft ceiling reserved for future enforcement — not yet enforced at run time. The pre-flight cost prompt currently triggers above $10 (skip with `--yes`) |
| `max_tokens` | `4096` | Completion cap per call (per-prompt `prompts[].max_tokens` overrides). Truncated calls are excluded from the regression statistics |
| `samples_per_example` | `1` (1–20) | Repeats each (prompt, example) this many times per model. Each sample pair is scored on its own; `scores.jsonl` keeps one row per example holding the mean over samples, with the per-sample scores and `delta_variance` under `metadata.samples`. Paired tests run over examples, so `n` is unchanged. Cost and calls multiply by it; the cache keys on the sample index so every sample is a live call |

### Slices

Slices come from the suite; there is nothing to configure. Every distinct example `tag` becomes a slice under its own name, alongside the implicit `all` slice, and every configured evaluator is analysed once overall and once per slice. Per-slice budgets go under [`migration_policy.slices`](#migration-policy-and-ci-gating), keyed by the tag:

```yaml
migration_policy:
  slices:
    security:                 # the example tag
      max_overall_regression_rate: 0.0
```

There is no top-level `slices:` key — it was removed (see [Top-level fields](#top-level-fields)). `overall` is reserved — it names the run-level scope in the run bundle — and is rejected as an example tag and as a `migration_policy.slices` key.

Slices holding exactly the same examples are collapsed to one before any test runs — duplicates restate the same numbers as if they were independent findings and skew the Benjamini–Hochberg correction anti-conservatively (extra copies of a p-value shrink every adjusted p-value in the family, so results look more significant than they are). `all` and any slice named under `migration_policy.slices` always survive; otherwise the provenance tag `captured` (written by `capture promote`) loses to an ordinary tag, then alphabetical order decides. Drops are reported on the terminal and as `collapsed_slices` in `analysis.json`. See [docs/methodology.md](docs/methodology.md).

### Per-suite evaluators

A project's suites are rarely homogeneous — one calls tools, six answer in prose — and one top-level `evaluators:` block either leaves the tool-calling suite unmeasured or hands the tool-free ones a tool evaluator with an empty denominator, which the policy reads as an *inconclusive* gate rather than as "not applicable here". So a `suites:` entry can carry its own evaluator block, and `capture sync` generates one per suite from what that suite's rows actually contain:

```yaml
# >>> evalshift suites (managed by `evalshift capture sync`) >>>
suites:
  briefing:                                       # tool-free: inherits the top level
    source: captured
    path: .evalshift/suites/briefing/golden.jsonl
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
# <<< evalshift suites <<<
```

Resolution is **family-level replacement**, applied by one method (`EvalShiftConfig.evaluators_for`) that `evaluate`, the report and the hosted bundle all route through, so the scored set and the reported set cannot drift apart. A family the suite does not mention is inherited from the top level; a family it mentions replaces the top-level one wholesale (no deep merge, no per-name merge); a family written as `[]` or `null` **removes** what would have been inherited — absent and `null` are different instructions. A run launched with a raw `--suite <path>`, or with a `--suite-name` that has no entry, resolves to the top-level block, so nothing that predates per-suite evaluators changes behaviour.

Derivation, on every `capture sync`: no row offered a toolset → **no block** (`structural` is never derived — nothing in a capture says what shape an answer must have); any row offered a toolset → `tool_selection` (`conformance: expected` + `divergence: set`); any row recorded call arguments → `tool_arguments` (`against: expected`, no `strategies:` — the default `auto` already grades free text by meaning). Generated names are stable by contract (`routing`, `routing_args`) because reports key on evaluator names across runs. Sync regenerates only the suites it just promoted and carries the rest of the region forward verbatim; `managed: false` on an entry freezes it, and sync prints the block it would have written instead. One hand edit survives regeneration: a managed entry's `evaluators.trace_invariants` block is copied verbatim into the entry sync rewrites, because no capture can derive a team's rules. The marker comment still says hand edits inside the region are overwritten — its text is matched exactly to find the region, so it cannot change — and this is the documented exception to it.

### Model ids

Model resolution is deliberately permissive: a small built-in registry maps aliases to canonical `provider/model` ids, and anything unknown is passed through with provider inferred from the prefix (`gemini-*` → Google, `claude-*` → Anthropic, `gpt-*`/`o1-*`/`o3-*` → OpenAI, `deepseek-*` → DeepSeek). LiteLLM is the call-time authority — **any model LiteLLM supports works**; the registry never gates. Before a live run the CLI checks that the inferred provider's API key env var is set.

---

## The golden suite

The suite is a JSONL file — one example per line. Default path `./golden.jsonl`, overridable with `--suite <path>` or `--suite-name <name>` (a key under `suites:` in config, e.g. a promoted capture suite).

```jsonl
{"id": "ex_security_01", "inputs": {"query": "User account_42 had 5 failed login attempts in the last hour"}, "tags": ["security"], "expected_tools": [{"tool_name": "notify_security_team", "match_strategy": "subset"}], "toolset_ref": "sha256:1a2b3c..."}
{"id": "ex_text_only_01", "inputs": {"query": "What is your refund policy?"}, "tags": ["text_only"], "expected_no_tools": true, "tools": []}
```

### Example fields

| Field | Type / default | Meaning |
|---|---|---|
| `id` | str, required, unique | Example id |
| `inputs` | dict, `{}` | Template-variable → value; must cover the prompt's `variables` |
| `tags` | list[str], `[]` | Slice labels |
| `expected` | dict \| None | Reference output (most evaluators compare source vs target directly and ignore this) |
| `expected_tools` | list \| None | Ground-truth tool calls, in order (agent prompts) |
| `expected_tool_rounds` | list[list] \| None | The full recorded agent loop, one list per model turn that emitted tool calls. `expected_tools` is `expected_tool_rounds[0]` — see [Capturing from production](#capturing-from-production) |
| `tool_result_fixtures` | list[list] \| None | Recorded results of the calls in `expected_tool_rounds`, one inner list per covered round, positionally aligned (`{tool_name, result, error}`). Written by `--rounds all`; when present `run` replays the example teacher-forced, one round per covered round plus the answer round. `None` = single-shot replay |
| `expected_tool_count` | int ≥ 0 \| None | Pin the total tool-call count |
| `expected_no_tools` | bool, `false` | Assert the model answers without any tool call — only meaningful when the example's toolset was non-empty; see below |
| `expected_parallel` | bool \| None | Assert parallel (or strictly sequential) tool calling |
| `history` | list \| None | Prior conversation turns for teacher-forced replay, including `tool_calls` / `tool` results (see [Multi-turn](#multi-turn-conversations)) |
| `conversation_id` | str \| None | Groups sibling turns of one conversation |
| `turn_index` | int ≥ 0 \| None | Position within the conversation |
| `generation_config` | dict \| None | Generation settings recorded by the SDK on the capture's first `model_call` (`temperature`, `response_mime_type`, `response_schema`, `top_p`, `response_format`, `max_output_tokens`, `max_tokens`, `tool_choice`, `parallel_tool_calls`, `tool_config`), copied verbatim by `capture promote`/`sync`. The runner translates it at dispatch and applies it to **both** the source and target calls: `temperature` overrides the registry default, `response_mime_type: "application/json"` (plus an optional `response_schema` dict) becomes a LiteLLM `response_format` (`json_schema` with a schema, `json_object` without), a litellm-shaped `response_format` dict passes through, and `tool_choice` (OpenAI string/object **or** Anthropic object) / `tool_config` (Gemini) / `parallel_tool_calls` normalise to an OpenAI-style `tool_choice` + `parallel_tool_calls` that LiteLLM maps per provider — see [Agent evals → Tool-choice constraints are replayed too](docs/agents.md#tool-choice-constraints-are-replayed-too). Folded into the response cache key — an example without the field keeps byte-identical cache keys. Keys the runner cannot translate are ignored with a warning; delete the field to disable the override |
| `toolset_ref` | str, required (exactly one of this / `tools`) | Content-addressed pointer (`sha256:<hex>`) to a toolset sidecar under `<base>/toolsets/`. What `capture promote`/`sync` write, carried verbatim from the source capture's first `model_call` |
| `tools` | list \| None, required (exactly one of this / `toolset_ref`) | The example's toolset, inlined — what you write by hand. `[]` is a real, valid value: "this example's agent had no tools available" is a first-class assertion, not an absence. Each entry is `{name, description, input_schema}`, plus an optional `strict: true` (OpenAI shape spells it `function.strict`) that is replayed to targets whose provider has a strict mode |

Every example must carry a toolset: every model call records the toolset it was offered, so a suite example — promoted or hand-authored — must record it too. `capture promote`/`sync` always write `toolset_ref`; write `tools` (even `tools: []`) when authoring a suite by hand.

Each entry in `expected_tools`:

```json
{"tool_name": "issue_refund",
 "arguments": {"order_id": "12345", "amount_usd": 42.5},
 "match_strategy": "subset",
 "provenance": "captured"}
```

- `arguments: null` → name-only check.
- `match_strategy`: `exact` (arguments must match exactly), `subset` (default; expected keys must be present and equal, extras allowed), `contains_per_field` (per-field containment).
- `provenance`: `captured` (default, what `capture promote`/`sync` write — transcribed from the source model's own call, unverified) or `reviewed` (a human has confirmed it). Scoring is identical; the flag only decides whether the run discloses that its ground truth is source-derived — see [Agent evaluation → Ground truth](#ground-truth).

Validators enforced at load: exactly one of `toolset_ref` / `tools` is required — neither, or both, fails to load; `expected_no_tools: true` is incompatible with non-empty `expected_tools`, a non-empty `expected_tool_rounds`, or a nonzero `expected_tool_count` — and so is an empty toolset, spelled `tools: []` or a `toolset_ref` naming the empty toolset (a call offered no tools cannot produce a tool call); `tool_result_fixtures` requires `expected_tool_rounds`, cannot cover more rounds than it has, and every covered round needs one result per expected call with a matching `tool_name`; `history` may contain at most one `system` message and it must come first; duplicate ids across the suite are rejected. The loader collects **all** schema errors before failing, so you fix a broken suite in one pass.

---

## Prompts

`prompts` is the template axis and `suites` the dataset axis: every prompt is rendered with every example of the suite being run, so both are always present. In a capture-first project the `replay` prompt that `init` writes (`content: "{input}"`) is a passthrough — a promoted example is `{"input": "<full rendered prompt>"}` — and that is what makes captured inputs replayable.

Two detection modes tell EvalShift where a prompt's body lives:

```yaml
prompts:
  - id: greeting
    detection: manual            # inline
    content: "Summarise: {text}"
    variables: [text]

  - id: customer_routing
    detection: python_string     # sourced from your code
    path: prompts.py
    variable: AGENT_SYSTEM_PROMPT
    variables: [query]
    max_tokens: 2048             # optional per-prompt override
```

- **`manual`** — the body is the inline `content`, verbatim.
- **`python_string`** — EvalShift **AST-walks** the `.py` file for a module-level assignment `VARIABLE = "..."` and takes the string literal. Your code is **never imported or executed**. Only a plain string constant is accepted; f-strings, concatenation (`"a" + "b"`), `.format()` calls, function calls, attribute access, and name references are all rejected with a labeled error (workaround: switch that prompt to `detection: manual`). If the variable is assigned more than once at module level, the last assignment wins.

`variables` declares the `{placeholder}` names the template uses; every example's `inputs` must supply them (validated up front, before any money is spent).

Nothing on the prompt switches it to an agent path — the toolset comes from
each golden-suite *example* instead (`toolset_ref` or inline `tools`), so
one prompt can dispatch some examples plainly and others with tools in the
same run. See [Agent evaluation](#agent-evaluation).

---

## Evaluators

Evaluators score each (source, target) output pair. Scores live in `[0, 1]`; the **delta = target − source**, so negative deltas are regressions. Every evaluator config accepts:

- `blocking: true|false` (default `true`) — **blocking** evaluators feed the migration-policy verdict and CI gates; **advisory** (`blocking: false`) evaluators are computed, reported, and summarised separately but can never fail a run on their own. The `init` scaffold ships semantic and judge as advisory deliberately: at small suite sizes their noise would gate the verdict.
- `applies_to: ["*"]` — prompt-id globs. Accepted by every family, enforced only by [`trace_invariants`](#trace-invariants); a no-op on the others.

When an evaluator's own measurement breaks (judge call fails, embedding call fails), the record is stored as **errored and excluded from the statistics** — not silently scored neutral. Upstream failures are recorded the same way: if a *model call* failed or was truncated, the pair gets an errored row (a 0.5/0.5 placeholder with the error attached), kept in `scores.jsonl` and excluded from the statistics. The run always completes.

### Structural (`evaluators.structural`, list) — free, no API calls

| Type | Score | Config |
|---|---|---|
| `json_schema` | 1.0 if output parses and validates (JSON Schema Draft 7), else 0.0 | `schema_path:` (path to a schema file, relative to the project root) |
| `regex` | 1.0 if `re.search` matches, else 0.0 | `pattern:` |
| `length` | 1.0 in bounds; linear decay outside (0.0 at 2× the boundary) | `min_chars` / `max_chars` (at least one) |

### Semantic (`evaluators.semantic`, single block)

Embeds both outputs and scores the target by cosine similarity to the source (source score pinned at 1.0). Config: `embedding_model` (default `text-embedding-3-small`; the Gemini scaffold uses `gemini/gemini-embedding-001`), `min_similarity` (default 0.9) — below it the pair is flagged `SEMANTIC_REGRESSION`. Within it, drift counts as *equivalent* for policy purposes. The yardstick is the source output, never the suite's `expected` field, so this measures drift rather than correctness: cosine distance can't tell "reworded" from "wrong", which is why the scaffold keeps it advisory (the library default is `blocking: true`; see [FAQ](docs/faq.md#why-does-a-hand-written-config-block-on-semantic-when-init-does-not)) and why correctness belongs to an `llm_judge` criterion.

On an agent turn where **both** models answered with tool calls and no prose, there is nothing to embed — the evaluator writes **no record at all** (no provider call) rather than erroring on an empty embedding input or inventing a score. A turn where only *one* side is empty is still scored: a target that went silent where the source answered is exactly the regression this evaluator exists to catch. Since the empty side cannot be embedded (the provider 400s on empty input), the pair scores **0.0 similarity by definition** with no embedding call — still gated by `min_similarity` as usual — and the record carries `empty_side: "source" | "target"` metadata plus an explanation the report shows verbatim. The mirrored case (source silent, target answered) scores identically.

### LLM-as-judge (`evaluators.llm_judge`, list)

Pairwise A/B comparison per criterion: the judge sees the two outputs anonymised and **order-randomised** (positional-bias mitigation) and picks a winner or a tie. Winner → scores (0, 1); tie → (0.5, 0.5). Config: `criterion_name`, `criterion_prompt`, `judge_model` (per criterion; built-in default `gemini-3.1-flash-lite-preview`). Write criteria symmetric (never mention "source"/"target") and include an explicit tie instruction; prefer a judge from a third model family so it isn't grading its own relatives. Multi-turn examples render the transcript to the judge (capped at 4,000 chars). Tool-only turns — both outputs empty — write **no record at all** and spend no judge call, since comparing two empty strings only ever returns a meaningless tie, and a fabricated tie is indistinguishable from a judged one.

### Tool-call evaluators — see [Agent evaluation](#agent-evaluation)

- **`tool_selection`** — did the model call the right tools? **Two independent axes, one record each** (`kind: tool_selection.conformance` / `tool_selection.divergence`), because a migration asks both and the answers differ. `conformance` grades **each side absolutely** against the example's ground truth: `expected` (default; `expected_tools` matched **in order**), `expected_set` (the same comparison, order-insensitive multiset recall, for parallel fan-outs), `off`. `expected_no_tools` examples score 1.0 iff zero calls, under either strategy — it is this axis's input, not a switch over the evaluator. `divergence` grades the **target against the source**, which is its own baseline at 1.0: `set` (default; Jaccard on the tool-name sets), `exact` (sequence equality), `first` (first call only), `off`. `set` is the default rather than `exact` so reordered identical calls do not read as drift. Both axes off is a config error. Why the split: with one switch the ground-truth branch won unconditionally, and a suite promoted from captures — where every row carries `expected_no_tools` — scored two models that called entirely different tools as `0.0 / 0.0`, a zero delta, filed as *equivalent*. A conformance record where **both** sides missed is tagged `TOOL_GROUND_TRUTH_MISS`: ground truth captured from the source model that the source model then fails is a broken harness, not a migration finding. Extra: `severity_floor: low|medium|high|critical` — a regression here can never be classified below the floor, regardless of effect size. In `report.html` the two axes render as separate, separately-labelled rows carrying what each one compares; an axis on which every pair was a `TOOL_GROUND_TRUTH_MISS` is headlined **Ground truth missed by both** rather than "Equivalent". The per-example table gains a `Tools called (source → target)` column and each top-regression card names both sides' tools, read off the record's own metadata — the whole content of a divergence finding is the two names, and nothing rendered them. On a teacher-forced multi-round replay both axes score **per round** (conformance against `expected_tool_rounds[k]`, "called nothing" for the answer round; divergence target-vs-source within round *k*), the record's scores are the mean over the replayed rounds (a round with no ground truth in which neither side called anything is skipped), and `metadata.rounds` lists `{round, expected_names, source_names, target_names, source_score, target_score}` per round; the report renders one `r1 a → a` line per round and `Round k:` prefixed trace diffs. Single-shot pairs are unchanged and carry no `rounds` key.
- **`tool_arguments`** — same tools, different arguments? Calls matched greedily by `(tool_name, nearest sequence_index)`; each argument field scored by a per-field strategy: `exact`, `subset`, `numeric` (relative error, linear decay to 0 at `numeric_tolerance`, default 0.05), `semantic` (embedding cosine, borrows the configured `semantic` evaluator's model and cache — without one it degrades to `exact`), or `auto`. Fields `strategies` does not name are scored by `default_strategy`, **`auto`** by default: a ladder that (1) scores two strings equal after normalizing case and whitespace at 1.0, (2) dispatches on the field's declared type in the toolset the example carries — identifiers (`*_id`/`*_ids`), enums, booleans and `date`/`date-time`/`uuid`/`email` formats → `exact`, numbers → `numeric`, objects/arrays → `subset`, and (3) *grades* whatever is left, which is free text: embedding similarity when a `semantic` block lent a model, `difflib` ratio when not, so partial credit survives with no embedding model. `default_strategy: exact` restores byte-equality scoring; a `strategies` entry always wins. A field present on one side only scores 0.5 (`optional_fields_scored: strict` restores 0.0): omitting an optional parameter is a real difference, not a wrong value. `against: expected` switches the comparison from drift-vs-source (where `source_score` is 1.0 by construction, so a hallucinating source defines the yardstick) to correctness — **both** sides scored against `expected_tools[].arguments`, with each expectation's `match_strategy` picking the compared keys (`exact` = union, `subset`/`contains_per_field` = recorded keys only). An expected call the model never made scores 0; an example with no expected arguments is skipped at a neutral 1.0/1.0. On a multi-round replay calls are paired **within** a round (a right call in the wrong round is a miss), the score is the mean over rounds that had something to score, and `metadata.rounds` carries the per-round detail. A ground-truth field **neither** side produced is dropped from that call's denominator on both sides and disclosed as `unmeasured_fields` in the per-call metadata — a stale expectation would otherwise cap the call below 1.0 forever, since no model change could lift it.
- **`tool_trace_structure`** — shape of the trace: call-count drift (`call_count_tolerance`, default 1), parallelism match, refusal alignment (a refusal mismatch forces severity ≥ high and flags `REFUSAL_REGRESSION`), and `expected_tool_count` when set. Toggles: `check_call_count` / `check_parallelism` / `check_refusals` (all default on).
- **`agent_trace`** — for imported external traces; see [External agent traces](#external-agent-traces).
- **`trace_invariants`** — hand-written rules checked on both sides, independent of what the source did; see [Trace invariants](#trace-invariants).

### Failure categories

Regressions carry machine-readable labels that the report and hosted diff group by: `FORMAT_FAILURE`, `SEMANTIC_REGRESSION`, `TOOL_SELECTION_DRIFT`, `TOOL_GROUND_TRUTH_MISS` (both models missed the same recorded tool ground truth — a broken-harness signal, not a migration finding; see [Migration policy and CI gating](#migration-policy-and-ci-gating)), `ARGUMENT_VALUE_DRIFT`, `TOOL_TRACE_STRUCTURE_DRIFT`, `TOOL_ORDER_DRIFT`, `DANGEROUS_ACTION_DRIFT`, `MISSING_VERIFICATION_STEP`, `UNNECESSARY_TOOL_CALL`, `REFUSAL_REGRESSION`, and `INVARIANT_VIOLATION` (the target broke a hand-written [trace rule](#trace-invariants), stamped whatever the source did). That is the complete set — every label the evaluators emit is declared in `evaluators/failures.py`. The machine labels live in `scores.jsonl`, `report.json` and the bundle; every rendered surface (the HTML report, decision prose, the run narrative) shows the plain-language display name instead — `TOOL_SELECTION_DRIFT` renders as "Different tools chosen" — with the mapping declared beside the labels in `evaluators/failures.py`.

`ARGUMENT_VALUE_DRIFT` counts **regressions**: it is stamped only when the target scored below the source. Under `against: expected` both models can miss the same recorded expectation by the same margin — a zero delta, and a fact about your ground truth rather than a migration defect, already reported as such. Policy budgets are unaffected by the label: `max_tool_argument_drift` counts calls whose *target* score fell below `tool_argument_drift_floor`.

### Cost

Structural and tool-call evaluators are free (pure computation over recorded outputs) — except that when an `evaluators.semantic` block exists, `tool_arguments` embeds free-text argument values (under the default `auto` strategy) and `semantic`-strategy fields with its model. Semantic costs one embedding call per output (one per pair when the two outputs are identical); `llm_judge` costs one judge-model call per (pair × criterion) — usually the dominant evaluation cost. Both go through the same cache as everything else, so re-running `evaluate` over an unchanged run costs nothing.

---

## Agent evaluation

Each golden-suite example carries its own toolset — a `toolset_ref` pointing at a content-addressed sidecar (what `capture promote`/`sync` write) or an inline `tools` list (`[]` is a real, valid "no tools offered" value). `run` resolves it per example and, for any example whose toolset is non-empty, sends the provider the tool definitions and records the response as a provider-agnostic `ToolTrace` (ordered `ToolCall`s with `tool_name`, `arguments`, `call_id`, `parent_call_id`, `sequence_index`, `round_index`, plus `final_text`, refusal info and `round_count` — a teacher-forced multi-round replay merges its rounds into one trace, one `raw.jsonl` row per example per model, tokens/cost/latency summed and `text` the last round's answer); the tool-call evaluators then score the trace against that example's ground truth. Two examples under the same prompt can carry different toolsets — or none — so one suite freely mixes agent and text-only rows. The same trace is visible in the local HTML report's side-by-side trace diffs, and — once the run is bundled and pushed — on the hosted run-detail page; see [Hosted EvalShift](#hosted-evalshift).

The canonical agent-migration failure this catches: the candidate model silently **stops calling `notify_security_team`** on security-sensitive tickets. `tool_selection` with `severity_floor: high` turns that into an unmissable red row.

### Toolset shape

A toolset — a `<base>/toolsets/<hex>.json` sidecar, or an example's inline
`tools` — is a list of tool dicts. Both provider shapes are accepted, and a
sidecar file may hold a flat list or `{"tools": [...]}`:

```yaml
# Anthropic-shape
- name: issue_refund
  description: Issue a refund on an existing order.
  input_schema:
    type: object
    properties:
      order_id:   {type: string}
      amount_usd: {type: number}
      reason:     {type: string}
    required: [order_id, amount_usd, reason]

# OpenAI-shape (equivalent)
- type: function
  function:
    name: issue_refund
    description: Issue a refund on an existing order.
    parameters: {type: object, properties: {...}}
```

The model client serialises to whatever shape the target provider expects, so
one toolset serves Anthropic, OpenAI, and Gemini models alike. You rarely
write a sidecar by hand — `capture promote`/`capture sync` write one per
distinct toolset your captures recorded, content-addressed by
`fingerprint_tools` so two captures offering the same tools share one file.
A hand-authored suite skips the sidecar and inlines the same shape directly
as an example's `tools:` list instead — keep a `tools.yaml` alongside your
suite as the human-readable source you copy from, if that helps; nothing
in `evalshift.yaml` reads it.

### Ground truth

Per-example expectations (`expected_tools`, `expected_no_tools`, `expected_tool_count`, `expected_parallel`) are described under [The golden suite](#the-golden-suite). You rarely write them by hand — `capture promote`/`capture sync` derive them from recorded production behaviour (`--strict-args`, `--names-only`, `--tool-count` control how strict the derived expectations are).

Derived expectations are marked `provenance: captured` (the default), meaning their arguments were transcribed verbatim from the source model's own recorded call — nobody has checked that they are *right*. On such a row the source scores 1.0 **by construction**, so an `against: expected` gate degenerates into what `against: source` already measured: target deviation from source. The run discloses that rather than letting `source_score: 1.0` read as evidence — when every scored row is `captured`, `migration_decision.json` gains a recommendation naming the count and the caveat. Set `provenance: reviewed` on a row once a human has confirmed its arguments; scoring is identical either way, and the disclosure goes silent as soon as one row is reviewed (a blanket disclaimer would understate the rows someone did check).

### Trace invariants

Every other tool evaluator treats "the source did it" as correct, so a wrong call both models make reads as agreement. `evaluators.trace_invariants` holds **both** sides to rules your team wrote, and a rule the target breaks counts against it whatever the source did. Use it for the few invariants where a silent miss is expensive — auth before a write, a deprecated endpoint, at most one charge, argument bounds — and keep production traces as the broad corpus. An ordering that matters belongs in an `order` rule, not in the recorded `expected_tools`, whose order `conformance: expected` grades as recorded, incidental or not.

```yaml
evaluators:
  trace_invariants:
    - name: payments_contract
      owner: "@payments-team"     # optional; shown on every violation
      applies_to: ["checkout-*"]  # prompt-id globs, enforced here; default ["*"]
      blocking: true              # default; false = reported, never gates
      traces: replayed            # default; `imported` checks `traces import` traces
      rules:
        - {id: auth-before-charge, type: order, before: authenticate, after: charge_card}
        - {id: no-legacy-refund, type: forbidden, tools: [refund_v1]}
        - {id: one-charge, type: call_count, tool: charge_card, max_calls: 1}
        - {id: must-book, type: required, tools: [create_booking]}
        - id: sane-amount
          type: arguments
          tool: charge_card
          json_schema: {type: object, required: [amount], properties: {amount: {type: number, minimum: 0.01, maximum: 5000}}}
```

Rules are checked per **response** (one model turn) against the tool calls already in its context: on a replay, the `history` turns' calls plus the *recorded* earlier rounds a teacher-forced replay fed back (never the model's own earlier rounds); an imported trace is one response with an empty context. Context calls are never violations, but they can satisfy or count toward a rule:

| `type` | Violated when | Context role |
|---|---|---|
| `forbidden` | the model calls any tool in `tools` (one violation per call) | none |
| `required` | some tool in `tools` is called in no response by the model | none — use `call_count` `min_calls` when the conversation may already contain the call |
| `order` | the model calls `after` with no `before` earlier, in the context or earlier in the same response | a context `before` satisfies it |
| `call_count` | `max_calls`: a model call takes the running count (context first) past it, one violation per excess call. `min_calls`: context plus model calls at the end of the last response fall short (one violation). Both equal = exact | context calls count toward both bounds |
| `arguments` | a model call to `tool` fails `json_schema` (Draft 2020-12, validated at config load); unparseable arguments fail any schema that requires a field | none |

`required` and `min_calls` are the two rules that are not vacuous on a trace that never mentions the tool, so on a single-shot replay (which sees only the first response) scope them with `applies_to`, or use `--rounds all` or imported traces. An entry whose `applies_to` matches no prompt in the run (with `traces: imported`, no imported trace pair) checks nothing and leaves no row, so `evaluate` prints a warning naming the entry and its `applies_to` (suppressed under `compare`'s quiet scoring); an entry that matched at least one prompt stays silent. A `traces: imported` entry on a run without `traces.jsonl` fails `evaluate` with an error naming that family. Each side scores the share of rules it broke zero times; the record's kind is `trace_invariants`, its metadata carries `rules_checked`, `source_violations`, `target_violations` (each `{rule_id, rule_type, tool, round_index, detail}`) and `owner`, and the target's violations stamp `INVARIANT_VIOLATION`. [`max_invariant_violations`](#migration-policy-and-ci-gating) turns a target violation of a blocking entry into a failed verdict. The HTML report's **Trace rules broken** panel and `report.json`'s `invariant_violations` list every rule the target broke, whatever the delta — advisory entries' violations included, though only blocking ones count toward the budget. Each `invariant_violations` row is `{prompt_id, example_id, evaluator_name, owner, rule_id, rule_type, tool, round_index, detail, source_also, blocking, sample}`; `source_also` says whether the source broke the same rule on that example, `blocking` is the entry's flag (`false` rows are reported but never gate, and the panel tags them **advisory**), and `sample` is `null` on a single-sample run and otherwise the violating sample's zero-based ordinal among the samples that scored successfully (not the raw `sample_index`). The **Overall, by evaluator** table tags every advisory (`blocking: false`) evaluator's row the same way and says what a trace-rule row measures (each side judged against the entry's rules, not against the other side); the **Top regression causes** panel (`failure_categories` in the decision) counts `INVARIANT_VIOLATION` as the budget does — distinct examples with a blocking target violation, never per row and never for advisory entries; and the per-prompt executive summary (with the header's mean score Δ, the mean of its rows) is drawn from blocking evaluators only — worst severity first, ties to the most negative effect size — falling back to advisory ones on a prompt that has nothing else. Full reference: [docs/configuration.md](docs/configuration.md#evaluatorstrace_invariants).

---

## Multi-turn conversations

EvalShift evaluates multi-turn agents by **teacher-forced replay**: each turn is one suite example carrying the conversation so far.

```jsonl
{"id": "conv1_t2", "inputs": {"input": "1pm works"}, "conversation_id": "conv_9f2", "turn_index": 2,
 "tools": [{"name": "get_calendar", "description": "List free slots on a day.", "input_schema": {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]}}],
 "history": [
   {"role": "system", "content": "You are a scheduling assistant."},
   {"role": "user", "content": "Can we move my appointment?"},
   {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_calendar", "arguments": {"day": "tue"}}]},
   {"role": "tool", "tool_call_id": "c1", "content": "{\"slots\": [\"1pm\"]}"},
   {"role": "assistant", "content": "Sure — what time works?"}
 ]}
```

When `history` is present, `run` sends the recorded prefix **verbatim**, followed by the current turn's rendered prompt as the final user message (via the messages-based client path; the tools variant for agent prompts). The candidate model never generates its own intermediate turns — both models see byte-identical context, and only the **current turn's** output is compared. That keeps every turn a clean paired measurement.

Rules and limitations:

- `history` may contain at most one `system` message, and it must be first. `history: null` means single-turn; `history: []` is a conversational example with no prefix.
- Re-driving a whole conversation (letting the candidate generate turn 1 and feeding *its own* reply into turn 2) is deliberately unsupported — it breaks the paired-comparison contract.
- Captures recorded with the SDK's messages-list convention recover `history` **verbatim**; captures that only recorded a bare string get history **reconstructed** from sibling turn captures (an approximation: assistant replies come from final outputs, intermediate tool exchanges are absent). Prefer the messages-list convention in your instrumentation.
- History carries the **agent loop**: an `assistant` turn may have `tool_calls` (`{id, name, arguments}`), and a `tool` message carries that call's result keyed by `tool_call_id` (required — an unpairable result is a load-time error). They are dispatched in the OpenAI wire shape and translated per provider by LiteLLM, so the candidate model sees the tool results production saw. A promoted `tool` message with no recorded id is keyed, in order, to the preceding assistant turn's still-unanswered call (how id-less providers such as Gemini pair them), with a warning; only a result with no call to answer gets a positional `_pos<N>` id.
- The report shows a `turn N` badge on conversational examples.

---

## External agent traces

If your agent runs outside EvalShift (LangChain, a custom loop, another language), you can still score its behaviour: run the model-call stage first, then attach full agent timelines to the run:

```bash
evalshift traces import <run-id> --source source_traces.jsonl --target target_traces.jsonl
```

Each JSONL line is one trace for a `(prompt_id, example_id, role)` with an `events` list. Event types: `model_call`, `tool_call`, `tool_result`, `retrieval`, `guardrail`, `final_output`, `error` — the same schema the evalshift-sdk writes. A `model_call` separates **offered** (`toolset_ref` / `tools_offered` — what was passed to the model), **requested** (`requested_tool_calls`, a list of `{name, arguments, call_id}` — what the model asked for in its response), and **executed** (the `tool_call` / `tool_result` events — what the app actually ran); the three can legitimately differ, and `requested_tool_calls: null` means the trace predates the field, not that no tools were requested. Events sort by `sequence_index` (duplicates rejected). Every `timestamp` must carry a UTC offset (`...Z` or `...+02:00`); an offset value is converted to UTC on import and a naive one is rejected with its file and line, because assuming UTC would silently relabel a trace recorded elsewhere. `--strict` fails when any completed pair lacks a trace pair.

Score imported traces with the `agent_trace` evaluator:

```yaml
evaluators:
  agent_trace:
    - name: safety
      check_tool_order: true          # LCS-normalised order similarity
      check_arguments: true           # per-field equality on matched calls
      check_missing_verification: true
      verification_tools: [confirm_with_user]
      dangerous_tools: [delete_record, transfer_funds]
```

Extra dangerous calls on the target flag `DANGEROUS_ACTION_DRIFT`; a dangerous call with no preceding verification tool flags `MISSING_VERIFICATION_STEP`; order changes flag `TOOL_ORDER_DRIFT`.

`dangerous_tools` compares the target against the source, so a dangerous call both sides made is not flagged. To hold both sides to rules instead, add a [`trace_invariants`](#trace-invariants) entry with `traces: imported`: each imported trace is checked as one response with an empty context, and a rule the target breaks counts toward `max_invariant_violations` whatever the source did.

Trace-aware debugging: `evalshift diff case <run> <example>` (side-by-side trace diff when traces exist, text diff otherwise), `evalshift inspect case <run> <example>`, `evalshift replay case <run> <example> --model target --trace`.

---

## Statistical methodology

Full contract in [docs/methodology.md](docs/methodology.md). Per `(prompt, evaluator, slice)` comparison over the paired per-example deltas:

1. **Sample guards.** A pair an evaluator measured nothing on (a tool-only turn, an example with no ground truth) has no row in `scores.jsonl` at all, so it is absent from `n`, from the slice aggregates and from the policy metrics by construction. The count is reconstructed from `state.json`'s `evaluator_coverage` and noted, so "8 of 10 rows not applicable" survives the rows themselves. n < 5 → skipped, severity `insufficient`. 5 ≤ n < 20 → tested, but flagged uncertain. Zero-variance deltas → skipped, severity `none`. **No applicable rows left** → severity `insufficient` with a `nothing measured:` note, never `none`; a run whose blocking evaluators all landed there cannot return `pass` — it is downgraded to `conditional_pass` (or stays `inconclusive`) with those evaluators named in `recommendations`, since an evaluator that compared nothing is unknown, not equivalent. Advisory (`blocking: false`) evaluators are exempt — their silence gates nothing by design, so they are neither named nor demote the verdict: `evaluator_coverage` carries each axis's `blocking` flag, and the synthesized comparison gets an `advisory:` note (zero rows leave the policy layer nothing else to read the flag from). The HTML report gives that row its own headline — **Nothing measured**, not the **Not enough data** the rest of `insufficient` gets.
2. **Normality screen.** Shapiro–Wilk on the deltas at α = 0.05: normal → **paired t-test**, non-normal → **Wilcoxon signed-rank**. n > 5000 skips the screen (CLT, t-test).
3. **Effect size.** Paired Cohen's d = mean(Δ)/std(Δ), with a 95% CI — analytical for the t-test, percentile bootstrap (2,000 resamples, seeded, deterministic) for Wilcoxon.
4. **Multiple comparisons.** Benjamini–Hochberg FDR correction at α = 0.05 across *all* testable comparisons in the run (equivalent to `statsmodels fdr_bh`).
5. **Severity** from the corrected p-value, |d|, and direction:

| Severity | Condition |
|---|---|
| `critical` | regression, p < 0.01 and \|d\| > 0.8 |
| `high` | regression, significant, \|d\| > 0.5 |
| `medium` | regression, significant, \|d\| > 0.2 |
| `low` | regression, significant, small effect |
| `improved` | significant, delta > 0 |
| `none` | not significant |
| `insufficient` | n < 5 |

Truncated outputs (hit the `max_tokens` cap) are excluded from the statistics; empty-but-complete outputs count. `severity_floor` on an evaluator (e.g. tool_selection) prevents downgrading its regressions below the floor.

**Sampling control.** Every test above assumes the only difference between the two arms is the model, which holds because EvalShift sends `temperature=0` on every call. Providers are beginning to withdraw the parameter (Google has announced it for Gemini 3+); since EvalShift also sets `drop_params=True`, a withdrawal would otherwise degrade silently — the call succeeds, sampling reverts to the provider default, and every p-value weakens with nothing saying so. Each arm is checked against LiteLLM at run start; affected ids land in `state.json` under `non_deterministic_models`, and the report shows a banner above the verdict plus a methodology note. A second failure mode is caught at call time rather than run start: reasoning-tier models (for example `gpt-5.6-terra`) advertise `temperature` but reject every value except their default with a 400, and `drop_params` does not cover them (LiteLLM special-cases only o-series names). The first such rejection makes EvalShift resend the call without `temperature` and stop sending it to that model for the rest of the process; the model joins `non_deterministic_models` and the same banner — this covers judge models too, merged when scoring completes. One call per affected model fails and is resent adapted, so nothing is lost, but sampling for that model is the provider default, not controlled. Such runs measure model change *plus* sampling noise — non-significant results are weak evidence there, and the fix is more examples, not a tighter threshold. EvalShift does not move sampling guidance into the system prompt to compensate: that would change the prompt under test, for one arm only.

**Recorded generation constraints.** A promoted capture can carry the generation config its original call used (`generation_config` on the suite example: `temperature`, `top_p`, `response_format` / `response_mime_type` / `response_schema`, `max_tokens` / `max_output_tokens`, `tool_choice` / `tool_config`, `parallel_tool_calls`). The replay sends those constraints, and `drop_params=True` means a model that never accepted one still answers — with the constraint missing. At run start EvalShift asks LiteLLM which of the parameters the suite actually recorded each arm supports (Gemini's spellings are mapped to their OpenAI names first; `temperature` is left to the sampling probe above, which owns its own banner). Anything a model positively lacks lands in `state.json` under `dropped_params` as `model id → [param, …]`, is logged once per (model, parameter) at `WARNING`, and gets a **Constraints not honoured** banner in the report next to the sampling one. A second, hard-coded source feeds the same record: parameters LiteLLM *claims* to support and then discards while building the provider body, which no probe can detect. Today that is two Gemini cases — `parallel_tool_calls`, and a tool's `strict` flag under the pseudo-parameter name **`tools.strict`** (a field on the `tools` array, not a generation parameter, so it has no probe of its own). Both sources merge into one sorted list per model. As with the sampling probe, uncertainty reads as "supported": an exception, a `None`, or an empty answer from LiteLLM records nothing, because a false banner on every report costs more than one missed warning. Such an arm measures *model change plus a missing constraint*; `migration_policy.fail_on_dropped_params: true` turns that from a caveat into a failing verdict for suites where the constraint is the contract.

---

## Migration policy and CI gating

`migration_policy` turns statistics into a decision. `analyze` writes `migration_decision.json` with a verdict: **`pass`**, **`conditional_pass`**, **`fail`**, or **`inconclusive`**.

Budgets (fractions, not percents):

| Field | Default | Meaning |
|---|---|---|
| `max_overall_regression_rate` | 0.30 | Share of blocking records that regressed |
| `max_critical_regressions` | 1 | Count of critical-severity regressions |
| `min_equivalence_rate` | 0.75 | Floor on the non-regression rate (equivalent **or improved** both count) |
| `max_tool_argument_drift` | 0.20 | Share of records with argument drift |
| `max_tool_divergence` | 0.20 | Share of `tool_selection.divergence` records where the target called different tools than the source |
| `max_invariant_violations` | 0 | **Count** of examples on which the target broke a rule of a blocking [`trace_invariants`](#trace-invariants) evaluator, whatever the source did. A breach always fails |
| `tool_argument_drift_floor` | 0.9 | Target argument score below which a call counts as drifted. Argument scores are continuous — without a floor every reworded query counts as drift. |
| `max_cost_increase` | 0.30 | Relative avg-cost increase, target vs source |
| `max_latency_increase` | 0.30 | Relative avg-latency increase |
| `fail_on_dropped_params` | `false` | Fail when `state.json` → `dropped_params` is non-empty — an arm could not honour a generation constraint the source capture recorded. Top level only: a model either accepts a parameter or does not, which no subset of examples can vary. |
| `slices` | `{}` | Per-slice overrides keyed by example tag — each distinct tag is a slice (unset fields inherit the top level) |

These defaults are a first-migration starting point, not a shipping gate: a fresh suite should *report* the regressions it found rather than fail on a couple of reworded tool arguments. `evalshift init` writes exactly these numbers (`--profile` picks a tighter set — `cost-reduction`, `quantization`, `provider-switch`, `local-model`); tighten them as the suite grows and the migration nears merge.

How the verdict is computed:

- **Only blocking evaluators gate quality.** Advisory records are summarised separately (`advisory`, `advisory_regressions`) and never flip the verdict. If *every* configured evaluator is advisory (the fresh-`init` state), the verdict is `inconclusive` with guidance to promote evaluators to blocking as the suite grows — **unless** the call-derived `max_cost_increase`/`max_latency_increase` budgets are breached, which still **fails** the run: those measure the calls, not the evaluators.
- **The four rate budgets are Wilson-CI-aware.** `max_overall_regression_rate`, `min_equivalence_rate`, `max_tool_argument_drift` and `max_tool_divergence` are each a count of records over a count of records, so each carries a 95% Wilson interval (`ci_low`/`ci_high`) on its rate — the first three are also the ones the hosted gate computes an interval for. A breach of one of them only **fails** when the interval confirms the breach (the CI's favourable bound still clears the budget → the suite is too small to be sure → `inconclusive`, not `fail`). A budget the observation *held* stays conclusive however wide its interval: a thin sample must never turn a clean run into a caveat. Cost and latency budgets are exact — a measured breach of one is never softened — but they are conclusive only when they measured something: their ratio falls back to `0.00` both when either arm has no error-free call and when **both** arms average zero (unpriced models, whose calls carry `cost_usd`/`latency_ms` of `0` on every row), and either way `conclusive: false`. The record-derived budgets (regression rate, equivalence rate, critical count, tool-argument drift, tool-selection divergence) report `conclusive: false` on a scope that scored zero records, since their `0/0` default measures nothing. The two per-axis budgets need one of *their own* rows: a scope that scored plenty but ran no `tool_arguments` evaluator, or set `divergence: off`, is unmeasured too.
- **A broken trace rule fails, however small the suite.** `max_invariant_violations` counts distinct examples — `(prompt_id, example_id)` pairs with any blocking `trace_invariants` row whose target score is below `1.0`, so a row averaged over repeated samples counts when any sample broke a rule, and an example that broke rules in two entries (each entry writes its own row) counts once. Its `denominator` is the distinct examples with a blocking `trace_invariants` row. It is conclusive by construction — no interval, no `1/n` warning — and a breach overrides `inconclusive` and `conditional_pass` to **`fail`**, even when every comparison is `insufficient`. The same holds per slice: a slice whose own `max_invariant_violations` breaches reads `fail` however few comparisons it has. The `reason` names the worst scope and its count. A violation the source shares still counts; it also adds a `recommendations` line naming the rule, since either both models break it or the rule no longer matches the toolset and its owner should review it. The budget's row appears only in scopes that scored at least one blocking `trace_invariants` row; advisory-only rules add none.
- **A rate ceiling finer than `1/n` is flagged as zero-tolerance.** Rates are counted over whole rows, so on a ten-row suite the achievable tool-argument drift rates are `{0.0, 0.1, …}` and a `0.01` budget is really "any drift at all". `analyze` adds a line to the recommendations — shown in the terminal and the HTML report — naming the budget, its value, the granularity and the denominator: `The tool-argument drift budget of 1% (max_tool_argument_drift in evalshift.yaml) is below the 10% granularity of 10 tool-argument comparisons — effective tolerance is zero at this sample size.` It applies to `max_overall_regression_rate`, `max_tool_argument_drift` and `max_tool_divergence`, per scope (slices use their own row counts). Nothing is emitted for a deliberate `0.0` budget, for a denominator of `0` (already reported by `conclusive: false`), for the count/ratio budgets, or for the `min_equivalence_rate` floor. No default changes and no verdict moves — see [methodology](docs/methodology.md).
- **A shared ground-truth miss is not equivalence.** A `tool_selection.conformance` row grades each side *absolutely* against the ground truth the suite recorded, so both sides can miss it at the same height — `0.0 / 0.0` on an example both models answered with a tool call the recording never made. The delta is zero, which used to read as equivalence and is the whole of the `equivalent_rate: 1.0` a real run once shipped over a suite where nine pairs in ten routed differently. Those rows now leave **every** policy rate: they measure the harness (wrong toolset attached, wrong prompt, a suite promoted from a different agent), not the migration. They are still reported — as a `TOOL_GROUND_TRUTH_MISS` count in `failure_categories` and as a recommendations line naming how many were excluded. Only the *shared-height* case is dropped: a conformance row the target lost ground on (`0.8 / 0.3`) is still a regression and one it improved on (`0.2 / 0.6`) is still an improvement. A run whose every blocking row was a shared miss is `inconclusive`, and its recommendation is **Fix the eval harness before collecting more examples** — more pairs from the same setup are more excluded rows, so the denominator stays empty however many are added. The exclusion is deliberately narrower than the diagnosis, which is why `evaluate` runs its own broken-harness check over the **source side alone**: a suite the source fails and the target happens to satisfy has a *positive* delta and no `TOOL_GROUND_TRUTH_MISS` tag, so the exclusion rule cannot see it and it is exactly as misconfigured.
- **A cost/latency ratio measured only from zeroes says so.** When the calls exist but every `cost_usd` (or `latency_ms`) is `0` on both models, `analyze` adds a recommendations line beside the `conclusive: false`: `The cost increase budget could not be measured: all 4 error-free calls across both models recorded a cost of 0, so its observed 0.00 is a default, not a measurement.` A run with no calls gets no line — the empty `raw.jsonl` already explains itself. Emitted once per run, since every scope reads the same calls.
- **Every budget reports its own `denominator`** — the sample `observed` was computed over. Scored records for the regression rate, the equivalence rate and the critical count — counted over *measurements*, so an evaluator scoring two axes contributes two rows per example; `tool_arguments` rows for tool drift; `tool_selection.divergence` rows for tool divergence; the distinct examples with a blocking `trace_invariants` row for the trace-rule count; the error-free calls behind both averages for the cost and latency ratios. Slices report their own counts. `0` means "counted, and the sample was empty", so `observed` is a default; a *missing* `denominator` means "no sample size reported" and is **not** zero — only bundles written before the field say that, and the hosted gate falls back to `conclusive` for them. It is the same number the `1/n` granularity warning is judged on, and it is orthogonal to `conclusive`: an all-zero cost ratio counted every call it averaged and still measured nothing, so it reports a positive denominator beside `conclusive: false`. The hosted gate derives its own Wilson interval from these denominators, over the same three rate budgets and with the same confidence constant the CLI uses, so a local verdict and a hosted one now agree on whether a breach was confirmed — see [methodology](docs/methodology.md).
- Any conclusive budget failure, or any blocking critical/high-severity comparison → **`fail`**. Any lower-severity blocking regression → **`conditional_pass`**. Otherwise → **`pass`**.
- **`migration_policy` is the single source of truth for these budgets.** `migration_decision.json`'s `policy` field is the resolved policy `analyze` computed the verdict under — every top-level budget with its default applied, plus `slices`. `bundle`/`push` do not read `migration_decision.json`; `bundle` re-resolves the verdict from `evalshift.yaml` at bundle time, so the bundle's `decision.policy` (see [Hosted EvalShift](#hosted-evalshift)) is the policy the config held when the bundle was built — editing `migration_policy` between `analyze` and `bundle` changes what the bundle carries. `push <run-id>` builds a bundle only when `run_bundle.json.gz` is missing — an existing bundle is uploaded as-is — so re-run `evalshift bundle <run-id>` after a config edit. Either way, that is what lets the hosted gate check a pull request against exactly the budgets the bundle's own verdict used, instead of a separate, web-edited policy. `null` when no `migration_policy` is configured, and on a `migration_decision.json` written before this field existed. `evalshift.yaml` is the source of truth for that policy, and the web app's project policy view is becoming a read-only display of the snapshot each run pushed. Because every budget is resolved, the snapshot carries `max_invariant_violations` (default `0`) even for a config with no trace rules; hosted EvalShift accepts and stores it (since 2026-10-08) but never re-evaluates it — a run that carries its own policy is answered with the CLI's decision, which did check it, and only runs pushed without a policy fall back to the server's six-budget re-check.
- **A slice budget gates the run exactly like an overall one.** The budgets under `migration_policy.slices` are evaluated on the same terms as the top-level ones: a conclusively breached slice budget **fails** the run, and an unconfirmed breach makes it `inconclusive` — the same Wilson rule, counted over that slice's own denominator. Since the overall rows can all be green in a run a slice budget fails, `recommendations` names the one that blocked (`The 'security' slice breached its overall regression rate budget (the share of scored comparisons where the target did worse): 20% over n=20 vs the 0% limit.`) and the `inconclusive` `reason` scope-qualifies it the same way. Per-slice verdicts under `slices[*].verdict` are unchanged, and a slice that fails on *comparison severity* rather than a budget still only downgrades an overall `pass` to `conditional_pass`.
- Semantic drift that stays above `min_similarity` counts as equivalent, not regression.

CI wiring (on `analyze` and `compare`):

- `--gate critical,high` — exit 1 when any comparison at those severities exists (allowed values: `critical`, `high`, `medium`, `low`).
- `--policy-gate` — exit 1 when the policy verdict is `fail` **or** `conditional_pass`, or when no `migration_policy` is configured. `inconclusive` exits 0.
- When `$GITHUB_STEP_SUMMARY` is set, `analyze` appends a markdown results table to the job summary.

---

## Run insights

`report` (and therefore `compare`) writes a plain-language explanation of the run: one summary each for the verdict, the advisory signal and the economics, a short list of **behavioural findings** taken from the worst regressions, and a recommendation. It is rendered at the top of `report.html` and uploaded with the bundle.

**The prose is machine-written; the figures in it are not.** Every number the narrative may mention is computed first and handed to the model pre-rendered as a display string (`+102%`, `$0.0204`, `< 0.0001`), with an instruction to copy them verbatim. The output is then scanned for numeric tokens that were not supplied — one is enough to reject the generation and retry, and two bad generations fall back to deterministic templated prose (`model: "none"`, no findings). A narrative therefore cannot contain a derived, rounded or invented number.

**Internal identifiers never reach the narrative.** Budgets and failure categories enter the prompt under their display names ("Cost increase", "Different tools chosen"), and the output is scanned for the identifiers the prompt could otherwise leak — FACTS keys like `cost_delta_pct`, `evalshift.yaml` budget fields like `max_tool_divergence`, machine category labels like `TOOL_SELECTION_DRIFT`. An echoed identifier rejects the generation exactly like an invented number. User-chosen evaluator names are exempt: they are the reader's own vocabulary.

**An absent rate is handed over as "not measured", never as a figure.** The equivalence, regression and improved rates share one denominator, and over an empty one they default to `0%` — which reads as "nothing regressed" when the truth is "nothing was compared". A run whose blocking evaluators produced no comparable row, or whose every row was excluded as a shared ground-truth miss, therefore gets a digit-free marker plus the reason, and the instruction forbids describing it as equivalent, consistent or free of regressions. The permit-list does the rest: with no rate rendered, a claim like "achieved a 100% equivalence rate" carries a figure that is not a fact and the generation is rejected.

**A gate that measured nothing is never counted as a gate that passed.** A budget handed an empty sample — a tool-divergence ceiling on a run that scored no divergence row, a cost ceiling on a run with no priced call — is within its limit by arithmetic: `0/0` is below every ceiling. Counted naively that renders "7 of 7 budgets passed", and a narrative restates it as "all hard constraints are met" while the report body two sections above names the blocking evaluators that scored nothing. So `budgets_passed` counts only budgets that were *measured*, the blind ones are named on their own `unmeasured_budgets` / `unmeasured_evaluators` facts with a digit-free `coverage_basis` saying why, and the instruction forbids claiming a clean sweep, a met constraint or a safe migration while that basis is present. The blind-gate set is the same one `migration_decision.json`'s own `recommendations` name, computed by one shared function, so the two surfaces cannot disagree. This is distinct from the rates rule above and not covered by it: that fires when the *whole run* measured nothing, this fires when only *some* gate did.

```yaml
defaults:
  insights_model: gemini-3.1-flash-lite-preview   # optional; falls back to judge_model
```

- **Cost**: one model call per run (a second only when the first generation is rejected). The narrative is cached in `insights.json` and keyed on the run's `config_hash` plus the model id, so re-running `report` or `push` costs nothing; changing either invalidates the cache and regenerates.
- **Skipped** silently when `--no-insights` is passed, and with a warning when no API key is set for the chosen model or the run directory has no usable `evalshift.yaml`. A skip is never an error.
- **Never fatal.** Any failure inside generation is logged and leaves the narrative empty; a run that already has its statistics is not worth failing over a missing paragraph.
- **What gets sent to the model**: the pre-rendered figures, plus the **worst 8 regressions'** inputs and both models' outputs (each truncated to 2000 characters). That is the same exposure `llm_judge` already has, but it is real — if your suite carries data you would not send to an LLM judge, run with `--no-insights`.
- `insights.json` is a cache envelope (`{"config_hash": …, "insight": {…}}`); only the inner `insight` object is uploaded. Do not hand-edit it — an envelope the CLI does not recognise is treated as a cache miss.

---

## Hosted EvalShift

The hosted service ([evalshift.dev](https://evalshift.dev), API at `https://api.evalshift.dev`) stores pushed runs, diffs them across branches, and comments on PRs. Strictly opt-in: nothing leaves your machine unless you run `push` (or `compare --push`). Provider API keys are never uploaded.

```bash
evalshift login                       # device-code browser flow
evalshift login --token es_...        # or paste a token (verified via GET /me)
evalshift whoami
evalshift push <run-id> --project my-org/my-project
evalshift compare --suite-name <suite> --push  # pipeline + push in one go
evalshift logout
```

- Credentials live in `~/.evalshift/credentials` (owner-only permissions). Precedence: CLI flags (`--host`/`--token`) > env (`EVALSHIFT_HOST`/`EVALSHIFT_TOKEN`) > credentials file. `--no-browser` prints the approval URL for remote shells.
- Re-running `login` (without `--token`) while the stored token for that host still works reuses it — no new token is minted — and prints `already logged in as <email>`. To switch accounts, run `evalshift logout` first.
- **`login` issues a personal token — don't use one in CI.** A personal token belongs to you and stops working when your membership does, which is correct on a workstation and fatal in a pipeline. For CI, mint a **service account key** in the web app (Settings → API tokens → Service accounts), scope it to the permissions the job needs, and pass it as `EVALSHIFT_TOKEN` from an encrypted secret instead of running `login` on the runner. Service accounts are org-owned, so the key survives the person who created it leaving.
- Projects are `org/project` slugs — from `--project`, or the `project:` key in config. Missing projects are auto-created when permissions allow (`--no-create-project` disables; project-scoped tokens can't auto-create).
- `evalshift bundle <run-id>` builds the upload artefact without uploading; `push --bundle <path>` uploads a prebuilt one. `run_bundle.json.gz` carries the manifest, per-example rows (inputs, both outputs, per-evaluator scores and the cost/latency deltas), each example's `traces` — the replay's own tool-call traces, one stream per model side with the ordered tool calls (names, arguments, call ids), round markers, any final text and refusal messages (no `model_call` events and no tool results; a stream over 256 KB keeps its leading events and is flagged `truncated`; imported agent traces from `traces import` stay local and are not uploaded) — the aggregate, `analysis`, the policy `decision` (whose `policy` field is the resolved `migration_policy` this run's verdict was computed under, or `null` when none is configured — see [Migration policy and CI gating](#migration-policy-and-ci-gating)), a run-level `economics` rollup (per-role calls, tokens, cost, latency), `methodology_notes`, the [insights](#run-insights) narrative, the evaluator config and the dataset snapshot. **`report.html` is not uploaded** — it is still written to the run directory for local viewing, and the hosted app renders the run from the data instead. Bundle bytes are deterministic: the same run always compresses identically. Pushes are idempotent on run id. On GitHub Actions, git metadata (`GITHUB_SHA`, branch refs) is baked into the bundle so the server can pair PR runs with base-branch baselines; elsewhere it comes from git, and `bundle` fails with `could not determine a valid 40-character git SHA; run inside git or set GITHUB_SHA` outside a git checkout. `push <run-id>` builds a bundle only when `run_bundle.json.gz` is missing and otherwise uploads the existing one as-is.
- `push` validates the bundle against the server's own schema **before** it opens a connection, so a stale, hand-edited or foreign bundle fails locally (`✗ bundle failed schema validation: ...`, exit 1) instead of after a full upload. A bundle at or over **50 MB** compressed prints a warning naming the server's **100 MB** hard limit and uploads anyway — the hard limit is configurable server-side, so the CLI quotes it rather than enforcing a stale copy.
- Two more notices, both before `push` reports success. A bundle with no `decision.policy` — no `migration_policy` configured — still uploads and renders like any gated run, but the hosted gate then has nothing of this run's own to check: unless the project still has an old web-app policy for the server to fall back on, it reports `inconclusive` and never blocks the pull request (unless the GitHub Action runs with `require-policy: true`, which fails the job for such a run). `push` says so first, before the network is touched — and so before it can tell the two apart, hence the hedge: `! this run carries no migration policy; unless this project still has an old web-app policy, the hosted gate reports inconclusive and never blocks — add migration_policy to evalshift.yaml`. And once the server's initiate response comes back — before the bundle is uploaded — if the project's only policy lives in the web app and `evalshift.yaml` has no `migration_policy` of its own, `push` prints that policy back as a ready-to-paste block: `! this project has a policy configured in the web app; move it into evalshift.yaml:` followed by a `migration_policy:` YAML block holding only the keys the web app actually set. See [docs/hosted.md](docs/hosted.md#bundle-and-push) for the full example.

### What uploads and what stays local

The full field-by-field data contract lives in [docs/hosted.md — Privacy model](docs/hosted.md#privacy-model--exactly-what-uploads); this is the summary. The CLI has **no telemetry** — no analytics, no crash reporting. Its only network traffic is (1) your configured model providers, with your own keys, during `run`/`evaluate`/`report`, and (2) the hosted API on `login`, `whoami`, and `push` — plus, with `captures.store` configured, reads from your own capture bucket.

**A push uploads**, inside `run_bundle.json.gz`: the manifest (run id, `org/project` slug, model ids, suite name, git SHA/branch/PR number, the local suite file path string, content hashes, timestamp, CLI version); per-example rows — the example's template `inputs` and `expected` output **verbatim**, both models' **full output text**, the replay's tool-call traces (tool names, arguments and call ids, round markers, final text and refusal messages, capped at 256 KB per side; imported agent traces are not uploaded), per-evaluator scores, error strings and one-line explanations, for each `trace_invariants` evaluator the rules each side broke (rule id, type, tool, round, detail; also for entries checked on imported traces, whose timelines otherwise stay local), per-side cost and latency, tags; aggregate/analysis/decision/economics (numbers, not content); methodology notes; the insights narrative (prose that can quote the regressions it summarizes); the evaluator config with every prompt body replaced by a `content_hash` (prompt names, file paths and variable names do ship, and so does each `llm_judge` `criterion_prompt`); and a dataset snapshot holding only metadata plus an `examples_hash`. Request metadata beside the bundle: the bearer token as an auth header to the configured host only, and the compressed size.

**Never uploads**: provider API keys, the hosted token (never inside a bundle), prompt bodies and system prompts, suite conversation histories, tool definitions/schemas, `raw.jsonl`, imported agent traces (`traces.jsonl`; only the verdicts computed on them upload — `agent_trace` scores and `trace_invariants` rule results — never the timelines; see above), the response cache, `.evalshift/captures/`, `state.json`, `report.json`, `report.html`.

**Still your responsibility**: `inputs`, `expected`, outputs and traces upload verbatim, so whatever customer data or secrets your suite or your models put in them uploads too. Redact at capture time (SDK redaction boundary) and inspect the exact bytes first: `evalshift bundle <run-id>`, then `gunzip -c .evalshift/runs/<run-id>/run_bundle.json.gz | jq .` — `push --bundle` uploads exactly the file you inspected.

### Plan limits

Local runs are always unlimited — plan limits apply only to what you push. Every new hosted account gets a 30-day Pro trial (no card) that covers every org it creates; after that each org needs its own Pro subscription. When a push exceeds your org's plan (monthly runs, seats, concurrent runs), the org's trial ended without a subscription, or the subscription has stopped paying, the server answers `402` and the CLI prints exactly what it said:

```
✗ EvalShift: this run needs a paid plan.
  The free trial for this organization ended on 2026-11-02. Subscribe to Pro to push runs and keep the CI gate — existing runs stay readable.
  Upgrade: https://app.evalshift.dev/app/acme/settings/billing
```

Exit code is 1 and nothing is uploaded. The CLI never decides entitlements itself and never retries a payment error — retrying would not change the answer. Your run and its `report.html` are already on disk under `.evalshift/runs/`; push it again once the org is subscribed (or, for the monthly quota, after the reset). An expired org stays readable — its runs, diffs and baselines still load. Transient upload failures (429, 5xx) *are* retried with backoff, which is a different thing entirely.

---

## GitHub Action

`evalshift init --ci` scaffolds `.github/workflows/evalshift.yml` — a production-shaped, self-documenting workflow (the setup checklist lives in its header comment) with three jobs:

- `discover` lists committed suites under `.evalshift/suites/*/golden.jsonl` — a suite added by `capture sync` is evaluated on the next run with no workflow edit, and a project with no suites yet skips green. Suites must be committed for CI to see them: keep `.evalshift/*` ignored but un-ignore `.evalshift/suites/` and `.evalshift/toolsets/`.
- `eval <suite>` is a matrix job per suite (the action evaluates one suite per invocation) with `fail-on: policy` and `evalshift-version` pinned to the CLI that scaffolded the project. Each suite is selected with `suite-name:` — see [Selecting a suite: name, not path](#selecting-a-suite-name-not-path). `max-parallel` defaults to 1; raise it toward the hosted plan's concurrent-run limit (Pro: 5; Enterprise: by contract). Only the first matrix job posts the PR comment — the comment marker is a constant, so multiple suites would overwrite one another.
- `evalshift gate` is the single check to require in branch protection: it fails if any suite failed and passes when evaluation was skipped (fork PR, no suites, or `EVALSHIFT_TOKEN` not yet set). Don't require the per-suite jobs (dynamic names) or the `evalshift/regression` commit status (last writer wins across suites).

Runs on pushes to main create the base-branch baselines PRs diff against, so the workflow cancels superseded runs on PRs only, never on main.

The action runs the pipeline, pushes the candidate run, finds the latest compatible base-branch run, fetches the hosted diff, maintains a single marked PR comment, and sets the `evalshift/regression` commit status. Inputs: `token` (required), `host`, `config` (default `evalshift.yaml`), `suite-name` (a `suites:` key; preferred) **or** `suite` (a path, default `golden.jsonl`) — mutually exclusive, `fail-on` (`policy` (default — hosted migration-policy verdict, falling back to regression gating when unreachable) | `never` | `regression` | `any-slice-regression`), `evalshift-version` (exact CLI version from PyPI), `python-version` (default `3.12`), `require-policy` (default `false`; under `fail-on: policy`, `true` fails the job when the pushed run carries no migration policy — by default such a run is reported as ungated with a workflow warning and passes), `branch` / `base-branch` (candidate and baseline branch overrides, auto-detected when omitted), `create-project` (default `true`), `comment` (default `true`), `github-token` (token for PR comments and the commit status; defaults to the workflow's `github.token`), `repo-private` (defaults to the GitHub context; used for the private-repo CI entitlement check). Under `fail-on: policy`, `pass`, `conditional_pass` and `inconclusive` pass and `fail` fails. With no baseline yet, the comment notes the push; under `regression` / `any-slice-regression` gating passes, while under the default `policy` mode the job still follows the run's policy verdict.

### Selecting a suite: name, not path

The action takes either `suite-name:` (a key under `suites:` in `evalshift.yaml`) or `suite:` (a path). They load the same rows, but only the **name** resolves that suite's own `evaluators:` block — the one `capture sync` writes for a tool-calling suite (`EvalShiftConfig.evaluators_for` maps a `None` name to the top-level block). Select a wired suite by path and it is silently scored with the top-level `evaluators:` instead; if those are `semantic` + `llm_judge` and the suite's rows are tool calls, nothing scores and the run fails at `analyze` with `scores.jsonl is empty`. There is no warning, because a bare path is a legitimate way to run a suite that has no entry under `suites:`.

A suite wired under `suites:` is therefore selected by name — which is what `init --ci` scaffolds (`suite-name: ${{ matrix.suite }}`, the matrix carrying directory names, which are the keys `capture sync` writes). `suite:` is for a one-off file outside the config. The name form needs an `evalshift-version` pin of `0.14.0` or newer.

### Pin drift

`evalshift.yaml` is `extra="forbid"` everywhere, so the CLI that *reads* the config in CI must be at least as new as the CLI that *wrote* it locally — a newer `capture sync` or `init` can add keys an older release rejects outright. The action installs an exact version (`evalshift-version`, or its own default when the input is absent), which is where drift creeps in: you upgrade locally, re-sync, and CI still installs last month's release.

The CLI checks for this wherever it writes or validates config — `capture sync`, `init` (without `--ci`, next to a workflow it didn't write — `init --ci` pins the scaffolding CLI itself and does not warn about the file it just wrote), `doctor` (a `ci pin` row), and `validate` — by parsing every `.github/workflows/*.yml` for `evalshift/evalshift-action` steps and comparing their `evalshift-version` with its own:

- **stale** — a literal pin is older than the local CLI. Fix: set `evalshift-version: "<local version>"` on the step.
- **unpinned** — a step has no `evalshift-version`, so the action default applies and may lag. Fix: add the pin.
- **ahead** — every pin is newer than the local CLI. Fix: `pip install -U evalshift`.

Equal pins, `${{ }}` expressions, unparseable versions, and an editable install without metadata (`0.0.0+unknown`) are silent. The check is advisory: it never edits a workflow and never changes an exit code, and in CI it is a no-op by construction (the running CLI *is* the pin). Config `version: 1` is not bumped for additive fields, nor for a removal that fails the load with a message naming the key — see [Configuration](docs/configuration.md#config-version-policy).

Secrets needed: a provider API key matching your config's models, and `EVALSHIFT_TOKEN` — a service account key from Settings → API tokens → Service accounts, scoped to `run:create` + `run:read` + `policy:read`, stored as an encrypted repository or environment secret. Not a personal token, never a literal in the workflow YAML, and never reachable from `pull_request_target`. Rotate by minting the successor first (24h grace), updating the secret, confirming a green run, then letting the old key expire. One thing a scoped key can't do, by design: auto-create the project (`project:create` is owner-only — pre-create it and set `create-project: false`). Full guidance: the action's [README](https://github.com/evalshift/evalshift-action#readme).

---

## Command reference

Common conventions: `-c/--config` defaults to `./evalshift.yaml`; run artefacts live under `.evalshift/runs/` (hidden `--runs-base` override on run-artefact commands); exit code 1 on handled errors.

### Pipeline

**`evalshift init`** — scaffold a minimal capture-first `evalshift.yaml`.
`-f/--force` · `-d/--directory <dir>` · `--ci` · `--wire-agents/--no-wire-agents` (default on) · `--provider gemini|openai|anthropic|deepseek` · `--profile model-upgrade|cost-reduction|local-model|quantization|provider-switch` (default `model-upgrade`)
Without `--ci`, warns after writing when an existing workflow under `.github/workflows/` pins an older or newer CLI than this one, or none at all (see [Pin drift](#pin-drift)); `init --ci` writes the pin itself and does not warn about the file it just wrote.

**`evalshift doctor`** — environment/config check. Exit 1 only on an invalid existing config or a `captures.store` whose client extra is missing. Row 2, `evalshift-sdk`, confirms `import evalshift` is the SDK (`warn` when missing or shadowed, never a failure). Reports the toolset each configured suite carries and flags a suite whose examples carry more than one distinct toolset. The suite-side checks cover every suite in the config's `suites:` block, falling back to `./golden.jsonl` when none are wired. Adds a `ci pin` row when a workflow uses the GitHub Action (`warn` on pin drift, never a failure) and a `judge family` row when an `llm_judge` judge shares a provider with a configured arm (`warn`, never a failure). Adds a `captures.store` row when the config names a store (`fail` on a missing extra, `warn` when the bucket cannot be listed).

**`evalshift run`** — paired evaluation run (costs money — calls real models).
`-f/--from <model>` · `-t/--to <model>` · `-c/--config` · `-s/--suite <file>` · `--suite-name <name>` · `--resume` · `-y/--yes`

**`evalshift evaluate <run-id>`** — score all pairs → `scores.jsonl`. `-c/--config`
Prints a red doctor-style **broken eval harness** row when the *source* model failed the recorded ground truth on at least half of at least four `tool_selection.conformance` rows — the suite's expectations were captured from the source model, so the source is the one side that should satisfy them, and when it does not the run measured the harness rather than the migration. It names the rate (`10 of 10 … (100%)`) and the likely causes. `evalshift compare` prints the same row immediately above the verdict block. Below four conformance rows the check stays silent: a 100% rate over three has a 95% Wilson lower bound of 0.44 and cannot support the claim.

**`evalshift analyze <run-id>`** — paired stats → `analysis.json` (+ `migration_decision.json`).
`-c/--config` · `--gate <severities>` · `--policy-gate`

**`evalshift report <run-id>`** — render `report.html` + `report.json`. `-c/--config` · `--open` · `--insights/--no-insights` (default on; one extra LLM call — see [Run insights](#run-insights))

**`evalshift compare`** — full pipeline under one live display: two models, **one** suite. It takes a single `--suite-name`/`--suite`, auto-selects when exactly one suite is wired, and prints a ready-to-run command per suite when several are. It does not run every suite — loop over the names, or use one CI job per suite. Formerly `all`; that name is hidden but still works, since scaffolded `EVALSHIFT.md` files reference it.
All `run` flags, plus `--gate` · `--policy-gate` · `--open` · `--push` · `--insights/--no-insights`

### Hosted

**`evalshift login`** — `--token es_...` · `--host <url>` · `--no-browser` · `--timeout <s>` (default 900)
**`evalshift logout`** — remove stored credentials.
**`evalshift whoami`** — `--host` · `--token`
**`evalshift bundle <run-id>`** — `-c/--config` · `-s/--suite` · `--suite-name` · `-o/--output` · `--project org/project`
**`evalshift push [run-id]`** — `--bundle <path>` · `--project` · `--host` · `--token` · `--create-project/--no-create-project` (default on) · `-c/--config` · `-s/--suite` · `--suite-name`

### Captures

**`evalshift capture list [suite]`** — `--json` · `-c/--config` · `--offline`
**`evalshift capture fetch`** — `--suite` · `--since <30m|24h|7d|ISO>` · `-c/--config` (needs `captures.store`)
**`evalshift capture promote <capture-id>`** — `--as <case-id>` · `--suite` · `--input-var` (default `input`) · `--tag` (repeatable) · `--strict-args` · `--names-only` · `--tool-count` · `--rounds {first,all}` (default `first`) · `--allow-errored` · `-f/--force`
**`evalshift capture sync`** — `--suite` · `--input-var` · `--tag` · `--strict-args` · `--names-only` · `--tool-count` · `--rounds {first,all}` · `--allow-errored` · `-c/--config` · `-f/--force` · `--write/--print` (default write) · `--keep-duplicates` · `--since` · `--offline`
**`evalshift capture clean [suite]`** — `--promoted` (default) · `--all` · `-y/--yes`
**`evalshift capture diff <cap-a> <cap-b>`**

### Traces, debugging

**`evalshift traces import <run-id>`** — `--source <file>` (required) · `--target <file>` (required) · `--strict`
**`evalshift inspect <run-id>`** / **`evalshift inspect case <run-id> <example-id>`** — `--failed`
**`evalshift diff case <run-id> <example-id>`**
**`evalshift replay case <run-id> <example-id>`** — `--model source|target` (default target) · `--trace`

### Housekeeping

**`evalshift runs clean`** — `--keep <n>` · `--older-than <days>` · `--suite <slug>` · `--dry-run` · `-y/--yes` · `--config`
**`evalshift cache clear`** — wipe the response cache.

### Hidden debug commands

**`evalshift validate`** — load config + suite + prompts, cross-check compatibility. `-s/--suite` · `-c/--config`. After the success line, prints the [Pin drift](#pin-drift) warning if a workflow pins an older or newer CLI, or none at all (advisory; exit code unchanged, and a no-op in CI where the running CLI is the pin).
**`evalshift test-call`** — one live smoke-test call. `-m/--model` (required) · `-p/--prompt` · `-t/--temperature` (0–2, default 0) · `--max-tokens` (1–8192, default 256) · `--tools <file>` (prints a ToolTrace)

---

## Compatibility and stability

EvalShift follows [Semantic Versioning](https://semver.org). From **1.0.0** onward, a breaking change to anything listed as public below requires a new major version. The point of writing the boundary down is that you can check it rather than infer it.

**Public surface — a breaking change here means a major release:**

| Surface | What is covered |
|---|---|
| `evalshift.yaml` | Every documented field, its type, and its meaning. Unknown keys are rejected (`extra="forbid"`), so the schema is a contract in both directions — a typo fails loudly rather than being ignored. |
| Command names and flags | Every command in the [Command reference](#command-reference) and its options, including the severities accepted by `--gate` and the verdicts that trip `--policy-gate`. |
| Exit codes | `0` success · `1` failure, or a gate breach under `--gate` / `--policy-gate` · `2` usage error (an unknown option or value, e.g. `init --provider <unknown>`). |
| Documented artifact fields | The `report.json`, `analysis.json`, `scores.jsonl` and `raw.jsonl` keys described in this document. |
| The run bundle | Shared with the hosted server and versioned in its own right — see [Hosted EvalShift](#hosted-evalshift). |

**Not public — these may change in any release:**

- The internal layout of `.evalshift/`: the response cache database, checkpoint files, and the fields of `state.json` beyond those documented here. It is generated state the CLI owns.
- The `evalshift_cli` Python package. EvalShift is a command-line tool; importing it is not a supported interface, and the package was renamed once already (`0.14.0`).
- The HTML report's markup, styling, and internal structure. Its *content* is described here; its DOM is not.
- Console output wording, progress rendering, and log formatting.

**Config schema evolution** has its own rule, and it is deliberately not tied to the CLI's major version: `version:` in `evalshift.yaml` bumps only when a field is renamed or given a different meaning. Additive fields ride the CLI version instead, and so do removals that fail the load with a message naming the key. See [Config version policy](docs/configuration.md#config-version-policy) for what that requires of your CI pin.

**Renames keep the old name.** When a command is renamed, the previous name stays registered as a hidden alias that still works — it stops being advertised, not accepted. `evalshift all` became `evalshift compare` in 1.0.0 and `all` still runs. Removing such an alias would itself be a breaking change, so it cannot happen inside a major version.

**Deprecations** are announced in `CHANGELOG.md` and warn on stderr at the point of use, so a pipeline that parses stdout is unaffected.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Google auth (either works) |
| `OPENAI_API_KEY` | — | OpenAI auth (also the default semantic embedding model) |
| `ANTHROPIC_API_KEY` | — | Anthropic auth |
| `DEEPSEEK_API_KEY` | — | DeepSeek auth |
| `EVALSHIFT_NONINTERACTIVE` | unset | Non-empty → skip the cost-confirmation prompt (implied `--yes`); set in scaffolded CI |
| `EVALSHIFT_MAX_RUNS` | unset | Override `retention.max_runs_per_suite`; `0`/`none`/`unlimited`/`off` disables count pruning |
| `EVALSHIFT_DIR` | `.evalshift` | Base dir for `<base>/captures` (what the `capture` commands read), `<base>/suites` (where promotion writes) and `<base>/toolsets` (toolset sidecars); `run` looks there first when resolving a `toolset_ref`, then beside the suite file. Does not move `.evalshift/runs/` or the response cache |
| `EVALSHIFT_HOST` | `https://api.evalshift.dev` | Hosted API base URL |
| `EVALSHIFT_TOKEN` | unset | Hosted token (beats the credentials file, loses to `--token`) |
| `EVALSHIFT_CREDENTIALS_PATH` | `~/.evalshift/credentials` | Credentials file override |
| `GITHUB_STEP_SUMMARY` | — | When set, `analyze` appends a markdown results table |

Keys are consumed by LiteLLM at call time; EvalShift itself never stores or transmits them.

---

## Troubleshooting / FAQ

### Will `run` cost me money?

Yes — every `run` calls a real model. Before dispatch you get a worst-case cost estimate (assumes every completion hits the registry `default_max_tokens`, 4096 — actual cost is usually much lower); above $10 it asks for confirmation. The cache makes repeat runs of unchanged calls free, tool-calling examples included. Cheapest iteration loop: small suite first, cache on.

### A model call failed mid-run

The error is recorded on that call in `raw.jsonl`; the run completes. At evaluate time the affected pair gets an errored row (a 0.5/0.5 placeholder with the error attached) that is excluded from the statistics, so it can't masquerade as a regression or an improvement. Re-running the same command re-uses cached successes (for a multi-round tool example, the rounds before the failed one) and retries the failures (errored calls in a *resumed* run are not retried — start a fresh run to retry them).

### `--resume` aborts with a config-hash mismatch

Resume requires the config and the suite path to match the original run — a changed config would corrupt the pairing. Start a fresh run. Upgrading the CLI counts as a config change when the release added config fields (they join the hashed canonical dump), so an older CLI's in-progress run cannot be resumed after upgrading. The suite's *contents* are not part of the hash, so an edited suite at the same path resumes without this error; start a fresh run after editing examples.

### Everything comes back severity `none`

Usually correct behaviour: no statistically significant difference. Check n (n < 5 → `insufficient`; n < 20 → noted as uncertain), and remember BH correction across many comparisons raises the significance bar. Zero-variance comparisons (every delta identical) are skipped as `none` by design.

### The policy verdict is `inconclusive`

Three common causes, all by design: (1) every configured evaluator is advisory (`blocking: false` — the fresh `init` state) so nothing gates quality — set `blocking: true` on at least one trusted evaluator; (2) a rate budget was breached but the Wilson CI can't confirm it at this suite size — grow the suite; (3) all comparisons were `insufficient` (n < 5). Note that case (1) still reads `fail` when the cost or latency budget is breached — those are computed from the run's calls and hold with no blocking evaluator at all — and that a breached `max_invariant_violations` is the exception to all three: a count, conclusive however small the sample, so a blocking trace rule the target broke reads `fail`, never `inconclusive`. `analyze` and `compare` print the specific reason and the recommended fix under the verdict line (also in `migration_decision.json` as `reason` / `recommendations`).

### Does EvalShift evaluate multi-turn conversations?

Yes — one example per turn with a recorded `history` prefix, replayed teacher-forced. See [Multi-turn conversations](#multi-turn-conversations). Full-conversation re-driving is deliberately not supported.

### Which models can I use?

Anything LiteLLM supports. The built-in registry only provides aliases and metadata; unknown ids pass through with provider inferred from the id prefix. Verify a model with `evalshift test-call -m <id>`.

### Does EvalShift work with DeepSeek?

Yes. Export `DEEPSEEK_API_KEY` and use DeepSeek's API ids, `deepseek-flash`
or `deepseek-v4-pro`. A bare `deepseek-*` id (what a capture records when your
app calls `api.deepseek.com` through the OpenAI client) gets the `deepseek/`
prefix automatically. `evalshift init --provider deepseek` scaffolds a
DeepSeek project. Three things differ from other providers:

- **Sampling is not controlled.** Both models run in thinking mode by
  default, which accepts `temperature` and ignores it. EvalShift keeps thinking
  on, because that is what your application runs, so DeepSeek arms are marked
  non-deterministic in the report. A DeepSeek judge is marked
  non-deterministic too. Raise `defaults.samples_per_example` when the verdict
  matters.
- **Replayed assistant turns carry an empty reasoning chain.** Every assistant
  turn replayed from the recording, tool rounds and chat history alike, is
  sent with the single-space `reasoning_content` placeholder the API accepts.
  The recording holds no DeepSeek reasoning to pass back. DeepSeek requires
  the field on any request with tools, where an empty chain may degrade
  multi-turn answer quality, and ignores it otherwise.
- **No embeddings.** DeepSeek has no embedding endpoint. The `semantic`
  evaluator needs an OpenAI or Gemini embedding model and its key, which is
  why the DeepSeek scaffold ships it commented out.

DeepSeek served by another host (self-hosted open weights, or a cloud region
of your choice) goes through that host's LiteLLM prefix (`hosted_vllm/`,
`azure_ai/`, `bedrock/`, ...) and its environment variables. Tool calls parse
the same way, but the key pre-check and the notes above apply to the
`deepseek/` API only. LiteLLM also reads `DEEPSEEK_API_BASE` to point the
`deepseek/` provider at a DeepSeek-compatible endpoint. A local Ollama model
named like `deepseek-r1` needs its prefix, `ollama/deepseek-r1`, when you name
it as a run arm; a capture that recorded the bare name is treated as the
DeepSeek API, and its estimated capture cost uses DeepSeek's API price.

### Do I need LangChain / a specific framework?

No. EvalShift compares model behaviour, not framework code: prompts come from config or AST-parsed source, each example's toolset from a capture-derived sidecar or an inline `tools:` list, ground truth from captures. Framework-side timelines can be scored via [external traces](#external-agent-traces).

### Where does my data go?

Nowhere, by default. Model inputs/outputs go to the providers you configured (that's the point); everything else stays under `.evalshift/` and `~/.evalshift/`. The hosted service only sees what `push` explicitly uploads. One thing worth knowing: [run insights](#run-insights) sends the worst regressions' inputs and outputs to `defaults.insights_model` — the same exposure as an `llm_judge` criterion. `--no-insights` turns it off. With `captures.store` configured, `capture sync`, `list` and `fetch` read from the bucket *you* named using *your* cloud credentials; nothing is sent to EvalShift, and `doctor` reports whether that store is reachable.

---

## Further reading

- [docs/](docs/) — the mkdocs site: [getting-started](docs/getting-started.md), [configuration](docs/configuration.md), [evaluators](docs/evaluators.md), [methodology](docs/methodology.md), [agents](docs/agents.md), [conversations](docs/conversations.md), [traces](docs/traces.md), [sdk](docs/sdk.md), [hosted](docs/hosted.md), [github-action](docs/github-action.md), [faq](docs/faq.md)
- [AGENTS.md](AGENTS.md) — repo orientation for AI coding agents; [CLAUDE.md](CLAUDE.md) — contributor workflow rules
- [CHANGELOG.md](CHANGELOG.md) — release history
- [evalshift-sdk](https://github.com/evalshift/evalshift-sdk) — the in-process capture SDK
- [llms-full.txt](llms-full.txt) — dense single-file reference for AI coding tools, hosted at <https://www.evalshift.dev/cli-llms-full.txt>
- License: Apache-2.0
