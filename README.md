# EvalShift

Open-source LLM migration and regression testing for AI agents.

[![CI](https://github.com/evalshift/evalshift-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/evalshift/evalshift-cli/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/evalshift.svg)](https://pypi.org/project/evalshift/)

**The SDK captures what your agent really did. The CLI replays it against a
candidate model and tells you what broke. The hosted app keeps the history.**

```
evalshift-sdk    captures real agent behavior in production
      ↓
evalshift CLI    replays it against a candidate model — scores, stats, report
      ↓
hosted (opt-in)  run history, diffs, PR gates
```

Migrating between LLM versions (say `gemini-2.5-flash` →
`gemini-3.1-flash-lite-preview`) means guessing which behaviors changed.
EvalShift removes the guess. It runs both models over the same golden suite,
scores the outputs with structural / semantic / LLM-as-judge / tool-call
evaluators, and produces a single-file HTML report with **defensible
statistics**: paired tests, Cohen's d, 95% CIs, and Benjamini-Hochberg
correction across every (prompt x evaluator x slice) comparison.

An eval is only worth the examples in it. That is why the
[capture SDK](https://github.com/evalshift/evalshift-sdk) is part of the
product rather than an add-on: it records real production runs — model calls,
tool calls, final outputs — to disk, and `evalshift capture sync` promotes them
into golden suites. Hand-written suites are fully supported too, but captured
traffic is the recommended starting point.

Local runs stay on your machine by default. Hosted commands are available when
you explicitly log in and push a run.

## How EvalShift fits together

Four pieces, released and documented independently:

| Piece | What it does for you | Reference |
| --- | --- | --- |
| **SDK** — PyPI `evalshift-sdk` | Records what your agent actually did in production — model calls, tool calls, final output — as capture files on disk. Those captures become your golden suite. | [docs/sdk.md](docs/sdk.md) |
| **CLI** — this repo, PyPI `evalshift` | Replays the suite on two models, scores, analyses, reports, bundles, pushes. | [DOCS.md](DOCS.md) |
| **GitHub Action** — `evalshift/evalshift-action@v0` | Runs the pipeline on pull requests, pushes the run, posts one PR comment, sets the `evalshift/regression` status. | [docs/github-action.md](docs/github-action.md) |
| **Hosted server** — `api.evalshift.dev`, web app at `evalshift.dev` | Optional. Stores pushed run bundles, diffs them across branches, drives PR comments and gating. | [docs/hosted.md](docs/hosted.md) |

The SDK and the CLI never call each other — the interface is files under
`.evalshift/captures/`, so either works without the other. The CLI (import
package `evalshift_cli`) depends on the SDK (import name `evalshift`), so one
environment holds both: `pip install evalshift` brings the SDK with it, and a
production agent that only records captures installs `evalshift-sdk` alone.

## For AI coding agents

Point your coding agent at the dense, single-file reference for the piece it is
working on:

- EvalShift CLI: <https://www.evalshift.dev/cli-llms-full.txt>
  (source of truth: [llms-full.txt](llms-full.txt) in this repo)
- EvalShift SDK: <https://www.evalshift.dev/sdk-llms-full.txt>
- EvalShift GitHub Action (CI): <https://www.evalshift.dev/ci-llms-full.txt>

## Status

**Stable and in production use.** Every command in the pipeline is shipped and
CI enforces a 90% coverage floor on the source. The CLI is published on PyPI as
`evalshift`, the capture SDK as `evalshift-sdk`, and the hosted service runs at
`api.evalshift.dev`.

The `evalshift.yaml` schema is versioned: `version: 1` changes only when a
field is renamed or given new semantics. A removed field does not bump it: a
config that still sets one fails to load with an error naming the key, so no
config is ever silently misread across releases. See
[Config version policy](docs/configuration.md#config-version-policy).

## Install

Requires Python 3.11+.

```bash
# Recommended
uv pip install evalshift     # or: pip install evalshift
```

That also installs the capture SDK that feeds the CLI its suites. A production
agent that only records captures needs just the SDK (stdlib-only):

```bash
uv pip install evalshift-sdk     # or: pip install evalshift-sdk
```

From source (for contributors):

```bash
git clone https://github.com/evalshift/evalshift-cli.git
cd evalshift-cli
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick start

### Use it on your agent

This is the workflow: record what your agent really does, then hold a
candidate model to it.

```bash
evalshift init                    # minimal capture-first evalshift.yaml
```

Instrument the agent with [evalshift-sdk](https://github.com/evalshift/evalshift-sdk)
— stdlib-only, Python 3.10+, installed with the CLI or on its own:

```python
from evalshift import capture


@capture.tool(name="issue_refund")
def issue_refund(order_id: str) -> dict: ...


@capture.agent(suite="support_agent", redact=True, tools=[])
def handle(message: str) -> str: ...
```

`redact=` is required on every entry point that opens a capture session (SDK
0.3.0+) — `@capture.agent`, `capture.agent_session`,
`capture.agent_session_async`, `EvalShiftCallbackHandler`: `True` masks emails,
API keys and bearer tokens before anything reaches disk, `False` records
verbatim, or pass your own `(value) -> value` callable. `@capture.tool` takes no
`redact` of its own — tool spans are masked by the redactor of the agent session
they run inside. `tools=` is required at the same entry points: the toolset the
agent was offered, or `[]` if it never calls tools.

Calling OpenAI, Anthropic or Google GenAI directly? Wrap the client once —
`wrap_openai(OpenAI())`, `wrap_anthropic(...)`, `wrap_genai(...)` (SDK 0.4.0+) —
and every model call is recorded with no per-call code: the tools offered, the
calls the model requested, usage and latency.

Nothing is recorded unless `EVALSHIFT_CAPTURE=1` is set, so the decorators are
safe to leave in production permanently:

```bash
EVALSHIFT_CAPTURE=1 python your_agent.py   # writes .evalshift/captures/
evalshift capture sync                     # captures → golden suites + wired config
evalshift compare --suite-name support_agent --to <candidate-model>
```

See [docs/sdk.md](docs/sdk.md) for the full capture contract, and
[`examples/capture-first/`](examples/capture-first/) for those three commands
checked in end to end — the instrumented agent, the captures it wrote, the
promoted suite, and the `suites:` block `capture sync` filled in. Can't
instrument the agent? A hand-written `golden.jsonl` works just as well — see
[Getting started](docs/getting-started.md).

### Driving the pipeline

`evalshift compare` drives the full five-stage pipeline — two models on one
suite per invocation — under a single
Rich Live region — stacked status rows, an inline progress bar for
the run stage, and a final verdict block that tells you whether the
candidate is significantly better, regressed, or showed no
significant change.

If you want to drive each stage by hand (useful in CI, or when
re-running just one stage after fixing config):

```bash
evalshift doctor
evalshift run --yes
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

Every artefact lives under `.evalshift/runs/<run-id>/` — `state.json`,
`raw.jsonl`, `scores.jsonl`, `analysis.json`, `migration_decision.json` (when
the config sets a `migration_policy`), `report.json`, `report.html`, and
`insights.json` (when insights ran). None of it leaves your machine unless you
opt in to hosted upload commands.

## Hosted EvalShift

Hosted EvalShift adds shared run history, web viewing, diffs, and GitHub PR
comments. It is optional: local CLI usage does not require an account.

```bash
# Sign in through the hosted web app, then approve CLI login in the browser.
# Defaults to https://api.evalshift.dev; pass --host to target another server.
evalshift login
evalshift whoami

# Add a hosted project to evalshift.yaml:
# project: acme/model-migration

# Run locally, then package and push the result.
evalshift compare --suite-name support_agent --yes --push
```

You can also drive the hosted steps manually:

```bash
evalshift bundle <run-id>
evalshift push <run-id>
evalshift push --bundle .evalshift/runs/<run-id>/run_bundle.json.gz
```

Credential precedence is explicit CLI flags, then `EVALSHIFT_HOST` /
`EVALSHIFT_TOKEN`, then `~/.evalshift/credentials`.

### What gets uploaded

Nothing, until you run `push` (or `compare --push`) — and the CLI itself has no
telemetry, analytics, or crash reporting. A push uploads one file,
`run_bundle.json.gz`, whose full field-by-field contract is documented in
[docs/hosted.md — Privacy model](docs/hosted.md#privacy-model--exactly-what-uploads).
The short version:

* **Uploads**: the run manifest (model ids, suite name, git SHA/branch/PR
  number, content hashes, CLI version); per-example rows — the example's
  template `inputs` and `expected` output verbatim, both models' full outputs,
  tool-call traces (names and arguments), scores, cost and latency; aggregate
  statistics, the analysis, the migration decision, economics, and the
  machine-written insights narrative.
* **Never uploads**: provider API keys, prompt bodies and system prompts,
  suite conversation histories, tool definitions/schemas, `raw.jsonl`,
  imported agent traces, the response cache, captures, and `report.html`.
  Prompt and dataset content is replaced by SHA-256 hashes so diffs still
  align across runs.
* **Can still be sensitive**: inputs, expected outputs, model outputs, and
  traces carry whatever content your suite or your models put in them. Redact
  at capture time (see the SDK's redaction boundary) and inspect before
  pushing: `evalshift bundle <run-id>` writes the exact bytes a push would
  upload — `gunzip -c .evalshift/runs/<run-id>/run_bundle.json.gz | jq .`.

## GitHub Action

`evalshift init --ci` scaffolds a production-shaped workflow: it discovers
every committed suite under `.evalshift/suites/`, evaluates each on every
pull request via [`evalshift/evalshift-action@v0`](https://github.com/evalshift/evalshift-action)
(one matrix job per suite), pushes the runs to hosted EvalShift, compares
against the latest compatible base-branch run, posts one PR comment, and
gates merges on your `migration_policy` through a single required
`evalshift gate` check. The full setup checklist — secrets, committing
suites, branch protection — is documented in the generated file itself.

```bash
evalshift init --ci
```

Then add repository secrets for `EVALSHIFT_TOKEN` and the provider keys your
models use; until they exist the workflow no-ops green with a notice.

See [`docs/github-action.md`](docs/github-action.md) for the workflow's
shape, `fail-on` modes, and baseline behavior.

## Agent migrations

Migrating an agent (a prompt that uses tools)? EvalShift detects
regressions in *which* tools the new model calls, *what* arguments it
passes, and in what order and parallelism within a response. Hand-written
`trace_invariants` rules (auth before a charge, a tool that must never be
called, at most one charge, argument bounds) hold *both* models to a
contract whatever the source did, gated by
`migration_policy.max_invariant_violations`. By default
each example is one model call scored against the first recorded round;
promote with `--rounds all` to replay every recorded round teacher-forced,
with the recorded tool results fed back and each round scored on its own.
The killer scenario: a routing agent that silently stops calling
`notify_security_team` after the migration — text-only eval reports green,
EvalShift marks it CRITICAL.

Each golden-suite example carries its own toolset — recorded automatically
by `capture promote` / `capture sync` from your production captures, or
inlined by hand for a hand-authored suite.

See [`docs/agents.md`](docs/agents.md) for the full walkthrough and
the [`examples/agent/`](examples/agent/) directory for a runnable
customer-support example.

## What the report looks like

Every run writes a single-file HTML report to
`.evalshift/runs/<run-id>/report.html` — see [Use it on your
agent](#use-it-on-your-agent) above, or the runnable walkthrough in
[`examples/agent/`](examples/agent/). The report (single file, no external
assets, works offline) has:

* **Migration verdict** — the policy decision up top: which budgets failed, the
  top regression causes, and the recommendation.
* **Executive summary** — one row per prompt with a severity badge, drawn from
  blocking evaluators (advisory ones only when nothing else scored the prompt).
* **Trace rules broken** — every hand-written trace rule the target broke,
  with its owner and whether the source broke it too; advisory rows are tagged.
* **What changed, in plain language** — the verdict, the economics and the
  behavioural drift explained by `defaults.insights_model`. Every figure in it
  is copied from the computed statistics, never generated. Needs a provider
  key; skip it with `--no-insights`.
* **Per-prompt deep dive** — aggregate stats, per-slice breakdown,
  top-5 worst regressions side-by-side.
* **Methodology appendix** — every test, p-value, effect size, and
  CI is documented.

## Why local-first?

Your prompts and suite stay local for `doctor`, `run`, `evaluate`, `analyze`,
and `report`. The only outbound calls in local mode are to the LLM providers you
configure — any provider LiteLLM supports, called with your own API keys — and,
if you set `captures.store`, reads from your own capture bucket.
Anthropic, DeepSeek, Google and OpenAI ids additionally get a curated pricing and
capability entry; everything else is passed through with the provider inferred
from the id.

`bundle` packages completed local artifacts into `run_bundle.json.gz` without
uploading them. `push` and `compare --push` upload that bundle to the hosted
backend associated with your token.

## Wiring the agent references into your project

The three references are listed at the top of this README. `evalshift init`
wires these links into your project automatically: it writes
`EVALSHIFT.md` and points existing agent files (`AGENTS.md`, `CLAUDE.md`,
`GEMINI.md`, `.cursorrules`, `.github/copilot-instructions.md`) at it, creating
`AGENTS.md` if none of those files exist. Disable with `--no-wire-agents`. In
*this* repo the same three links live in
[AGENTS.md](AGENTS.md).

## Documentation

* [DOCS.md](DOCS.md) — consolidated single-file reference for everything below
* [Getting started](docs/getting-started.md) — install + first run walkthrough
* [Configuration reference](docs/configuration.md) — every `evalshift.yaml` field
* [Evaluators](docs/evaluators.md) — when to use which family
* [Agent migrations](docs/agents.md) — tool-call evaluation, per-example toolsets
* [Multi-turn conversations](docs/conversations.md) — teacher-forced replay
* [Agent traces](docs/traces.md) — bring-your-own agent timelines
* [Capture SDK](docs/sdk.md) — instrument your agent, promote captures to suites
* [Methodology](docs/methodology.md) — the statistical machinery
* [Hosted EvalShift](docs/hosted.md) — login, bundle, push, and the privacy
  model: exactly what data uploads and what never leaves your machine
* [GitHub Action](docs/github-action.md) — PR comments + hosted regression gate
* [FAQ](docs/faq.md) — common questions
* [llms-full.txt](llms-full.txt) — dense single-file reference for AI coding
  tools, hosted at <https://www.evalshift.dev/cli-llms-full.txt>

Further reading on the EvalShift blog:

* [How to test an LLM model migration before you ship it](https://www.evalshift.dev/blog/test-llm-model-migration-before-you-ship)
* [What actually breaks when you switch LLMs](https://www.evalshift.dev/blog/what-breaks-when-you-switch-llms)
* [How many eval cases do you need?](https://www.evalshift.dev/blog/how-many-eval-cases-do-you-need)
* [When to trust an LLM judge](https://www.evalshift.dev/blog/when-to-trust-an-llm-judge)

Runnable projects under [`examples/`](examples/):

| Example | Shows |
| --- | --- |
| [`capture-first/`](examples/capture-first/) | The recommended flow: an SDK-instrumented agent, its captures, and the golden suite + managed `suites:` block `evalshift capture sync` produced from them. |
| [`simple/`](examples/simple/) | The smallest hand-authored project — one `python_string` prompt, one length evaluator. |
| [`agent/`](examples/agent/) | A hand-authored agent suite: six tools, per-example `toolset_ref`, `tool_selection` scoring, slices. |
| [`agent-traces/`](examples/agent-traces/) | Bring-your-own agent timelines scored with the `agent_trace` evaluator, for agents the SDK cannot instrument. |

## Non-goals

* Hosted provider-key storage
* Multi-criterion judge in a single call
* Custom evaluator plugin system
* Comparing more than 2 models in one run
* Auto-detection of LangChain / LlamaIndex prompt patterns

## License

[Apache-2.0](LICENSE). Free for any use, commercial included — no share-back
requirement, and an explicit patent grant. The capture SDK
([`evalshift-sdk`](https://github.com/evalshift/evalshift-sdk)), the piece
you import into your own application, is MIT.

Earlier releases stay under the license they shipped with: `0.3.0` and earlier
under the MIT License, `0.4.0` through `0.14.1` under AGPL-3.0-or-later.
