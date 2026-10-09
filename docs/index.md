# EvalShift

Open-source LLM migration and regression testing for AI agents.

```
evalshift-sdk    captures real agent behavior in production
      ↓
evalshift CLI    replays it against a candidate model — scores, stats, report
      ↓
hosted (opt-in)  run history, diffs, PR gates
```

Compare an existing model against a replacement before you migrate.
Replay golden suites, evaluate semantic and structural behavior,
detect tool-call regressions, and gate model changes in CI.

Run your prompts on two LLMs and find out, with statistical confidence,
what regressed.

EvalShift is a local-first CLI that helps engineering teams migrate
safely between LLM versions. Point it at your prompts and a golden
suite of inputs; it runs both models, scores the outputs with
structural / semantic / LLM-as-judge / tool-call evaluators, and
produces a single-file HTML report with **defensible statistics**:
paired tests, Cohen's d, 95% CIs, and Benjamini-Hochberg correction
across every comparison.

Hosted commands are available when you explicitly log in
and push a run. Local runs remain local by default.

## The four pieces

| Piece | What it is | Docs |
| --- | --- | --- |
| **SDK** (`evalshift-sdk`) | Records what your agent actually did in production — model, tool, and final-output calls — to `.evalshift/captures/`. The CLI promotes those into golden suites. | [Capture SDK](sdk.md) |
| **CLI** (`evalshift`) | Runs the suite on two models, scores, analyses, reports. | this site |
| **GitHub Action** | Runs the pipeline on pull requests, comments, gates the check. | [GitHub Action](github-action.md) |
| **Hosted server** | Optional. Stores pushed runs, diffs branches, drives PR comments. | [Hosted EvalShift](hosted.md) |

Data flow: SDK captures → CLI runs and bundles → hosted server stores and diffs
→ web app displays.

## The pipeline

The recommended path is capture-first:

```
evalshift init                        # scaffold a capture-first evalshift.yaml
# instrument your agent with evalshift-sdk, run it with EVALSHIFT_CAPTURE=1
evalshift capture sync                # promote captures into a golden suite
evalshift compare --suite-name <suite> --to <candidate-model>
                                      # doctor → run → evaluate → analyze → report
```

Or drive each stage by hand:

```
evalshift doctor      # verify env + config
evalshift run         # call both models on every example
evalshift evaluate    # score every (source, target) pair
evalshift analyze     # paired tests + BH correction
evalshift report      # single-file HTML report
```

Each stage writes its artefact under `.evalshift/runs/<run-id>/`:

| Stage      | Artefact         |
| ---------- | ---------------- |
| `run`      | `raw.jsonl`      |
| `evaluate` | `scores.jsonl`   |
| `analyze`  | `analysis.json` (+ `migration_decision.json` with a `migration_policy`) |
| `report`   | `report.html` + `report.json` (+ `insights.json`) |

Optional steps add their own: `traces import` writes `traces.jsonl`,
`bundle`/`push` write `run_bundle.json.gz`, and `push` keeps a
`push_state.json` only while an upload is in flight.

Hosted commands add optional sharing and CI workflows:

```
evalshift login       # authenticate the CLI for hosted upload
evalshift whoami      # show hosted identity and org roles
evalshift bundle      # package a completed local run
evalshift push        # upload a bundle to hosted EvalShift
evalshift compare --suite-name <suite> --push  # run locally, then push
```

Housekeeping:

```
evalshift inspect <run-id> [--failed]   # per-row deltas (--failed: negative or errored only)
evalshift bundle <run-id> -o out.json.gz   # write the bundle somewhere else
evalshift cache clear                   # wipe the local response cache
evalshift runs clean                    # prune old runs (--keep, --older-than, --dry-run)
```

## Local-first by design

For local commands, your prompts and your suite never leave your machine.
The only outbound calls are to the LLM providers you configure — any provider
LiteLLM supports — using your own API keys — and, if you set `captures.store`,
reads from your own capture bucket with your own cloud credentials. `bundle` packages artifacts locally;
hosted upload happens only when you run `push` or `compare --push`.

## For AI coding agents

Each piece publishes a dense, single-file, machine-readable reference:

* CLI — <https://www.evalshift.dev/cli-llms-full.txt>
* SDK — <https://www.evalshift.dev/sdk-llms-full.txt>
* GitHub Action (CI) — <https://www.evalshift.dev/ci-llms-full.txt>

`evalshift init` wires all three into your project's agent files (`AGENTS.md`,
`CLAUDE.md`, `GEMINI.md`, `.cursorrules`,
`.github/copilot-instructions.md`) via a generated `EVALSHIFT.md`, creating
`AGENTS.md` if none of those files exist; disable with
`--no-wire-agents`.

## Where to next

* [Getting started](getting-started.md) — a 60-second walkthrough.
* [Configuration](configuration.md) — every `evalshift.yaml` field.
* [Evaluators](evaluators.md) — when to use each evaluator family.
* [Agent migrations](agents.md) — tool-call evaluation and suite ground truth.
* [Capture SDK](sdk.md) — instrument your agent, promote captures to suites.
* [Multi-turn conversations](conversations.md) — teacher-forced replay.
* [Agent traces](traces.md) — bring-your-own agent timelines.
* [Methodology](methodology.md) — the statistical machinery.
* [Hosted EvalShift](hosted.md) — login, bundle, push, privacy.
* [GitHub Action](github-action.md) — PR comments and hosted gates.
* [FAQ](faq.md) — common questions.
