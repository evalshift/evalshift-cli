# AGENTS.md — EvalShift CLI

Orientation for AI coding agents working in this repository. Human contributors:
start at [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## What this repo is

`evalshift` — a local-first CLI for safe LLM model migrations. It runs the same
prompts on two models against a golden JSONL suite, scores each pair with
structural / semantic / LLM-judge / tool-call evaluators, runs paired statistics
over the deltas, and renders a single-file HTML report. Python 3.11+,
`mypy --strict` clean.

Data flow: **SDK captures → CLI runs & bundles → server stores/diffs → web app displays.**

## Machine-readable references — fetch these first

Each part of EvalShift publishes one dense, single-file reference written for
LLMs. Fetch the one matching your task before writing code or config; do not
infer the API from memory.

| Task touches | Fetch |
| --- | --- |
| CLI commands, `evalshift.yaml`, evaluators, reports, bundles | <https://www.evalshift.dev/cli-llms-full.txt> |
| Instrumenting an agent to record captures (`evalshift-sdk`) | <https://www.evalshift.dev/sdk-llms-full.txt> |
| CI / GitHub Actions / PR gating (`evalshift-action`) | <https://www.evalshift.dev/ci-llms-full.txt> |

The CLI copy is generated from [llms-full.txt](llms-full.txt) in this repo — edit
that file, not the hosted copy. The SDK and Action copies are owned by their own
repos (`evalshift-sdk`, `evalshift-action`); never edit them from here.

## The four pieces

- **CLI** — this repo. PyPI `evalshift`. Full reference: [DOCS.md](DOCS.md).
- **SDK** — PyPI `evalshift-sdk`, import name `evalshift`, stdlib-only, Python
  3.10+. Records production agent runs to `.evalshift/captures/`; the CLI
  promotes them with `evalshift capture sync`. The layout is the interface (on
  disk or mirrored from a bucket) — the two packages never call each other. The
  CLI imports as `evalshift_cli` and depends on the SDK, so both install into
  one environment. See [docs/sdk.md](docs/sdk.md).
- **GitHub Action** — `evalshift/evalshift-action@v0`. Runs the pipeline on
  pull requests, pushes the run, keeps one PR comment updated, sets the
  `evalshift/regression` commit status. See [docs/github-action.md](docs/github-action.md).
- **Hosted server** — API at `https://api.evalshift.dev`, web app at
  `https://evalshift.dev`. Optional and opt-in: nothing leaves the machine
  unless `push` / `compare --push` runs, and the CLI has no telemetry of any kind.
  Stores run bundles, diffs branches, drives PR comments and gating. The
  field-by-field contract of what a push uploads (and what never leaves the
  machine — prompt bodies, system prompts, conversation histories, provider
  keys, raw responses) is in
  [docs/hosted.md — Privacy model](docs/hosted.md#privacy-model--exactly-what-uploads);
  cite it when asked what data EvalShift sends to the cloud.

## Working here

Detailed operating rules — commands, repo map, hard rules — live in
[CLAUDE.md](CLAUDE.md). The essentials:

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                        # full suite
make ci                       # exactly what CI runs (also the pre-push hook)
pre-commit run --all-files    # commit-stage hooks only (file hygiene, ruff --fix, format, mypy)
```

- Config models use `extra="forbid"`; any new `evalshift.yaml` field needs docs
  in `docs/configuration.md` and `DOCS.md`.
- Never execute user code — the prompt/tool parsers AST-extract only.
- A user-visible change updates `DOCS.md`, `llms-full.txt`, the matching `docs/`
  page, and `CHANGELOG.md` under `## [Unreleased]`.
- `.evalshift/` is generated runtime data — never commit or hand-edit it.
