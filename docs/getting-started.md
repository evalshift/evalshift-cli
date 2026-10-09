# Getting started

This page walks you from a fresh shell to a working `evalshift.yaml` in
under a minute.

## 1. Install

EvalShift is a Python 3.11+ package. We recommend [`uv`][uv] for dependency
management, but `pip` works too.

```bash
# uv (recommended)
uv pip install evalshift

# or pip
pip install evalshift
```

Verify the install:

```bash
evalshift --version
```

That also installs the capture SDK (`evalshift-sdk`, import name `evalshift`):
the CLI depends on it, so the same environment can instrument your agent. A
production agent that only records captures installs the SDK alone:

```bash
uv pip install evalshift-sdk
```

It is optional for this walkthrough — you can hand-write `golden.jsonl`
instead (see step 5) — but it is how real projects build their suite
fastest.

## 2. Set provider API keys

EvalShift calls the provider you configure — any provider LiteLLM supports — directly using your own keys.
Local runs do not send prompts or outputs to an EvalShift-operated server.
Provider responses are cached locally in `~/.evalshift/cache.db`, so re-running an
unchanged suite (agent suites included) makes no new calls.

Set whichever providers you intend to use:

```bash
export ANTHROPIC_API_KEY=<anthropic-api-key>
export OPENAI_API_KEY=<openai-api-key>
export GEMINI_API_KEY=<gemini-api-key>
export DEEPSEEK_API_KEY=<deepseek-api-key>
```

## 3. Scaffold your project

```bash
mkdir my-eval
cd my-eval
evalshift init --provider gemini   # or openai, anthropic, deepseek
```

`--provider` picks the model ids the scaffold uses. Omit it and `init` asks on
a terminal, or defaults to `gemini` when there is no terminal to ask on.

You can also pick a migration profile:

```bash
evalshift init --profile cost-reduction
```

The default `model-upgrade` profile scaffolds a `migration_policy` block
that powers the verdict in `analyze`, `compare`, and `report`.

This writes a single, minimal, capture-first `evalshift.yaml`: a
passthrough `replay` prompt, an advisory LLM-judge evaluator and — for the
Gemini and OpenAI scaffolds — an advisory semantic evaluator (the Anthropic
and DeepSeek scaffolds write it commented out, since neither provider has an
embedding endpoint), an empty managed `suites:` block for `capture sync` to fill, and the migration
policy. `init` refuses to clobber an existing `evalshift.yaml`; pass
`--force` to overwrite, or `--directory my-eval/` to scaffold into a
different folder.

## 4. Verify your environment

```bash
evalshift doctor
```

You'll see a short table:

* Green ✓ — check passes.
* Yellow ✗ — informational warning (e.g. an unset API key, or no
  `evalshift.yaml` here yet). Doctor still exits 0.
* Red ✗ — hard failure (e.g. an `evalshift.yaml` that doesn't validate;
  run `evalshift validate` to see each problem). Doctor exits 1.

The second row, `evalshift-sdk`, confirms that `import evalshift` in this
environment is the capture SDK — yellow when it is missing or shadowed by an
older CLI install.

If a workflow under `.github/workflows/` uses the GitHub Action, the table
also has a `ci pin` row — yellow when CI pins an older or newer CLI than
yours (or none at all); see [Pin drift](github-action.md#pin-drift).

If everything is green or yellow, you're ready to run.

## 5. Record what your agent actually does

`init`'s `suites:` block starts empty — `run` needs a golden suite to
dispatch against. Instrument your agent with [evalshift-sdk][sdk]
(installed in step 1):

```python
from evalshift import capture


@capture.agent(suite="support_agent", redact=True, tools=[])
def handle(message: str) -> str: ...
```

Then exercise the agent with capture turned on:

```bash
EVALSHIFT_CAPTURE=1 python your_agent.py   # writes .evalshift/captures/
```

Running on Fargate, Lambda or Kubernetes, where the disk does not outlive the task? Add
`EVALSHIFT_SINK=s3://<bucket>/<prefix>` (or `gs://…` / `az://…`) next to it and put the same
URI under `captures: {store: …}` in `evalshift.yaml`; step 6 then fetches from the bucket
first. See [`captures`](configuration.md#captures).

Captures are off unless `EVALSHIFT_CAPTURE=1` is set, so the decorator can
stay in production code. If the agent calls OpenAI, Anthropic or Google GenAI
directly, wrap the client once — `wrap_openai(OpenAI())`, `wrap_anthropic`,
`wrap_genai` (SDK 0.4.0+) — and every model call is recorded with no further
code. Full contract: [Capture SDK](sdk.md). If you can't
instrument the agent, write `golden.jsonl` by hand instead — see
[Configuration](configuration.md).

## 6. Promote captures into a suite

```bash
evalshift capture sync
```

`capture sync` promotes every recorded capture into
`.evalshift/suites/<suite>/golden.jsonl` and injects the matching
`suites:` block into `evalshift.yaml`. See
[Configuration](configuration.md) for the full capture lifecycle. If a
workflow under `.github/workflows/` pins an older or newer CLI than the one
you just synced with, or none at all, it ends with an advisory warning and the
fix — the exact `evalshift-version` line to set, or `pip install -U evalshift`
when CI is ahead — see [Pin drift](github-action.md#pin-drift).

## 7. Run the pipeline

The fast path is one command:

```bash
evalshift compare --suite-name support_agent --to <candidate-model> --yes --open
```

This runs `doctor → run → evaluate → analyze → report` under a single
Rich Live region with a progress bar for the run stage and a final
verdict block. Warnings raised along the way (LiteLLM deprecation
notices, insights retries) are held back and printed as one `⚠` section
directly under the pipeline block; errors are never deferred.
`run`/`compare` estimate worst-case cost up front and prompt for
confirmation above $10 (skip with `--yes`, or set `EVALSHIFT_NONINTERACTIVE=1`
in CI).

If you want to drive each stage by hand (useful when re-running just
one stage after fixing config, or in CI where you stage artefacts):

```bash
evalshift run --suite-name support_agent --to <candidate-model>
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

`evalshift compare` accepts every flag the underlying commands do
(`--from/--to`, `--config`, `--suite`, `--suite-name`, `--yes`, `--resume`,
`--gate`, `--policy-gate`, `--open`, `--push`).

One invocation compares two models on **one** suite. It picks the suite for
you when `evalshift.yaml` wires exactly one; with several, name it with
`--suite-name` (the error lists a ready-to-run command per suite). To cover
every suite, loop:

```bash
for s in support_agent billing_agent; do
  evalshift compare --suite-name "$s" --yes --push
done
```

## 8. Optional: import agent traces

If your own agent runtime already records source and target timelines,
attach them to a completed run before `evaluate`:

```bash
evalshift traces import <run-id> \
  --source source-traces.jsonl \
  --target target-traces.jsonl
evalshift evaluate <run-id>
evalshift analyze <run-id>
evalshift report <run-id> --open
```

Configure `evaluators.agent_trace` to compare tool order, argument
drift, extra dangerous actions, and missing verification steps, or
`evaluators.trace_invariants` with `traces: imported` to hold both sides to
rules you write. See [Agent traces](traces.md) for the JSONL schema.

## 9. Optional: push to hosted EvalShift

Hosted EvalShift adds shared run history, web viewing, diffs, and
GitHub PR comments. Sign in through the hosted web app, then approve CLI login
in the browser:

```bash
evalshift login --host <hosted-api-url>
evalshift whoami
```

Add a hosted project path to `evalshift.yaml`:

```yaml
project: acme/model-migration
```

Then push a completed run:

```bash
evalshift compare --suite-name <suite> --yes --push
```

Or package and push manually:

```bash
evalshift bundle <run-id>
evalshift push <run-id>
```

See [Hosted EvalShift](hosted.md) and [GitHub Action](github-action.md) for CI
setup and privacy details.

[uv]: https://docs.astral.sh/uv/
[sdk]: https://github.com/evalshift/evalshift-sdk
