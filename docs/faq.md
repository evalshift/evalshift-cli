# FAQ

## Does EvalShift send my prompts to EvalShift servers?

**Not during local runs.** `doctor`, `run`, `evaluate`, `analyze`, and
`report` operate locally. Every provider API call goes directly from your
machine to the LLM provider you configured — any provider LiteLLM supports —
using your own API keys.

The local SQLite cache at `~/.evalshift/cache.db` only contains
provider responses for *your* prompts and inputs.

Hosted uploads are explicit. `bundle` packages the completed
local run artifacts into `run_bundle.json.gz` without uploading them. `push`
and `compare --push` upload that bundle to the hosted backend for your project.

One local stage does call a provider with your data beyond the run itself:
`report` generates the run-insights narrative, sending the worst regressions'
inputs and outputs to `defaults.insights_model`. That is your provider, not
EvalShift's, and `--no-insights` turns it off.

## What is the summary at the top of the report, and can I trust its numbers?

It is the run-insights narrative — plain-language prose written by
`defaults.insights_model` (falling back to `judge_model`) explaining the
verdict, the advisory signal, the economics and what changed behaviourally in
the worst regressions.

The prose is machine-written; the numbers are not. Every figure is computed
from the run and handed to the model pre-rendered as a string to copy
verbatim, and any numeric token in the output that was not supplied causes the
generation to be rejected and retried. After two bad generations the CLI ships
deterministic templated prose instead (shown as model `none`). So a figure in
that block is the same figure as in the tables below it, or it is not there at
all.

It costs one model call per run, is cached in `insights.json`, and skips
itself when no API key is configured. Disable it with
`evalshift report --no-insights` or `evalshift compare --no-insights`.

## What happens if a single LLM call fails?

The orchestrator records the error in `raw.jsonl` (with `error="..."`)
and moves on. The run still completes. In the evaluation phase the pair
gets an errored row (a 0.5/0.5 placeholder with `error` set): it stays in
`scores.jsonl` for inspection and is excluded from the statistics, so a
failed call can't masquerade as a regression or an improvement.

## What models does EvalShift support?

Anything LiteLLM supports. The `evalshift_cli.models.registry` provides
friendly aliases and sane defaults for common models (Claude, DeepSeek,
Gemini, GPT), but **the registry is advisory, not gating**. A model id
that isn't in the registry — for example a fresh preview from a
vendor playground — gets passed through to LiteLLM with a
prefix-inferred provider. LiteLLM is the source of truth at call
time.

## Does EvalShift work with DeepSeek?

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

## Can I resume a run after Ctrl+C / a crash?

Yes. `evalshift run --resume` finds the latest in-progress run for
the project, validates that the config and the suite path haven't changed
since, and continues from where it left off. Already-completed calls
(including ones that errored at the LLM layer) are skipped.

A config change or a different suite path between attempts aborts the
resume — start a fresh run instead. Upgrading the CLI counts when the release
added config fields (they join the hashed canonical dump), so an older CLI's
in-progress run cannot be resumed after upgrading. The suite's *contents* are not checked,
so after editing examples, start a fresh run yourself.

## How do I push a run to hosted EvalShift?

Sign in through the hosted web app, then approve CLI login in the browser:

```bash
evalshift login          # defaults to https://api.evalshift.dev
evalshift whoami
```

Set `project: org-slug/project-slug` in `evalshift.yaml` or pass
`--project org-slug/project-slug`, then run:

```bash
evalshift compare --suite-name <suite> --yes --push
```

See [Hosted EvalShift](hosted.md) for credential precedence, bundle contents,
and troubleshooting.

## Why does the `max cost` row in `evalshift compare` look so much higher than the actual `Total cost` in the report?

The pre-flight figure is a **worst-case ceiling**, not a forecast.
`evalshift compare` (and `evalshift run`) prices each call as if the model
emits its full registry `default_max_tokens` of completion (4096).
Real completions — especially agent-style runs that produce short
tool-call decisions — are usually far shorter than the cap, so the
actual `Total cost` in the report typically lands well below the
displayed ceiling.

The figure is conservative on purpose: the cost-confirmation prompt
(triggered above $10) wants to over-warn rather than under-warn. If
you see `≤ $0.17` and the run actually cost $0.03, that's expected.

## How do I lower the cost of a run?

* **Set the SQLite cache to be on** (it's the default). A re-run of
  the exact same configuration makes no run-stage calls, agent suites
  included: examples that offer tools are cached one entry per replayed
  round. Evaluate-stage embedding and judge calls are cached too.
* **Use cheaper models.** The model registry assigns sensible
  defaults but you can drop everything to flash/mini/haiku tier.
* **Skip the LLM judge.** Structural and semantic evaluators are
  much cheaper. Drop the `evaluators.llm_judge` section to disable
  the judge entirely.
* **Cap with `max_cost_usd`** in `defaults` (a future tightening
  will hard-enforce; currently a soft ceiling).

## What does "passthrough" mean next to my model id in `evalshift test-call`?

It means the id you passed isn't in EvalShift's curated registry.
The id is sent to LiteLLM as-is (with provider prefix inferred from
the prefix). If LiteLLM doesn't know the model either, you'll get a
clean error from the provider when you make the call.

## Why does my Cohen's d show as 0 with severity "none"?

Two common causes:

1. **Every delta is identical.** When the variance in deltas is near
   zero, the test is skipped and severity defaults to `none`.
2. **Your sample size is too small.** With `n < 5` the test is
   skipped and severity is `insufficient` (not `none`).

If you expected a real signal, double-check your evaluator output
range — many "all the same" cases are evaluators returning a constant.

## Why is the migration-policy verdict `inconclusive`?

Three common causes, all by design:

1. **Every configured evaluator is advisory** (`blocking: false` — the
   fresh `evalshift init` state), so nothing gates quality. Promote
   evaluators to blocking as your suite grows. The cost and latency
   budgets still apply — they read the run's calls, so a breach there
   reports `fail`, not `inconclusive`.
2. **A rate budget was breached but the 95% Wilson interval can't
   confirm it** at this suite size — grow the suite.
3. **All comparisons were `insufficient`** (n < 5).

A breached `max_invariant_violations` is the exception to all three: it is a
count, conclusive however small the sample, so a blocking hand-written trace
rule the target broke reads `fail`, never `inconclusive` — even when every
comparison is insufficient.

`analyze` and `compare` print the specific reason and the recommended fix
under the verdict line, and record them in `migration_decision.json`
(`reason` / `recommendations`).

## Does EvalShift work with LangChain agents?

You don't need LangChain to use EvalShift. Each golden-suite example
carries its own toolset — a `toolset_ref` pointing at a sidecar, or an
inline `tools` list (Anthropic-shape or OpenAI-shape, either works). The
usual path is automatic: capture your agent with the `evalshift-sdk` (works
regardless of framework — LangChain, a manual loop, anything) and
`capture promote` / `capture sync` record the toolset it was actually
offered. Writing a suite by hand instead? Inline `tools:` directly on each
example — see [Agent migrations](agents.md#suite-ground-truth). If your
tools are defined as LangChain `Tool` objects, export them to JSON Schema
once for that inline list. Framework-side agent timelines can also be
scored via [external traces](traces.md).

## Does EvalShift evaluate multi-turn conversations?

Yes — one suite example per turn, each carrying a recorded `history`
prefix that is replayed teacher-forced: both models see byte-identical
context, and only the current turn's output is compared. See
[Multi-turn conversations](conversations.md). Full-conversation
re-driving (feeding the candidate's own replies into later turns) is
deliberately not supported — it breaks the paired-comparison contract.

## Why does a hand-written config block on semantic when init does not?

Because the two defaults differ on purpose. The **library default** for
every evaluator's `blocking` is `true`, so a hand-written `evalshift.yaml`
that lists `semantic:` (or an `llm_judge` entry) without the key gates the
verdict on it. `evalshift init` writes `blocking: false` for both: the
semantic score measures drift from the *source* output, not correctness,
so a right answer in different words reads as a regression, and on the
small suites a fresh capture starts with that noise (and judge noise)
would dominate the verdict. Use a judge criterion for correctness.

The library default is not flipped to match because that would silently
turn a failing migration into a passing one for every existing config that
relies on the omitted key — a gate loosened under a minor release. It
stays `true` until a `version: 2` schema. To get init's behaviour in a
hand-written file, say so:

```yaml
evaluators:
  semantic:
    embedding_model: text-embedding-3-small
    blocking: false
  llm_judge:
    - criterion_name: helpfulness
      criterion_prompt: Which answer helps the user more?
      blocking: false
```

See [`blocking`](configuration.md#blocking-every-evaluator).

## Where is `evalshift validate` / `evalshift test-call` in `--help`?

They're hidden — they're development aids, not part of the supported
user pipeline. Both still run if you invoke them by name. `validate` also
prints the [CI pin drift](github-action.md#pin-drift) warning after its
success line (advisory; the exit code is unchanged).
