# Hosted EvalShift

Hosted EvalShift is optional. Local commands stay local until you explicitly
package and push a completed run.

Use hosted when you want shared run history, web viewing, diff links, and
GitHub pull request comments.

## Login

Register or sign in through the hosted web app, then authenticate the CLI:

```bash
evalshift login          # defaults to https://api.evalshift.dev
evalshift whoami
```

`login` prints a short code, opens the browser approval page, waits for you to
approve the CLI login, then stores the returned API token in
`~/.evalshift/credentials` with owner-only file permissions. Use
`--no-browser` on remote shells where the browser cannot open automatically.
`--timeout <seconds>` (default 900) bounds how long it waits for the approval.

Re-running `login` while the stored token for that host still works reuses it
instead of minting a new one, and prints `already logged in as <email>`. To
switch accounts, run `evalshift logout` first, then `evalshift login`.

You can still paste an existing hosted API token manually:

```bash
evalshift login --token <hosted-api-token> --host <hosted-api-url>
```

Manual token login verifies the token with `GET /me` before writing it.

### Personal tokens vs service account keys

`login` gives you a **personal token**: it is yours, it carries whatever your membership
allows, and it stops working the moment that membership does. That is the right credential
for a workstation and the wrong one for CI — a pipeline outlives the person who set it up,
and a personal token takes the pipeline down with the leaver.

For CI, mint a **service account key** instead. In the hosted web app: Settings → API
tokens → Service accounts. A service account is an org-owned machine identity that never
consumes a seat and can only ever hold the `member` or `viewer` role, so no CI credential is
owner-equivalent. Scope the key to the permission keys the job actually needs (`run:create`
plus `run:read` plus `policy:read` covers pushing a run, reading a diff, and the default
`fail-on: policy` gate), store it as an encrypted CI
secret, and pass it as `EVALSHIFT_TOKEN` — do not run `evalshift login` on a runner.

Keys rotate with an overlap: mint the successor, update the secret, confirm a green run,
then let the predecessor expire.

To remove local hosted credentials from this machine:

```bash
evalshift logout
```

Credential precedence is:

1. Explicit CLI flags: `--host` and `--token`.
2. Environment variables: `EVALSHIFT_HOST` and `EVALSHIFT_TOKEN`.
3. Stored credentials: `~/.evalshift/credentials`.

The default host is production (`https://api.evalshift.dev`). Use `--host` or
`EVALSHIFT_HOST` to point at staging or a local development server
(`http://localhost:8000`).

## Configure a project

Hosted push needs a project path in `org-slug/project-slug` form. Put it in
`evalshift.yaml`:

```yaml
project: acme/model-migration
```

Or pass it for a single command:

```bash
evalshift push <run-id> --project acme/model-migration
```

`evalshift.yaml` used to accept a `thresholds:` block here. It was removed: it
gated nothing, and a config that still sets it fails to load. Delete the key —
[`migration_policy`](configuration.md#migration_policy) is the single source of
truth for gating, and `push` carries the resolved policy in the bundle.

## Bundle and push

`bundle` packages a completed local run from `.evalshift/runs/<run-id>/`:

```bash
evalshift bundle <run-id>
```

The output is `.evalshift/runs/<run-id>/run_bundle.json.gz`. It carries:

- the run manifest (ids, models, suite, git metadata, config and dataset hashes,
  and `cli_version` — the evalshift version that produced the run, so the web app
  can tell a run that recorded no output from one produced by a CLI that could not
  record it),
- one row per example: inputs, both models' outputs, per-evaluator scores, and
  the cost/latency deltas,
- each example's `traces` — the replay's own tool-call traces, one stream per
  model side: the ordered tool calls with their names, arguments and call ids,
  any final text, refusal messages and round markers. No `model_call` events
  or tool results are included. A stream over 256 KB keeps its leading events
  and is flagged `truncated`. Imported agent traces (`traces import`) stay
  local and are not uploaded; the verdicts `evaluate` computes on them do —
  scores, explanations, and for `trace_invariants` the broken rules (tool
  names and, for `arguments` rules, the quoted offending value; see
  [What `push` sends](#what-push-sends-block-by-block)),
- the aggregate, `analysis`, and the policy `decision`,
- `economics` — a run-level per-role rollup of calls, tokens, cost and latency,
- `methodology_notes` and the evaluator config and dataset snapshot,
- `insights` — the machine-written narrative, when one was generated.

`report.html` is **not** uploaded. It is still written to
`.evalshift/runs/<run-id>/` for local viewing; the hosted app renders the run
from the data instead. Bundle bytes are deterministic — the same run always
compresses to the same file.

**Keep the CLI current.** Hosted EvalShift validates bundle shape, not CLI
version: every block of the bundle is parsed strictly, so a bundle missing a
required field — or carrying one the server no longer knows — is rejected on
upload. In practice that means bundles built by a CLI older than 0.10.0, where
the current shape landed, do not upload. Upgrade with
`pip install --upgrade evalshift`.

Push a local run:

```bash
evalshift push <run-id>
```

`push <run-id>` builds `run_bundle.json.gz` only when the run directory has
none; an existing bundle is uploaded as-is. After editing `evalshift.yaml`
(a `migration_policy` budget, say), re-run `evalshift bundle <run-id>` before
pushing, or the upload carries the numbers from when the bundle was built.

`bundle` — and `push <run-id>` when it has to build one — records git
metadata and needs a commit to point at: it fails with `could not determine a
valid 40-character git SHA; run inside git or set GITHUB_SHA` outside a git
checkout unless `GITHUB_SHA` is set. The branch comes from `GITHUB_HEAD_REF`,
then `GITHUB_REF_NAME`, then git, falling back to `local`; the pull request
number from `GITHUB_REF` (`refs/pull/<n>/…`) or the event payload at
`GITHUB_EVENT_PATH`.

Push a prebuilt bundle:

```bash
evalshift push --bundle .evalshift/runs/<run-id>/run_bundle.json.gz
```

Run the whole local pipeline, then push:

```bash
evalshift compare --suite-name <suite> --yes --push
```

`push` validates the bundle against the same schema the server uses before it
opens a connection, so a stale, hand-edited, or foreign bundle fails locally in
under a second instead of after a full upload:

```
✗ bundle failed schema validation: 'budget_results' is a required property
```

`push` exits 1 and nothing is uploaded.

A bundle at or over **50 MB** compressed prints a warning and uploads anyway:

```
! bundle is 62.4 MB compressed, over the 50 MB soft limit; the server's hard limit is 100 MB and it rejects anything larger.
```

The hard limit is the server's to enforce and is configurable there, so the CLI
quotes it rather than applying it.

Two more notices can appear before `push` reports success. A bundle with no
`migration_policy` configured carries no `decision.policy`, so the hosted gate
has nothing of this run's own to check: unless the project still has an old
web-app policy for the server to fall back on, the gate reports `inconclusive`
and the pull request it belongs to is never blocked (unless the GitHub Action
runs with `require-policy: true`, which fails the job for such a run) — a
silence that reads exactly like a passing gate unless `push` says so. The warning prints before
the network is touched, so it cannot yet know which of the two this project is;
it hedges accordingly, and prints once:

```
! this run carries no migration policy; unless this project still has an old web-app policy, the hosted gate reports inconclusive and never blocks — add migration_policy to evalshift.yaml
```

And once the server's initiate response comes back — before the bundle is
uploaded — if the project's only policy was
configured in the web app and `evalshift.yaml` has no `migration_policy` of
its own, `push` prints that policy back as the block to paste into the file:

```
! this project has a policy configured in the web app; move it into evalshift.yaml:
migration_policy:
  max_overall_regression_rate: 0.3
  min_equivalence_rate: 0.75
```

Only the budgets the web app actually set are printed — writing the CLI's
other defaults into the file would pin values that are meant to move with the
CLI. Once `evalshift.yaml` has its own `migration_policy`, the yaml is the
source of truth and this hint stops appearing.

On success, `push` prints only the hosted run URL. If the backend already has
an available run with the same id, the CLI treats that as idempotent success
and prints the existing URL.

## Project auto-create

If the hosted project is missing, `push` can create it automatically when:

- The token can see the org.
- The token has permission to create projects in that org.
- `--create-project` is enabled, which is the default.

Disable auto-create when you want strict CI behavior:

```bash
evalshift push <run-id> --no-create-project
```

Project-scoped tokens cannot auto-create projects.

When auto-create fails, the error names the project slug, **the host the CLI
actually talked to**, and the HTTP status and sentence the server returned —
because host resolution has four sources (`--host`, `EVALSHIFT_HOST`, the
credentials file, then the `https://api.evalshift.dev` default) and pushing to
the wrong one fails exactly like a permissions problem.

## Plan limits

Running locally is always unlimited. Plan limits apply only to what you push.

Every new hosted account gets a 30-day Pro trial (no card) that covers every
org it creates; after that each org needs its own Pro subscription.

When a push exceeds your organization's plan — monthly runs, seats, concurrent
runs — the org's trial ended without a subscription, or the subscription has
stopped paying, the server refuses the push with HTTP 402 and the CLI prints
exactly what the server said:

```
✗ EvalShift: this run needs a paid plan.
  The free trial for this organization ended on 2026-11-02. Subscribe to Pro to push runs and keep the CI gate — existing runs stay readable.
  Upgrade: https://app.evalshift.dev/app/acme/settings/billing
```

`push` exits 1 and nothing is uploaded. The CLI does not evaluate entitlements
itself and does not retry a payment error — retrying cannot change the answer.
Your run and its `report.html` stay on disk under `.evalshift/runs/`, so you can
push the same run id again once the org is subscribed (or, for the monthly
quota, after the counter resets). An expired org stays readable — its runs,
diffs and baselines still load.

Transient upload failures (HTTP 429 and 5xx from object storage) are a separate
case and *are* retried with exponential backoff.

## Privacy model — exactly what uploads

This section is the canonical data contract for hosted EvalShift. If you need
to clear a push with a security or compliance team, this is the section to
hand them.

The CLI contains **no telemetry**: no analytics, no crash reporting, no
phone-home of any kind. It opens two kinds of network connections (three with
`captures.store`), all initiated by you:

1. **Your model providers** (whichever you configure — any provider LiteLLM
   supports), using your own API keys: `run` sends the rendered prompts and
   conversation histories to both models, `evaluate` sends outputs to the
   embedding and `llm_judge` models, and `report` sends the worst regressions'
   inputs and outputs to `defaults.insights_model` unless you pass
   `--no-insights`. This traffic goes to your providers, never to EvalShift.
2. **Hosted EvalShift** (`https://api.evalshift.dev`, or your `--host`), only
   when you run `login`, `whoami`, `push`, or `compare --push`. The local
   commands — `doctor`, `run`, `evaluate`, `analyze`, `report`, `bundle` —
   send nothing to EvalShift-operated services.

With [`captures.store`](configuration.md#captures) configured, `capture sync`,
`capture list`, `capture fetch` and `doctor` also read from that object store —
a bucket you own, with your own cloud credentials. Nothing from it goes to
EvalShift.

### What `push` sends, block by block

`push` uploads `run_bundle.json.gz` plus two pieces of request metadata: the
bearer token (an `Authorization` header, sent only to the configured host) and
the compressed bundle size. (`login` additionally sends a client name that
includes your machine's hostname, so you can recognize the session in the web
app.)

The bundle itself contains:

| Block | What is inside |
| --- | --- |
| `manifest` | Run id, `org/project` slug, source and target model ids, suite name, git commit SHA, branch name, PR number, the **local suite file path** as a string (it can reveal directory or user names), two content hashes, the run timestamp, and the CLI version. |
| `examples[]` — one row per prompt × example | The example's template variables (`inputs`) **verbatim**; its `expected` reference output **verbatim**; both models' **full output text**; tool-call traces from the replay: each side's tool calls (names, arguments, call ids) with round markers, any final text and refusal messages, capped at 256 KB per side. Imported agent traces (`traces import`) are not uploaded; per-evaluator scores, error strings and one-line explanations (`target won`, `tie`, `target broke 1 trace rule(s): no-delete`); for each `trace_invariants` evaluator, the rules each side broke — rule id, rule type, tool, round, a one-line detail that may quote an argument value, and the sample ordinal on a repeated-sampling run — so the hosted report can name the rule, not only count it (`rules_checked`, `owner` and `failure_categories` are not repeated on the score row; they already ship in `evaluator_config` / `decision`) — including rules checked on imported traces (`traces: imported`), so an entry can name a tool, and for an `arguments` rule quote the offending value, from a timeline that otherwise stays local; per-side cost and latency; tags and slice names. |
| `aggregate`, `analysis`, `decision`, `economics` | Pass/fail counts, statistical comparisons, the migration verdict, and per-role token/cost/latency rollups. Numbers and verdict labels, not content. `decision.policy` is the resolved `migration_policy` this run's verdict was computed under — every top-level budget with its default applied, plus `slices` — or `null` when no `migration_policy` is configured. It therefore carries `max_invariant_violations` (default `0`) even when the config has no trace rules; hosted EvalShift accepts and stores it (since 2026-10-08) but never re-evaluates it — a run that carries its own policy is answered with the CLI's decision, which did check it, and only runs pushed without a policy fall back to the server's six-budget re-check. It is what lets the hosted gate check a pull request against the exact budgets the verdict used, instead of a separate policy configured elsewhere. |
| `methodology_notes` | The model ids and the statistical-contract sentences shown in every report. |
| `insights` | The machine-written run narrative, when one was generated. It is prose *about* your run and can paraphrase or quote the regressions it summarizes. |
| `evaluator_config` | Config version; the prompt list **metadata only** — prompt names, file paths, and variable names, with every prompt body replaced by a `content_hash`; the whole `defaults` block (model ids, concurrency, cache flag, cost ceiling, max_tokens, samples_per_example); an always-empty `slices` list (a legacy key, kept so `eval_config_hash` does not change — the top-level `slices:` config key was removed); and the full evaluators block — which includes each `llm_judge` entry's `criterion_prompt` text, so keep judge criteria free of secrets. An empty `trace_invariants` list is left out of it, so a config without trace rules keeps its `eval_config_hash`; adding rules changes the hash like any other evaluator edit. |
| `dataset_snapshot` | Suite path, example count, slice names, and one `examples_hash`. **No example content.** |

### What never leaves your machine

- **Provider API keys and the hosted token.** Neither is ever inside a bundle.
  Keys go only to their own providers; the token goes only to the configured
  host as an auth header.
- **Prompt bodies and system prompts.** A `manual` prompt's `content` is
  replaced by `content_hash`; a `python_string` prompt's body never enters the
  config at all (only its file path and variable name do).
- **Suite conversation histories** (`history`, including any embedded system
  message). The dataset snapshot ships hashes, not examples.
- **Tool definitions.** Toolsets — names, descriptions, JSON schemas — are not
  in the bundle; only the calls a model actually made at run time appear, in
  the traces.
- **Local artefacts**: `raw.jsonl` (the raw provider requests and responses),
  imported agent traces (`traces.jsonl`; only the verdicts computed on them
  upload — `agent_trace` scores and `trace_invariants` rule results — never
  the timelines; see the `examples[]` row above), the SQLite response cache,
  `.evalshift/captures/` (and, with `captures.store` configured, the bucket
  *you* named that they are mirrored from — read with your credentials, never
  uploaded to EvalShift), `state.json`, `report.json`, and `report.html`.

The content hashes that replace this data (`dataset_hash`, `examples_hash`,
`prompts[].content_hash`) are SHA-256 digests, so hosted diffs and baselines
still align across runs without the content itself uploading.

### The fields that do upload can still be sensitive

`inputs`, `expected`, both model outputs, and tool traces upload **verbatim**.
If your suite rows contain customer data, or a model echoes a secret it was
given at run time, that content is in the bundle — EvalShift cannot tell the
difference. Before pushing runs built from production captures:

- redact at capture time with the [SDK redaction boundary](sdk.md) so
  sensitive values never reach disk, and
- inspect the exact bytes a push would upload:

```bash
evalshift bundle <run-id>
gunzip -c .evalshift/runs/<run-id>/run_bundle.json.gz | jq . | less
```

`push --bundle` uploads exactly the file you inspected. If a run must not
leave the machine, simply never push it — every local artefact, the HTML
report included, works without an account.

Run insights are a separate exposure from the hosted upload: generating them
sends the worst regressions' inputs and outputs to `defaults.insights_model`,
the same way an `llm_judge` criterion sends outputs to its judge. Disable with
`evalshift report --no-insights` (or `compare --no-insights`).

Do not put hosted API tokens or provider API keys in config files. Use the
credential file locally and repository secrets in CI.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `missing hosted token` | No flag, env var, or credentials file token is available. | Run `evalshift login --host <hosted-api-url>`, paste a token with `evalshift login --token <hosted-api-token> --host <hosted-api-url>`, or set `EVALSHIFT_TOKEN`. |
| `host uses plain http` warning | The host is non-local HTTP. | Use HTTPS for non-local hosts. |
| `hosted project is required` | No `project` in config and no `--project` flag. | Add `project: org/project` or pass `--project`. |
| `could not determine a valid 40-character git SHA` | `bundle` (or `push <run-id>` building a bundle) ran outside a git checkout, or in one with no commits, and `GITHUB_SHA` is unset. | Run inside a git repository with at least one commit, or set `GITHUB_SHA`. |
| `project was not found` | The project does not exist and auto-create is disabled or not allowed. | Ask an owner to create it, use an org-scoped owner token, or enable auto-create. |
| `cannot auto-create <slug> at <host>` | The message names the host it talked to and the server's status. Most often the host is not the one you meant: with no `--host` and no `EVALSHIFT_HOST`, an unset credentials file falls back to `https://api.evalshift.dev`, where your org does not exist. | Run `evalshift whoami` and check the host it prints. If it is wrong, `evalshift login --host <hosted-api-url>`. If the host is right and the status is 403, the token lacks org access — see [Project auto-create](#project-auto-create). |
| `Your account has been suspended` | An EvalShift admin suspended the account this token belongs to. Its browser sessions and personal tokens are refused; service-account keys in your org keep working. | Contact support@evalshift.dev. Running `evalshift login` again does not help. |
| `this run needs a paid plan` | The org's plan does not cover this push, its 30-day trial ended without a subscription, or the subscription has stopped paying. | Open the upgrade URL printed with the message; for the monthly quota you can also wait for the reset and push the same run id again. See [Plan limits](#plan-limits). |
| `this run carries no migration policy` warning | No `migration_policy` is configured in `evalshift.yaml`, so the bundle has no `decision.policy`. | Add `migration_policy` to `evalshift.yaml` (see [Configuration](configuration.md#migration_policy)). Until then the gate has only whatever old web-app policy the project still has; with none, it reports `inconclusive` and never blocks the pull request unless the GitHub Action sets `require-policy: true`. |
