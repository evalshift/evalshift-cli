# GitHub Action

The reusable EvalShift GitHub Action runs the CLI in CI, pushes the run to
hosted EvalShift, compares it with a compatible base-branch run, and updates
the pull request with a regression summary.

## Scaffold the workflow

From a new EvalShift project:

```bash
evalshift init --ci      # capture-first project
```

This writes `.github/workflows/evalshift.yml` — a production-shaped,
self-documenting workflow (its header comment carries the full setup
checklist) with three jobs:

- **`discover`** — lists every committed suite under
  `.evalshift/suites/*/golden.jsonl`. A suite added by `evalshift capture
  sync` is evaluated on the next run with no workflow edit; a project with no
  suites yet skips green with a notice instead of failing.
- **`eval <suite>`** — a matrix job per discovered suite (the action
  evaluates one suite per invocation), selected with `suite-name:` — the key
  the suite is wired under in `evalshift.yaml`, not its path, so the suite's
  own evaluator block travels with it (see [Selecting a
  suite](#selecting-a-suite-name-not-path)). Runs `fail-on: policy`, so the verdict
  is the one this run computed against its own `migration_policy` block in
  `evalshift.yaml` and pushed inside the bundle. `evalshift-version` is pinned
  to the CLI that scaffolded the project: the CLI that *reads* the config in CI must be at least as new
  as the CLI that *wrote* it locally (`extra: forbid` rejects newer keys), and
  the CLI warns when the pin falls behind (or runs ahead of the local CLI) — see [Pin drift](#pin-drift). `max-parallel` defaults to 1 —
  raise it toward your hosted plan's concurrent-run limit (Pro: 5;
  Enterprise: by contract). The PR comment is posted by the first matrix job only: the
  comment marker is a constant, so multiple suites would overwrite one
  another's summary.
- **`evalshift gate`** — the join job to require in branch protection. It
  fails if any suite failed and passes when evaluation was skipped (fork PR
  — no secret access, no suites committed, or `EVALSHIFT_TOKEN` not set
  yet). Require this check, not the per-suite jobs (dynamic names) and not
  the `evalshift/regression` commit status (with several suites the last
  writer wins that status).

The workflow keys off `${{ secrets.<PROVIDER>_API_KEY }}` for the provider
chosen at `init` time. An Anthropic or DeepSeek project borrows its
embedding model from OpenAI or Gemini, so `init --ci` also wires that
provider's key (`OPENAI_API_KEY` or `GEMINI_API_KEY`) as a second, optional
secret: left unset it arrives empty, and the run skips the advisory
`semantic` evaluator with a warning instead of failing. Add further keys
under the eval job's `env:` if you point a judge or embedding model at
another family. Suites must be committed
for CI to see them — keep runtime data ignored and un-ignore just the suites:

```gitignore
.evalshift/*
!.evalshift/suites/
!.evalshift/toolsets/
```

Runs on pushes to the main branch create the base-branch baselines pull
requests diff against; the workflow's `concurrency` block therefore cancels
superseded runs on PRs only, never on main.

The scaffold refuses to overwrite existing files. If you already have
an `evalshift.yaml`, run `evalshift init --ci --directory` somewhere scratch
and copy `.github/workflows/evalshift.yml` across instead of overwriting
your project.

## Required secrets

- `EVALSHIFT_TOKEN`: a **service account key** from the web app (Settings → API
  tokens → Service accounts), scoped to `run:create` + `run:read` + `policy:read`.
  Not a personal token — that one dies with its owner's membership and takes the
  pipeline with it. `policy:read` is what lets the default `fail-on: policy` gate
  read the hosted verdict; without it the check falls back to `fail-on: regression`
  silently. A scoped key cannot auto-create the hosted project (`project:create` is
  owner-only), so create the project once in the web app and set
  `create-project: false`.
- Provider API keys used by the source, target, judge, or embedding models.
  A missing key for a *blocking* judge or embedding model fails the run before
  any call; an advisory one is skipped with a warning.

Do not hard-code tokens in workflow YAML. Keep them in GitHub encrypted secrets
— repository or, better for production repos, environment secrets. Never expose
them to a `pull_request_target` workflow: that trigger runs the base repo's
workflow with secrets in scope against fork code.

Rotate on a schedule: rotate the key in the web app (the old one keeps working
for a 24-hour grace window), update the GitHub secret, confirm a green run, then
let the old key expire.

## Project config

The Action expects the CLI project to know where hosted runs should land:

```yaml
project: acme/model-migration
```

The default workflow runs the local pipeline, finds the latest run id, and
pushes that run to hosted EvalShift.

The action passes the hosted token through environment variables so command
output does not expose it.

## Baselines and PR comments

On pull requests, the Action:

- Pushes the candidate run.
- Looks for the latest compatible run on the base branch.
- Fetches the hosted diff if a baseline exists.
- Creates or updates one PR comment marked by EvalShift.
- Sets commit status `evalshift/regression`.

If no compatible baseline exists, the comment explains that the run was pushed
but there is no baseline yet. Under `regression` / `any-slice-regression`,
gating passes. Under the default `policy` mode, the job still follows the
run's policy verdict.

## `fail-on` modes

| Mode | Behavior |
| --- | --- |
| `policy` (default) | Ask hosted EvalShift for the migration-policy verdict — the verdict the run itself computed against the `migration_policy` limits in `evalshift.yaml` and carried in its bundle, returned rather than re-scored. `fail` fails; `pass`/`conditional_pass`/`inconclusive` pass. A run pushed with no `migration_policy` is reported as ungated with a workflow warning and passes, unless `require-policy: true`. If the policy check is unreachable, falls back to `regression` gating and says so. |
| `never` | Do not fail the workflow for hosted regressions. |
| `regression` | Fail when the hosted diff reports one or more regressed examples. |
| `any-slice-regression` | Fail when any slice pass rate moves down. |

The Action can still fail for setup errors, provider auth errors, invalid
config, upload failures, or finalize failures.

## Inputs

Common inputs:

| Input | Default | Description |
| --- | --- | --- |
| `token` | required | Hosted EvalShift API token. |
| `host` | hosted default | Hosted API base URL. |
| `config` | `evalshift.yaml` | Config path. |
| `suite-name` | — | Name of a suite wired under `suites:` in `evalshift.yaml`. Preferred — see [Selecting a suite](#selecting-a-suite-name-not-path). Needs a CLI pin of `0.14.0` or newer. |
| `suite` | `golden.jsonl` | Suite path, for a file that is not wired into the config (one suite per invocation). Mutually exclusive with `suite-name`. |
| `fail-on` | `policy` | Gate mode — see the table above. |
| `require-policy` | `false` | Fail the job when the pushed run carries no migration policy (`policy` mode only). |
| `evalshift-version` | action default (may lag) | Exact CLI version installed from PyPI. Always set it: it must be at least as new as the CLI that writes your `evalshift.yaml` (reader ≥ writer). `init --ci` pins it to the scaffolding CLI. |
| `create-project` | `true` | Allow project auto-create when permissions allow it. |
| `comment` | `true` | Post or update the PR comment on pull requests. |

See the action repository README for the full input list.

## Selecting a suite: name, not path

The action takes either `suite-name:` (a key under `suites:` in
`evalshift.yaml`) or `suite:` (a path). They load the same rows, but only the
name resolves that suite's **own `evaluators:` block** — the one `evalshift
capture sync` writes for a tool-calling suite:

```yaml
suites:
  planner:
    source: captured
    path: .evalshift/suites/planner/golden.jsonl
    evaluators:
      tool_selection:
        - name: routing
          conformance: expected
          divergence: set
```

Select that suite by path and it is scored with the **top-level** `evaluators:`
instead. There is no warning — a bare path is a legitimate way to run a suite
that has no entry under `suites:`. If the top level is `semantic` + `llm_judge`
and the suite's rows are tool calls, nothing scores at all and the run fails at
`analyze` with `scores.jsonl is empty`.

So: a suite with an entry under `suites:` is selected by name, which is what
`init --ci` scaffolds. `suite:` is for a one-off file outside the config.

## Pin drift

`evalshift.yaml` rejects unknown keys (`extra: forbid`), so a config written by a
newer CLI can fail outright on an older one. The rule is **reader ≥ writer**:
the version the action installs in CI must be at least as new as the CLI you
run `capture sync` and `init` with locally. Upgrading locally without bumping
`evalshift-version` is the common way to break this.

The CLI checks for it wherever it writes or validates config — `capture sync`,
`init` (without `--ci`, next to a workflow it did not write; `init --ci` pins
the scaffolding CLI itself and does not warn about the file it just wrote),
`doctor` (a `ci pin` row), and `validate`. It parses every `.github/workflows/*.yml` for
`evalshift/evalshift-action` steps and compares their `evalshift-version`
with its own:

```text
⚠ CI installs evalshift 0.12.1 (.github/workflows/evalshift.yml, job evalshift) but the local CLI is 0.13.1 — an older CLI rejects config keys a newer one writes.
  Fix: set `evalshift-version: "0.13.1"` on the evalshift/evalshift-action step.
```

| Status | Trigger | Fix |
| --- | --- | --- |
| stale | a literal pin is older than the local CLI | set `evalshift-version: "<local>"` |
| unpinned | a step has no `evalshift-version` (action default applies and may lag) | add the pin |
| ahead | every pin is newer than the local CLI | `pip install -U evalshift` |

Equal pins, `${{ }}` expressions, unparseable values, and an editable install
without metadata are silent. The check is advisory only: the CLI never edits
your workflow and never changes an exit code, and in CI it is a no-op by
construction — the running CLI *is* the pin. Config `version: 1` is not
bumped for additive fields; see the
[config version policy](configuration.md#config-version-policy).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No PR comment | Missing `pull-requests: write` or `issues: write`. | Add both permissions to the workflow. |
| No commit status | Missing `statuses: write`. | Add the permission. |
| Push fails with missing project | Project does not exist and token cannot auto-create it. | Create the project in the web app or use an org-scoped owner token for first setup. |
| No compatible baseline | Base branch has not pushed a compatible run yet. | Merge or run EvalShift on the base branch once. |
