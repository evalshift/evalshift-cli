# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `evaluators.trace_invariants`: hand-written rules over tool-call traces —
  `forbidden`, `required`, `order`, `call_count` (min/max/exact per tool) and
  `arguments` (JSON Schema) — checked on both sides of every pair, on replayed
  traces or (`traces: imported`) on imported ones. A rule the target breaks
  counts whatever the source did. Its `applies_to` is enforced: a prompt
  outside the globs is never checked.
- `migration_policy.max_invariant_violations` (default `0`, per-slice
  overridable): examples on which the target broke a blocking trace rule. A
  breach fails the run even when the suite is too small for statistics, and a
  slice whose own budget breaches reads `fail` however few comparisons it has.
  Pushed bundles' `decision.policy` now always carries the field, so pushing
  needs an evalshift-server that accepts `max_invariant_violations` in policy
  snapshots; this CLI release waits on that server deploy.
- The HTML report lists every broken trace rule in a "Trace rules broken"
  panel, with its owner and whether the source broke it too — advisory
  (`blocking: false`) entries included, though only blocking ones count
  toward the budget. `report.json` carries the same rows under a new
  top-level `invariant_violations` key.
- `evalshift init` scaffolds `max_invariant_violations: 0` in every
  `--profile` policy block, beside the other budgets.

### Changed

- Upgrading moves the resume `config_hash`: the new config fields are part of
  every config's canonical dump, so a run left in progress by an older CLI
  cannot be `--resume`d after upgrading (it fails with the usual config-hash
  mismatch). Start a fresh run; cached responses are reused.
### Fixed

- `capture sync` no longer drops a managed suite's hand-written
  `evaluators.trace_invariants` when it regenerates the entry. The marker
  comment still says hand edits are overwritten; this block is the one
  exception.

## [1.2.2] - 2026-10-05

### Fixed

- `evalshift push` printed a traceback instead of the upgrade prompt when hosted EvalShift
  refused the run with HTTP 402 right after auto-creating the project (over the monthly run
  or concurrent-run limit, or a subscription that stopped paying). It now prints the
  server's message and the billing link and exits 1. A 402 while auto-creating the
  project — what an organization whose trial has ended gets — now shows the same prompt
  instead of an access-permission hint.
- `evalshift push` printed a traceback when the connection failed while auto-creating the
  project; it now prints the error and exits 1.

### Changed

- The EvalShift repositories moved from the `babaliauskas` GitHub account to
  the `evalshift` organization: <https://github.com/evalshift/evalshift-cli>,
  `evalshift-sdk` and `evalshift-action`. Links, package metadata and the
  workflow `evalshift init --ci` scaffolds now use
  `evalshift/evalshift-action@v0`. GitHub redirects the old names, so existing
  workflows keep working.
- The CI pin check (`init`, `validate`, `doctor`, `capture sync`) recognizes
  `evalshift/evalshift-action` steps as well as the old
  `babaliauskas/evalshift-action` name, matching owner and repository names
  case-insensitively as GitHub does.
- Docs and the `init --ci` workflow comment describe hosted EvalShift's current plans: a
  30-day Pro trial for every new account, then Pro per organization (the hosted Free and
  Team plans are gone). Local runs stay free and unlimited.

## [1.2.1] - 2026-10-01

### Changed

- Top regressions in the HTML report now read as expandable: the worst one
  in each section starts open, each row has a boxed caret and a
  "Show details" / "Hide details" hint, and the header reacts on hover and
  shows a keyboard focus ring. Printing expands every regression in
  Chromium-based browsers.

## [1.2.0] - 2026-10-01

Shipped as a minor again. The `slices` removal below is breaking by the
letter of SemVer — a config that still sets the key stops loading — but the
field never had any effect: `analyze` has always built one slice per
distinct example tag plus `all`, so `name`, `filter` and `applies_to` renamed,
filtered or scoped nothing. A major would announce a migration that, for
anyone who never wrote `slices:`, does not exist; those who did get a
load-time error naming the key and the fix, the same way `thresholds` was
handled in 1.1.0.

### Added

- DeepSeek is a supported provider. `deepseek-flash` and `deepseek-v4-pro`
  are in the model registry; bare `deepseek-*` ids (as recorded by a capture
  of an app calling `api.deepseek.com` through the OpenAI client) resolve to
  `deepseek/…`; `DEEPSEEK_API_KEY` is checked before a run and shown by
  `evalshift doctor`; tool-call evals parse DeepSeek responses; a DeepSeek
  judge grading a DeepSeek arm gets the judge-family warning; and
  `evalshift init --provider deepseek` scaffolds a DeepSeek project. DeepSeek's
  default thinking mode ignores `temperature`, so DeepSeek arms and a DeepSeek
  judge carry the report's non-determinism banner, and every assistant turn
  replayed from the recording (tool rounds and chat history) is sent with a
  placeholder `reasoning_content`: DeepSeek requires it on tool requests and
  ignores it otherwise.

### Changed

- `evalshift doctor`'s `evalshift.yaml` row now ends a failure with "— run
  `evalshift validate` for details". The row has room for the summary only
  ("1 schema problem found"), which names neither the offending key nor the
  fix; `validate` prints both.

- Repeat runs of agent suites are now served from the response cache. Every
  example that offers tools used to bypass the cache, a leftover from v0.2
  when a cache entry could not hold a parsed tool trace, so each `run` of an
  agent suite paid for every call again, one per replayed round. Each replayed
  round is now its own entry. It is keyed on the canonical model, the prompt
  and inputs, the exact message list that round sends (history, current turn,
  and the recorded rounds and fixture results fed back), the tool list exactly
  as sent and in order (including `strict`), `generation_config` (so
  `tool_choice` and `parallel_tool_calls`), the effective temperature and
  `max_tokens`, the round index and the sample index. A hit restores the
  parsed trace, tokens, cost, latency and finish reason, so the `raw.jsonl`
  row is identical to the live one apart from `cached`, which is true only
  when every round hit, and the new `cached_rounds` count. A row with any
  round served from the cache carries latency from an earlier run, so it stays
  out of the report's live latency figures and its latency delta is marked not
  comparable, in `report.json` and in the bundle alike. Errors are never
  cached: the next run re-sends a failed round and serves the rounds before it
  from the cache. Truncated responses are cached and stay flagged, and
  `defaults.cache: false` still sends everything live. Existing
  `~/.evalshift/cache.db` files keep working: the new `trace_json` column is
  added in place on open, and every cached text response still hits.

### Removed

- **The top-level `slices:` key is gone from `evalshift.yaml`, and its
  removal is breaking.** The block (`name`, `filter`, `applies_to`) was
  validated and recorded in the run bundle, but analysis never read it:
  slices have always come from example `tags`, one per distinct tag plus
  `all`, so none of its fields renamed, filtered or scoped anything, and a
  run reports the same slices without it. **Migration: delete the block.**
  Per-slice budgets, the one thing it looked like it configured, go under
  `migration_policy.slices`, keyed by tag.

  A config that still sets `slices:` now **fails to load** instead of being
  quietly ignored, the same way `thresholds` has since 1.1.0, naming what
  happened and the fix:

  ```text
  `slices` was removed: it never had any effect. Slices come from example `tags` automatically (one per distinct tag, plus `all`). Delete it from evalshift.yaml; per-slice budgets go under migration_policy.slices, keyed by tag.
  ```

  `evalshift validate` prints it; `evalshift doctor` fails its config row
  and points at `validate`. `version:` stays `1`, under the same config
  version policy. The two shipped example configs that set the block no
  longer do, and `tests/unit/test_docs_currency.py` now fails if any doc or
  example shows a top-level `slices:` or `thresholds:` key.

  Hosted baselines are unaffected for every config that never set the key
  (or set `slices: []`): the bundle's `evaluator_config` still carries
  `"slices": []`, so `eval_config_hash` is byte-identical to what earlier
  CLIs computed and existing baselines keep matching. Deleting a *non-empty*
  block does change that hash, so runs pushed afterwards are not comparable
  to baselines pushed before until the base branch pushes a run with the
  edited config.

### Fixed

- The scaffolded CI workflow's key guidance (`evalshift init --ci`) named only
  `run:create` + `run:read`. That workflow defaults to `fail-on: policy`, which
  reads the hosted policy-check endpoint (`policy:read`); a key scoped to only
  `run:create` + `run:read` got a 403 there and the action silently fell back
  to `fail-on: regression`. The scaffold, `DOCS.md`, `llms-full.txt`,
  `docs/github-action.md`, and `docs/hosted.md` now name `run:create` +
  `run:read` + `policy:read` for the CI key.

- A run whose target was served entirely from the cache while the source ran
  live showed a -100% latency change in the HTML report header and in the
  run insights. Latency averages cover only calls measured live on this run,
  so a role with none of them averages 0, which means unmeasured, not
  instant. Both now say the latency change is not comparable unless both
  roles measured some latency live.

- Two `evalshift` processes opening the response cache at the same moment
  (parallel CI jobs, or a `run` beside an `evaluate`) could crash one of them
  with `table cached_calls already exists` when the cache file was new. Both
  had checked for the table, and the slower one then tried to create it too.
  Opening the cache now treats that error, and its migration counterpart
  `duplicate column name`, as "another process already did it" and carries
  on; any other schema error is still raised.

- Under SQLAlchemy 2.1, which fresh installs resolve (`sqlalchemy>=2.0`), a
  `CacheStore` opened on an in-memory SQLite database could silently lose
  concurrent writes. The default on-disk cache used by CLI runs was not
  affected. In-memory databases get SQLAlchemy's single-connection
  `StaticPool`, so concurrent sessions shared one transaction, and one
  session's rollback-on-return could discard another's uncommitted write.
  SQLAlchemy 2.1's aiosqlite rework (sqlalchemy#10415) made this happen
  routinely, so some cached writes were missing on the next lookup. Two
  orchestrator cache tests failed on every fresh checkout, and the
  embeddings-cache test failed intermittently. `CacheStore` now runs its
  sessions one at a time on a single-connection pool. This works the same
  on SQLAlchemy 2.0 and 2.1.

- `capture sync` priced calls recorded under a bare id at $0 when LiteLLM
  keys the model under both spellings but can only price the provider-prefixed
  one (DeepSeek). The price lookup now tries the provider-prefixed id first.

- README.md, DOCS.md, llms-full.txt, four `docs/` pages, and AGENTS.md
  advertised `evalshift all --push` as the command to run. `all` has been a
  hidden alias for `compare` since 1.0.0 — it still works, and always will —
  but it is hidden from `evalshift --help`, so the docs were teaching a name
  the CLI does not advertise. Every advertised example now reads
  `evalshift compare --push`; nothing about the CLI changed. `AGENTS.md` is
  now also guarded by `tests/unit/test_docs_currency.py`.

- README.md, docs/index.md, docs/faq.md, docs/hosted.md, and
  docs/getting-started.md described the LLM providers a run can call as
  "Anthropic, OpenAI, Google" (or, in docs/getting-started.md, the
  Oxford-comma "Anthropic, OpenAI, and Google") — read as a compatibility
  list, when every call actually dispatches through LiteLLM and those three
  are just the entries with a curated pricing table. A separate FAQ answer in
  the same `docs/faq.md` ("what models does EvalShift support?") already used
  the correct framing ("Anything LiteLLM supports."); all five sites now name
  LiteLLM as the boundary too, with Anthropic, OpenAI, and Google called out
  as the ones that get curated pricing and capability data. The
  `test_docs_currency.py` absence check now matches the Oxford-comma phrasing
  too, not just the plain three-item list. Nothing about the CLI changed.

- DOCS.md, llms-full.txt, and docs/configuration.md quoted the scaffolded
  `replay` prompt's content as `"{{input}}"` — the escaped form that appears
  literally in `init.py`'s `str.format` template, not what ends up on disk.
  Because the doubled brace is a format escape, the `evalshift.yaml` that
  `evalshift init` actually writes contains `"{input}"`, as `test_init.py`
  already asserted. All three sites now quote `"{input}"`, matching the other
  places in DOCS.md and llms-full.txt that already had it right. Nothing
  about the CLI changed.

- README.md claimed "the test suite covers 92% of the source," a literal
  nothing checked; the suite now measures 94%, and nothing enforced either
  number. Reworded the Status section to say "CI enforces a 90% coverage
  floor on the source" instead of quoting a point-in-time percentage, and
  added `--cov-fail-under=90` to the three CI entry points
  (`.github/workflows/ci.yml`, `Makefile`'s `test` target, and
  `.pre-commit-config.yaml`'s pre-push mirror) rather than to
  `[tool.coverage.report]`, so a targeted local run such as
  `pytest tests/unit/test_init.py` is not failed by a floor meant for the
  whole source tree. The floor sits below the measured 94% on purpose, so a
  real regression fails CI while an honest refactor does not.

- An audit of the docs against the code found them wrong in a few dozen
  places, and all of them now say what the CLI does. Nothing about CLI
  behaviour changed. The one that matters most concerns slices: the docs
  said a top-level `slices:` block picks examples by tag, names the slice,
  and scopes it to prompts. It did none of that. The block was validated and
  recorded in the run bundle, but analysis never read it. Slices come from
  example `tags`, one per distinct tag plus `all`, and per-slice budgets are
  keyed by tag under `migration_policy.slices`. The docs now say so, and the
  block itself is removed in this release (see Removed above). Imported
  agent traces (`traces import`) stay local; the bundle carries only the
  replay's own tool-call trace, without tool results or `model_call` events.
  The response cache served only tool-less examples, so every `run` of an
  agent suite was live and full price (it now serves them too; see Changed).
  `--resume` hashes the suite's path, not its contents. `push <run-id>`
  uploads an existing bundle as-is instead of rebuilding it, and `bundle`
  needs a git SHA. `--policy-gate` also fails when no `migration_policy` is
  configured. The `init` profile table had the wrong `model-upgrade` numbers
  and no tool-divergence column. The multi-turn suite example failed to load
  because it had no `tools`. The failure-label list was missing
  `TOOL_GROUND_TRUTH_MISS`. Upstream model-call failures and evaluator
  failures are handled the same way, as errored rows excluded from the
  statistics. The GitHub Action docs gained `require-policy` and the other
  missing inputs. `record_model_call` examples now pass the required
  `tools=`. DOCS.md's header said version 1.0.1; a new check in
  `tests/unit/test_docs_currency.py` keeps the version in DOCS.md and
  llms-full.txt equal to the package's.

## [1.1.0] - 2026-09-19

Shipped as a minor deliberately. The `thresholds` removal below is breaking by
the letter of SemVer — a config that sets the key stops loading — but the key
was free-form, gated nothing, and travelled no further than a project-settings
blob nobody read. A major would have signalled a migration that, for anyone who
never wrote `thresholds:`, does not exist. Those who did get an error naming
the key and the fix, which is the whole migration.

### Added

- `migration_decision.json` and the bundle now carry the resolved
  `migration_policy` the verdict was computed under, as `decision.policy`. The
  hosted server previously gated pull requests on a separate, web-edited
  policy that nobody configures, silently turning the PR gate off; the CLI's
  own budgets riding inside the bundle is what lets the server check a
  migration against the same numbers the CLI's verdict used. `null` when no
  `migration_policy` is configured, and on a `migration_decision.json`
  written by an older CLI. This release needs a hosted server that accepts
  `decision.policy`: local pre-flight validation passes for a policy-carrying
  bundle, so pushing a gated run at a host that has not deployed that change
  yet uploads in full and only then fails at finalize — self-hosted and
  staging deployments behind `EVALSHIFT_HOST` should upgrade the server first.

- `evalshift push` — and `evalshift compare --push`, which prints through the
  same console — now warn when the run being pushed carries no migration
  policy: unless the project still has an old web-app policy for the server to
  fall back on, the hosted gate reports `inconclusive` and the pull request it
  belongs to is never blocked, which is otherwise indistinguishable from a gate
  that passed. Projects whose only policy was configured in the web app get
  that policy printed back as the `migration_policy:` block to paste into
  `evalshift.yaml` — printed only while the yaml has no policy of its own, and
  validated first, so what is shown is config the CLI accepts.

### Removed

- **`thresholds` is gone from `evalshift.yaml`, and its removal is breaking.**
  The key was free-form and gated nothing: the migration verdict has always
  come from `migration_policy`, and `thresholds` only ever travelled to the
  hosted API as project settings. Keeping it meant a config could carry numbers
  that read like a gate and decided nothing — the same confusion
  `decision.policy` above exists to end. Nothing replaces it. **Migration:
  delete the block**, and express any gate you meant by it as a
  `migration_policy` budget.

  The break has two axes. A config that still sets `thresholds:` now **fails to
  load** instead of being quietly ignored, naming what happened and the fix:

  ```text
  `thresholds` was removed: it was free-form and gated nothing. Delete it from evalshift.yaml; migration_policy is the single source of truth for gating.
  ```

  Dropping the key silently would have preserved the very impression the
  removal is meant to end — a config that looks gated and is not. The second
  axis: `push_bundle()` in `evalshift_cli.hosted.push` no longer takes a
  `thresholds` keyword.

  `version:` stays `1`. The config version policy now says so explicitly: the
  literal marks a config that is still valid but would be read with the wrong
  meaning, and a removal that fails the load while naming the key is the
  opposite of that. Bumping it would have forced an edit on every config,
  including the majority that never set `thresholds`.

- The hosted traffic that carried it goes with it: `push` no longer sends
  `thresholds` when it uploads a run or creates a project, the CLI ignores the
  server's `canonical_thresholds` response field, and the "local thresholds
  differ from the hosted canonical thresholds" warning is gone along with the
  setting it compared against.

## [1.0.1] - 2026-09-18

### Fixed

- `evalshift init --ci` now scaffolds the eval job with `suite-name: ${{ matrix.suite }}`
  instead of `suite: .evalshift/suites/<name>/golden.jsonl`. The path form loads the same
  rows but resolves no suite *name*, so the suite was scored with the top-level
  `evaluators:` and its own block under `suites:` — the tool evaluators `capture sync`
  writes for a tool-calling suite — never loaded. On a capture-first project whose top
  level is `semantic` + `llm_judge`, that scored no rows at all and CI failed at `analyze`
  with `scores.jsonl is empty`, naming nothing that pointed at the selection. Regenerate
  the workflow (`evalshift init --ci --force`) or change the input by hand; the step needs
  `babaliauskas/evalshift-action@v0` at a release that offers `suite-name`, and an
  `evalshift-version` pin of 0.14.0 or newer.

- `evalshift login` reuses a working stored token instead of minting a new one. Every
  browser approval mints a fresh personal token on the server and the CLI holds only one
  at a time, so re-running `login` stranded the previous token — two "EvalShift CLI
  <host>" tokens after two logins. `login` now verifies the stored credential for the
  requested host first and reports "already logged in" when it still works; a 401/403
  falls through to the browser flow, other failures are reported as-is, and an explicit
  `--token` still always saves.

## [1.0.0] - 2026-09-18

### Changed

- **First stable release.** EvalShift now follows Semantic Versioning against a
  written contract: `evalshift.yaml`, command names and flags, exit codes, the
  documented artifact fields, and the run bundle are public surface, and a
  breaking change to any of them requires a major release. What is explicitly
  *not* public — the internals of `.evalshift/`, the `evalshift_cli` Python
  package, report markup, console wording — is listed too, so the boundary is
  checkable rather than implied. See
  [Compatibility and stability](DOCS.md#compatibility-and-stability). The config
  schema keeps its own evolution rule, unchanged: `version:` bumps only for
  breaking schema changes, additive fields ride the CLI version.

- **`evalshift all` is now `evalshift compare`.** The old name promised
  something the command never did: it runs all pipeline *stages*
  (`doctor → run → evaluate → analyze → report`) against **one** suite, so a
  project with several wired suites typed what looked like a complete command
  and got an error. `compare` says what it does — compare two models on a
  suite — and nobody expects a comparison to fan out.

  **`all` keeps working and is not scheduled for removal.** It is hidden from
  `--help` and prints a one-line notice on stderr (so CI parsing stdout is
  unaffected) pointing at the new name. Both names bind the same function, so
  there is no second code path. The alias is permanent because `evalshift init`
  writes `EVALSHIFT.md` into user repos telling agents to run `evalshift all`,
  and that file is never regenerated.

  Scripts, CI jobs, and scaffolded projects need no change. Docs, `--help`,
  `llms-full.txt`, the `EVALSHIFT.md` template, and the `init` / `capture sync`
  next-step hints all teach `compare` now.

- **Suite selection explains itself when it cannot pick one.** `run` and
  `compare` used to fail with a single line when `evalshift.yaml` wired more
  than one suite. They now print a framed panel that names the wired suites and
  lists one ready-to-run command per suite, rebuilt from the flags actually
  typed — so `evalshift compare --yes --push` is answered with
  `evalshift compare --yes --push --suite-name <suite>`, and a stale
  `--suite-name` is stripped rather than duplicated. Invoked under the legacy
  `all` name, the panel also spells out that `all` meant all *stages*, not all
  suites. An unknown `--suite-name` gets the same treatment, and `bundle` /
  `push` share it. Behaviour is unchanged: one suite per invocation,
  auto-selected when exactly one is wired.

- Docs no longer show a bare pipeline invocation in the hosted-push
  walkthroughs; every example names a suite, and `docs/getting-started.md`
  shows the shell loop for covering every suite.

## [0.15.0] - 2026-09-15

### Changed

- **License: AGPL-3.0-or-later → Apache-2.0.** The CLI is a client for a
  closed-source hosted service, so the copyleft protected nothing that mattered
  while tripping the blanket AGPL bans and dependency/SBOM scanners many
  companies run over their CI environments. Apache-2.0 drops that friction and
  adds an explicit patent grant. Nothing about how you use the tool changes,
  and the capture SDK (`evalshift-sdk`) was already MIT. Releases `0.4.0`
  through `0.14.1` remain available under AGPL-3.0-or-later.

## [0.14.1] - 2026-09-10

### Fixed

- `capture sync` now pairs a history tool result that was recorded without a
  `tool_call_id` with the preceding assistant turn's next unanswered tool
  call, in order, instead of assigning it a positional `_pos<N>` id. Gemini
  puts no ids on function responses, so every promoted Gemini conversation
  with a tool round carried an id nothing could match and LiteLLM rejected
  the replay of every later turn on both models (`Missing corresponding tool
  call for tool response message`) before it reached the provider. Re-run
  `evalshift capture sync` to regenerate affected suites. A result with no
  preceding call to answer still gets `_pos<N>` with a warning.
- LiteLLM's "Give Feedback / Get Help" banner, which the library prints on
  every failed call outside any logger, no longer floods the pipeline output.

## [0.14.0] - 2026-09-10

### Added

- `defaults.samples_per_example` (default `1`, max `20`): send every
  `(prompt, example)` to each model N times. Each sample is its own live call
  (`raw.jsonl` rows carry `sample_index`; the cache key includes it, so a
  single-sample run keeps every existing cache entry and resume skips per
  sample). `evaluate` scores source sample *i* against target sample *i* and
  folds the samples of one example into one `scores.jsonl` row: the mean
  `source_score` / `target_score` / `delta` over the samples that scored, the
  per-sample lists and the population `delta_variance` under
  `metadata.samples`, and the explanation prefixed `mean of k samples`. The
  paired tests still run over examples, so `n` is unchanged. The report shows
  an `N samples per example` pill, `report.json` carries
  `samples_per_example`, the non-determinism banner suggests the setting on a
  single-sample run, and example rows (report, bundle, insights, `inspect`)
  show sample 0. The cost estimate and pre-flight call count multiply by N.
- Teacher-forced multi-round replay. `capture promote` / `capture sync
  --rounds all` now carry the recorded tool results on the promoted example as
  `tool_result_fixtures` — one inner list per covered round, positionally
  aligned with `expected_tool_rounds`, each entry `{tool_name, result, error}`
  — pairing each round's calls with that round's `tool_result` events by
  `call_id` first and by tool name within the round second. `evalshift run`
  then replays such an example round by round: round *k* is sent the prompt
  (and any `history` prefix) followed by the *recorded* rounds `1..k-1` as
  assistant tool calls and `tool` results, so source, target and the recording
  all see identical context; the candidate's own calls are never fed back. The
  replay covers every round the fixtures cover plus the round after it, which
  — when every tool round is covered — is the answer round, where the recorded
  agent called nothing and produced its final text. Fixture coverage stops at
  the first round with a call that has no recorded result, with a warning
  naming the rounds the replay will cover. One `raw.jsonl` row per example per
  model as before: tokens, cost and latency summed, `text` the last round's
  answer, the `ToolTrace` carrying `round_count` and a `round_index` on every
  call. A model error in round *k* fails the example as `round k/n: …`.
  `tool_selection`, `tool_arguments` and `tool_trace_structure` score each
  round against its own ground truth (conformance against
  `expected_tool_rounds[k]`, "called nothing" for the answer round; divergence
  and argument pairing within a round) and record the mean over replayed
  rounds with per-round detail under `metadata.rounds`, so
  `max_tool_divergence` counts an example as diverged when any round diverged.
  The HTML report and `report.json` show one line per round in the tools
  column and prefix trace-diff items with `Round k:`; bundle trace events now
  carry the real `round`. The cost pre-flight counts one call per replayed
  round. Suites without the field (every suite written before it, and every
  `--rounds first` promotion) replay single-shot exactly as before.
- `cache_key` accepts a `round_index` (hashed only when set), so tool-call
  caching can land later without a cache migration; the tool path still
  bypasses the cache.

- Runs now record which generation parameters the models cannot honour, instead
  of `drop_params: True` making them vanish. A promoted capture can pin the
  generation config its original call used (`temperature`, `top_p`,
  `response_format` / `response_mime_type` / `response_schema`, `max_tokens` /
  `max_output_tokens`, `tool_choice` / `tool_config`, `parallel_tool_calls`),
  and the replay sends it — but LiteLLM's `drop_params` lets a model that never
  accepted one of those answer anyway, minus the constraint, so the arm
  measured a model change *plus* a missing constraint with nothing saying so.
  At run start EvalShift now asks LiteLLM (`models.capabilities.unsupported_params`,
  the generalisation of the existing `honors_temperature` probe) which of the
  parameters the suite actually recorded each arm supports, mapping provider
  spellings to their OpenAI names first. Anything a model positively lacks
  lands in `state.json` under `dropped_params` (`model id → [param, …]`), is
  logged once per (model, parameter) at `WARNING` rather than per call, reaches
  `report.json` as `dropped_params`, and renders as a **Constraints not
  honoured** banner beside the sampling banner in the HTML report. Uncertainty
  reads as "supported" — an exception, a `None`, or an empty answer records
  nothing — on the same reasoning as the sampling probe: a false banner on
  every report costs more than one missed warning. `temperature` stays with
  `non_deterministic_models`, which owns its own banner and probes
  unconditionally. Calls are unaffected: `drop_params` is still on, so nothing
  that used to succeed now fails.
  Alongside the probe, a small hard-coded table
  (`models.capabilities._KNOWN_LITELLM_GAPS`) covers what the probe cannot see:
  parameters LiteLLM *reports as supported* and then discards while building
  the provider's request body. Verified against litellm 1.100.0, that is two
  Gemini cases — `parallel_tool_calls` (filtered out against
  `GenerationConfig`'s fields) and a tool's `strict` flag (dropped by
  `_map_function`, since Gemini function declarations have no strict mode) —
  and both now land in `dropped_params` for a Gemini arm whose suite recorded
  them, where previously they only produced a per-dispatch log line. The tool
  flag is recorded under the pseudo-parameter name **`tools.strict`**, because
  it is a field on the `tools` array rather than a generation parameter. Probe
  and table merge into one sorted list per model. The two dispatch-time
  warnings in `models/client.py` are gone, since the run-start record now says
  the same thing and reaches the report and the policy; a `tool_choice` on a
  tool-less example is still stripped with a warning at dispatch and is
  deliberately *not* recorded as a dropped parameter — it is a fact about the
  suite, not about either target.
- `migration_policy.fail_on_dropped_params` (bool, default `false`) turns that
  record into a gate: when set and `dropped_params` is non-empty, the verdict
  is `fail` with a reason naming each model and parameter, whatever the scores
  said. For suites where the constraint *is* the contract — captures that
  pinned `response_format` measure nothing useful against a target that will
  not produce structured output. Top-level only (a model either accepts a
  parameter or does not, which no slice can vary), and runs recorded before
  `dropped_params` existed are never failed by it.
- Trace models accept `requested_tool_calls` on a `model_call` event — the tool
  calls the model asked for *in its response*, as
  `{name, arguments, call_id}` entries. It sits alongside the two notions that
  already existed and is none of them: `toolset_ref` / `tools_offered` is what
  was **offered** to the model, the `tool_call` / `tool_result` events are what
  the app **executed**, and this is what was **requested**. `null` (the default)
  means the trace predates the field; `[]` means the model requested no tools.
  The capture reader gates on the schema *major* only, so a capture written at
  the SDK's new `2.1.0` schema loads unchanged. Trace models are `extra="forbid"`,
  so the CLI has to accept the field before any SDK writes it.
- `capture promote` / `capture sync` now use those model-requested calls as the
  tool-call ground truth whenever a capture carries them, and record which
  yardstick a case used as `promotion_source` (`"requested"` | `"executed"`,
  default `"executed"`) on the promoted-case file. The executed `tool_call`
  events have already passed through the application — its filtering, retries,
  re-ordering, and its own function signatures — so they show what the *app*
  did, while a golden case has to state what a *model* should produce. On the
  requested path each `model_call` is one round (rounds that requested nothing
  are dropped, exactly as tool-less executed rounds are) and arguments are
  carried verbatim: wrapper unwrapping never runs on them, because nothing
  stands between the model and its own requested call. Every `model_call` in a
  run has to carry the field for the capture to be scored against requested
  calls — pass `[]` for a round in which the model requested no tools, since
  `null` means *not recorded* rather than *nothing requested*. Fallback to the
  executed calls is silent for a capture that predates the field, and warns for
  one where only *some* `model_call` events carry it (the whole capture falls
  back rather than mixing yardsticks). When requested and executed calls
  disagree, the requested ones win and promotion warns, naming the tools on
  both sides.
- Replay now carries the tool-choice constraints production used, instead of
  debug-logging and dropping them. A recorded `tool_choice` reaches the target
  in whichever of the three spellings the capture holds — an OpenAI string
  (`"auto"` / `"none"` / `"required"`) or object, an Anthropic object
  (`{"type": "auto"|"any"|"tool", "name"?, "disable_parallel_tool_use"?}`), or
  Gemini's `tool_config` (`function_calling_config.mode`, with
  `allowed_function_names`) — all normalised to one OpenAI-style intent plus a
  `parallel_tool_calls` bool, which LiteLLM then maps onto each provider's own
  shape (Anthropic's `tool_choice` object carrying `disable_parallel_tool_use`,
  Gemini's `toolConfig`). Normalising rather than passing through is what lets a
  capture recorded against one provider replay meaningfully against a target on
  another. An Anthropic `disable_parallel_tool_use: true` becomes
  `parallel_tool_calls: false`; an explicit top-level `parallel_tool_calls`
  wins over the inferred one.
- `ToolSpec` gained `strict`, so a toolset sidecar or inline `tools` entry
  carrying `strict: true` (top-level in the canonical/Anthropic shape,
  `function.strict` in the OpenAI shape) is replayed instead of being rejected
  as an unknown key. Both serialisers emit it only when set, so non-strict tools
  serialise byte-identically to before and toolset fingerprints are unchanged.
- Generation-config keys the runner cannot translate are now logged at
  **warning** rather than debug — once per distinct key set, so a whole-suite
  replay says it once. Constraints a target provider genuinely cannot express
  are named rather than dropped in silence: a Gemini target warns for
  `parallel_tool_calls` (`generateContent` has no such switch) and for a tool's
  `strict` flag (Gemini function declarations have no strict mode), once per
  model and key. A `tool_choice` that reaches an example with no toolset is
  dropped with a warning — there is nothing to constrain.
- `doctor` row `evalshift-sdk`: reports the SDK version the `evalshift` import
  name resolves to in this environment; `warn` (never a failure) when the SDK
  is missing, fails to import, or is shadowed by an older CLI's leftover files
  or a local `evalshift/` directory.
- Judge-family warning (`evalshift_cli.models.family`): when an `llm_judge`
  `judge_model` resolves to the same provider as `defaults.source_model` or
  `target_model`, `doctor` prints a warn-level `judge family` row (one per
  distinct judge; `ok` "from a third family" when none overlaps; no row when
  either arm is unset) and `validate` prints the same line after its success
  line — LLM judges prefer their own relatives' output (self-preference bias),
  so verdicts lean toward that arm. Advisory, never a failure: `init` scaffolds
  a same-provider judge on purpose. The report repeats the note as a third
  banner, only for judges that actually contributed `scores.jsonl` rows, and
  `report.json` carries it as `judge_family_overlap`. Provider `other` (an id
  the registry cannot place) never matches.
- CI pin-drift check (`evalshift_cli.utils.ci_pin`): `capture sync`, `init` (without
  `--ci`), `doctor` (new `ci pin` row), and `validate` now parse
  `.github/workflows/*.yml` for `babaliauskas/evalshift-action` steps and warn
  when the `evalshift-version` pin is older than the local CLI (`stale`),
  absent (`unpinned` — the action default may lag), or newer than the local CLI
  (`ahead`), printing the exact `evalshift-version: "<v>"` line to set. Advisory
  only: the CLI never edits a workflow and exit codes are unchanged. Rationale:
  `extra="forbid"` config means the reader in CI must be at least as new as the
  writer locally.
- `packaging>=23` is now a declared dependency (version comparison).
- `examples/capture-first/` — the first example to use the managed `suites:`
  block. It checks in the whole capture-first flow: an SDK-instrumented agent,
  the three captures and the toolset sidecar it recorded, the promoted cases and
  `golden.jsonl` that `evalshift capture sync` wrote, and the unedited
  `evalshift.yaml` from `init` with sync's derived `tool_selection` /
  `tool_arguments` block filled in. Its own `.gitignore` shows how a project
  commits `.evalshift/suites/` and `.evalshift/toolsets/` while keeping runs and
  the cache ignored.
- Promoted case files now record what the captured run cost: `cost_usd` (the
  run's `model_call` events summed) and `cost_source`. The SDK never prices
  anything — a `model_call`'s `cost_usd` is `0.0` unless the app's own
  instrumentation set it, and the provider client wrappers record tokens but no
  cost by design — so `capture promote` / `capture sync` now price each event
  that recorded tokens but no cost from litellm's price table for its own
  `model_id`, tagging the case `cost_source: "estimated"`. A recorded non-zero
  cost is kept as recorded (`"recorded"`), never re-estimated. A model litellm
  does not price — local / self-hosted, the normal case for open-source models
  — stays at `0.0` with no tag and no warning; the pricer is never called for
  it, so `capture sync` neither prints litellm's provider banner nor opens a
  socket to a local Ollama daemon. Existing case files parse unchanged
  (both fields default), and the `golden.jsonl` example never carries the
  figure.

### Fixed

- LiteLLM warnings no longer print in the middle of the `evalshift all`
  pipeline block (and again in the deferred-warnings section) on
  litellm >= 1.100. That release routes records below WARNING to `sys.stdout`
  by re-pointing its handler's stream per record; because only `sys.stderr`
  and `sys.__stderr__` counted as console streams, a single INFO record left
  the handler unrecognised and `deferred_console_warnings()` stopped
  detaching it. `sys.stdout`/`sys.__stdout__` now count too.

### Changed

- `--rounds all` no longer flattens every recorded round into `expected_tools`.
  `expected_tools` is now `expected_tool_rounds[0]` under both settings, and
  `--rounds all` means teacher-forced multi-round replay instead (see *Added*).
  The flattened list was only ever right for comparing against an externally
  produced multi-round trace, which the `agent_trace` evaluator does from
  imported traces. `--tool-count` under `--rounds all` pins the total over the
  rounds the replay reaches rather than over every recorded round.
- **Breaking (packaging):** the CLI's import package is now `evalshift_cli`.
  The distribution (`evalshift`) and the `evalshift` command are unchanged.
  The import name `evalshift` belongs to the capture SDK, which the CLI now
  declares as a dependency (`evalshift-sdk>=0.3.0`), so both install into one
  environment and `pip install evalshift` brings the SDK with it — the
  two-virtualenv rule is gone from every install page. What a user can
  notice: `python -m evalshift` is now `python -m evalshift_cli`, and scripts
  that imported CLI internals (`from evalshift.models.client import …`) must
  import from `evalshift_cli`. No shim is possible — shipping any `evalshift/`
  file would recreate the collision — so this ships as a minor bump (0.14.0).
- Docs: the capture guides (`README.md`, `DOCS.md`, `docs/sdk.md`,
  `docs/getting-started.md`, `llms-full.txt`) now cover the SDK 0.4.0 provider
  client wrappers (`wrap_openai` / `wrap_anthropic` / `wrap_genai`), the
  `requested_tool_calls` promotion path and `capture sync --rounds`, and no
  longer tell users to install `evalshift-sdk` in a separate venv — the CLI
  depends on it since the `evalshift_cli` rename. The capture-first example
  README explains why its cases are `promotion_source: executed`.
- Documented the config version policy: `version: 1` bumps only for breaking
  changes; additive fields ride on the CLI version and the CI pin check is the
  mechanism that keeps CI's reader at least as new as the local writer.

## [0.13.1] - 2026-08-28

### Changed

- `evalshift init --ci` now scaffolds a production-shaped GitHub Actions
  workflow instead of a single-suite example: dynamic suite discovery under
  `.evalshift/suites/` (a project with no suites yet skips green), one matrix
  job per suite, a single `evalshift gate` join check for branch protection,
  `fail-on: policy` (the action's real default), `evalshift-version` pinned to
  the scaffolding CLI, a provider API key matching `--provider`, PR-only
  run cancellation so base-branch baselines survive, and the full setup
  checklist documented in the generated file's header.

## [0.13.0] - 2026-08-28

Initial public release.
