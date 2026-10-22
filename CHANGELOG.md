# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`captures.rounds` in `evalshift.yaml`.** `first` (the default) or `all` sets
  how many recorded agent rounds the cases `capture sync` promotes are
  replayed over. It was a `--rounds` flag only, so teacher-forced replay had
  to be repeated on every sync. With `first`, an agent that answers after a
  tool call is scored on the call alone, and `semantic` and `llm_judge` see
  no answer text. On a real converter suite that left them 2 of 17 rows.
  `capture sync --rounds` still overrides it for one run; `capture promote`
  takes only the flag. The setting applies to cases promoted from then on.
  When it is `all` and promoted cases still replay one round although their
  captures carry the tool results to replay more, sync says how many. It
  names the command that would upgrade them: `capture sync --force`, plus
  `--rounds all` or `--suite <S>` when the run used them. It also warns that
  `--force` rebuilds every promoted case whose capture is local, dropping
  hand edits and `provenance: reviewed` marks. Reviewed, stale and
  `promote --as` cases are not counted. An unreadable config now also says
  sync is running without `captures.rounds`. `init` scaffolds the key
  commented out. Older CLIs reject the key, so update a CI
  `evalshift-version:` pin first.
- **Bounded capture suites.** Two optional settings under `captures:`,
  `max_age: 14d` and `max_cases: 200`, together keep each captured suite a
  capped, route-balanced sample of recent production on
  `defaults.source_model`. Setting either one turns the feature on; without
  them `capture sync` keeps and promotes exactly what it did before.
- **Filter.** With either limit set, `capture sync` promotes only captures
  recorded on `defaults.source_model`, and with `max_age` set, only captures
  recorded within it. Model ids match on bare names: lower-cased, provider
  prefix and trailing snapshot date dropped, no alias lookup. A conversation
  turn that recorded no messages list of its own is promoted only when every
  earlier turn of its conversation is local, since its history is rebuilt
  from them (`not eligible: N missing earlier turns`, with a warning naming
  the likely causes: the fetch window or budget, the bucket's lifecycle rule,
  `capture clean`, turns written flat and promoted on another machine, the
  SDK's per-turn sample
  rate or its `max_captures` / `capture_ttl` eviction, or a `turn_index`
  numbered from 1). `max_age` alone caps
  nothing. `capture promote` ignores the limits.
- **Plan.** With `max_cases` set, sync never grows a suite past the cap and
  swaps stale cases (too old, or recorded on another model) for fresh
  captures as they arrive. A stale case leaves at once when a fresh capture of
  the same question is added in its place; if its fresh copy is not added,
  the stale case leaves only to make room, like any other stale case. So a
  suite never shrinks because its cases went stale (an over-cap suite is still
  trimmed to `max_cases`). A conversation enters whole when it fits; once no
  whole conversation fits, the slots left take the next one's leading turns,
  so one long conversation cannot leave a suite nearly empty. Cases a
  human marked `provenance: reviewed` are never retired; in a captured suite
  the mark goes in the case file (`.evalshift/suites/<suite>/<case>.json`),
  not in `golden.jsonl`, which sync regenerates. Sync knows a case by its
  file, so a renamed or copied case file is retired (or kept) on its own.
  The same captures give the same suite on any machine, and syncing again
  with nothing new changes no file.
  `max_cases` also trims suites sync did not build: suites promoted by hand
  with `capture promote`, and suites whose entry is `managed: false` (that
  flag freezes the YAML entry, not the case files). A suite whose case files
  cannot be read is left untouched, with a warning, and the other suites
  still sync. A suite planned only because its directory holds case files
  is wired into `suites:` only if it already has an entry there.
- **Guardrail.** If none of a suite's cases and captures was recorded on
  `defaults.source_model`, sync warns (`none of N case(s) and capture(s) were
  recorded on <model> …; not filtering by model — check
  defaults.source_model`) and does not filter that suite by model rather than
  empty it.
- **Fetch.** The limits also bound what the automatic fetch of
  `capture sync` and `capture list` downloads from `captures.store`:
  `max_age` skips older captures, and `max_cases` takes at most
  `3 × max_cases` captures per suite inside that window, newest first
  (`skipped 2400 beyond
  each suite's newest 600`). Captures already local or promoted count toward
  that budget, so each sync downloads at most that many per suite, however
  many captures the bucket holds, and a re-sync with nothing new downloads
  nothing. Toolset sidecars are always fetched; `--since` replaces the age
  cutoff, not the budget; `capture fetch` applies neither limit.
- **Output.** Sync prints one line per suite: `alpha: 200 cases — kept 150,
  retired 50 (32 older than 14d, 18 on another model), added 50 of 2940
  eligible; not eligible: …`. With `max_cases` set, a capture's build
  warnings print only when it becomes a case.
- `doctor` gains `suite freshness` rows for each `source: captured` suite when
  either limit is set: `ok` (`alpha: 200 cases, captured 2d–12d ago`), or one
  `warn` per finding — the newest case is older than `max_age`, cases were
  recorded on another model than `defaults.source_model` (or none on it), the
  suite is over `max_cases`, or case files could not be read. A
  `captures limits` warning fires when `captures.store` is set and
  `captures.max_cases` is not (`suites grow with every capture sync; set
  captures.max_cases and captures.max_age`). `compare` prints the same
  findings above the verdict for the suite it runs, named with `--suite-name`
  or auto-selected (not for `--suite <path>`), so a team that only runs CI
  sees its suite went stale.
- Promoted case files record `captured_at` and `source_model`, with or without
  the limits. `capture sync` backfills them on existing cases whose capture is
  still local (`backfilled capture time and model on N case(s)`).
- **Conversation folders.** evalshift-sdk 0.7.0 files a
  `capture.agent_session` conversation's turns in one folder,
  `captures/<suite>/conv_<16 hex>/`, on disk and in a bucket. The CLI reads
  it beside the flat layout, which older SDKs, `EVALSHIFT_CAPTURE_LAYOUT=flat`,
  the `@capture.agent` decorator and standalone captures keep using, so
  existing buckets need no migration. Every fetch (`capture sync`'s,
  `capture list`'s and `capture fetch`) moves whole conversations: a
  conversation with a new turn brings its promoted turns along, whatever their
  age while the bucket still holds them, so a fresh checkout rebuilds the new
  turn's history instead of reporting it as missing earlier turns. The
  `max_cases` budget of sync's and list's fetch counts captures but takes a
  conversation at a time, newest first; once none fits whole, what is left
  takes the earliest turns of the next one whose turns that fit include one
  not yet promoted (one whose would all be promoted is passed over), and goes
  unused when none qualifies.

### Changed

- `evalshift init` scaffolds `captures: {max_age: 14d, max_cases: 200}` (with a
  commented `store:`). Existing configs are untouched. Without the two keys,
  `capture sync` keeps and promotes exactly what it did before.
- `capture sync` now reads `evalshift.yaml` under `--offline` too, for the
  limits. When the file cannot be read and has a top-level `captures:` key,
  sync says so (`could not read <path> (<reason>); syncing without
  captures.max_age / captures.max_cases.`), online and offline.
- The first `capture sync` on this CLI writes `captured_at` and
  `source_model` into case files, even without the limits, and older CLIs
  reject case files carrying them (and the new `captures` keys). Everyone
  who runs `capture` commands, teammates and CI alike, upgrades together:
  bump the CLI (and a pinned CI action's `evalshift-version`) everywhere
  before the first sync. `captures.max_age` accepts at most `36500d`; omit
  it for no age limit. CI's `compare` reads only `golden.jsonl`, whose
  format is unchanged. A run left in progress by an
  older CLI cannot be `--resume`d across the upgrade (config-hash mismatch);
  start a fresh one.
- `capture clean` (the default `--promoted` scope) keeps a conversation's
  promoted captures while any of its turns is unpromoted, since a later turn
  rebuilds its history from them, and says how many it kept. It removes the
  conversation folders it empties.
- `capture list <suite>`, `capture sync --suite`, `capture fetch --suite`,
  `capture clean SUITE` and `capture promote --suite` read the directory the
  SDK writes the suite to, so `team/support` finds `captures/team_support/`.

### Fixed

- Tool-calling replays work again on a fresh install. From 1.106.0, LiteLLM
  imports its MCP proxy module on every completion that carries tools, and
  that module needs the `mcp` package, which only LiteLLM's proxy extra
  installs. So every tool-calling call failed with `ModelError` (`No module
  named 'mcp'`) before any request was sent, and every retry failed the same
  way. The dependency is now `litellm>=1.77,<1.106`.
- The report's narrative no longer calls a reply that called tools an "empty
  output". A first-round tool replay ends at the tool call, so its text is
  empty, and the narrative model was shown that blank. A real run's findings
  read "the target model returned empty outputs where the source model
  provided specific information" for replies that each called a tool. The
  regression samples now show a reply as its tool calls with their arguments,
  then any text, and the instruction says a tool call is not an empty answer.
  Each call's arguments are capped at 200 characters and a sample lists at
  most 6 calls ("+N more call(s)"), so the text always survives the
  2000-character sample cap. Malformed arguments read `<malformed arguments>`,
  and a multi-round reply marks each call's round.
- The narrative no longer presents a run's total cost as a cost per call. The
  figures it is given were each model's whole-run total, under keys that did
  not say so, and the instruction's own example phrase was "cost per call".
  The same run read "the cost per call increased from $0.1045 to $0.1056", for
  21 calls each at about $0.005. The keys now name the totals, and the
  Economics panel and the templated fallback say "Total".
- A change that missed significance is no longer headlined as equivalent.
  Severity `none` means the corrected p-value missed 0.05, whatever the effect
  size, and the report read every such row as "✓ Equivalent — No meaningful
  difference between models": a −17% tool-selection change (Cohen's d −0.51,
  n = 21) got a green tick directly under a FAIL verdict. A `none` row now
  reads Equivalent only when |d| < 0.2. A larger effect renders as "~
  Unconfirmed drop" or "~ Unconfirmed gain", with its magnitude, in a warning
  colour. A zero-variance row whose pairs all moved by the same non-zero
  amount, where d is a placeholder 0 and a uniform −1.0 drop read "✓
  Equivalent" beside "−100%", now carries a `uniform drop:` / `uniform gain:`
  note from `analyze` and renders as "⚠ Uniform drop/gain". It also outranks
  other non-significant rows for a prompt's executive-summary row, where its
  placeholder d had lost to a tested d of −0.1. The `analyze` terminal table
  uses the same reading for its glyph, and `compare` adds a callout under "~
  no significant change" naming each sub-metric that dropped on every pair.
  These are display only: severities and verdicts are unchanged. The note is
  written by `analyze`, so re-run `analyze` (or `compare`) on an existing run
  to get it; `report` alone reads the old `analysis.json`.
- A row where no test ran no longer reads "Unclear" in the Confidence
  column. When every tested delta is the same value, `analyze` skips the test
  and stores a placeholder p-value of 1.0, which the column read as "Unclear".
  A real converter run showed "Unclear" on four rows where both models made
  the same tool calls on all 17 pairs. Such a row now shows "—", its tooltip
  says no test ran, and an all-zero row reads "Equivalent — Both models
  scored the same on every pair."
- The executive summary no longer puts a ✓ on a "Ground truth missed by
  both" row. Its rows never got the shared-miss annotation the evaluator
  table's rows did, so a prompt whose summary row was that axis read "✓
  Equivalent" one table above "! Ground truth missed by both".
- An existing run's narrative regenerates on its next `report`, because the
  facts it is built from changed. Re-run `report` before `push` to replace an
  already-written narrative.
- The PyPI project page's links work. 35 of the README's 51 links were
  relative, so on pypi.org they resolved under `pypi.org/project/evalshift/`
  and 404ed; they now point at the files on GitHub. The project's Homepage
  and Documentation links now go to <https://www.evalshift.dev> and
  <https://www.evalshift.dev/docs/cli> instead of the GitHub repo.

## [1.5.0] - 2026-10-15

### Added

- `evalshift report` and `evalshift compare` end with a one-line request to star the
  project on GitHub, with the repo link. It prints on interactive terminals only, so CI
  logs, the GitHub Action's output and anything piped stay free of it.

## [1.4.3] - 2026-10-13

### Fixed

- Failure categories count examples, not scored rows. `tool_selection` stamps
  `TOOL_SELECTION_DRIFT` on both its conformance and its divergence row, so one example
  that picked another tool read "Different tools chosen (2 examples)" in the narrative
  and 2 in the report's top regression causes. Each label is now counted once per
  distinct example, as `INVARIANT_VIOLATION` already was.
- The cached run narrative no longer outlives the figures it describes. `insights.json`
  was keyed on the run's `config_hash`, which is fixed when the run is created, while
  `evaluate` and `analyze` re-read `evalshift.yaml` every time: loosening a budget and
  re-running `analyze` + `report` rendered a PASS verdict beside a cached "FAIL under the
  configured policy" paragraph. The key now also covers the narrative's facts (scores,
  analysis and the persisted decision), so any change to them regenerates it on the next
  `report`; a plain re-render still reuses it. Existing caches miss once and regenerate.

## [1.4.2] - 2026-10-12

### Fixed

- A `semantic` evaluator set to `blocking: true` no longer fails runs whose every row
  stays within `min_similarity`. The source is pinned at 1.0, so any rewording is a
  small negative delta, and the paired significance test ran on those raw deltas:
  nine rows at 0.90–0.98 tested as a `critical` regression and failed a run whose every
  budget passed. Rows within `min_similarity` now enter the test as no change — the
  rule the budgets already used — and the comparison notes how many. `delta_avg_score`
  and the aggregates still show the raw drift. Re-analysing an older run can lower its
  semantic severities.
- A fail caused by comparison severity alone now carries a `reason` naming each
  comparison, so the report no longer shows "7 of 7 budgets within policy" under FAIL.
  The templated recommendation names the actual cause instead of always claiming a
  budget breach — a breached budget, including a slice's own (`the 'security' slice
  overall regression rate budget was breached`), or else the overall blocking
  regressions — and the summary's critical count reads the blocking comparisons.
- Tool evaluators (including a top-level `trace_invariants` inherited by every suite) no
  longer attempt pairs that were offered no tools. On a tool-free suite they used to
  book every pair as unmeasured, demote a clean `pass` to `conditional_pass`, and blame
  "tool calls and no prose". They now skip those pairs, and `evaluate` (and `compare`,
  under its verdict) warns that the evaluator checked nothing because no example offered
  tools. Coverage is written by `evaluate`, so an older run needs `evaluate` re-run to
  drop the stale coverage; `analyze` alone keeps it, and the `conditional_pass` with it.
- The run-insights narrative no longer guesses why a run failed. The prompt listed each
  budget's limit but not its result, so a FAIL on the latency ceiling alone was explained
  as "failed to meet the budget for Trace-rule violations" — a budget that held. Every
  budget, slice rows included, now enters the prompt with its status, the breached ones
  are listed as `failed_budgets`, and the decision's `reason` travels with them in plain
  words, without `evalshift.yaml` keys. Only a breached budget may be named as the cause,
  and on a FAIL a summary or recommendation that blames a budget that held is rejected
  and retried, then replaced by the templated prose. Naming a held budget to clear it
  ("Cost increase stayed within its limit", "not Cost increase") is not blame, and
  neither is a figure "over the source". Narratives are cached on the run's
  config hash and model, not the CLI version: delete a run's `insights.json` and re-run
  `report` to rewrite an existing one.

## [1.4.1] - 2026-10-11

### Fixed

- `semantic` no longer reads formatting as drift. Before embedding, both
  outputs drop a code fence wrapping the whole output, and JSON objects and
  arrays are re-serialized canonically (sorted keys). A source that fences
  pretty-printed JSON against a target returning the same JSON compact used
  to score ~0.73 and get flagged; it now scores 1.0. Semantic scores for
  structured outputs change on upgrade — expect higher similarity where the
  models only differed in layout.

## [1.4.0] - 2026-10-11

### Added

- `capture sync --repair` rebuilds the `inputs` and `history` of cases an
  earlier sync built from stringified message blocks, keeping every other
  field and the case's input variable name, and re-downloads exactly the
  captures it needs from `captures.store` even though they are promoted. It
  also rebuilds cases an earlier sync promoted with empty `inputs` (OpenAI
  `exclude_none`, Responses, LangChain), which get no warning.
  `capture sync` and `validate` warn when a suite still holds stringified
  cases; without `--repair`, sync leaves both kinds untouched. A repaired case that now
  duplicates another promoted case is named
  (`repaired case <a> duplicates <b>`) and nothing is deleted.
- `evalshift init` writes the `semantic` evaluator active for every provider.
  Anthropic and DeepSeek have no embeddings endpoint, so the scaffold borrows
  OpenAI's or Gemini's — whichever key you already have — and `init --ci`
  wires that key as a second secret.
- `compare`, `run` and `evaluate` check every judge and embedding model's API
  key before the first call. An advisory evaluator without one is skipped with
  a line naming the env var to export; a blocking one stops the run, and so
  does a config whose every evaluator would be skipped. A `semantic` that is a
  gate only because a blocking `tool_arguments` entry uses its embedder is
  reported as such, naming that entry. The skip
  is recorded in `state.json` (`skipped_evaluators`) and repeated in the
  verdict's recommendations, the HTML report and the bundle.
- `doctor` gains an `evaluator keys` row.
- When nothing gates the verdict, the recommendation now names the advisory
  judge and says, from its own sample size, whether it is ready for
  `blocking: true` (20 pairs per prompt) or how many examples it still needs.
  A judge that measured nothing (every call errored) is said to have measured
  nothing rather than told to collect more examples.

### Changed

- A `semantic`/`llm_judge` entry without `blocking: false` is a gate (the
  library default). If its model has no API key, `compare`, `run`, `evaluate`
  and `doctor` now exit 1 before any call — previously the run completed with
  an unmeasured axis, landed on `inconclusive` and passed `--policy-gate`, so
  CI that was green can turn red on upgrade. The bare `text-embedding-3-small`
  default is checked for the first time. Export the key, or set
  `blocking: false` to have it skipped with a warning.
- Docs: the SDK's object-store variable is `EVALSHIFT_CAPTURE_STORE` from SDK 0.6.0, matching
  `captures.store` here; SDK 0.5.0 reads the same value from `EVALSHIFT_SINK`, which keeps
  working on newer SDKs too.
- Every store message names the package to install instead of a pip extra: `pip install boto3`,
  `google-cloud-storage`, `azure-storage-blob azure-identity` (both Azure packages in one
  command). The `[s3]` / `[gcs]` / `[azure]` extras still exist but are no longer documented.

### Fixed

- `capture sync` and `capture promote` now read every wrapped provider's own
  message shape. Claude follow-up turns no longer get Python-repr strings
  (`"[{'id': 'toolu_1', … 'type': 'tool_use'}]"`) in their history: `tool_use`
  blocks become the assistant's `tool_calls`, `tool_result` blocks become `tool`
  messages, thinking is dropped, and a prompt-cached `system` prompt or a
  text-block user turn becomes plain text. OpenAI Chat histories with
  `content: null` no longer abort the whole sync with a traceback, and OpenAI
  Responses and LangChain captures recover their `inputs` instead of promoting
  them empty.
- A turn whose recorded messages carry an image, a document, audio, a
  server-tool result or an unknown block type is skipped (`capture promote`
  refuses it) instead of being promoted with that content stringified or lost;
  `--allow-errored` does not apply.
- An OpenAI `developer` message is now kept as the system prompt instead of
  being dropped: it merges with a leading system prompt (a Responses
  `instructions=` included), and a system or `developer` message after the
  first turn is dropped with a warning instead of aborting the sync; run
  `capture sync --force` to refresh cases promoted before.
- A capture no valid case can be built from is skipped with the reason
  (`capture promote` refuses it) instead of aborting the whole `capture sync`
  with a validation traceback; `--allow-errored` does not apply.
- A judge or embedding model with no API key no longer fails silently on every
  pair: it is skipped (advisory) or refused (blocking) before any call.
- The bare `text-embedding-3-small` default now resolves to OpenAI, so its key
  is checked.
- `doctor`'s `captures.store` row no longer fails the command — and with it `compare` and
  every GitHub Action run — when the store's client library is not installed; it warns, naming
  the package to install. A run on a committed suite never reads the bucket; `capture fetch`,
  `list` and `sync`, which do, still exit 1.
- A suspended account is named as such. When the server refuses a token with
  `account_suspended`, the CLI prints its message — "Your account has been suspended.
  Contact support@evalshift.dev." `evalshift login` with a stored token for that account
  exits 1 with it instead of calling the token no longer valid and starting a browser
  sign-in, and push auto-create shows it without the token-access hint. Service account
  keys are not affected by a user's suspension.

## [1.3.0] - 2026-10-09

### Added

- `captures.store` in `evalshift.yaml`: name the object store the capture SDK
  ships captures to — `s3://<bucket>/<prefix>`, `gs://<bucket>/<prefix>` or
  `az://<account>/<container>/<prefix>` — and `capture sync` and `capture list`
  mirror new captures and toolset sidecars from it into `.evalshift/` before
  they run, with no flags. Only objects missing locally are downloaded
  (captures are immutable, toolsets content-addressed), captures already
  promoted are skipped so `capture clean` never causes a re-download, and
  every file is written atomically. New `capture fetch` does only that step;
  `--since 30m|24h|7d|<ISO>` limits a fetch to recently written captures and
  `--offline` skips it. Keys from a bucket are validated against the SDK's
  layout and anything else is ignored as malformed — including a suite
  directory containing `:`, `/`, `\`, NUL or `..`, or starting or ending with
  `.` or a space, and a key with a trailing newline — and no destination can
  land outside `.evalshift/`; folder-marker keys ending in `/` (the S3
  console's "Create folder", ADLS directories) are skipped without being
  counted. A list or download failure exits 1 with a hint naming the
  provider's credential chain (`aws sso login` / `AWS_PROFILE` / the CI OIDC
  role, `gcloud auth application-default login`, `az login`); a local write
  failure names the file and asks you to check disk space and permissions on
  its directory instead. Extras:
  `evalshift[s3]`, `[gcs]`, `[azure]` (azure-storage-blob and azure-identity);
  a missing one exits 1 naming the `pip install` to run. Credentials come from
  each provider's default chain and are rejected in the URI: a value with `@`
  or `?` fails to load, naming the accepted forms. No rejected value is echoed
  back (it may be a pasted key or connection string); at most its scheme is
  named.
  `capture list --json` keeps stdout pure JSON (the fetch line and warnings
  go to stderr). `doctor` reports whether the store is installed (`fail`, exit 1,
  when the extra is missing) and reachable (`warn` when it cannot be listed).
  Without `captures.store` nothing changes; a `sync` or `list` whose
  `evalshift.yaml` cannot be read or does not validate skips the fetch,
  warning only when the file has a top-level `captures:` key. Older CLIs
  reject the `captures:` key, so bump the CLI (and a pinned CI action's
  `evalshift-version`) before adding it.
- `evaluators.trace_invariants`: hand-written rules over tool-call traces —
  `forbidden`, `required`, `order`, `call_count` (`min_calls` and/or
  `max_calls` per tool; equal bounds = an exact count) and
  `arguments` (JSON Schema) — checked on both sides of every pair, on replayed
  traces or (`traces: imported`) on imported ones. A rule the target breaks
  counts whatever the source did. Its `applies_to` is enforced: a prompt
  outside the globs is never checked, and `evaluate` warns, naming the entry
  and its `applies_to`, when the globs matched nothing in the run (so a typo
  is not a silent clean run). A `traces: imported` entry on a run without
  `traces.jsonl` fails `evaluate` with an error naming that family.
- `migration_policy.max_invariant_violations` (default `0`, per-slice
  overridable): distinct examples on which the target broke a blocking trace
  rule, so an example that broke rules in two entries counts once. A breach
  fails the run even when the suite is too small for statistics, and a
  slice whose own budget breaches reads `fail` however few comparisons it has.
  `analyze` adds a recommendation when the source broke the same rules the
  target did: review the rule with its owner rather than raising the budget.
  Pushed bundles' `decision.policy` now always carries the field; hosted
  EvalShift accepts and stores it since 2026-10-08 (it never re-evaluates it —
  a run that carries its own policy is answered with the CLI's decision).
- The HTML report lists every broken trace rule in a "Trace rules broken"
  panel, with its owner and whether the source broke it too — advisory
  (`blocking: false`) entries included, though only blocking ones count
  toward the budget; those rows are tagged "advisory". `report.json` carries
  the same rows under a new top-level `invariant_violations` key, each with
  the entry's `blocking` flag. The "Overall, by evaluator" table tags every
  advisory (`blocking: false`) evaluator's row the same way and says what a
  trace-rule row measures, and the "Top
  regression causes" panel counts `INVARIANT_VIOLATION` as the budget does —
  distinct examples with a blocking target violation — so the narrative never
  quotes a figure the verdict did not use.
- `evalshift init` scaffolds `max_invariant_violations: 0` in every
  `--profile` policy block, beside the other budgets. Older CLIs reject the
  field, so bump the CI action's `evalshift-version` pin to this release
  before committing a freshly scaffolded config.
- `capture sync` keeps a managed entry's hand-written
  `evaluators.trace_invariants` block when it regenerates the entry. The
  marker comment still says hand edits are overwritten; this block is the one
  exception.
- `push` now uploads each score row's one-line `explanation` and, for
  `trace_invariants` rows, the rules each side broke (`violations.source` /
  `violations.target`: rule id, type, tool, round, detail, and sample on
  repeated-sampling runs). Hosted EvalShift shows them on the example detail
  and on share links instead of only the violation count. `rules_checked`,
  `owner` and `failure_categories` are not repeated on the score row (they
  already ship in `evaluator_config` / `decision`). Requires a hosted service
  that accepts the two fields (hosted EvalShift was updated before this
  release); nothing else in the bundle changed.

### Changed

- The report's per-prompt Executive summary — and the header's mean score Δ,
  which averages its rows — now picks each prompt's worst severity with ties
  broken toward the most negative effect size (the order the hero panel and
  the narrative already used) instead of whichever evaluator `analyze` listed
  first, and skips advisory (`blocking: false`) evaluators while the prompt
  has a gating one. An advisory rule the target satisfied by calling no tools
  at all could otherwise read as the prompt's "+9% improved" above a failed
  verdict.
- Upgrading moves the resume `config_hash`: the new config fields are part of
  every config's canonical dump, so a run left in progress by an older CLI
  cannot be `--resume`d after upgrading (it fails with the usual config-hash
  mismatch). Start a fresh run; cached responses are reused.

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
