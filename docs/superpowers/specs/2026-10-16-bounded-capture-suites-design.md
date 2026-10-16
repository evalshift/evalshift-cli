# Bounded capture suites: `captures.max_age` + `captures.max_cases`

**Date:** 2026-10-16 · **Status:** approach A approved in chat, spec for maintainer review · **Repos:** evalshift-cli (behaviour + docs); evalshift-sdk, evalshift-action, evalshift-server, evalshift-client (nothing — follow-ups recorded)

## Problem

A team that ships captures to a bucket (`EVALSHIFT_CAPTURE_STORE`) and switches
`EVALSHIFT_CAPTURE=1` on in production for a couple of days at a time writes thousands of
objects per agent each time. The SDK's 200-newest-per-suite bound applies to local disk only
(`docs/sdk.md:106-120`); the bucket keeps everything a lifecycle rule does not expire. Today:

1. **One sync promotes everything.** `capture sync` promotes every local capture that survives
   the no-events / blocked / errored / duplicate filters (`cli/commands/capture.py:1213-1318`), and
   the fetch downloads every object not already promoted (`captures/remote.py:297-335`). Nothing
   bounds a suite, so every CI run replays thousands of rows on both models plus the judge.
2. **Suites only grow.** No code deletes a promoted case — `capture clean` unlinks capture files
   and orphan sidecars only (`capture.py:830, 898`). Week 2's sync adds to week 1's.
3. **Stale cases are invisible.** A case's ground truth is what the production model did when the
   capture was recorded, but neither `PromotedCase` nor `SuiteExample` records when that was or on
   which model (`captures/models.py:53-93`). `created_at` is never parsed (a string tie-break only,
   `captures/promote.py:1308`) and `model_id` is read only for pricing (`promote.py:552`). After a
   production migration the old-model cases keep gating CI, and a team that only runs CI — which
   reads `golden.jsonl` and never touches the bucket — never learns its suite is stale.
4. **An empty checkout re-promotes the whole bucket.** The promoted set is derived only from the
   committed case files (`captures/reader.py:410-423`). With those gone, every object is downloaded
   and promoted again, however old.

## Goals

- A suite is a **bounded, deterministic, route-balanced sample of recent production on the current
  source model**, refreshed deliberately by `capture sync` + a reviewed PR.
- Staleness is **visible where CI-only teams look**: `doctor` and `compare` output.
- A config that does not set the new keys behaves exactly as 1.5.0.

Non-goals: the CLI deleting from the bucket (it stays read-only on the store); CI syncing on its
own; freshness in `report.html` / the bundle / the PR comment (follow-up, cross-repo).

## Decision

### 1. Config

```yaml
captures:
  store: s3://acme-evals/support-agent   # optional, unchanged
  max_age: 14d                           # optional
  max_cases: 200                         # optional, per suite
```

- `max_age` — a duration `N` + `m|h|d` (the relative grammar `--since` already accepts), parsed to a
  `timedelta`, must be > 0. A new `utils/duration.py::parse_duration(text) -> timedelta` owns the
  grammar; `captures/remote.py::parse_since` is refactored onto it and keeps its `--since` error
  text. The config field uses an `Annotated[timedelta, BeforeValidator(...)]`, so a typo fails as
  `captures.max_age: expected a duration like 30m, 24h or 14d`.
- `max_cases` — `int`, `ge=1`.
- Both default to `None`, meaning today's behaviour: no eligibility filter, no cap, no retirement.
- **Setting either one turns bounded suites on** — the eligibility filters (§3), the plan (§5) and
  the freshness findings (§7). `defaults.source_model` alone changes nothing: it is set in nearly
  every config, and keying model filtering on it would switch the feature on for every existing
  user.
- Both are valid without `store` — they govern how captures become suites, from disk or bucket.
- No per-suite override in v1: the managed `suites:` region is regenerated on every sync and only
  `evaluators.trace_invariants` is carried forward (`capture.py:981-1005`). Follow-up.
- Nothing is added to `defaults` — the hosted `eval_config_hash` hashes `defaults`
  (`hosted/bundle.py:258-290`), so a new field there would orphan every hosted baseline. The
  `captures` block is not hashed.

### 2. What a promoted case remembers

`PromotedCase` gains two fields (it is `extra="forbid"`, see Compatibility):

| Field | Value |
| --- | --- |
| `captured_at: str = ""` | the envelope's `created_at`, verbatim (the SDK writes ISO 8601 UTC) |
| `source_model: str = ""` | `model_id` of the capture's **first** `model_call` — the call whose input the replay re-sends, the same event that already supplies `toolset_ref` and `generation_config` (`promote.py:299-301, 457-461`) |

`capture sync` and `capture promote` write both. `SuiteExample` is **not** changed, so
`golden.jsonl` stays byte-compatible with the (possibly older) CLI that CI runs.

**Backfill.** When sync sees an existing case with an empty field whose capture is local, it fills
the field and rewrites the case file; nothing else on the case changes (`promoted_at` included).

**Legacy cases.** No `captured_at` → its age is taken from `promoted_at`, an upper bound on capture
time: the case looks younger than it is, so it is retired later, never earlier. Neither timestamp
parseable → age unknown → never age-stale. No `source_model` → never model-stale.

### 3. Staleness and eligibility

One clock instant `now` per sync run (a module-level `_now()` so tests can pin it).

- **Age-stale** (a kept case) / **too old** (a candidate capture): captured before `now − max_age`.
  Only when `max_age` is set.
- **Model-stale** / **other model**: `source_model` is known and
  `same_model(source_model, defaults.source_model)` is false. Only when bounded suites are on (§1)
  and `defaults.source_model` is set.

`same_model(a, b)` — new `models/matching.py`, no config import. True when either:

1. the **bare names** are equal, where bare = lowercase, drop everything up to the last `/`, then
   strip one trailing snapshot date `-YYYYMMDD` or `-YYYY-MM-DD`; or
2. `resolve_model(a).id == resolve_model(b).id` (registry aliases).

| a | b | match | via |
| --- | --- | --- | --- |
| `claude-sonnet-5-20260301` | `claude-sonnet-5` | yes | bare |
| `anthropic/claude-sonnet-5` | `claude-sonnet-5` | yes | bare |
| `google/gemini-3.8-flash` | `gemini-3.8-flash` | yes | bare |
| `gpt-5-2026-08-07` | `openai/gpt-5` | yes | bare |
| `claude-sonnet-5` | `claude-sonnet-5-5` | **no** | — |

**Guardrail.** Per suite: when at least one kept case or candidate has a known model and every one
that does would be model-stale, the model check is switched off for that suite and sync warns
`⚠ planner: none of 212 cases/captures were recorded on claude-sonnet-5-5 (seen: claude-sonnet-5-20260301 ×212); not filtering by model — check defaults.source_model`.
A wrong or unrecognised `source_model` can therefore never empty or freeze a suite. The same rule
holds mid-rollout: until at least one new-model capture exists, nothing is filtered.

**Eligible candidate** = a local capture that is not promoted, survives today's filters (events,
blocked, errored, dedup — §6), and is neither too old nor on another model.

### 4. Fetch is bounded by time, not count

Effective fetch cutoff = `--since` when given, else `now − max_age` when set, else none — applied
to the object's `last_modified` exactly as `--since` is today (`remote.py:324-331`). Promoted ids
are still skipped.

The fetch does **not** sample: a listing yields only key, `last_modified` and size
(`remote.py:91-103`; capture ids are random `cap_<uuid4 hex>`), so conversation, route and model are
unknown until download — and a turn without its own messages list rebuilds its history from its
sibling captures on disk (`promote.py:1369-1399`), so a key-level sample would silently truncate
histories. Volume inside the window is governed by `EVALSHIFT_SAMPLE_RATE` in the agent. A hard
download cap is a follow-up.

### 5. The per-suite plan (only when `max_cases` is set)

New pure module `captures/selection.py` — no I/O, no clock:

```python
def plan_suite(
    kept: Sequence[KeptCase],
    candidates: Sequence[CandidateGroup],
    *,
    max_cases: int,
) -> SuitePlan: ...
```

`KeptCase`: name, captured instant (or `None`), stale reason (`None | "age" | "model"`), reviewed
flag, route, content key. `CandidateGroup`: group key, its turns (capture ids, in turn order),
route, content keys. `SuitePlan`: groups to add, `(case name, reason)` to retire, counts.

- **Unit = conversation group.** All candidate turns sharing a `conversation_id` within the suite;
  a capture without one is its own group. A group is added whole or not at all; one larger than
  the remaining slots is skipped and the next tried.
- **Route** of a case or group = the sorted set of tool names in its round-1 `expected_tools`
  across its turns, `∅` when none.
- **Order is deterministic, no RNG:** within a route, groups are taken in `sha256(group key)` order.
  Same inputs ⇒ same plan on any machine; rerunning sync with nothing new changes nothing.
- **Partition kept cases:** `P` = reviewed — any of the case's expected tool calls carries
  `provenance: reviewed` (never retired), `F` = fresh, `S` = stale.
- **Fill:** `slots = max(0, max_cases − |P| − |F|)`. Repeatedly pick the route with the fewest cases
  in `P ∪ F ∪ added` among routes that still have a group that fits (ties → route string), add its
  next group, until slots or groups run out.
- **Retire (replacement only):** `excess = |P| + |F| + |added| + |S| − max_cases`. While
  `excess > 0`, retire from `S` in this order: (a) a stale case whose content key equals an added
  turn's (its own replacement), (b) model-stale, (c) age-stale oldest first; ties by name. A stale
  case is thus only ever retired to make room — **a suite never shrinks because it went stale.**
- **Trim:** if `excess > 0` once `S` is exhausted (a legacy or over-cap suite), retire from `F`: the
  most-represented route each time, oldest first within it. `P` is never trimmed; if `|P| > max_cases`
  sync warns and keeps them.

Without `max_cases`: every eligible candidate is added (today's behaviour, minus too-old /
other-model captures when those filters are set), and nothing is retired.

### 6. Sync integration

The per-suite loop (`capture.py:1171-1321`) becomes **collect → plan → apply**:

1. Build examples as today with `build_conversation_examples` over every local record of the suite,
   promoted ones included, so history reconstruction keeps its siblings.
2. Classify each built record: already promoted (kept) / no events, blocked, errored (counted as
   today) / too old / other model / eligible.
3. Dedup: seed from **`P ∪ F` only**, not `S`, so a question re-asked on the new model can replace
   its stale twin (rule 5a); within-run first-wins as today. The final suite holds no duplicate
   content keys unless `--keep-duplicates`.
4. `plan_suite`.
5. Apply: write added cases (with the §2 fields), unlink retired case files (new
   `promote.retire_promoted_case`), backfill, then `rebuild_golden_jsonl` and
   `derive_suite_evaluators` as today. `--repair` runs as today; the plan sees the post-repair set.

**Which suites are planned:** every suite with local captures (as today) plus, when `max_cases` is
set, every `<base>/suites/<name>/` holding case files, so an over-cap suite is trimmed even with no
new captures. `--suite` narrows both. The "no captures found" early return applies only when no
limits are set.

**Config:** sync now loads `evalshift.yaml` even under `--offline` to read `captures.*` and
`defaults.source_model`. Missing or invalid config ⇒ no limits (today's behaviour), with the existing
warning when the text has a top-level `captures:` key (`capture.py:373-380`).

**Flags:** no new flags — the limits are durable settings and live in YAML. `--since` keeps
overriding the fetch window; `--force`, `--keep-duplicates`, `--allow-errored`, `--repair`, `--print`
keep their meaning.

**Output:** the existing summary line is unchanged. When limits are set, one line per planned suite:

```
  planner: 200 cases — kept 150, retired 50 (32 older than 14d, 18 on another model), added 50 of 2940 eligible; not eligible: 812 older than 14d, 40 on another model
```


### 7. Freshness signals for CI-only teams

New `captures/freshness.py::suite_freshness(suite_dir, *, max_age, max_cases, source_model, now) -> list[FreshnessFinding]`,
reading the case files beside `golden.jsonl` (they are committed with it). Computed only when
bounded suites are on (§1), and only for `source: captured` suites. Best-effort: unreadable
case files become one finding, never an exception. Findings:

- `stale_age` — the newest case is older than `max_age` (`planner: newest case captured 23d ago (max_age 14d); refresh with evalshift capture sync`).
- `stale_model` — `k of n cases recorded on claude-sonnet-5; defaults.source_model is claude-sonnet-5-5`.
- `over_cap` — `planner: 2,940 cases, over captures.max_cases 200`.

**doctor:** one `suite freshness` row per named captured suite with case files — `ok`
(`planner: 200 cases, captured 2–12 days ago`) or `warn` per finding — plus a `captures limits`
`warn` row when `store` is set and `max_cases` is not (`suites grow with every sync; set captures.max_cases and captures.max_age`). Never `fail`.

**compare:** compare discards doctor `warn` rows (`compare.py:551-552`), so it calls
`suite_freshness` itself for the suite it runs and prints each finding as a `⚠` line above the
verdict, beside the tool-free warnings (`compare.py:716-719`). That is the CI log.

### 8. `init`

New projects get the limits active (the scaffold currently writes no `captures:` block,
`cli/commands/init.py:111-173`):

```yaml
# How captures become suites. Each suite is a capped, route-balanced sample of
# production from the last max_age, on defaults.source_model; `capture sync`
# replaces stale cases as fresh captures arrive. Set store: if the SDK ships
# captures to a bucket (EVALSHIFT_CAPTURE_STORE).
captures:
  max_age: 14d
  max_cases: 200
  # store: s3://<bucket>/<prefix>
```

Existing configs are not rewritten; doctor's `captures limits` row nudges them.

### 9. Bucket retention (docs only)

`docs/configuration.md#captures` and `docs/sdk.md` recommend a lifecycle rule that expires capture
objects after a period ≥ `max_age` (e.g. 30 days for `max_age: 14d`) — it bounds storage and how
long production user data sits in the bucket — with an `aws s3api put-bucket-lifecycle-configuration`
example and one-line GCS / Azure pointers. The CLI stays read-only on the bucket.

## Trade-offs recorded

- **Coverage-weighted, not traffic-weighted.** Route balancing means rate budgets such as
  `max_overall_regression_rate` are measured over a balanced sample. Chosen because a rare tool path
  regressing is what a migration gate exists to catch; a traffic-weighted 200-case sample gives a
  1%-traffic route about two cases.
- **Replacement-only retirement** means a suite can stay stale indefinitely if production stops
  capturing. doctor and compare make that visible rather than emptying the suite.
- **Time-bounded fetch** downloads everything inside `max_age`, not just what the plan uses; correct
  for conversations, bounded by the window and the SDK sample rate.

## Not in scope (recorded follow-ups)

- Freshness in `report.html`, the bundle, the hosted run page and the PR comment (bundle field;
  evalshift-server + evalshift-action).
- Per-suite `max_cases` / `max_age` (carry-forward in the managed region).
- `capture sync --dry-run` (retirement deletes committed files; git restores them).
- A hard download cap inside the window.
- A traffic-proportional sampling mode.
- `code_version`-based staleness (opaque string, no ordering).
- Sweeping, or even reporting, the toolset sidecars a retirement leaves unreferenced
  (`capture clean` already sweeps them).

## Docs that change

- `DOCS.md`: Capturing from production (`captures:` example + sync steps), Top-level fields
  (`captures` row), How it works (doctor bullet), Command reference (sync, doctor, compare).
- `llms-full.txt`: the same sections plus the `captures:` schema block.
- `docs/configuration.md`: `## captures` field table, retention paragraph, Capture lifecycle.
- `docs/sdk.md`: §2 bucket-retention note, §3 promote.
- `docs/conversations.md`: conversations are sampled and kept whole.
- `CHANGELOG.md` `## [Unreleased]`: Added (`captures.max_age`, `captures.max_cases`, freshness
  rows/lines, case fields), Changed (sync loads config under `--offline`; `init` scaffolds limits).

## Compatibility

- **Opt-in.** Without the new keys, sync promotes and keeps exactly what 1.5.0 does. Two
  differences remain: case files gain `captured_at` / `source_model`, and sync loads the config
  under `--offline`.
- **New config keys** are rejected by older CLIs (`extra="forbid"`): reader ≥ writer, bump the CI
  action's `evalshift-version` — the same as `captures.store` in 1.3.0.
- **New `PromotedCase` fields**: an older CLI's `capture` commands fail closed on a new case file
  (`_iter_promoted_cases` raises; the fetch skip list degrades to empty, `capture.py:182-190`).
  Everyone who runs `capture` commands on the repo needs this version or later. CI is unaffected —
  it reads `golden.jsonl` only, and that format does not change.
- `compute_config_hash` covers the whole config, so new defaulted fields change it for every config
  and a `--resume` across the upgrade reports a mismatch (as in 1.3.0). The hosted
  `eval_config_hash` does not hash `captures` and does not move.
- No SDK, action, server or client change.

## Testing

- **Unit:** `parse_duration`; `CapturesConfig` fields (valid, invalid, dotted error location);
  `same_model` table; `plan_suite` properties — deterministic, idempotent, cap respected,
  replacement-only, reviewed pinned, twin preferred, groups whole, route balance, trim order;
  `suite_freshness` findings incl. legacy cases; promote writes the new fields; backfill;
  doctor rows; compare lines; init scaffold (plus the existing docs-match test).
- **Helpers:** `fake_remote_store.capture_payload` and `test_cli_capture._write_capture` gain a
  `model_id` parameter (both hardcode `"m"` today).
- **Integration** (`tests/integration/test_capture_lifecycle.py`), clock pinned:
  week 1 sync caps at `max_cases`; week 2 inside `max_age` changes nothing; clock past `max_age`
  with new captures replaces stale cases; `defaults.source_model` changed replaces model-stale
  cases; a fresh checkout with committed suites fetches only new captures; a wrong `source_model`
  trips the guardrail and filters nothing.
