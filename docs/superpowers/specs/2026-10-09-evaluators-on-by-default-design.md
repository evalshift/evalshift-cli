# Judge and semantic on by default; say which key is missing; promote with evidence

**Date:** 2026-10-09 · **Status:** draft for maintainer review · **Repos:** evalshift-cli only
(the hosted server never reads `state.json`; the bundle already carries `recommendations` as
free strings)

## Problem

`llm_judge` and `semantic` are the evaluators that tell a user whether the new model's prose is
as good as the old one's, and a fresh project gets the least out of them:

1. **`semantic` is off for Anthropic and DeepSeek projects.** `init` writes it commented out
   (`_SEMANTIC_BLOCK_DISABLED`) because neither provider has an embeddings endpoint. The only
   explanation is a YAML comment, so most users never turn it on — even with an
   `OPENAI_API_KEY` already exported.
2. **A missing judge or embedding key fails silently.** `compare` preflights keys only for the
   two arms (`compare.py:492`). With no key for `judge_model`, every judge call raises
   `EvaluatorError`, the axis records nothing but errors, and — advisory, as scaffolded —
   nothing gates on it and nothing is printed. The user pays the retries and never learns
   the evaluator did not run.
3. **The "promote" advice is generic.** A fresh project is all-advisory, so every run is
   `inconclusive` with the same line: "Set blocking: true on at least one trusted evaluator…".
   It does not say which evaluator, or whether the suite is big enough yet — although the
   policy layer holds each comparison's `n`. Repeated identically, it is learnt as noise.

## Decision

- **Everything on by default; tell the user what is missing.** `init` always writes an
  active `semantic` block. A missing key never stops an *advisory* evaluator's run — it skips
  that evaluator before any call and names the env var to export.
- **A missing key for a gate is an error, before anything is spent.** A *blocking* evaluator
  whose model has no key fails the preflight exactly as a missing arm key does today.
  Silently skipping a gate in CI would turn a broken config into a green check.
- **Advice from the run's own numbers.** When nothing gates, the recommendation names the
  advisory judge and its smallest per-prompt `n` against the existing
  `MIN_N_RELIABLE = 20`: ready to promote, or how far off.
- **One channel.** All post-run advice goes through `MigrationDecision.recommendations`, which
  already reaches `analyze`/`compare` output, the HTML report, `migration_decision.json` and the
  bundle. No new surface.

Not doing: changing the verdict or exit-code semantics; auto-promoting the judge at `n ≥ 20`
(a CI gate that flips with no config change is a surprise, and `blocking` is public config);
choosing a third-family judge in `init` (needs a second key; `doctor` already warns on a
same-family judge); rewriting existing users' `evalshift.yaml`; a "you could enable
`semantic`" hint for configs without one (nags users who removed it on purpose); a new config
field.

## 1. Key check and skip

### Shared helper

`models/registry.py` gains `missing_api_keys(model: str, env: Mapping[str, str]) -> tuple[str, ...]`:
the provider's env-var aliases when none is set (an empty string counts as unset), `()` when one
is set or the provider has no known key (`other`). It replaces the three copies that exist
today — `compare._missing_keys`, the check at `run.py:234`, and `insights.stage._missing_api_keys`
— with no behaviour change to their callers.

Provider inference fix: a bare id starting with `text-embedding-` resolves to provider `openai`.
Today the config default `embedding_model: text-embedding-3-small` resolves to `other`, so its
key could never be checked.

### `evaluator_key_gaps`

New module `src/evalshift_cli/evaluators/keys.py`:

```python
@dataclass(frozen=True, slots=True)
class EvaluatorKeyGap:
    evaluator_name: str      # "semantic" / "llm_judge.<criterion_name>", as the evaluator names itself
    kind: str                # "semantic" | "llm_judge"
    model: str               # as written in config
    env_vars: tuple[str, ...]
    blocking: bool           # effective — see below
    note: str = ""           # side effect of skipping, e.g. the tool_arguments fallback

def evaluator_key_gaps(evaluators: EvaluatorsConfig, env: Mapping[str, str]) -> list[EvaluatorKeyGap]: ...
```

One gap per `semantic` / `llm_judge` entry whose model has no key. Callers pass the
**resolved** set (`cfg.evaluators_for(suite_name)`) so a suite's own block is what is checked.

**Effective `blocking` for `semantic`.** `tool_arguments` borrows the semantic evaluator's
embedder (`evaluate.py:404`). Skipping `semantic` therefore changes how `tool_arguments` scores:

| `tool_arguments` uses | without the embedder | gap treated as |
| --- | --- | --- |
| an explicit `semantic` strategy (in `strategies` or as `default_strategy`) on a **blocking** entry | degrades to `exact` — a blocking gate silently scores differently | **blocking** |
| `auto` (the default) | `difflib` ratio, an existing documented fallback | as `semantic.blocking`, with `note` saying arguments are compared with difflib instead of embeddings |
| an explicit `semantic` strategy on an advisory entry | `exact` | as `semantic.blocking`, with `note` |

### Where it runs

1. **`compare` and `run` preflight**, before the first model call, beside the arm-key check.
   A blocking gap prints `✗ missing API key for <model> (<evaluator_name>); export <VARS>.` and
   exits 1, the same way and same exit code as an arm. An advisory gap prints one line and
   continues:

   ```
   ⚠ semantic skipped: no API key for openai/text-embedding-3-small — export OPENAI_API_KEY to enable it.
   ```

   followed by `note` on the next line when set.
2. **`run_evaluate`** (shared by `compare`, `evaluate`, `all`). It computes the gaps for the run's
   resolved set before `_build_evaluators`, drops advisory-gap entries from the config it builds
   from (no calls, no retries), and raises the existing `ConfigError` for a blocking gap — so
   `evaluate` over an old run behaves the same. If every evaluator is dropped, it raises
   `ConfigError` naming the missing vars, not the misleading `NoEvaluatorsError` "no evaluators
   configured". Under `quiet` (compare) it does not print; compare already printed at preflight.

### What gets recorded

- `RunState` gains `skipped_evaluators: list[SkippedEvaluator] = []` with
  `SkippedEvaluator(evaluator_name, kind, model, env_vars, note)`, written by `run_evaluate`.
  Defaulted so old `state.json` files load. (A newer run read by an older CLI fails on the
  unknown field — `RunState` forbids extras — which is acceptable: runs are local and read by
  the CLI that wrote them.)
- `EvaluateResult` gains the same tuple.
- A skipped evaluator writes **no** `EvaluatorCoverage` entry: it was never attempted, and an
  attempted-but-empty advisory axis already has a meaning (`advisory:` note) this must not
  borrow.

## 2. Post-run advice

`evaluate_migration_policy` and `inconclusive_decision` gain
`skipped_evaluators: Sequence[SkippedEvaluator] = ()`. Both callers that build a decision pass
`state.skipped_evaluators`: `analyze.py:151` and `hosted/bundle.py:137/148`, so the local and
hosted recommendations stay identical. (`insights/stage.py:286` passes it too.)

### Promotion advice (replaces the generic `enable_blocking` line)

Only when `no_blocking_records` holds — the state a fresh project is in. Once anything gates,
none of this is emitted, so a deliberately-advisory judge is never nagged.

For each advisory `llm_judge` evaluator with comparisons, take the overall-slice comparisons
(one per prompt) and the **smallest** `n` among them, `n_min`, and its prompt:

- `n_min ≥ MIN_N_RELIABLE`:
  `The equivalence judge scored at least {n_min} pairs on every prompt — enough to gate. Set blocking: true on it in evalshift.yaml to get a pass/fail verdict.`
- otherwise:
  `The equivalence judge is advisory, so this run has no pass/fail verdict. It becomes reliable at {MIN_N_RELIABLE} pairs per prompt; {prompt_id} has {n_min}. Collect more examples, then set blocking: true on it.`
  (`; {prompt_id} has` is omitted on a single-prompt run in favour of `this run has {n_min}`.)

The criterion is named by `criterion_name` (the part after `llm_judge.`). When there is no
advisory judge with comparisons (only `semantic`, or the judge was skipped), the existing
generic line is kept. `semantic` is never suggested for promotion: it measures drift, not
correctness.

### Skipped-evaluator lines

One per skipped evaluator, under every verdict, appended after the verdict's own advice like the
existing notes (never substituted):

```
semantic was skipped: no API key for openai/text-embedding-3-small. Export OPENAI_API_KEY to enable it.
```

with `note` appended when set.

### Not changing

The insights narrative: it mirrors *unmeasured blocking* evaluators, a different fact, and is
not handed `recommendations`. The bundle schema: `recommendations` is already `string[]`.

## 3. `init` writes everything on

- `render_minimal_config(*, profile, provider, env=None)` — `env` defaults to `os.environ`, so
  tests pass a mapping and the scaffold is deterministic under test.
- **Embedding model.** Gemini and OpenAI keep their own. Anthropic and DeepSeek choose from the
  keys present: `OPENAI_API_KEY` → `openai/text-embedding-3-small`; else
  `GEMINI_API_KEY`/`GOOGLE_API_KEY` → `gemini/gemini-embedding-001`; else
  `openai/text-embedding-3-small` anyway (written active; `compare` skips it until the key
  exists). `_SEMANTIC_BLOCK_DISABLED` is removed.
- **Block comment** for the borrowed case says which provider it uses and why, needs which key,
  and that `compare` skips it and says so without one.
- **Next steps.** When the embedding key is missing, one extra line:
  `semantic uses openai/text-embedding-3-small — export OPENAI_API_KEY to enable it; compare skips it until then.`
- **`init --ci`.** When the embedding provider differs from the main provider, the workflow's
  `env:` gains `<EMBED_KEY>: ${{ secrets.<EMBED_KEY> }}` and the CI next-steps line lists it as
  optional. An unset secret arrives as an empty string, which `missing_api_keys` treats as
  unset — the run warns and skips rather than failing.
- **`doctor`** gains an `evaluator keys` row from `evaluator_key_gaps`: `fail` for a blocking gap,
  `warn` for an advisory one, `ok` (`judge and semantic keys present`) otherwise; silent with no
  loadable config.

Side effect, new projects only: with an embedding key present, `tool_arguments`' `auto`
strategy now compares unequal strings by embeddings instead of `difflib`. Existing configs are
untouched.

## Tests

- `tests/unit/test_model_registry.py` — `missing_api_keys` (set, empty string, Gemini alias,
  `other`); `text-embedding-*` → openai.
- `tests/unit/test_evaluator_keys.py` (new) — gaps for judge/semantic; resolved per-suite set;
  effective-blocking table for `tool_arguments` (three rows) and its `note`.
- `tests/unit/test_evaluate_command.py` — advisory gap dropped with no client call; blocking gap
  raises `ConfigError`; all-dropped raises `ConfigError` naming the vars;
  `state.skipped_evaluators` written.
- `tests/unit/test_compare_command.py` — preflight `⚠` line and continue; blocking `✗` exits 1 before
  any model call.
- `tests/unit/test_policy.py` — promotion advice at `n_min` 24 and 8 (smallest prompt
  named; single-prompt wording); none once something gates; generic line when only `semantic`
  is advisory; skipped lines under `pass`/`fail`/`inconclusive`; `inconclusive_decision`
  carries them; bundle and analyze decisions equal.
- `tests/unit/test_init.py` — each provider × {OpenAI key, Gemini key, none}; CI workflow env;
  next-steps line; existing scaffold snapshots updated.
- `tests/unit/test_doctor.py` — the `evaluator keys` row's three states.

Tests live flat under `tests/unit/`, beside the existing files named above.

## Docs (same change)

`DOCS.md` and `llms-full.txt` (init scaffold, evaluator keys, skip behaviour, recommendations,
`state.json` field), `docs/configuration.md` (semantic default and the `tool_arguments`
interaction), `docs/faq.md` (the "why inconclusive" entry), the scaffolded `EVALSHIFT.md` if it
describes the evaluator defaults, and `CHANGELOG.md` under `## [Unreleased]`.
