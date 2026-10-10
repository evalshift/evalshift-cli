# Judge and Semantic On by Default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `init` scaffolds `semantic` and `llm_judge` active for every provider; a judge or embedding model with no API key is skipped (advisory) or refused (blocking) *before any call*, with the env var named; and the all-advisory `inconclusive` verdict tells the user, from the run's own `n`, when the judge is ready to gate.

**Architecture:** One helper (`missing_api_keys`) answers "which env var is missing for this model". One module (`evaluators/keys.py`) turns an `EvaluatorsConfig` into key gaps and owns every user-facing message about them. `run_evaluate` drops advisory gaps and records them in `state.json` (`skipped_evaluators`); `compare`/`run` preflight the same gaps before the first model call; the policy layer reads `skipped_evaluators` and `n` to write `recommendations`, which already reach the terminal, report, `migration_decision.json` and bundle.

**Tech Stack:** Python 3.11, Typer, Rich, Pydantic v2, pytest, mypy `--strict`, ruff.

**Spec:** `docs/superpowers/specs/2026-10-09-evaluators-on-by-default-design.md`

## Global Constraints

- Every module starts with `from __future__ import annotations`; public functions/classes get Google-style docstrings; `mypy --strict src/evalshift_cli` stays clean.
- Config is public API: **no new `evalshift.yaml` field**, `extra="forbid"` untouched, library defaults (`blocking: True`) unchanged.
- Verdict logic and exit codes are unchanged except: a *blocking* evaluator whose model has no key now exits 1 before any model call (same code as a missing arm key).
- An empty-string env var counts as unset.
- `RunState` gains exactly one field, `skipped_evaluators`, defaulted to `[]`.
- Message strings are exactly as written in the tasks below (tests assert them).
- `MIN_N_RELIABLE` (20) from `analysis/statistics.py` is the promotion threshold — never a new constant.
- Conventional Commits; every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Commit with the venv active (`source .venv/bin/activate`) so the pre-commit hooks run.
- Green means all four: `ruff check .`, `ruff format --check .`, `mypy --strict src/evalshift_cli`, `pytest`.

## Review Focus

1. **Only `GOOGLE_API_KEY` set, Gemini judge** — the legacy alias must satisfy the check; no skip, no warning. → Task 1 test `test_google_alias_satisfies_gemini`, Task 2 test `test_gemini_judge_accepts_google_alias`.
2. **CI secret unset → empty string** — `OPENAI_API_KEY=""` must count as missing (skip + warn), never as present (which would make every embedding call fail). → Task 1 test `test_empty_string_counts_as_unset`.
3. **A suite block that removes the judge** (`llm_judge: []` under `suites.<name>.evaluators`) — `compare --suite-name <name>` must not warn about a judge that will not run. → Task 2 test `test_gaps_follow_the_resolved_suite_set`.
4. **A `state.json` written before this change** — must still load (resume, `analyze`, `report`, bundle). → Task 3 test `test_state_without_skipped_evaluators_still_loads`.
5. **Two judges, one keyless** — only the keyless one is skipped; the other still scores. → Task 3 test `test_only_the_keyless_judge_is_skipped`.

---

### Task 1: `missing_api_keys` — one key check, three callers

**Files:**
- Modify: `src/evalshift_cli/models/registry.py` (add helper; `_infer_provider_and_canonical` lines 250–283; `__all__`)
- Modify: `src/evalshift_cli/cli/commands/compare.py:252-268` (`_missing_keys`)
- Modify: `src/evalshift_cli/cli/commands/run.py` (`_missing_api_keys`, ~line 220)
- Modify: `src/evalshift_cli/insights/stage.py:103` and delete `_missing_api_keys` (lines 359–364)
- Test: `tests/unit/test_model_registry.py`

**Interfaces:**
- Produces: `missing_api_keys(model: str, env: Mapping[str, str]) -> tuple[str, ...]` in `evalshift_cli.models.registry`. Returns the provider's env-var aliases (primary first) when none holds a non-empty value; `()` otherwise or for provider `"other"`.
- Produces: `resolve_model("text-embedding-3-small").provider == "openai"`, `.id == "openai/text-embedding-3-small"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_model_registry.py` (add `missing_api_keys` to the existing `from evalshift_cli.models.registry import ...` line):

```python
class TestMissingApiKeys:
    def test_returns_the_aliases_when_none_is_set(self) -> None:
        assert missing_api_keys("gemini-2.5-flash", {}) == ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def test_returns_empty_when_the_primary_is_set(self) -> None:
        assert missing_api_keys("claude-sonnet-5", {"ANTHROPIC_API_KEY": "k"}) == ()

    def test_google_alias_satisfies_gemini(self) -> None:
        assert missing_api_keys("gemini-2.5-flash", {"GOOGLE_API_KEY": "k"}) == ()

    def test_empty_string_counts_as_unset(self) -> None:
        # An unset GitHub secret arrives as "" — that is not a key.
        assert missing_api_keys("gpt-4o-mini", {"OPENAI_API_KEY": ""}) == ("OPENAI_API_KEY",)

    def test_unknown_provider_needs_no_key(self) -> None:
        assert missing_api_keys("mistral/mistral-large", {}) == ()


class TestEmbeddingProviderInference:
    def test_bare_openai_embedding_id_resolves_to_openai(self) -> None:
        meta = resolve_model("text-embedding-3-small")
        assert meta.provider == "openai"
        assert meta.id == "openai/text-embedding-3-small"

    def test_bare_embedding_id_now_has_a_checkable_key(self) -> None:
        assert missing_api_keys("text-embedding-3-small", {}) == ("OPENAI_API_KEY",)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/unit/test_model_registry.py -k "MissingApiKeys or EmbeddingProviderInference" -v`
Expected: FAIL — `ImportError: cannot import name 'missing_api_keys'`.

- [ ] **Step 3: Implement**

In `src/evalshift_cli/models/registry.py`, add `from collections.abc import Mapping` to the imports, then add after `resolve_model`:

```python
def missing_api_keys(model: str, env: Mapping[str, str]) -> tuple[str, ...]:
    """Return the env vars that would authenticate ``model`` when none is set.

    The single answer to "can this model be called?" — the arm preflight in
    ``run``/``compare``, the insights stage and the evaluator key checks all
    ask it, so an alias accepted in one place is accepted everywhere.

    Args:
        model: Any model id or alias :func:`resolve_model` accepts.
        env: Environment mapping, typically ``os.environ``. An empty-string
            value counts as unset: an unset CI secret arrives as ``""``.

    Returns:
        The provider's env-var aliases, primary first, when none of them holds
        a non-empty value. ``()`` when one does, or when the provider is
        ``"other"`` and the key it would need is unknown.
    """
    keys = PROVIDER_ENV_VARS.get(resolve_model(model).provider, ())
    if not keys or any(env.get(key) for key in keys):
        return ()
    return keys
```

In `_infer_provider_and_canonical`, add one bullet to the docstring decision tree after the `deepseek-` bullet:

```
    * If it starts with ``text-embedding-`` → openai, prefix ``openai/``
      (OpenAI's embedding ids; the config default is the bare form).
```

and the matching branch just before the final `return id_or_alias, "other"`:

```python
    if id_or_alias.startswith("text-embedding-"):
        return f"openai/{id_or_alias}", "openai"
```

Add `"missing_api_keys"` to `__all__` (alphabetical, after `"list_supported"`).

Replace the body of `compare._missing_keys` (keep its signature and return type):

```python
    seen: set[str] = set()
    missing: list[tuple[str, Provider, tuple[str, ...]]] = []
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        keys = missing_api_keys(m, env)
        if keys:
            missing.append((m, resolve_model(m).provider, keys))
    return missing
```

and import `missing_api_keys` in compare's existing `from evalshift_cli.models.registry import (...)` block. Do the same in `run.py`'s `_missing_api_keys` (identical body; keep its docstring, which already says provider `"other"` is skipped). Drop `PROVIDER_ENV_VARS` from either import list if it becomes unused (ruff will say).

In `insights/stage.py`, delete `_missing_api_keys` and change line 103 to:

```python
        missing_keys = missing_api_keys(model, env if env is not None else os.environ)
```

importing `missing_api_keys` in place of `PROVIDER_ENV_VARS` (keep `resolve_model` only if still used elsewhere in the file).

- [ ] **Step 4: Run the tests and the three callers' suites**

Run: `pytest tests/unit/test_model_registry.py tests/unit/test_run_command.py tests/unit/test_compare_command.py tests/unit/test_insights_stage.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/models/registry.py src/evalshift_cli/cli/commands/compare.py \
  src/evalshift_cli/cli/commands/run.py src/evalshift_cli/insights/stage.py \
  tests/unit/test_model_registry.py
git commit -m "refactor(models): one missing_api_keys check; bare text-embedding ids are OpenAI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `evaluators/keys.py` — key gaps and their messages

**Files:**
- Create: `src/evalshift_cli/evaluators/keys.py`
- Test: `tests/unit/test_evaluator_keys.py` (new)

**Interfaces:**
- Consumes: `missing_api_keys` (Task 1).
- Produces (all in `evalshift_cli.evaluators.keys`):
  - `SEMANTIC_EVALUATOR_NAME = "semantic.cosine"`, `SEMANTIC_KIND = "semantic"`, `LLM_JUDGE_KIND = "llm_judge"`
  - `@dataclass(frozen=True, slots=True) class EvaluatorKeyGap: evaluator_name: str; kind: str; label: str; model: str; env_vars: tuple[str, ...]; blocking: bool; note: str = ""`
  - `evaluator_key_gaps(evaluators: EvaluatorsConfig, env: Mapping[str, str]) -> list[EvaluatorKeyGap]` — semantic first, then judges in config order.
  - `without_gaps(evaluators: EvaluatorsConfig, gaps: Sequence[EvaluatorKeyGap]) -> EvaluatorsConfig`
  - `skip_warning(*, label: str, model: str, env_vars: Sequence[str], note: str = "") -> str` (Rich markup)
  - `missing_key_error(*, label: str, model: str, env_vars: Sequence[str]) -> str` (Rich markup)
  - `skip_recommendation(*, label: str, model: str, env_vars: Sequence[str], note: str = "") -> str` (plain text)
  - `AUTO_FALLBACK_NOTE`, `EXACT_FALLBACK_NOTE` (str constants)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_evaluator_keys.py`:

```python
"""Tests for :mod:`evalshift_cli.evaluators.keys`."""

from __future__ import annotations

from typing import Any

from evalshift_cli.config.models import EvalShiftConfig, EvaluatorsConfig
from evalshift_cli.evaluators.keys import (
    AUTO_FALLBACK_NOTE,
    EXACT_FALLBACK_NOTE,
    EvaluatorKeyGap,
    evaluator_key_gaps,
    missing_key_error,
    skip_recommendation,
    skip_warning,
    without_gaps,
)

_JUDGE: dict[str, Any] = {
    "criterion_name": "equivalence",
    "criterion_prompt": "which is better?",
    "judge_model": "gpt-4o-mini",
    "blocking": False,
}


def _evaluators(**families: Any) -> EvaluatorsConfig:
    return EvaluatorsConfig.model_validate(families)


class TestEvaluatorKeyGaps:
    def test_no_gaps_when_every_key_is_set(self) -> None:
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small"}, llm_judge=[_JUDGE]
        )
        assert evaluator_key_gaps(evaluators, {"OPENAI_API_KEY": "k"}) == []

    def test_keyless_semantic_and_judge_are_both_reported(self) -> None:
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small", "blocking": False},
            llm_judge=[_JUDGE],
        )
        assert evaluator_key_gaps(evaluators, {}) == [
            EvaluatorKeyGap(
                evaluator_name="semantic.cosine",
                kind="semantic",
                label="semantic",
                model="openai/text-embedding-3-small",
                env_vars=("OPENAI_API_KEY",),
                blocking=False,
            ),
            EvaluatorKeyGap(
                evaluator_name="llm_judge.equivalence",
                kind="llm_judge",
                label="llm_judge.equivalence",
                model="gpt-4o-mini",
                env_vars=("OPENAI_API_KEY",),
                blocking=False,
            ),
        ]

    def test_bare_default_embedding_id_is_checked(self) -> None:
        # The config default `text-embedding-3-small` used to resolve to
        # provider "other", so its key was never checked.
        gaps = evaluator_key_gaps(_evaluators(semantic={}), {})
        assert [g.env_vars for g in gaps] == [("OPENAI_API_KEY",)]

    def test_gemini_judge_accepts_google_alias(self) -> None:
        judge = {**_JUDGE, "judge_model": "gemini-2.5-flash"}
        assert evaluator_key_gaps(_evaluators(llm_judge=[judge]), {"GOOGLE_API_KEY": "k"}) == []

    def test_blocking_flag_is_carried(self) -> None:
        judge = {**_JUDGE, "blocking": True}
        (gap,) = evaluator_key_gaps(_evaluators(llm_judge=[judge]), {})
        assert gap.blocking is True

    def test_gaps_follow_the_resolved_suite_set(self) -> None:
        cfg = EvalShiftConfig.model_validate(
            {
                "prompts": [{"id": "a", "detection": "manual", "content": "hi"}],
                "evaluators": {"llm_judge": [_JUDGE]},
                "suites": {"tools": {"path": "s.jsonl", "evaluators": {"llm_judge": []}}},
            }
        )
        assert evaluator_key_gaps(cfg.evaluators_for("tools"), {}) == []
        assert len(evaluator_key_gaps(cfg.evaluators_for(None), {})) == 1


class TestSemanticToolArgumentsInteraction:
    def _gap(self, **tool_arguments: Any) -> EvaluatorKeyGap:
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small", "blocking": False},
            tool_arguments=[{"name": "args", **tool_arguments}],
        )
        (gap,) = evaluator_key_gaps(evaluators, {})
        return gap

    def test_blocking_explicit_semantic_strategy_makes_the_gap_blocking(self) -> None:
        gap = self._gap(strategies={"query": "semantic"}, blocking=True)
        assert gap.blocking is True
        assert gap.note == EXACT_FALLBACK_NOTE

    def test_blocking_semantic_default_strategy_makes_the_gap_blocking(self) -> None:
        gap = self._gap(default_strategy="semantic", blocking=True)
        assert gap.blocking is True

    def test_advisory_explicit_semantic_strategy_stays_advisory_with_a_note(self) -> None:
        gap = self._gap(strategies={"query": "semantic"}, blocking=False)
        assert gap.blocking is False
        assert gap.note == EXACT_FALLBACK_NOTE

    def test_auto_strategy_stays_advisory_and_notes_difflib(self) -> None:
        gap = self._gap(blocking=True)  # default_strategy is auto
        assert gap.blocking is False
        assert gap.note == AUTO_FALLBACK_NOTE


class TestWithoutGaps:
    def test_drops_only_the_named_evaluators(self) -> None:
        other = {**_JUDGE, "criterion_name": "tone", "judge_model": "claude-sonnet-5"}
        evaluators = _evaluators(
            semantic={"embedding_model": "openai/text-embedding-3-small"},
            llm_judge=[_JUDGE, other],
            structural=[{"type": "length", "min_chars": 1}],
        )
        gaps = evaluator_key_gaps(evaluators, {"ANTHROPIC_API_KEY": "k"})
        kept = without_gaps(evaluators, gaps)
        assert kept.semantic is None
        assert [j.criterion_name for j in kept.llm_judge] == ["tone"]
        assert len(kept.structural) == 1

    def test_no_gaps_returns_the_same_instance(self) -> None:
        evaluators = _evaluators(llm_judge=[_JUDGE])
        assert without_gaps(evaluators, []) is evaluators


class TestMessages:
    def test_skip_warning(self) -> None:
        assert skip_warning(
            label="semantic", model="openai/text-embedding-3-small", env_vars=("OPENAI_API_KEY",)
        ) == (
            "[yellow]⚠[/yellow] semantic skipped: no API key for "
            "openai/text-embedding-3-small — export OPENAI_API_KEY to enable it."
        )

    def test_skip_warning_appends_the_note_on_its_own_line(self) -> None:
        text = skip_warning(label="semantic", model="m", env_vars=("K",), note="n.")
        assert text.endswith("\n  n.")

    def test_missing_key_error(self) -> None:
        assert missing_key_error(
            label="llm_judge.equivalence", model="gpt-4o-mini", env_vars=("OPENAI_API_KEY",)
        ) == (
            "[red]✗[/red] missing API key for [bold]gpt-4o-mini[/bold] "
            "(llm_judge.equivalence); export OPENAI_API_KEY."
        )

    def test_skip_recommendation_joins_aliases(self) -> None:
        assert skip_recommendation(
            label="llm_judge.equivalence",
            model="gemini-2.5-flash",
            env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        ) == (
            "llm_judge.equivalence was skipped: no API key for gemini-2.5-flash. "
            "Export GEMINI_API_KEY or GOOGLE_API_KEY to enable it."
        )
```

Before running, open `src/evalshift_cli/config/models.py` and confirm the `suites.<name>` entry's path key and `ToolArgumentsEvaluatorConfig`'s required fields; if `path` / `name` differ, adjust only the test dicts.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_evaluator_keys.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalshift_cli.evaluators.keys'`.

- [ ] **Step 3: Implement `src/evalshift_cli/evaluators/keys.py`**

```python
"""Which judge / embedding models in an evaluator set have no API key.

``semantic`` and ``llm_judge`` are the evaluators that call a model of their
own, so they are the ones a missing key silently breaks: every call raises,
the axis records only errors, and — advisory, as ``init`` scaffolds them —
nothing gates on it. This module finds those gaps *before* any call and owns
every message about them, so the preflight line, the evaluate stage's skip
and the decision's recommendation cannot drift apart.

An advisory gap is skipped; a blocking one is refused. ``semantic``'s
effective blocking flag also accounts for ``tool_arguments``, which borrows
its embedder: skipping it would silently move a blocking ``semantic``
strategy onto exact matching.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evalshift_cli.config.models import EvaluatorsConfig, ToolArgumentsEvaluatorConfig
from evalshift_cli.models.registry import missing_api_keys

SEMANTIC_EVALUATOR_NAME = "semantic.cosine"
SEMANTIC_KIND = "semantic"
LLM_JUDGE_KIND = "llm_judge"

EXACT_FALLBACK_NOTE = (
    "tool_arguments fields using the semantic strategy fall back to exact matching until then."
)
AUTO_FALLBACK_NOTE = (
    "tool_arguments compares free-text arguments with difflib instead of embeddings until then."
)


@dataclass(frozen=True, slots=True)
class EvaluatorKeyGap:
    """One ``semantic`` / ``llm_judge`` entry whose model has no API key.

    Attributes:
        evaluator_name: The name the evaluator stamps on its records
            (``semantic.cosine``, ``llm_judge.<criterion_name>``).
        kind: ``"semantic"`` or ``"llm_judge"``.
        label: How messages name it: ``semantic`` or the judge's evaluator name.
        model: The model id as written in config.
        env_vars: The env vars that would satisfy it, primary first.
        blocking: Effective flag — see the module docstring for ``semantic``.
        note: A side effect of skipping it, or ``""``.
    """

    evaluator_name: str
    kind: str
    label: str
    model: str
    env_vars: tuple[str, ...]
    blocking: bool
    note: str = ""


def evaluator_key_gaps(
    evaluators: EvaluatorsConfig,
    env: Mapping[str, str],
) -> list[EvaluatorKeyGap]:
    """Find every ``semantic`` / ``llm_judge`` entry whose model has no key.

    Args:
        evaluators: An already-resolved set (``cfg.evaluators_for(suite)``),
            so a suite's own block is what gets checked.
        env: Environment mapping, typically ``os.environ``.

    Returns:
        One gap per keyless entry: ``semantic`` first, then judges in config
        order. Empty when every model can be called.
    """
    gaps: list[EvaluatorKeyGap] = []
    semantic = evaluators.semantic
    if semantic is not None:
        keys = missing_api_keys(semantic.embedding_model, env)
        if keys:
            blocking, note = _semantic_effect(semantic.blocking, evaluators.tool_arguments)
            gaps.append(
                EvaluatorKeyGap(
                    evaluator_name=SEMANTIC_EVALUATOR_NAME,
                    kind=SEMANTIC_KIND,
                    label=SEMANTIC_KIND,
                    model=semantic.embedding_model,
                    env_vars=keys,
                    blocking=blocking,
                    note=note,
                )
            )
    for judge in evaluators.llm_judge:
        keys = missing_api_keys(judge.judge_model, env)
        if keys:
            name = f"{LLM_JUDGE_KIND}.{judge.criterion_name}"
            gaps.append(
                EvaluatorKeyGap(
                    evaluator_name=name,
                    kind=LLM_JUDGE_KIND,
                    label=name,
                    model=judge.judge_model,
                    env_vars=keys,
                    blocking=judge.blocking,
                )
            )
    return gaps


def without_gaps(
    evaluators: EvaluatorsConfig,
    gaps: Sequence[EvaluatorKeyGap],
) -> EvaluatorsConfig:
    """Return ``evaluators`` minus the entries named by ``gaps``.

    Args:
        evaluators: The resolved set the gaps were computed from.
        gaps: The gaps to drop (normally the advisory ones).

    Returns:
        ``evaluators`` itself when there is nothing to drop, else a copy with
        ``semantic`` cleared and/or the keyless judges removed.
    """
    if not gaps:
        return evaluators
    names = {gap.evaluator_name for gap in gaps}
    update: dict[str, object] = {}
    if SEMANTIC_EVALUATOR_NAME in names:
        update["semantic"] = None
    judges = [
        j for j in evaluators.llm_judge if f"{LLM_JUDGE_KIND}.{j.criterion_name}" not in names
    ]
    if len(judges) != len(evaluators.llm_judge):
        update["llm_judge"] = judges
    return evaluators.model_copy(update=update)


def skip_warning(*, label: str, model: str, env_vars: Sequence[str], note: str = "") -> str:
    """Terminal line (Rich markup) for an advisory evaluator that will be skipped."""
    text = (
        f"[yellow]⚠[/yellow] {label} skipped: no API key for {model} — "
        f"export {' or '.join(env_vars)} to enable it."
    )
    return f"{text}\n  {note}" if note else text


def missing_key_error(*, label: str, model: str, env_vars: Sequence[str]) -> str:
    """Terminal line (Rich markup) for a blocking evaluator with no key."""
    return (
        f"[red]✗[/red] missing API key for [bold]{model}[/bold] ({label}); "
        f"export {' or '.join(env_vars)}."
    )


def skip_recommendation(
    *, label: str, model: str, env_vars: Sequence[str], note: str = ""
) -> str:
    """Plain-text ``recommendations`` line for an evaluator the run skipped."""
    text = (
        f"{label} was skipped: no API key for {model}. "
        f"Export {' or '.join(env_vars)} to enable it."
    )
    return f"{text} {note}" if note else text


def _uses(ta: ToolArgumentsEvaluatorConfig, strategy: str) -> bool:
    return ta.default_strategy == strategy or strategy in ta.strategies.values()


def _semantic_effect(
    semantic_blocking: bool,
    tool_arguments: Sequence[ToolArgumentsEvaluatorConfig],
) -> tuple[bool, str]:
    """Effective blocking flag and side-effect note for a keyless ``semantic``."""
    if any(ta.blocking and _uses(ta, "semantic") for ta in tool_arguments):
        return True, EXACT_FALLBACK_NOTE
    if any(_uses(ta, "semantic") for ta in tool_arguments):
        return semantic_blocking, EXACT_FALLBACK_NOTE
    if any(_uses(ta, "auto") for ta in tool_arguments):
        return semantic_blocking, AUTO_FALLBACK_NOTE
    return semantic_blocking, ""


__all__ = [
    "AUTO_FALLBACK_NOTE",
    "EXACT_FALLBACK_NOTE",
    "LLM_JUDGE_KIND",
    "SEMANTIC_EVALUATOR_NAME",
    "SEMANTIC_KIND",
    "EvaluatorKeyGap",
    "evaluator_key_gaps",
    "missing_key_error",
    "skip_recommendation",
    "skip_warning",
    "without_gaps",
]
```

Note the `test_auto_strategy_stays_advisory_and_notes_difflib` case: an `auto` blocking `tool_arguments` keeps `semantic`'s own (advisory) flag, because `auto` without embeddings is the documented `difflib` path, not a silent downgrade.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_evaluator_keys.py -v && mypy --strict src/evalshift_cli/evaluators/keys.py`
Expected: PASS; mypy `Success`.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/evaluators/keys.py tests/unit/test_evaluator_keys.py
git commit -m "feat(evaluators): find judge/embedding models with no API key

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: evaluate stage skips advisory gaps, refuses blocking ones, records them

**Files:**
- Modify: `src/evalshift_cli/runner/models.py` (new `SkippedEvaluator`; `RunState.skipped_evaluators`; docstring)
- Modify: `src/evalshift_cli/config/loader.py:33` (`ConfigErrorKind` gains `"missing_key"`)
- Modify: `src/evalshift_cli/cli/commands/evaluate.py` (`run_evaluate`, `EvaluateResult`, new `preflight_evaluator_keys`)
- Test: `tests/unit/test_evaluate_command.py`

**Interfaces:**
- Consumes: Task 2's `evaluator_key_gaps`, `without_gaps`, `skip_warning`, `missing_key_error`, `EvaluatorKeyGap`.
- Produces:
  - `class SkippedEvaluator(_StrictModel)` in `evalshift_cli.runner.models` with fields `evaluator_name: str`, `kind: str`, `label: str`, `model: str`, `env_vars: list[str]`, `note: str = ""`.
  - `RunState.skipped_evaluators: list[SkippedEvaluator]` (default `[]`).
  - `EvaluateResult.skipped: tuple[SkippedEvaluator, ...] = ()`.
  - `preflight_evaluator_keys(console: Console, evaluators_cfg: EvaluatorsConfig, env: Mapping[str, str]) -> list[EvaluatorKeyGap]` in `evalshift_cli.cli.commands.evaluate` — prints, returns advisory gaps, raises `typer.Exit(code=1)` on any blocking gap.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_evaluate_command.py` (add `SkippedEvaluator` to the `evalshift_cli.runner.models` import):

```python
_ALL_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
)


def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


def _judges_config(tmp_path: Path, *judges: tuple[str, str, bool], structural: bool = True) -> None:
    """Write a config with the given ``(criterion, judge_model, blocking)`` judges."""
    structural_block = (
        "\n          structural:\n            - type: length\n              min_chars: 1"
        if structural
        else ""
    )
    judge_block = "".join(
        f"\n            - criterion_name: {name}"
        f"\n              criterion_prompt: which is better?"
        f"\n              judge_model: {model}"
        f"\n              blocking: {str(blocking).lower()}"
        for name, model, blocking in judges
    )
    (tmp_path / "evalshift.yaml").write_text(
        f"""
        version: 1
        prompts:
          - id: greet
            detection: manual
            content: "Hi {{name}}"
            variables: [name]
        defaults:
          source_model: gemini-2.5-flash
          target_model: gemini-2.5-pro
        evaluators:{structural_block}
          llm_judge:{judge_block}
        """,
        encoding="utf-8",
    )


class TestEvaluatorKeyGaps:
    def test_advisory_keyless_judge_is_skipped_without_a_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from evalshift_cli.models import client as client_module

        _clear_keys(monkeypatch)
        _judges_config(tmp_path, ("equivalence", "gpt-4o-mini", False))
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)

        async def no_call(**_: Any) -> Any:
            raise AssertionError("a skipped judge must not be called")

        monkeypatch.setattr(client_module.litellm, "acompletion", no_call)

        result = runner.invoke(app, ["evaluate", run_id])

        assert result.exit_code == 0, result.stdout
        flat = " ".join(result.stdout.split())
        assert "llm_judge.equivalence skipped: no API key for gpt-4o-mini" in flat
        assert "export OPENAI_API_KEY to enable it." in flat
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        rows = (run_dir / SCORES_FILENAME).read_text(encoding="utf-8").splitlines()
        assert all("llm_judge" not in row for row in rows)
        assert read_state(run_dir).skipped_evaluators == [
            SkippedEvaluator(
                evaluator_name="llm_judge.equivalence",
                kind="llm_judge",
                label="llm_judge.equivalence",
                model="gpt-4o-mini",
                env_vars=["OPENAI_API_KEY"],
            )
        ]

    def test_blocking_keyless_judge_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clear_keys(monkeypatch)
        _judges_config(tmp_path, ("equivalence", "gpt-4o-mini", True))
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["evaluate", run_id])

        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "missing API key for a blocking evaluator" in flat
        assert "OPENAI_API_KEY" in flat
        assert not (tmp_path / ".evalshift" / "runs" / run_id / SCORES_FILENAME).exists()

    def test_every_evaluator_skipped_names_the_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _clear_keys(monkeypatch)
        _judges_config(tmp_path, ("equivalence", "gpt-4o-mini", False), structural=False)
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["evaluate", run_id])

        assert result.exit_code == 1
        flat = " ".join(result.stdout.split())
        assert "every configured evaluator was skipped" in flat
        assert "OPENAI_API_KEY" in flat
        assert "no evaluators configured" not in flat

    def test_only_the_keyless_judge_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from evalshift_cli.models import client as client_module

        _clear_keys(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.setattr(
            "evalshift_cli.cache.schema.DEFAULT_CACHE_PATH", tmp_path / "cache.db"
        )
        _judges_config(
            tmp_path,
            ("equivalence", "gpt-4o-mini", False),
            ("tone", "gemini-2.5-flash", False),
        )
        run_id = _scaffold_run(tmp_path)
        monkeypatch.chdir(tmp_path)

        async def fake_acompletion(**_: Any) -> Any:
            return _judge_response('{"winner": "A"}')

        monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(
            client_module.litellm, "completion_cost", lambda completion_response=None, **_: 0.0
        )

        result = runner.invoke(app, ["evaluate", run_id])

        assert result.exit_code == 0, result.stdout
        run_dir = tmp_path / ".evalshift" / "runs" / run_id
        names = {
            json.loads(line)["evaluator_name"]
            for line in (run_dir / SCORES_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert "llm_judge.tone" in names
        assert "llm_judge.equivalence" not in names
        assert [s.evaluator_name for s in read_state(run_dir).skipped_evaluators] == [
            "llm_judge.equivalence"
        ]

    def test_state_without_skipped_evaluators_still_loads(self, tmp_path: Path) -> None:
        run_id = _scaffold_run(tmp_path)
        state_path = tmp_path / ".evalshift" / "runs" / run_id / "state.json"
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        raw.pop("skipped_evaluators", None)
        state_path.write_text(json.dumps(raw), encoding="utf-8")
        assert read_state(state_path.parent).skipped_evaluators == []
```

(`_judge_response` is the existing helper this file's judge tests already use.)

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_evaluate_command.py -k TestEvaluatorKeyGaps -v`
Expected: FAIL — `ImportError: cannot import name 'SkippedEvaluator'`.

- [ ] **Step 3: Add `SkippedEvaluator` and the `RunState` field**

In `src/evalshift_cli/runner/models.py`, after `EvaluatorCoverage`:

```python
class SkippedEvaluator(_StrictModel):
    """An evaluator the evaluate stage did not run because its model had no key.

    Written by ``evaluate`` for an *advisory* ``semantic`` / ``llm_judge``
    entry whose judge or embedding model had no API key — a blocking one is
    refused instead. Recorded, not just printed, because ``analyze``, the
    report and the bundle run later and each must still say what was missing.

    Attributes:
        evaluator_name: The name its records would have carried.
        kind: ``"semantic"`` or ``"llm_judge"``.
        label: How messages name it.
        model: The judge or embedding model id as written in config.
        env_vars: The env vars that would have satisfied it, primary first.
        note: A side effect of the skip (e.g. the ``tool_arguments``
            fallback), or ``""``.
    """

    evaluator_name: str
    kind: str
    label: str
    model: str
    env_vars: list[str]
    note: str = ""
```

In `RunState`'s docstring, after `samples_per_example`, add:

```
        skipped_evaluators: Advisory ``semantic`` / ``llm_judge`` entries the
            ``evaluate`` stage skipped because their model had no API key.
            Empty when every evaluator ran. See :class:`SkippedEvaluator`.
```

and the field at the end of the class:

```python
    # Written by `evalshift evaluate` alongside evaluator_coverage. Defaulted
    # so state.json files written before this field existed still load under
    # extra="forbid".
    skipped_evaluators: list[SkippedEvaluator] = Field(default_factory=list)
```

In `src/evalshift_cli/config/loader.py:33`:

```python
ConfigErrorKind = Literal[
    "missing", "not_a_file", "yaml_parse", "not_a_mapping", "schema", "missing_key"
]
```

- [ ] **Step 4: Wire the gaps into `run_evaluate`**

In `src/evalshift_cli/cli/commands/evaluate.py`: add `import os` and `from collections.abc import Mapping`; extend the loader import to `from evalshift_cli.config.loader import ConfigError, ConfigErrorDetail, load_config`; add

```python
from evalshift_cli.evaluators.keys import (
    EvaluatorKeyGap,
    evaluator_key_gaps,
    missing_key_error,
    skip_warning,
    without_gaps,
)
```

and add `SkippedEvaluator` to the existing `evalshift_cli.runner.models` import.

Add to `EvaluateResult` (after `harness_check`):

```python
    #: Advisory evaluators dropped because their model had no API key; the
    #: same list ``state.json`` carries as ``skipped_evaluators``.
    skipped: tuple[SkippedEvaluator, ...] = ()
```

In `run_evaluate`, replace

```python
    evaluators = _build_evaluators(
        cfg.evaluators_for(state.suite_name),
        project_root,
        judge_client=judge_client,
        suite_path=Path(state.suite_path),
    )
    if not evaluators:
        raise NoEvaluatorsError(
```

with

```python
    resolved = cfg.evaluators_for(state.suite_name)
    # Before any evaluator is built, so a keyless judge or embedder costs no
    # call and no retry. A blocking one is refused: skipping a gate would turn
    # a broken config into a green check.
    gaps = evaluator_key_gaps(resolved, os.environ)
    blocking_gaps = [gap for gap in gaps if gap.blocking]
    if blocking_gaps:
        raise ConfigError(
            config_path,
            "missing_key",
            "missing API key for a blocking evaluator",
            details=[_gap_detail(gap) for gap in blocking_gaps],
        )
    skipped_gaps = [gap for gap in gaps if not gap.blocking]
    evaluators = _build_evaluators(
        without_gaps(resolved, skipped_gaps),
        project_root,
        judge_client=judge_client,
        suite_path=Path(state.suite_path),
    )
    if not evaluators and skipped_gaps:
        raise ConfigError(
            config_path,
            "missing_key",
            "every configured evaluator was skipped: none of their models has an API key",
            details=[_gap_detail(gap) for gap in skipped_gaps],
        )
    if not quiet:
        for gap in skipped_gaps:
            console.print(
                skip_warning(label=gap.label, model=gap.model, env_vars=gap.env_vars, note=gap.note)
            )
    skipped = tuple(_skipped_record(gap) for gap in skipped_gaps)
    if not evaluators:
        raise NoEvaluatorsError(
```

In the existing `write_state(... update={...})` dict add `"skipped_evaluators": list(skipped),`, and pass `skipped=skipped` to the returned `EvaluateResult(...)`.

Add the helpers below `run_evaluate`:

```python
def preflight_evaluator_keys(
    console: Console,
    evaluators_cfg: EvaluatorsConfig,
    env: Mapping[str, str],
) -> list[EvaluatorKeyGap]:
    """Check every judge / embedding model's key before any model call.

    Shared by ``run`` and ``compare`` so the line a user sees before money is
    spent is the one :func:`run_evaluate` would act on.

    Args:
        console: Where to print the per-gap lines.
        evaluators_cfg: The resolved evaluator set for this run's suite.
        env: Environment mapping, typically ``os.environ``.

    Returns:
        The advisory gaps — the evaluators the evaluate stage will skip.

    Raises:
        typer.Exit: Code 1 when any blocking evaluator's model has no key.
    """
    gaps = evaluator_key_gaps(evaluators_cfg, env)
    blocking = [gap for gap in gaps if gap.blocking]
    for gap in blocking:
        console.print(missing_key_error(label=gap.label, model=gap.model, env_vars=gap.env_vars))
    if blocking:
        console.print(
            "  Export the key, or set [bold]blocking: false[/bold] on the evaluator "
            "to run without it."
        )
        raise typer.Exit(code=1)
    skipped = [gap for gap in gaps if not gap.blocking]
    for gap in skipped:
        console.print(
            skip_warning(label=gap.label, model=gap.model, env_vars=gap.env_vars, note=gap.note)
        )
    return skipped


def _gap_detail(gap: EvaluatorKeyGap) -> ConfigErrorDetail:
    return ConfigErrorDetail(
        location=f"evaluators.{gap.kind}",
        message=(
            f"{gap.label} uses {gap.model}; export {' or '.join(gap.env_vars)}"
            + (" (or set blocking: false to run without it)" if gap.blocking else "")
        ),
    )


def _skipped_record(gap: EvaluatorKeyGap) -> SkippedEvaluator:
    return SkippedEvaluator(
        evaluator_name=gap.evaluator_name,
        kind=gap.kind,
        label=gap.label,
        model=gap.model,
        env_vars=list(gap.env_vars),
        note=gap.note,
    )
```

Add `"preflight_evaluator_keys"` to the module's `__all__`.

- [ ] **Step 5: Run the new tests**

Run: `pytest tests/unit/test_evaluate_command.py -k TestEvaluatorKeyGaps -v`
Expected: PASS.

- [ ] **Step 6: Run the whole unit suite and fix key fallout in tests only**

Run: `pytest tests/unit -q -x`

Existing tests that configure a judge or `semantic` (blocking by library default) and never set its key now hit the blocking refusal — e.g. the `with_judge=True` tests in `test_evaluate_command.py` (Gemini judge → `GEMINI_API_KEY`; the DeepSeek case → `DEEPSEEK_API_KEY`). Fix each by adding `monkeypatch.setenv("<KEY>", "test-key")` in that test. **Never** relax `evaluator_key_gaps` or `run_evaluate` to make an old test pass. Re-run until green.

- [ ] **Step 7: Commit**

```bash
git add src/evalshift_cli/runner/models.py src/evalshift_cli/config/loader.py \
  src/evalshift_cli/cli/commands/evaluate.py tests/unit/
git commit -m "feat(evaluate): skip a keyless advisory judge/embedder, refuse a keyless gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `compare` and `run` preflight the evaluator keys

**Files:**
- Modify: `src/evalshift_cli/cli/commands/compare.py` (after the arm-key check ~line 500; stage-4 `except`; `_evaluator_family_summary`)
- Modify: `src/evalshift_cli/cli/commands/run.py` (after the arm-key check ~line 154)
- Test: `tests/unit/test_compare_command.py`, `tests/unit/test_run_command.py`

**Interfaces:**
- Consumes: `preflight_evaluator_keys` (Task 3), `EvaluatorKeyGap`, `SEMANTIC_EVALUATOR_NAME`, `LLM_JUDGE_KIND` (Task 2).
- Produces: `_evaluator_family_summary(evaluators: EvaluatorsConfig, skipped: Sequence[EvaluatorKeyGap] = ()) -> str` — marks `semantic (no key)` / `judge (no key)` (the latter only when every judge is skipped).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_compare_command.py`:

```python
_JUDGE_YAML = (
    "              max_chars: 200\n"
    "          llm_judge:\n"
    "            - criterion_name: equivalence\n"
    "              criterion_prompt: which is better?\n"
    "              judge_model: gpt-4o-mini\n"
    "              blocking: {blocking}"
)


class TestEvaluatorKeyPreflight:
    def _with_judge(self, tmp_path: Path, *, blocking: bool) -> None:
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "              max_chars: 200",
                _JUDGE_YAML.format(blocking=str(blocking).lower()),
            )
            + "\n        migration_policy:\n"
            + "          max_critical_regressions: 0\n",
            encoding="utf-8",
        )

    def test_keyless_advisory_judge_warns_and_the_run_completes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        self._with_judge(tmp_path, blocking=False)
        _patch_client(monkeypatch)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert "llm_judge.equivalence skipped: no API key for gpt-4o-mini" in flat
        assert (
            "llm_judge.equivalence was skipped: no API key for gpt-4o-mini. "
            "Export OPENAI_API_KEY to enable it."
        ) in flat

    def test_keyless_blocking_judge_exits_before_any_model_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        self._with_judge(tmp_path, blocking=True)
        _patch_client(monkeypatch)
        calls: list[str] = []

        async def spy_complete(self: ModelClient, **kwargs: Any) -> CompletionResult:
            calls.append(str(kwargs["model"]))
            raise AssertionError("no model call may happen before the key preflight")

        monkeypatch.setattr(ModelClient, "complete", spy_complete)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["all", "--yes"])

        assert result.exit_code == 1
        flat = " ".join(result.output.split())
        assert "missing API key for gpt-4o-mini (llm_judge.equivalence); export OPENAI_API_KEY." in flat
        assert calls == []


class TestFamilySummarySkipped:
    def test_marks_a_skipped_semantic_and_an_all_skipped_judge(self) -> None:
        evaluators = EvaluatorsConfig.model_validate(
            {
                "semantic": {"embedding_model": "openai/text-embedding-3-small"},
                "llm_judge": [
                    {"criterion_name": "e", "criterion_prompt": "p", "judge_model": "gpt-4o-mini"}
                ],
            }
        )
        from evalshift_cli.evaluators.keys import evaluator_key_gaps

        gaps = evaluator_key_gaps(evaluators, {})
        assert _evaluator_family_summary(evaluators, gaps) == "semantic (no key) · judge (no key)"
        assert _evaluator_family_summary(evaluators) == "semantic · judge"
```

(Import `EvaluatorsConfig` from `evalshift_cli.config.models` at the top if the file does not already.)

Append to `tests/unit/test_run_command.py`:

```python
class TestEvaluatorKeyPreflight:
    def test_keyless_blocking_judge_exits_before_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _scaffold(tmp_path)
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "          concurrency: 4",
                "          concurrency: 4\n"
                "        evaluators:\n"
                "          llm_judge:\n"
                "            - criterion_name: equivalence\n"
                "              criterion_prompt: which is better?\n"
                "              judge_model: gpt-4o-mini",
            ),
            encoding="utf-8",
        )
        _patch_client(monkeypatch)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["run", "--yes"])

        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.stdout
        assert not (tmp_path / ".evalshift" / "runs").exists()
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_compare_command.py -k "EvaluatorKeyPreflight or FamilySummarySkipped" tests/unit/test_run_command.py -k EvaluatorKeyPreflight -v`
Expected: FAIL — advisory test lacks the `⚠` line; blocking tests exit later / make calls; `_evaluator_family_summary` takes 1 positional argument.

- [ ] **Step 3: Implement**

`compare.py` — import `preflight_evaluator_keys` in the existing `from evalshift_cli.cli.commands.evaluate import (...)` block, and `from evalshift_cli.evaluators.keys import LLM_JUDGE_KIND, SEMANTIC_EVALUATOR_NAME, EvaluatorKeyGap`; add `from collections.abc import Sequence`. Directly after the arm-key `raise typer.Exit(code=1)` block:

```python
    # The judge and embedding models make calls of their own; check them now,
    # before the doctor stage and long before the first arm call is paid for.
    skipped_gaps = preflight_evaluator_keys(console, cfg.evaluators_for(suite_name), os.environ)
```

Stage 4: change `rows[3].payload = _evaluator_family_summary(cfg.evaluators_for(suite_name))` to pass `skipped_gaps`, and widen the stage's `except`:

```python
                except (NoEvaluatorsError, NoPairsError) as exc:
                    rows[3].status = "failed"
                    rows[3].payload = str(exc)
                    update(live)
                    raise typer.Exit(code=1) from exc
                except ConfigError as exc:
                    # The preflight above makes this unreachable with the same
                    # environment; kept so a config edited mid-run fails the
                    # row instead of escaping the Live region.
                    rows[3].status = "failed"
                    rows[3].payload = exc.summary
                    update(live)
                    raise typer.Exit(code=1) from exc
```

Replace `_evaluator_family_summary`:

```python
def _evaluator_family_summary(
    evaluators: EvaluatorsConfig,
    skipped: Sequence[EvaluatorKeyGap] = (),
) -> str:
    """Name the evaluator families in an already-resolved set.

    Takes the resolved set rather than the whole config so the pipeline row
    names what *this* suite is scored with — on a heterogeneous project the
    top-level block is not what runs. A family whose model has no key is
    marked ``(no key)``: listed as configured, never as having scored.
    """
    skipped_names = {gap.evaluator_name for gap in skipped}
    families: list[str] = []
    if evaluators.structural:
        families.append("structural")
    if evaluators.semantic is not None:
        families.append(
            "semantic (no key)" if SEMANTIC_EVALUATOR_NAME in skipped_names else "semantic"
        )
    if evaluators.llm_judge:
        all_skipped = all(
            f"{LLM_JUDGE_KIND}.{j.criterion_name}" in skipped_names for j in evaluators.llm_judge
        )
        families.append("judge (no key)" if all_skipped else "judge")
    if evaluators.tool_selection or evaluators.tool_arguments or evaluators.tool_trace_structure:
        families.append("tool-call")
    return " · ".join(families) if families else "(none)"
```

`run.py` — import `from evalshift_cli.cli.commands.evaluate import preflight_evaluator_keys`; after the arm-key `raise typer.Exit(code=1)`:

```python
    preflight_evaluator_keys(console, cfg.evaluators_for(suite_name), os.environ)
```

- [ ] **Step 4: Run the tests, then the two command suites**

Run: `pytest tests/unit/test_compare_command.py tests/unit/test_run_command.py tests/unit/test_compare_suite_name.py tests/unit/test_compare_alias.py -q`
Expected: PASS. Fix any existing test that now trips the preflight by setting the key in that test (same rule as Task 3 Step 6).

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/cli/commands/compare.py src/evalshift_cli/cli/commands/run.py tests/unit/
git commit -m "feat(compare,run): check judge and embedding keys before the first call

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: recommendations — promotion advice from `n`, skipped-evaluator lines

**Files:**
- Modify: `src/evalshift_cli/analysis/policy.py` (`evaluate_migration_policy` ~369, `inconclusive_decision` ~604, `_recommendations` ~1581; new `_promotion_advice`)
- Modify: `src/evalshift_cli/cli/commands/analyze.py:151`
- Modify: `src/evalshift_cli/hosted/bundle.py:137-155`
- Modify: `src/evalshift_cli/insights/stage.py:286`
- Test: `tests/unit/test_policy.py`, `tests/unit/test_analyze_command.py`, `tests/unit/test_bundle_shape.py`

**Interfaces:**
- Consumes: `SkippedEvaluator` (Task 3), `skip_recommendation` (Task 2), `MIN_N_RELIABLE` (`analysis/statistics.py`).
- Produces: keyword `skipped_evaluators: Sequence[SkippedEvaluator] = ()` on both `evaluate_migration_policy` and `inconclusive_decision`.

- [ ] **Step 1: Extend the test helper and write the failing tests**

In `tests/unit/test_policy.py`, give `_comparison` two new keyword parameters and pass them through:

```python
def _comparison(
    *,
    severity: str,
    slice_name: str = "all",
    evaluator_name: str = "structural.length",
    prompt_id: str = "p",
    delta_avg_score: float = -0.1,
    notes: list[str] | None = None,
    n: int = 30,
    kind: str = "",
) -> ComparisonResult:
    return ComparisonResult(
        prompt_id=prompt_id,
        evaluator_name=evaluator_name,
        slice_name=slice_name,
        n=n,
        test="paired_t",
        statistic=1.0,
        p_value=0.01,
        p_value_corrected=0.01,
        effect_size=-0.8 if severity not in {"improved", "none"} else 0.2,
        effect_size_ci_low=-1.0,
        effect_size_ci_high=-0.2,
        delta_avg_score=delta_avg_score,
        severity=severity,  # type: ignore[arg-type]
        notes=notes or [],
        kind=kind,
    )
```

Add `SkippedEvaluator` to the `evalshift_cli.runner.models` import, then append:

```python
_SKIPPED_SEMANTIC = SkippedEvaluator(
    evaluator_name="semantic.cosine",
    kind="semantic",
    label="semantic",
    model="openai/text-embedding-3-small",
    env_vars=["OPENAI_API_KEY"],
)
_SKIPPED_LINE = (
    "semantic was skipped: no API key for openai/text-embedding-3-small. "
    "Export OPENAI_API_KEY to enable it."
)


def _judge(prompt_id: str, n: int) -> ComparisonResult:
    return _comparison(
        severity="none",
        evaluator_name="llm_judge.equivalence",
        kind="llm_judge",
        prompt_id=prompt_id,
        n=n,
    )


def _advisory_judge_records() -> list[EvalRecord]:
    return [
        _record(example_id=f"a{i}", delta=0.0, evaluator_name="llm_judge.equivalence", blocking=False)
        for i in range(3)
    ]


def _decide(
    comparisons: list[ComparisonResult],
    records: list[EvalRecord],
    skipped: tuple[SkippedEvaluator, ...] = (),
) -> MigrationDecision:
    return evaluate_migration_policy(
        run_id="r1",
        source_model="src",
        target_model="tgt",
        policy=MigrationPolicy(),
        comparisons=comparisons,
        records=records,
        calls=[],
        skipped_evaluators=skipped,
    )


class TestPromotionAdvice:
    def test_ready_judge_is_named_with_its_smallest_n(self) -> None:
        decision = _decide([_judge("summarize", 24), _judge("classify", 31)], _advisory_judge_records())
        assert decision.recommendations == [
            "The equivalence judge scored at least 24 pairs on every prompt — enough to "
            "gate. Set blocking: true on it in evalshift.yaml to get a pass/fail verdict.",
        ]

    def test_small_judge_names_the_smallest_prompt(self) -> None:
        decision = _decide([_judge("summarize", 8), _judge("classify", 31)], _advisory_judge_records())
        assert decision.recommendations == [
            "The equivalence judge is advisory, so it does not gate this run. It becomes "
            "reliable at 20 pairs per prompt; summarize has 8. Collect more examples, then "
            "set blocking: true on it.",
        ]

    def test_single_prompt_wording(self) -> None:
        decision = _decide([_judge("p", 8)], _advisory_judge_records())
        assert "this run has 8" in decision.recommendations[0]

    def test_no_advice_once_something_gates(self) -> None:
        decision = _decide(
            [_judge("p", 8), _comparison(severity="none", delta_avg_score=0.0)],
            [*_advisory_judge_records(), _record(example_id="b1", delta=0.0)],
        )
        assert not any("judge" in r for r in decision.recommendations)

    def test_only_semantic_advisory_keeps_the_generic_line(self) -> None:
        decision = _decide(
            [_comparison(severity="none", evaluator_name="semantic.cosine", kind="semantic")],
            [_record(example_id="a1", delta=0.0, evaluator_name="semantic.cosine", blocking=False)],
        )
        assert decision.recommendations == [
            "Set blocking: true on at least one trusted evaluator in "
            "evalshift.yaml to get a pass/fail verdict.",
        ]


class TestSkippedEvaluatorLines:
    def test_inconclusive(self) -> None:
        decision = _decide([_judge("p", 8)], _advisory_judge_records(), (_SKIPPED_SEMANTIC,))
        assert decision.verdict == "inconclusive"
        assert decision.recommendations[-1] == _SKIPPED_LINE

    def test_pass(self) -> None:
        decision = _decide(
            [_comparison(severity="none", delta_avg_score=0.0)],
            [_record(example_id=f"b{i}", delta=0.0) for i in range(5)],
            (_SKIPPED_SEMANTIC,),
        )
        assert decision.verdict == "pass"
        assert decision.recommendations[0] == "Safe to migrate under the configured policy."
        assert decision.recommendations[-1] == _SKIPPED_LINE

    def test_fail(self) -> None:
        decision = _decide(
            [_comparison(severity="critical")],
            [_record(example_id=f"b{i}", delta=-0.5) for i in range(5)],
            (_SKIPPED_SEMANTIC,),
        )
        assert decision.verdict == "fail"
        assert _SKIPPED_LINE in decision.recommendations

    def test_no_policy_decision_carries_them(self) -> None:
        decision = inconclusive_decision(
            run_id="r1",
            source_model="src",
            target_model="tgt",
            comparisons=[],
            records=[],
            calls=[],
            skipped_evaluators=(_SKIPPED_SEMANTIC,),
        )
        assert _SKIPPED_LINE in decision.recommendations
```

If `test_pass` / `test_fail` don't reach the stated verdict with these inputs, copy the record/comparison shape from an existing test in this file that asserts `"pass"` / `"fail"`, keeping the `skipped_evaluators` argument and the assertions on `_SKIPPED_LINE`.

Append to `tests/unit/test_analyze_command.py` inside the class that holds `test_all_advisory_inconclusive_prints_reason_and_fix` (import `SkippedEvaluator` from `evalshift_cli.runner.models`):

```python
    def test_skipped_evaluator_reaches_the_decision_and_the_terminal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cwd, run_id = _scaffold(tmp_path)
        config_path = cwd / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + "\n        migration_policy:\n"
            + "          max_overall_regression_rate: 1.0\n",
            encoding="utf-8",
        )
        run_dir = cwd / ".evalshift" / "runs" / run_id
        state = read_state(run_dir)
        state.skipped_evaluators = [
            SkippedEvaluator(
                evaluator_name="semantic.cosine",
                kind="semantic",
                label="semantic",
                model="openai/text-embedding-3-small",
                env_vars=["OPENAI_API_KEY"],
            )
        ]
        write_state(run_dir, state)
        monkeypatch.chdir(cwd)

        result = runner.invoke(app, ["analyze", run_id])

        assert result.exit_code == 0, result.stdout
        line = (
            "semantic was skipped: no API key for openai/text-embedding-3-small. "
            "Export OPENAI_API_KEY to enable it."
        )
        decision = json.loads((run_dir / "migration_decision.json").read_text(encoding="utf-8"))
        assert line in decision["recommendations"]
        assert line in " ".join(result.stdout.split())
```

Append to `tests/unit/test_bundle_shape.py` (import `read_state`, `write_state` from `evalshift_cli.runner.checkpoint` and `SkippedEvaluator` from `evalshift_cli.runner.models`):

```python
@pytest.mark.parametrize("with_policy", [False, True])
def test_the_bundle_decision_names_skipped_evaluators(
    run_fixture: RunFixture, with_policy: bool
) -> None:
    """The bundle recomputes the decision; it must say what ``analyze`` says."""
    state = read_state(run_fixture.run_dir)
    state.skipped_evaluators = [
        SkippedEvaluator(
            evaluator_name="semantic.cosine",
            kind="semantic",
            label="semantic",
            model="openai/text-embedding-3-small",
            env_vars=["OPENAI_API_KEY"],
        )
    ]
    write_state(run_fixture.run_dir, state)
    config = _with_migration_policy(run_fixture) if with_policy else run_fixture.config
    recommendations = _decision(_load(run_fixture.build(config_path=config).path))[
        "recommendations"
    ]
    assert any(r.startswith("semantic was skipped:") for r in recommendations)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_policy.py -k "PromotionAdvice or SkippedEvaluatorLines" tests/unit/test_bundle_shape.py -k skipped tests/unit/test_analyze_command.py -k skipped -v`
Expected: FAIL — `unexpected keyword argument 'skipped_evaluators'`.

- [ ] **Step 3: Implement in `policy.py`**

Imports: add `MIN_N_RELIABLE` to the existing `evalshift_cli.analysis.statistics` import, `SkippedEvaluator` to the `evalshift_cli.runner.models` import, and `from evalshift_cli.evaluators.keys import LLM_JUDGE_KIND, skip_recommendation`.

`evaluate_migration_policy`: add the parameter after `dropped_params` and document it:

```python
    skipped_evaluators: Sequence[SkippedEvaluator] = (),
```

```
        skipped_evaluators: ``state.json``'s advisory evaluators the evaluate
            stage skipped for a missing API key; each gets one
            ``recommendations`` line naming the env var, whatever the verdict.
```

In the `_recommendations(...)` call add `promotion=_promotion_advice(advisory_comparisons),`, and in the `recommendations=[...]` list add as the last element:

```python
            # Config gaps the user can fix in one export — appended under
            # every verdict, never substituted for the verdict's own advice.
            *_skipped_notes(skipped_evaluators),
```

`inconclusive_decision`: add the same parameter and append `*_skipped_notes(skipped_evaluators),` as the last element of its `recommendations` list.

`_recommendations`: add parameter `promotion: list[str] | None = None` and use it where `enable_blocking` is used today:

```python
        if no_blocking_records:
            return promotion or [enable_blocking]
```

```python
    if no_blocking_records:
        out.extend(promotion or [enable_blocking])
```

New helpers (next to `_recommendations`):

```python
def _promotion_advice(advisory_comparisons: Sequence[ComparisonResult]) -> list[str]:
    """Say, from each advisory judge's own ``n``, whether it is ready to gate.

    Only consulted when nothing gates (see :func:`_recommendations`), so a
    judge deliberately kept advisory beside a real gate is never nagged.
    Reads the overall (``all``) slice: one comparison per prompt, and the
    smallest ``n`` decides, because a gate is only as reliable as its
    thinnest prompt. ``semantic`` is never suggested: it measures drift, not
    correctness.
    """
    by_judge: dict[str, list[ComparisonResult]] = {}
    for c in advisory_comparisons:
        is_judge = c.kind == LLM_JUDGE_KIND or (
            not c.kind and c.evaluator_name.startswith(f"{LLM_JUDGE_KIND}.")
        )
        if is_judge and c.slice_name == "all":
            by_judge.setdefault(c.evaluator_name, []).append(c)
    out: list[str] = []
    for name in sorted(by_judge):
        rows = by_judge[name]
        smallest = min(rows, key=lambda c: (c.n, c.prompt_id))
        criterion = name.removeprefix(f"{LLM_JUDGE_KIND}.")
        if smallest.n >= MIN_N_RELIABLE:
            out.append(
                f"The {criterion} judge scored at least {smallest.n} pairs on every prompt "
                "— enough to gate. Set blocking: true on it in evalshift.yaml to get a "
                "pass/fail verdict."
            )
            continue
        where = (
            f"{smallest.prompt_id} has {smallest.n}"
            if len({c.prompt_id for c in rows}) > 1
            else f"this run has {smallest.n}"
        )
        out.append(
            f"The {criterion} judge is advisory, so it does not gate this run. It becomes "
            f"reliable at {MIN_N_RELIABLE} pairs per prompt; {where}. Collect more "
            "examples, then set blocking: true on it."
        )
    return out


def _skipped_notes(skipped: Sequence[SkippedEvaluator]) -> list[str]:
    return [
        skip_recommendation(label=s.label, model=s.model, env_vars=s.env_vars, note=s.note)
        for s in skipped
    ]
```

Callers — pass `skipped_evaluators=state.skipped_evaluators` in:
- `analyze.py:151` (`evaluate_migration_policy(...)`),
- `hosted/bundle.py` both calls (`evaluate_migration_policy(...)` and `inconclusive_decision(...)`),
- `insights/stage.py` `_decision` (`inconclusive_decision(...)`).

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_policy.py tests/unit/test_analyze_command.py tests/unit/test_bundle_shape.py tests/unit/test_compare_command.py -q`
Expected: PASS (the existing "Set blocking: true" tests still pass: their advisory evaluators are structural/semantic, not judges).

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/analysis/policy.py src/evalshift_cli/cli/commands/analyze.py \
  src/evalshift_cli/hosted/bundle.py src/evalshift_cli/insights/stage.py tests/unit/
git commit -m "feat(policy): advise promoting the judge from its own n; name skipped evaluators

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `doctor` — an `evaluator keys` row

**Files:**
- Modify: `src/evalshift_cli/cli/commands/doctor.py` (constant, `_evaluator_key_checks`, `run_checks`, `__all__`)
- Test: `tests/unit/test_doctor.py`

**Interfaces:**
- Consumes: `evaluator_key_gaps`, `EvaluatorKeyGap` (Task 2).
- Produces: `EVALUATOR_KEYS_CHECK: Final = "evaluator keys"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_doctor.py` (import `EVALUATOR_KEYS_CHECK` from `evalshift_cli.cli.commands.doctor`):

```python
def _key_rows(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.name == EVALUATOR_KEYS_CHECK]


class TestEvaluatorKeysCheck:
    def test_blocking_keyless_judge_fails(self, tmp_path: Path) -> None:
        _write_judge_config(tmp_path, judges=("openai/gpt-4.1-mini",))
        (row,) = _key_rows(run_checks(cwd=tmp_path, env=_empty_env()))
        assert row.status == "fail"
        assert "openai/gpt-4.1-mini" in row.detail
        assert "OPENAI_API_KEY" in row.detail

    def test_advisory_keyless_judge_warns(self, tmp_path: Path) -> None:
        _write_judge_config(tmp_path, judges=("openai/gpt-4.1-mini",))
        path = tmp_path / CONFIG_FILENAME
        path.write_text(path.read_text(encoding="utf-8") + "      blocking: false\n", encoding="utf-8")
        (row,) = _key_rows(run_checks(cwd=tmp_path, env=_empty_env()))
        assert row.status == "warn"
        assert "skipped until then" in row.detail

    def test_ok_when_every_key_is_set(self, tmp_path: Path) -> None:
        _write_judge_config(tmp_path, judges=("openai/gpt-4.1-mini",))
        (row,) = _key_rows(run_checks(cwd=tmp_path, env={"OPENAI_API_KEY": "k"}))
        assert row.status == "ok"

    def test_silent_without_judge_or_semantic(self, tmp_path: Path) -> None:
        _write_judge_config(tmp_path, judges=())
        assert _key_rows(run_checks(cwd=tmp_path, env=_empty_env())) == []

    def test_silent_without_config(self, tmp_path: Path) -> None:
        assert _key_rows(run_checks(cwd=tmp_path, env=_empty_env())) == []
```

Also fix the existing `TestJudgeFamilyCheck.test_warning_never_fails_the_command`: it runs the real `doctor` on a blocking Gemini judge, which now (correctly) fails without a key. Add `monkeypatch.setenv("GEMINI_API_KEY", "test-key")` before `runner.invoke`.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_doctor.py -k "EvaluatorKeysCheck" -v`
Expected: FAIL — `ImportError: cannot import name 'EVALUATOR_KEYS_CHECK'`.

- [ ] **Step 3: Implement**

In `doctor.py`, import `from evalshift_cli.evaluators.keys import EvaluatorKeyGap, evaluator_key_gaps`, then next to `JUDGE_FAMILY_CHECK`:

```python
# Row name for the judge / embedding key check (one row per keyless model).
EVALUATOR_KEYS_CHECK: Final = "evaluator keys"
```

In `run_checks`, after `results.extend(_judge_family_checks(cwd))`:

```python
    results.extend(_evaluator_key_checks(cwd, env))
```

New function after `_judge_family_checks`:

```python
def _evaluator_key_checks(cwd: Path, env: Mapping[str, str]) -> list[CheckResult]:
    """Report every judge / embedding model that has no API key.

    Checks the top-level evaluator set and every named suite's resolved set,
    since a suite can bring its own judge. ``fail`` for a blocking evaluator
    (``evaluate`` refuses it), ``warn`` for an advisory one (``evaluate``
    skips it), one ``ok`` row when every such model has a key. Silent with
    no loadable config, or with no ``semantic`` / ``llm_judge`` anywhere.
    """
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return []
    try:
        cfg = load_config(cfg_path)
    except ConfigError:
        return []
    sets = [cfg.evaluators, *(cfg.evaluators_for(name) for name in cfg.suites)]
    if not any(s.semantic is not None or s.llm_judge for s in sets):
        return []
    gaps: dict[tuple[str, str], EvaluatorKeyGap] = {}
    for evaluators in sets:
        for gap in evaluator_key_gaps(evaluators, env):
            key = (gap.evaluator_name, gap.model)
            prior = gaps.get(key)
            if prior is None or (gap.blocking and not prior.blocking):
                gaps[key] = gap
    if not gaps:
        return [
            CheckResult(
                name=EVALUATOR_KEYS_CHECK,
                status="ok",
                detail="judge and embedding models have API keys",
            ),
        ]
    return [
        CheckResult(
            name=EVALUATOR_KEYS_CHECK,
            status="fail" if gap.blocking else "warn",
            detail=(
                f"{gap.label} uses {gap.model}; export {' or '.join(gap.env_vars)}"
                + ("" if gap.blocking else " (skipped until then)")
            ),
        )
        for gap in gaps.values()
    ]
```

Add `"EVALUATOR_KEYS_CHECK"` to `__all__`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_doctor.py tests/unit/test_compare_command.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/cli/commands/doctor.py tests/unit/test_doctor.py
git commit -m "feat(doctor): report judge and embedding models with no API key

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `init` writes `semantic` active for every provider

**Files:**
- Modify: `src/evalshift_cli/cli/commands/init.py` (`_SEMANTIC_BLOCK`, delete `_SEMANTIC_BLOCK_DISABLED`, `render_minimal_config`, new `scaffold_embedding_model`, `init` next steps)
- Modify: `src/evalshift_cli/cli/commands/_scaffold.py` (CI template env line, `render_ci_workflow`)
- Test: `tests/unit/test_init.py`

**Interfaces:**
- Consumes: `missing_api_keys` (Task 1), `PROVIDER_ENV_VARS`, `resolve_model`.
- Produces:
  - `scaffold_embedding_model(provider: str, env: Mapping[str, str]) -> str` in `init.py`.
  - `render_minimal_config(*, profile: str, provider: str = "gemini", env: Mapping[str, str] | None = None) -> str`.
  - `render_ci_workflow(*, provider: str, version: str, embedding_api_key: str | None = None) -> str`.

- [ ] **Step 1: Isolate the tests from the developer's environment**

The scaffold now depends on which keys are set. In `tests/unit/test_init.py`, make `in_tmp` clear them:

```python
@pytest.fixture
def in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run inside ``tmp_path`` with no provider keys set."""
    monkeypatch.chdir(tmp_path)
    for keys in PROVIDER_ENV_VARS.values():
        for key in keys:
            monkeypatch.delenv(key, raising=False)
    return tmp_path
```

- [ ] **Step 2: Write the failing tests**

Replace `test_anthropic_provider_comments_out_semantic` and `test_deepseek_provider_writes_deepseek_ids_and_comments_out_semantic` with:

```python
    @pytest.mark.parametrize(
        ("env_key", "expected"),
        [
            (None, "openai/text-embedding-3-small"),
            ("OPENAI_API_KEY", "openai/text-embedding-3-small"),
            ("GEMINI_API_KEY", "gemini/gemini-embedding-001"),
            ("GOOGLE_API_KEY", "gemini/gemini-embedding-001"),
        ],
    )
    @pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
    def test_providers_without_embeddings_borrow_one_and_keep_semantic_on(
        self,
        in_tmp: Path,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        env_key: str | None,
        expected: str,
    ) -> None:
        if env_key is not None:
            monkeypatch.setenv(env_key, "k")
        result = runner.invoke(app, ["init", "--provider", provider])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == expected
        assert cfg.evaluators.semantic.blocking is False
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# semantic:" not in body
        label = {"anthropic": "Anthropic", "deepseek": "DeepSeek"}[provider]
        assert f"{label} has no embeddings endpoint" in body

    def test_anthropic_keeps_its_own_ids(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "anthropic"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "claude-sonnet-5"
        assert cfg.evaluators.llm_judge[0].judge_model == "claude-opus-4-8"

    def test_deepseek_keeps_its_own_ids(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "deepseek"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "deepseek-flash"
        assert cfg.evaluators.llm_judge[0].judge_model == "deepseek-v4-pro"
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "Anthropic has no" not in body

    def test_borrowed_embedding_without_its_key_is_named_in_next_steps(
        self, in_tmp: Path
    ) -> None:
        result = runner.invoke(app, ["init", "--provider", "anthropic"])
        flat = " ".join(result.stdout.split())
        assert (
            "semantic uses openai/text-embedding-3-small — export OPENAI_API_KEY to "
            "enable it; compare skips it until then."
        ) in flat

    def test_borrowed_embedding_with_its_key_needs_no_next_step(
        self, in_tmp: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        result = runner.invoke(app, ["init", "--provider", "anthropic"])
        assert "skips it until then" not in result.stdout

    def test_own_embedding_needs_no_next_step(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "gemini"])
        assert "skips it until then" not in result.stdout

    def test_render_is_pure_given_env(self) -> None:
        text = render_minimal_config(
            profile="model-upgrade", provider="anthropic", env={"GEMINI_API_KEY": "k"}
        )
        assert yaml.safe_load(text)["evaluators"]["semantic"]["embedding_model"] == (
            "gemini/gemini-embedding-001"
        )
```

(Import `render_minimal_config` alongside `_PROVIDER_MODELS, PROVIDERS` from `evalshift_cli.cli.commands.init`.)

In the CI workflow test class (the one with `self._workflow`), add:

```python
    def test_borrowed_embedding_key_is_wired_as_a_secret(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp, "--provider", "anthropic")
        assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in body
        assert "__EMBEDDING_API_KEY__" not in body

    def test_own_embedding_adds_no_second_secret(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp, "--provider", "gemini")
        assert "OPENAI_API_KEY" not in body
        assert "__EMBEDDING_API_KEY__" not in body
```

- [ ] **Step 3: Run to verify they fail**

Run: `pytest tests/unit/test_init.py -v`
Expected: the new tests FAIL (semantic is `None` for anthropic/deepseek; no next-steps line; no second secret).

- [ ] **Step 4: Implement `init.py`**

Imports: add `import os`, `from collections.abc import Mapping`, and `from evalshift_cli.models.registry import PROVIDER_ENV_VARS, missing_api_keys, resolve_model`.

Replace `_SEMANTIC_BLOCK` and delete `_SEMANTIC_BLOCK_DISABLED`:

```python
_SEMANTIC_BLOCK: Final = """\
  # Embedding-based drift score between source and target outputs. Advisory
  # (blocking: false): it reports and ranks drift but never fails a run by
  # itself — cosine distance can't tell "reworded" from "wrong".
{borrowed_note}  semantic:
    embedding_model: {embedding_model}
    # Cosine similarity below which a target output is flagged as a
    # semantic regression. Defaults to 0.9; lower it to tolerate more drift.
    min_similarity: 0.9
    blocking: false"""

_BORROWED_EMBEDDING_NOTE: Final = """\
  # {provider_label} has no embeddings endpoint, so this uses {embedding_label}'s and
  # needs {key}. Without it, `evalshift compare` skips this evaluator and says so.
"""

# Tried in order when the provider has no embeddings endpoint of its own; the
# first key already exported decides, so a user with OpenAI or Gemini set up
# gets a working semantic evaluator with nothing to edit.
_BORROWED_EMBEDDINGS: Final[tuple[tuple[str, str], ...]] = (
    ("OPENAI_API_KEY", "openai/text-embedding-3-small"),
    ("GEMINI_API_KEY", "gemini/gemini-embedding-001"),
    ("GOOGLE_API_KEY", "gemini/gemini-embedding-001"),
)
_DEFAULT_BORROWED_EMBEDDING: Final = "openai/text-embedding-3-small"
_PROVIDER_LABELS: Final[dict[str, str]] = {"anthropic": "Anthropic", "deepseek": "DeepSeek"}
_EMBEDDING_LABELS: Final[dict[str, str]] = {"openai": "OpenAI", "google": "Gemini"}
```

Add:

```python
def scaffold_embedding_model(provider: str, env: Mapping[str, str]) -> str:
    """Pick the embedding model ``init`` writes for ``provider``.

    The provider's own when it has one; otherwise the first of OpenAI's or
    Gemini's whose key is already set, else OpenAI's — written active either
    way, so ``compare`` names the missing key instead of the YAML hiding it.

    Args:
        provider: One of :data:`PROVIDERS`.
        env: Environment mapping, typically ``os.environ``.
    """
    own = _PROVIDER_MODELS[provider]["embedding_model"]
    if own:
        return own
    for key, model in _BORROWED_EMBEDDINGS:
        if env.get(key):
            return model
    return _DEFAULT_BORROWED_EMBEDDING


def _borrowed_embedding_key(provider: str, embedding_model: str) -> str | None:
    """Primary env var of a borrowed embedding provider, or ``None`` when it is the provider's own."""
    if _PROVIDER_MODELS[provider]["embedding_model"]:
        return None
    return PROVIDER_ENV_VARS[resolve_model(embedding_model).provider][0]
```

`render_minimal_config` — new signature `(*, profile: str, provider: str = "gemini", env: Mapping[str, str] | None = None)`, add to its docstring `env: Environment used to pick a borrowed embedding model; defaults to os.environ.`, and replace the `semantic_block = (...)` expression with:

```python
    embedding_model = scaffold_embedding_model(provider, os.environ if env is None else env)
    borrowed_key = _borrowed_embedding_key(provider, embedding_model)
    borrowed_note = (
        _BORROWED_EMBEDDING_NOTE.format(
            provider_label=_PROVIDER_LABELS[provider],
            embedding_label=_EMBEDDING_LABELS[resolve_model(embedding_model).provider],
            key=borrowed_key,
        )
        if borrowed_key is not None
        else ""
    )
    semantic_block = _SEMANTIC_BLOCK.format(
        embedding_model=embedding_model, borrowed_note=borrowed_note
    )
```

In `init()`, after `target.mkdir(...)`:

```python
    embedding_model = scaffold_embedding_model(provider, os.environ)
    embedding_key = _borrowed_embedding_key(provider, embedding_model)
```

and pass `embedding_api_key=embedding_key` to `render_ci_workflow(...)`. After the three numbered next-step lines:

```python
    if embedding_key is not None and missing_api_keys(embedding_model, os.environ):
        console.print(
            f"     [bold]semantic[/bold] uses [cyan]{embedding_model}[/cyan] — export "
            f"[bold]{embedding_key}[/bold] to enable it; compare skips it until then.",
        )
```

In the `if ci:` next-steps message, change `f"[bold]{PROVIDER_API_KEY_ENVS[provider]}[/bold] and "` to

```python
            f"[bold]{PROVIDER_API_KEY_ENVS[provider]}[/bold]"
            + (f", [bold]{embedding_key}[/bold] (for semantic)" if embedding_key else "")
            + " and "
```

- [ ] **Step 5: Implement `_scaffold.py`**

In `CI_WORKFLOW_TEMPLATE`'s `evalshift` job `env:` block, add the line directly under `      __PROVIDER_API_KEY__: ${{ secrets.__PROVIDER_API_KEY__ }}`:

```
      __EMBEDDING_API_KEY__: ${{ secrets.__EMBEDDING_API_KEY__ }}
```

Add above `render_ci_workflow`:

```python
# Present only when ``init`` borrowed another provider's embedding model; an
# unset secret arrives as "" and the CLI then skips semantic with a warning.
_EMBEDDING_KEY_LINE: Final = "      __EMBEDDING_API_KEY__: ${{ secrets.__EMBEDDING_API_KEY__ }}\n"
```

and replace `render_ci_workflow`:

```python
def render_ci_workflow(
    *, provider: str, version: str, embedding_api_key: str | None = None
) -> str:
    """Render the ``--ci`` GitHub Actions workflow for a provider.

    Args:
        provider: Key of :data:`PROVIDER_API_KEY_ENVS`; selects the provider
            API key the workflow wires through as a secret.
        version: CLI version to pin via the action's ``evalshift-version``
            input, normally :data:`evalshift_cli.__version__`.
        embedding_api_key: The env var of a borrowed embedding provider, wired
            through as a second secret; ``None`` when the provider embeds itself.
    """
    template = (
        CI_WORKFLOW_TEMPLATE.replace(_EMBEDDING_KEY_LINE, "")
        if embedding_api_key is None
        else CI_WORKFLOW_TEMPLATE.replace("__EMBEDDING_API_KEY__", embedding_api_key)
    )
    return template.replace("__PROVIDER_API_KEY__", PROVIDER_API_KEY_ENVS[provider]).replace(
        "__EVALSHIFT_VERSION__", version
    )
```

Confirm `Final` is imported in `_scaffold.py` (it is, for `PROVIDER_API_KEY_ENVS`).

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/test_init.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_remote.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/evalshift_cli/cli/commands/init.py src/evalshift_cli/cli/commands/_scaffold.py tests/unit/test_init.py
git commit -m "feat(init): scaffold semantic on for every provider, borrowing an embedder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: docs, changelog, full verification

**Files:**
- Modify: `DOCS.md`, `llms-full.txt`, `docs/configuration.md`, `docs/faq.md`, `CHANGELOG.md`
- Modify (if they describe the scaffold's evaluators): the `EVALSHIFT.md` guide source under `src/evalshift_cli/cli/commands/_agents.py` or its template

- [ ] **Step 1: Find every stale statement**

Run:

```bash
grep -n -i "commented out for anthropic\|ships commented out\|no embedding endpoint\|Set blocking: true\|evaluator_coverage\|skipped_evaluators\|semantic" \
  DOCS.md llms-full.txt docs/configuration.md docs/faq.md docs/agents.md | head -80
grep -rn -i "semantic" src/evalshift_cli/cli/commands/_agents.py src/evalshift_cli/**/templates 2>/dev/null | head
```

Known stale lines: `llms-full.txt:140` (`commented out for anthropic/deepseek`) and `llms-full.txt:762-763` (DeepSeek scaffold ships semantic commented out).

- [ ] **Step 2: Update the docs**

Make these statements true everywhere they appear, in each file's own style (no new sections unless the file has no home for it):

1. **init scaffold:** `semantic` and `llm_judge` are active and advisory for every provider. Anthropic/DeepSeek borrow an embedding model: OpenAI's if `OPENAI_API_KEY` is set, else Gemini's if `GEMINI_API_KEY`/`GOOGLE_API_KEY` is set, else OpenAI's; `init --ci` wires the borrowed key as a second secret.
2. **Missing evaluator keys:** before any model call, `compare`/`run` check every `semantic`/`llm_judge` model's key. Advisory → skipped with `⚠ <label> skipped: no API key for <model> — export <VAR> to enable it.`; blocking → exit 1. A keyless `semantic` counts as blocking when a blocking `tool_arguments` entry uses the `semantic` strategy; with `auto`, arguments fall back to difflib. `evaluate` applies the same rule. `doctor` has an `evaluator keys` row (fail/warn/ok).
3. **`state.json`:** new `skipped_evaluators` list (`evaluator_name`, `kind`, `label`, `model`, `env_vars`, `note`) — add it to the artefact table next to `evaluator_coverage` in `DOCS.md`.
4. **Recommendations:** with nothing gating, an advisory judge gets promotion advice from its smallest per-prompt `n` vs. 20 (both wordings); each skipped evaluator adds one line under every verdict. Update the FAQ entry on `inconclusive` (DOCS.md ~line 991 and `docs/faq.md`) to mention the judge advice.
5. `docs/configuration.md` `evaluators.semantic`: the scaffold default and the `tool_arguments` interaction.

Add to `CHANGELOG.md` under `## [Unreleased]`:

```markdown
### Added

- `evalshift init` writes the `semantic` evaluator active for every provider.
  Anthropic and DeepSeek have no embeddings endpoint, so the scaffold borrows
  OpenAI's or Gemini's — whichever key you already have — and `init --ci`
  wires that key as a second secret.
- `compare`, `run` and `evaluate` check every judge and embedding model's API
  key before the first call. An advisory evaluator without one is skipped with
  a line naming the env var to export; a blocking one stops the run. The skip
  is recorded in `state.json` (`skipped_evaluators`) and repeated in the
  verdict's recommendations, the HTML report and the bundle.
- `doctor` gains an `evaluator keys` row.
- When nothing gates the verdict, the recommendation now names the advisory
  judge and says, from its own sample size, whether it is ready for
  `blocking: true` (20 pairs per prompt) or how many examples it still needs.

### Fixed

- A judge or embedding model with no API key no longer fails silently on every
  pair: it is skipped (advisory) or refused (blocking) before any call.
- The bare `text-embedding-3-small` default now resolves to OpenAI, so its key
  is checked.
```

- [ ] **Step 3: Run the docs-currency tests**

Run: `pytest tests/unit/test_docs_currency.py tests/unit/test_init.py -q`
Expected: PASS.

- [ ] **Step 4: Full verification**

Run (venv active):

```bash
ruff check . && ruff format --check . && mypy --strict src/evalshift_cli && pytest -q
pre-commit run --all-files
make ci
```

Expected: all green. Paste the tail of each output into the handoff.

- [ ] **Step 5: Commit**

```bash
git add DOCS.md llms-full.txt docs/ CHANGELOG.md src/evalshift_cli/cli/commands/_agents.py
git commit -m "docs: judge and semantic on by default; missing evaluator keys

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Drop `_agents.py` from `git add` if Step 1 found nothing to change there.)
