# Bounded Capture Suites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `captures.max_age` + `captures.max_cases` turn each captured suite into a bounded, deterministic, route-balanced sample of recent production on the current source model, refreshed by `capture sync`, with staleness visible in `doctor` and `compare`.

**Architecture:** Two pure modules carry the logic — `captures/staleness.py` (what makes a case stale: age vs `max_age`, model vs `defaults.source_model` through `models/matching.py`) and `captures/selection.py` (`plan_suite`: fill free slots route-balanced in `sha256` order, retire stale cases only to make room, trim over-cap suites). `capture sync` becomes collect → plan → apply when `max_cases` is set; with neither limit set it is byte-for-byte today's sync. Promoted case files gain `captured_at` / `source_model`; `captures/freshness.py` reads them for the doctor rows and the compare lines.

**Tech Stack:** Python 3.11+, pydantic v2, Typer + Rich, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-16-bounded-capture-suites-design.md` — read it first; this plan argues from it.

## Global Constraints

- Python floor 3.11 (no 3.12+ syntax). Every module starts with `from __future__ import annotations`; public functions/classes get Google-style docstrings; `mypy --strict src/evalshift_cli` stays clean.
- **Opt-in rule (spec §1):** every new sync behaviour — the too-old / other-model filters, planning, retirement — and every freshness finding applies **only when `captures.max_age` or `captures.max_cases` is set**. `defaults.source_model` alone changes nothing. Writing and backfilling `captured_at` / `source_model` on case files happens always.
- Config key names, verbatim: `captures.max_age` (duration `N` + `m|h|d`, > 0), `captures.max_cases` (int ≥ 1). Both default `None`. Nothing is added to `defaults` (the hosted `eval_config_hash` hashes it).
- `SuiteExample` / `golden.jsonl` do **not** change. Only `PromotedCase` gains fields.
- The CLI never writes to or deletes from a bucket.
- No new CLI flags (durable settings live in `evalshift.yaml`; `--since` stays the one-off fetch override).
- User-facing strings, verbatim:
  - config error: `expected a duration like 30m, 24h or 14d`
  - fetch skip label: `older than captures.max_age (14d)` (the `14d` is `format_duration(max_age)`)
  - per-suite line: `  {suite}: {total} cases — kept {kept}, retired {n}[ ({reasons})], added {added} of {eligible} eligible[; not eligible: {reasons}]`; with `max_age` only: `  {suite}: not eligible: {reasons}`; reasons are `N older than 14d`, `N on another model`, `N over max_cases`, zero counts left out
  - guardrail: `{suite}: none of {n} case(s) and capture(s) were recorded on {source_model} (seen: {models}); not filtering by model — check defaults.source_model`
  - backfill clause: `, backfilled capture time and model on {n} case(s)`
  - limits skipped (`--offline` only): `could not read {path} ({reason}); syncing without captures.max_age / captures.max_cases.`
  - doctor rows `suite freshness` / `captures limits`; finding texts are in Task 9.
- Gate ("green" = all four, as CI): `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest -q` (coverage floor 90). `make ci` is the exact CI mirror.
- Work on branch `feat/bounded-capture-suites` in a worktree (superpowers:using-git-worktrees). Before every commit confirm the branch is not `main`. Conventional Commits, each message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `--no-verify`; if the pre-commit hook cannot find its Python in the worktree, commit with `uv run --with pre-commit git commit …`.
- `docs/superpowers/` is gitignored: add the spec and this plan with `git add -f` (Task 11).
- Subagents implementing or reviewing tasks run with `model: opus`.

## Review Focus

- Re-running `capture sync` with nothing new must not touch a single file — Task 6 `test_planning_twice_changes_nothing`, Task 8 `test_syncing_again_with_nothing_new_changes_nothing`.
- A `defaults.source_model` that no capture ran on (typo, alias, mid-rollout) must neither empty nor freeze a suite — Task 8 `test_a_source_model_no_capture_ran_on_filters_nothing`, Task 10 `test_a_wrong_source_model_trips_the_guardrail`.
- A conversation larger than the free slots is never split — Task 6 `test_a_group_is_added_whole_or_not_at_all`, Task 8 `test_a_conversation_enters_whole_or_not_at_all`.
- `capture sync --suite alpha` must not retire or trim any other suite — Task 8 `test_suite_flag_leaves_other_suites_alone`.
- Capture timestamps in `Z`, a non-UTC offset, naive, or garbage — Task 5 `test_parse_instant_normalises_to_utc`, `test_parse_instant_is_none_for_anything_unparseable`.

---

### Task 1: Duration grammar and the two config fields

**Files:**
- Create: `src/evalshift_cli/utils/duration.py`
- Modify: `src/evalshift_cli/captures/remote.py` (`parse_since`, drop `_DURATION`)
- Modify: `src/evalshift_cli/config/models.py` (`CapturesConfig`, imports)
- Test: `tests/unit/test_utils_duration.py` (new), `tests/unit/test_config_captures.py`

**Interfaces:**
- Produces: `utils.duration.DURATION_FORMS: str = "30m, 24h or 14d"`, `parse_duration(text: str) -> timedelta`, `format_duration(delta: timedelta) -> str`; `CapturesConfig.max_age: timedelta | None`, `CapturesConfig.max_cases: int | None`.

- [ ] **Step 1: Write the failing duration tests**

Create `tests/unit/test_utils_duration.py`:

```python
"""The duration grammar shared by ``--since`` and ``captures.max_age``."""

from __future__ import annotations

from datetime import timedelta

import pytest

from evalshift_cli.utils.duration import format_duration, parse_duration


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30m", timedelta(minutes=30)),
        ("24h", timedelta(hours=24)),
        ("14d", timedelta(days=14)),
        (" 7d ", timedelta(days=7)),
        ("0d", timedelta(0)),
    ],
)
def test_parse_duration_accepts_minutes_hours_days(text: str, expected: timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "14", "2w", "1.5d", "-1d", "d", "14 d", "P14D"])
def test_parse_duration_rejects_anything_else(text: str) -> None:
    with pytest.raises(ValueError, match="expected a duration like 30m, 24h or 14d"):
        parse_duration(text)


def test_parse_duration_overflow_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="too large"):
        parse_duration("9999999999999d")


@pytest.mark.parametrize(
    ("delta", "text"),
    [(timedelta(days=14), "14d"), (timedelta(hours=36), "36h"), (timedelta(minutes=90), "90m")],
)
def test_format_duration_uses_the_largest_whole_unit(delta: timedelta, text: str) -> None:
    assert format_duration(delta) == text
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_utils_duration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalshift_cli.utils.duration'`

- [ ] **Step 3: Implement `utils/duration.py`**

```python
"""The duration grammar shared by ``--since`` and ``captures.max_age``: ``N`` + ``m|h|d``."""

from __future__ import annotations

import re
from datetime import timedelta

#: How an error names the accepted forms.
DURATION_FORMS = "30m, 24h or 14d"

_DURATION = re.compile(r"(\d+)([mhd])")
_UNITS = {"m": timedelta(minutes=1), "h": timedelta(hours=1), "d": timedelta(days=1)}


def parse_duration(text: str) -> timedelta:
    """Parse ``30m`` / ``24h`` / ``14d`` (surrounding whitespace allowed) into a timedelta.

    Raises:
        ValueError: when ``text`` is not that form, or is too large for a timedelta.
    """
    match = _DURATION.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"expected a duration like {DURATION_FORMS}, got {text!r}")
    try:
        return _UNITS[match.group(2)] * int(match.group(1))
    except OverflowError as exc:
        raise ValueError(f"duration {text!r} is too large") from exc


def format_duration(delta: timedelta) -> str:
    """Render ``delta`` in the largest whole unit :func:`parse_duration` reads (``14d``, ``36h``)."""
    seconds = int(delta.total_seconds())
    for suffix, unit in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % unit == 0:
            return f"{seconds // unit}{suffix}"
    return f"{seconds}s"
```

- [ ] **Step 4: Move `parse_since` onto it**

In `src/evalshift_cli/captures/remote.py` delete the `_DURATION = re.compile(r"(\d+)([mhd])")` line, import `from evalshift_cli.utils.duration import parse_duration`, and replace the body of `parse_since` (keep its signature and docstring):

```python
    reference = now if now is not None else datetime.now(UTC)
    cleaned = text.strip()
    try:
        delta: timedelta | None = parse_duration(cleaned)
    except ValueError:
        delta = None
    try:
        if delta is not None:
            return reference - delta
        parsed = datetime.fromisoformat(cleaned)
    except (ValueError, OverflowError) as exc:  # OverflowError: a duration reaching before year 1
        raise ValueError(
            f"--since expects a duration like 30m, 24h, 7d or an ISO date/datetime, got {text!r}"
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
```

If `re` / `timedelta` imports become unused or newly needed, fix them (`ruff check` says which).

- [ ] **Step 5: Run the duration and `--since` tests**

Run: `uv run pytest tests/unit/test_utils_duration.py tests/unit/test_captures_remote.py -q`
Expected: PASS (the existing `parse_since` tests at `test_captures_remote.py:333-362` pin the unchanged behaviour).

- [ ] **Step 6: Write the failing config tests**

Append to `tests/unit/test_config_captures.py` (add `from datetime import timedelta` to the imports):

```python
def test_limits_default_to_off() -> None:
    cfg = EvalShiftConfig(prompts=[_PROMPT])
    assert cfg.captures.max_age is None
    assert cfg.captures.max_cases is None


def test_limits_parse_without_a_store() -> None:
    cfg = EvalShiftConfig(prompts=[_PROMPT], captures={"max_age": "14d", "max_cases": 200})
    assert cfg.captures.max_age == timedelta(days=14)
    assert cfg.captures.max_cases == 200
    assert cfg.captures.store is None


@pytest.mark.parametrize("value", ["2w", "14", 14, "0d", "P14D"])
def test_max_age_rejects_anything_but_a_positive_duration(value: object) -> None:
    with pytest.raises(ValidationError, match="max_age"):
        EvalShiftConfig(prompts=[_PROMPT], captures={"max_age": value})


@pytest.mark.parametrize("value", [0, -5])
def test_max_cases_must_be_positive(value: int) -> None:
    with pytest.raises(ValidationError, match="max_cases"):
        EvalShiftConfig(prompts=[_PROMPT], captures={"max_cases": value})


def test_loader_reports_max_age_error_at_its_location(tmp_path: Path) -> None:
    path = tmp_path / "evalshift.yaml"
    path.write_text(
        "prompts:\n  - id: p\n    detection: manual\n    content: hi\n"
        "captures:\n  max_age: 2w\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        load_config(path)
    [detail] = info.value.details
    assert detail.location == "captures.max_age"
    assert "30m, 24h or 14d" in detail.message
```

- [ ] **Step 7: Run them to see them fail**

Run: `uv run pytest tests/unit/test_config_captures.py -q`
Expected: FAIL — `Extra inputs are not permitted` for `max_age` / `max_cases`.

- [ ] **Step 8: Add the fields**

In `src/evalshift_cli/config/models.py`: add `from datetime import timedelta`, add `BeforeValidator` to the pydantic import, and `from evalshift_cli.utils.duration import DURATION_FORMS, parse_duration`. Above `class CapturesConfig` add:

```python
def _duration_from_yaml(value: object) -> object:
    """``captures.max_age`` is written ``14d``; a bare number would silently mean seconds."""
    if value is None or isinstance(value, timedelta):
        return value
    if not isinstance(value, str):
        raise ValueError(f"expected a duration like {DURATION_FORMS}")
    return parse_duration(value)
```

In `CapturesConfig`, extend the docstring's `Attributes:` with

```
        max_age: Captures recorded longer ago than this (``30m``, ``24h``, ``14d``) never enter a
            suite, and promoted cases that old are stale: ``capture sync`` replaces them as fresh
            captures arrive. Also the fetch window. ``None`` (the default): no age limit.
        max_cases: The most cases ``capture sync`` keeps in each captured suite. ``None`` (the
            default): suites grow with every sync. Setting either limit turns bounded suites on.
```

and add the fields and validator under `store`:

```python
    max_age: Annotated[timedelta | None, BeforeValidator(_duration_from_yaml)] = None
    max_cases: int | None = Field(default=None, ge=1)

    @field_validator("max_age")
    @classmethod
    def _positive_max_age(cls, value: timedelta | None) -> timedelta | None:
        if value is not None and value <= timedelta(0):
            raise ValueError(f"must be greater than 0 (expected a duration like {DURATION_FORMS})")
        return value
```

- [ ] **Step 9: Run the config tests, then the gate's static checks**

Run: `uv run pytest tests/unit/test_config_captures.py tests/unit/test_config_models.py tests/unit/test_config_loader.py -q && uv run mypy --strict src/evalshift_cli && uv run ruff check . && uv run ruff format --check .`
Expected: PASS / clean.

- [ ] **Step 10: Commit**

```bash
git add src/evalshift_cli/utils/duration.py src/evalshift_cli/captures/remote.py src/evalshift_cli/config/models.py tests/unit/test_utils_duration.py tests/unit/test_config_captures.py
git commit -m "feat(config): captures.max_age and captures.max_cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `init` scaffolds the limits

**Files:**
- Modify: `src/evalshift_cli/cli/commands/init.py` (new `CAPTURES_SCAFFOLD`, `render_minimal_config`)
- Modify: `tests/unit/test_cli_capture.py` (`_write_min_config`), `tests/unit/test_cli_capture_remote.py` (`_write_config`), `tests/unit/test_cli_capture_repair.py` (both `render_minimal_config` calls)
- Test: `tests/unit/test_init.py`

**Interfaces:**
- Consumes: `CapturesConfig.max_age` / `max_cases` (Task 1).
- Produces: `cli.commands.init.CAPTURES_SCAFFOLD: Final[str]` — the exact block, ending in `\n`; tests strip it with `.replace(CAPTURES_SCAFFOLD + "\n", "")` to get today's unbounded sync.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_init.py` (import `re`, `timedelta`, `yaml`, `CAPTURES_SCAFFOLD` from `evalshift_cli.cli.commands.init`, `SUITES_MARKER_BEGIN` from `evalshift_cli.cli.commands._suites`, `EvalShiftConfig` from `evalshift_cli.config.models` — skip any already imported):

```python
def test_scaffold_bounds_capture_suites() -> None:
    text = render_minimal_config(profile="model-upgrade")
    assert CAPTURES_SCAFFOLD in text
    cfg = EvalShiftConfig.model_validate(yaml.safe_load(text))
    assert cfg.captures.max_age == timedelta(days=14)
    assert cfg.captures.max_cases == 200
    assert cfg.captures.store is None
    # The managed suites region stays last: capture sync rewrites only the tail.
    block = re.search(r"(?m)^captures:$", text)
    assert block is not None
    assert block.start() < text.index(SUITES_MARKER_BEGIN)
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/unit/test_init.py::test_scaffold_bounds_capture_suites -q`
Expected: FAIL — `ImportError: cannot import name 'CAPTURES_SCAFFOLD'`

- [ ] **Step 3: Implement**

In `src/evalshift_cli/cli/commands/init.py`, after `_MINIMAL_YAML_BODY` add:

```python
#: The scaffold's ``captures:`` block, between the migration policy and the suites region.
#: Public so a test that needs today's unbounded sync can strip it from the scaffold.
CAPTURES_SCAFFOLD: Final = """\
# How captures become suites. Each suite stays a capped sample of production
# from the last max_age, recorded on defaults.source_model: `capture sync`
# swaps stale cases for fresh captures as they arrive. Does the SDK ship
# captures to a bucket (EVALSHIFT_CAPTURE_STORE on the agent)? Uncomment
# store: and name the same bucket.
captures:
  max_age: 14d
  max_cases: 200
  # store: s3://<bucket>/<prefix>
"""
```

and change the last line of `render_minimal_config` to

```python
    return (
        header
        + body.rstrip()
        + "\n\n"
        + policy.rstrip()
        + "\n\n"
        + CAPTURES_SCAFFOLD
        + "\n"
        + suites
        + "\n"
    )
```

- [ ] **Step 4: Keep the existing sync tests on unbounded sync**

In `tests/unit/test_cli_capture.py` import `CAPTURES_SCAFFOLD` beside `render_minimal_config` and change `_write_min_config` to:

```python
def _write_min_config(path: Path) -> None:
    """Write a minimal init-style config (with the managed suites markers).

    Without the scaffold's capture limits: these tests pin unbounded sync, and their
    fixtures' June 2026 captures are older than the scaffold's 14d max_age.
    """
    text = render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
    path.write_text(text, encoding="utf-8")
```

Apply the same `.replace(CAPTURES_SCAFFOLD + "\n", "")` to the `render_minimal_config(...)` result in `tests/unit/test_cli_capture_remote.py::_write_config` and to both `render_minimal_config(...)` calls in `tests/unit/test_cli_capture_repair.py` (importing `CAPTURES_SCAFFOLD` in each). `_write_config` then appends its own `captures:` block as before.

- [ ] **Step 5: Run init and every capture CLI test**

Run: `uv run pytest tests/unit/test_init.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_remote.py tests/unit/test_cli_capture_repair.py tests/unit/test_docs_currency.py -q`
Expected: PASS. If a `test_init.py` test pins the exact scaffold text, update it to include `CAPTURES_SCAFFOLD`.

- [ ] **Step 6: Commit**

```bash
git add src/evalshift_cli/cli/commands/init.py tests/unit/test_init.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_remote.py tests/unit/test_cli_capture_repair.py
git commit -m "feat(init): scaffold captures.max_age and captures.max_cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Model matching

**Files:**
- Create: `src/evalshift_cli/models/matching.py`
- Test: `tests/unit/test_model_matching.py` (new)

**Interfaces:**
- Consumes: `models.registry.resolve_model(id_or_alias: str) -> ModelMetadata` (existing).
- Produces: `bare_model_name(model_id: str) -> str`, `same_model(a: str, b: str) -> bool`, `model_filter_applies(recorded: Iterable[str], source_model: str | None) -> bool`, `describe_models(recorded: Iterable[str], *, limit: int = 3) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_model_matching.py`:

```python
"""Matching a capture's recorded model id against ``defaults.source_model``."""

from __future__ import annotations

import pytest

from evalshift_cli.models.matching import describe_models, model_filter_applies, same_model


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("claude-sonnet-5-20260301", "claude-sonnet-5"),
        ("anthropic/claude-sonnet-5", "claude-sonnet-5"),
        ("google/gemini-3.8-flash", "gemini-3.8-flash"),
        ("gpt-5-2026-08-07", "openai/gpt-5"),
        ("Claude-Sonnet-5", "claude-sonnet-5"),
    ],
)
def test_same_model_matches_prefix_case_and_snapshot_variants(a: str, b: str) -> None:
    assert same_model(a, b)
    assert same_model(b, a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("claude-sonnet-5", "claude-sonnet-5-5"),
        ("gpt-5", "gpt-5-mini"),
        ("", "claude-sonnet-5"),
        ("m", "claude-sonnet-5"),
    ],
)
def test_same_model_keeps_distinct_models_apart(a: str, b: str) -> None:
    assert not same_model(a, b)


def test_filter_applies_when_any_recorded_model_matches() -> None:
    assert model_filter_applies(["claude-sonnet-5-20260301", "gpt-5"], "claude-sonnet-5")


def test_filter_is_off_without_a_source_model() -> None:
    assert not model_filter_applies(["claude-sonnet-5"], None)


def test_filter_is_off_when_nothing_recorded_a_model() -> None:
    assert not model_filter_applies(["", ""], "claude-sonnet-5")


def test_guardrail_filter_is_off_when_no_recorded_model_matches() -> None:
    assert not model_filter_applies(["claude-sonnet-5"] * 3, "claude-sonnet-5-5")


def test_describe_models_counts_the_most_recorded_first() -> None:
    recorded = ["a"] * 3 + ["b"] + ["c"] * 2 + ["d"] + [""]
    assert describe_models(recorded) == "a ×3, c ×2, b ×1, +1 more"
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_model_matching.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalshift_cli.models.matching'`

- [ ] **Step 3: Implement `models/matching.py`**

```python
"""Whether two model ids name the same model, for capture-suite staleness.

A capture records whatever id the provider echoed back -- often a dated snapshot
(``claude-sonnet-5-20260301``) or a provider-prefixed id -- while ``defaults.source_model``
is usually the bare alias. Matching errs toward "same": a false match keeps a stale case a
little longer, a false mismatch would retire good ones.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from evalshift_cli.models.registry import resolve_model

_SNAPSHOT_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")


def bare_model_name(model_id: str) -> str:
    """``model_id`` lower-cased, without its provider prefix or a trailing snapshot date."""
    name = model_id.strip().lower().rsplit("/", 1)[-1]
    return _SNAPSHOT_SUFFIX.sub("", name)


def same_model(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` share a bare name or resolve to one registry id."""
    if not a.strip() or not b.strip():
        return False
    if bare_model_name(a) == bare_model_name(b):
        return True
    return resolve_model(a).id == resolve_model(b).id


def model_filter_applies(recorded: Iterable[str], source_model: str | None) -> bool:
    """Whether filtering one suite by ``source_model`` is safe (the spec §3 guardrail).

    False without a source model, when nothing recorded a model, and when not one recorded
    model matches it -- a wrong or unrecognised ``defaults.source_model``, or a migration whose
    first new-model capture has not arrived yet, must never empty or freeze a suite.
    """
    if not source_model:
        return False
    return any(same_model(model, source_model) for model in recorded if model)


def describe_models(recorded: Iterable[str], *, limit: int = 3) -> str:
    """``a ×3, c ×2, b ×1, +1 more``: the most-recorded ids first, at most ``limit`` of them."""
    counts = Counter(model for model in recorded if model)
    shown = ", ".join(f"{model} ×{n}" for model, n in counts.most_common(limit))
    rest = len(counts) - limit
    return f"{shown}, +{rest} more" if rest > 0 else shown
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_model_matching.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/models/matching.py tests/unit/test_model_matching.py
git commit -m "feat(models): match recorded model ids against the source model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Promoted cases remember when and on what their capture ran

**Files:**
- Modify: `src/evalshift_cli/captures/models.py` (`PromotedCase`)
- Modify: `src/evalshift_cli/captures/promote.py` (`capture_provenance`, `retire_promoted_case`, `__all__`)
- Modify: `src/evalshift_cli/cli/commands/capture.py` (`capture promote` + `capture sync` write the fields; `_load_promoted`, `_backfill_provenance`; summary clause)
- Modify: `tests/unit/fake_remote_store.py` (`capture_payload(model_id=...)`), `tests/unit/test_cli_capture.py` (`_events` / `_write_capture` gain `model_id`)
- Test: `tests/unit/test_captures_models.py`, `tests/unit/test_captures_promote.py`, `tests/unit/test_cli_capture.py`

**Interfaces:**
- Produces: `PromotedCase.captured_at: str = ""`, `PromotedCase.source_model: str = ""`; `promote.capture_provenance(envelope: CaptureEnvelope) -> tuple[str, str]` (`(created_at, first model_call's model_id or "")`); `promote.retire_promoted_case(suite: str, name: str, *, base: Path | None = None) -> bool`; in `capture.py`: `_load_promoted(suite: str, *, base: Path | None) -> tuple[list[PromotedCase], str | None]` (cases, error type name), `_backfill_provenance(cases: list[PromotedCase], envelopes: Mapping[str, CaptureEnvelope], *, base: Path | None) -> tuple[list[PromotedCase], int]`; test helpers `capture_payload(..., model_id: str = "m")`, `_write_capture(..., model_id: str = "m")`.

- [ ] **Step 1: Give the test helpers a model id**

`tests/unit/fake_remote_store.py::capture_payload`: add keyword `model_id: str = "m"` and use it in place of the hardcoded `"model_id": "m"`.
`tests/unit/test_cli_capture.py`: add `model_id: str = "m"` to `_events` (used in its `"model_id"`) and to `_write_capture`, which passes it on: `_events(tools=..., model_input=model_input, model_id=model_id)`.

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_captures_models.py` — append:

```python
def test_a_case_written_before_capture_provenance_still_parses() -> None:
    case = PromotedCase.model_validate(
        {"name": "c", "suite": "s", "from_capture": "cap_1", "example": {"id": "c", "tools": []}}
    )
    assert case.captured_at == ""
    assert case.source_model == ""
```

(import `PromotedCase` from `evalshift_cli.captures.models` if the file does not already.)

`tests/unit/test_captures_promote.py` — add `capture_provenance`, `retire_promoted_case`, `write_promoted_case` to the `evalshift_cli.captures.promote` import and `from evalshift_cli.suite.models import SuiteExample`, then append:

```python
class TestCaptureProvenance:
    def test_reads_created_at_and_the_first_model_calls_model(self) -> None:
        envelope = _envelope(
            events=[
                _model_call(0, model_input="hi", model_id="claude-sonnet-5-20260301"),
                _model_call(1, model_input="again", model_id="claude-haiku-5"),
                _final("done", 2),
            ],
            created_at="2026-10-01T12:00:00+00:00",
        )
        assert capture_provenance(envelope) == (
            "2026-10-01T12:00:00+00:00",
            "claude-sonnet-5-20260301",
        )

    def test_a_capture_without_a_model_call_has_no_model(self) -> None:
        assert capture_provenance(_envelope(events=[_final("done", 0)]))[1] == ""


class TestRetirePromotedCase:
    def test_deletes_the_case_file(self, tmp_path: Path) -> None:
        case = PromotedCase(
            name="cap_1",
            suite="alpha",
            from_capture="cap_1",
            example=SuiteExample(id="cap_1", tools=[]),
        )
        path = write_promoted_case(case, base=tmp_path)
        assert retire_promoted_case("alpha", "cap_1", base=tmp_path) is True
        assert not path.exists()

    def test_an_absent_case_is_not_an_error(self, tmp_path: Path) -> None:
        assert retire_promoted_case("alpha", "cap_nope", base=tmp_path) is False
```

`tests/unit/test_cli_capture.py` — append:

```python
def test_sync_records_when_and_on_what_each_capture_ran(tmp_path: Path) -> None:
    _write_capture(
        tmp_path,
        capture_id="cap_1",
        suite="alpha",
        created_at="2026-10-01T09:30:00+00:00",
        model_id="claude-sonnet-5-20260301",
    )
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    case = PromotedCase.model_validate_json(
        (tmp_path / "suites" / "alpha" / "cap_1.json").read_text(encoding="utf-8")
    )
    assert case.captured_at == "2026-10-01T09:30:00+00:00"
    assert case.source_model == "claude-sonnet-5-20260301"


def test_sync_backfills_a_case_written_before_the_fields_existed(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", suite="alpha", model_id="claude-sonnet-5")
    config = tmp_path / "evalshift.yaml"
    _write_min_config(config)
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    case_path = tmp_path / "suites" / "alpha" / "cap_1.json"
    legacy = json.loads(case_path.read_text(encoding="utf-8"))
    del legacy["captured_at"], legacy["source_model"]
    case_path.write_text(json.dumps(legacy), encoding="utf-8")

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    case = PromotedCase.model_validate_json(case_path.read_text(encoding="utf-8"))
    assert case.captured_at == "2026-06-16T12:00:00+00:00"
    assert case.source_model == "claude-sonnet-5"
    assert case.promoted_at == legacy["promoted_at"]
    assert "backfilled capture time and model on 1 case(s)" in " ".join(result.stdout.split())


def test_promote_records_when_and_on_what_the_capture_ran(tmp_path: Path) -> None:
    _write_capture(tmp_path, capture_id="cap_1", model_id="gpt-5-2026-08-07")

    assert _invoke(["promote", "cap_1", "--as", "case1"], tmp_path).exit_code == 0

    case = PromotedCase.model_validate_json(
        (tmp_path / "suites" / "support_agent" / "case1.json").read_text(encoding="utf-8")
    )
    assert case.source_model == "gpt-5-2026-08-07"
    assert case.captured_at == "2026-06-16T12:00:00+00:00"
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/unit/test_captures_models.py tests/unit/test_captures_promote.py tests/unit/test_cli_capture.py -q -k "provenance or Provenance or Retire or records_when or backfills"`
Expected: FAIL — unknown attributes / missing imports.

- [ ] **Step 4: Add the fields**

In `src/evalshift_cli/captures/models.py`, inside `PromotedCase` after `cost_source`:

```python
    # When the source capture was recorded (its ``created_at``, verbatim) and the model its
    # first ``model_call`` ran on -- the call a replay re-sends. With captures.max_age /
    # captures.max_cases set, ``capture sync`` uses both to tell a stale case from a fresh
    # one. Empty on case files written before they existed; sync backfills them while the
    # capture is still local.
    captured_at: str = ""
    source_model: str = ""
```

- [ ] **Step 5: Add the promote helpers**

In `src/evalshift_cli/captures/promote.py`, after `write_promoted_case`:

```python
def capture_provenance(envelope: CaptureEnvelope) -> tuple[str, str]:
    """``(captured_at, source_model)`` for a case promoted from ``envelope``.

    ``source_model`` is the first ``model_call``'s ``model_id`` -- the call whose input a replay
    re-sends -- or ``""`` when the capture recorded none.
    """
    first = next((e for e in envelope.trace.events if isinstance(e, ModelCallEvent)), None)
    return envelope.created_at, first.model_id if first is not None else ""


def retire_promoted_case(suite: str, name: str, *, base: Path | None = None) -> bool:
    """Delete ``<base>/suites/<suite>/<name>.json``; ``False`` when it was already gone.

    The caller rebuilds ``golden.jsonl`` afterwards (:func:`rebuild_golden_jsonl`).
    """
    path = promoted_suite_dir(suite, base=base) / f"{_safe_segment(name)}.json"
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
```

Add both names to `__all__` (alphabetical order).

- [ ] **Step 6: Write the fields from `promote` and `sync`, and backfill**

In `src/evalshift_cli/cli/commands/capture.py` import `capture_provenance` from `evalshift_cli.captures.promote`. In **both** `PromotedCase(...)` constructions (`capture promote`, near line 725; `capture sync`, near line 1298) add, computed just before:

```python
    captured_at, source_model = capture_provenance(envelope)
```

and the keyword arguments `captured_at=captured_at, source_model=source_model,` (before `example=`).

Add these helpers below `_promoted_content_owners`:

```python
def _load_promoted(suite: str, *, base: Path | None) -> tuple[list[PromotedCase], str | None]:
    """The suite's promoted cases, plus the error's type name when its files could not be read.

    A suite with no directory yet has no cases. Only the exception's type name is kept: its
    text can carry raw bytes of a case file.
    """
    suite_dir = promoted_suite_dir(suite, base=base)
    if not suite_dir.is_dir():
        return [], None
    try:
        return iter_promoted_cases(suite_dir), None
    except (ValueError, OSError) as exc:  # pydantic's ValidationError is a ValueError
        return [], type(exc).__name__


def _backfill_provenance(
    cases: list[PromotedCase],
    envelopes: Mapping[str, CaptureEnvelope],
    *,
    base: Path | None,
) -> tuple[list[PromotedCase], int]:
    """Fill ``captured_at`` / ``source_model`` on cases written before the fields existed.

    Only an empty field is filled, only from the case's own capture, and only while that
    capture is local; nothing else on the case changes (``promoted_at`` included). Returns
    the cases as they now stand and how many files were rewritten.
    """
    out: list[PromotedCase] = []
    rewritten = 0
    for case in cases:
        envelope = envelopes.get(case.from_capture)
        if envelope is None or (case.captured_at and case.source_model):
            out.append(case)
            continue
        captured_at, source_model = capture_provenance(envelope)
        filled = case.model_copy(
            update={
                "captured_at": case.captured_at or captured_at,
                "source_model": case.source_model or source_model,
            }
        )
        if filled == case:
            out.append(case)
            continue
        write_promoted_case(filled, base=base, force=True)
        out.append(filled)
        rewritten += 1
    return out, rewritten
```

In `capture_sync`: add `backfilled = 0` with the other counters. In the per-suite loop, right after the `duplicate_turn_warnings` loop and before `built_pairs = ...`, add:

```python
        envelopes = {r.envelope.capture_id: r.envelope for r in records_by_suite[suite_name]}
        kept_cases, unreadable = _load_promoted(suite_name, base=base)
        if unreadable is None:
            kept_cases, rewritten = _backfill_provenance(kept_cases, envelopes, base=base)
            backfilled += rewritten
```

In the summary block, right after the `wired_generation` clause:

```python
    if backfilled:
        summary += f", backfilled capture time and model on {backfilled} case(s)"
```

- [ ] **Step 7: Run the new tests and the capture suites**

Run: `uv run pytest tests/unit/test_captures_models.py tests/unit/test_captures_promote.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_remote.py tests/unit/test_cli_capture_repair.py tests/unit/test_cli_capture_native_shapes.py -q`
Expected: PASS. A test that compares a whole case file to a literal dict needs the two new keys (`""` or the fixture's values) — update it.

- [ ] **Step 8: Commit**

```bash
git add src/evalshift_cli/captures/models.py src/evalshift_cli/captures/promote.py src/evalshift_cli/cli/commands/capture.py tests/unit/fake_remote_store.py tests/unit/test_captures_models.py tests/unit/test_captures_promote.py tests/unit/test_cli_capture.py
git commit -m "feat(captures): promoted cases record their capture time and model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Staleness rules

**Files:**
- Create: `src/evalshift_cli/captures/staleness.py`
- Test: `tests/unit/test_captures_staleness.py` (new)

**Interfaces:**
- Consumes: `PromotedCase.captured_at/promoted_at/source_model` (Task 4), `same_model` (Task 3).
- Produces: `StaleReason = Literal["age", "model"]`; `NO_TOOLS_ROUTE = "∅"`; `parse_instant(text: str) -> datetime | None`; `case_instant(case: PromotedCase) -> datetime | None`; `is_reviewed(example: SuiteExample) -> bool`; `examples_route(examples: Iterable[SuiteExample]) -> str`; `@dataclass(frozen=True, slots=True) StalenessRules(now: datetime, max_age: timedelta | None = None, source_model: str | None = None)` with `stale_reason(instant: datetime | None, model: str) -> StaleReason | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_captures_staleness.py`:

```python
"""What makes a promoted case stale (spec §3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.staleness import (
    NO_TOOLS_ROUTE,
    StalenessRules,
    case_instant,
    examples_route,
    is_reviewed,
    parse_instant,
)
from evalshift_cli.suite.models import ExpectedToolCall, SuiteExample

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)
#: A non-empty toolset (expected_tools may not sit beside an empty one).
_REF = "sha256:" + "cd" * 32


def _example(*tools: str, reviewed: bool = False) -> SuiteExample:
    calls = [
        ExpectedToolCall(tool_name=t, provenance="reviewed" if reviewed else "captured")
        for t in tools
    ]
    return SuiteExample(id="c", toolset_ref=_REF, expected_tools=calls or None)


def _case(*, captured_at: str = "", promoted_at: str = "") -> PromotedCase:
    return PromotedCase(
        name="c",
        suite="s",
        from_capture="cap",
        captured_at=captured_at,
        promoted_at=promoted_at,
        example=_example(),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-01T12:00:00+00:00", datetime(2026, 10, 1, 12, tzinfo=UTC)),
        ("2026-10-01T12:00:00Z", datetime(2026, 10, 1, 12, tzinfo=UTC)),
        ("2026-10-01T14:00:00+02:00", datetime(2026, 10, 1, 12, tzinfo=UTC)),
        ("2026-10-01T12:00:00", datetime(2026, 10, 1, 12, tzinfo=UTC)),
    ],
)
def test_parse_instant_normalises_to_utc(text: str, expected: datetime) -> None:
    assert parse_instant(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "yesterday", "2026-13-01"])
def test_parse_instant_is_none_for_anything_unparseable(text: str) -> None:
    assert parse_instant(text) is None


def test_case_instant_prefers_the_capture_time() -> None:
    case = _case(captured_at="2026-10-01T00:00:00+00:00", promoted_at="2026-10-05T00:00:00+00:00")
    assert case_instant(case) == datetime(2026, 10, 1, tzinfo=UTC)


def test_case_instant_falls_back_to_the_promotion_time() -> None:
    case = _case(promoted_at="2026-10-05T00:00:00+00:00")
    assert case_instant(case) == datetime(2026, 10, 5, tzinfo=UTC)


def test_case_instant_is_none_when_neither_parses() -> None:
    assert case_instant(_case()) is None


def test_is_reviewed_reads_any_expectation() -> None:
    assert is_reviewed(_example("a", reviewed=True))
    assert not is_reviewed(_example("a"))
    assert not is_reviewed(_example())


def test_route_is_the_sorted_union_of_expected_tool_names() -> None:
    assert examples_route([_example("b", "a")]) == "a,b"
    assert examples_route([_example("a"), _example("c")]) == "a,c"
    assert examples_route([_example()]) == NO_TOOLS_ROUTE


def test_an_old_case_is_age_stale() -> None:
    rules = StalenessRules(now=NOW, max_age=timedelta(days=14))
    assert rules.stale_reason(NOW - timedelta(days=15), "m") == "age"
    assert rules.stale_reason(NOW - timedelta(days=13), "m") is None
    assert rules.stale_reason(None, "m") is None


def test_another_model_is_model_stale_before_age() -> None:
    rules = StalenessRules(now=NOW, max_age=timedelta(days=14), source_model="claude-sonnet-5-5")
    old = NOW - timedelta(days=30)
    assert rules.stale_reason(old, "claude-sonnet-5-20260301") == "model"
    assert rules.stale_reason(old, "claude-sonnet-5-5") == "age"
    assert rules.stale_reason(NOW, "") is None


def test_no_limits_means_nothing_is_stale() -> None:
    assert StalenessRules(now=NOW).stale_reason(NOW - timedelta(days=999), "x") is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_captures_staleness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalshift_cli.captures.staleness'`

- [ ] **Step 3: Implement `captures/staleness.py`**

```python
"""What makes a promoted case stale -- shared by ``capture sync`` and the freshness checks.

See ``docs/superpowers/specs/2026-10-16-bounded-capture-suites-design.md`` §3.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.models.matching import same_model
from evalshift_cli.suite.models import SuiteExample

StaleReason = Literal["age", "model"]
"""Why a case is stale: older than ``captures.max_age``, or recorded on another model."""

#: The route of a case that expects no tool call.
NO_TOOLS_ROUTE = "∅"


def parse_instant(text: str) -> datetime | None:
    """An ISO 8601 timestamp as an aware UTC datetime; naive means UTC; ``None`` if unparseable."""
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def case_instant(case: PromotedCase) -> datetime | None:
    """When ``case``'s capture was recorded.

    ``captured_at``, else ``promoted_at`` -- an upper bound on the capture time, so a legacy
    case looks younger than it is and is retired later, never earlier.
    """
    return parse_instant(case.captured_at) or parse_instant(case.promoted_at)


def is_reviewed(example: SuiteExample) -> bool:
    """Whether a human marked any of ``example``'s tool expectations ``provenance: reviewed``."""
    calls = list(example.expected_tools or [])
    for round_calls in example.expected_tool_rounds or []:
        calls.extend(round_calls)
    return any(call.provenance == "reviewed" for call in calls)


def examples_route(examples: Iterable[SuiteExample]) -> str:
    """The sorted, comma-joined round-1 tool names across ``examples``; ``∅`` for none."""
    names = sorted({call.tool_name for e in examples for call in e.expected_tools or []})
    return ",".join(names) if names else NO_TOOLS_ROUTE


@dataclass(frozen=True, slots=True)
class StalenessRules:
    """The limits one sync (or one freshness check) measures cases against.

    Attributes:
        now: The single instant ages are measured from.
        max_age: ``captures.max_age``; ``None`` turns age staleness off.
        source_model: The model a case must have been recorded on; ``None`` turns model
            staleness off (no ``defaults.source_model``, or the guardrail tripped).
    """

    now: datetime
    max_age: timedelta | None = None
    source_model: str | None = None

    def stale_reason(self, instant: datetime | None, model: str) -> StaleReason | None:
        """``model`` before ``age``; ``None`` when fresh or when the fact is unknown."""
        if self.source_model and model and not same_model(model, self.source_model):
            return "model"
        if self.max_age is not None and instant is not None and instant < self.now - self.max_age:
            return "age"
        return None
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_captures_staleness.py -q && uv run mypy --strict src/evalshift_cli`
Expected: PASS / clean. (If importing `captures.staleness` from `captures/__init__` order causes a cycle, do not add it to `captures/__init__.py` — import it by module path.)

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/captures/staleness.py tests/unit/test_captures_staleness.py
git commit -m "feat(captures): staleness rules for promoted cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The per-suite planner

**Files:**
- Create: `src/evalshift_cli/captures/selection.py`
- Test: `tests/unit/test_captures_selection.py` (new)

**Interfaces:**
- Consumes: `StaleReason` (Task 5).
- Produces: `RetireReason = Literal["age", "model", "over_cap"]`; frozen dataclasses `KeptCase(name: str, instant: datetime | None, stale: StaleReason | None, reviewed: bool, route: str, content_key: str)`, `CandidateGroup(key: str, capture_ids: tuple[str, ...], route: str, content_keys: frozenset[str])` with property `size: int`, `SuitePlan(add: tuple[CandidateGroup, ...] = (), retire: tuple[tuple[str, RetireReason], ...] = (), reviewed_over_cap: int = 0)` with property `added_cases: int`; `plan_suite(kept: Sequence[KeptCase], candidates: Sequence[CandidateGroup], *, max_cases: int) -> SuitePlan`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_captures_selection.py`:

```python
"""``plan_suite``: which captures enter a bounded suite and which cases leave (spec §5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from evalshift_cli.captures.selection import CandidateGroup, KeptCase, SuitePlan, plan_suite
from evalshift_cli.captures.staleness import StaleReason

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _kept(
    name: str,
    *,
    days: int = 0,
    stale: StaleReason | None = None,
    reviewed: bool = False,
    route: str = "a",
    key: str | None = None,
) -> KeptCase:
    return KeptCase(
        name=name,
        instant=T0 + timedelta(days=days),
        stale=stale,
        reviewed=reviewed,
        route=route,
        content_key=key or f"k-{name}",
    )


def _group(
    key: str, *, route: str = "a", size: int = 1, content: set[str] | None = None
) -> CandidateGroup:
    ids = tuple(f"{key}-t{i}" for i in range(size)) if size > 1 else (key,)
    return CandidateGroup(
        key=key,
        capture_ids=ids,
        route=route,
        content_keys=frozenset(content or {f"k-{i}" for i in ids}),
    )


def _added(plan: SuitePlan) -> list[str]:
    return [g.key for g in plan.add]


def test_fills_free_slots_up_to_the_cap() -> None:
    plan = plan_suite([], [_group(f"c{i}") for i in range(5)], max_cases=3)
    assert plan.added_cases == 3
    assert plan.retire == ()


def test_the_plan_is_deterministic_whatever_the_input_order() -> None:
    groups = [_group(f"c{i}", route="ab"[i % 2]) for i in range(7)]
    assert plan_suite([], groups, max_cases=4) == plan_suite([], groups[::-1], max_cases=4)


def test_the_least_represented_route_fills_first() -> None:
    kept = [_kept(f"f{i}") for i in range(3)]
    groups = [_group(f"a{i}") for i in range(5)] + [_group(f"b{i}", route="b") for i in range(2)]
    plan = plan_suite(kept, groups, max_cases=6)
    assert [g.route for g in plan.add] == ["b", "b", "a"]


def test_a_group_is_added_whole_or_not_at_all() -> None:
    plan = plan_suite([], [_group("conv", size=3), _group("solo", route="b")], max_cases=2)
    assert _added(plan) == ["solo"]
    assert plan.added_cases == 1


def test_stale_cases_stay_when_nothing_replaces_them() -> None:
    kept = [_kept(f"s{i}", stale="age") for i in range(3)]
    assert plan_suite(kept, [], max_cases=3) == SuitePlan()


def test_stale_cases_leave_only_to_make_room_oldest_first() -> None:
    kept = [_kept("s1", days=0, stale="age"), _kept("s2", days=1, stale="age"), _kept("s3", days=2, stale="age")]
    plan = plan_suite(kept, [_group("new")], max_cases=3)
    assert _added(plan) == ["new"]
    assert plan.retire == (("s1", "age"),)


def test_another_model_leaves_before_old_age() -> None:
    kept = [_kept("old", days=0, stale="age"), _kept("mod", days=5, stale="model")]
    plan = plan_suite(kept, [_group("new")], max_cases=2)
    assert plan.retire == (("mod", "model"),)


def test_a_stale_twin_is_replaced_even_under_the_cap() -> None:
    kept = [_kept("twin", stale="age", key="k-q")]
    plan = plan_suite(kept, [_group("new", content={"k-q"})], max_cases=10)
    assert _added(plan) == ["new"]
    assert plan.retire == (("twin", "age"),)


def test_reviewed_cases_are_never_retired() -> None:
    kept = [_kept("r1", stale="age", reviewed=True), _kept("r2", stale="model", reviewed=True)]
    plan = plan_suite(kept, [_group(f"c{i}") for i in range(3)], max_cases=1)
    assert plan.add == ()
    assert plan.retire == ()
    assert plan.reviewed_over_cap == 1


def test_an_over_cap_suite_is_trimmed_from_its_largest_route_oldest_first() -> None:
    kept = [
        _kept("a1", days=0),
        _kept("a2", days=1),
        _kept("a3", days=2),
        _kept("b1", days=0, route="b"),
    ]
    plan = plan_suite(kept, [], max_cases=2)
    assert plan.retire == (("a1", "over_cap"), ("a2", "over_cap"))


def test_planning_twice_changes_nothing() -> None:
    kept = [_kept(f"f{i}") for i in range(3)]
    assert plan_suite(kept, [_group("c1"), _group("c2")], max_cases=3) == SuitePlan()


@pytest.mark.parametrize("max_cases", [1, 2, 3, 5, 8])
def test_a_suite_never_ends_over_the_cap_unless_reviewed_cases_force_it(max_cases: int) -> None:
    kept = [
        _kept("r", reviewed=True),
        _kept("f1"),
        _kept("f2", route="b"),
        _kept("s1", stale="age"),
        _kept("s2", stale="model", route="b"),
    ]
    groups = [_group("c1"), _group("c2", route="c"), _group("conv", size=2, route="b")]
    plan = plan_suite(kept, groups, max_cases=max_cases)
    names = [name for name, _ in plan.retire]
    assert len(names) == len(set(names))
    assert "r" not in names
    assert len(kept) - len(plan.retire) + plan.added_cases <= max_cases
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_captures_selection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalshift_cli.captures.selection'`

- [ ] **Step 3: Implement `captures/selection.py`**

```python
"""Which captures enter a bounded suite and which promoted cases leave it.

Pure: no I/O, no clock. ``capture sync`` builds the inputs from disk and applies the plan.
See ``docs/superpowers/specs/2026-10-16-bounded-capture-suites-design.md`` §5.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from evalshift_cli.captures.staleness import StaleReason

RetireReason = Literal["age", "model", "over_cap"]
"""Why a case left: too old, recorded on another model, or trimmed to ``max_cases``."""


@dataclass(frozen=True, slots=True)
class KeptCase:
    """A promoted case already in the suite, as the planner sees it.

    Attributes:
        name: The case name (its file stem) -- what a retirement deletes.
        instant: When its capture was recorded (``staleness.case_instant``), if known.
        stale: Why it is stale, or ``None`` when fresh.
        reviewed: A human marked one of its expectations ``reviewed``: never retired.
        route: Its tool route (``staleness.examples_route``).
        content_key: ``promote.example_content_key`` of its example.
    """

    name: str
    instant: datetime | None
    stale: StaleReason | None
    reviewed: bool
    route: str
    content_key: str


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    """Eligible captures that enter a suite together: one conversation, or one lone capture.

    Attributes:
        key: Stable identity used for the deterministic order.
        capture_ids: Its turns' case names (the capture ids), in turn order.
        route: The union route of its turns.
        content_keys: Each turn's ``example_content_key``.
    """

    key: str
    capture_ids: tuple[str, ...]
    route: str
    content_keys: frozenset[str]

    @property
    def size(self) -> int:
        """How many cases the group adds."""
        return len(self.capture_ids)


@dataclass(frozen=True, slots=True)
class SuitePlan:
    """What one sync does to one suite.

    Attributes:
        add: Groups to promote, in the order they were picked.
        retire: ``(case name, reason)`` for every case to delete.
        reviewed_over_cap: Reviewed cases beyond ``max_cases``, kept anyway.
    """

    add: tuple[CandidateGroup, ...] = ()
    retire: tuple[tuple[str, RetireReason], ...] = ()
    reviewed_over_cap: int = 0

    @property
    def added_cases(self) -> int:
        """How many cases ``add`` promotes."""
        return sum(group.size for group in self.add)


def _hash_order(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _when(case: KeptCase) -> float:
    """Sort key, oldest first; an unknown instant counts as oldest of all."""
    return case.instant.timestamp() if case.instant is not None else float("-inf")


def plan_suite(
    kept: Sequence[KeptCase],
    candidates: Sequence[CandidateGroup],
    *,
    max_cases: int,
) -> SuitePlan:
    """Plan one suite: fill free slots route-balanced; retire stale cases only to make room.

    1. Reviewed cases are pinned; every other case is fresh or stale.
    2. Free slots (``max_cases`` minus pinned and fresh) are filled one group at a time from
       the route least represented so far (ties by route name), each route's groups in
       ``sha256(key)`` order. A group is added whole, or skipped when it does not fit.
    3. A stale case whose content an added turn replays is retired outright, so no duplicate
       content survives. Other stale cases leave only while the suite is over ``max_cases``:
       recorded-on-another-model first, then oldest.
    4. Still over (a legacy or over-cap suite): fresh cases are trimmed from the most
       represented route, oldest first. Reviewed cases never leave.

    Same inputs, same plan, on any machine.
    """
    pinned = [c for c in kept if c.reviewed]
    fresh = [c for c in kept if not c.reviewed and c.stale is None]
    stale = [c for c in kept if not c.reviewed and c.stale is not None]

    counts: Counter[str] = Counter(c.route for c in (*pinned, *fresh))
    queues: dict[str, list[CandidateGroup]] = {}
    for group in sorted(candidates, key=lambda g: _hash_order(g.key)):
        queues.setdefault(group.route, []).append(group)

    slots = max(0, max_cases - len(pinned) - len(fresh))
    added: list[CandidateGroup] = []
    while slots > 0:
        fitting = [r for r, queue in queues.items() if any(g.size <= slots for g in queue)]
        if not fitting:
            break
        route = min(fitting, key=lambda r: (counts[r], r))
        queue = queues[route]
        group = queue.pop(next(i for i, g in enumerate(queue) if g.size <= slots))
        added.append(group)
        counts[route] += group.size
        slots -= group.size

    retire: list[tuple[str, RetireReason]] = []
    replayed = frozenset(key for group in added for key in group.content_keys)
    remaining: list[KeptCase] = []
    for case in stale:
        if case.stale is not None and case.content_key in replayed:
            retire.append((case.name, case.stale))
        else:
            remaining.append(case)

    excess = len(pinned) + len(fresh) + sum(g.size for g in added) + len(remaining) - max_cases
    for case in sorted(remaining, key=lambda c: (c.stale != "model", _when(c), c.name)):
        if excess <= 0:
            break
        if case.stale is not None:
            retire.append((case.name, case.stale))
            excess -= 1

    pool = sorted(fresh, key=lambda c: (_when(c), c.name))
    while excess > 0 and pool:
        route = min({c.route for c in pool}, key=lambda r: (-counts[r], r))
        victim = next(c for c in pool if c.route == route)
        pool.remove(victim)
        counts[route] -= 1
        retire.append((victim.name, "over_cap"))
        excess -= 1

    return SuitePlan(
        add=tuple(added),
        retire=tuple(retire),
        reviewed_over_cap=max(0, len(pinned) - max_cases),
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_captures_selection.py -q && uv run mypy --strict src/evalshift_cli`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/captures/selection.py tests/unit/test_captures_selection.py
git commit -m "feat(captures): plan_suite picks and retires cases for a bounded suite

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Sync reads the limits; the fetch window follows `max_age`

**Files:**
- Modify: `src/evalshift_cli/captures/remote.py` (`FetchSummary.since_label`, `fetch_captures(since_label=...)`)
- Modify: `src/evalshift_cli/cli/commands/capture.py` (`_now`, `_SyncLimits`, `_sync_limits`, `_maybe_fetch(window=...)`, `_fetch_into_base(since_label=...)`, top of `capture_sync`)
- Test: `tests/unit/test_captures_remote.py`, `tests/unit/test_cli_capture_remote.py`

**Interfaces:**
- Consumes: `CapturesConfig.max_age/max_cases` (Task 1), `format_duration` (Task 1).
- Produces: `FetchSummary.since_label: str = "--since"`; `fetch_captures(..., since_label: str = "--since")`; in `capture.py`: `_now() -> datetime`, `@dataclass(frozen=True, slots=True) _SyncLimits(max_age: timedelta | None = None, max_cases: int | None = None, source_model: str | None = None)` with property `active: bool`; `_sync_limits(config_path: Path, *, console: Console, warn: bool) -> _SyncLimits`; inside `capture_sync` the locals `limits: _SyncLimits` and `now: datetime` that Task 8 uses.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_captures_remote.py` — append (import `FetchSummary` if needed):

```python
def test_skipped_old_names_the_window_it_was_measured_against() -> None:
    summary = FetchSummary(
        store_uri="s3://b/p", skipped_old=3, since_label="captures.max_age (14d)"
    )
    assert "3 older than captures.max_age (14d)" in summary.describe()


def test_skipped_old_defaults_to_the_since_flag() -> None:
    assert "1 older than --since" in FetchSummary(store_uri="s3://b/p", skipped_old=1).describe()
```

`tests/unit/test_cli_capture_remote.py` — append (import `datetime`, `UTC`):

```python
_NOW = datetime(2026, 6, 20, tzinfo=UTC)


def _with_max_age(config: Path) -> Path:
    config.write_text(config.read_text(encoding="utf-8") + "  max_age: 14d\n", encoding="utf-8")
    return config


def test_sync_fetches_only_inside_max_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: FakeRemoteStore
) -> None:
    remote.modified["captures/alpha/cap_1.json"] = datetime(2026, 5, 1, tzinfo=UTC)
    config = _with_max_age(_write_config(tmp_path / "evalshift.yaml"))
    monkeypatch.setattr(capture_module, "_now", lambda: _NOW)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    flat = " ".join(result.stdout.split())
    assert "fetched 1 capture(s)" in flat
    assert "1 older than captures.max_age (14d)" in flat
    assert not (tmp_path / "captures" / "alpha" / "cap_1.json").exists()


def test_an_explicit_since_overrides_max_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: FakeRemoteStore
) -> None:
    remote.modified["captures/alpha/cap_1.json"] = datetime(2026, 5, 1, tzinfo=UTC)
    config = _with_max_age(_write_config(tmp_path / "evalshift.yaml"))
    monkeypatch.setattr(capture_module, "_now", lambda: _NOW)

    result = _invoke(["sync", "--since", "2026-04-01", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "fetched 2 capture(s)" in result.stdout


def test_max_age_without_a_store_is_not_a_fetch_error(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "evalshift.yaml", store=None)
    config.write_text(
        config.read_text(encoding="utf-8") + "\ncaptures:\n  max_age: 14d\n", encoding="utf-8"
    )
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout


def test_offline_sync_says_it_ignores_limits_it_cannot_read(tmp_path: Path) -> None:
    config = tmp_path / "evalshift.yaml"
    config.write_text("captures:\n  max_age: 2w\n", encoding="utf-8")
    result = _invoke(["sync", "--offline", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "syncing without captures.max_age / captures.max_cases" in " ".join(
        result.stdout.split()
    )
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_captures_remote.py tests/unit/test_cli_capture_remote.py -q -k "window or since_flag or max_age or offline_sync_says"`
Expected: FAIL — `FetchSummary` has no `since_label`; `capture_module` has no `_now`.

- [ ] **Step 3: Label the skip**

In `captures/remote.py`: add `since_label: str = "--since"` as the last field of `FetchSummary`; in `describe()` change the `(self.skipped_old, "older than --since")` pair to `(self.skipped_old, f"older than {self.since_label}")`; add keyword `since_label: str = "--since"` to `fetch_captures` (document it in the docstring: "how the summary names the ``since`` cutoff") and construct `FetchSummary(store_uri=store.uri, since_label=since_label)`.

- [ ] **Step 4: Read the limits in sync and pass the window**

In `cli/commands/capture.py` add `from dataclasses import dataclass`, `timedelta` to the `datetime` import, and `from evalshift_cli.utils.duration import format_duration`. Replace `_now_iso` with:

```python
def _now() -> datetime:
    """The one instant a sync measures capture age from (tests pin it)."""
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()
```

Below `_warn_store_skipped` add:

```python
@dataclass(frozen=True, slots=True)
class _SyncLimits:
    """``captures.max_age`` / ``captures.max_cases`` and ``defaults.source_model``, for sync.

    ``active`` is the opt-in switch (spec §1): with neither limit set, sync behaves exactly as
    it did before the limits existed, whatever ``defaults.source_model`` says.
    """

    max_age: timedelta | None = None
    max_cases: int | None = None
    source_model: str | None = None

    @property
    def active(self) -> bool:
        return self.max_age is not None or self.max_cases is not None


def _sync_limits(config_path: Path, *, console: Console, warn: bool) -> _SyncLimits:
    """Read the limits from ``config_path``; none at all when it is missing or unreadable.

    ``warn`` is for ``--offline``: online, the fetch step has already loaded the same file and
    warned about it, so a second line would only repeat it.
    """
    if not config_path.exists():
        return _SyncLimits()
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        reason = exc.summary
    except (OSError, UnicodeDecodeError) as exc:
        reason = type(exc).__name__  # its text can carry raw bytes of the file
    else:
        return _SyncLimits(
            max_age=cfg.captures.max_age,
            max_cases=cfg.captures.max_cases,
            source_model=cfg.defaults.source_model,
        )
    if warn and _mentions_captures(config_path):
        console.print(
            f"[yellow]⚠[/yellow] could not read {escape(str(config_path))} ({escape(reason)}); "
            "syncing without captures.max_age / captures.max_cases.",
        )
    return _SyncLimits()
```

`_fetch_into_base`: add keyword `since_label: str = "--since"` and pass `since_label=since_label` to `fetch_captures`.

`_maybe_fetch`: add keyword `window: tuple[datetime, str] | None = None` (docstring: "the cutoff ``captures.max_age`` implies, and how to name it; an explicit ``--since`` wins") and replace its body with:

```python
    since_dt = _resolve_since(since)
    since_label = "--since"
    if since_dt is None and window is not None:
        since_dt, since_label = window
    store = None if offline else _configured_store(config_path, console=console, strict=False)
    if store is None:
        if since is not None:
            console.print(
                "[red]✗[/red] --since needs captures.store in evalshift.yaml and no --offline: "
                "there is nothing to fetch from.",
            )
            raise typer.Exit(code=1)
        return
    _fetch_into_base(
        store,
        base=base,
        suite=suite,
        since=since_dt,
        since_label=since_label,
        console=console,
        force_ids=force_ids,
    )
```

At the top of `capture_sync`, between `_check_rounds(rounds)` and `selection = ...`:

```python
    limits = _sync_limits(config_path, console=console, warn=offline)
    now = _now()
    window = (
        (now - limits.max_age, f"captures.max_age ({format_duration(limits.max_age)})")
        if limits.max_age is not None
        else None
    )
```

and pass `window=window` in the `_maybe_fetch(...)` call.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/test_captures_remote.py tests/unit/test_cli_capture_remote.py tests/unit/test_cli_capture.py -q && uv run mypy --strict src/evalshift_cli`
Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add src/evalshift_cli/captures/remote.py src/evalshift_cli/cli/commands/capture.py tests/unit/test_captures_remote.py tests/unit/test_cli_capture_remote.py
git commit -m "feat(capture): sync reads captures limits; max_age bounds the fetch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Sync filters, plans and retires

**Files:**
- Modify: `src/evalshift_cli/cli/commands/capture.py` (`capture_sync` + helpers `_suites_with_cases`, `_suite_rules`, `_candidate_groups`, `_reason_counts`, `_plan_line`)
- Test: `tests/unit/test_cli_capture_limits.py` (new)

**Interfaces:**
- Consumes: `_SyncLimits`, `limits`, `now` (Task 7); `_load_promoted`, `_backfill_provenance`, `capture_provenance`, `retire_promoted_case` (Task 4); `StalenessRules`, `StaleReason`, `case_instant`, `examples_route`, `is_reviewed`, `parse_instant` (Task 5); `plan_suite`, `KeptCase`, `CandidateGroup`, `SuitePlan` (Task 6); `model_filter_applies`, `describe_models` (Task 3).
- Produces: the sync behaviour and output lines in Global Constraints. No new public API.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_cli_capture_limits.py`:

```python
"""``capture sync`` with ``captures.max_age`` / ``captures.max_cases`` set (bounded suites)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config
from evalshift_cli.cli.main import app
from evalshift_cli.suite.loader import load_jsonl
from tests.unit.fake_remote_store import TOOLSET_HEX, TOOLSET_PAYLOAD, capture_payload

runner = CliRunner()

WEEK_1 = "2026-06-10T12:00:00+00:00"
WEEK_3 = "2026-06-29T12:00:00+00:00"


def _at(monkeypatch: pytest.MonkeyPatch, day: int) -> None:
    """Pin sync's clock to noon UTC on 2026-06-<day>."""
    monkeypatch.setattr(capture_module, "_now", lambda: datetime(2026, 6, day, 12, tzinfo=UTC))


def _put(
    base: Path,
    cid: str,
    *,
    suite: str = "alpha",
    created_at: str = WEEK_1,
    model_id: str = "claude-sonnet-5-20260301",
    text: str | None = None,
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> None:
    """Write one local capture and the shared toolset sidecar it points at."""
    sidecar = base / "toolsets" / f"{TOOLSET_HEX}.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(TOOLSET_PAYLOAD)
    path = base / "captures" / suite / f"{cid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        capture_payload(
            cid,
            suite,
            model_input=text if text is not None else f"question {cid}?",
            created_at=created_at,
            model_id=model_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
    )


def _config(
    path: Path,
    *,
    max_age: str | None = "14d",
    max_cases: int | None = 3,
    source_model: str = "claude-sonnet-5",
) -> Path:
    text = render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
    text = re.sub(r"(?m)^  source_model: .*$", f"  source_model: {source_model}", text)
    limits = (f"  max_age: {max_age}\n" if max_age else "") + (
        f"  max_cases: {max_cases}\n" if max_cases else ""
    )
    if limits:
        text += "\ncaptures:\n" + limits
    path.write_text(text, encoding="utf-8")
    return path


def _sync(tmp_path: Path, *args: str) -> Any:
    result = runner.invoke(
        app,
        [
            "capture",
            "sync",
            "--config",
            str(tmp_path / "evalshift.yaml"),
            "--base",
            str(tmp_path),
            *args,
        ],
    )
    assert result.exit_code == 0, result.stdout
    return result


def _flat(result: Any) -> str:
    return " ".join(result.stdout.split())


def _cases(tmp_path: Path, suite: str = "alpha") -> set[str]:
    return {p.stem for p in (tmp_path / "suites" / suite).glob("*.json")}


def test_a_first_sync_caps_the_suite_at_max_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert len(load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").examples) == 3
    assert "alpha: 3 cases — kept 0, retired 0, added 3 of 5 eligible" in _flat(result)


def test_syncing_again_with_nothing_new_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    suite_dir = tmp_path / "suites" / "alpha"
    before = {p.name: p.read_bytes() for p in suite_dir.iterdir()}

    result = _sync(tmp_path)

    assert {p.name: p.read_bytes() for p in suite_dir.iterdir()} == before
    assert "alpha: 3 cases — kept 3, retired 0, added 0 of 2 eligible" in _flat(result)


def test_stale_cases_are_swapped_for_fresh_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"old_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new_0", created_at=WEEK_3)
    _put(tmp_path, "new_1", created_at=WEEK_3)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"old_2", "new_0", "new_1"}
    assert (
        "alpha: 3 cases — kept 1, retired 2 (2 older than 14d), added 2 of 2 eligible"
        in _flat(result)
    )


def test_a_stale_suite_stays_full_when_nothing_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"old_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert "alpha: 3 cases — kept 3, retired 0, added 0 of 0 eligible" in _flat(result)


def test_cases_recorded_on_the_old_model_leave_after_a_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"old_{i}")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new_0", created_at="2026-06-13T12:00:00+00:00", model_id="claude-sonnet-5-5")
    _config(tmp_path / "evalshift.yaml", source_model="claude-sonnet-5-5")
    _at(monkeypatch, 14)

    result = _sync(tmp_path)

    assert "new_0" in _cases(tmp_path)
    assert len(_cases(tmp_path)) == 3
    assert "retired 1 (1 on another model), added 1 of 1 eligible" in _flat(result)


def test_a_source_model_no_capture_ran_on_filters_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", source_model="gpt-9")
    _at(monkeypatch, 12)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    flat = _flat(result)
    assert "alpha: none of 3 case(s) and capture(s) were recorded on gpt-9" in flat
    assert "(seen: claude-sonnet-5-20260301 ×3)" in flat
    assert "not filtering by model — check defaults.source_model" in flat


def test_captures_outside_the_window_are_not_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(tmp_path, "old")
    _put(tmp_path, "new", created_at=WEEK_3)
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"new"}
    assert "added 1 of 1 eligible; not eligible: 1 older than 14d" in _flat(result)


def test_max_age_alone_filters_without_capping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"new_{i}", created_at=WEEK_3)
    _put(tmp_path, "old")
    _config(tmp_path / "evalshift.yaml", max_cases=None)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 5
    assert "alpha: not eligible: 1 older than 14d" in _flat(result)


def test_an_over_cap_suite_is_trimmed_without_new_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    for path in (tmp_path / "captures" / "alpha").glob("*.json"):
        path.unlink()
    _config(tmp_path / "evalshift.yaml", max_cases=3)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 3
    assert len(load_jsonl(tmp_path / "suites" / "alpha" / "golden.jsonl").examples) == 3
    assert (
        "alpha: 3 cases — kept 3, retired 2 (2 over max_cases), added 0 of 0 eligible"
        in _flat(result)
    )


def test_suite_flag_leaves_other_suites_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for suite in ("alpha", "beta"):
        for i in range(5):
            _put(tmp_path, f"{suite}_{i}", suite=suite)
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None)
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _config(tmp_path / "evalshift.yaml", max_cases=3)

    _sync(tmp_path, "--suite", "alpha")

    assert len(_cases(tmp_path, "alpha")) == 3
    assert len(_cases(tmp_path, "beta")) == 5


def test_a_conversation_enters_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for turn in range(3):
        _put(tmp_path, f"conv_t{turn}", conversation_id="c1", turn_index=turn)
    _put(tmp_path, "solo")
    _config(tmp_path / "evalshift.yaml", max_cases=2)
    _at(monkeypatch, 12)

    _sync(tmp_path)

    assert _cases(tmp_path) == {"solo"}


def test_the_same_question_captured_fresh_replaces_its_stale_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(tmp_path, "old", text="where is order 1?")
    _config(tmp_path / "evalshift.yaml")
    _at(monkeypatch, 12)
    _sync(tmp_path)
    _put(tmp_path, "new", text="where is order 1?", created_at=WEEK_3)
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert _cases(tmp_path) == {"new"}
    assert "retired 1 (1 older than 14d), added 1 of 1 eligible" in _flat(result)


def test_without_limits_sync_promotes_everything_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        _put(tmp_path, f"cap_{i}")
    _config(tmp_path / "evalshift.yaml", max_age=None, max_cases=None, source_model="gpt-9")
    _at(monkeypatch, 30)

    result = _sync(tmp_path)

    assert len(_cases(tmp_path)) == 5
    assert "cases — kept" not in _flat(result)
    assert "not filtering by model" not in _flat(result)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_cli_capture_limits.py -q`
Expected: FAIL — suites are not capped, no per-suite lines (`test_without_limits_sync_promotes_everything_as_before` already passes).

- [ ] **Step 3: Add the helpers**

In `cli/commands/capture.py` add imports:

```python
from collections import Counter
from collections.abc import Sequence  # extend the existing collections.abc import
from evalshift_cli.captures.promote import retire_promoted_case  # extend the existing import
from evalshift_cli.captures.selection import CandidateGroup, KeptCase, SuitePlan, plan_suite
from evalshift_cli.captures.staleness import (
    StaleReason,
    StalenessRules,
    case_instant,
    examples_route,
    is_reviewed,
    parse_instant,
)
from evalshift_cli.models.matching import describe_models, model_filter_applies
```

and below `_backfill_provenance`:

```python
def _suites_with_cases(*, base: Path | None, only: str | None) -> set[str]:
    """Suites under ``<base>/suites/`` holding at least one case file (``only`` narrows)."""
    root = suites_root(base)
    if not root.is_dir():
        return set()
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and any(path.glob("*.json")) and (only is None or path.name == only)
    }


def _suite_rules(
    suite: str,
    limits: _SyncLimits,
    *,
    now: datetime,
    recorded_models: list[str],
    console: Console,
) -> StalenessRules:
    """One suite's staleness rules, with the model guardrail applied (spec §3)."""
    source = limits.source_model
    if source and not model_filter_applies(recorded_models, source):
        known = [m for m in recorded_models if m]
        if known:
            console.print(
                f"[yellow]⚠[/yellow] {escape(suite)}: none of {len(known)} case(s) and "
                f"capture(s) were recorded on {escape(source)} "
                f"(seen: {escape(describe_models(known))}); not filtering by model — "
                "check defaults.source_model",
                soft_wrap=True,
            )
        source = None
    return StalenessRules(now=now, max_age=limits.max_age, source_model=source)


def _candidate_groups(pending: Sequence[PromotedCase]) -> list[CandidateGroup]:
    """Group pending cases by conversation (a lone case is its own group), turn order kept."""
    turns: dict[str, list[PromotedCase]] = {}
    for case in pending:
        key = (
            f"conversation:{case.conversation_id}"
            if case.conversation_id
            else f"capture:{case.name}"
        )
        turns.setdefault(key, []).append(case)
    return [
        CandidateGroup(
            key=key,
            capture_ids=tuple(c.name for c in cases),
            route=examples_route(c.example for c in cases),
            content_keys=frozenset(example_content_key(c.example) for c in cases),
        )
        for key, cases in turns.items()
    ]


def _reason_counts(counts: Counter[str], *, max_age: timedelta | None) -> str:
    """``2 older than 14d, 1 on another model, 3 over max_cases``; zero counts left out."""
    labels = {
        "age": f"older than {format_duration(max_age)}" if max_age is not None else "too old",
        "model": "on another model",
        "over_cap": "over max_cases",
    }
    return ", ".join(f"{counts[k]} {labels[k]}" for k in ("age", "model", "over_cap") if counts[k])


def _plan_line(
    suite: str,
    plan: SuitePlan | None,
    *,
    kept: int,
    eligible: int,
    ineligible: Counter[str],
    max_age: timedelta | None,
) -> str | None:
    """The per-suite summary line (spec §6); ``plan`` is ``None`` with ``max_age`` alone."""
    not_eligible = _reason_counts(ineligible, max_age=max_age)
    if plan is None:
        return f"  {suite}: not eligible: {not_eligible}" if not_eligible else None
    retired = Counter(reason for _, reason in plan.retire)
    detail = _reason_counts(retired, max_age=max_age)
    stayed = kept - len(plan.retire)
    return (
        f"  {suite}: {stayed + plan.added_cases} cases — kept {stayed}, "
        f"retired {len(plan.retire)}"
        + (f" ({detail})" if detail else "")
        + f", added {plan.added_cases} of {eligible} eligible"
        + (f"; not eligible: {not_eligible}" if not_eligible else "")
    )
```

- [ ] **Step 4: Wire them into `capture_sync`**

Edit `capture_sync` in this order (anchors are existing code):

(a) The early return becomes:

```python
    if not records and not (
        limits.max_cases is not None and _suites_with_cases(base=base, only=suite)
    ):
        where = f" for suite {suite!r}" if suite else ""
        console.print(f"[dim]no captures found{where} under .evalshift/captures/.[/dim]")
        return
```

(b) Next to the other counters add `plan_lines: list[str] = []`.

(c) Replace `for suite_name in sorted(records_by_suite):` with:

```python
    planned = set(records_by_suite)
    if limits.max_cases is not None:
        # An over-cap suite is trimmed even when no new capture arrived for it.
        planned |= _suites_with_cases(base=base, only=suite)
    suites_seen |= planned
    for suite_name in sorted(planned):
        suite_records = records_by_suite.get(suite_name, [])
```

and inside the loop replace every `records_by_suite[suite_name]` with `suite_records`.

(d) Right after the Task 4 backfill lines (`if unreadable is None: ...`), add:

```python
        if unreadable is not None and limits.active:
            console.print(
                f"[yellow]⚠[/yellow] {escape(suite_name)}: could not read its promoted cases "
                f"({unreadable}); left it unchanged.",
            )
            continue
        kept_from = {c.from_capture for c in kept_cases}
        stale_by_name: dict[str, StaleReason | None] = {}
        rules: StalenessRules | None = None
        if limits.active:
            recorded = [c.source_model for c in kept_cases] + [
                capture_provenance(r.envelope)[1]
                for r in suite_records
                if r.envelope.capture_id not in kept_from
            ]
            rules = _suite_rules(
                suite_name, limits, now=now, recorded_models=recorded, console=console
            )
            stale_by_name = {
                c.name: rules.stale_reason(case_instant(c), c.source_model) for c in kept_cases
            }
        ineligible: Counter[str] = Counter()
        pending: list[PromotedCase] = []
```

(e) Replace the disk seed `seen_examples.update(_promoted_content_owners(suite_name, base=base, console=console))` with:

```python
            owners = _promoted_content_owners(suite_name, base=base, console=console)
            # A stale case does not claim its content: the same question captured fresh
            # replaces it (spec §5, rule 3).
            seen_examples.update(
                (key, owner) for key, owner in owners.items() if stale_by_name.get(owner) is None
            )
```

(f) In the per-record loop, after the `if built.blocked is not None: ... continue` block and before `content_key = (suite_name, example_content_key(built.example))`, add:

```python
            if rules is not None and envelope.capture_id not in kept_from:
                captured_at, recorded_model = capture_provenance(envelope)
                reason = rules.stale_reason(parse_instant(captured_at), recorded_model)
                if reason is not None:
                    ineligible[reason] += 1
                    continue
            if limits.max_cases is not None and envelope.capture_id in kept_from and not force:
                skipped_existing += 1
                continue
```

(g) Right after the `case = PromotedCase(...)` construction and before its `try: write_promoted_case(...)`, add:

```python
            if limits.max_cases is not None and envelope.capture_id not in kept_from:
                pending.append(case)
                continue
```

(h) After the per-record loop's `if repaired_here: ...` block, still inside the suite loop, add:

```python
        line: str | None = None
        if limits.max_cases is not None:
            plan = plan_suite(
                [
                    KeptCase(
                        name=c.name,
                        instant=case_instant(c),
                        stale=stale_by_name.get(c.name),
                        reviewed=is_reviewed(c.example),
                        route=examples_route([c.example]),
                        content_key=example_content_key(c.example),
                    )
                    for c in kept_cases
                ],
                _candidate_groups(pending),
                max_cases=limits.max_cases,
            )
            chosen = {cid for group in plan.add for cid in group.capture_ids}
            for case in pending:
                if case.name not in chosen:
                    continue
                try:
                    write_promoted_case(case, base=base, force=force)
                except FileExistsError:
                    skipped_existing += 1
                    continue
                promoted += 1
                if case.example.generation_config:
                    wired_generation += 1
            for name, _reason in plan.retire:
                retire_promoted_case(suite_name, name, base=base)
            if plan.reviewed_over_cap:
                console.print(
                    f"[yellow]⚠[/yellow] {escape(suite_name)}: {plan.reviewed_over_cap} "
                    f"reviewed case(s) over captures.max_cases {limits.max_cases}; reviewed "
                    "cases are never retired, so it stays over.",
                )
            line = _plan_line(
                suite_name,
                plan,
                kept=len(kept_cases),
                eligible=len(pending),
                ineligible=ineligible,
                max_age=limits.max_age,
            )
        elif limits.active:
            line = _plan_line(
                suite_name,
                None,
                kept=len(kept_cases),
                eligible=0,
                ineligible=ineligible,
                max_age=limits.max_age,
            )
        if line is not None:
            plan_lines.append(line)
```

(i) Right after `console.print(summary + ".")`:

```python
    for line in plan_lines:
        console.print(escape(line), soft_wrap=True)
```

- [ ] **Step 5: Run the new tests and every capture test**

Run: `uv run pytest tests/unit/test_cli_capture_limits.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_remote.py tests/unit/test_cli_capture_repair.py tests/unit/test_cli_capture_native_shapes.py tests/integration/test_capture_to_verdict.py -q`
Expected: PASS. If a `test_cli_capture_limits.py` count is off, debug with superpowers:systematic-debugging before touching the expectation — the expected strings are the spec's contract.

- [ ] **Step 6: Static checks**

Run: `uv run mypy --strict src/evalshift_cli && uv run ruff check . && uv run ruff format --check .`
Expected: clean. (`capture_sync` may trip ruff's complexity rules; if so, follow how the file already silences them for this function rather than adding a new suppression style.)

- [ ] **Step 7: Commit**

```bash
git add src/evalshift_cli/cli/commands/capture.py tests/unit/test_cli_capture_limits.py
git commit -m "feat(capture): sync keeps each suite to captures.max_cases fresh cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Freshness in `doctor` and `compare`

**Files:**
- Create: `src/evalshift_cli/captures/freshness.py`
- Modify: `src/evalshift_cli/cli/commands/doctor.py` (constants, `_suite_freshness_checks`, `run_checks`)
- Modify: `src/evalshift_cli/cli/commands/compare.py` (`_freshness_lines`, verdict block)
- Test: `tests/unit/test_captures_freshness.py` (new), `tests/unit/test_doctor.py`, `tests/unit/test_compare_command.py`

**Interfaces:**
- Consumes: `case_instant` (Task 5), `model_filter_applies`, `same_model`, `describe_models` (Task 3), `format_duration` (Task 1).
- Produces: `FindingKind = Literal["unreadable", "stale_age", "stale_model", "no_source_model_case", "over_cap"]`; frozen dataclasses `FreshnessFinding(kind: FindingKind, message: str)`, `SuiteFreshness(suite: str, cases: int, newest: datetime | None, oldest: datetime | None, findings: tuple[FreshnessFinding, ...])` with `describe(now: datetime) -> str`; `suite_freshness(suite: str, suite_dir: Path, *, max_age: timedelta | None, max_cases: int | None, source_model: str | None, now: datetime) -> SuiteFreshness | None`; doctor `SUITE_FRESHNESS_CHECK = "suite freshness"`, `CAPTURES_LIMITS_CHECK = "captures limits"`; compare `_freshness_lines(cfg: EvalShiftConfig, suite_name: str | None, suite_path: Path, source_model: str) -> list[str]`.

- [ ] **Step 1: Write the failing module tests**

Create `tests/unit/test_captures_freshness.py`:

```python
"""``suite_freshness``: what doctor and compare say about a committed capture suite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from evalshift_cli.captures.freshness import SuiteFreshness, suite_freshness
from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.promote import write_promoted_case
from evalshift_cli.suite.models import SuiteExample

NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


def _write(base: Path, i: int, *, days: int, model: str = "claude-sonnet-5") -> None:
    write_promoted_case(
        PromotedCase(
            name=f"cap_{i}",
            suite="alpha",
            from_capture=f"cap_{i}",
            captured_at=(NOW - timedelta(days=days)).isoformat(),
            source_model=model,
            example=SuiteExample(id=f"cap_{i}", inputs={"input": f"q{i}"}, tools=[]),
        ),
        base=base,
    )


def _check(base: Path, **limits: Any) -> SuiteFreshness:
    options: dict[str, Any] = {
        "max_age": timedelta(days=14),
        "max_cases": 5,
        "source_model": "claude-sonnet-5",
        **limits,
    }
    fresh = suite_freshness("alpha", base / "suites" / "alpha", now=NOW, **options)
    assert fresh is not None
    return fresh


def test_no_case_files_means_nothing_to_report(tmp_path: Path) -> None:
    assert (
        suite_freshness(
            "alpha",
            tmp_path / "suites" / "alpha",
            max_age=timedelta(days=14),
            max_cases=5,
            source_model="claude-sonnet-5",
            now=NOW,
        )
        is None
    )


def test_a_fresh_suite_has_no_findings(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=2)
    _write(tmp_path, 1, days=12)
    fresh = _check(tmp_path)
    assert fresh.findings == ()
    assert fresh.describe(NOW) == "alpha: 2 cases, captured 2d–12d ago"


def test_a_suite_past_max_age_says_how_to_refresh(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=23)
    [finding] = _check(tmp_path).findings
    assert finding.kind == "stale_age"
    assert finding.message == (
        "alpha: newest case captured 23d ago (captures.max_age 14d); "
        "refresh with evalshift capture sync"
    )


def test_cases_on_another_model_are_counted(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=1)
    _write(tmp_path, 1, days=1, model="claude-sonnet-4-5")
    [finding] = _check(tmp_path).findings
    assert finding.kind == "stale_model"
    assert finding.message.startswith(
        "alpha: 1 of 2 cases recorded on another model than claude-sonnet-5"
    )


def test_no_case_on_the_source_model_is_its_own_finding(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=1, model="gpt-5")
    [finding] = _check(tmp_path).findings
    assert finding.kind == "no_source_model_case"
    assert "no case recorded on claude-sonnet-5 (seen: gpt-5 ×1)" in finding.message


def test_a_suite_over_its_cap_is_reported(tmp_path: Path) -> None:
    for i in range(3):
        _write(tmp_path, i, days=1)
    [finding] = _check(tmp_path, max_cases=2).findings
    assert finding.message == (
        "alpha: 3 cases, over captures.max_cases 2; evalshift capture sync trims it"
    )


def test_an_unreadable_case_file_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    _write(tmp_path, 0, days=1)
    (tmp_path / "suites" / "alpha" / "broken.json").write_text("{", encoding="utf-8")
    fresh = _check(tmp_path)
    assert fresh.cases == 1
    assert [f.kind for f in fresh.findings] == ["unreadable"]


def test_a_legacy_case_ages_from_its_promotion(tmp_path: Path) -> None:
    write_promoted_case(
        PromotedCase(
            name="old",
            suite="alpha",
            from_capture="old",
            promoted_at=(NOW - timedelta(days=30)).isoformat(),
            example=SuiteExample(id="old", inputs={"input": "q"}, tools=[]),
        ),
        base=tmp_path,
    )
    assert [f.kind for f in _check(tmp_path).findings] == ["stale_age"]
```


- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_captures_freshness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'evalshift_cli.captures.freshness'`

- [ ] **Step 3: Implement `captures/freshness.py`**

```python
"""How fresh a committed capture suite is, for ``doctor`` and ``compare`` (spec §7).

Reads the promoted case files committed beside ``golden.jsonl`` -- the only place a case's
capture time and model live -- so a team that only runs CI still learns its suite went stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.staleness import case_instant
from evalshift_cli.models.matching import describe_models, model_filter_applies, same_model
from evalshift_cli.utils.duration import format_duration

FindingKind = Literal["unreadable", "stale_age", "stale_model", "no_source_model_case", "over_cap"]


@dataclass(frozen=True, slots=True)
class FreshnessFinding:
    """One thing that is stale about a suite, worded for a doctor row or a compare line."""

    kind: FindingKind
    message: str


@dataclass(frozen=True, slots=True)
class SuiteFreshness:
    """A suite's case count, capture-time span and findings."""

    suite: str
    cases: int
    newest: datetime | None
    oldest: datetime | None
    findings: tuple[FreshnessFinding, ...]

    def describe(self, now: datetime) -> str:
        """``alpha: 200 cases, captured 2d–12d ago`` (the span left out when unknown)."""
        text = f"{self.suite}: {self.cases} cases"
        if self.newest is not None and self.oldest is not None:
            text += f", captured {_age(now, self.newest)}–{_age(now, self.oldest)} ago"
        return text


def _age(now: datetime, then: datetime) -> str:
    return f"{max(0, (now - then).days)}d"


def suite_freshness(
    suite: str,
    suite_dir: Path,
    *,
    max_age: timedelta | None,
    max_cases: int | None,
    source_model: str | None,
    now: datetime,
) -> SuiteFreshness | None:
    """Read ``suite_dir``'s case files and report what is stale; ``None`` when there are none.

    Never raises: an unreadable case file becomes one ``unreadable`` finding and is left out
    of every count.
    """
    paths = sorted(suite_dir.glob("*.json")) if suite_dir.is_dir() else []
    cases: list[PromotedCase] = []
    unreadable = 0
    for path in paths:
        try:
            cases.append(PromotedCase.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):  # ValidationError and UnicodeDecodeError are ValueErrors
            unreadable += 1
    if not cases and not unreadable:
        return None
    instants = [i for i in (case_instant(c) for c in cases) if i is not None]
    newest = max(instants, default=None)
    oldest = min(instants, default=None)
    findings: list[FreshnessFinding] = []
    if unreadable:
        findings.append(
            FreshnessFinding(
                "unreadable",
                f"{suite}: {unreadable} case file(s) could not be read; "
                "the counts leave them out",
            )
        )
    if max_age is not None and newest is not None and newest < now - max_age:
        findings.append(
            FreshnessFinding(
                "stale_age",
                f"{suite}: newest case captured {_age(now, newest)} ago "
                f"(captures.max_age {format_duration(max_age)}); "
                "refresh with evalshift capture sync",
            )
        )
    models = [c.source_model for c in cases]
    if source_model and any(models):
        if model_filter_applies(models, source_model):
            other = sum(1 for m in models if m and not same_model(m, source_model))
            if other:
                findings.append(
                    FreshnessFinding(
                        "stale_model",
                        f"{suite}: {other} of {len(cases)} cases recorded on another model "
                        f"than {source_model}; evalshift capture sync replaces them as new "
                        "captures arrive",
                    )
                )
        else:
            findings.append(
                FreshnessFinding(
                    "no_source_model_case",
                    f"{suite}: no case recorded on {source_model} "
                    f"(seen: {describe_models(models)}); check defaults.source_model, "
                    "or record new captures",
                )
            )
    if max_cases is not None and len(cases) > max_cases:
        findings.append(
            FreshnessFinding(
                "over_cap",
                f"{suite}: {len(cases)} cases, over captures.max_cases {max_cases}; "
                "evalshift capture sync trims it",
            )
        )
    return SuiteFreshness(
        suite=suite, cases=len(cases), newest=newest, oldest=oldest, findings=tuple(findings)
    )
```

Run: `uv run pytest tests/unit/test_captures_freshness.py -q` — Expected: PASS.

- [ ] **Step 4: Write the failing doctor tests**

Append to `tests/unit/test_doctor.py` (imports: `from datetime import UTC, datetime, timedelta`; `PromotedCase`; `rebuild_golden_jsonl`, `write_promoted_case` from `evalshift_cli.captures.promote`; `SuiteExample`; `CAPTURES_LIMITS_CHECK`, `SUITE_FRESHNESS_CHECK` from the doctor module — skip any already imported):

```python
class TestSuiteFreshnessCheck:
    def _project(
        self, tmp_path: Path, *, captures: str, ages_days: list[int], model: str = "claude-sonnet-5"
    ) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "prompts:\n  - id: p\n    detection: manual\n    content: '{input}'\n"
            "    variables: [input]\n"
            "defaults:\n  source_model: claude-sonnet-5\n"
            f"{captures}"
            "suites:\n  alpha:\n    source: captured\n"
            "    path: .evalshift/suites/alpha/golden.jsonl\n",
            encoding="utf-8",
        )
        now = datetime.now(UTC)
        for i, days in enumerate(ages_days):
            write_promoted_case(
                PromotedCase(
                    name=f"cap_{i}",
                    suite="alpha",
                    from_capture=f"cap_{i}",
                    captured_at=(now - timedelta(days=days)).isoformat(),
                    source_model=model,
                    example=SuiteExample(id=f"cap_{i}", inputs={"input": f"q{i}"}, tools=[]),
                ),
                base=tmp_path / ".evalshift",
            )
        rebuild_golden_jsonl(tmp_path / ".evalshift" / "suites" / "alpha")

    def _rows(self, tmp_path: Path, name: str = SUITE_FRESHNESS_CHECK) -> list[CheckResult]:
        return [r for r in run_checks(cwd=tmp_path, env=_empty_env()) if r.name == name]

    def test_a_fresh_suite_is_ok(self, tmp_path: Path) -> None:
        self._project(tmp_path, captures="captures:\n  max_age: 14d\n  max_cases: 5\n", ages_days=[2, 12])
        [row] = self._rows(tmp_path)
        assert row.status == "ok"
        assert row.detail == "alpha: 2 cases, captured 2d–12d ago"

    def test_a_stale_suite_warns(self, tmp_path: Path) -> None:
        self._project(tmp_path, captures="captures:\n  max_age: 14d\n", ages_days=[23, 30])
        [row] = self._rows(tmp_path)
        assert row.status == "warn"
        assert "newest case captured 23d ago (captures.max_age 14d)" in row.detail

    def test_no_rows_without_limits(self, tmp_path: Path) -> None:
        self._project(tmp_path, captures="", ages_days=[30])
        assert self._rows(tmp_path) == []
        assert self._rows(tmp_path, CAPTURES_LIMITS_CHECK) == []

    def test_a_bucket_without_limits_gets_a_nudge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.unit.fake_remote_store import populated_store

        self._project(tmp_path, captures="captures:\n  store: s3://acme-evals/p\n", ages_days=[1])
        monkeypatch.setattr(doctor_module, "open_store", lambda parsed: populated_store("cap_1"))
        [row] = self._rows(tmp_path, CAPTURES_LIMITS_CHECK)
        assert row.status == "warn"
        assert row.detail == (
            "suites grow with every capture sync; set captures.max_cases and captures.max_age"
        )
```

(If `CheckResult` is not yet imported in the test module, import it from the doctor module.)

- [ ] **Step 5: Run them to see them fail**

Run: `uv run pytest tests/unit/test_doctor.py -q -k SuiteFreshness`
Expected: FAIL — `ImportError` for the two constants.

- [ ] **Step 6: Implement the doctor rows**

In `cli/commands/doctor.py` add `from datetime import UTC, datetime` and `from evalshift_cli.captures.freshness import suite_freshness`; next to the other row-name constants:

```python
SUITE_FRESHNESS_CHECK: Final = "suite freshness"
CAPTURES_LIMITS_CHECK: Final = "captures limits"
```

below `_captures_store_check`:

```python
def _suite_freshness_checks(cwd: Path) -> list[CheckResult]:
    """Bounded-suite rows (spec §7). Never ``fail``: freshness is advice.

    ``captures limits`` warns when ``captures.store`` is set without ``max_age`` /
    ``max_cases`` -- every sync then adds to the suite. With a limit set, each
    ``source: captured`` suite gets one ``ok`` row, or one ``warn`` row per finding.
    """
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return []
    try:
        cfg = load_config(cfg_path)
    except ConfigError:
        return []
    captures = cfg.captures
    if captures.max_age is None and captures.max_cases is None:
        if captures.store is None:
            return []
        return [
            CheckResult(
                name=CAPTURES_LIMITS_CHECK,
                status="warn",
                detail="suites grow with every capture sync; "
                "set captures.max_cases and captures.max_age",
            )
        ]
    now = datetime.now(UTC)
    rows: list[CheckResult] = []
    for name, path in _named_suite_paths(cwd, cfg):
        entry = cfg.suites.get(name)
        if entry is None or entry.source != "captured":
            continue
        fresh = suite_freshness(
            name,
            path.parent,
            max_age=captures.max_age,
            max_cases=captures.max_cases,
            source_model=cfg.defaults.source_model,
            now=now,
        )
        if fresh is None:
            continue
        if not fresh.findings:
            rows.append(
                CheckResult(name=SUITE_FRESHNESS_CHECK, status="ok", detail=fresh.describe(now))
            )
        rows.extend(
            CheckResult(name=SUITE_FRESHNESS_CHECK, status="warn", detail=f.message)
            for f in fresh.findings
        )
    return rows
```

and in `run_checks`, right after `results.extend(_captures_store_check(cwd))`:

```python
    results.extend(_suite_freshness_checks(cwd))
```

Run: `uv run pytest tests/unit/test_doctor.py -q` — Expected: PASS (fix any test that pins the full row list).

- [ ] **Step 7: Write the failing compare tests**

Append to `tests/unit/test_compare_command.py` (imports: `PromotedCase`, `rebuild_golden_jsonl`, `write_promoted_case`, `SuiteExample`, `EvalShiftConfig`, and `_freshness_lines` from `evalshift_cli.cli.commands.compare` — skip any already imported):

```python
def _captured_cases(tmp_path: Path, suite: str, n: int) -> Path:
    for i in range(n):
        write_promoted_case(
            PromotedCase(
                name=f"ex{i}",
                suite=suite,
                from_capture=f"cap_{i}",
                example=SuiteExample(id=f"ex{i}", inputs={"name": f"User{i}"}, tools=[]),
            ),
            base=tmp_path / ".evalshift",
        )
    return rebuild_golden_jsonl(tmp_path / ".evalshift" / "suites" / suite)


class TestFreshnessLines:
    def _cfg(self, captures: dict[str, object]) -> EvalShiftConfig:
        return EvalShiftConfig(
            prompts=[{"id": "p", "detection": "manual", "content": "{name}", "variables": ["name"]}],
            captures=captures,
            suites={"greet": {"source": "captured", "path": ".evalshift/suites/greet/golden.jsonl"}},
        )

    def test_an_over_cap_suite_gets_a_line(self, tmp_path: Path) -> None:
        golden = _captured_cases(tmp_path, "greet", 3)
        [line] = _freshness_lines(self._cfg({"max_cases": 2}), "greet", golden, "gemini-2.5-flash")
        assert "greet: 3 cases, over captures.max_cases 2" in line

    def test_no_lines_without_limits(self, tmp_path: Path) -> None:
        golden = _captured_cases(tmp_path, "greet", 3)
        assert _freshness_lines(self._cfg({}), "greet", golden, "gemini-2.5-flash") == []

    def test_no_lines_for_a_suite_given_by_path(self, tmp_path: Path) -> None:
        golden = _captured_cases(tmp_path, "greet", 3)
        assert _freshness_lines(self._cfg({"max_cases": 2}), None, golden, "gemini-2.5-flash") == []
```

and inside the end-to-end test class that holds `test_all_prints_the_tool_free_warning_it_scored_quietly`:

```python
    def test_compare_warns_when_the_suite_is_over_its_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _scaffold(tmp_path)
        config_path = tmp_path / "evalshift.yaml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + "        captures:\n          max_cases: 2\n"
            + "        suites:\n          greet:\n            source: captured\n"
            + "            path: .evalshift/suites/greet/golden.jsonl\n",
            encoding="utf-8",
        )
        _captured_cases(tmp_path, "greet", 3)
        _patch_client(monkeypatch)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["compare", "--suite-name", "greet", "--yes"])

        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert "greet: 3 cases, over captures.max_cases 2; evalshift capture sync trims it" in flat
        assert flat.count("over captures.max_cases") == 1
```

- [ ] **Step 8: Run them to see them fail**

Run: `uv run pytest tests/unit/test_compare_command.py -q -k "Freshness or over_its_cap"`
Expected: FAIL — `ImportError: cannot import name '_freshness_lines'`

- [ ] **Step 9: Implement the compare lines**

In `cli/commands/compare.py` add `from datetime import UTC, datetime`, `from rich.markup import escape`, `EvalShiftConfig` to the `evalshift_cli.config.models` import, and `from evalshift_cli.captures.freshness import suite_freshness`. Add above `compare_command`:

```python
def _freshness_lines(
    cfg: EvalShiftConfig, suite_name: str | None, suite_path: Path, source_model: str
) -> list[str]:
    """``⚠`` lines for a stale captured suite (spec §7).

    compare drops doctor's ``warn`` rows, so it asks for the findings itself. Only for a suite
    selected by name whose entry is ``source: captured``, and only with a capture limit set.
    """
    captures = cfg.captures
    if suite_name is None or (captures.max_age is None and captures.max_cases is None):
        return []
    entry = cfg.suites.get(suite_name)
    if entry is None or entry.source != "captured":
        return []
    fresh = suite_freshness(
        suite_name,
        suite_path.parent,
        max_age=captures.max_age,
        max_cases=captures.max_cases,
        source_model=source_model,
        now=datetime.now(UTC),
    )
    if fresh is None:
        return []
    return [f"[yellow]⚠[/yellow] {escape(f.message)}" for f in fresh.findings]
```

In `compare_command`, right after the `suite = load_jsonl(suite_path)` try/except:

```python
    freshness = _freshness_lines(cfg, suite_name, suite_path, source_model)
```

and in the verdict block, right after the `tool_free_warnings` block:

```python
    # And again: a stale suite gates on ground truth production no longer produces.
    if freshness:
        for line in freshness:
            console.print(line, soft_wrap=True)
        console.print()
```

- [ ] **Step 10: Run doctor, compare and freshness tests, then static checks**

Run: `uv run pytest tests/unit/test_captures_freshness.py tests/unit/test_doctor.py tests/unit/test_compare_command.py -q && uv run mypy --strict src/evalshift_cli && uv run ruff check . && uv run ruff format --check .`
Expected: PASS / clean.

- [ ] **Step 11: Commit**

```bash
git add src/evalshift_cli/captures/freshness.py src/evalshift_cli/cli/commands/doctor.py src/evalshift_cli/cli/commands/compare.py tests/unit/test_captures_freshness.py tests/unit/test_doctor.py tests/unit/test_compare_command.py
git commit -m "feat(doctor,compare): warn when a captured suite is stale or over its cap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Lifecycle integration test

**Files:**
- Test: `tests/integration/test_capture_lifecycle.py` (new)

**Interfaces:**
- Consumes: everything above, through the CLI only; `FakeRemoteStore`, `capture_payload`, `TOOLSET_KEY`, `TOOLSET_PAYLOAD` from `tests/unit/fake_remote_store.py`.

- [ ] **Step 1: Write the test**

Create `tests/integration/test_capture_lifecycle.py`:

```python
"""Weeks of production captures in a bucket stay one bounded, fresh suite (spec, Testing)."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.captures.store_uri import StoreURI
from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config
from evalshift_cli.cli.main import app
from tests.unit.fake_remote_store import (
    TOOLSET_KEY,
    TOOLSET_PAYLOAD,
    FakeRemoteStore,
    capture_payload,
)

runner = CliRunner()


def _day(day: int, month: int = 6) -> datetime:
    return datetime(2026, month, day, 12, tzinfo=UTC)


class _Bucket:
    """A fake bucket the SDK 'writes' to between syncs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.store = FakeRemoteStore({TOOLSET_KEY: TOOLSET_PAYLOAD})
        monkeypatch.setattr(capture_module, "open_store", lambda parsed: self._open(parsed))

    def _open(self, parsed: StoreURI) -> FakeRemoteStore:
        return self.store

    def record(self, cid: str, when: datetime, *, model: str = "claude-sonnet-5-20260301") -> None:
        key = f"captures/planner/{cid}.json"
        self.store.objects[key] = capture_payload(
            cid,
            "planner",
            model_input=f"plan my day {cid}",
            created_at=when.isoformat(),
            model_id=model,
        )
        self.store.modified[key] = when


def _config(path: Path, *, source_model: str = "claude-sonnet-5") -> None:
    text = render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
    text = re.sub(r"(?m)^  source_model: .*$", f"  source_model: {source_model}", text)
    text += "\ncaptures:\n  store: s3://acme-evals/testapp\n  max_age: 14d\n  max_cases: 4\n"
    path.write_text(text, encoding="utf-8")


def _sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now: datetime) -> Any:
    monkeypatch.setattr(capture_module, "_now", lambda: now)
    result = runner.invoke(
        app,
        ["capture", "sync", "--config", str(tmp_path / "evalshift.yaml"), "--base", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    return result


def _cases(tmp_path: Path) -> set[str]:
    return {p.stem for p in (tmp_path / "suites" / "planner").glob("*.json")}


def test_weeks_of_captures_stay_one_bounded_fresh_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml")

    # Week 1: six captures, a suite of four.
    for i in range(6):
        bucket.record(f"w1_{i}", _day(10))
    _sync(tmp_path, monkeypatch, _day(12))
    week_1 = _cases(tmp_path)
    assert len(week_1) == 4

    # Week 2, inside max_age: more captures, but nothing is stale -- the suite holds still.
    for i in range(6):
        bucket.record(f"w2_{i}", _day(17))
    _sync(tmp_path, monkeypatch, _day(19))
    assert _cases(tmp_path) == week_1

    # Two weeks later: week-1 cases are stale and fresh captures replace them.
    for i in range(3):
        bucket.record(f"w4_{i}", _day(1, month=7))
    _sync(tmp_path, monkeypatch, _day(2, month=7))
    cases = _cases(tmp_path)
    assert len(cases) == 4
    assert {f"w4_{i}" for i in range(3)} <= cases

    # A fresh checkout: suites committed, local mirror gone. Promoted captures are not
    # downloaded again, and nothing outside max_age is.
    shutil.rmtree(tmp_path / "captures")
    bucket.store.gets.clear()
    result = _sync(tmp_path, monkeypatch, _day(2, month=7))
    downloaded = {key.rsplit("/", 1)[-1].removesuffix(".json") for key in bucket.store.gets}
    assert not downloaded & cases
    assert not any(cid.startswith(("w1_", "w2_")) for cid in downloaded)
    assert _cases(tmp_path) == cases
    assert "promoted" in result.stdout


def test_a_production_migration_retires_the_old_models_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml")
    for i in range(4):
        bucket.record(f"old_{i}", _day(10))
    _sync(tmp_path, monkeypatch, _day(12))

    _config(tmp_path / "evalshift.yaml", source_model="claude-sonnet-5-5")
    for i in range(2):
        bucket.record(f"new_{i}", _day(13), model="claude-sonnet-5-5-20260601")
    result = _sync(tmp_path, monkeypatch, _day(14))

    assert {"new_0", "new_1"} <= _cases(tmp_path)
    assert len(_cases(tmp_path)) == 4
    assert "retired 2 (2 on another model)" in " ".join(result.stdout.split())


def test_a_wrong_source_model_trips_the_guardrail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml", source_model="gpt-9")
    for i in range(3):
        bucket.record(f"cap_{i}", _day(10))

    result = _sync(tmp_path, monkeypatch, _day(12))

    assert len(_cases(tmp_path)) == 3
    assert "not filtering by model — check defaults.source_model" in " ".join(
        result.stdout.split()
    )
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/integration/test_capture_lifecycle.py -q`
Expected: PASS. A failure here is a real integration bug in Tasks 7–8: debug it (superpowers:systematic-debugging) rather than loosening the assertions.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_capture_lifecycle.py
git commit -m "test(capture): weeks of bucket captures stay one bounded fresh suite

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Docs, changelog, spec and plan

**Files:**
- Modify: `CHANGELOG.md`, `DOCS.md`, `llms-full.txt`, `docs/configuration.md`, `docs/sdk.md`, `docs/conversations.md`
- Add (forced, gitignored): `docs/superpowers/specs/2026-10-16-bounded-capture-suites-design.md`, `docs/superpowers/plans/2026-10-16-bounded-capture-suites.md`

- [ ] **Step 1: CHANGELOG**

Under `## [Unreleased]` in `CHANGELOG.md`:

```markdown
### Added

- **Bounded capture suites.** Two optional settings under `captures:` —
  `max_age: 14d` and `max_cases: 200` — keep each captured suite a capped,
  route-balanced sample of recent production on `defaults.source_model`.
  `capture sync` only promotes captures recorded within `max_age` on the source
  model, never grows a suite past `max_cases`, and swaps stale cases (too old, or
  recorded on another model) for fresh captures as they arrive — never shrinking
  a suite that has nothing to replace them with. A conversation enters whole or
  not at all; cases a human marked `provenance: reviewed` are never retired; the
  same captures give the same suite on any machine. `max_age` also bounds what
  `sync` downloads from `captures.store` (`--since` still overrides it). Sync
  prints one line per suite: kept / retired / added, and what was not eligible.
  If no capture was recorded on `defaults.source_model`, sync warns and does not
  filter by model rather than empty the suite.
- `doctor` gains a `suite freshness` row per captured suite (newest case older
  than `max_age`, cases on another model, over `max_cases`) and a
  `captures limits` warning when `captures.store` is set without limits;
  `compare` prints the same findings above the verdict, so a CI-only team sees
  its suite went stale.
- Promoted case files record `captured_at` and `source_model`; `capture sync`
  backfills them on existing cases whose capture is still local.

### Changed

- `evalshift init` scaffolds `captures: {max_age: 14d, max_cases: 200}` (with a
  commented `store:`). Existing configs are untouched — without the two keys,
  `capture sync` behaves exactly as before.
- `capture sync` now reads `evalshift.yaml` under `--offline` too, for the limits.
- Older CLIs reject the new `captures` keys and the new case-file fields: bump
  the CLI everywhere `capture` commands run (and a pinned CI action's
  `evalshift-version`) before adding them. CI's `compare` reads only
  `golden.jsonl`, whose format is unchanged.
```

- [ ] **Step 2: Reference docs**

- `docs/configuration.md` `## captures` (line ~1066): add `max_age` and `max_cases` rows to the field table (types, defaults `None`, the one-sentence meaning from the `CapturesConfig` docstring); add a subsection **Bounded suites** that states the opt-in rule, eligibility (§3), the plan (§5 rules 1–4, in prose), the guardrail, the per-suite line format with an example, and the fetch window; extend the bucket-retention paragraph (line ~1104) with an expiration rule for the `captures/` prefix only (not `toolsets/`, which committed suites reference):

  ```bash
  aws s3api put-bucket-lifecycle-configuration --bucket acme-evals \
    --lifecycle-configuration '{"Rules": [{"ID": "expire-evalshift-captures",
      "Status": "Enabled", "Filter": {"Prefix": "support-agent/captures/"},
      "Expiration": {"Days": 30}}]}'
  ```

  plus one line each for GCS (Object Lifecycle Management `age` condition) and Azure (lifecycle management `daysAfterModificationGreaterThan`), and the advice "expire after a period ≥ `max_age`".
- `DOCS.md`: `## Capturing from production` — extend the `captures:` YAML example with `max_age` / `max_cases` and add a short "Bounded suites" paragraph linking to `docs/configuration.md#captures`; the sync steps list gains a step "With `max_age` / `max_cases` set: filter, plan, retire" (one paragraph); `### Top-level fields` `captures` row names the two keys; the doctor bullet under `## How it works` names the `suite freshness` and `captures limits` rows; `## Command reference` sync/doctor/compare entries mention them; `evalshift init` description says it scaffolds the limits.
- `llms-full.txt`: mirror every DOCS.md change in its matching section (init 137-158, doctor 162-198, compare 235-239, captures 327-420, and the `captures:` schema block ~801-811).
- `docs/sdk.md` §2: after "Bucket retention is a lifecycle rule", add "— and `captures.max_age` / `captures.max_cases` in `evalshift.yaml` keep the suites built from it bounded (see [configuration](configuration.md#captures))."; §3: show the two keys in the `captures:` example.
- `docs/conversations.md`: one sentence — with `captures.max_cases` set, a conversation's new turns enter a suite together or not at all.

- [ ] **Step 3: Docs tests**

Run: `uv run pytest tests/unit/test_docs_currency.py tests/unit/test_init.py -q`
Expected: PASS.

- [ ] **Step 4: Commit (spec and plan forced in)**

```bash
git add CHANGELOG.md DOCS.md llms-full.txt docs/configuration.md docs/sdk.md docs/conversations.md
git add -f docs/superpowers/specs/2026-10-16-bounded-capture-suites-design.md docs/superpowers/plans/2026-10-16-bounded-capture-suites.md
git commit -m "docs: bounded capture suites (captures.max_age / captures.max_cases)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Verify and open the PR

- [ ] **Step 1: Full gate**

Run: `make ci` then `pre-commit run --all-files`
Expected: both green; show the output (superpowers:verification-before-completion).

- [ ] **Step 2: Real-project smoke (read-only on the bucket)**

In `/home/lukas/repos/testApp` with this branch's CLI (`uv run --project /path/to/worktree evalshift ...`), on a scratch copy of its `evalshift.yaml` with `max_age: 14d` / `max_cases: 200` added: `evalshift capture sync --print` then `evalshift doctor`. Confirm the per-suite lines and freshness rows read sensibly. Do not commit anything in testApp; restore its config.

- [ ] **Step 3: Code review**

Use superpowers:requesting-code-review on the branch (reviewer on `model: opus`), with the spec and this plan as context and the Review Focus list as the checklist. Address findings via superpowers:receiving-code-review.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/bounded-capture-suites
gh pr create --title "feat(capture): bounded capture suites (captures.max_age / captures.max_cases)" --body "<summary of the CHANGELOG entry + test plan + link to the spec>

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

The maintainer merges the PR; the release (version bump, tag) is a separate PR.

## Execution notes

What shipped differs from the tasks above in these ways (controller rulings during Tasks 1–10; the spec and the user docs describe the shipped behaviour):

1. Model matching (Task 3): `same_model` compares bare names only (lower-cased, provider prefix dropped, one trailing `-YYYYMMDD` / `-YYYY-MM-DD` dropped). The registry/alias fallback shown in Task 3 was removed, so a `defaults.source_model` spelled as an alias trips the guardrail warning.
2. Planner (Task 6): a stale case whose content an added capture replays is retired outright, even under the cap; other stale cases leave only while `excess > 0`. A case with an unknown capture time counts as oldest when ordering retirement and trim (spec §5 amended).
3. Dedup with limits (Task 8): a stale case gives up its content claim only when `max_cases` is set and the case is not reviewed; with `max_age` alone a fresh re-capture of a stale case's question is skipped as a duplicate.
4. Twin retirement applies under `--keep-duplicates` too.
5. Fetch window (Task 7): only `capture sync`'s automatic fetch follows `max_age`; `capture list` and `capture fetch` do not; `--since` overrides it.
6. Limits warning (Task 7): `could not read {path} ({reason}); syncing without captures.max_age / captures.max_cases.` prints online too, not only under `--offline`.
7. With `max_cases` set, sync also trims suites it did not build: hand-promoted suites and suites whose entry is `managed: false`. Reviewed cases are never retired.
8. A suite directory whose case files cannot be read is left untouched (its `golden.jsonl` and `suites:` entry stay); sync warns and continues.
9. doctor `captures limits` row (Task 9, per spec §7): warns when `captures.store` is set and `captures.max_cases` is not, with `suites grow with every capture sync; set captures.max_cases and captures.max_age`, or `…; set captures.max_cases` when only `max_age` is set.
10. compare freshness lines (Task 9): printed for the suite compare runs when named with `--suite-name` or auto-selected (single wired suite), not for `--suite <path>`; measured against `defaults.source_model`, not `--from`.
