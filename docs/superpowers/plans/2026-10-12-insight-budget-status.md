# Insight Budget Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The machine-written run narrative names the budget that actually failed a run, and a narrative that blames a budget that held never ships.

**Architecture:** The prompt's FACTS block gains each budget's *status* beside its limit (`breached` / `within` / `measured nothing` / unconfirmed), the `failed_budgets` list the templated fallback already uses, and the decision's own `reason`. The instruction lets only a breached budget be named as the cause. A third validator, `validate_blame`, scans a `fail` run's verdict summary and recommendation for a held budget that shares a clause with a non-negated blame word; a hit rejects the generation like an invented number, the retry names the misblamed labels, and two rejections ship the templated prose (which is already correct).

**Tech Stack:** Python 3.11+, dataclasses, `re`, pytest (async tests via the repo's pytest-asyncio config), uv.

**Spec:** No separate spec file — the design was agreed in chat on 2026-10-12 ("status + rule + check"). The motivating run is `r_20261012_converter_d418e5` in `/home/lukas/repos/testApp`: verdict `fail` on `max_latency_increase` alone (+175.3% vs +30%), every other budget within; the narrative said the run "failed to meet the budget for Trace-rule violations" and recommended "address the Trace-rule violations" — a budget that passed at 0 against a limit of 1. The prompt listed each budget's limit with no result, so the model guessed.

## Global Constraints

- Python floor 3.11: no 3.12+ syntax. Every module starts with `from __future__ import annotations`; public functions get Google-style docstrings; `mypy --strict src/evalshift_cli` stays clean.
- No config, suite, `state.json`, `scores.jsonl`, `insights.json` or bundle schema change. `Insight` gets no new field. `Facts` is in-memory only and may grow.
- Everything the model may copy stays digit-free or enters `Facts.allowed_numbers`: the four status words carry no digits; slice-budget limits and the `verdict_reason` text's figures join the allow-list.
- New FACTS keys (`failed_budgets`, `verdict_reason`) join `_FACTS_FIELD_NAMES` so an echo is rejected like `coverage_basis` is today.
- User-facing strings, verbatim:
  - budget line: `budget "{label}": limit {limit}, {status}`
  - status words: `breached`, `within`, `measured nothing`, `over its limit but not confirmed at this sample size`
  - retry hint: `Your previous answer blamed budgets that held: {labels}. Only the budgets under failed_budgets were breached; a budget marked within may be described as within policy or left out, never as a cause of the verdict or as something to address.`
- The blame scan runs only when `facts.verdict == "fail"`, only over `verdict_summary` and `recommendation`, only for labels whose status is `within`.
- Gate ("green" = all four, as CI): `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest -q` (coverage floor 90). `make ci` is the exact CI mirror.
- Commit with `uv run --with pre-commit git commit …` (the shared hook points at a deleted venv). Never `--no-verify`. Conventional Commits, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. `docs/superpowers/` is gitignored: add this plan with `git add -f`.
- Always `git -C /home/lukas/repos/evalshift/evalshift-cli` (= `$WT`). Before each commit confirm the branch is `fix/insight-budget-status`, not `main`.

## Review Focus

- A `fail` whose summary *mentions* a held budget without blaming it ("Trace-rule violations stayed within budget") must ship on the first attempt — Task 3 `test_a_fail_may_mention_a_held_budget_without_blaming_it`.
- A negated accusation ("was not breached", "no breach of") is not blame — Task 3 `test_a_negated_blame_is_not_blame`.
- The shipped sentence and the shipped recommendation are both caught, including lower-case and unhyphenated spellings of the label — Task 3 `test_blame_catches_the_shipped_sentence`, `test_blame_catches_advice_to_address_a_held_budget`, `test_blame_matches_case_and_hyphen_variants`.
- A `verdict_reason` carrying figures (`effect size -2.15, adjusted p 0.0004`) must not get a generation that copies them rejected as invented — Task 1 `test_the_decision_reason_travels_with_the_facts`, Task 2 `test_a_copied_reason_figure_is_not_an_invented_number`.
- A slice budget's limit (only ever rendered in the new status lines) is a copyable figure — Task 1 `test_a_slice_budget_is_listed_under_its_scope`.

---

### Task 1: Facts carry each budget's status and the decision's reason

**Files:**
- Modify: `src/evalshift_cli/insights/facts.py` (`Facts`, `build_facts`, `_failed_budget_label` → `_scoped_budget_label`, new `_budget_status`, `_allowed_numbers`, `__all__`)
- Modify: `tests/unit/insights_factories.py` (new `converter_decision`, `SEVERITY_REASON`)
- Test: `tests/unit/test_insight_facts.py`

**Interfaces:**
- Consumes: `BudgetResult.measured/passed/conclusive/scope`, `gating_budget_results(decision)`, `MigrationDecision.reason` (all existing in `analysis/policy.py`).
- Produces: `facts.BUDGET_BREACHED = "breached"`, `BUDGET_WITHIN = "within"`, `BUDGET_UNMEASURED = "measured nothing"`, `BUDGET_UNCONFIRMED = "over its limit but not confirmed at this sample size"`; `Facts.budget_statuses: list[tuple[str, str, str]]` (display label, rendered limit, status), one per gating budget row in policy order; `Facts.verdict_reason: str` (`""` when the decision has none); factories `converter_decision() -> MigrationDecision`, `SEVERITY_REASON: str`.

- [ ] **Step 1: Add the shared factories**

In `tests/unit/insights_factories.py`, after `unmeasured_budget` (line ~257) add:

```python
#: A severity-only fail's ``MigrationDecision.reason``, as the policy words it.
#: Carries figures: a generation that copies them must not be rejected.
SEVERITY_REASON = (
    "semantic.cosine regressed at critical severity on prompt replay "
    "(effect size -2.15, adjusted p 0.0004). A critical or high regression from a "
    "blocking evaluator fails the run on its own, whatever the budgets say."
)


def converter_decision() -> MigrationDecision:
    """r_20261012_converter_d418e5: the latency ceiling breached, every other budget held.

    The run whose narrative blamed "Trace-rule violations" — a budget that
    passed at 0 against a limit of 1 — because the prompt listed each budget's
    limit without its result. Six held budgets, one breached, one blind.
    """
    return decision(
        verdict="fail",
        budget_results=[
            *budgets()[:5],
            BudgetResult(
                name="max_latency_increase", observed=1.753, allowed=0.30, passed=False
            ),
            BudgetResult(name="max_invariant_violations", observed=0.0, allowed=1.0, passed=True),
            unmeasured_budget(),
        ],
    )
```

`decision` is defined later in the same file; a module-level function body resolves it at call time, so the order is fine.

- [ ] **Step 2: Write the failing tests**

In `tests/unit/test_insight_facts.py` add `MigrationDecision` to the `evalshift_cli.analysis.policy` import, add `from dataclasses import replace`, add `BUDGET_BREACHED, BUDGET_UNCONFIRMED, BUDGET_UNMEASURED, BUDGET_WITHIN` to the `evalshift_cli.insights.facts` import (alongside `build_facts`), and add `SEVERITY_REASON` and `converter_decision` to the `tests.unit.insights_factories` import. Append:

```python
# ---------------------------------------------------------------------------
# Budget status and the decision's reason — what a FAIL's cause is read from
# ---------------------------------------------------------------------------


def _status_of(facts_statuses: list[tuple[str, str, str]], label: str) -> tuple[str, str]:
    (hit,) = [(limit, status) for name, limit, status in facts_statuses if name == label]
    return hit


def test_every_gating_budget_carries_a_status(sample_run: dict[str, Any]) -> None:
    """The converter run: one breached, one blind, the rest held."""
    sample_run["decision"] = converter_decision()
    facts = build_facts(**sample_run)

    assert len(facts.budget_statuses) == 8
    assert _status_of(facts.budget_statuses, "Latency increase") == ("+30%", BUDGET_BREACHED)
    assert _status_of(facts.budget_statuses, "Trace-rule violations") == ("1", BUDGET_WITHIN)
    assert _status_of(facts.budget_statuses, "Cost increase") == ("+200%", BUDGET_WITHIN)
    assert _status_of(facts.budget_statuses, "Tool-selection divergence") == (
        "20%",
        BUDGET_UNMEASURED,
    )
    assert facts.failed_budgets == ["Latency increase"]


def test_an_unconfirmed_breach_has_its_own_status(sample_run: dict[str, Any]) -> None:
    over = BudgetResult(
        name="max_overall_regression_rate",
        observed=0.1,
        allowed=0.03,
        passed=False,
        conclusive=False,
        ci_low=0.0,
        ci_high=0.3,
    )
    sample_run["decision"] = decision(
        verdict="inconclusive", budget_results=[over, *budgets()[1:]]
    )
    facts = build_facts(**sample_run)
    assert _status_of(facts.budget_statuses, "Overall regression rate") == (
        "3%",
        BUDGET_UNCONFIRMED,
    )
    assert facts.failed_budgets == []


def test_a_slice_budget_is_listed_under_its_scope(sample_run: dict[str, Any]) -> None:
    """A slice row is a gating budget too, and its limit is a copyable figure."""
    facts = build_facts(**_slice_budget_run(sample_run, extra=[]))
    assert _status_of(facts.budget_statuses, "'security' slice overall regression rate") == (
        "0%",
        BUDGET_BREACHED,
    )
    assert "0%" in facts.allowed_numbers


def test_the_decision_reason_travels_with_the_facts(sample_run: dict[str, Any]) -> None:
    sample_run["decision"] = replace(decision(verdict="fail"), reason=SEVERITY_REASON)
    facts = build_facts(**sample_run)
    assert facts.verdict_reason == SEVERITY_REASON
    # Its figures are facts the model may copy.
    assert {"-2.15", "2.15", "0.0004"} <= facts.allowed_numbers


def test_a_decision_without_a_reason_renders_none(sample_run: dict[str, Any]) -> None:
    assert build_facts(**sample_run).verdict_reason == ""
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/unit/test_insight_facts.py -q -k "status or reason or scope"`
Expected: FAIL at import — `ImportError: cannot import name 'BUDGET_BREACHED'`.

- [ ] **Step 4: Implement the facts**

In `src/evalshift_cli/insights/facts.py`:

1. After `_COUNT_BUDGETS` (line ~84) add:

```python
#: Budget status words, rendered beside each budget's limit in the FACTS
#: block. Digit-free plain English — the model copies them into prose — and
#: the only vocabulary the blame validator keys on.
BUDGET_BREACHED = "breached"
BUDGET_WITHIN = "within"
BUDGET_UNMEASURED = "measured nothing"
BUDGET_UNCONFIRMED = "over its limit but not confirmed at this sample size"
```

2. In `Facts`, after `overall_blocking_evaluators` add:

```python
    #: One ``(display label, rendered limit, status)`` per gating budget row —
    #: the overall rows and the distinct slice rows, in policy order. Handed to
    #: the model beside each limit so a FAIL's cause is read off the line, not
    #: guessed: a narrative once blamed "Trace-rule violations" on a run where
    #: that budget held and only the latency ceiling was breached, because the
    #: prompt listed limits without results.
    budget_statuses: list[tuple[str, str, str]] = field(default_factory=list)
    #: The decision's own explanation of its verdict
    #: (``MigrationDecision.reason``), or ``""`` when it carries none. The same
    #: sentence the report hero renders, so the two surfaces cannot disagree.
    verdict_reason: str = ""
```

3. In `build_facts`, directly after the `failed_budgets = _unique(...)` statement add:

```python
    budget_statuses = [
        (_scoped_budget_label(b), _budget_limit(b.name, b.allowed), _budget_status(b))
        for b in gating_budget_results(decision)
    ]
    verdict_reason = decision.reason or ""
```

   and in the `failed_budgets` comprehension rename the call `_failed_budget_label(b)` → `_scoped_budget_label(b)`.

4. In the `Facts(...)` return, change the `allowed_numbers=_allowed_numbers(` call to pass two new keywords after `budget_limits=budget_limits,`:

```python
            slice_limits=[limit for _, limit, _ in budget_statuses],
            verdict_reason=verdict_reason,
```

   and add `budget_statuses=budget_statuses,` and `verdict_reason=verdict_reason,` after `overall_blocking_evaluators=...`.

5. Rename `_failed_budget_label` → `_scoped_budget_label` and replace its docstring with:

```python
    """A budget's display name, scoped when it is a slice's own row.

    A bare "overall regression rate" for a slice row would send the reader to
    the overall row — green in exactly the run that names it. Worded so the
    templated advice reads "the 'security' slice overall regression rate
    budget was breached" and the FACTS block lists the row under that name.
    """
```

6. After `_budget_limit` add:

```python
def _budget_status(budget: BudgetResult) -> str:
    """Which of the four things a budget row says, as a word the prose may use.

    ``measured`` first: the ``0/0`` default passes by arithmetic, and calling
    it ``within`` is the clean-sweep claim the coverage facts exist to stop.
    """
    if not budget.measured:
        return BUDGET_UNMEASURED
    if budget.passed:
        return BUDGET_WITHIN
    if budget.conclusive:
        return BUDGET_BREACHED
    return BUDGET_UNCONFIRMED
```

7. In `_allowed_numbers`, add the parameters `slice_limits: Sequence[str],` and `verdict_reason: str,` after `budget_limits: dict[str, str],`, extend the docstring's first sentence list with "the slice budgets' limits (rendered only in the status lines) and the decision reason's own figures", and add to `sources` after `*budget_limits.values(),`:

```python
        *slice_limits,
        verdict_reason,
```

8. Add `"BUDGET_BREACHED", "BUDGET_UNCONFIRMED", "BUDGET_UNMEASURED", "BUDGET_WITHIN",` to `__all__` (alphabetical, before `"NOT_AVAILABLE"`).

- [ ] **Step 5: Run the facts tests and the type check**

Run: `uv run pytest tests/unit/test_insight_facts.py -q && uv run mypy --strict src/evalshift_cli/insights`
Expected: all pass (the five new tests plus every existing one — `failed_budgets` behaviour is unchanged by the rename); mypy `Success`.

- [ ] **Step 6: Commit**

```bash
WT=/home/lukas/repos/evalshift/evalshift-cli
git -C $WT branch --show-current   # fix/insight-budget-status
git -C $WT add src/evalshift_cli/insights/facts.py tests/unit/insights_factories.py tests/unit/test_insight_facts.py
cd $WT && uv run --with pre-commit git commit -m "feat(insights): facts carry each budget's status and the decision's reason

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The prompt shows the statuses and the instruction rules on them

**Files:**
- Modify: `src/evalshift_cli/insights/generator.py` (`_INSTRUCTION`, `_FACTS_FIELD_NAMES`, `_facts_block`)
- Test: `tests/unit/test_insight_generator.py`

**Interfaces:**
- Consumes: `Facts.budget_statuses`, `Facts.failed_budgets`, `Facts.verdict_reason` (Task 1).
- Produces: FACTS lines `budget "{label}": limit {limit}, {status}`, `failed_budgets: {a, b}`, `verdict_reason: {text}`; the instruction rule quoted in Step 3.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_insight_generator.py` add `from dataclasses import replace`, and add `SEVERITY_REASON`, `converter_decision` to the `tests.unit.insights_factories` import. Change the existing assertion at line ~424 from `assert 'budget "Cost increase": +200%' in prompt` to `assert 'budget "Cost increase": limit +200%, within' in prompt` (it pinned the limit-only line this task replaces). Append:

```python
# ---------------------------------------------------------------------------
# A FAIL's cause is a fact, not an inference
# ---------------------------------------------------------------------------


async def test_each_budget_line_carries_its_status(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The converter run: the model is told which budget broke and which held.

    The shipped narrative blamed "Trace-rule violations" because every budget
    line read as a bare limit — nothing said which one the run had failed on.
    """
    sample_run["decision"] = converter_decision()
    fake_client.queue_responses(_payload())
    await _generate(build_facts(**sample_run), fake_client)

    prompt = fake_client.prompts[0]
    assert '\nbudget "Latency increase": limit +30%, breached\n' in prompt
    assert '\nbudget "Trace-rule violations": limit 1, within\n' in prompt
    assert '\nbudget "Tool-selection divergence": limit 20%, measured nothing\n' in prompt
    assert "\nfailed_budgets: Latency increase\n" in prompt


async def test_a_passing_run_lists_no_failed_budgets_and_no_reason(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(_payload())
    await _generate(sample_facts, fake_client)

    prompt = fake_client.prompts[0]
    assert "\nfailed_budgets: " not in prompt
    assert "\nverdict_reason: " not in prompt
    assert '\nbudget "Cost increase": limit +200%, within\n' in prompt


async def test_the_verdict_reason_reaches_the_prompt(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    sample_run["decision"] = replace(decision(verdict="fail"), reason=SEVERITY_REASON)
    fake_client.queue_responses(_payload())
    await _generate(build_facts(**sample_run), fake_client)

    assert f"\nverdict_reason: {SEVERITY_REASON}\n" in fake_client.prompts[0]


async def test_a_copied_reason_figure_is_not_an_invented_number(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The reason's own figures are facts; copying them must not cost a retry."""
    sample_run["decision"] = replace(decision(verdict="fail"), reason=SEVERITY_REASON)
    fake_client.queue_responses(
        _payload(verdict_summary="semantic.cosine regressed with effect size -2.15 (p 0.0004).")
    )
    insight = await _generate(build_facts(**sample_run), fake_client)

    assert insight.verdict_summary.endswith("(p 0.0004).")
    assert fake_client.call_count == 1


async def test_the_instruction_rules_on_budget_status(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """A plain-English accusation carries no digit; the rule has to be spelled out."""
    fake_client.queue_responses(_payload())
    await _generate(sample_facts, fake_client)

    instruction = fake_client.prompts[0].split("FACTS (")[0]
    assert '"breached"' in instruction
    assert '"within"' in instruction
    assert "failed_budgets" in instruction
    assert "verdict_reason" in instruction


async def test_an_echoed_failed_budgets_key_is_rejected_and_retried(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    sample_run["decision"] = converter_decision()
    fake_client.queue_responses(
        _payload(verdict_summary="failed_budgets: Latency increase."),
        _payload(verdict_summary="The latency ceiling of +30% was exceeded."),
    )
    insight = await _generate(build_facts(**sample_run), fake_client)

    assert insight.verdict_summary == "The latency ceiling of +30% was exceeded."
    assert fake_client.call_count == 2
```

`decision` is already imported in this file from the factories (check the import block; add it if not).

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_insight_generator.py -q -k "status or failed_budgets or reason"`
Expected: `test_each_budget_line_carries_its_status`, `test_a_passing_run_lists_no_failed_budgets_and_no_reason`, `test_the_verdict_reason_reaches_the_prompt`, `test_the_instruction_rules_on_budget_status`, `test_an_echoed_failed_budgets_key_is_rejected_and_retried` FAIL (old line format; no `failed_budgets`/`verdict_reason` lines; echo not rejected). `test_a_copied_reason_figure_is_not_an_invented_number` passes already (Task 1 put the figures on the allow-list). The pre-existing test at line ~424 FAILS on its updated assertion.

- [ ] **Step 3: Implement the prompt**

In `src/evalshift_cli/insights/generator.py`:

1. In `_INSTRUCTION`, after the `unmeasured_budgets` rule (the bullet ending `and say their silence is unknown.`) insert:

```
- Every "budget" line ends with its status. Only a budget whose status is
  "breached" — the same ones listed under "failed_budgets" — may be named as
  why the run failed or as something to fix. A budget marked "within" held:
  never write that it was breached, exceeded, failed, missed, violated or
  must be addressed. When "verdict_reason" is present it is the decision's
  own explanation of the verdict, and your verdict summary must agree with it.
```

2. Add `"failed_budgets",` and `"verdict_reason",` to `_FACTS_FIELD_NAMES` (after `"coverage_basis",`).

3. In `_facts_block`, after the `coverage_basis` line add:

```python
    if facts.failed_budgets:
        lines.append(f"failed_budgets: {', '.join(facts.failed_budgets)}")
    if facts.verdict_reason:
        lines.append(f"verdict_reason: {facts.verdict_reason}")
```

   and replace the budget loop and its comment (`# Budgets and categories reach the model …` through `lines.append(f'budget "{BUDGET_LABELS.get(name, name)}": {limit}')`) with:

```python
    # Budgets and categories reach the model under their display names: it is
    # told to copy tokens verbatim, so whatever appears here is what a reader
    # may end up seeing in the narrative. Each budget carries its status, so
    # the cause of a FAIL is read off the line rather than guessed from the
    # limit — a narrative once blamed a budget that held.
    for label, limit, status in facts.budget_statuses:
        lines.append(f'budget "{label}": limit {limit}, {status}')
```

   `BUDGET_LABELS` stays imported — `banned_identifiers` still uses it.

- [ ] **Step 4: Run the generator tests and the type check**

Run: `uv run pytest tests/unit/test_insight_generator.py tests/unit/test_insight_facts.py tests/unit/test_insights_stage.py -q && uv run mypy --strict src/evalshift_cli/insights`
Expected: all pass; mypy `Success`.

- [ ] **Step 5: Commit**

```bash
WT=/home/lukas/repos/evalshift/evalshift-cli
git -C $WT branch --show-current   # fix/insight-budget-status
git -C $WT add src/evalshift_cli/insights/generator.py tests/unit/test_insight_generator.py
cd $WT && uv run --with pre-commit git commit -m "fix(insights): tell the narrative which budget failed the run

The prompt listed each budget's limit with no result, so a FAIL on the
latency ceiling alone was explained as 'failed to meet the budget for
Trace-rule violations' — a budget that held. Each budget line now ends
with its status, the breached ones ride on failed_budgets, the decision's
reason travels with them, and the instruction allows only a breached
budget to be named as the cause.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: A fail that blames a held budget is rejected, retried, then templated

**Files:**
- Modify: `src/evalshift_cli/insights/generator.py` (new `_BLAME_RE`, `_NEGATION_RE`, `_CLAUSE_RE`, `_normalise`, `validate_blame`; `GenerationError.misblamed`; `generate_insight`; `build_prompt`; `_build_insight`; `__all__`)
- Test: `tests/unit/test_insight_generator.py`

**Interfaces:**
- Consumes: `Facts.verdict`, `Facts.budget_statuses`, `facts.BUDGET_WITHIN` (Task 1); `fallback_insight` (existing).
- Produces: `validate_blame(text: str, within: Sequence[str]) -> list[str]`; `GenerationError(message, *, tokens=None, misblamed=None)` with `.misblamed: list[str]`; `build_prompt(facts, *, invented=None, misblamed=None)`.

- [ ] **Step 1: Write the failing validator tests**

Append to `tests/unit/test_insight_generator.py` (and add `validate_blame` to the `evalshift_cli.insights.generator` import):

```python
# ---------------------------------------------------------------------------
# The blame validator
# ---------------------------------------------------------------------------

_HELD = ["Trace-rule violations", "Cost increase"]


def test_blame_catches_the_shipped_sentence() -> None:
    text = (
        "The migration failed because the target model exceeded the latency ceiling "
        "of +30% and failed to meet the budget for Trace-rule violations."
    )
    assert validate_blame(text, _HELD) == ["Trace-rule violations"]


def test_blame_catches_advice_to_address_a_held_budget() -> None:
    text = (
        "Investigate the cause of the latency increase and address the Trace-rule "
        "violations before re-evaluating."
    )
    assert validate_blame(text, _HELD) == ["Trace-rule violations"]


def test_a_held_budget_may_be_called_within() -> None:
    assert validate_blame("Trace-rule violations stayed within budget.", _HELD) == []


def test_a_contrast_clause_separates_the_blame() -> None:
    text = "Latency increase was exceeded while Trace-rule violations stayed within budget."
    assert validate_blame(text, _HELD) == []


def test_a_sentence_boundary_separates_the_blame() -> None:
    text = "The latency ceiling was exceeded. Trace-rule violations and Cost increase held."
    assert validate_blame(text, _HELD) == []


def test_a_negated_blame_is_not_blame() -> None:
    assert validate_blame("The Trace-rule violations budget was not breached.", _HELD) == []
    assert validate_blame("No breach of Cost increase was observed.", _HELD) == []


def test_blame_matches_case_and_hyphen_variants() -> None:
    assert validate_blame("the run failed on trace rule violations", _HELD) == [
        "Trace-rule violations"
    ]


def test_a_budget_not_in_the_held_list_may_be_blamed() -> None:
    assert validate_blame("Latency increase was breached.", _HELD) == []


def test_every_misblamed_budget_is_returned_once_in_list_order() -> None:
    text = "Cost increase was exceeded and Trace-rule violations were violated; Cost increase failed."
    assert validate_blame(text, _HELD) == ["Trace-rule violations", "Cost increase"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_insight_generator.py -q -k blame`
Expected: FAIL at import — `ImportError: cannot import name 'validate_blame'`.

- [ ] **Step 3: Implement the validator**

In `src/evalshift_cli/insights/generator.py`:

1. Add `from collections.abc import Sequence` to the imports (after `import re`), and `BUDGET_WITHIN` to the `evalshift_cli.insights.facts` import.

2. After `_JSON_OBJECT_RE` add:

```python
#: Words that lay a verdict at a budget's door. A budget marked "within" that
#: shares a clause with one of these, un-negated, is a cause the model
#: invented: a shipped narrative said a run "failed to meet the budget for
#: Trace-rule violations" when that budget held and only latency broke.
_BLAME_RE = re.compile(
    r"\b(?:breach\w*|exceed\w*|fail\w*|violat\w*|address\w*|miss(?:ed|es|ing)?|"
    r"unmet|not met|over (?:the|its)|above (?:the|its)|broke|broken)\b",
    re.IGNORECASE,
)

#: A blame word shortly after one of these is a denial ("was not breached",
#: "no breach of"), not an accusation. Bounded so a "not" three clauses back
#: cannot excuse an accusation.
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without)\b[^.;,]{0,24}$", re.IGNORECASE)

#: Clause boundaries. A contrast word starts a new claim, so "latency was
#: exceeded while Trace-rule violations stayed within budget" blames only
#: latency. A bare "and" is deliberately not one: "exceeded latency and failed
#: to meet Trace-rule violations" is the shipped sentence, and both halves
#: accuse.
_CLAUSE_RE = re.compile(
    r"[.;:!?]\s+|\s(?:while|but|whereas|although|though|however)\s|,\s*(?:and|but)\s",
    re.IGNORECASE,
)
```

3. After `validate_numbers` add:

```python
def validate_blame(text: str, within: Sequence[str]) -> list[str]:
    """Return the budgets in ``within`` that ``text`` blames for the verdict.

    A budget that held may be mentioned — "Trace-rule violations stayed within
    budget" is accurate — but not accused: it may not share a clause with a
    blame word unless that word is negated. A plain-English accusation carries
    no digit for :func:`validate_numbers` and no identifier for
    :func:`validate_identifiers`, so this is the only check that sees it.

    A safety net, not a parser: it catches the shipped phrasing and its close
    variants, and a false match costs one retry. Labels are matched
    case-insensitively with hyphens and runs of whitespace folded to one space.

    Args:
        text: One prose field of a generation.
        within: Display labels of the budgets whose status is
            :data:`~evalshift_cli.insights.facts.BUDGET_WITHIN`.

    Returns:
        The wrongly blamed labels, each once, in ``within`` order. Empty means
        valid.
    """
    needles = {label: _normalise(label) for label in within}
    blamed: set[str] = set()
    for clause in (_normalise(c) for c in _CLAUSE_RE.split(text)):
        present = [label for label, needle in needles.items() if needle in clause]
        if not present:
            continue
        # Every held label comes out before the scan, not just the one under
        # test: "Trace-rule violations" would otherwise read as a violation
        # of whichever budget shares its clause.
        rest = clause
        for needle in needles.values():
            rest = rest.replace(needle, " ")
        if any(not _NEGATION_RE.search(rest[: m.start()]) for m in _BLAME_RE.finditer(rest)):
            blamed.update(present)
    return [label for label in within if label in blamed]


def _normalise(text: str) -> str:
    """Lower-case with hyphens and whitespace runs folded, for label matching."""
    return re.sub(r"[-\s]+", " ", text.lower())
```

4. Add `"validate_blame",` to `__all__` (after `"validate_numbers",`).

- [ ] **Step 4: Run the validator tests**

Run: `uv run pytest tests/unit/test_insight_generator.py -q -k blame`
Expected: 9 passed.

- [ ] **Step 5: Write the failing integration tests**

Append to `tests/unit/test_insight_generator.py`:

```python
# ---------------------------------------------------------------------------
# Blame in a generation: retry with the labels named, then the template
# ---------------------------------------------------------------------------

_SHIPPED_SUMMARY = (
    "The migration failed because the target model exceeded the latency ceiling of "
    "+30% and failed to meet the budget for Trace-rule violations."
)


async def test_a_fail_that_blames_a_held_budget_is_rejected_and_retried(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    sample_run["decision"] = converter_decision()
    fake_client.queue_responses(
        _payload(verdict_summary=_SHIPPED_SUMMARY),
        _payload(verdict_summary="The migration failed because the latency ceiling was exceeded."),
    )
    insight = await _generate(build_facts(**sample_run), fake_client)

    assert insight.verdict_summary.endswith("latency ceiling was exceeded.")
    assert fake_client.call_count == 2
    assert (
        "Your previous answer blamed budgets that held: Trace-rule violations."
        in fake_client.prompts[1]
    )


async def test_a_misblaming_recommendation_is_rejected(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    sample_run["decision"] = converter_decision()
    fake_client.queue_responses(
        _payload(recommendation="Address the Trace-rule violations before re-evaluating."),
        _payload(recommendation="Bring latency back under the ceiling, then re-run."),
    )
    insight = await _generate(build_facts(**sample_run), fake_client)

    assert insight.recommendation == "Bring latency back under the ceiling, then re-run."
    assert fake_client.call_count == 2


async def test_two_misblaming_generations_fall_back_to_the_template(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The template already names the breached budget, so it is the safe floor."""
    sample_run["decision"] = converter_decision()
    bad = _payload(recommendation="Address the Trace-rule violations before re-evaluating.")
    fake_client.queue_responses(bad, bad)
    insight = await _generate(build_facts(**sample_run), fake_client)

    assert insight.model == "none"
    assert insight.recommendation.startswith(
        "Do not migrate under the configured policy — the latency increase budget was breached."
    )


async def test_a_fail_may_mention_a_held_budget_without_blaming_it(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    sample_run["decision"] = converter_decision()
    fake_client.queue_responses(
        _payload(
            verdict_summary=(
                "The latency ceiling of +30% was exceeded; Trace-rule violations stayed "
                "within budget."
            )
        )
    )
    await _generate(build_facts(**sample_run), fake_client)

    assert fake_client.call_count == 1


async def test_blame_is_not_checked_outside_a_fail(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The scan is scoped to a FAIL (Global Constraints): only a FAIL assigns a cause.

    The sentence is false on a pass, but it accuses no budget of failing a
    run that did not fail; the scope is pinned here so widening it is a
    deliberate change.
    """
    fake_client.queue_responses(
        _payload(verdict_summary="PASS although the Cost increase budget was exceeded.")
    )
    await _generate(build_facts(**sample_run), fake_client)

    assert fake_client.call_count == 1


async def test_an_invented_number_and_a_misblame_are_both_named_on_retry(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    sample_run["decision"] = converter_decision()
    fake_client.queue_responses(
        _payload(verdict_summary=_SHIPPED_SUMMARY, economics_summary="Cost rose +43%."),
        _payload(),
    )
    await _generate(build_facts(**sample_run), fake_client)

    retry = fake_client.prompts[1]
    assert "+43%" in retry
    assert "blamed budgets that held: Trace-rule violations" in retry
```

- [ ] **Step 6: Run them to see them fail**

Run: `uv run pytest tests/unit/test_insight_generator.py -q -k "misblam or blames or blame_is_not or mention_a_held"`
Expected: `test_a_fail_that_blames_a_held_budget_is_rejected_and_retried`, `test_a_misblaming_recommendation_is_rejected`, `test_two_misblaming_generations_fall_back_to_the_template`, `test_an_invented_number_and_a_misblame_are_both_named_on_retry` FAIL (the bad generation ships, `call_count == 1`); the other two pass already.

- [ ] **Step 7: Wire the validator into generation**

In `src/evalshift_cli/insights/generator.py`:

1. `GenerationError`: extend the docstring's `Attributes` with

```
        misblamed: Display labels of budgets that held but were blamed for
            the verdict — fed back the same way, under their own hint.
```

   and change `__init__` to:

```python
    def __init__(
        self,
        message: str,
        *,
        tokens: list[str] | None = None,
        misblamed: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.tokens = tokens or []
        self.misblamed = misblamed or []
```

2. `generate_insight`: declare `misblamed: list[str] = []` beside `invented`, pass `misblamed=misblamed` to `build_prompt`, and in the `except GenerationError as exc:` branch add `misblamed = exc.misblamed` after `invented = exc.tokens`.

3. `build_prompt`: signature `def build_prompt(facts: Facts, *, invented: list[str] | None = None, misblamed: list[str] | None = None) -> str:` (wrap to the file's line width), add to the docstring's Args:

```
        misblamed: Budgets that held but a previous attempt blamed for the
            verdict. Named separately from ``invented``: the label itself is
            allowed in prose, the accusation is not.
```

   and after the `if invented:` block add:

```python
    if misblamed:
        sections.append(
            "Your previous answer blamed budgets that held: "
            + ", ".join(dict.fromkeys(misblamed))
            + ". Only the budgets under failed_budgets were breached; a budget "
            "marked within may be described as within policy or left out, never "
            "as a cause of the verdict or as something to address."
        )
```

4. `_build_insight`: replace the block from `banned = banned_identifiers(facts)` through the `raise GenerationError(...)` with:

```python
    banned = banned_identifiers(facts)
    offending: list[str] = []
    for field in (*prose.values(), *(t for f in findings for t in (f.title, f.detail))):
        offending.extend(validate_numbers(field, facts.allowed_numbers))
        offending.extend(t for t in validate_identifiers(field, banned) if t not in offending)
    # Only a FAIL assigns a cause, and only its verdict summary and
    # recommendation do the assigning; the advisory and economics summaries
    # describe signals, and findings describe behaviour.
    misblamed: list[str] = []
    if facts.verdict == "fail":
        held = [label for label, _, status in facts.budget_statuses if status == BUDGET_WITHIN]
        for field in (prose["verdict_summary"], prose["recommendation"]):
            misblamed.extend(t for t in validate_blame(field, held) if t not in misblamed)
    if offending or misblamed:
        problems: list[str] = []
        if offending:
            problems.append(f"tokens that may not be shown: {', '.join(offending)}")
        if misblamed:
            problems.append(f"budgets blamed although they held: {', '.join(misblamed)}")
        raise GenerationError(
            "response contained " + "; ".join(problems),
            tokens=offending,
            misblamed=misblamed,
        )
```

5. Update the module docstring's second paragraph opening `Two attempts, then` to read: `Three checks on the way out — invented figures, echoed identifiers, and on a FAIL a held budget blamed for the verdict (:func:`validate_blame`). Two attempts, then` (keep the rest of the sentence).

- [ ] **Step 8: Full gate, then commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest -q`
Expected: green.

```bash
WT=/home/lukas/repos/evalshift/evalshift-cli
git -C $WT branch --show-current   # fix/insight-budget-status
git -C $WT add src/evalshift_cli/insights/generator.py tests/unit/test_insight_generator.py
cd $WT && uv run --with pre-commit git commit -m "fix(insights): reject a narrative that blames a budget that held

On a FAIL the verdict summary and recommendation are scanned for a
'within' budget sharing a clause with a non-negated blame word. A hit is
rejected like an invented number, the retry names the labels, and two
rejections ship the templated prose, which already names the breached
budget.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Docs and changelog

**Files:**
- Modify: `DOCS.md` (Run insights, ~l.782), `llms-full.txt` (Run insights, ~l.1190), `docs/faq.md` (~l.31-35), `CHANGELOG.md` (`## [Unreleased]` → `### Fixed`)

- [ ] **Step 1: `DOCS.md`**

After the paragraph beginning `**A gate that measured nothing is never counted as a gate that passed.**` (ends `…this fires when only *some* gate did.`) and before the ```` ```yaml ```` block, insert:

```markdown
**A FAIL's cause is read off the facts, not inferred.** Every budget enters the prompt with its status beside its limit — `breached`, `within`, `measured nothing`, or over its limit but not confirmed at the sample size — together with the `failed_budgets` list (overall and slice rows, named as the decision prose names them) and, when the decision carries one, its `reason`. The instruction allows only a breached budget to be named as why the run failed or as something to fix. A narrative once explained a FAIL as "failed to meet the budget for Trace-rule violations" on a run where that budget held and only the latency ceiling was breached: the prompt had listed limits without results, so the model guessed. On a `fail` the verdict summary and recommendation are also scanned for a held budget that shares a clause with a blame word (breached, exceeded, failed, violated, address…) and is not negated; a match rejects the generation like an invented number, the retry names the misblamed budgets, and two rejections ship the templated prose, which names the breached budget itself. The scan is a safety net, not a parser: it catches the shipped phrasing and its close variants, and a false match costs one retry. A narrative is cached per run (see below), so an existing run keeps its old prose until its `insights.json` is deleted.
```

- [ ] **Step 2: `llms-full.txt`**

After the `- A BLIND GATE is not a passed gate.` bullet (ends `…this one fires when only some gate died.`) insert, keeping the file's two-space continuation indent:

```
- A FAIL's CAUSE is a fact, not an inference. Each `budget "<label>": limit <x>, <status>`
  line carries status breached | within | measured nothing | over its limit but not
  confirmed at this sample size; `failed_budgets: <labels>` lists the breached rows
  (overall and slice, as decision prose names them); `verdict_reason: <text>` carries
  MigrationDecision.reason when set (its figures join the allow-list). Instruction: only a
  "breached" budget may be named as why the run failed or as something to fix; a "within"
  budget is never breached/exceeded/failed/violated/to be addressed; verdict_summary must
  agree with verdict_reason. On verdict fail, verdict_summary and recommendation are
  scanned (generator.validate_blame): a "within" label sharing a clause (split on sentence
  punctuation, while/but/whereas/although/though/however, ", and" / ", but") with a
  non-negated blame word (breach*, exceed*, fail*, violat*, address*, miss*, unmet, not
  met, over/above the/its, broke/broken) rejects the generation; the retry names the
  misblamed labels; 2 rejections -> templated prose. Heuristic by design: it catches the
  shipped phrasing and close variants. Cached narratives do not regenerate: delete
  insights.json to rewrite an existing run's prose.
```

- [ ] **Step 3: `docs/faq.md`**

In the answer to `## What is the summary at the top of the report, and can I trust its numbers?`, after the sentence ending `…or it is not there at all.` add a new paragraph:

```markdown
The same goes for the cause of a FAIL. The model is told which budgets were
breached and which held, and on a failed run a summary or recommendation that
blames a budget that held is rejected and retried the same way, then replaced
by the templated prose.
```

- [ ] **Step 4: `CHANGELOG.md`**

Under `## [Unreleased]` → `### Fixed`, after the last existing bullet, add:

```markdown
- The run-insights narrative no longer guesses why a run failed. The prompt listed each
  budget's limit but not its result, so a FAIL on the latency ceiling alone was explained
  as "failed to meet the budget for Trace-rule violations" — a budget that held. Every
  budget now enters the prompt with its status, the breached ones are listed as
  `failed_budgets`, the decision's `reason` travels with them, and on a FAIL a summary or
  recommendation that blames a budget that held is rejected and retried, then replaced by
  the templated prose. Narratives are cached: delete a run's `insights.json` and re-run
  `report` to rewrite an existing one.
```

- [ ] **Step 5: Check the doc tripwires, then commit**

Run: `uv run pytest -q -k "docs or llms or currency" && uv run --with pre-commit pre-commit run --all-files`
Expected: green.

```bash
WT=/home/lukas/repos/evalshift/evalshift-cli
git -C $WT branch --show-current   # fix/insight-budget-status
git -C $WT add DOCS.md llms-full.txt docs/faq.md CHANGELOG.md
cd $WT && uv run --with pre-commit git commit -m "docs(insights): a FAIL's cause is a fact the narrative is given

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Verify on the converter run, then PR

**Files:** none changed (a backup in the scratchpad).

- [ ] **Step 1: Regenerate the shipped narrative with the branch build**

```bash
WT=/home/lukas/repos/evalshift/evalshift-cli
RUN=/home/lukas/repos/testApp/.evalshift/runs/r_20261012_converter_d418e5
cp "$RUN/insights.json" "$SCRATCH/converter-insights.before.json"
rm "$RUN/insights.json"            # the cache key does not change with the CLI
cd /home/lukas/repos/testApp && uv run --project $WT evalshift report r_20261012_converter_d418e5
```

`$SCRATCH` is the session scratchpad. The global `evalshift` on PATH is the PyPI 1.4.1 wheel, so it **must** be `uv run --project $WT`. The narrative model is `gemini-3.1-flash-lite-preview`; if `report` warns `skipping insights: no API key`, ask the user to run the last line with `!` in their shell.

Expected in the new `insights.json`: `verdict_summary` and `recommendation` attribute the FAIL to latency only; "Trace-rule violations" appears nowhere as a cause (it may appear as "within budget"). If `insight.model == "none"`, the generation was rejected twice and the templated `recommendation` reads `Do not migrate under the configured policy — the latency increase budget was breached. …` — also a pass for this check; note it in the PR.

- [ ] **Step 2: Full CI mirror**

Run: `cd $WT && make ci`
Expected: ruff, format, mypy, pytest all green; coverage ≥ 90.

- [ ] **Step 3: Push and open the PR** (only after the user approves pushing)

```bash
cd $WT && git add -f docs/superpowers/plans/2026-10-12-insight-budget-status.md
uv run --with pre-commit git commit -m "docs(plan): insight budget status

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
uv run --with pre-commit git push -u origin fix/insight-budget-status
gh -R evalshift/evalshift-cli pr create --base main --head fix/insight-budget-status \
  --title "fix(insights): the narrative names the budget that failed the run" \
  --body-file "$SCRATCH/pr-body.md"
```

The PR body follows `.github/PULL_REQUEST_TEMPLATE.md`: what changes (status lines, `failed_budgets`, `verdict_reason`, the instruction rule, `validate_blame` + retry + template floor), why (the converter run's before/after narrative, quoted), the test plan with the gate ticked, a note that the scan is heuristic and the prompt change is the primary fix, the cache caveat (delete `insights.json` to regenerate), and it ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Release as 1.4.2 together with PR #50 via a separate `chore(release): 1.4.2` PR after merge.
