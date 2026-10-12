"""Tests for the generated half of the insights package.

Never hits a real model: the generator's whole reason for existing beside two
pure modules is that the prompt is built from facts and the response is
validated back against them, so a queued-response double exercises every path.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from evalshift_cli.analysis.policy import (
    BlockingRegression,
    BudgetResult,
    FailureCategoryCount,
    SliceDecision,
)
from evalshift_cli.insights.facts import Facts, build_facts
from evalshift_cli.insights.generator import generate_insight, validate_blame, validate_numbers
from evalshift_cli.models.client import ModelClient
from tests.unit.insights_factories import (
    SEVERITY_REASON,
    FakeModelClient,
    budgets,
    converter_decision,
    decision,
    unmeasured_budget,
    unmeasured_comparison,
)


@pytest.fixture
def fake_client() -> FakeModelClient:
    return FakeModelClient()


def _generate(facts: Facts, client: FakeModelClient) -> Any:
    return generate_insight(facts, model="m", client=cast(ModelClient, client))


def _payload(**overrides: Any) -> str:
    """A valid response body, overridable field by field."""
    body: dict[str, Any] = {
        "verdict_summary": "PASS under the configured policy.",
        "advisory_summary": "Semantic similarity fell on every prompt.",
        "economics_summary": "Cost rose +102%.",
        "recommendation": "Safe to migrate under the configured policy.",
        "findings": [],
    }
    body.update(overrides)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# The validator
# ---------------------------------------------------------------------------


def test_validate_numbers_accepts_copied_figures() -> None:
    allowed = frozenset({"+102%", "$0.0101", "21", "102"})
    assert validate_numbers("Cost rose +102% to $0.0101 across 21 calls.", allowed) == []


def test_validate_numbers_flags_an_invented_figure() -> None:
    allowed = frozenset({"+102%", "21"})
    assert validate_numbers("Cost rose +43% across 21 calls.", allowed) == ["+43%"]


def test_validate_numbers_ignores_prose_without_digits() -> None:
    assert validate_numbers("Latency is not comparable on a cached replay.", frozenset()) == []


def test_validate_numbers_accepts_a_bare_numeral_from_a_rendered_figure() -> None:
    """``+102%`` was supplied; writing "102" out of it is copying, not deriving."""
    assert validate_numbers("A 102 percent increase.", frozenset({"102"})) == []


def test_validate_numbers_accepts_a_thousands_separator() -> None:
    """``facts`` admits ``20695`` for a rendered ``20,695`` — both must pass."""
    assert validate_numbers("20,695 input tokens.", frozenset({"20695"})) == []


def test_a_percentage_is_not_admitted_by_the_counting_integers() -> None:
    """The permit-list holds 0..n so "15 of 21" is writable.

    Tolerating a unit-stripped match would make ``+7%`` legal on any run with
    seven or more examples — exactly the mistyped figure this rejects.
    """
    assert validate_numbers("Cost rose +7%.", frozenset({"7", "21", "+102%"})) == ["+7%"]


def test_an_ascii_hyphen_is_read_as_the_rendered_negative() -> None:
    """The facts render ``−2.51`` (U+2212) and also admit the bare ``2.51``."""
    assert validate_numbers("Effect size -2.51.", frozenset({"−2.51", "2.51"})) == []


def test_validate_numbers_matches_a_unicode_minus() -> None:
    """The facts render negatives with U+2212; a hyphen-only regex would miss it."""
    assert validate_numbers("Effect size −2.51.", frozenset({"+2.51"})) == ["−2.51"]


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


async def test_generator_returns_the_narrative_with_provenance(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """``model`` and ``generated_at`` mark the prose as machine-written."""
    fake_client.queue_responses(
        _payload(findings=[{"kind": "negative", "title": "Drops actions", "detail": "d"}])
    )
    insight = await _generate(sample_facts, fake_client)

    assert insight.model == "m"
    assert insight.generated_at.tzinfo is not None
    assert insight.verdict_summary == "PASS under the configured policy."
    assert [finding.kind for finding in insight.findings] == ["negative"]
    assert fake_client.call_count == 1


async def test_the_prompt_carries_the_facts_and_the_regression_samples(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """A finding describes behavior, which needs the outputs — not just figures."""
    fake_client.queue_responses(_payload())
    await _generate(sample_facts, fake_client)

    prompt = fake_client.prompts[0]
    assert "+102%" in prompt
    assert "$0.0101" in prompt
    assert sample_facts.regression_samples[0].example_id in prompt
    assert sample_facts.regression_samples[0].target_output[:40] in prompt


async def test_generator_retries_once_on_an_invented_number(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(
        _payload(economics_summary="Cost rose +43%."),  # invalid
        _payload(economics_summary="Cost rose +102%."),  # valid
    )
    insight = await _generate(sample_facts, fake_client)

    assert insight.economics_summary == "Cost rose +102%."
    assert fake_client.call_count == 2


async def test_the_retry_names_the_offending_token(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """Handing the token back is what makes the second attempt worth paying for."""
    fake_client.queue_responses(
        _payload(economics_summary="Cost rose +43%."),
        _payload(economics_summary="Cost rose +102%."),
    )
    await _generate(sample_facts, fake_client)

    assert "+43%" in fake_client.prompts[1]


async def test_an_echoed_facts_key_is_rejected_and_retried(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """The shipped run wrote "a cost_delta_pct of −61.7%" into the report.

    The keys exist to address figures inside the prompt; a reader must never
    see one. Like an invented number, an echoed key is named in the retry.
    """
    fake_client.queue_responses(
        _payload(economics_summary="Cost fell, a cost_delta_pct of +102%."),
        _payload(economics_summary="Cost rose +102%."),
    )
    insight = await _generate(sample_facts, fake_client)

    assert insight.economics_summary == "Cost rose +102%."
    assert fake_client.call_count == 2
    assert "cost_delta_pct" in fake_client.prompts[1]


async def test_a_machine_failure_category_in_a_finding_is_rejected(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """SCREAMING_CASE category labels are grouping keys, not prose."""
    fake_client.queue_responses(
        _payload(
            findings=[
                {"kind": "negative", "title": "Drift", "detail": "Shows TOOL_SELECTION_DRIFT."}
            ]
        ),
        _payload(),
    )
    insight = await _generate(sample_facts, fake_client)

    assert insight.findings == []
    assert fake_client.call_count == 2


async def test_a_config_budget_name_in_prose_is_rejected(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(
        _payload(recommendation="Loosen max_tool_divergence before migrating."),
        _payload(),
    )
    insight = await _generate(sample_facts, fake_client)

    assert "max_tool_divergence" not in insight.recommendation
    assert fake_client.call_count == 2


async def test_generator_falls_back_after_two_bad_generations(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(
        _payload(economics_summary="Cost rose +43%."),
        _payload(economics_summary="Cost rose +7%."),
    )
    insight = await _generate(sample_facts, fake_client)

    assert insight.model == "none"
    assert "$0.0101" in insight.economics_summary
    assert fake_client.call_count == 2


async def test_generator_falls_back_on_malformed_json(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses("not json at all", "still not json")
    insight = await _generate(sample_facts, fake_client)
    assert insight.model == "none"


async def test_generator_reads_json_out_of_a_fenced_response(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """Models fence JSON even when told not to; that is not a failed generation."""
    fake_client.queue_responses(f"```json\n{_payload()}\n```")
    insight = await _generate(sample_facts, fake_client)

    assert insight.model == "m"
    assert fake_client.call_count == 1


async def test_generator_enforces_the_server_length_and_count_caps(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """A model that overruns a cap must not produce a bundle the server 400s."""
    fake_client.queue_responses(
        _payload(
            verdict_summary="x" * 5000,
            findings=[{"kind": "negative", "title": f"t{i}", "detail": "d"} for i in range(14)],
        ),
    )
    insight = await _generate(sample_facts, fake_client)

    assert len(insight.verdict_summary) <= 2000
    assert len(insight.findings) <= 10
    assert all(len(finding.title) <= 200 for finding in insight.findings)
    assert all(len(finding.detail) <= 2000 for finding in insight.findings)
    assert insight.model == "m"
    assert fake_client.call_count == 1


async def test_generator_falls_back_on_an_empty_prose_field(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(_payload(recommendation=""), _payload(recommendation=""))
    insight = await _generate(sample_facts, fake_client)
    assert insight.model == "none"


async def test_generator_falls_back_on_a_missing_prose_field(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """A key the server requires is as fatal as an empty one, and as recoverable."""
    body = json.loads(_payload())
    del body["advisory_summary"]
    fake_client.queue_responses(json.dumps(body), json.dumps(body))
    insight = await _generate(sample_facts, fake_client)
    assert insight.model == "none"


async def test_generator_rejects_an_unknown_finding_kind(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(
        _payload(findings=[{"kind": "catastrophic", "title": "t", "detail": "d"}]),
        _payload(findings=[{"kind": "negative", "title": "t", "detail": "d"}]),
    )
    insight = await _generate(sample_facts, fake_client)
    assert [finding.kind for finding in insight.findings] == ["negative"]


async def test_an_unknown_kind_is_dropped_when_other_findings_survive(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """One bad icon class is not worth paying for a second generation."""
    fake_client.queue_responses(
        _payload(
            findings=[
                {"kind": "catastrophic", "title": "bad", "detail": "d"},
                {"kind": "warning", "title": "good", "detail": "d"},
            ]
        ),
    )
    insight = await _generate(sample_facts, fake_client)

    assert [finding.title for finding in insight.findings] == ["good"]
    assert fake_client.call_count == 1


async def test_no_findings_at_all_is_a_valid_generation(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """ "Return fewer findings than pad" only works if zero is accepted."""
    fake_client.queue_responses(_payload(findings=[]))
    insight = await _generate(sample_facts, fake_client)

    assert insight.findings == []
    assert insight.model == "m"
    assert fake_client.call_count == 1


async def test_an_invented_number_inside_a_finding_is_caught(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """Findings are prose on the same page; the permit-list covers them too."""
    fake_client.queue_responses(
        _payload(findings=[{"kind": "negative", "title": "t", "detail": "Seen in 97 examples."}]),
        _payload(findings=[{"kind": "negative", "title": "t", "detail": "Seen in 15 examples."}]),
    )
    insight = await _generate(sample_facts, fake_client)

    assert insight.findings[0].detail == "Seen in 15 examples."
    assert fake_client.call_count == 2


async def test_a_json_array_response_is_not_a_generation(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """Valid JSON of the wrong shape fails the same way malformed JSON does."""
    fake_client.queue_responses("[1, 2, 3]", "[1, 2, 3]")
    insight = await _generate(sample_facts, fake_client)
    assert insight.model == "none"


async def test_a_braced_but_broken_object_is_not_a_generation(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """The brace scan finds something; it still has to parse."""
    fake_client.queue_responses('here you go: {"verdict_summary": }', 'again: {"x": }')
    insight = await _generate(sample_facts, fake_client)
    assert insight.model == "none"


async def test_findings_of_the_wrong_type_are_rejected(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(_payload(findings="a string"), _payload(findings="a string"))
    insight = await _generate(sample_facts, fake_client)
    assert insight.model == "none"


async def test_a_non_object_finding_is_dropped(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    fake_client.queue_responses(
        _payload(findings=["just a string", {"kind": "positive", "title": "t", "detail": "d"}]),
    )
    insight = await _generate(sample_facts, fake_client)

    assert [finding.kind for finding in insight.findings] == ["positive"]
    assert fake_client.call_count == 1


async def test_a_finding_with_no_detail_is_dropped(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """``detail`` is ``min_length=1`` server-side, same as ``title``."""
    fake_client.queue_responses(
        _payload(findings=[{"kind": "negative", "title": "t", "detail": "   "}]),
        _payload(findings=[{"kind": "negative", "title": "t", "detail": "d"}]),
    )
    insight = await _generate(sample_facts, fake_client)

    assert [finding.detail for finding in insight.findings] == ["d"]


async def test_a_clean_run_prompts_against_inventing_findings(
    fake_client: FakeModelClient, passing_run: dict[str, Any]
) -> None:
    """No example regressed, so there is no behavior to describe."""
    facts = build_facts(**passing_run)
    fake_client.queue_responses(_payload())
    await _generate(facts, fake_client)

    assert "SAMPLED REGRESSIONS: none" in fake_client.prompts[0]


async def test_the_prompt_names_the_blocking_evaluators_and_failure_categories(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """A narrative that cannot name what blocked the run is not worth writing."""
    sample_run["decision"] = decision(
        verdict="fail",
        blocking=[
            BlockingRegression(
                prompt_id="replay",
                evaluator_name="semantic.cosine",
                slice_name="all",
                severity="high",
                delta_avg_score=-0.031,
                effect_size=-2.51,
            ),
        ],
        categories=[FailureCategoryCount(category="missing_field", count=15)],
    )
    fake_client.queue_responses(_payload())
    await _generate(build_facts(**sample_run), fake_client)

    prompt = fake_client.prompts[0]
    assert "blocking_evaluators: semantic.cosine" in prompt
    # Categories and budgets reach the model already humanised, so the tokens
    # it can copy into prose are the ones a reader may see.
    assert 'failure_category "Missing field": 15' in prompt
    assert 'budget "Cost increase": limit +200%, within' in prompt
    assert "failure_category.missing_field" not in prompt
    assert "max_cost_increase" not in prompt


async def test_a_response_with_no_findings_key_is_valid(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """An omitted list reads the same as an empty one — no findings to show."""
    body = json.loads(_payload())
    del body["findings"]
    fake_client.queue_responses(json.dumps(body))
    insight = await _generate(sample_facts, fake_client)

    assert insight.findings == []
    assert insight.model == "m"


async def test_an_unmeasured_run_tells_the_model_not_to_claim_equivalence(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The permit-list stops a *figure* being invented; not a claim in words.

    "The target model achieved a 100% equivalence rate" is rejected because of
    its number. "No behavioral differences were observed" carries no digits at
    all and reads exactly the same to the engineer deciding to migrate, so the
    absence of a denominator has to be stated in the prompt as well.
    """
    sample_run["decision"] = decision(verdict="inconclusive", n_records=0, equivalent_rate=0.0)
    fake_client.queue_responses(_payload())
    await _generate(build_facts(**sample_run), fake_client)

    prompt = fake_client.prompts[0]
    assert "rates_basis: " in prompt
    assert "equivalence_rate_pct: not measured" in prompt
    assert "equivalent" in prompt.lower().split("facts (")[0]


async def test_a_blind_gate_is_named_in_the_prompt(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The shipped "passed all 7 of 7 budgets" was copied, not invented.

    The model was handed a clean sweep and restated it. Nothing in the FACTS
    block said one of those budgets had been counted over an empty sample, or
    that two blocking evaluators had produced no comparable row at all.
    """
    sample_run["decision"] = decision(
        verdict="conditional_pass",
        budget_results=[*budgets(), unmeasured_budget()],
    )
    sample_run["comparisons"] = [
        unmeasured_comparison("llm_judge.equivalence"),
        unmeasured_comparison("semantic.cosine"),
    ]
    fake_client.queue_responses(_payload())
    await _generate(build_facts(**sample_run), fake_client)

    prompt = fake_client.prompts[0]
    assert "budgets_passed: 6" in prompt
    assert "budgets_passed: 7" not in prompt
    assert "\nunmeasured_budgets: Tool-selection divergence\n" in prompt
    assert "\nunmeasured_evaluators: llm_judge.equivalence, semantic.cosine\n" in prompt
    assert "\ncoverage_basis: " in prompt


async def test_the_instruction_forbids_calling_a_blind_gate_a_pass(
    fake_client: FakeModelClient, sample_facts: Facts
) -> None:
    """Same reason as the unmeasured rates: a claim in words carries no digits.

    "All hard constraints are met" is what the shipped run recommended. It has
    no numeric token for the validator to reject, so the rule has to be in the
    instruction rather than in the permit-list.
    """
    fake_client.queue_responses(_payload())
    await _generate(sample_facts, fake_client)

    instruction = fake_client.prompts[0].split("FACTS (")[0]
    assert "unmeasured_budgets" in instruction
    assert "unmeasured_evaluators" in instruction


async def test_a_fully_measured_run_carries_no_coverage_facts(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The lines appear only where there is something blind to report."""
    fake_client.queue_responses(_payload())
    await _generate(build_facts(**sample_run), fake_client)

    prompt = fake_client.prompts[0]
    assert "budgets_unmeasured: 0" in prompt
    assert "\nunmeasured_budgets: " not in prompt
    assert "\nunmeasured_evaluators: " not in prompt
    assert "\ncoverage_basis: " not in prompt


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
    text = (
        "Cost increase was exceeded and Trace-rule violations were violated; Cost increase failed."
    )
    assert validate_blame(text, _HELD) == ["Trace-rule violations", "Cost increase"]


def test_a_breached_slice_label_does_not_accuse_its_held_overall_row() -> None:
    """The slice label contains the overall one; the longer match owns its span."""
    held = ["Cost increase"]
    others = ["'checkout' slice cost increase"]
    assert (
        validate_blame(
            "The 'checkout' slice cost increase budget was breached.", held, others=others
        )
        == []
    )
    assert (
        validate_blame(
            "The run failed because the 'checkout' slice cost increase exceeded its +10% limit.",
            held,
            others=others,
        )
        == []
    )


def test_a_breached_label_with_a_blame_word_does_not_accuse_its_neighbour() -> None:
    """ "violations" is part of a label, not an accusation, once the label is masked.

    The bare "and" keeps both labels in one clause, so the masking alone saves
    it — without ``others`` the same sentence accuses "Cost increase".
    """
    text = "Trace-rule violations rose to 3 and cost increase stayed inside its limit."
    assert validate_blame(text, ["Cost increase"], others=["Trace-rule violations"]) == []
    assert validate_blame(text, ["Cost increase"]) == ["Cost increase"]


def test_a_slice_label_matches_whatever_quotes_the_model_used() -> None:
    """The model may drop the slice's quotes or swap them; the slice still owns its span."""
    others = ["'checkout' slice cost increase"]
    for text in (
        "The checkout slice cost increase budget was breached.",
        "The \u2018checkout\u2019 slice cost increase budget was breached.",
        'The "checkout" slice cost increase budget was breached.',
    ):
        assert validate_blame(text, ["Cost increase"], others=others) == [], text


def test_the_retry_hints_own_phrasing_is_not_blame() -> None:
    """The hint invites "within policy"; a reply phrased that way must pass."""
    text = (
        "The run failed because latency exceeded its ceiling, with Cost increase and "
        "Trace-rule violations within policy."
    )
    assert validate_blame(text, _HELD) == []


def test_a_dash_or_however_separates_the_blame() -> None:
    assert validate_blame("Latency breached its ceiling — Cost increase held.", _HELD) == []
    assert validate_blame("Latency was exceeded, however, Cost increase held.", _HELD) == []


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


async def test_a_slice_budget_fail_may_name_the_breached_slice_budget(
    fake_client: FakeModelClient, sample_run: dict[str, Any]
) -> None:
    """The overall "Cost increase" held; its 'checkout' slice row broke.

    The slice label contains the held overall one, so a substring match would
    accuse the overall row in every run a slice budget failed.
    """
    base = decision(verdict="fail")
    slice_row = BudgetResult(
        name="max_cost_increase", observed=1.02, allowed=0.10, passed=False, scope="checkout"
    )
    sample_run["decision"] = replace(
        base,
        slices={
            "checkout": SliceDecision(
                name="checkout", verdict="fail", metrics=base.overall, budget_results=[slice_row]
            )
        },
    )
    facts = build_facts(**sample_run)
    assert facts.failed_budgets == ["'checkout' slice cost increase"]
    fake_client.queue_responses(
        _payload(
            verdict_summary=(
                "The run failed because the 'checkout' slice cost increase exceeded its +10% limit."
            ),
            recommendation="Bring the 'checkout' slice cost increase under its limit.",
        )
    )
    await _generate(facts, fake_client)

    assert fake_client.call_count == 1
