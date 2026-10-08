"""Pin the hosted bundle's payload shape.

The server model is the contract; these tests assert what leaves the CLI so a
field rename cannot be discovered as a 400 at finalize, after the bundle has
already been uploaded.
"""

from __future__ import annotations

import gzip
import json
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.runner.checkpoint import append_call
from evalshift_cli.runner.models import Call
from evalshift_cli.suite.models import Suite
from tests.conftest import RunFixture


def _load(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, dict)
    return data


def test_bundle_has_no_version_or_report_html(built_bundle_path: Path) -> None:
    bundle = _load(built_bundle_path)
    assert "bundle_version" not in bundle
    assert "schema_version" not in bundle
    assert "report_html" not in bundle
    manifest = bundle["manifest"]
    assert isinstance(manifest, dict)
    assert "bundle_version" not in manifest
    assert "size_bytes" not in manifest


def test_manifest_carries_the_suite_path(built_bundle_path: Path) -> None:
    """The hosted methodology tab shows which suite file the run replayed."""
    manifest = _load(built_bundle_path)["manifest"]
    assert isinstance(manifest, dict)
    assert str(manifest["suite_path"]).endswith(".jsonl")


def test_report_html_still_lands_on_disk(run_fixture: RunFixture) -> None:
    """Dropping it from the bundle must not stop the local report being written."""
    run_fixture.build()
    assert (run_fixture.run_dir / "report.html").exists()


def test_bundle_bytes_are_deterministic(tmp_path: Path, run_fixture: RunFixture) -> None:
    """No fixed-point loop means the same run always compresses identically."""
    first = run_fixture.build(output=tmp_path / "a.json.gz")
    second = run_fixture.build(output=tmp_path / "b.json.gz")
    assert first.path.read_bytes() == second.path.read_bytes()


def test_report_html_is_not_required_to_build(run_fixture: RunFixture) -> None:
    (run_fixture.run_dir / "report.html").unlink()
    result = run_fixture.build()
    assert result.size_bytes > 0


def test_size_bytes_reports_the_compressed_file_size(run_fixture: RunFixture) -> None:
    """``size_bytes`` is request metadata now, not a field inside the payload."""
    result = run_fixture.build()
    assert result.size_bytes == result.path.stat().st_size


def _examples(bundle: dict[str, object]) -> list[dict[str, Any]]:
    examples = bundle["examples"]
    assert isinstance(examples, list)
    return examples


def test_example_rows_carry_split_ids_and_hoisted_metrics(built_bundle_path: Path) -> None:
    row = _examples(_load(built_bundle_path))[0]

    assert row["prompt_id"] == "greet"
    assert row["example_id"] == "ex1"
    assert row["tags"] == ["security"]
    assert "metrics" not in row
    assert "severity" not in row
    assert set(row["scores"][0]) == {
        "evaluator_name",
        "kind",
        "source_score",
        "target_score",
        "delta",
        "error",
        "explanation",
        "violations",
    }
    for key in (
        "worst_delta_score",
        "delta_cost_usd",
        "delta_latency_ms",
        "latency_comparable",
        "truncated",
        "target_empty_output",
        "tool_match",
        "turn_index",
    ):
        assert key in row


def test_bundle_carries_economics_and_methodology(built_bundle_path: Path) -> None:
    bundle = _load(built_bundle_path)
    economics = bundle["economics"]
    assert isinstance(economics, dict)
    assert set(economics) == {"source", "target"}
    # Run-level, not per-prompt: both example pairs are counted.
    assert economics["source"]["calls"] == 2
    assert economics["source"]["cached_calls"] == 1
    assert economics["source"]["total_input_tokens"] == 22
    notes = bundle["methodology_notes"]
    assert isinstance(notes, list)
    assert any("Wilcoxon" in note for note in notes)


def test_latency_delta_is_zero_and_flagged_when_cached(built_bundle_path: Path) -> None:
    rows = _examples(_load(built_bundle_path))
    cached = [row for row in rows if not row["latency_comparable"]]
    assert cached, "fixture should contain at least one cached pair"
    assert all(row["delta_latency_ms"] == 0 for row in cached)
    live = [row for row in rows if row["latency_comparable"]]
    assert [row["delta_latency_ms"] for row in live] == [10]


def test_target_empty_output_is_flagged(built_bundle_path: Path) -> None:
    """The cached target returns no text despite spending output tokens."""
    rows = {row["example_id"]: row for row in _examples(_load(built_bundle_path))}
    assert rows["ex2"]["target_empty_output"] is True
    assert rows["ex1"]["target_empty_output"] is False


def test_worst_delta_score_is_the_most_negative_evaluator_delta(
    built_bundle_path: Path,
) -> None:
    rows = {row["example_id"]: row for row in _examples(_load(built_bundle_path))}
    assert rows["ex2"]["worst_delta_score"] == -0.25
    assert rows["ex1"]["worst_delta_score"] == 0.0


def test_aggregate_still_rolls_up_cost_and_latency_after_hoisting(
    built_bundle_path: Path,
) -> None:
    """Hoisting the metrics out of ``metrics`` must not empty the aggregate.

    The server requires all nine figures as non-null numbers, so a reader that
    still looked inside a deleted ``metrics`` dict would ship a valid bundle
    full of zeros — wrong in a way schema validation cannot catch.
    """
    aggregate = _load(built_bundle_path)["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["cost_usd_source"] == pytest.approx(0.04)
    assert aggregate["cost_usd_target"] == pytest.approx(0.07)
    assert aggregate["cost_usd_delta"] == pytest.approx(0.03)
    assert aggregate["latency_ms_source_p95"] == 100.0
    assert aggregate["latency_ms_target_p95"] == 110.0


def test_bundle_carries_insights_when_present(built_bundle_path: Path) -> None:
    insights = _load(built_bundle_path)["insights"]
    assert isinstance(insights, dict)
    assert insights["verdict_summary"]
    assert insights["model"]


def test_bundle_insights_are_null_when_absent(run_fixture: RunFixture) -> None:
    (run_fixture.run_dir / "insights.json").unlink(missing_ok=True)
    result = run_fixture.build()
    assert _load(result.path)["insights"] is None


def test_bundle_insights_carry_no_cache_metadata(built_bundle_path: Path) -> None:
    """``Insights`` is ``extra="forbid"`` server-side — a stray key is a 400."""
    insights = _load(built_bundle_path)["insights"]
    assert isinstance(insights, dict)
    assert set(insights) == {
        "model",
        "generated_at",
        "verdict_summary",
        "advisory_summary",
        "economics_summary",
        "findings",
        "recommendation",
    }


def test_an_unreadable_insights_file_is_treated_as_absent(run_fixture: RunFixture) -> None:
    """A hand-edited cache must not be able to fail a push."""
    (run_fixture.run_dir / "insights.json").write_text("{not json", encoding="utf-8")
    result = run_fixture.build()
    assert _load(result.path)["insights"] is None


def test_built_bundle_validates_against_the_vendored_schema(built_bundle_path: Path) -> None:
    """The end-to-end guard: what we build is what the server accepts.

    ``build_bundle`` validates before writing, so this can only fail if that gate
    is removed — which is exactly the regression worth catching.
    """
    schema = json.loads(
        resources.files("evalshift_cli.hosted")
        .joinpath("bundle_manifest.schema.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator(schema["bundle"]).validate(_load(built_bundle_path))


def _vendored_schema() -> dict[str, Any]:
    schema = json.loads(
        resources.files("evalshift_cli.hosted")
        .joinpath("bundle_manifest.schema.json")
        .read_text(encoding="utf-8")
    )
    assert isinstance(schema, dict)
    return schema


def _decision(bundle: dict[str, object]) -> dict[str, Any]:
    decision = bundle["decision"]
    assert isinstance(decision, dict)
    return decision


def _with_migration_policy(run_fixture: RunFixture) -> Path:
    """Re-point the fixture run at a config that declares a policy.

    ``write_project_files`` deliberately ships none, so the shared bundle is an
    ``inconclusive_decision`` with no budgets at all — nothing to assert a
    denominator on.

    Includes a ``checkout`` slice override so a schema-validated bundle
    actually exercises ``SliceMigrationPolicy`` — itself
    ``additionalProperties: false`` server-side — end to end, not just the
    top-level ``MigrationPolicy``. The override sets a non-default value so
    the dumped slice is distinguishable from an all-``null`` one.
    """
    config = run_fixture.root / "evalshift_policy.yaml"
    config.write_text(
        run_fixture.config.read_text(encoding="utf-8").rstrip()
        + "\n        migration_policy:\n"
        + "          max_overall_regression_rate: 0.10\n"
        + "          slices:\n"
        + "            checkout:\n"
        + "              max_overall_regression_rate: 0.05\n",
        encoding="utf-8",
    )
    return config


def test_the_vendored_schema_matches_the_server_export() -> None:
    """The CLI's copy is a copy — the server owns the spec.

    Skipped when the server repo is not checked out beside this one; in that
    layout ``make check-schema`` is the same assertion with a clearer message.
    """
    server_schema = (
        Path(__file__).resolve().parents[3]
        / "evalshift-server"
        / "schemas"
        / "bundle_manifest.schema.json"
    )
    if not server_schema.exists():
        pytest.skip("evalshift-server is not checked out beside this repo")
    assert json.loads(server_schema.read_text(encoding="utf-8")) == _vendored_schema()


def test_every_emitted_budget_carries_an_integer_denominator(run_fixture: RunFixture) -> None:
    """``null`` means "no sample size reported", and the CLI always has one.

    A bundle this CLI writes must never send the governed gate back to its
    pre-denominator fallback, so every budget — in every scope — reports a
    count, even when that count is ``0``. ``build_bundle`` validates against
    the vendored schema before writing, so this also proves the emitted field
    survives an ``additionalProperties: false`` object.
    """
    path = run_fixture.build(config_path=_with_migration_policy(run_fixture)).path
    decision = _decision(_load(path))
    scopes = [decision["budget_results"]]
    scopes.extend(s["budget_results"] for s in decision["slices"].values())
    emitted = [budget for scope in scopes for budget in scope]
    assert emitted, "the fixture run should evaluate budgets"
    for budget in emitted:
        assert isinstance(budget["denominator"], int), budget["name"]
        assert budget["denominator"] >= 0, budget["name"]


_RESOLVED_POLICY_KEYS = {
    "max_overall_regression_rate",
    "max_critical_regressions",
    "min_equivalence_rate",
    "max_tool_argument_drift",
    "max_tool_divergence",
    "max_invariant_violations",
    "tool_argument_drift_floor",
    "max_cost_increase",
    "max_latency_increase",
    "fail_on_dropped_params",
    "slices",
}

# The nine per-slice ratio/count budgets: the ten top-level fields minus
# ``fail_on_dropped_params`` (not overridable per slice — see
# ``MigrationPolicy.fail_on_dropped_params``'s docstring) and minus ``slices``
# itself (a slice cannot nest another slice).
_RESOLVED_SLICE_POLICY_KEYS = _RESOLVED_POLICY_KEYS - {"fail_on_dropped_params", "slices"}


def test_a_bundle_with_a_configured_policy_carries_it_on_the_decision(
    run_fixture: RunFixture,
) -> None:
    """The hosted gate reads ``decision.policy``, not a separately-edited web policy.

    ``evaluate_migration_policy`` stamps the resolved policy on the decision
    (see ``analysis/policy.py``); this proves it survives all the way through
    ``build_bundle`` — ``decision.to_dict()`` — and the vendored-schema
    validation ``build_bundle`` runs before writing. The fixture configures a
    ``checkout`` slice override, so this also exercises the nested
    ``SliceMigrationPolicy`` `$def` — itself ``additionalProperties: false``
    server-side — not just the top-level ``MigrationPolicy``.
    """
    path = run_fixture.build(config_path=_with_migration_policy(run_fixture)).path
    policy = _decision(_load(path))["policy"]
    assert isinstance(policy, dict)
    assert set(policy) == _RESOLVED_POLICY_KEYS
    assert policy["max_overall_regression_rate"] == 0.10
    slices = policy["slices"]
    assert isinstance(slices, dict)
    checkout = slices["checkout"]
    assert set(checkout) == _RESOLVED_SLICE_POLICY_KEYS
    assert checkout["max_overall_regression_rate"] == 0.05
    Draft202012Validator(_vendored_schema()["bundle"]).validate(_load(path))


def test_a_bundle_with_no_configured_policy_has_no_decision_policy(
    built_bundle_path: Path,
) -> None:
    """``write_project_files`` ships no ``migration_policy``: the run is an
    ``inconclusive_decision``, which resolves nothing to stamp."""
    assert _decision(_load(built_bundle_path))["policy"] is None


def test_a_bundle_without_denominators_still_validates(run_fixture: RunFixture) -> None:
    """Backwards compatibility, from the other side.

    Bundles written before this field existed omit the key entirely.
    ``denominator`` is optional in the schema precisely so those keep
    validating; making it required would reject the installed estate.
    """
    bundle = _load(run_fixture.build(config_path=_with_migration_policy(run_fixture)).path)
    decision = _decision(bundle)
    for budget in decision["budget_results"]:
        budget.pop("denominator")
    for slice_decision in decision["slices"].values():
        for budget in slice_decision["budget_results"]:
            budget.pop("denominator")
    Draft202012Validator(_vendored_schema()["bundle"]).validate(bundle)


def _example_by_id(bundle: dict[str, object], example_id: str) -> dict[str, Any]:
    return next(row for row in _examples(bundle) if row["example_id"] == example_id)


def _tool_call_pair(run_fixture: RunFixture) -> None:
    """Append a source/target pair that answered with tool calls, not text."""
    for role, model_id, tool_name, arguments in (
        ("source", "gemini/gemini-2.5-flash", "get_daily_briefing", {}),
        (
            "target",
            "gemini/gemini-3.1-flash-lite-preview",
            "get_schedule",
            {"date": "2026-08-17"},
        ),
    ):
        append_call(
            run_fixture.run_dir,
            Call(
                run_id=run_fixture.run_id,
                prompt_id="greet",
                example_id="ex_tools",
                model_id=model_id,
                role=role,
                text="",
                finish_reason="tool_calls",
                trace=ToolTrace(
                    calls=[ToolCall(tool_name=tool_name, arguments=arguments, sequence_index=0)]
                ),
            ),
        )


def test_tool_only_call_produces_a_trace_stream(run_fixture: RunFixture) -> None:
    """The regression this feature exists to fix.

    A turn ending in tool calls has ``text == ""``. Before this change the
    bundle carried only that empty string, so the hosted run-detail page showed
    an empty output pane for every agent-suite example.
    """
    _tool_call_pair(run_fixture)

    bundle = _load(run_fixture.build().path)
    example = _example_by_id(bundle, "ex_tools")

    assert example["source_output"] == ""
    assert [t["side"] for t in example["traces"]] == ["source", "target"]
    assert example["traces"][0]["events"][0]["name"] == "get_daily_briefing"
    assert example["traces"][1]["events"][0]["arguments"] == {"date": "2026-08-17"}


def test_text_only_call_produces_no_trace_stream(built_bundle_path: Path) -> None:
    example = _example_by_id(_load(built_bundle_path), "ex1")

    assert example["traces"] == []
    assert example["source_output"] == "hello ada"


def test_manifest_carries_the_cli_version(built_bundle_path: Path) -> None:
    manifest = _load(built_bundle_path)["manifest"]

    assert isinstance(manifest, dict)
    assert manifest["cli_version"]
    assert str(manifest["cli_version"])[0].isdigit()


# ---------------------------------------------------------------------------
# S4 — what ``passed`` and ``tool_match`` mean once a pair can score 0 or 2 rows
# ---------------------------------------------------------------------------


def _pair_calls(example_id: str) -> list[Call]:
    return [
        Call(
            run_id="r1",
            prompt_id="replay",
            example_id=example_id,
            model_id="m-source" if role == "source" else "m-target",
            role=role,  # type: ignore[arg-type]
            text="",
            finish_reason="tool_calls",
        )
        for role in ("source", "target")
    ]


def _record(example_id: str, kind: str, source: float, target: float, **meta: Any) -> EvalRecord:
    return EvalRecord(
        run_id="r1",
        prompt_id="replay",
        example_id=example_id,
        evaluator_name="routing",
        kind=kind,
        source_score=source,
        target_score=target,
        delta=target - source,
        metadata=meta,
    )


def _rows(scores: list[EvalRecord], example_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    from evalshift_cli.hosted.bundle import _build_examples

    calls = [call for example_id in example_ids for call in _pair_calls(example_id)]
    rows = _build_examples(
        suite=Suite(),
        calls=calls,
        scores=scores,
        tool_evaluator_names=frozenset({"routing"}),
    )
    return {row["example_id"]: row for row in rows}


NO_DELETE: dict[str, Any] = {
    "rule_id": "no-delete",
    "rule_type": "forbidden",
    "tool": "delete_file",
    "round_index": 0,
    "detail": "delete_file was called",
}


def _rule_record(
    example_id: str,
    *,
    target_violations: list[dict[str, Any]],
    source_violations: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> EvalRecord:
    """A ``trace_invariants`` row as ``TraceInvariantsEvaluator._record`` writes it."""
    broken = sorted({v["rule_id"] for v in target_violations})
    return EvalRecord(
        run_id="r1",
        prompt_id="replay",
        example_id=example_id,
        evaluator_name="contracts",
        kind="trace_invariants",
        source_score=1.0 if not source_violations else 0.5,
        target_score=1.0 if not target_violations else 0.5,
        delta=(1.0 if not target_violations else 0.5) - (1.0 if not source_violations else 0.5),
        explanation=(
            f"target broke {len(broken)} trace rule(s): {', '.join(broken)}"
            if broken
            else "target kept every trace rule"
        ),
        metadata={
            "rules_checked": ["no-delete"],
            "source_violations": source_violations or [],
            "target_violations": target_violations,
            "owner": "payments",
            **({"failure_categories": ["INVARIANT_VIOLATION"]} if target_violations else {}),
        },
        error=error,
    )


def test_trace_rule_rows_carry_the_explanation_and_both_sides_violations() -> None:
    row = _rows([_rule_record("ex1", target_violations=[NO_DELETE])], ("ex1",))["ex1"]
    score = row["scores"][0]
    assert score["explanation"] == "target broke 1 trace rule(s): no-delete"
    assert score["violations"] == {"source": [], "target": [NO_DELETE]}


def test_the_source_side_violations_ride_along() -> None:
    """A rule both sides broke still fails the target; the source list is what lets a reader see that."""
    row = _rows(
        [_rule_record("ex1", target_violations=[NO_DELETE], source_violations=[NO_DELETE])],
        ("ex1",),
    )["ex1"]
    assert row["scores"][0]["violations"]["source"] == [NO_DELETE]


def test_rules_checked_owner_and_failure_categories_stay_local() -> None:
    """They are in the uploaded evaluator config and decision already — the row must not repeat them."""
    row = _rows([_rule_record("ex1", target_violations=[NO_DELETE])], ("ex1",))["ex1"]
    score = row["scores"][0]
    assert "metadata" not in score
    assert set(score["violations"]) == {"source", "target"}


def test_other_evaluators_carry_null_violations_and_their_explanation() -> None:
    record = _record("ex1", "semantic", 1.0, 0.8).model_copy(update={"explanation": "cosine 0.80"})
    score = _rows([record], ("ex1",))["ex1"]["scores"][0]
    assert score["violations"] is None
    assert score["explanation"] == "cosine 0.80"


def test_an_empty_explanation_is_sent_as_null() -> None:
    score = _rows([_record("ex1", "semantic", 1.0, 0.8)], ("ex1",))["ex1"]["scores"][0]
    assert score["explanation"] is None


def test_an_errored_trace_rule_row_carries_no_violations() -> None:
    """An evaluator that broke measured nothing; an empty target list would read as a clean target."""
    record = _rule_record("ex1", target_violations=[], error="trace missing")
    record = record.model_copy(update={"metadata": {}})
    assert _rows([record], ("ex1",))["ex1"]["scores"][0]["violations"] is None


def test_violation_entries_are_filtered_to_the_server_contract() -> None:
    """A future key on ``Violation.to_dict()`` must not reach a schema that forbids unknown properties."""
    row = _rows(
        [_rule_record("ex1", target_violations=[{**NO_DELETE, "sample": 1, "severity": "high"}])],
        ("ex1",),
    )["ex1"]
    assert row["scores"][0]["violations"]["target"] == [{**NO_DELETE, "sample": 1}]


def test_a_populated_violations_row_validates_against_the_vendored_schema() -> None:
    """A drift guard, not a red-first test: it passes on arrival and must keep passing.

    The fixture bundle's rows carry no violations, so the end-to-end schema
    test never sees a populated ``ScoreViolation``. This one does — with the
    optional ``sample`` ordinal set — so a CLI-side field the server does not
    accept, or a re-vendored schema that tightens the shape, fails here
    instead of as a 422 after upload.
    """
    row = _rows([_rule_record("ex1", target_violations=[{**NO_DELETE, "sample": 1}])], ("ex1",))[
        "ex1"
    ]["scores"][0]
    assert row["violations"]["target"], "the guard is vacuous without a populated entry"
    Draft202012Validator(
        {"$ref": "#/$defs/ExampleScore", "$defs": _vendored_schema()["bundle"]["$defs"]}
    ).validate(row)


def test_a_pair_nothing_measured_is_not_reported_as_passed() -> None:
    """``all()`` over an empty list is ``True`` — absence read as success.

    Every text evaluator on a tool-only turn now writes no row at all, so an
    example can reach the bundle with zero scores. ``passed`` is a required
    non-nullable boolean in the server's manifest schema, so "unknown" cannot
    be expressed; ``False`` is the honest half of what is available.
    """
    rows = _rows([], ("ex_silent",))
    assert rows["ex_silent"]["scores"] == []
    assert rows["ex_silent"]["passed"] is False
    assert rows["ex_silent"]["score"] is None
    assert rows["ex_silent"]["worst_delta_score"] is None


def test_passed_reflects_both_axes_not_whichever_came_last() -> None:
    """A clean conformance row must not absolve a divergence regression."""
    rows = _rows(
        [
            _record("ex_diverged", "tool_selection.conformance", 1.0, 1.0),
            _record("ex_diverged", "tool_selection.divergence", 1.0, 0.0),
        ],
        ("ex_diverged",),
    )
    assert rows["ex_diverged"]["passed"] is False
    assert rows["ex_diverged"]["worst_delta_score"] == pytest.approx(-1.0)


def test_two_axis_scores_stay_distinguishable_in_the_bundle() -> None:
    """Both axes share ``evaluator_name``; only ``kind`` tells them apart.

    The server keys score rows on ``(evaluator_name, kind)`` — dropping the
    kind here made a divergence-enabled run collide on the server's unique
    constraint and 500 on finalize.
    """
    rows = _rows(
        [
            _record("ex_diverged", "tool_selection.conformance", 1.0, 1.0),
            _record("ex_diverged", "tool_selection.divergence", 1.0, 0.0),
        ],
        ("ex_diverged",),
    )
    assert [
        (score["evaluator_name"], score["kind"]) for score in rows["ex_diverged"]["scores"]
    ] == [
        ("routing", "tool_selection.conformance"),
        ("routing", "tool_selection.divergence"),
    ]


def test_a_shared_ground_truth_miss_does_not_fail_the_pair() -> None:
    """Both models missing the suite's ground truth is not a target failure.

    ``0.0 / 0.0`` on the conformance axis means the recording called tools
    neither model called — the suite is wrong, not the migration — and the
    divergence axis says the target did exactly what the source did.
    """
    rows = _rows(
        [
            _record(
                "ex_shared",
                "tool_selection.conformance",
                0.0,
                0.0,
                failure_categories=["TOOL_GROUND_TRUTH_MISS"],
            ),
            _record("ex_shared", "tool_selection.divergence", 1.0, 1.0),
        ],
        ("ex_shared",),
    )
    assert rows["ex_shared"]["passed"] is True
    assert rows["ex_shared"]["tool_match"] is True


def test_tool_match_does_not_demand_a_full_score_on_every_axis() -> None:
    """The old rule was ``all(target_score >= 1.0)`` over every tool row.

    That was a single-axis predicate. With two rows per example it silently
    became "conform to the ground truth *and* match the source", so a suite
    whose ground truth both models fail forced ✗ on every pair — including
    the pairs where the migration changed nothing at all.
    """
    rows = _rows(
        [
            _record(
                "ex_shared",
                "tool_selection.conformance",
                0.0,
                0.0,
                failure_categories=["TOOL_GROUND_TRUTH_MISS"],
            ),
            _record("ex_shared", "tool_selection.divergence", 1.0, 1.0),
            _record("ex_dropped", "tool_selection.conformance", 1.0, 0.0),
            _record("ex_dropped", "tool_selection.divergence", 1.0, 1.0),
        ],
        ("ex_shared", "ex_dropped"),
    )
    assert rows["ex_shared"]["tool_match"] is True
    # The target alone lost the ground truth the source held: a real finding.
    assert rows["ex_dropped"]["tool_match"] is False


class TestBundleResolvesPerSuiteEvaluators:
    """The bundle snapshots what actually scored the run, not the top level.

    ``evaluator_config`` is what the hosted app shows as this run's
    methodology, and ``eval_config_hash`` is derived from it — a top-level
    snapshot would describe evaluators that never ran on this suite.
    """

    _CONFIG = """
        version: 1
        project: acme/model-migration
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        evaluators:
          structural:
            - type: length
              min_chars: 1
        suites:
          main_chat:
            path: ./golden.jsonl
            evaluators:
              structural: []
              tool_selection:
                - name: routing
    """

    def _bundle(self, project: RunFixture, *, suite_name: str | None) -> dict[str, Any]:
        from evalshift_cli.runner.checkpoint import read_state, write_state

        project.config.write_text(self._CONFIG, encoding="utf-8")
        state = read_state(project.run_dir)
        write_state(project.run_dir, state.model_copy(update={"suite_name": suite_name}))
        return _load(project.build().path)

    def _evaluators(self, bundle: dict[str, Any]) -> dict[str, Any]:
        config = bundle["evaluator_config"]
        assert isinstance(config, dict)
        evaluators = config["evaluators"]
        assert isinstance(evaluators, dict)
        return evaluators

    def test_snapshot_is_the_resolved_set(self, run_fixture: RunFixture) -> None:
        evaluators = self._evaluators(self._bundle(run_fixture, suite_name="main_chat"))
        assert [e["name"] for e in evaluators["tool_selection"]] == ["routing"]
        assert evaluators["structural"] == []

    def test_snapshot_is_the_top_level_for_a_raw_suite_path_run(
        self, run_fixture: RunFixture
    ) -> None:
        evaluators = self._evaluators(self._bundle(run_fixture, suite_name=None))
        assert evaluators["tool_selection"] == []
        assert [e["type"] for e in evaluators["structural"]] == ["length"]

    def test_examples_flag_tool_match_from_the_suites_evaluators(
        self, run_fixture: RunFixture
    ) -> None:
        """``tool_match`` keys on the resolved names, like the local report."""
        from evalshift_cli.hosted import bundle as bundle_module

        seen: dict[str, frozenset[str]] = {}
        real = bundle_module._build_examples

        def _spy(**kwargs: Any) -> Any:
            seen["names"] = kwargs["tool_evaluator_names"]
            return real(**kwargs)

        bundle_module._build_examples = _spy  # type: ignore[assignment]
        try:
            self._bundle(run_fixture, suite_name="main_chat")
        finally:
            bundle_module._build_examples = real  # type: ignore[assignment]
        assert seen["names"] == frozenset({"routing"})

    #: ``manifest.eval_config_hash`` for :attr:`_CONFIG` resolved for
    #: ``main_chat``, computed by the CLI at 5b608f2 — before
    #: ``trace_invariants`` existed. A suite override that never mentions the
    #: family must not start hashing its empty default.
    _HASH_BEFORE_TRACE_INVARIANTS = (
        "sha256:18b43295c4f2d1343c322522348565762fffd670a009dfc4bda861de15e6896f"
    )

    def test_eval_config_hash_of_a_suite_resolved_run_is_unchanged(
        self, run_fixture: RunFixture
    ) -> None:
        manifest = self._bundle(run_fixture, suite_name="main_chat")["manifest"]
        assert isinstance(manifest, dict)
        assert manifest["eval_config_hash"] == self._HASH_BEFORE_TRACE_INVARIANTS


class TestBundleShipsNoSuiteContent:
    """Suite content stays local; only content hashes go on the wire.

    ``dataset_snapshot.examples`` was 90%+ of a conversational bundle — every
    history and system prompt verbatim — and the server never wrote a row from
    it nor the web app a pixel. The manifest hashes stay computed over the
    *full* snapshots so their values are unchanged for a given suite/config
    and cross-version diffs stay ``direct``.
    """

    def test_dataset_snapshot_carries_a_hash_not_examples(self, built_bundle_path: Path) -> None:
        snapshot = _load(built_bundle_path)["dataset_snapshot"]
        assert isinstance(snapshot, dict)
        assert set(snapshot) == {"suite_path", "size", "slices", "examples_hash"}
        assert snapshot["size"] == 2
        assert str(snapshot["examples_hash"]).startswith("sha256:")

    def test_inline_prompt_bodies_are_stripped(self, built_bundle_path: Path) -> None:
        config = _load(built_bundle_path)["evaluator_config"]
        assert isinstance(config, dict)
        (prompt,) = config["prompts"]
        assert prompt["content"] is None
        assert str(prompt["content_hash"]).startswith("sha256:")
        with gzip.open(built_bundle_path, "rt", encoding="utf-8") as handle:
            assert "Hello {name}" not in handle.read()

    def test_conversation_history_never_reaches_the_wire(
        self, tmp_path: Path, run_fixture: RunFixture
    ) -> None:
        """A history's system prompt must not appear anywhere in the bytes."""
        sentinel = "TOP-SECRET-SYSTEM-PROMPT"
        _rewrite_suite_history(run_fixture.suite, sentinel)
        path = run_fixture.build(output=tmp_path / "with_history.json.gz").path
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            assert sentinel not in handle.read()

    def test_dataset_hash_is_still_content_derived(
        self, tmp_path: Path, run_fixture: RunFixture
    ) -> None:
        """Same size, same tags, different content ⇒ different ``dataset_hash``.

        This is the invariant that forbids hashing the wire form: metadata
        alone is identical between these two builds.
        """
        before = run_fixture.build(output=tmp_path / "before.json.gz")
        _rewrite_suite_history(run_fixture.suite, "a system prompt")
        after = run_fixture.build(output=tmp_path / "after.json.gz")
        assert before.manifest["dataset_hash"] != after.manifest["dataset_hash"]
        snapshot_before = _load(before.path)["dataset_snapshot"]
        snapshot_after = _load(after.path)["dataset_snapshot"]
        assert isinstance(snapshot_before, dict)
        assert isinstance(snapshot_after, dict)
        assert {k: v for k, v in snapshot_before.items() if k != "examples_hash"} == {
            k: v for k, v in snapshot_after.items() if k != "examples_hash"
        }
        assert snapshot_before["examples_hash"] != snapshot_after["examples_hash"]


class TestEvaluatorConfigKeepsTheLegacySlicesKey:
    """Removing the top-level ``slices:`` key must not move ``eval_config_hash``.

    The server pairs a run with its baseline only on an equal
    ``eval_config_hash``, and that hash is computed over the whole
    ``evaluator_config`` snapshot — which has always carried a ``"slices"``
    key. Dropping the key would change the hash of every config, including
    the ones that never set ``slices:``, and orphan every hosted baseline.
    """

    _CONFIG = """
        version: 1
        project: acme/model-migration
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini/gemini-2.5-flash
          target_model: gemini/gemini-3.1-flash-lite-preview
        evaluators:
          structural:
            - type: length
              min_chars: 1
          llm_judge:
            - criterion_name: tone
              criterion_prompt: Which reply is friendlier?
        migration_policy:
          max_overall_regression_rate: 0.10
          slices:
            checkout:
              max_overall_regression_rate: 0.05
    """

    #: ``manifest.eval_config_hash`` for :attr:`_CONFIG`, computed by the CLI at
    #: origin/main af4e6fe — the last commit that still accepted ``slices:``.
    #: If this changes, every hosted baseline stops matching its next run.
    _HASH_BEFORE_THE_REMOVAL = (
        "sha256:37936052974aecda1876fb406cb98a60c30b91b403b5ca59f6b53f0f1c1827e5"
    )

    def _bundle(self, run_fixture: RunFixture) -> dict[str, Any]:
        run_fixture.config.write_text(self._CONFIG, encoding="utf-8")
        return _load(run_fixture.build().path)

    def test_eval_config_hash_of_a_slices_less_config_is_unchanged(
        self, run_fixture: RunFixture
    ) -> None:
        manifest = self._bundle(run_fixture)["manifest"]
        assert isinstance(manifest, dict)
        assert manifest["eval_config_hash"] == self._HASH_BEFORE_THE_REMOVAL

    def test_evaluator_config_still_ships_an_empty_slices_list(
        self, run_fixture: RunFixture
    ) -> None:
        config = self._bundle(run_fixture)["evaluator_config"]
        assert isinstance(config, dict)
        assert config["slices"] == []


class TestEvaluatorConfigHashesTraceInvariantsOnlyWhenSet:
    """``trace_invariants`` enters ``eval_config_hash`` only when configured.

    An empty family is dropped from the snapshot so configs that never wrote
    the key keep the hash their hosted baselines were recorded under (pinned
    by :class:`TestEvaluatorConfigKeepsTheLegacySlicesKey`). Configured rules
    are methodology and must move the hash like any other evaluator.
    """

    _BASE = TestEvaluatorConfigKeepsTheLegacySlicesKey._CONFIG
    _WITH_RULES = """
        version: 1
        project: acme/model-migration
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini/gemini-2.5-flash
          target_model: gemini/gemini-3.1-flash-lite-preview
        evaluators:
          structural:
            - type: length
              min_chars: 1
          llm_judge:
            - criterion_name: tone
              criterion_prompt: Which reply is friendlier?
          trace_invariants:
            - name: payments_contract
              rules:
                - id: no-refund
                  type: forbidden
                  tools: [refund]
        migration_policy:
          max_overall_regression_rate: 0.10
          slices:
            checkout:
              max_overall_regression_rate: 0.05
    """

    def _bundle(self, run_fixture: RunFixture, text: str) -> dict[str, Any]:
        run_fixture.config.write_text(text, encoding="utf-8")
        return _load(run_fixture.build().path)

    def test_an_unconfigured_family_is_absent_from_the_snapshot(
        self, run_fixture: RunFixture
    ) -> None:
        config = self._bundle(run_fixture, self._BASE)["evaluator_config"]
        assert isinstance(config, dict)
        assert "trace_invariants" not in config["evaluators"]

    def test_configured_rules_ship_and_move_the_hash(self, run_fixture: RunFixture) -> None:
        bundle = self._bundle(run_fixture, self._WITH_RULES)
        config = bundle["evaluator_config"]
        assert isinstance(config, dict)
        [family] = config["evaluators"]["trace_invariants"]
        assert family["name"] == "payments_contract"
        assert [rule["id"] for rule in family["rules"]] == ["no-refund"]
        manifest = bundle["manifest"]
        assert isinstance(manifest, dict)
        assert (
            manifest["eval_config_hash"]
            != TestEvaluatorConfigKeepsTheLegacySlicesKey._HASH_BEFORE_THE_REMOVAL
        )


def _rewrite_suite_history(suite_path: Path, system_prompt: str) -> None:
    """Give every suite example a conversation prefix carrying ``system_prompt``."""
    rows = [
        json.loads(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["history"] = [{"role": "system", "content": system_prompt}]
    suite_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_bundle_latency_is_incomparable_when_a_side_replayed_rounds() -> None:
    from evalshift_cli.hosted.bundle import _build_examples

    def call(role: str, example_id: str, latency: int, cached_rounds: int = 0) -> Call:
        return Call(
            run_id="r_20260601_abc123",
            prompt_id="p",
            example_id=example_id,
            model_id="m",
            role=role,  # type: ignore[arg-type]
            latency_ms=latency,
            cached_rounds=cached_rounds,
        )

    rows = _build_examples(
        suite=Suite(examples=[]),
        calls=[
            call("source", "live", 100),
            call("target", "live", 130),
            call("source", "mixed", 100),
            call("target", "mixed", 130, cached_rounds=2),
        ],
        scores=[],
        tool_evaluator_names=frozenset(),
    )
    by_id = {r["example_id"]: r for r in rows}
    assert by_id["live"]["latency_comparable"] is True
    assert by_id["live"]["delta_latency_ms"] == 30
    assert by_id["mixed"]["latency_comparable"] is False
    assert by_id["mixed"]["delta_latency_ms"] == 0
    # The count itself stays local: the bundle row schema has no such field.
    assert "cached_rounds" not in by_id["mixed"]
