"""Tests for the insights stage: the cache envelope and the ``report`` wiring.

Never hits a real model — ``run_report`` takes an injectable client so the whole
skip/cache/generate/never-fatal flow is exercised against queued responses.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest

from evalshift_cli.analysis.policy import evaluate_migration_policy
from evalshift_cli.cli.commands.analyze import MIGRATION_DECISION_FILENAME
from evalshift_cli.cli.commands.report import run_report
from evalshift_cli.config.loader import load_config
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.insights.models import Insight, insight_from_dict
from evalshift_cli.insights.stage import INSIGHTS_FILENAME, build_run_facts, read_bundle_insight
from evalshift_cli.models.client import ModelClient, ModelError
from evalshift_cli.runner.checkpoint import iter_calls, read_state
from tests.conftest import RunFixture
from tests.unit.insights_factories import (
    FakeModelClient,
    generation_payload,
    unmeasured_comparison,
)


class ExplodingModelClient:
    """A client whose every call fails, the way a revoked key behaves."""

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, **_: Any) -> Any:
        self.call_count += 1
        raise ModelError("provider said no")


@pytest.fixture
def fake_client() -> FakeModelClient:
    client = FakeModelClient()
    # More than any single test needs: the cache tests assert on call_count,
    # not on the queue running dry.
    client.queue_responses(*[generation_payload() for _ in range(6)])
    return client


@pytest.fixture
def project(run_fixture: RunFixture, monkeypatch: pytest.MonkeyPatch) -> RunFixture:
    """``run_fixture`` with a key for the default judge model's provider."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (run_fixture.run_dir / INSIGHTS_FILENAME).unlink(missing_ok=True)
    return run_fixture


def _report(
    project: RunFixture,
    *,
    client: Any | None = None,
    insights: bool = True,
) -> None:
    run_report(
        run_id=project.run_id,
        config_path=project.config,
        runs_base=project.runs_base,
        insights=insights,
        client=cast(ModelClient, client) if client is not None else None,
    )


def _envelope(project: RunFixture) -> dict[str, Any]:
    raw = (project.run_dir / INSIGHTS_FILENAME).read_text(encoding="utf-8")
    payload: Any = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def test_report_writes_insights_json(project: RunFixture, fake_client: FakeModelClient) -> None:
    _report(project, client=fake_client)
    assert (project.run_dir / INSIGHTS_FILENAME).exists()
    assert fake_client.call_count == 1


def test_report_reuses_cached_insights(project: RunFixture, fake_client: FakeModelClient) -> None:
    """Re-running report or push must not pay for generation twice."""
    _report(project, client=fake_client)
    calls_after_first = fake_client.call_count
    _report(project, client=fake_client)
    assert fake_client.call_count == calls_after_first


def test_no_insights_flag_skips_generation(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    _report(project, client=fake_client, insights=False)
    assert not (project.run_dir / INSIGHTS_FILENAME).exists()
    assert fake_client.call_count == 0


def test_generation_failure_is_never_fatal(
    project: RunFixture, caplog: pytest.LogCaptureFixture
) -> None:
    exploding = ExplodingModelClient()
    with caplog.at_level("WARNING"):
        _report(project, client=exploding)

    assert (project.run_dir / "report.html").exists()
    assert not (project.run_dir / INSIGHTS_FILENAME).exists()
    assert "insights" in caplog.text.lower()


def test_a_missing_api_key_skips_generation(
    run_fixture: RunFixture,
    fake_client: FakeModelClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No key for the insights model is a skip, not a failed provider call."""
    (run_fixture.run_dir / INSIGHTS_FILENAME).unlink(missing_ok=True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with caplog.at_level("WARNING"):
        _report(run_fixture, client=fake_client)

    assert fake_client.call_count == 0
    assert not (run_fixture.run_dir / INSIGHTS_FILENAME).exists()
    assert "insights" in caplog.text.lower()


def test_a_cached_insight_is_reused_without_a_key(
    project: RunFixture,
    fake_client: FakeModelClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key gates generation, not reuse — a bundled run must still ship prose."""
    _report(project, client=fake_client)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    _report(project, client=fake_client)
    assert fake_client.call_count == 1
    assert (project.run_dir / INSIGHTS_FILENAME).exists()


def test_stale_cached_insights_are_regenerated(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """A ``config_hash`` mismatch is a miss, not a silently reused narrative."""
    _report(project, client=fake_client)
    path = project.run_dir / INSIGHTS_FILENAME
    payload = _envelope(project)
    payload["config_hash"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    before = fake_client.call_count
    _report(project, client=fake_client)
    assert fake_client.call_count == before + 1
    assert _envelope(project)["config_hash"] != "sha256:" + "0" * 64


def test_a_changed_insights_model_invalidates_the_cache(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """Provenance is part of the narrative, so the model id is part of the key."""
    _report(project, client=fake_client)
    config = project.config.read_text(encoding="utf-8")
    project.config.write_text(
        config.replace(
            "defaults:\n",
            "defaults:\n          insights_model: gemini-2.5-flash\n",
        ),
        encoding="utf-8",
    )

    _report(project, client=fake_client)
    assert fake_client.call_count == 2


def test_a_reanalysed_decision_invalidates_the_cache(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """A policy edit then ``analyze`` + ``report`` on the same run.

    ``state.config_hash`` is fixed when the run is created, so it cannot see the
    edit: the report rendered the new verdict beside the cached narrative's old
    one -- "FAIL under the configured policy" next to a PASS verdict block.
    """
    _report(project, client=fake_client)
    _write_policy(project, "  max_overall_regression_rate: 0.0\n")
    _persist_decision(project)

    _report(project, client=fake_client)
    assert fake_client.call_count == 2


def test_rescored_records_invalidate_the_cache(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """``evaluate`` again on the same run (a new key, an edited evaluator)."""
    _report(project, client=fake_client)
    scores = project.run_dir / "scores.jsonl"
    rows = [line for line in scores.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    _report(project, client=fake_client)
    assert fake_client.call_count == 2


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_envelope_separates_the_cache_key_from_the_insight(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """``config_hash`` is cache metadata and must never sit beside the prose."""
    _report(project, client=fake_client)
    payload = _envelope(project)
    assert set(payload) == {"config_hash", "insight"}
    assert "config_hash" not in payload["insight"]


def test_read_bundle_insight_returns_the_inner_member(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    _report(project, client=fake_client)
    insight = read_bundle_insight(project.run_dir)
    assert insight is not None
    assert insight["verdict_summary"]


def test_read_bundle_insight_treats_an_unrecognised_envelope_as_absent(
    project: RunFixture,
) -> None:
    """Never ``dict``-and-pop: an envelope we don't understand is a miss."""
    (project.run_dir / INSIGHTS_FILENAME).write_text(
        json.dumps({"model": "m", "verdict_summary": "flat, not wrapped"}),
        encoding="utf-8",
    )
    assert read_bundle_insight(project.run_dir) is None


def test_read_bundle_insight_is_none_without_the_file(project: RunFixture) -> None:
    assert read_bundle_insight(project.run_dir) is None


def test_insight_from_dict_rejects_an_unknown_key() -> None:
    """The server's model is ``extra="forbid"``; a stray key is a cache miss here."""
    payload = {
        "model": "m",
        "generated_at": "2026-08-03T09:14:22Z",
        "verdict_summary": "v",
        "advisory_summary": "a",
        "economics_summary": "e",
        "recommendation": "r",
        "findings": [],
        "config_hash": "sha256:0",
    }
    with pytest.raises(ValueError, match="unexpected"):
        insight_from_dict(payload)


def test_insight_round_trips_through_the_envelope(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    _report(project, client=fake_client)
    restored = insight_from_dict(_envelope(project)["insight"])
    assert isinstance(restored, Insight)
    assert restored.verdict_summary


# ---------------------------------------------------------------------------
# The decision the narrative describes
# ---------------------------------------------------------------------------


def _write_policy(project: RunFixture, body: str | None) -> None:
    """Rewrite the fixture config, with ``migration_policy`` set or removed."""
    base = textwrap.dedent(project.config.read_text(encoding="utf-8")).split("\nmigration_policy:")[
        0
    ]
    tail = "" if body is None else f"\nmigration_policy:\n{body}"
    project.config.write_text(base + tail, encoding="utf-8")


def _persist_decision(project: RunFixture) -> str:
    """Run the policy the way ``analyze`` does and persist its decision."""
    state = read_state(project.run_dir)
    scores = [
        EvalRecord.model_validate_json(line)
        for line in (project.run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    policy = load_config(project.config).migration_policy
    assert policy is not None
    decision = evaluate_migration_policy(
        run_id=state.run_id,
        source_model=state.models.source,
        target_model=state.models.target,
        policy=policy,
        comparisons=[],
        records=scores,
        calls=list(iter_calls(project.run_dir)),
    )
    (project.run_dir / MIGRATION_DECISION_FILENAME).write_text(
        json.dumps(decision.to_dict(), indent=2), encoding="utf-8"
    )
    return decision.verdict


def test_the_narrative_describes_the_decision_the_report_renders(project: RunFixture) -> None:
    """One document cannot carry two verdicts.

    ``report.html``'s verdict block renders the decision ``analyze`` persisted.
    Recomputing it here from whatever the config says *now* lets a policy edit
    between ``analyze`` and ``report`` put "fail" in the verdict block and
    "inconclusive" in the prose beside it.
    """
    _write_policy(project, "  max_overall_regression_rate: 0.0\n")
    persisted = _persist_decision(project)
    assert persisted == "fail"

    # The user drops the policy and re-runs report only. ``analyze`` did not
    # re-run and does not delete what it wrote, so migration_decision.json
    # still holds the strict verdict the report will render.
    _write_policy(project, None)
    facts = build_run_facts(project.run_dir)
    assert facts.verdict == persisted
    assert facts.rendered["budgets_total"] == "7"


def _write_unmeasured_analysis(project: RunFixture, evaluator_name: str) -> None:
    """Replace ``analysis.json`` with one evaluator that scored no pair."""
    comparison = unmeasured_comparison(evaluator_name)
    (project.run_dir / "analysis.json").write_text(
        json.dumps({"comparisons": [asdict(comparison)]}), encoding="utf-8"
    )


def _mark_advisory(project: RunFixture, evaluator_name: str) -> None:
    """Rewrite ``scores.jsonl`` with ``blocking: false`` on one evaluator."""
    path = project.run_dir / "scores.jsonl"
    rows = [
        EvalRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path.write_text(
        "".join(
            row.model_copy(
                update={"blocking": row.evaluator_name != evaluator_name}
            ).model_dump_json()
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_a_blind_gate_on_disk_reaches_the_narrative(project: RunFixture) -> None:
    """``build_run_facts`` reads both halves of the answer off the run directory.

    The comparison says the evaluator scored nothing; ``scores.jsonl`` says
    whether it was gating. Either half alone gets the set wrong.
    """
    _write_unmeasured_analysis(project, "length")
    facts = build_run_facts(project.run_dir)
    assert facts.unmeasured_evaluators == ["length"]
    assert "measured nothing" in facts.coverage_basis.lower()


def test_an_advisory_evaluator_on_disk_is_not_a_blind_gate(project: RunFixture) -> None:
    """Same comparison, ``blocking: false`` in ``scores.jsonl`` — not a gate."""
    _write_unmeasured_analysis(project, "length")
    _mark_advisory(project, "length")
    facts = build_run_facts(project.run_dir)
    assert facts.unmeasured_evaluators == []
    assert facts.coverage_basis == ""


def test_the_narrative_falls_back_when_no_decision_was_persisted(project: RunFixture) -> None:
    """``analyze`` writes nothing without a policy, so the run still needs one."""
    (project.run_dir / MIGRATION_DECISION_FILENAME).unlink(missing_ok=True)
    facts = build_run_facts(project.run_dir)
    assert facts.verdict == "inconclusive"
    assert facts.rendered["budgets_total"] == "0"


def test_an_unreadable_persisted_decision_falls_back(project: RunFixture) -> None:
    """A hand-edited artifact must not fail the report it is read for."""
    (project.run_dir / MIGRATION_DECISION_FILENAME).write_text("{not json", encoding="utf-8")
    facts = build_run_facts(project.run_dir)
    assert facts.verdict == "inconclusive"


# ---------------------------------------------------------------------------
# The HTML report
# ---------------------------------------------------------------------------


def test_the_html_report_renders_the_narrative(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    _report(project, client=fake_client)
    html = (project.run_dir / "report.html").read_text(encoding="utf-8")
    assert "PASS under the configured policy." in html
    assert "Machine-written" in html


def test_the_narratives_verdict_paragraph_leads_the_panel(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """The verdict read is the lead, and it is tinted by the run's own verdict.

    It is the one paragraph a reader has to come away with, so it carries the
    larger type and the accent rail; the advisory and economics reads sit under
    it at body weight.
    """
    _write_policy(project, "  max_overall_regression_rate: 0.0\n")
    verdict = _persist_decision(project)
    _report(project, client=fake_client)
    html = (project.run_dir / "report.html").read_text(encoding="utf-8")

    # Tinted by the decision ``analyze`` persisted, not by a fresh one — the
    # rail beside the prose and the headline above it are the same verdict.
    assert f'class="insight-block insight-block-lead verdict-{verdict}"' in html
    # The advisory and economics reads stay at body weight.
    assert html.count('class="insight-block"') == 2


def test_the_verdict_lead_stays_neutral_without_a_decision(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """No configured policy still gets the lead treatment, with no accent.

    The accent says which way the verdict went; with no verdict to report it
    would be decoration standing in for a fact.
    """
    _report(project, client=fake_client)
    html = (project.run_dir / "report.html").read_text(encoding="utf-8")
    assert 'class="insight-block insight-block-lead verdict-unknown"' in html


def test_a_tool_call_reply_reaches_the_samples_as_its_calls(project: RunFixture) -> None:
    """A single-shot tool replay records ``text: ""``; the sample must not read empty.

    The narrative model was shown a blank target output for every reply that
    called a tool, and reported that the target "returned empty outputs".
    """
    raw = project.run_dir / "raw.jsonl"
    lines = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]
    for line in lines:
        if line["role"] == "target":
            line["text"] = ""
            line["trace"] = {
                "calls": [{"tool_name": "get_tasks", "arguments": {}, "sequence_index": 0}],
            }
    raw.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    samples = build_run_facts(project.run_dir).regression_samples

    assert samples
    assert {sample.target_output for sample in samples} == {"[tool calls] get_tasks({})"}


def test_the_html_report_escapes_model_output(project: RunFixture) -> None:
    """The prose is model output rendered into HTML."""
    client = FakeModelClient()
    client.queue_responses(generation_payload(recommendation="<script>alert(1)</script>"))
    _report(project, client=client)

    html = (project.run_dir / "report.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_the_html_report_renders_without_a_narrative(project: RunFixture) -> None:
    _report(project, insights=False)
    html = (project.run_dir / "report.html").read_text(encoding="utf-8")
    assert "Machine-written" not in html
    assert "Executive summary" in html


def test_report_json_carries_no_narrative(
    project: RunFixture, fake_client: FakeModelClient
) -> None:
    """``report.json`` is the computed payload; the prose lives beside it."""
    _report(project, client=fake_client)
    payload: Any = json.loads((project.run_dir / "report.json").read_text(encoding="utf-8"))
    assert "insights" not in payload


def test_report_still_works_without_a_config(
    project: RunFixture, fake_client: FakeModelClient, tmp_path: Path
) -> None:
    """A foreign run dir has no config, so no model to generate with."""
    run_report(
        run_id=project.run_id,
        config_path=tmp_path / "nope.yaml",
        runs_base=project.runs_base,
        insights=True,
        client=cast(ModelClient, fake_client),
    )
    assert fake_client.call_count == 0
    assert (project.run_dir / "report.html").exists()
