"""Wire the insights package into a run directory.

Owns ``insights.json``, which is a **cache envelope and not the bundle block**.
The cache needs a ``config_hash`` to know whether it is still valid; the
server's ``Insights`` model is ``extra="forbid"`` and rejects any key it does
not define. So the two shapes stay separated on disk::

    {"config_hash": "sha256:…", "insight": { …the server's seven keys… }}

and only the inner member ever reaches a bundle. Reading it back goes through
:func:`~evalshift_cli.insights.models.insight_from_dict` rather than a
``dict``-and-pop, so an envelope this module does not recognise is a cache
miss instead of a 400 at finalize.

Nothing here is allowed to fail a run: :func:`ensure_insight` swallows every
exception and returns ``None``. The narrative is prose around figures that were
already computed — a run that has its statistics is not worth failing over a
missing paragraph.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evalshift_cli.analysis.policy import MigrationDecision, inconclusive_decision
from evalshift_cli.analysis.statistics import ComparisonResult
from evalshift_cli.cli.commands.analyze import ANALYSIS_FILENAME, MIGRATION_DECISION_FILENAME
from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME
from evalshift_cli.config.models import EvalShiftConfig
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.insights.facts import ExampleFact, Facts, build_facts
from evalshift_cli.insights.generator import generate_insight
from evalshift_cli.insights.models import (
    Insight,
    insight_from_dict,
    insight_to_dict,
)
from evalshift_cli.models.client import ModelClient
from evalshift_cli.models.registry import missing_api_keys
from evalshift_cli.reports.economics import build_economics
from evalshift_cli.runner.checkpoint import iter_calls, read_state
from evalshift_cli.runner.models import Call, RunState, representative_calls
from evalshift_cli.suite.loader import SuiteError, load_jsonl
from evalshift_cli.suite.models import Suite

log = logging.getLogger(__name__)

INSIGHTS_FILENAME: str = "insights.json"

#: Envelope keys. A file carrying anything else is not one we wrote.
_ENVELOPE_KEYS: frozenset[str] = frozenset({"config_hash", "insight"})


def ensure_insight(
    run_dir: Path,
    *,
    cfg: EvalShiftConfig | None,
    enabled: bool = True,
    client: ModelClient | None = None,
    env: Mapping[str, str] | None = None,
) -> Insight | None:
    """Return the run's narrative, generating and caching it if needed.

    Args:
        run_dir: The completed run directory. ``insights.json`` is read from
            and written to here.
        cfg: The loaded config, or ``None`` when it could not be loaded — a
            report rendered against a foreign run directory has no model to
            generate with, so that is a skip.
        enabled: ``False`` for ``--no-insights``. Skips silently: the user
            asked for nothing, so there is nothing to warn about.
        client: Model client to generate with. Defaults to a real
            :class:`~evalshift_cli.models.client.ModelClient`.
        env: Environment to read provider API keys from. Defaults to
            ``os.environ``.

    Returns:
        The narrative, or ``None`` when generation was skipped or failed.
        Never raises — see the module docstring.
    """
    if not enabled:
        return None
    if cfg is None:
        log.warning("skipping insights: no usable evalshift.yaml for this run")
        return None

    try:
        state = read_state(run_dir)
        model = cfg.defaults.insights_model or cfg.defaults.judge_model
        config_hash = insights_config_hash(run_config_hash=state.config_hash, model=model)

        cached = load_cached_insight(run_dir, config_hash=config_hash)
        if cached is not None:
            return cached

        missing_keys = missing_api_keys(model, env if env is not None else os.environ)
        if missing_keys:
            log.warning(
                "skipping insights: no API key for %s; export %s",
                model,
                " or ".join(missing_keys),
            )
            return None

        facts = build_run_facts(run_dir, state=state)
        insight = asyncio.run(
            generate_insight(facts, model=model, client=client or ModelClient()),
        )
        write_insight(run_dir, insight, config_hash=config_hash)
    # Deliberately bare: a narrative never fails a run. See the module docstring.
    except Exception as exc:
        log.warning("skipping insights: generation failed (%s)", exc)
        return None
    return insight


def insights_config_hash(*, run_config_hash: str, model: str) -> str:
    """The cache key for a run's narrative.

    ``run_config_hash`` is the run's own ``state.config_hash`` — the canonical
    config plus the suite path — which moves whenever anything the figures are
    derived from moves. The model id is folded in on top because provenance is
    part of the narrative: prose attributed to a model that no longer writes it
    is stale even when every figure in it still holds.
    """
    return _sha256({"config_hash": run_config_hash, "model": model})


def _sha256(value: Any) -> str:
    """Hash a JSON value the way the bundle spec hashes one.

    Deliberately local rather than imported from ``hosted.bundle``: that module
    reads this one for its ``insights`` block, and the cycle would be real.
    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_cached_insight(run_dir: Path, *, config_hash: str) -> Insight | None:
    """Return the cached narrative when it is still valid, else ``None``.

    A missing file, an envelope shape this module did not write, a hash
    mismatch and an unparseable insight are all the same answer: a miss.
    """
    envelope = _read_envelope(run_dir)
    if envelope is None or envelope.get("config_hash") != config_hash:
        return None
    return _insight_from_envelope(envelope)


def write_insight(run_dir: Path, insight: Insight, *, config_hash: str) -> Path:
    """Persist ``insight`` in its cache envelope and return the path."""
    path = run_dir / INSIGHTS_FILENAME
    payload = {"config_hash": config_hash, "insight": insight_to_dict(insight)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_bundle_insight(run_dir: Path) -> dict[str, Any] | None:
    """The ``insights`` block for a bundle, or ``None`` when there is none.

    Re-serialised through :class:`~evalshift_cli.insights.models.Insight` rather
    than handed over as read: a hand-edited cache file cannot then smuggle a
    key past the server's ``extra="forbid"`` model. Validity of the cache is
    not checked — a narrative written for an earlier config still describes
    this run's artifacts, and ``report`` is what refreshes it.
    """
    envelope = _read_envelope(run_dir)
    if envelope is None:
        return None
    insight = _insight_from_envelope(envelope)
    return None if insight is None else insight_to_dict(insight)


def build_run_facts(run_dir: Path, *, state: RunState | None = None) -> Facts:
    """Render the FACTS block for a completed run directory.

    Reads the same artifacts the report does — including the decision, which is
    taken from ``migration_decision.json`` rather than recomputed. The narrative
    sits beside a verdict block rendered from that same file, and a document
    cannot carry two verdicts: recomputing from whatever the config says *now*
    would let a policy edit between ``analyze`` and ``report`` disagree with the
    paragraph next to it.

    ``analyze`` only writes that file when a ``migration_policy`` is configured,
    so a run without one falls back to the ``inconclusive`` decision — there is
    always a verdict and a budget table to describe.
    """
    state = state or read_state(run_dir)
    calls = list(iter_calls(run_dir))
    scores = _read_scores(run_dir)
    comparisons = _read_comparisons(run_dir)
    suite = _load_suite(state.suite_path)

    return build_facts(
        decision=_decision(
            run_dir=run_dir, state=state, comparisons=comparisons, scores=scores, calls=calls
        ),
        comparisons=comparisons,
        # Run-level, not per-prompt: every call in the run rolls up here.
        economics=build_economics(calls),
        examples=_example_facts(calls=calls, scores=scores, suite=suite, state=state),
        # Every scored row, advisory ones included: `blocking` is what tells a
        # gate that measured nothing from an advisory axis that did.
        records=scores,
        state=state,
    )


# ---------------------------------------------------------------------------
# Envelope I/O
# ---------------------------------------------------------------------------


def _read_envelope(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / INSIGHTS_FILENAME
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or set(raw) != _ENVELOPE_KEYS:
        log.debug("ignoring unrecognised %s in %s", INSIGHTS_FILENAME, run_dir)
        return None
    return raw


def _insight_from_envelope(envelope: Mapping[str, Any]) -> Insight | None:
    payload = envelope.get("insight")
    if not isinstance(payload, dict):
        return None
    try:
        return insight_from_dict(payload)
    except ValueError as exc:
        log.debug("ignoring unusable cached insight: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Run artifacts
# ---------------------------------------------------------------------------


def _read_scores(run_dir: Path) -> list[EvalRecord]:
    text = (run_dir / SCORES_FILENAME).read_text(encoding="utf-8")
    return [EvalRecord.model_validate_json(line) for line in text.splitlines() if line.strip()]


def _read_comparisons(run_dir: Path) -> list[ComparisonResult]:
    raw: Any = json.loads((run_dir / ANALYSIS_FILENAME).read_text(encoding="utf-8"))
    rows = raw.get("comparisons", []) if isinstance(raw, dict) else []
    return [ComparisonResult(**row) for row in rows]


def _load_suite(suite_path: str) -> Suite:
    """Load the run's suite, tolerating a moved or edited file.

    Mirrors ``reports/json.py::_load_suite``: without it the narrative loses
    the example inputs it quotes, which is a degraded finding rather than a
    reason to skip the whole block.
    """
    try:
        return load_jsonl(Path(suite_path))
    except (SuiteError, FileNotFoundError, OSError):
        return Suite()


def _decision(
    *,
    run_dir: Path,
    state: RunState,
    comparisons: list[ComparisonResult],
    scores: list[EvalRecord],
    calls: list[Call],
) -> MigrationDecision:
    """The decision ``analyze`` persisted, or an ``inconclusive`` stand-in."""
    persisted = _read_persisted_decision(run_dir)
    if persisted is not None:
        return persisted
    return inconclusive_decision(
        run_id=state.run_id,
        source_model=state.models.source,
        target_model=state.models.target,
        comparisons=comparisons,
        records=scores,
        calls=calls,
    )


def _read_persisted_decision(run_dir: Path) -> MigrationDecision | None:
    """Read ``migration_decision.json``, or ``None`` when it is unusable.

    Absent, unparseable and unrecognised are the same answer. A hand-edited
    artifact degrades the narrative to the ``inconclusive`` fallback rather
    than failing a report that already has its statistics.
    """
    try:
        raw: Any = json.loads(
            (run_dir / MIGRATION_DECISION_FILENAME).read_text(encoding="utf-8"),
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return MigrationDecision.from_dict(raw)
    except ValueError as exc:
        log.debug("ignoring unusable %s: %s", MIGRATION_DECISION_FILENAME, exc)
        return None


def _example_facts(
    *,
    calls: Sequence[Call],
    scores: Sequence[EvalRecord],
    suite: Suite,
    state: RunState,
) -> list[ExampleFact]:
    """One fact per (prompt, example) pair — the bundle's notion of an example.

    Labelled with the bare example id on a single-prompt run and with
    ``prompt/example`` otherwise, so a narrative naming an example says
    something a reader can find in a multi-prompt report.
    """
    inputs_by_id = {
        example.id: json.dumps(example.inputs, sort_keys=True, ensure_ascii=False)
        for example in suite.examples
    }
    texts: dict[tuple[str, str], dict[str, str]] = {}
    # Sample 0's text on a repeated-sampling run; the fact is one per example.
    for call in representative_calls(calls):
        texts.setdefault((call.prompt_id, call.example_id), {})[call.role] = call.text
    worst: dict[tuple[str, str], float] = {}
    for score in scores:
        if score.error is not None:
            continue
        key = (score.prompt_id, score.example_id)
        worst[key] = min(worst.get(key, score.delta), score.delta)

    multi_prompt = len(state.prompt_ids) > 1
    return [
        ExampleFact(
            example_id=f"{prompt_id}/{example_id}" if multi_prompt else example_id,
            worst_delta_score=worst.get((prompt_id, example_id)),
            input_text=inputs_by_id.get(example_id),
            source_output=pair.get("source"),
            target_output=pair.get("target"),
        )
        for (prompt_id, example_id), pair in sorted(texts.items())
    ]


__all__ = [
    "INSIGHTS_FILENAME",
    "build_run_facts",
    "ensure_insight",
    "insights_config_hash",
    "load_cached_insight",
    "read_bundle_insight",
    "write_insight",
]
