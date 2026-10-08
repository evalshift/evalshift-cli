"""Build hosted run bundles from local EvalShift run artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError

from evalshift_cli.analysis.policy import evaluate_migration_policy, inconclusive_decision
from evalshift_cli.analysis.statistics import ComparisonResult
from evalshift_cli.cli.commands._suites import derive_suite_slug
from evalshift_cli.cli.commands.analyze import ANALYSIS_FILENAME
from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.config.models import EvalShiftConfig, EvaluatorsConfig
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.trace_invariants import KIND as KIND_INVARIANTS
from evalshift_cli.hosted.trace_events import from_tool_trace
from evalshift_cli.insights.stage import read_bundle_insight
from evalshift_cli.reports.economics import (
    build_economics,
    is_empty_output,
    methodology_notes,
    role_economics_to_dict,
)
from evalshift_cli.runner.checkpoint import (
    RAW_FILENAME,
    STATE_FILENAME,
    iter_calls,
    read_state,
    run_dir_for,
)
from evalshift_cli.runner.models import Call, representative_calls
from evalshift_cli.suite.loader import SuiteError, load_jsonl
from evalshift_cli.suite.models import Suite, SuiteExample

BUNDLE_FILENAME = "run_bundle.json.gz"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class BundleError(Exception):
    """Raised when a local run cannot be packaged for hosted upload."""


@dataclass(frozen=True, slots=True)
class BundleBuildResult:
    run_id: str
    path: Path
    manifest: dict[str, Any]
    size_bytes: int


def canonical_json(value: Any) -> bytes:
    """Return canonical JSON bytes for hashing and deterministic bundles."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_hash(value: Any) -> str:
    """Return the bundle-spec SHA-256 hash for a JSON value."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def build_bundle(
    run_id: str,
    *,
    config_path: Path,
    suite_path: Path | None = None,
    suite_name: str | None = None,
    runs_base: Path,
    output: Path | None = None,
    project: str | None = None,
    env: Mapping[str, str] | None = None,
) -> BundleBuildResult:
    """Package a completed local run as ``run_bundle.json.gz``.

    When ``suite_path`` is ``None`` the suite recorded in the run's
    ``state.json`` is used, so the bundle can be built from any working
    directory and named/promoted-capture suites resolve automatically.
    """
    run_dir = run_dir_for(run_id, runs_base)
    _require_artifacts(run_dir)
    state = read_state(run_dir)
    if suite_path is None:
        suite_path = Path(state.suite_path)
    try:
        cfg = load_config(config_path)
        suite = load_jsonl(suite_path)
    except (ConfigError, SuiteError) as exc:
        raise BundleError(str(exc)) from exc

    project_slug = project or cfg.project
    if not project_slug:
        raise BundleError(
            "hosted project is required; pass --project or set `project` in evalshift.yaml"
        )

    if state.status != "completed":
        raise BundleError(f"run {run_id} is {state.status!r}; only completed runs can be bundled")
    calls = list(iter_calls(run_dir))
    scores = _read_scores(run_dir)
    analysis_raw = _read_json(run_dir / ANALYSIS_FILENAME)

    # Everything downstream describes *this* run, so the evaluator set is the
    # one that scored it: the suite's own block where it declares one.
    resolved_evaluators = cfg.evaluators_for(state.suite_name)
    evaluator_config = _evaluator_config_snapshot(cfg, resolved_evaluators)
    dataset_snapshot = _dataset_snapshot(suite_path, suite)
    examples = _build_examples(
        suite=suite,
        calls=calls,
        scores=scores,
        tool_evaluator_names=resolved_evaluators.tool_evaluator_names,
    )
    aggregate = _build_aggregate(
        examples, state_started=state.started_at, state_ended=state.last_checkpoint_at
    )
    analysis = _build_analysis(analysis_raw)
    comparisons = [ComparisonResult(**c) for c in analysis_raw.get("comparisons", [])]
    if cfg.migration_policy is not None:
        decision = evaluate_migration_policy(
            run_id=run_id,
            source_model=state.models.source,
            target_model=state.models.target,
            policy=cfg.migration_policy,
            comparisons=comparisons,
            records=scores,
            calls=calls,
            dropped_params=state.dropped_params,
        )
    else:
        decision = inconclusive_decision(
            run_id=run_id,
            source_model=state.models.source,
            target_model=state.models.target,
            comparisons=comparisons,
            records=scores,
            calls=calls,
        )
    git = _git_metadata(env or os.environ)

    created_at = _format_z(state.started_at)
    suite_slug = derive_suite_slug(suite_name=suite_name, suite_path=suite_path)
    if not suite_slug:
        # The server requires a suite name. Failing here beats uploading a
        # bundle that finalize will reject.
        raise BundleError("could not derive a suite name; pass --suite-name or use a named suite")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "project_slug": project_slug,
        "source_model": state.models.source,
        "target_model": state.models.target,
        "suite_name": suite_slug,
        "git_sha": git["git_sha"],
        "branch": git["branch"],
        "pr_number": git["pr_number"],
        "suite_path": str(suite_path),
        "eval_config_hash": compute_hash(evaluator_config),
        "dataset_hash": compute_hash(dataset_snapshot),
        "created_at": created_at,
        "cli_version": _cli_version(),
    }
    # Run-level, not per-prompt: every call in the run rolls up here.
    # ``reports/json.py`` calls the same builder per prompt for its
    # per-prompt section, and that stays as it is.
    economics = build_economics(calls)
    bundle: dict[str, Any] = {
        "manifest": manifest,
        "aggregate": aggregate,
        "examples": examples,
        "analysis": analysis,
        "decision": decision.to_dict(),
        "economics": {
            "source": role_economics_to_dict(economics.source),
            "target": role_economics_to_dict(economics.target),
        },
        "methodology_notes": methodology_notes(state),
        # Nullable, and deliberately not in ``_require_artifacts``: a run with
        # no narrative (``--no-insights``, no API key) is a valid run. The
        # cache envelope's ``config_hash`` stays on disk.
        "insights": read_bundle_insight(run_dir),
        # The manifest hashes above are computed over the *full* snapshots;
        # what ships is the wire form of each — suite examples and inline
        # prompt bodies replaced by content hashes. See the wire builders.
        "evaluator_config": _wire_evaluator_config(evaluator_config),
        "dataset_snapshot": _wire_dataset_snapshot(dataset_snapshot),
    }
    # ``mtime=0`` is what makes the bytes reproducible across builds.
    compressed = gzip.compress(canonical_json(bundle), compresslevel=9, mtime=0)
    validate_bundle(bundle)

    target = output or run_dir / BUNDLE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(compressed)
    return BundleBuildResult(
        run_id=run_id,
        path=target,
        manifest=dict(bundle["manifest"]),
        size_bytes=len(compressed),
    )


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a gzipped hosted bundle from disk."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"failed to read bundle {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BundleError(f"bundle {path} must contain a JSON object")
    return data


def _require_artifacts(run_dir: Path) -> None:
    required = [
        STATE_FILENAME,
        RAW_FILENAME,
        SCORES_FILENAME,
        ANALYSIS_FILENAME,
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise BundleError(f"missing run artifact(s) in {run_dir}: {', '.join(missing)}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_scores(run_dir: Path) -> list[EvalRecord]:
    scores_path = run_dir / SCORES_FILENAME
    scores: list[EvalRecord] = []
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            scores.append(EvalRecord.model_validate_json(line))
    return scores


def _evaluator_config_snapshot(
    cfg: EvalShiftConfig,
    evaluators: EvaluatorsConfig,
) -> dict[str, Any]:
    """Snapshot the config that produced a run, with its resolved evaluators.

    ``evaluators`` is passed in already resolved rather than read off ``cfg``:
    a run scored under a per-suite block must not be published alongside the
    top-level one it did not use — ``eval_config_hash`` is derived from this
    dict, so two suites of the same project are meant to hash differently.
    """
    dumped = cfg.model_dump(mode="json")
    evaluators_dumped = evaluators.model_dump(mode="json")
    # Added in 1.3 with an empty default. Hashing the empty list would move
    # ``eval_config_hash`` for every config that never wrote the key and
    # orphan every hosted baseline -- the same reason ``slices`` below is kept.
    if not evaluators_dumped.get("trace_invariants"):
        evaluators_dumped.pop("trace_invariants", None)
    return {
        "version": dumped["version"],
        "prompts": dumped["prompts"],
        "defaults": dumped["defaults"],
        "evaluators": evaluators_dumped,
        # Legacy constant. This key used to carry the top-level ``slices:``
        # block, which was removed from evalshift.yaml because nothing ever
        # read it; a config that still sets it no longer loads. The key stays,
        # always ``[]``, because ``eval_config_hash`` is computed over this
        # whole dict and the server pairs a run with its baseline only on an
        # equal hash: dropping it would change the hash of every config --
        # including the ones that never set ``slices:``, for which ``[]`` is
        # exactly what was hashed before -- and orphan every hosted baseline.
        "slices": [],
    }


def _dataset_snapshot(path: Path, suite: Suite) -> dict[str, Any]:
    """The full dataset snapshot: what ``dataset_hash`` is computed over.

    Never shipped — ``_wire_dataset_snapshot`` is what goes in the bundle.
    ``examples`` must stay in the hashed form even so: the hash is the
    comparability key for diffs and baselines, and hashing metadata alone
    would file two suites that differ only in example content as identical.
    """
    tags = sorted({tag for example in suite.examples for tag in example.tags})
    return {
        "suite_path": str(path),
        "size": len(suite.examples),
        "slices": tags,
        "examples": [example.model_dump(mode="json") for example in suite.examples],
    }


def _wire_dataset_snapshot(full: dict[str, Any]) -> dict[str, Any]:
    """The ``dataset_snapshot`` that goes on the wire: metadata, no content.

    The suite's examples — conversation histories and system prompts included —
    stay local: the server writes no table row from this block and the web app
    renders nothing from it, so uploading them bought exposure and nothing
    else. ``manifest.dataset_hash`` is still computed over the full snapshot,
    so its value for a given suite is unchanged across CLI versions and runs
    pushed before this change still diff as ``direct`` against new ones.
    ``examples_hash`` carries the content fingerprint at row-array granularity
    for producers that want to verify a local suite against a pushed run.
    """
    return {
        "suite_path": full["suite_path"],
        "size": full["size"],
        "slices": full["slices"],
        "examples_hash": compute_hash(full["examples"]),
    }


def _wire_evaluator_config(full: dict[str, Any]) -> dict[str, Any]:
    """The ``evaluator_config`` that goes on the wire: prompt bodies hashed out.

    Same contract as ``_wire_dataset_snapshot``: ``eval_config_hash`` is
    computed over the full snapshot, inline prompt bodies included, and each
    non-null ``prompts[].content`` ships as a ``content_hash`` instead of the
    text. Everything else — evaluators, defaults, the legacy empty ``slices``
    — is methodology, not content, and ships as-is. A ``python_string``
    prompt's body never entered the config at all (its dump carries ``path`` +
    ``variable``), so only ``manual`` prompts have anything to strip.
    """
    prompts: list[dict[str, Any]] = []
    for prompt in full["prompts"]:
        entry = dict(prompt)
        content = entry.get("content")
        if content is not None:
            entry["content"] = None
            entry["content_hash"] = compute_hash(content)
        prompts.append(entry)
    wire = dict(full)
    wire["prompts"] = prompts
    return wire


def _cli_version() -> str:
    """Installed ``evalshift`` version, or ``"0.0.0"`` when it cannot be read.

    Stamped on the manifest so the web app can tell "this run recorded no
    output" from "the CLI that produced this run could not record output".
    """
    try:
        return version("evalshift")
    except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
        return "0.0.0"


def _build_examples(
    *,
    suite: Suite,
    calls: list[Call],
    scores: list[EvalRecord],
    tool_evaluator_names: frozenset[str],
) -> list[dict[str, Any]]:
    """Assemble the bundle's ``examples[]`` rows.

    Delta conventions match ``reports/json.py::_build_example_rows`` exactly —
    target minus source, with latency forced to 0 and flagged as
    incomparable whenever either side replayed any round from cache
    (``Call.latency_replayed``). ``cached_rounds`` itself is not uploaded.
    """
    # One output per (prompt, example, role) — sample 0 on a repeated-sampling
    # run — so a later sample never overwrites the one shown.
    calls_by_pair: dict[tuple[str, str], dict[str, Call]] = defaultdict(dict)
    for call in representative_calls(calls):
        calls_by_pair[(call.prompt_id, call.example_id)][call.role] = call
    scores_by_pair: dict[tuple[str, str], list[EvalRecord]] = defaultdict(list)
    for score in scores:
        scores_by_pair[(score.prompt_id, score.example_id)].append(score)
    suite_by_id = {example.id: example for example in suite.examples}

    rows: list[dict[str, Any]] = []
    for prompt_id, example_id in sorted(calls_by_pair):
        pair = calls_by_pair[(prompt_id, example_id)]
        source = pair.get("source")
        target = pair.get("target")
        example = suite_by_id.get(example_id)
        pair_scores = scores_by_pair.get((prompt_id, example_id), [])
        errors = [item.error for item in (source, target) if item is not None and item.error]
        errors.extend(score.error for score in pair_scores if score.error)
        scored = [item for item in pair_scores if item.error is None]
        passed = _passed(pair_scores, errors=errors)
        score_values = [item.target_score for item in scored]
        row_score = sum(score_values) / len(score_values) if score_values else None
        worst = min((item.delta for item in scored), default=None)
        latency_comparable = (
            source is not None
            and target is not None
            and not source.latency_replayed
            and not target.latency_replayed
        )
        delta_latency = (
            target.latency_ms - source.latency_ms
            if latency_comparable and source is not None and target is not None
            else 0
        )
        traces = [
            stream
            for stream in (
                from_tool_trace(source.trace if source else None, side="source"),
                from_tool_trace(target.trace if target else None, side="target"),
            )
            if stream is not None
        ]
        rows.append(
            {
                "prompt_id": prompt_id,
                "example_id": example_id,
                "slice": _slice_name(example),
                "tags": list(example.tags) if example is not None else [],
                "input": canonical_json(
                    example.inputs if example else {"example_id": example_id}
                ).decode("utf-8"),
                "expected": _expected_text(example),
                "source_output": source.text if source else None,
                "target_output": target.text if target else None,
                "traces": traces,
                "error": "; ".join(errors) if errors else None,
                "passed": passed,
                "score": row_score,
                "worst_delta_score": worst,
                "cost_usd_source": source.cost_usd if source else None,
                "cost_usd_target": target.cost_usd if target else None,
                "delta_cost_usd": (
                    target.cost_usd - source.cost_usd
                    if source is not None and target is not None
                    else None
                ),
                "latency_ms_source": source.latency_ms if source else None,
                "latency_ms_target": target.latency_ms if target else None,
                "delta_latency_ms": delta_latency,
                "latency_comparable": latency_comparable,
                "truncated": bool(
                    (source is not None and source.truncated)
                    or (target is not None and target.truncated)
                ),
                "target_empty_output": target is not None and is_empty_output(target),
                "tool_match": _tool_match(scored, tool_evaluator_names),
                "turn_index": example.turn_index if example is not None else None,
                "scores": [
                    {
                        "evaluator_name": item.evaluator_name,
                        # The server keys score rows on (evaluator_name, kind):
                        # one evaluator can measure two axes — conformance and
                        # divergence — under the same user-chosen name.
                        "kind": item.kind,
                        "source_score": item.source_score,
                        "target_score": item.target_score,
                        "delta": item.delta,
                        "error": item.error,
                        # Added 2026-10-08; hosted EvalShift accepts both since
                        # its 2026-10 deploy that added the columns. Everything
                        # else in ``metadata`` stays local — see
                        # ``_score_violations``.
                        "explanation": item.explanation or None,
                        "violations": _score_violations(item),
                    }
                    for item in pair_scores
                ],
            }
        )
    return rows


def _passed(pair_scores: list[EvalRecord], *, errors: list[str]) -> bool:
    """Whether this pair came through every measurement without losing ground.

    ``all()`` over an empty list is ``True``, so a pair *nothing* measured
    used to arrive here as a pass. That was harmless while every evaluator
    wrote a row for every pair; now that an evaluator with nothing to compare
    writes nothing at all, an example can reach the bundle with zero scores —
    and "no evidence" reporting as "passed" is exactly the silence-reads-as-
    success defect this work exists to remove.

    ``Example.passed`` is a required non-nullable boolean in the server's
    manifest schema, so "unknown" cannot be expressed here; ``False`` is the
    honest half of what is available, and the row's ``score``,
    ``worst_delta_score`` and ``scores`` are all empty beside it to say why.

    Both tool-selection axes count, and neither excuses the other: a clean
    conformance row does not absolve a divergence regression.
    """
    if errors or not pair_scores:
        return False
    return all(score.delta >= 0 for score in pair_scores)


def _tool_match(
    scored: list[EvalRecord],
    tool_evaluator_names: frozenset[str],
) -> bool | None:
    """Whether the target held every tool measurement the source held.

    ``None`` when no tool evaluator scored it at all — an absent signal, not
    a failed one. Mirrors ``reports/json.py::_tool_match``, including why it
    is a *signed* test: ``all(target_score >= 1.0)`` read as one question
    while ``tool_selection`` emitted one row per example and silently became
    "conform to the recorded ground truth *and* match the source" once it
    emitted two, forcing ✗ onto every pair of a captures-promoted suite whose
    ground truth both models fail.
    """
    tool_records = [item for item in scored if item.evaluator_name in tool_evaluator_names]
    if not tool_records:
        return None
    return all(item.delta >= 0 for item in tool_records)


#: The fields a ``violations`` entry may carry on the wire — an allowlist, not
#: the metadata keys the lists are read from. The server's ``ScoreViolation``
#: model is ``extra="forbid"``, so anything else the evaluator ever adds to
#: ``Violation.to_dict()`` must be dropped here, not discovered as a 422 after
#: the bundle has uploaded.
_VIOLATION_WIRE_FIELDS = ("rule_id", "rule_type", "tool", "round_index", "detail", "sample")


def _score_violations(record: EvalRecord) -> dict[str, list[dict[str, Any]]] | None:
    """Which rules each side broke, for a ``trace_invariants`` row that measured.

    Read off the ``source_violations`` / ``target_violations`` metadata the
    evaluator wrote (``Violation.to_dict()``, plus a ``sample`` ordinal on a
    repeated-sampling run). ``None`` on every other evaluator and on an errored
    row — an errored row has no verdict, and an empty ``target`` list would read
    as a clean one. ``rules_checked``, ``owner`` and ``failure_categories``
    stay local: the first two ship in ``evaluator_config``, the third in
    ``decision``.
    """
    if record.kind != KIND_INVARIANTS or record.error is not None:
        return None

    def entries(key: str) -> list[dict[str, Any]]:
        raw = record.metadata.get(key) or []
        return [
            {field: item[field] for field in _VIOLATION_WIRE_FIELDS if field in item}
            for item in raw
            if isinstance(item, dict)
        ]

    return {"source": entries("source_violations"), "target": entries("target_violations")}


def _slice_name(example: SuiteExample | None) -> str | None:
    if example is None or not example.tags:
        return None
    return example.tags[0]


def _expected_text(example: SuiteExample | None) -> str | None:
    if example is None or example.expected is None:
        return None
    return canonical_json(example.expected).decode("utf-8")


def _build_analysis(analysis_raw: dict[str, Any]) -> dict[str, Any]:
    """Reshape ``analysis.json`` into the bundle's ``analysis`` block.

    No statistics are recomputed; the already-emitted comparisons and per-slice
    aggregates are carried verbatim so the server never re-derives them.
    """
    return {
        "n_records": analysis_raw.get("n_records", 0),
        "n_examples": analysis_raw.get("n_examples", 0),
        "comparisons": analysis_raw.get("comparisons", []),
        "slice_aggregates": analysis_raw.get("aggregates", {}),
    }


def _build_aggregate(
    examples: list[dict[str, Any]],
    *,
    state_started: datetime,
    state_ended: datetime | None,
) -> dict[str, Any]:
    total = len(examples)
    passed = sum(1 for example in examples if example["passed"])
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        if example.get("slice"):
            by_slice[str(example["slice"])].append(example)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "duration_seconds": _duration_seconds(state_started, state_ended),
        "slices": [
            {
                "name": name,
                "total": len(rows),
                "passed": sum(1 for row in rows if row["passed"]),
                "pass_rate": (sum(1 for row in rows if row["passed"]) / len(rows) if rows else 0.0),
            }
            for name, rows in sorted(by_slice.items())
        ],
        **_cost_latency_rollup(examples),
    }


def _cost_latency_rollup(examples: list[dict[str, Any]]) -> dict[str, float]:
    """Summarise per-example cost/latency upstream so the server never aggregates."""
    cost_source = _metric_values(examples, "cost_usd_source")
    cost_target = _metric_values(examples, "cost_usd_target")
    lat_source = _metric_values(examples, "latency_ms_source")
    lat_target = _metric_values(examples, "latency_ms_target")
    cost_s = float(sum(cost_source))
    cost_t = float(sum(cost_target))
    lat_s_p50 = _percentile(lat_source, 50)
    lat_s_p95 = _percentile(lat_source, 95)
    lat_t_p50 = _percentile(lat_target, 50)
    lat_t_p95 = _percentile(lat_target, 95)
    return {
        "cost_usd_source": cost_s,
        "cost_usd_target": cost_t,
        "cost_usd_delta": cost_t - cost_s,
        "latency_ms_source_p50": lat_s_p50,
        "latency_ms_source_p95": lat_s_p95,
        "latency_ms_target_p50": lat_t_p50,
        "latency_ms_target_p95": lat_t_p95,
        "latency_ms_p50_delta": lat_t_p50 - lat_s_p50,
        "latency_ms_p95_delta": lat_t_p95 - lat_s_p95,
    }


def _metric_values(examples: list[dict[str, Any]], key: str) -> list[float]:
    """Read one cost/latency figure off every row that has it.

    Reads the row directly: these live at the top level of an example since
    the ``metrics`` sub-dict was flattened. The server requires all nine
    aggregate figures as non-null numbers, so reading a key that no longer
    exists would emit a schema-valid bundle full of zeros.
    """
    values: list[float] = []
    for example in examples:
        value = example.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; ``0.0`` for an empty series."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return float(ordered[rank - 1])


def _duration_seconds(started: datetime, ended: datetime | None) -> float:
    if ended is None:
        return 0.0
    return max(0.0, (ended - started).total_seconds())


def _git_metadata(env: Mapping[str, str]) -> dict[str, Any]:
    sha = (env.get("GITHUB_SHA") or _git("rev-parse", "HEAD") or "").lower()
    if not _GIT_SHA_RE.match(sha):
        raise BundleError(
            "could not determine a valid 40-character git SHA; run inside git or set GITHUB_SHA",
        )
    branch = (
        env.get("GITHUB_HEAD_REF")
        or env.get("GITHUB_REF_NAME")
        or _git("rev-parse", "--abbrev-ref", "HEAD")
        or "local"
    )
    return {"git_sha": sha, "branch": branch, "pr_number": _github_pr_number(env)}


def _github_pr_number(env: Mapping[str, str]) -> int | None:
    ref = env.get("GITHUB_REF", "")
    match = re.match(r"refs/pull/([1-9][0-9]*)/", ref)
    if match:
        return int(match.group(1))
    event_path = env.get("GITHUB_EVENT_PATH")
    if event_path:
        try:
            payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        number = payload.get("number") if isinstance(payload, dict) else None
        return number if isinstance(number, int) and number > 0 else None
    return None


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _format_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def validate_bundle(bundle: dict[str, Any]) -> None:
    """Validate a decoded bundle against the vendored server schema.

    Public because ``push`` runs it too: a bundle handed to ``--bundle`` may
    have been built by an older CLI, by another tool, or edited by hand, and the
    server's rejection would otherwise arrive only after a full upload.

    Raises:
        BundleError: If the bundle does not satisfy the schema.
    """
    schema_path = resources.files("evalshift_cli.hosted").joinpath("bundle_manifest.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        Draft202012Validator(schema["bundle"]).validate(bundle)
    except JsonSchemaValidationError as exc:
        raise BundleError(f"bundle failed schema validation: {exc.message}") from exc


__all__ = [
    "BUNDLE_FILENAME",
    "BundleBuildResult",
    "BundleError",
    "build_bundle",
    "canonical_json",
    "compute_hash",
    "load_bundle",
    "validate_bundle",
]
