"""Report-data assembly: stitches together everything Phase 7 needs.

Reads the artefacts produced by earlier phases:

* ``state.json`` — run metadata (Phase 4).
* ``raw.jsonl`` — per-call records (Phase 4).
* ``scores.jsonl`` — per (pair, evaluator) records (Phase 5).
* ``analysis.json`` — statistical comparisons (Phase 6).

…and produces a single :class:`ReportData` payload that the HTML
renderer consumes. Also writes a ``report.json`` artefact alongside it
so external tools can read the report data without parsing HTML.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evalshift_cli.analysis.policy import is_shared_ground_truth_miss
from evalshift_cli.analysis.statistics import ADVISORY_NOTE_PREFIX, AXIS_NOTE_PREFIX
from evalshift_cli.cli.commands.analyze import ANALYSIS_FILENAME, MIGRATION_DECISION_FILENAME
from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.evaluators.tool_selection import KIND_DIVERGENCE
from evalshift_cli.models.family import judge_family_overlaps
from evalshift_cli.reports.economics import (
    PromptEconomics,
    RoleEconomics,
    build_economics,
    is_empty_output,
    methodology_notes,
    role_economics_to_dict,
)
from evalshift_cli.runner.checkpoint import iter_calls, read_state
from evalshift_cli.runner.models import Call, representative_calls
from evalshift_cli.suite.loader import SuiteError, load_jsonl
from evalshift_cli.suite.models import Suite, SuiteExample
from evalshift_cli.traces.diff import TraceDiff, diff_traces
from evalshift_cli.traces.loader import (
    TRACES_FILENAME,
    TraceKey,
    TraceLoadError,
    index_traces,
    load_traces_jsonl,
)
from evalshift_cli.traces.models import AgentTrace

REPORT_JSON_FILENAME: str = "report.json"

#: Prefix of the display note marking an aggregate row whose every
#: measurement was a shared ground-truth miss. Written here rather than in
#: ``analysis`` because it is a *rendering* decision — the statistics are
#: right, the headline they earn is not: a ``tool_selection.conformance``
#: comparison over ten ``0.0 / 0.0`` rows is a true zero delta, and calling
#: it "Equivalent — no meaningful difference between models" one line under a
#: critical regression on the same evaluator name is how this run shipped.
#: Rides in ``notes`` on the same rule as ``UNMEASURED_NOTE_PREFIX`` and
#: ``AXIS_NOTE_PREFIX``.
SHARED_GROUND_TRUTH_NOTE_PREFIX: str = "shared ground-truth miss:"

#: Metadata key pairs the tool-selection axes record their names under, in
#: the order they are looked for. ``exact``/``expected``/``expected_set``
#: write ordered name lists, ``set`` writes sorted sets, ``first`` writes a
#: single name per side. One of the three is always present on a
#: tool-selection record, and the report reads whichever it finds rather
#: than re-deriving names from the call trace: the trace is a positional
#: diff and the record is what was actually scored.
_TOOL_NAME_KEYS: tuple[tuple[str, str], ...] = (
    ("source_names", "target_names"),
    ("source_set", "target_set"),
    ("source_first", "target_first"),
)


@dataclass(frozen=True, slots=True)
class RoundToolChange:
    """One round of a teacher-forced replay, as the evaluator scored it.

    ``round`` is zero-based, matching the evaluator's metadata; the report
    labels it ``r1``, ``r2``, … because a user counts rounds from one.
    """

    round: int
    source_names: list[str]
    target_names: list[str]
    expected_names: list[str]

    @property
    def diverged(self) -> bool:
        """Whether the two sides called different tools in this round."""
        return self.source_names != self.target_names


@dataclass(frozen=True, slots=True)
class ToolChange:
    """The tools each side called on one example, as the evaluator saw them.

    The report's whole finding on a divergence regression: "the target called
    something else" is unactionable without the two names. Read off the
    evaluator's record metadata rather than the call trace so the names shown
    are the names scored.
    """

    source_names: list[str]
    target_names: list[str]
    # Populated only for a teacher-forced multi-round replay, from the
    # evaluator's ``metadata["rounds"]``. ``None`` — never ``[]`` — for a
    # single-shot pair, so the report keeps rendering it as one line.
    rounds: list[RoundToolChange] | None = None

    @property
    def diverged(self) -> bool:
        """Whether the two sides called different tools."""
        return self.source_names != self.target_names


@dataclass(frozen=True, slots=True)
class ToolDiff:
    """One tool-call diff item for a top regression."""

    kind: str
    tool_name: str
    message: str
    source_arguments: dict[str, Any] | None = None
    target_arguments: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TopRegression:
    """One concrete (prompt, example, evaluator) regression worth showing."""

    prompt_id: str
    example_id: str
    evaluator_name: str
    delta: float
    source_text: str
    target_text: str
    # The axis slug this record came from (``EvalRecord.kind``). One
    # evaluator name can score two of them — a conformance failure and a
    # divergence regression are different findings and must not render as
    # the same card with the same label.
    kind: str = ""
    # Populated on a tool-selection regression, from the record's own
    # metadata. See :class:`ToolChange`.
    tool_change: ToolChange | None = None
    # Why it was flagged: the two scores for this pair plus whatever the
    # evaluator wrote (llm_judge rationale, or metadata like raw_cosine).
    source_score: float = 0.0
    target_score: float = 0.0
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # v0.2 — populated when the regression is on a tool evaluator. The
    # HTML report renders these as a side-by-side trace diff in place
    # of the source/target text panes.
    source_trace: ToolTrace | None = None
    target_trace: ToolTrace | None = None
    tool_diffs: list[ToolDiff] = field(default_factory=list)
    # CLI Phase 2 — imported bring-your-own-agent trace support.
    source_agent_trace: AgentTrace | None = None
    target_agent_trace: AgentTrace | None = None
    trace_diff: TraceDiff | None = None
    # True when either side's output was cut off at the token cap. Such
    # pairs are excluded from the statistics upstream (error-marked), so
    # this flag is a belt-and-suspenders annotation for the report.
    truncated: bool = False
    # True when the target returned empty text despite spending output
    # tokens with no error and a "stop" finish reason (a thinking-only
    # response). Tool-call responses are excluded — see `_is_empty_output`.
    # Unlike `truncated`, this is NOT excluded from statistics — an empty
    # answer is a genuine regression signal — but the report flags it so
    # readers don't mistake it for a scoring bug.
    target_empty_output: bool = False
    # Multi-turn — the conversation prefix (as plain dicts, one per
    # ChatMessage) that preceded this example's current turn, and its
    # zero-based position within that conversation. Both are ``None`` for
    # single-turn examples or when the example can't be found in the suite.
    history: list[dict[str, Any]] | None = None
    turn_index: int | None = None
    # The example's rendered input (template variables), so a regression can
    # be reviewed against what the models were actually asked. Truncated at
    # ``INPUT_TEXT_MAX_CHARS``; ``None`` when the example has no input vars
    # or can't be found in the suite.
    input_text: str | None = None


@dataclass(frozen=True, slots=True)
class ExampleRow:
    """One row in the per-example breakdown table.

    Captures only deltas (target − source) — the report is a migration
    diff, so absolute values per role are surfaced upstream in
    ``PromptEconomics``.
    """

    example_id: str
    tags: list[str]
    delta_latency_ms: int
    latency_comparable: bool  # False when either side replayed a round from cache (delta 0)
    delta_cost_usd: float
    worst_delta_score: float | None
    # Whether the target held every tool measurement the source did — i.e.
    # no tool evaluator scored it *below* the source. ``None`` when no tool
    # evaluator scored this example. See :func:`_tool_match`.
    tool_match: bool | None
    # Which tools each side called, when a divergence axis scored this
    # example. This column is the plan's table: one line per example, source
    # called / target called, for every example rather than the worst five.
    tool_change: ToolChange | None = None
    # Multi-turn — zero-based turn position within the source conversation.
    # None for single-turn examples or when the example isn't in the suite.
    turn_index: int | None = None


@dataclass(frozen=True, slots=True)
class InvariantViolationRow:
    """One trace rule the target broke on one example.

    Listed whatever the delta: a rule both models broke scores ``delta == 0``
    and would never surface among the top regressions, yet a *blocking* entry
    still counts against ``max_invariant_violations`` under a policy.
    ``blocking`` is the record's flag: ``False`` rows come from a
    ``blocking: false`` entry and are reported but never gate.
    """

    prompt_id: str
    example_id: str
    evaluator_name: str
    owner: str | None
    rule_id: str
    rule_type: str
    tool: str
    round_index: int
    detail: str
    source_also: bool
    blocking: bool
    # Zero-based ordinal among the scored samples, set only on a row reduced
    # over ``samples_per_example > 1``: the same rule then appears once per
    # sample that broke it. ``None`` on a single-sample run.
    sample: int | None = None


@dataclass(frozen=True, slots=True)
class PromptSection:
    """Per-prompt slice of the report payload."""

    prompt_id: str
    aggregate_rows: list[dict[str, Any]]  # serialised SliceAggregate per evaluator
    slice_rows: list[dict[str, Any]]  # serialised ComparisonResult rows
    top_regressions: list[TopRegression]
    economics: PromptEconomics
    example_rows: list[ExampleRow]
    # v0.2 — when at least one Call has a populated trace, render a
    # "Tool Trace Comparison" section with example-level diffs.
    has_tool_traces: bool = False


@dataclass(frozen=True, slots=True)
class ReportData:
    """Everything the HTML template needs in one place."""

    run_id: str
    started_at: str
    source_model: str
    target_model: str
    suite_path: str
    n_examples: int
    n_calls: int
    cached_calls: int
    failed_calls: int
    truncated_calls: int
    total_cost_usd: float

    executive_summary: list[dict[str, Any]] = field(default_factory=list)
    migration_decision: dict[str, Any] | None = None
    prompt_sections: list[PromptSection] = field(default_factory=list)
    methodology_notes: list[str] = field(default_factory=list)
    # Models whose sampling is non-deterministic; drives the report banner.
    non_deterministic_models: list[str] = field(default_factory=list)
    # Canonical model id -> generation parameters LiteLLM dropped on that arm
    # because the model does not accept them; drives the second banner.
    dropped_params: dict[str, list[str]] = field(default_factory=dict)
    # ``defaults.samples_per_example`` the run was made with: how many
    # ``raw.jsonl`` rows each (prompt, example, role) has. Every example row
    # below shows sample 0; the totals above count every sample.
    samples_per_example: int = 1
    # Judges that scored this run and share a provider with an arm, each
    # ``{"judge_model", "provider", "roles"}`` — drives the one-line
    # self-preference note under the banners. Empty when no judge overlaps,
    # no judge contributed a score, or the report ran without a config.
    judge_family_overlap: list[dict[str, Any]] = field(default_factory=list)
    # Every trace rule the target broke, one row per violation, whatever the
    # delta. Empty when no ``trace_invariants`` evaluator scored the run.
    invariant_violations: list[InvariantViolationRow] = field(default_factory=list)


def build_report_payload(
    run_dir: Path,
    *,
    tool_evaluator_names: frozenset[str] = frozenset(),
    judge_models: Mapping[str, str] | None = None,
) -> ReportData:
    """Assemble :class:`ReportData` from the artefacts in ``run_dir``.

    Args:
        run_dir: The completed run directory.
        tool_evaluator_names: Names of evaluators that score tool-call
            correctness (configured under ``tool_selection`` /
            ``tool_arguments`` / ``tool_trace_structure`` in
            evalshift.yaml). Used to populate the per-example
            ``tool_match`` flag. Empty set is fine — every example's
            ``tool_match`` will then be ``None``.
        judge_models: Configured ``llm_judge`` evaluator name →
            ``judge_model``, for the suite the run was scored with. Only
            judges that actually wrote a ``scores.jsonl`` row are checked
            for a family overlap with the arms. ``None`` (no config) leaves
            ``judge_family_overlap`` empty.

    Raises:
        FileNotFoundError: If any required artefact is missing. The
            caller (``evalshift report``) is expected to surface a
            helpful error message.
    """
    state = read_state(run_dir)
    analysis = _read_analysis(run_dir)
    migration_decision = _read_migration_decision(run_dir)
    scores = _read_scores(run_dir)
    calls = list(iter_calls(run_dir))
    suite = _load_suite(state.suite_path)
    agent_traces = _read_agent_traces(run_dir)

    cached = sum(1 for c in calls if c.cached)
    failed = sum(1 for c in calls if c.error is not None)
    truncated = sum(1 for c in calls if c.truncated)
    cost = sum(c.cost_usd for c in calls)

    advisory = _advisory_evaluator_names(analysis, scores)
    summary = _build_executive_summary(analysis, advisory)
    sections = _build_prompt_sections(
        analysis,
        scores,
        calls,
        suite,
        tool_evaluator_names,
        agent_traces,
        advisory=advisory,
    )

    return ReportData(
        run_id=state.run_id,
        started_at=state.started_at.isoformat(),
        source_model=state.models.source,
        target_model=state.models.target,
        suite_path=state.suite_path,
        n_examples=_n_examples(calls),
        n_calls=len(calls),
        cached_calls=cached,
        failed_calls=failed,
        truncated_calls=truncated,
        total_cost_usd=cost,
        executive_summary=summary,
        migration_decision=migration_decision,
        prompt_sections=sections,
        methodology_notes=methodology_notes(state),
        non_deterministic_models=list(state.non_deterministic_models),
        dropped_params={k: list(v) for k, v in state.dropped_params.items()},
        samples_per_example=state.samples_per_example,
        judge_family_overlap=_judge_family_overlap(
            scores,
            judge_models or {},
            source_model=state.models.source,
            target_model=state.models.target,
        ),
        invariant_violations=_invariant_violations(scores),
    )


def _invariant_violations(scores: list[EvalRecord]) -> list[InvariantViolationRow]:
    """One row per trace rule the target broke, read off ``trace_invariants`` records.

    ``source_also`` is whether the source broke the same rule (by ``rule_id``,
    on any sample) on the same example: a rule both sides break is more
    likely a rule to review than a migration regression. Errored records
    carry no verdict and list nothing.
    """
    rows: list[InvariantViolationRow] = []
    for r in scores:
        if r.kind != "trace_invariants" or r.error is not None:
            continue
        source_rules = {
            v.get("rule_id")
            for v in r.metadata.get("source_violations") or []
            if isinstance(v, dict)
        }
        owner = r.metadata.get("owner")
        for v in r.metadata.get("target_violations") or []:
            if not isinstance(v, dict):
                continue
            sample = v.get("sample")
            rows.append(
                InvariantViolationRow(
                    prompt_id=r.prompt_id,
                    example_id=r.example_id,
                    evaluator_name=r.evaluator_name,
                    owner=owner if isinstance(owner, str) else None,
                    rule_id=str(v.get("rule_id", "")),
                    rule_type=str(v.get("rule_type", "")),
                    tool=str(v.get("tool", "")),
                    round_index=int(v.get("round_index", 0)),
                    detail=str(v.get("detail", "")),
                    source_also=v.get("rule_id") in source_rules,
                    blocking=r.blocking,
                    sample=sample if isinstance(sample, int) else None,
                ),
            )
    return sorted(
        rows,
        key=lambda row: (
            row.prompt_id,
            row.example_id,
            row.rule_id,
            row.round_index,
            -1 if row.sample is None else row.sample,
        ),
    )


def _judge_family_overlap(
    scores: list[EvalRecord],
    judge_models: Mapping[str, str],
    *,
    source_model: str,
    target_model: str,
) -> list[dict[str, Any]]:
    """Family overlaps for the judges that contributed at least one row.

    Selected on ``kind == "llm_judge"``, falling back to the legacy
    ``llm_judge.`` name prefix for rows checkpointed before ``kind`` existed.
    A judge that is configured but scored nothing (``applies_to`` matched no
    prompt, or the criterion post-dates the run) has no verdict to caveat.
    """
    contributed = {
        r.evaluator_name
        for r in scores
        if r.kind == "llm_judge" or (not r.kind and r.evaluator_name.startswith("llm_judge."))
    }
    judges = [judge_models[name] for name in sorted(contributed) if name in judge_models]
    return [
        {"judge_model": o.judge_model, "provider": o.provider, "roles": list(o.roles)}
        for o in judge_family_overlaps(
            judge_models=judges, source_model=source_model, target_model=target_model
        )
    ]


def write_report_json(report: ReportData, run_dir: Path) -> Path:
    """Persist the report payload as ``report.json``."""
    out_path = run_dir / REPORT_JSON_FILENAME
    out_path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_analysis(run_dir: Path) -> dict[str, Any]:
    path = run_dir / ANALYSIS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing {ANALYSIS_FILENAME} in {run_dir}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{ANALYSIS_FILENAME} must contain a JSON object")
    return data


def _read_migration_decision(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / MIGRATION_DECISION_FILENAME
    if not path.exists():
        return None
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{MIGRATION_DECISION_FILENAME} must contain a JSON object")
    return data


def _read_scores(run_dir: Path) -> list[EvalRecord]:
    path = run_dir / SCORES_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing {SCORES_FILENAME} in {run_dir}")
    out: list[EvalRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            out.append(EvalRecord.model_validate_json(text))
    return out


def _read_agent_traces(run_dir: Path) -> dict[TraceKey, AgentTrace]:
    path = run_dir / TRACES_FILENAME
    if not path.exists():
        return {}
    try:
        return dict(index_traces(load_traces_jsonl(path)))
    except TraceLoadError:
        return {}


#: Cap on the rendered input attached to each top-regression card. Keeps the
#: single-file report bounded when suite inputs are very large; the marker
#: points readers at the suite file for the full text.
INPUT_TEXT_MAX_CHARS: int = 8_000


def _render_input_text(inputs: dict[str, Any], *, example_id: str) -> str | None:
    """Render an example's input vars for display on a regression card.

    Single-variable examples (the replay/capture common case) render the
    value verbatim; multi-variable examples fall back to pretty-printed
    JSON. Returns ``None`` when there are no input vars, and truncates at
    :data:`INPUT_TEXT_MAX_CHARS` with a pointer to the suite file.
    """
    if not inputs:
        return None
    if len(inputs) == 1:
        text = str(next(iter(inputs.values())))
    else:
        text = json.dumps(inputs, indent=2, ensure_ascii=False, default=str)
    if len(text) > INPUT_TEXT_MAX_CHARS:
        text = (
            text[:INPUT_TEXT_MAX_CHARS]
            + f"\n… truncated — full input in the suite file (id={example_id})"
        )
    return text


def _load_suite(suite_path: str) -> Suite:
    """Load the suite referenced by RunState; tolerate failures.

    The report is read-only; if the suite has been moved/edited since the
    run, fall back to an empty Suite so we still render economics +
    statistics. Tag/expected-tools data simply won't be available.
    """
    try:
        return load_jsonl(Path(suite_path))
    except (SuiteError, FileNotFoundError, OSError):
        return Suite()


def _n_examples(calls: list[Call]) -> int:
    return len({c.example_id for c in calls})


def _advisory_evaluator_names(analysis: dict[str, Any], scores: list[EvalRecord]) -> set[str]:
    """Names of the ``blocking: false`` evaluators, as the policy gate sees them.

    From the records when there are records -- every row carries the config
    flag -- and from the ``advisory:`` note ``analyze`` stamps on the
    synthesized comparison of an advisory evaluator that scored nothing.
    Mirrors :func:`evalshift_cli.analysis.policy._advisory_evaluator_names`,
    which works on parsed comparisons rather than the raw ``analysis.json``
    rows this module renders.
    """
    names = {record.evaluator_name for record in scores if not record.blocking}
    for comparison in analysis.get("comparisons", []):
        notes = comparison.get("notes") or []
        if any(isinstance(note, str) and note.startswith(ADVISORY_NOTE_PREFIX) for note in notes):
            names.add(comparison["evaluator_name"])
    return names


def _build_executive_summary(
    analysis: dict[str, Any],
    advisory: set[str],
) -> list[dict[str, Any]]:
    """One row per prompt: the comparison that speaks for it.

    The worst severity wins; ties go to the most negative effect size, then
    the smaller corrected p-value -- the order :func:`reports.html._headline_comparison`
    and the insights use, so this table, the header's average score change
    and the hero panel beside them quote the same row rather than whichever
    evaluator ``analyze`` happened to list first. Advisory (``blocking:
    false``) evaluators are passed over while the prompt has a gating one: a
    rule the target satisfied by calling no tools at all once read as the
    prompt's "+9% improved" directly above a failed verdict.
    """
    severity_rank = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
        "improved": 4,
        "none": 5,
        "insufficient": 6,
    }

    def order(comparison: dict[str, Any]) -> tuple[int, float, float]:
        return (
            severity_rank.get(comparison["severity"], 99),
            float(comparison.get("effect_size", 0.0)),
            float(comparison.get("p_value_corrected", 1.0)),
        )

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for c in analysis["comparisons"]:
        by_prompt.setdefault(c["prompt_id"], []).append(c)
    rows: list[dict[str, Any]] = []
    for prompt in sorted(by_prompt):
        gating = [c for c in by_prompt[prompt] if c["evaluator_name"] not in advisory]
        # A prompt scored only by advisory evaluators keeps its row -- the
        # table is one line per prompt -- and that row is then the best
        # advisory one, tagged as such where it is listed in full.
        rows.append(min(gating or by_prompt[prompt], key=order))
    return rows


def _build_prompt_sections(
    analysis: dict[str, Any],
    scores: list[EvalRecord],
    calls: list[Call],
    suite: Suite,
    tool_evaluator_names: frozenset[str],
    agent_traces: dict[TraceKey, AgentTrace],
    *,
    advisory: set[str],
) -> list[PromptSection]:
    # One displayed output per (prompt, example, role): sample 0. Keyed by
    # role below, so every sample must not overwrite the one before it.
    calls = representative_calls(calls)
    by_prompt_calls: dict[tuple[str, str], dict[str, Call]] = {}
    for c in calls:
        by_prompt_calls.setdefault((c.prompt_id, c.example_id), {})[c.role] = c

    by_prompt_records: dict[str, list[EvalRecord]] = {}
    for rec in scores:
        by_prompt_records.setdefault(rec.prompt_id, []).append(rec)

    tags_by_example_id: dict[str, list[str]] = {ex.id: list(ex.tags) for ex in suite.examples}
    examples_by_id: dict[str, SuiteExample] = {ex.id: ex for ex in suite.examples}

    sections: list[PromptSection] = []
    for prompt_id in sorted(by_prompt_records):
        prompt_records = by_prompt_records[prompt_id]
        prompt_comparisons = [c for c in analysis["comparisons"] if c["prompt_id"] == prompt_id]
        # Aggregate rows: pick the implicit "all" slice rows for this prompt.
        # Only these are annotated: the slice table renders significant
        # severities only, and a shared miss is always ``none``.
        # ``advisory`` rides on the row so the evaluator table can tag a
        # ``blocking: false`` entry the way the broken-rules panel does.
        aggregates_all = [
            {**row, "advisory": row["evaluator_name"] in advisory}
            for row in _annotate_shared_misses(
                [c for c in prompt_comparisons if c["slice_name"] == "all"],
                prompt_records,
            )
        ]
        # Slice rows: every non-all comparison for this prompt.
        slice_rows = [c for c in prompt_comparisons if c["slice_name"] != "all"]
        # Top regressions: the 5 records with the most-negative delta.
        ranked = sorted(
            prompt_records,
            key=lambda r: (
                r.delta,
                0
                if (
                    (r.prompt_id, r.example_id, "source") in agent_traces
                    and (r.prompt_id, r.example_id, "target") in agent_traces
                )
                else 1,
            ),
        )[:5]
        top: list[TopRegression] = []
        for r in ranked:
            if r.delta >= 0 or r.error is not None:
                continue
            calls_for_pair = by_prompt_calls.get((r.prompt_id, r.example_id), {})
            src = calls_for_pair.get("source")
            tgt = calls_for_pair.get("target")
            source_agent_trace = agent_traces.get((r.prompt_id, r.example_id, "source"))
            target_agent_trace = agent_traces.get((r.prompt_id, r.example_id, "target"))
            trace_diff = (
                diff_traces(source_agent_trace, target_agent_trace)
                if source_agent_trace is not None and target_agent_trace is not None
                else None
            )
            example = examples_by_id.get(r.example_id)
            top.append(
                TopRegression(
                    prompt_id=r.prompt_id,
                    example_id=r.example_id,
                    evaluator_name=r.evaluator_name,
                    delta=r.delta,
                    source_text=src.text if src else "",
                    target_text=tgt.text if tgt else "",
                    kind=r.kind,
                    tool_change=_tool_change(r.metadata),
                    source_score=r.source_score,
                    target_score=r.target_score,
                    explanation=r.explanation,
                    metadata=r.metadata,
                    source_trace=src.trace if src else None,
                    target_trace=tgt.trace if tgt else None,
                    tool_diffs=_build_tool_diffs(
                        src.trace if src else None, tgt.trace if tgt else None
                    ),
                    source_agent_trace=source_agent_trace,
                    target_agent_trace=target_agent_trace,
                    trace_diff=trace_diff,
                    truncated=(bool(src and src.truncated) or bool(tgt and tgt.truncated)),
                    target_empty_output=bool(tgt and is_empty_output(tgt)),
                    history=(
                        [m.model_dump(exclude_none=True) for m in example.history]
                        if example is not None and example.history is not None
                        else None
                    ),
                    turn_index=example.turn_index if example is not None else None,
                    input_text=(
                        _render_input_text(example.inputs, example_id=r.example_id)
                        if example is not None
                        else None
                    ),
                ),
            )
        # v0.2 — does this prompt have any traces at all? Drives the
        # "Tool Trace Comparison" subsection in the HTML template.
        has_traces = any(
            (sides.get("source") and sides["source"].trace is not None)
            or (sides.get("target") and sides["target"].trace is not None)
            for (pid, _ex_id), sides in by_prompt_calls.items()
            if pid == prompt_id
        )
        prompt_calls = [c for c in calls if c.prompt_id == prompt_id]
        example_rows = _build_example_rows(
            prompt_id=prompt_id,
            calls=prompt_calls,
            records=prompt_records,
            tags_by_example_id=tags_by_example_id,
            tool_evaluator_names=tool_evaluator_names,
            examples_by_id=examples_by_id,
        )
        sections.append(
            PromptSection(
                prompt_id=prompt_id,
                aggregate_rows=aggregates_all,
                slice_rows=slice_rows,
                top_regressions=top,
                economics=build_economics(prompt_calls),
                example_rows=example_rows,
                has_tool_traces=has_traces,
            ),
        )
    return sections


def _build_example_rows(
    *,
    prompt_id: str,
    calls: list[Call],
    records: list[EvalRecord],
    tags_by_example_id: dict[str, list[str]],
    tool_evaluator_names: frozenset[str],
    examples_by_id: dict[str, SuiteExample],
) -> list[ExampleRow]:
    """Assemble one ExampleRow per example for a single prompt.

    Δ values are target − source. Latency is reported as 0 when either
    side replayed any round from the cache (`Call.latency_replayed`: its
    `latency_ms` is the original run's, not this one's) so the figure stays
    meaningful only on live × live pairs.
    """
    sides_by_example: dict[str, dict[str, Call]] = {}
    for c in calls:
        sides_by_example.setdefault(c.example_id, {})[c.role] = c

    records_by_example: dict[str, list[EvalRecord]] = {}
    for r in records:
        records_by_example.setdefault(r.example_id, []).append(r)

    rows: list[ExampleRow] = []
    for example_id in sides_by_example:
        sides = sides_by_example[example_id]
        src = sides.get("source")
        tgt = sides.get("target")
        if src is None or tgt is None:
            continue  # incomplete pair; skip rather than misreport
        latency_comparable = not (src.latency_replayed or tgt.latency_replayed)
        delta_lat = tgt.latency_ms - src.latency_ms if latency_comparable else 0
        delta_cost = tgt.cost_usd - src.cost_usd

        ex_records = records_by_example.get(example_id, [])
        scored = [r for r in ex_records if r.error is None]
        worst = min((r.delta for r in scored), default=None)

        example = examples_by_id.get(example_id)
        rows.append(
            ExampleRow(
                example_id=example_id,
                tags=tags_by_example_id.get(example_id, []),
                delta_latency_ms=delta_lat,
                latency_comparable=latency_comparable,
                delta_cost_usd=delta_cost,
                worst_delta_score=worst,
                tool_match=_tool_match(scored, tool_evaluator_names),
                tool_change=_example_tool_change(scored),
                turn_index=example.turn_index if example is not None else None,
            ),
        )

    # Worst regression first (most-negative delta score), tiebreak on
    # most-expensive (highest +delta cost), then example_id for stability.
    rows.sort(
        key=lambda r: (
            r.worst_delta_score if r.worst_delta_score is not None else 0.0,
            -r.delta_cost_usd,
            r.example_id,
        ),
    )
    return rows


def _tool_match(
    scored: list[EvalRecord],
    tool_evaluator_names: frozenset[str],
) -> bool | None:
    """Whether the target held every tool measurement the source held.

    ``None`` when no tool evaluator scored the example — an absent signal,
    not a failed one. Mirrored exactly in ``hosted/bundle.py::_tool_match``.

    Signed, not absolute. This was ``all(target_score >= 1.0)``, which read
    as one question while ``tool_selection`` emitted one row per example. It
    now emits two, against two different baselines, so the same expression
    silently became "conform to the recorded ground truth *and* match the
    source" — and on a suite promoted from captures, whose ground truth both
    models routinely fail, that forced ✗ onto every pair including the ones
    the migration left untouched. A shared ``0.0 / 0.0`` says the suite is
    wrong, not the target.

    ``delta >= 0`` asks the question the column actually sits in: this table
    is a migration diff, every other cell in it is target minus source, and
    the tool cell now agrees. A target that misses ground truth the source
    held is still ✗ — that is a negative delta on the conformance axis — and
    so is any divergence below the source's fixed 1.0.
    """
    tool_records = [r for r in scored if r.evaluator_name in tool_evaluator_names]
    if not tool_records:
        return None
    return all(r.delta >= 0 for r in tool_records)


def _tool_change(metadata: dict[str, Any]) -> ToolChange | None:
    """Read the tools each side called off one record's metadata.

    Returns ``None`` for a record that carries no tool names — every
    non-tool evaluator, and the ``expected_no_tools`` conformance mode,
    which records call *counts* rather than names.
    """
    for source_key, target_key in _TOOL_NAME_KEYS:
        if source_key in metadata or target_key in metadata:
            return ToolChange(
                source_names=_name_list(metadata.get(source_key)),
                target_names=_name_list(metadata.get(target_key)),
                rounds=_round_changes(metadata.get("rounds")),
            )
    return None


def _round_changes(value: Any) -> list[RoundToolChange] | None:
    """Read a record's per-round breakdown, or ``None`` for a single-shot pair.

    Written by the tool evaluators only when the replay spanned more than one
    round, so its absence is the signal to render the flat one-line form.
    """
    if not isinstance(value, list) or not value:
        return None
    rounds: list[RoundToolChange] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            return None
        rounds.append(
            RoundToolChange(
                round=int(entry.get("round", index)),
                source_names=_name_list(entry.get("source_names")),
                target_names=_name_list(entry.get("target_names")),
                expected_names=_name_list(entry.get("expected_names")),
            ),
        )
    return rounds


def _name_list(value: Any) -> list[str]:
    """Normalise a metadata tool-name value to a list of names."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _example_tool_change(scored: list[EvalRecord]) -> ToolChange | None:
    """The example's source-vs-target tool names, preferring the divergence axis.

    Divergence is the axis that asks the migration question, so its names are
    the ones the per-example table wants. Any other tool record that carries
    names is a fallback for a run with ``divergence: off``.
    """
    with_names = [(r, _tool_change(r.metadata)) for r in scored]
    candidates = [(r, change) for r, change in with_names if change is not None]
    for record, change in candidates:
        if record.kind == KIND_DIVERGENCE:
            return change
    return candidates[0][1] if candidates else None


def _annotate_shared_misses(
    rows: list[dict[str, Any]],
    records: list[EvalRecord],
) -> list[dict[str, Any]]:
    """Flag aggregate rows whose every measurement was a shared ground-truth miss.

    Such a row is a genuine zero delta and the statistics are not wrong; the
    *headline* is. Left alone it renders "✓ Equivalent — No meaningful
    difference between models" for ten pairs on which both models called a
    tool the recording never made, directly beneath a critical regression
    carrying the same evaluator name.

    Rows are matched to records on ``(prompt_id, evaluator_name)`` plus the
    axis, when the comparison names one. It only names one where the
    evaluator contributed several — which is exactly when the match would
    otherwise be ambiguous.
    """
    by_key: dict[tuple[str, str, str], list[EvalRecord]] = {}
    for record in records:
        if record.error is None:
            by_key.setdefault((record.prompt_id, record.evaluator_name, record.kind), []).append(
                record
            )

    out: list[dict[str, Any]] = []
    for row in rows:
        axis = _axis_of(row.get("notes", []))
        measured = [
            record
            for (prompt_id, name, kind), group in by_key.items()
            if prompt_id == row["prompt_id"]
            and name == row["evaluator_name"]
            and (not axis or kind == axis)
            for record in group
        ]
        misses = [record for record in measured if is_shared_ground_truth_miss(record)]
        if measured and len(misses) == len(measured):
            out.append(
                {
                    **row,
                    "notes": [
                        *row.get("notes", []),
                        f"{SHARED_GROUND_TRUTH_NOTE_PREFIX} all {len(misses)} rows — both "
                        f"models missed the suite's recorded tool calls by the same margin, "
                        f"so this axis measured the suite and not the migration.",
                    ],
                },
            )
        else:
            out.append(row)
    return out


def _axis_of(notes: Any) -> str:
    """The axis slug a comparison names in ``notes``, or ``""``."""
    if not isinstance(notes, list):
        return ""
    for note in notes:
        if isinstance(note, str) and note.startswith(AXIS_NOTE_PREFIX):
            return note[len(AXIS_NOTE_PREFIX) :].strip()
    return ""


def _build_tool_diffs(
    source_trace: ToolTrace | None,
    target_trace: ToolTrace | None,
) -> list[ToolDiff]:
    """Positional diff of two traces, one round at a time.

    A teacher-forced replay is several responses, and position 1 of round 2
    is not position 3 of the trace: diffing the flattened call lists reports
    a whole loop as shifted the moment one round makes an extra call. Each
    round is diffed on its own and its messages say which round they are
    about.
    """
    if source_trace is None and target_trace is None:
        return []
    rounds = max(
        source_trace.round_count if source_trace else 1,
        target_trace.round_count if target_trace else 1,
    )
    if rounds == 1:
        return _diff_calls(
            source_trace.calls if source_trace else [],
            target_trace.calls if target_trace else [],
        )
    diffs: list[ToolDiff] = []
    for index in range(rounds):
        diffs.extend(
            _diff_calls(
                _calls_of_round(source_trace, index),
                _calls_of_round(target_trace, index),
                prefix=f"Round {index + 1}: ",
            ),
        )
    return diffs


def _calls_of_round(trace: ToolTrace | None, index: int) -> list[ToolCall]:
    """One round's calls, or nothing when that side never replayed it."""
    if trace is None or index >= trace.round_count:
        return []
    return trace.round(index).calls


def _diff_calls(
    source_calls: list[ToolCall],
    target_calls: list[ToolCall],
    *,
    prefix: str = "",
) -> list[ToolDiff]:
    diffs: list[ToolDiff] = []

    max_len = max(len(source_calls), len(target_calls))
    for index in range(max_len):
        src = source_calls[index] if index < len(source_calls) else None
        tgt = target_calls[index] if index < len(target_calls) else None
        if src is None and tgt is not None:
            diffs.append(
                ToolDiff(
                    kind="extra_tool",
                    tool_name=tgt.tool_name,
                    message=f"{prefix}Target added {tgt.tool_name} at position {index + 1}.",
                    target_arguments=tgt.arguments,
                ),
            )
            continue
        if src is not None and tgt is None:
            diffs.append(
                ToolDiff(
                    kind="missing_tool",
                    tool_name=src.tool_name,
                    message=f"{prefix}Target omitted {src.tool_name} at position {index + 1}.",
                    source_arguments=src.arguments,
                ),
            )
            continue
        if src is None or tgt is None:
            continue
        if src.tool_name != tgt.tool_name:
            diffs.append(
                ToolDiff(
                    kind="tool_order_or_selection",
                    tool_name=tgt.tool_name,
                    message=(
                        f"{prefix}Position {index + 1}: source called {src.tool_name}, "
                        f"target called {tgt.tool_name}."
                    ),
                    source_arguments=src.arguments,
                    target_arguments=tgt.arguments,
                ),
            )
        elif src.arguments != tgt.arguments:
            diffs.append(
                ToolDiff(
                    kind="argument_drift",
                    tool_name=src.tool_name,
                    message=f"{prefix}Arguments changed for {src.tool_name} at position {index + 1}.",
                    source_arguments=src.arguments,
                    target_arguments=tgt.arguments,
                ),
            )
    return diffs


def _tool_change_to_dict(change: ToolChange | None) -> dict[str, Any] | None:
    """Serialise a tool change, adding ``rounds`` only when the replay had them.

    A single-shot pair keeps the exact two-key object every existing reader
    of ``report.json`` was written against.
    """
    if change is None:
        return None
    payload: dict[str, Any] = {
        "source_names": change.source_names,
        "target_names": change.target_names,
    }
    if change.rounds is not None:
        payload["rounds"] = [
            {
                "round": r.round,
                "source_names": r.source_names,
                "target_names": r.target_names,
                "expected_names": r.expected_names,
                "diverged": r.diverged,
            }
            for r in change.rounds
        ]
    return payload


def _to_jsonable(report: ReportData) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "started_at": report.started_at,
        "source_model": report.source_model,
        "target_model": report.target_model,
        "suite_path": report.suite_path,
        "n_examples": report.n_examples,
        "n_calls": report.n_calls,
        "cached_calls": report.cached_calls,
        "failed_calls": report.failed_calls,
        "truncated_calls": report.truncated_calls,
        "total_cost_usd": report.total_cost_usd,
        "executive_summary": report.executive_summary,
        "migration_decision": report.migration_decision,
        "prompt_sections": [
            {
                "prompt_id": ps.prompt_id,
                "aggregate_rows": ps.aggregate_rows,
                "slice_rows": ps.slice_rows,
                "has_tool_traces": ps.has_tool_traces,
                "economics": {
                    "source": role_economics_to_dict(ps.economics.source),
                    "target": role_economics_to_dict(ps.economics.target),
                },
                "example_rows": [
                    {
                        "example_id": er.example_id,
                        "tags": er.tags,
                        "delta_latency_ms": er.delta_latency_ms,
                        "latency_comparable": er.latency_comparable,
                        "delta_cost_usd": er.delta_cost_usd,
                        "worst_delta_score": er.worst_delta_score,
                        "tool_match": er.tool_match,
                        "tool_change": _tool_change_to_dict(er.tool_change),
                        "turn_index": er.turn_index,
                    }
                    for er in ps.example_rows
                ],
                "top_regressions": [
                    {
                        "prompt_id": tr.prompt_id,
                        "example_id": tr.example_id,
                        "evaluator_name": tr.evaluator_name,
                        "kind": tr.kind,
                        "tool_change": _tool_change_to_dict(tr.tool_change),
                        "delta": tr.delta,
                        "truncated": tr.truncated,
                        "target_empty_output": tr.target_empty_output,
                        "history": tr.history,
                        "turn_index": tr.turn_index,
                        "input_text": tr.input_text,
                        "source_score": tr.source_score,
                        "target_score": tr.target_score,
                        "explanation": tr.explanation,
                        "metadata": tr.metadata,
                        "source_text": tr.source_text,
                        "target_text": tr.target_text,
                        "source_trace": (tr.source_trace.model_dump() if tr.source_trace else None),
                        "target_trace": (tr.target_trace.model_dump() if tr.target_trace else None),
                        "source_agent_trace": (
                            tr.source_agent_trace.model_dump(mode="json")
                            if tr.source_agent_trace
                            else None
                        ),
                        "target_agent_trace": (
                            tr.target_agent_trace.model_dump(mode="json")
                            if tr.target_agent_trace
                            else None
                        ),
                        "trace_diff": (
                            {
                                "prompt_id": tr.trace_diff.prompt_id,
                                "example_id": tr.trace_diff.example_id,
                                "items": [
                                    {
                                        "kind": item.kind,
                                        "category": item.category,
                                        "source_index": item.source_index,
                                        "target_index": item.target_index,
                                        "source_name": item.source_name,
                                        "target_name": item.target_name,
                                        "field": item.field,
                                        "source_value": item.source_value,
                                        "target_value": item.target_value,
                                    }
                                    for item in tr.trace_diff.items
                                ],
                            }
                            if tr.trace_diff
                            else None
                        ),
                        "tool_diffs": [
                            {
                                "kind": d.kind,
                                "tool_name": d.tool_name,
                                "message": d.message,
                                "source_arguments": d.source_arguments,
                                "target_arguments": d.target_arguments,
                            }
                            for d in tr.tool_diffs
                        ],
                    }
                    for tr in ps.top_regressions
                ],
            }
            for ps in report.prompt_sections
        ],
        "methodology_notes": report.methodology_notes,
        "non_deterministic_models": report.non_deterministic_models,
        "dropped_params": report.dropped_params,
        "samples_per_example": report.samples_per_example,
        "judge_family_overlap": report.judge_family_overlap,
        "invariant_violations": [asdict(v) for v in report.invariant_violations],
    }


__all__ = [
    "REPORT_JSON_FILENAME",
    "SHARED_GROUND_TRUTH_NOTE_PREFIX",
    "InvariantViolationRow",
    "PromptEconomics",
    "PromptSection",
    "ReportData",
    "RoleEconomics",
    "RoundToolChange",
    "ToolChange",
    "ToolDiff",
    "TopRegression",
    "build_report_payload",
    "write_report_json",
]
