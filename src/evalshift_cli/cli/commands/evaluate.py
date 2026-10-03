"""Implementation of ``evalshift evaluate <run-id>``.

Loads ``raw.jsonl`` from a completed run, pairs source and target calls
by ``(prompt_id, example_id, sample_index)``, runs every configured
evaluator, and appends one :class:`EvalRecord` per (example x evaluator)
to ``scores.jsonl``.

A repeated-sampling run (``defaults.samples_per_example > 1``) has several
sample pairs per example. Each is scored on its own, exactly as a
single-sample pair is, and the cells are then folded back to **one row per
example**: the scores become the mean over the successful samples, with
the per-sample values and the within-pair variance kept under
``metadata.samples``. Every downstream consumer — analysis, policy,
slicing, report, bundle — therefore still sees one row per example, so the
statistical ``n`` stays the number of examples and repeated sampling
never inflates power. See :func:`_reduce_sample_cells`.

Failed source or target calls (those with ``error != None`` in
``raw.jsonl``) are recorded with ``error="upstream call failed"`` and a
neutral 0.5/0.5 score so the analysis layer can decide how to handle
them rather than silently skipping them.

An evaluator that *measured nothing* on a pair writes no record at all.
Because an absent row leaves no trace, this stage also records
:class:`EvaluatorCoverage` — attempted versus recorded, plus the pairs
that produced nothing — back into ``state.json``, so the analysis layer
can still report the not-applicable count and still refuse to call an
unmeasured evaluator a pass.

Coverage is booked **per axis**, not per evaluator: ``tool_selection``
scores conformance and divergence independently and writes a row for
each, and folding two measurements into one tally would let ``recorded``
exceed ``attempted`` while hiding an axis that measured nothing behind
one that measured everything.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from evalshift_cli.cache.store import CacheStore
from evalshift_cli.captures.reader import CaptureError
from evalshift_cli.cli.commands.doctor import (
    CONFIG_FILENAME,
    CheckResult,
    render_results,
    source_conformance_check,
)
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.config.models import EvalShiftConfig, EvaluatorsConfig
from evalshift_cli.evaluators.agent_trace import AgentTraceEvaluator
from evalshift_cli.evaluators.base import EvalRecord, Evaluator, EvaluatorError, PairedScore
from evalshift_cli.evaluators.failures import INVARIANT_VIOLATION
from evalshift_cli.evaluators.llm_judge import PairwiseJudgeEvaluator
from evalshift_cli.evaluators.semantic import CosineSimilarityEvaluator, _cosine
from evalshift_cli.evaluators.structural import (
    JsonSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
)
from evalshift_cli.evaluators.tool_arguments import (
    EmbeddingsFn,
    ToolArgumentsEvaluator,
    ToolsetResolver,
)
from evalshift_cli.evaluators.tool_models import ToolSpec
from evalshift_cli.evaluators.tool_selection import ToolSelectionEvaluator
from evalshift_cli.evaluators.tool_trace_structure import ToolTraceStructureEvaluator
from evalshift_cli.evaluators.trace_invariants import build_trace_invariants_evaluator
from evalshift_cli.models.capabilities import honors_temperature
from evalshift_cli.models.client import ModelClient
from evalshift_cli.models.registry import resolve_model
from evalshift_cli.runner.checkpoint import (
    CheckpointError,
    iter_calls,
    read_state,
    run_dir_for,
    write_state,
)
from evalshift_cli.runner.models import Call, EvaluatorCoverage, UnmeasuredPair
from evalshift_cli.runner.orchestrator import resolve_example_tools, toolset_base_candidates
from evalshift_cli.suite.loader import SuiteError, load_jsonl
from evalshift_cli.suite.models import Suite, SuiteExample
from evalshift_cli.traces.loader import (
    TRACES_FILENAME,
    TraceLoadError,
    load_traces_jsonl,
    pairs_for_prompt_examples,
)

SCORES_FILENAME: str = "scores.jsonl"


@dataclass(frozen=True, slots=True)
class _PairedCalls:
    """Source + target calls for one (prompt, example, sample) pair.

    ``sample_index`` is ``0`` on a single-sample run. On a repeated-sampling
    run sample *i* of the source is paired with sample *i* of the target —
    never cross-sample — so each pair is a like-for-like comparison.
    """

    prompt_id: str
    example_id: str
    source: Call
    target: Call
    sample_index: int = 0


@dataclass(frozen=True, slots=True)
class _ScoredCell:
    """One (pair x evaluator) attempt and whatever it produced.

    ``record`` is ``None`` when the evaluator measured nothing — the cell
    still exists, which is how :func:`_coverage_for` knows the attempt was
    made at all once the row is gone. ``blocking`` is the evaluator's config
    flag, carried here so coverage still knows an axis was advisory when
    every one of its cells is recordless.
    """

    prompt_id: str
    example_id: str
    evaluator_name: str
    kind: str
    record: EvalRecord | None
    blocking: bool = True
    #: Which sample pair produced the cell; ``0`` after
    #: :func:`_reduce_sample_cells` has folded the samples of one example.
    sample_index: int = 0


@dataclass(frozen=True, slots=True)
class EvaluateResult:
    """Outcome of scoring a completed run."""

    run_id: str
    output_path: Path
    n_records: int
    #: Distinct ``(prompt, example)`` pairs scored — not sample pairs.
    n_pairs: int
    evaluator_names: tuple[str, ...]
    #: Per-(evaluator, axis) attempted-vs-recorded counts, also persisted
    #: into the run's ``state.json`` for the analysis stage.
    coverage: tuple[EvaluatorCoverage, ...] = ()
    #: The broken-harness finding, when the source model failed the ground
    #: truth recorded from it (see
    #: :func:`~evalshift_cli.cli.commands.doctor.source_conformance_check`).
    #: Carried as well as printed because ``evalshift compare`` scores quietly,
    #: inside a Live grid this table would fight with, and renders it itself
    #: immediately above the verdict it invalidates.
    harness_check: CheckResult | None = None


class NoEvaluatorsError(ValueError):
    """Raised when ``evalshift.yaml`` declares no evaluators."""


class NoPairsError(ValueError):
    """Raised when ``raw.jsonl`` yields no (source, target) pairs."""


def run_evaluate(
    *,
    run_id: str,
    config_path: Path,
    runs_base: Path,
    console: Console,
    quiet: bool = False,
) -> EvaluateResult:
    """Score a completed run. Raises typed errors on failure.

    The standalone Typer command catches and pretty-prints these; the
    aggregate ``evalshift compare`` command catches them too and renders a
    failed pipeline row.
    """
    cfg = load_config(config_path)

    run_dir = run_dir_for(run_id, runs_base)
    state = read_state(run_dir)

    if state.status != "completed" and not quiet:
        console.print(
            f"[yellow]⚠[/yellow] run is {state.status!r}; results may be incomplete.",
        )

    project_root = config_path.resolve().parent
    # One client shared by every judge so temperature rejections discovered
    # while judging are collected in one place and merged into state below.
    judge_client = ModelClient()
    # One resolution point for the run's evaluator set: which suite this run
    # was launched against decides what scores it, and that answer must be
    # the same one report and bundle reach later.
    evaluators = _build_evaluators(
        cfg.evaluators_for(state.suite_name),
        project_root,
        judge_client=judge_client,
        suite_path=Path(state.suite_path),
    )
    if not evaluators:
        raise NoEvaluatorsError(
            "no evaluators configured. Add at least one entry under evaluators: in evalshift.yaml.",
        )

    pairs = _pair_calls(run_dir)
    if not pairs:
        raise NoPairsError("no (source, target) pairs found in raw.jsonl")

    # Load the suite so tool evaluators (which consume ``SuiteExample``)
    # can find the row matching each pair. v0.1 evaluators don't need it
    # but loading it is cheap and makes the dispatch uniform.
    examples_by_id = _load_examples_by_id(state.suite_path)

    call_evaluators = [e for e in evaluators if not _is_agent_trace_evaluator(e)]
    agent_trace_evaluators = [e for e in evaluators if _is_agent_trace_evaluator(e)]

    records, coverage = asyncio.run(
        _score_everything(
            console=console,
            cfg=cfg,
            run_dir=run_dir,
            run_id=run_id,
            call_evaluators=call_evaluators,
            agent_trace_evaluators=agent_trace_evaluators,
            pairs=pairs,
            examples_by_id=examples_by_id,
            quiet=quiet,
        ),
    )

    output_path = run_dir / SCORES_FILENAME
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")

    # Coverage is the only record that an unmeasured pair was ever
    # attempted, so it must outlive this process alongside the scores.
    # Judge calls can discover temperature-rejecting models after the run
    # phase already wrote its state; merge them here so the report banner
    # covers the judge model too. A judge that ignores temperature without
    # rejecting it (DeepSeek thinking mode) never shows up at runtime, so
    # the judges are also checked the same way the run's arms were.
    ignoring_judges = {
        resolve_model(e.judge_model).id
        for e in evaluators
        if isinstance(e, PairwiseJudgeEvaluator) and not honors_temperature(e.judge_model)
    }
    runtime_nondet = sorted(
        (set(judge_client.temperature_rejected_models) | ignoring_judges)
        - set(state.non_deterministic_models)
    )
    write_state(
        run_dir,
        state.model_copy(
            update={
                "evaluator_coverage": coverage,
                "non_deterministic_models": [
                    *state.non_deterministic_models,
                    *runtime_nondet,
                ],
            },
        ),
    )

    # This is the first stage that can answer the question at all — it needs
    # the scores, and it must be asked before anything downstream turns them
    # into a verdict.
    harness_check = source_conformance_check(records)
    if harness_check is not None and not quiet:
        render_results([harness_check], console)

    return EvaluateResult(
        run_id=run_id,
        output_path=output_path,
        n_records=len(records),
        n_pairs=len({(pair.prompt_id, pair.example_id) for pair in pairs}),
        evaluator_names=tuple(e.name for e in evaluators),
        coverage=tuple(coverage),
        harness_check=harness_check,
    )


def evaluate(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to evaluate (e.g. r_20260601_abc123)."),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help=f"Path to evalshift.yaml (default: ./{CONFIG_FILENAME}).",
            file_okay=True,
            dir_okay=False,
        ),
    ] = Path(CONFIG_FILENAME),
    runs_base: Annotated[
        Path,
        typer.Option(
            "--runs-base",
            help="Base directory for run state (advanced).",
            hidden=True,
            file_okay=False,
        ),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Score every (source, target) pair from a completed run."""
    console = Console()

    try:
        result = run_evaluate(
            run_id=run_id,
            config_path=config_path,
            runs_base=runs_base,
            console=console,
        )
    except ConfigError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc
    except CheckpointError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except EvaluatorError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except (NoEvaluatorsError, NoPairsError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]✓[/green] wrote {result.n_records} eval records to {result.output_path}",
    )
    console.print(
        f"[bold]Next:[/bold] [cyan]evalshift analyze {run_id}[/cyan]",
    )


def _build_evaluators(
    evaluators_cfg: EvaluatorsConfig,
    project_root: Path,
    *,
    judge_client: ModelClient,
    suite_path: Path | None = None,
) -> list[Evaluator]:
    """Instantiate every evaluator in an already-resolved evaluator set.

    Takes the resolved :class:`EvaluatorsConfig` rather than the whole config
    so per-suite resolution happens once, in the caller, instead of being
    re-derived here (see
    :meth:`~evalshift_cli.config.models.EvalShiftConfig.evaluators_for`).
    """
    # Tool evaluators (v0.2) implement ``score_pair`` rather than the
    # text-evaluator ``score`` method, so they don't satisfy the
    # :class:`Evaluator` Protocol structurally. We treat them as
    # duck-typed ``Evaluator``-likes — the dispatch in ``_score_one``
    # checks for ``score_pair`` before calling.
    out: list[Evaluator] = []

    def _add(evaluator: Evaluator, *, blocking: bool) -> None:
        # The config's ``blocking`` flag rides along on the evaluator
        # instance so the scoring loop can stamp it onto each EvalRecord
        # without re-resolving config.
        setattr(evaluator, "blocking", blocking)  # noqa: B010
        out.append(evaluator)

    for s in evaluators_cfg.structural:
        if s.type == "json_schema":
            assert s.schema_path is not None  # pydantic invariant
            schema_path = Path(s.schema_path)
            if not schema_path.is_absolute():
                schema_path = project_root / schema_path
            _add(JsonSchemaEvaluator(schema_path=schema_path), blocking=s.blocking)
        elif s.type == "regex":
            assert s.pattern is not None
            _add(RegexEvaluator(pattern=s.pattern), blocking=s.blocking)
        else:  # length
            _add(
                LengthEvaluator(min_chars=s.min_chars, max_chars=s.max_chars),
                blocking=s.blocking,
            )

    # Kept as a local so the tool-arguments ``semantic`` field strategy can
    # borrow it below. It must be this *same instance*: ``_run_scoring``
    # attaches the run cache after evaluators are built, so a second embedder
    # would pay for every embedding twice.
    semantic_evaluator: CosineSimilarityEvaluator | None = None
    if evaluators_cfg.semantic is not None:
        semantic_evaluator = CosineSimilarityEvaluator(
            embedding_model=evaluators_cfg.semantic.embedding_model,
            min_similarity=evaluators_cfg.semantic.min_similarity,
        )
        _add(semantic_evaluator, blocking=evaluators_cfg.semantic.blocking)

    for j in evaluators_cfg.llm_judge:
        _add(
            PairwiseJudgeEvaluator(
                criterion_name=j.criterion_name,
                criterion_prompt=j.criterion_prompt,
                judge_model=j.judge_model,
                client=judge_client,
            ),
            blocking=j.blocking,
        )

    # v0.2 — tool-call evaluators. Each implements ``score_pair`` rather
    # than ``score`` because they consume the (source, target) ToolTrace
    # pair directly.
    for ts in evaluators_cfg.tool_selection:
        _add(ToolSelectionEvaluator(ts), blocking=ts.blocking)  # type: ignore[arg-type]
    # ``strategies: {<field>: semantic}`` is unreachable without this — the
    # evaluator's own fallback degrades to exact string equality, silently.
    # With no semantic evaluator configured there is no embedding model to
    # borrow, and that fallback is the honest answer.
    embeddings_fn = _make_embeddings_fn(semantic_evaluator)
    # ``auto``'s schema rung reads the toolset the capture already recorded.
    # One resolver, shared by every tool_arguments evaluator, so N of them
    # share one sidecar cache.
    toolset_resolver = _make_toolset_resolver(suite_path)
    for ta in evaluators_cfg.tool_arguments:
        _add(
            ToolArgumentsEvaluator(  # type: ignore[arg-type]
                ta,
                embeddings_fn=embeddings_fn,
                toolset_resolver=toolset_resolver,
            ),
            blocking=ta.blocking,
        )
    for tts in evaluators_cfg.tool_trace_structure:
        _add(ToolTraceStructureEvaluator(tts), blocking=tts.blocking)  # type: ignore[arg-type]
    for agent_trace in evaluators_cfg.agent_trace:
        _add(AgentTraceEvaluator(agent_trace), blocking=agent_trace.blocking)  # type: ignore[arg-type]
    # ``traces: replayed`` builds a ``score_pair`` evaluator and ``imported``
    # a ``score_trace_pair`` one, so the method-based dispatch below routes
    # each to its own path without knowing about the family.
    for ti in evaluators_cfg.trace_invariants:
        _add(build_trace_invariants_evaluator(ti), blocking=ti.blocking)  # type: ignore[arg-type]

    return out


def _make_toolset_resolver(suite_path: Path | None) -> ToolsetResolver | None:
    """Build the per-example toolset resolver ``auto`` dispatches on.

    Backed by the orchestrator's own
    :func:`~evalshift_cli.runner.orchestrator.resolve_example_tools`, so the
    evaluator resolves a ``toolset_ref`` exactly the way dispatch did — same
    candidate base directories, same sidecar integrity check. The cache is
    closed over, so N examples sharing a toolset read it once per run.

    Args:
        suite_path: The run's golden suite, whose directory is one of the
            candidate locations for a sidecar. ``None`` (no suite in hand)
            yields ``None`` — schema dispatch is skipped and ``auto`` falls
            back to its schema-free ladder.

    Returns:
        The resolver, or ``None`` when there is no suite path to resolve
        against. The resolver itself returns ``None`` for an example whose
        sidecar cannot be resolved: a missing toolset degrades scoring, it
        does not fail the run.
    """
    if suite_path is None:
        return None
    bases = toolset_base_candidates(suite_path=suite_path)
    cache: dict[str, list[ToolSpec]] = {}

    def _resolve(example: SuiteExample) -> list[ToolSpec] | None:
        try:
            return list(resolve_example_tools(example, toolset_bases=bases, toolset_cache=cache))
        except CaptureError:
            return None

    return _resolve


def _make_embeddings_fn(
    semantic: CosineSimilarityEvaluator | None,
) -> EmbeddingsFn | None:
    """Adapt a semantic evaluator into a ``(a, b) -> cosine similarity`` callable.

    Thin wrapper over the evaluator's own ``_embed`` + :func:`_cosine` rather
    than a second embedding path, so argument embeddings share its model *and*
    its cache — one embedding per distinct string per run.

    Args:
        semantic: The configured semantic evaluator, or ``None`` when the
            project has none.

    Returns:
        The callable, or ``None`` when there is no evaluator to borrow a model
        from — which leaves ``ToolArgumentsEvaluator`` on its exact-match
        fallback.
    """
    if semantic is None:
        return None

    async def embeddings_fn(a: str, b: str) -> float:
        if a == b:
            return 1.0
        source, target = await asyncio.gather(semantic._embed(a), semantic._embed(b))
        return _cosine(source, target)

    return embeddings_fn


def _evaluator_blocking(evaluator: Evaluator) -> bool:
    """The config ``blocking`` flag stamped by :func:`_build_evaluators`."""
    return bool(getattr(evaluator, "blocking", True))


def _evaluator_kind(evaluator: Evaluator) -> str:
    """The evaluator's type slug, which the analysis layer selects rows on.

    Text evaluators return a :class:`PairedScore` and never build a record of
    their own, so their kind can only be stamped here. Tool evaluators build
    their own records and already stamp it; restamping is a no-op that keeps
    every path through this module identical.
    """
    return str(getattr(evaluator, "kind", ""))


def _evaluator_kinds(evaluator: Evaluator) -> tuple[str, ...]:
    """Every axis this evaluator attempts on a pair, one slug each.

    All but ``tool_selection`` measure one thing and write at most one row.
    ``tool_selection`` measures conformance and divergence independently and
    writes a row for each, so it declares both slugs — and each gets its own
    :class:`EvaluatorCoverage`. Coverage has to be per axis: ``attempted``
    counts axes, not examples, or ``recorded`` would exceed it, and "k of n
    pairs were not measurable" is a statement about one measurement rather
    than about an evaluator that makes two.
    """
    kinds = getattr(evaluator, "kinds", None)
    if kinds is None:
        return (_evaluator_kind(evaluator),)
    return tuple(str(k) for k in kinds)


def _is_tool_evaluator(evaluator: Evaluator) -> bool:
    """True iff ``evaluator`` consumes :class:`ToolTrace` pairs (v0.2)."""
    return hasattr(evaluator, "score_pair")


def _is_agent_trace_evaluator(evaluator: Evaluator) -> bool:
    """True iff ``evaluator`` consumes imported :class:`AgentTrace` pairs."""
    return hasattr(evaluator, "score_trace_pair")


def _applies(evaluator: Evaluator, prompt_id: str) -> bool:
    """Whether ``evaluator`` is scoped to ``prompt_id`` at all.

    An evaluator outside its scope was never asked, so it gets no cell: no
    row, and no coverage attempt that would later read as "unmeasured".
    Only evaluators that define ``applies`` are scoped; every other family
    keeps today's behaviour, where ``applies_to`` is accepted but unenforced.

    Args:
        evaluator: Any built evaluator.
        prompt_id: The prompt a pair was produced from.

    Returns:
        ``False`` only when the evaluator defines ``applies`` and it rejects
        ``prompt_id``.
    """
    applies = getattr(evaluator, "applies", None)
    return True if applies is None else bool(applies(prompt_id))


def _pair_calls(run_dir: Path) -> list[_PairedCalls]:
    """Pair each source row with the target row of the same sample.

    Sorted by ``(prompt_id, example_id, sample_index)``, so the samples of
    one example are adjacent and in order — which is what lets
    :func:`_reduce_sample_cells` keep ``scores.jsonl`` in the same
    example-major order a single-sample run writes.
    """
    by_key: dict[tuple[str, str, int], dict[str, Call]] = {}
    for call in iter_calls(run_dir):
        key = (call.prompt_id, call.example_id, call.sample_index)
        by_key.setdefault(key, {})[call.role] = call
    pairs: list[_PairedCalls] = []
    for (prompt_id, example_id, sample_index), sides in sorted(by_key.items()):
        if "source" in sides and "target" in sides:
            pairs.append(
                _PairedCalls(
                    prompt_id=prompt_id,
                    example_id=example_id,
                    source=sides["source"],
                    target=sides["target"],
                    sample_index=sample_index,
                ),
            )
    return pairs


def _load_examples_by_id(suite_path: str) -> dict[str, SuiteExample]:
    """Load the suite and index examples by id; tolerate failures."""
    try:
        suite: Suite = load_jsonl(suite_path)
    except (SuiteError, FileNotFoundError, OSError):
        return {}
    return {ex.id: ex for ex in suite.examples}


async def _score_everything(
    *,
    console: Console,
    cfg: EvalShiftConfig,
    run_dir: Path,
    run_id: str,
    call_evaluators: list[Evaluator],
    agent_trace_evaluators: list[Evaluator],
    pairs: list[_PairedCalls],
    examples_by_id: dict[str, SuiteExample],
    quiet: bool,
) -> tuple[list[EvalRecord], list[EvaluatorCoverage]]:
    """Single async entrypoint for the scoring stage.

    Owns the cache lifetime: the judge and semantic evaluators make their
    own model calls, which the run-stage cache never covered, so re-running
    ``evaluate`` over an unchanged run used to pay full price every time.

    Returns:
        The records to write, and the per-evaluator coverage describing
        which attempts produced them.
    """
    cache = await CacheStore.open() if cfg.defaults.cache else None
    try:
        for evaluator in call_evaluators:
            if isinstance(evaluator, PairwiseJudgeEvaluator | CosineSimilarityEvaluator):
                evaluator.cache = cache
        cells = await _score_all(
            console,
            call_evaluators,
            pairs,
            run_id,
            examples_by_id,
            concurrency=cfg.defaults.concurrency,
            quiet=quiet,
        )
        if agent_trace_evaluators:
            cells.extend(
                await _score_agent_traces(
                    run_dir=run_dir,
                    evaluators=agent_trace_evaluators,
                    pairs=pairs,
                    run_id=run_id,
                ),
            )
        return [c.record for c in cells if c.record is not None], _coverage_for(cells)
    finally:
        if cache is not None:
            await cache.close()


def _coverage_for(cells: list[_ScoredCell]) -> list[EvaluatorCoverage]:
    """Fold scored cells into one :class:`EvaluatorCoverage` per axis.

    Keyed by ``(evaluator_name, kind)`` because that is what the analysis
    layer groups comparisons by — coverage has to line up with the rows it
    accounts for. An evaluator scoring two axes gets two entries: they are
    separate measurements with separate denominators, and one entry for both
    could report ``recorded`` above ``attempted`` while hiding an axis that
    measured nothing behind one that measured everything. Insertion order
    follows the first attempt, so the output order is the configured
    evaluator order.
    """
    by_axis: dict[tuple[str, str], EvaluatorCoverage] = {}
    for cell in cells:
        key = (cell.evaluator_name, cell.kind)
        coverage = by_axis.get(key)
        if coverage is None:
            coverage = EvaluatorCoverage(
                evaluator_name=cell.evaluator_name,
                kind=cell.kind,
                attempted=0,
                recorded=0,
                blocking=cell.blocking,
            )
            by_axis[key] = coverage
        coverage.attempted += 1
        if cell.record is not None:
            coverage.recorded += 1
        else:
            coverage.unmeasured.append(
                UnmeasuredPair(prompt_id=cell.prompt_id, example_id=cell.example_id),
            )
    return list(by_axis.values())


async def _score_all(
    console: Console,
    evaluators: list[Evaluator],
    pairs: list[_PairedCalls],
    run_id: str,
    examples_by_id: dict[str, SuiteExample],
    *,
    concurrency: int,
    quiet: bool = False,
) -> list[_ScoredCell]:
    """Score every (pair x evaluator) combination, up to ``concurrency`` at once.

    Semantic and judge evaluators each make their own network calls, so
    scoring serially made the evaluate stage the slowest part of the
    pipeline by a wide margin. Work is dispatched under a semaphore and
    gathered, which restores the original pair-major / evaluator-minor
    record order regardless of completion order — ``scores.jsonl`` must
    stay byte-stable across runs.

    Returns one :class:`_ScoredCell` per combination, including the ones
    that produced no record.
    """
    if not evaluators:
        return []

    work = [
        (pair, evaluator)
        for pair in pairs
        for evaluator in evaluators
        if _applies(evaluator, pair.prompt_id)
    ]
    sem = asyncio.Semaphore(concurrency)
    progress: Progress | None = None
    task_id: TaskID | None = None

    if not quiet:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        task_id = progress.add_task(
            description=f"[cyan]evaluate {run_id}[/cyan]",
            total=len(work),
        )

    async def _score_bounded(pair: _PairedCalls, evaluator: Evaluator) -> list[_ScoredCell]:
        async with sem:
            records = await _score_one(evaluator, pair, run_id, examples_by_id)
        if progress is not None and task_id is not None:
            progress.advance(task_id)
        return _cells_for(pair, evaluator, records)

    with ExitStack() as stack:
        if progress is not None:
            stack.enter_context(progress)
        per_combination = await asyncio.gather(
            *(_score_bounded(pair, evaluator) for pair, evaluator in work),
        )
    return _reduce_sample_cells([cell for cells in per_combination for cell in cells])


def _reduce_sample_cells(cells: list[_ScoredCell]) -> list[_ScoredCell]:
    """Fold the per-sample cells of each example into one cell.

    Grouped by ``(prompt_id, example_id, evaluator_name, kind)`` in
    first-seen order — pair-major input keeps the output example-major, the
    order a single-sample run already writes. A group of one cell (every
    group on a single-sample run, and every agent-trace cell) is returned
    untouched, so ``scores.jsonl`` is byte-identical to before repeated
    sampling existed.

    For a group of several samples:

    * samples that measured nothing (no record) are ignored; a group with no
      record at all stays one recordless cell, so coverage still counts the
      example as attempted and unmeasured;
    * ``source_score`` / ``target_score`` / ``delta`` are the means over the
      samples that scored without error;
    * ``metadata`` is the first successful sample's, plus ``samples`` —
      ``n`` (sample pairs attempted), ``scored`` (without error), the three
      per-sample lists, and ``delta_variance`` (population variance of the
      deltas, ``0.0`` for one);
    * ``explanation`` is the first successful sample's, prefixed with
      ``mean of k samples``;
    * ``error`` is set only when every scored sample errored, joining the
      distinct messages; the neutral 0.5/0.5 row is kept.

    Args:
        cells: One cell per ``(sample pair x evaluator axis)``.

    Returns:
        One cell per ``(example x evaluator axis)``, ``sample_index`` 0.
    """
    groups: dict[tuple[str, str, str, str], list[_ScoredCell]] = {}
    for cell in cells:
        key = (cell.prompt_id, cell.example_id, cell.evaluator_name, cell.kind)
        groups.setdefault(key, []).append(cell)
    return [_reduce_one_group(group) for group in groups.values()]


def _reduce_one_group(group: list[_ScoredCell]) -> _ScoredCell:
    if len(group) == 1:
        return group[0]
    first = group[0]
    records = [cell.record for cell in group if cell.record is not None]
    if not records:
        return replace(first, record=None, sample_index=0)

    successful = [record for record in records if record.error is None]
    if not successful:
        errors = list(dict.fromkeys(record.error for record in records if record.error))
        reduced = records[0].model_copy(
            update={
                "error": "; ".join(errors),
                "metadata": {
                    **records[0].metadata,
                    "samples": _samples_metadata(len(group), []),
                },
            },
        )
        return replace(first, record=reduced, sample_index=0)

    lead = successful[0]
    explanation = f"mean of {len(successful)} samples"
    if lead.explanation:
        explanation = f"{explanation} · {lead.explanation}"
    reduced = lead.model_copy(
        update={
            "source_score": _mean([r.source_score for r in successful]),
            "target_score": _mean([r.target_score for r in successful]),
            "delta": _mean([r.delta for r in successful]),
            "explanation": explanation,
            "metadata": {
                **lead.metadata,
                **_merged_violations(successful),
                "samples": _samples_metadata(len(group), successful),
            },
        },
    )
    return replace(first, record=reduced, sample_index=0)


_VIOLATION_KEYS = ("source_violations", "target_violations")


def _merged_violations(records: list[EvalRecord]) -> dict[str, Any]:
    """Every scored sample's trace-rule violations, not only the lead's.

    A reduced row's scores are means over samples, so one broken sample in
    three pulls ``target_score`` below 1.0 and the gate counts it -- but the
    row's metadata is the lead sample's, which may be the clean one. Without
    this the report would show a failed example and no violation to read.
    ``sample`` is the ordinal among the scored samples. Empty for rows that
    carry no violation lists, which is every family but ``trace_invariants``.

    Args:
        records: The successfully scored samples of one example, lead first.

    Returns:
        Metadata keys to overlay on the lead's: the merged violation lists,
        and ``failure_categories`` with ``INVARIANT_VIOLATION`` when any
        sample's target broke a rule.
    """
    merged: dict[str, Any] = {}
    for key in _VIOLATION_KEYS:
        if not any(key in record.metadata for record in records):
            continue
        merged[key] = [
            {**violation, "sample": ordinal}
            for ordinal, record in enumerate(records)
            for violation in record.metadata.get(key) or []
            if isinstance(violation, dict)
        ]
    if merged.get("target_violations"):
        categories = [
            c for c in records[0].metadata.get("failure_categories") or [] if isinstance(c, str)
        ]
        if INVARIANT_VIOLATION not in categories:
            categories.append(INVARIANT_VIOLATION)
        merged["failure_categories"] = categories
    return merged


def _samples_metadata(n: int, successful: list[EvalRecord]) -> dict[str, Any]:
    deltas = [r.delta for r in successful]
    return {
        "n": n,
        "scored": len(successful),
        "source_scores": [r.source_score for r in successful],
        "target_scores": [r.target_score for r in successful],
        "deltas": deltas,
        "delta_variance": _population_variance(deltas),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _cells_for(
    pair: _PairedCalls,
    evaluator: Evaluator,
    records: list[EvalRecord],
) -> list[_ScoredCell]:
    """One cell per axis the evaluator attempted, holding what it produced.

    A single-axis evaluator has nothing to match: its one record — if it
    wrote one — belongs to its one slug. That is also the only shape that
    works for the text evaluators, whose records carry ``kind: ""`` because
    they were never given a slug. Multi-axis records are matched on the slug
    each one stamped itself, so an axis that measured nothing is a cell with
    no record rather than a missing cell.

    A record under a slug the evaluator did not declare still gets a cell of
    its own. Only cells reach ``scores.jsonl``, so dropping it would delete
    a real measurement to keep the bookkeeping tidy.
    """
    kinds = _evaluator_kinds(evaluator)
    if len(kinds) == 1:
        by_kind: dict[str, EvalRecord | None] = {kinds[0]: records[0] if records else None}
    else:
        by_kind = dict.fromkeys(kinds)
        for record in records:
            by_kind[record.kind] = record
    return [
        _ScoredCell(
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            evaluator_name=evaluator.name,
            kind=kind,
            record=record,
            blocking=_evaluator_blocking(evaluator),
            sample_index=pair.sample_index,
        )
        for kind, record in by_kind.items()
    ]


async def _score_agent_traces(
    *,
    run_dir: Path,
    evaluators: list[Evaluator],
    pairs: list[_PairedCalls],
    run_id: str,
) -> list[_ScoredCell]:
    traces_path = run_dir / TRACES_FILENAME
    if not traces_path.exists():
        raise EvaluatorError(
            "agent_trace evaluators require imported traces. "
            f"Run: evalshift traces import {run_id} --source ... --target ...",
        )
    try:
        traces = load_traces_jsonl(traces_path)
    except TraceLoadError as exc:
        raise EvaluatorError(str(exc)) from exc
    # Imported traces have no sample dimension: one trace pair per example,
    # however many sample pairs the run produced.
    trace_pairs = pairs_for_prompt_examples(
        traces,
        prompt_examples=list(dict.fromkeys((pair.prompt_id, pair.example_id) for pair in pairs)),
    )
    cells: list[_ScoredCell] = []
    for trace_pair in trace_pairs:
        for evaluator in evaluators:
            if not _applies(evaluator, trace_pair.prompt_id):
                continue
            try:
                record = await evaluator.score_trace_pair(  # type: ignore[attr-defined]
                    run_id=run_id,
                    source_trace=trace_pair.source,
                    target_trace=trace_pair.target,
                )
                record = record.model_copy(
                    update={
                        "blocking": _evaluator_blocking(evaluator),
                        "kind": _evaluator_kind(evaluator),
                    },
                )
            except Exception as exc:
                record = EvalRecord(
                    run_id=run_id,
                    prompt_id=trace_pair.prompt_id,
                    example_id=trace_pair.example_id,
                    evaluator_name=evaluator.name,
                    kind=_evaluator_kind(evaluator),
                    source_score=0.5,
                    target_score=0.5,
                    delta=0.0,
                    error=f"evaluator error: {exc}",
                    blocking=_evaluator_blocking(evaluator),
                )
            cells.append(
                _ScoredCell(
                    prompt_id=trace_pair.prompt_id,
                    example_id=trace_pair.example_id,
                    evaluator_name=evaluator.name,
                    kind=_evaluator_kind(evaluator),
                    record=record,
                    blocking=_evaluator_blocking(evaluator),
                ),
            )
    return cells


def _error_records(
    evaluator: Evaluator,
    pair: _PairedCalls,
    run_id: str,
    error: str,
) -> list[EvalRecord]:
    """One errored row per axis the evaluator would have measured.

    A broken measurement is still a measurement per axis: two axes over a
    truncated pair are two things that could not be measured, and giving
    them one row between them would leave the other axis looking merely
    not-applicable.
    """
    return [
        EvalRecord(
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            evaluator_name=evaluator.name,
            kind=kind,
            source_score=0.5,
            target_score=0.5,
            delta=0.0,
            error=error,
            blocking=_evaluator_blocking(evaluator),
        )
        for kind in _evaluator_kinds(evaluator)
    ]


async def _score_one(
    evaluator: Evaluator,
    pair: _PairedCalls,
    run_id: str,
    examples_by_id: dict[str, SuiteExample],
) -> list[EvalRecord]:
    """Score one pair with one evaluator; empty when it measured nothing.

    Returns one record per axis the evaluator measured — at most one for
    every evaluator but ``tool_selection``, which scores conformance and
    divergence independently.
    """
    upstream_failed = pair.source.error is not None or pair.target.error is not None
    if upstream_failed:
        upstream = "; ".join(x for x in (pair.source.error, pair.target.error) if x is not None)
        return _error_records(evaluator, pair, run_id, f"upstream call failed: {upstream}")

    # A call whose output was cut off at the token cap is a broken
    # measurement, not a model verdict — scoring truncated text would
    # manufacture a false regression. Route it through the ``error`` path
    # (like an upstream failure) so slicing.py and policy.py exclude it
    # from the paired statistics and regression metrics.
    if pair.source.truncated or pair.target.truncated:
        return _error_records(evaluator, pair, run_id, "output truncated (token cap)")

    # v0.2 — tool evaluators consume the (source, target) ToolTrace pair.
    # Plain prompts have ``call.trace is None`` and we skip cleanly with
    # a neutral record so the analysis layer doesn't see noise.
    if _is_tool_evaluator(evaluator):
        return await _score_one_tool(evaluator, pair, run_id, examples_by_id)

    example = examples_by_id.get(pair.example_id)
    input_vars = example.inputs if example is not None else {}
    history = (
        [m.model_dump(exclude_none=True) for m in example.history]
        if example is not None and example.history
        else None
    )

    try:
        score: PairedScore | None = await evaluator.score(
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            input_vars=input_vars,
            source_output=pair.source.text,
            target_output=pair.target.text,
            history=history,
        )
    except Exception as exc:
        return _error_records(evaluator, pair, run_id, f"evaluator error: {exc}")

    if score is None:
        # Nothing was measured on this pair. Writing a row would mean
        # inventing a score; the attempt survives in EvaluatorCoverage.
        return []

    return [
        EvalRecord(
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example_id=pair.example_id,
            evaluator_name=evaluator.name,
            kind=_evaluator_kind(evaluator),
            source_score=score.source_score,
            target_score=score.target_score,
            delta=score.delta,
            explanation=score.explanation,
            metadata=score.metadata,
            blocking=_evaluator_blocking(evaluator),
        ),
    ]


async def _score_one_tool(
    evaluator: Evaluator,
    pair: _PairedCalls,
    run_id: str,
    examples_by_id: dict[str, SuiteExample],
) -> list[EvalRecord]:
    """Dispatch a v0.2 tool evaluator. Skips plain (text) calls cleanly.

    ``score_pair`` returns one record, ``None``, or — for the two-axis
    ``tool_selection`` — a list of them; all three normalise to a list here.
    """
    if pair.source.trace is None or pair.target.trace is None:
        # No tool trace on either side → nothing for this evaluator to
        # measure, so nothing to report. The 1.0/1.0 this used to write
        # claimed a perfect tool match on a plain text prompt.
        return []
    # tools=[]: this placeholder stands in for an example the loaded suite has
    # no record of at all, so there is no real toolset to carry -- not an
    # assertion that production offered none. See SuiteExample.toolset_ref /
    # .tools docstring for why one of the two is required on every example.
    example = examples_by_id.get(pair.example_id) or SuiteExample(id=pair.example_id, tools=[])
    try:
        scored: EvalRecord | list[EvalRecord] | None = await evaluator.score_pair(  # type: ignore[attr-defined]
            run_id=run_id,
            prompt_id=pair.prompt_id,
            example=example,
            source_trace=pair.source.trace,
            target_trace=pair.target.trace,
        )
    except Exception as exc:
        return _error_records(evaluator, pair, run_id, f"evaluator error: {exc}")
    if scored is None:
        return []
    records = scored if isinstance(scored, list) else [scored]
    blocking = _evaluator_blocking(evaluator)
    fallback_kind = _evaluator_kind(evaluator)
    # A multi-axis evaluator stamps the axis slug itself; restamping would
    # collapse both axes onto the family slug and undo the split.
    return [
        record.model_copy(
            update={"blocking": blocking, "kind": record.kind or fallback_kind},
        )
        for record in records
    ]


__all__ = [
    "SCORES_FILENAME",
    "EvaluateResult",
    "NoEvaluatorsError",
    "NoPairsError",
    "evaluate",
    "run_evaluate",
]
