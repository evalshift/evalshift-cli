"""Implementation of ``evalshift analyze <run-id>``.

Loads ``scores.jsonl`` from a completed evaluation, groups records into
slices, runs the statistics pipeline, and writes ``analysis.json`` to
the run directory. The Phase 7 report renderer reads that file.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from evalshift_cli.analysis.policy import MigrationDecision, evaluate_migration_policy
from evalshift_cli.analysis.slicing import (
    SliceAggregate,
    UnmeasuredCounts,
    aggregates,
    build_slices,
    build_unmeasured,
    dedupe_slices,
)
from evalshift_cli.analysis.statistics import ComparisonResult, analyze
from evalshift_cli.cli.commands.doctor import CONFIG_FILENAME
from evalshift_cli.cli.commands.evaluate import SCORES_FILENAME
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.runner.checkpoint import (
    CheckpointError,
    iter_calls,
    read_state,
    run_dir_for,
)
from evalshift_cli.suite.loader import SuiteError, load_jsonl

ANALYSIS_FILENAME: str = "analysis.json"
MIGRATION_DECISION_FILENAME: str = "migration_decision.json"

GATE_SEVERITIES: frozenset[str] = frozenset({"critical", "high", "medium", "low"})


class MissingScoresError(FileNotFoundError):
    """Raised when ``scores.jsonl`` is missing for the requested run."""


class EmptyScoresError(ValueError):
    """Raised when ``scores.jsonl`` parses to zero records."""


@dataclass(frozen=True, slots=True)
class AnalyzeResult:
    """Outcome of running paired stats over a scored run."""

    run_id: str
    output_path: Path
    comparisons: tuple[ComparisonResult, ...]
    n_records: int
    migration_decision: MigrationDecision | None = None
    collapsed_slices: Mapping[str, str] = field(default_factory=dict)


def run_analyze(
    *,
    run_id: str,
    config_path: Path,
    runs_base: Path,
) -> AnalyzeResult:
    """Run paired stats over scored evaluations and persist analysis.json.

    Pure of side effects beyond the run directory; raises typed errors
    on every failure mode the standalone command surfaces.
    """
    cfg = load_config(config_path)

    run_dir = run_dir_for(run_id, runs_base)
    state = read_state(run_dir)

    scores_path = run_dir / SCORES_FILENAME
    if not scores_path.exists():
        raise MissingScoresError(
            f"no {SCORES_FILENAME} in {run_dir}; run `evalshift evaluate {run_id}` first.",
        )

    records = _load_score_records(scores_path)
    if not records:
        raise EmptyScoresError(f"{scores_path} is empty")

    try:
        suite = load_jsonl(state.suite_path)
    except SuiteError:
        from evalshift_cli.suite.models import Suite

        suite = Suite()

    # Budgeted slice names are the only ones the user spells out today, so
    # they are the ones deduplication must never collapse away.
    budgeted = (
        frozenset(cfg.migration_policy.slices) if cfg.migration_policy is not None else frozenset()
    )
    # Pairs the evaluate stage handed an evaluator that produced no row.
    # They are the only trace those pairs left, and without them an
    # evaluator that measured nothing would vanish from the analysis
    # instead of reporting `severity: insufficient`.
    coverage = state.evaluator_coverage
    sliced, collapsed_slices = dedupe_slices(
        build_slices(records=records, suite=suite, coverage=coverage),
        preferred=budgeted,
    )
    unmeasured: UnmeasuredCounts = {
        name: counts
        for name, counts in build_unmeasured(coverage=coverage, suite=suite).items()
        if name not in collapsed_slices
    }
    comparisons = analyze(
        sliced_by_slice=sliced,
        unmeasured_by_slice=unmeasured,
        # An axis with zero rows leaves nothing in scores.jsonl carrying the
        # config `blocking` flag; coverage is the only place it survives.
        advisory_axes={(c.evaluator_name, c.kind) for c in coverage if not c.blocking},
    )

    aggregates_by_slice = {
        name: asdict(_aggregate_for(name, sliced.get(name, []))) for name in sliced
    }
    payload: dict[str, Any] = {
        "run_id": run_id,
        "models": {
            "source": state.models.source,
            "target": state.models.target,
        },
        "n_examples": len(suite),
        "n_records": len(records),
        "slices": list(sliced.keys()),
        "collapsed_slices": dict(collapsed_slices),
        "aggregates": aggregates_by_slice,
        "comparisons": [_comparison_to_dict(c) for c in comparisons],
    }
    out_path = run_dir / ANALYSIS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    migration_decision: MigrationDecision | None = None
    if cfg.migration_policy is not None:
        migration_decision = evaluate_migration_policy(
            run_id=run_id,
            source_model=state.models.source,
            target_model=state.models.target,
            policy=cfg.migration_policy,
            comparisons=list(comparisons),
            records=records,
            calls=list(iter_calls(run_dir)),
            dropped_params=state.dropped_params,
            skipped_evaluators=state.skipped_evaluators,
        )
        decision_path = run_dir / MIGRATION_DECISION_FILENAME
        decision_path.write_text(
            json.dumps(migration_decision.to_dict(), indent=2),
            encoding="utf-8",
        )

    return AnalyzeResult(
        run_id=run_id,
        output_path=out_path,
        comparisons=tuple(comparisons),
        n_records=len(records),
        migration_decision=migration_decision,
        collapsed_slices=collapsed_slices,
    )


def analyze_command(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to analyze."),
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
    gate: Annotated[
        str,
        typer.Option(
            "--gate",
            help=(
                "CI gate: comma-separated severities that should fail the "
                "command with exit 1 (e.g. 'critical,high'). "
                "Allowed values: critical, high, medium, low."
            ),
        ),
    ] = "",
    policy_gate: Annotated[
        bool,
        typer.Option(
            "--policy-gate",
            help="CI gate: fail when migration_policy verdict is fail or conditional_pass.",
        ),
    ] = False,
) -> None:
    """Run paired statistical tests over scored evaluations."""
    console = Console()

    try:
        gate_severities = _parse_gate(gate)
    except ValueError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        result = run_analyze(
            run_id=run_id,
            config_path=config_path,
            runs_base=runs_base,
        )
    except ConfigError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc
    except CheckpointError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except MissingScoresError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except EmptyScoresError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    comparisons = list(result.comparisons)
    _print_summary(console, comparisons, result.output_path, run_id)
    _write_step_summary(comparisons, result.output_path, run_id)

    collapsed_note = format_collapsed_slices(result.collapsed_slices)
    if collapsed_note is not None:
        console.print(collapsed_note, style="dim")

    if result.migration_decision is not None:
        console.print(
            f"[bold]Migration verdict:[/bold] {result.migration_decision.verdict}",
        )
        if result.migration_decision.reason is not None:
            console.print(f"  {result.migration_decision.reason}", style="dim")
        for rec in result.migration_decision.recommendations:
            console.print(f"  {rec}", style="dim")

    if policy_gate:
        if result.migration_decision is None:
            console.print(
                "[red]✗ policy gate failed:[/red] no migration_policy configured",
            )
            raise typer.Exit(code=1)
        if result.migration_decision.verdict in {"fail", "conditional_pass"}:
            console.print(
                f"[red]✗ policy gate failed:[/red] {result.migration_decision.verdict}",
            )
            raise typer.Exit(code=1)

    if gate_severities:
        offending = [c for c in comparisons if c.severity in gate_severities]
        if offending:
            console.print(
                f"[red]✗ gate failed:[/red] "
                f"{len(offending)} comparison(s) at severity "
                f"{{{', '.join(sorted(gate_severities))}}}",
            )
            raise typer.Exit(code=1)


def format_collapsed_slices(collapsed: Mapping[str, str]) -> str | None:
    """Summarise deduplicated slices in one line, or ``None`` if none were.

    Grouped by the slice each dropped name collapsed into, so the common
    case — a promoted suite whose tags all cover every example — reads as a
    single clause instead of one per tag.
    """
    if not collapsed:
        return None
    by_kept: dict[str, list[str]] = defaultdict(list)
    for dropped, kept in collapsed.items():
        by_kept[kept].append(dropped)
    clauses = [
        f"dropped {', '.join(sorted(dropped))} (identical to {kept})"
        for kept, dropped in sorted(by_kept.items())
    ]
    return f"slices: {'; '.join(clauses)}"


def _load_score_records(path: Path) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            out.append(EvalRecord.model_validate_json(text))
    return out


def _aggregate_for(name: str, sliced: list[Any]) -> SliceAggregate:
    return aggregates(sliced, name)


def _comparison_to_dict(c: ComparisonResult) -> dict[str, Any]:
    """Serialise one comparison for ``analysis.json``.

    One field per :class:`ComparisonResult` field, no more: the hosted
    bundle carries these objects verbatim and its ``Comparison`` schema is
    ``additionalProperties: false``. ``kind`` is part of the row's identity
    there — the server keys ``run_comparisons`` on it — so dropping it
    collides the axes of a multi-axis evaluator at finalize.
    """
    return {
        "prompt_id": c.prompt_id,
        "evaluator_name": c.evaluator_name,
        "kind": c.kind,
        "slice_name": c.slice_name,
        "n": c.n,
        "test": c.test,
        "statistic": c.statistic,
        "p_value": c.p_value,
        "p_value_corrected": c.p_value_corrected,
        "effect_size": c.effect_size,
        "effect_size_ci_low": c.effect_size_ci_low,
        "effect_size_ci_high": c.effect_size_ci_high,
        "delta_avg_score": c.delta_avg_score,
        "severity": c.severity,
        "notes": c.notes,
    }


_SEVERITY_GLYPHS: dict[str, tuple[str, str]] = {
    "critical": ("✗", "red"),
    "high": ("✗", "red"),
    "medium": ("⚠", "yellow"),
    "low": ("⚠", "yellow"),
    "improved": ("↑", "green"),
    "none": ("✓", "green"),
    "insufficient": ("?", "dim"),
}


def _print_summary(
    console: Console,
    comparisons: list[ComparisonResult],
    out_path: Path,
    run_id: str,
) -> None:
    table = Table(title=f"analysis · {run_id}", show_lines=False)
    table.add_column("severity", no_wrap=True)
    table.add_column("prompt")
    table.add_column("evaluator")
    table.add_column("slice")
    table.add_column("n", justify="right")
    table.add_column("Δ avg", justify="right")
    table.add_column("|d|", justify="right")
    table.add_column("p_corr", justify="right")
    for c in comparisons:
        # Skip the implicit all-slice if there's a more specific row;
        # but for the MVP just show every row to keep things simple.
        glyph, style = _SEVERITY_GLYPHS.get(c.severity, ("?", "dim"))
        table.add_row(
            f"[{style}]{glyph} {c.severity}[/{style}]",
            c.prompt_id,
            c.evaluator_name,
            c.slice_name,
            str(c.n),
            f"{c.delta_avg_score:+.3f}",
            f"{abs(c.effect_size):.2f}",
            f"{c.p_value_corrected:.3f}",
        )
    console.print(table)
    console.print(
        f"[green]✓[/green] wrote analysis to [bold]{out_path}[/bold]",
    )
    console.print(
        f"[bold]Next:[/bold] [cyan]evalshift report {run_id}[/cyan]",
    )


def _parse_gate(raw: str) -> frozenset[str]:
    if not raw.strip():
        return frozenset()
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    unknown = sorted(set(parts) - GATE_SEVERITIES)
    if unknown:
        allowed = ", ".join(sorted(GATE_SEVERITIES))
        raise ValueError(
            f"unknown --gate severity {unknown}; allowed: {allowed}",
        )
    return frozenset(parts)


def _write_step_summary(
    comparisons: list[ComparisonResult],
    out_path: Path,
    run_id: str,
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        f"## evalshift · `{run_id}`",
        "",
        "| severity | prompt | evaluator | slice | n | Δ avg | \\|d\\| | p_corr |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in comparisons:
        lines.append(
            f"| {c.severity} | {c.prompt_id} | {c.evaluator_name} | "
            f"{c.slice_name} | {c.n} | {c.delta_avg_score:+.3f} | "
            f"{abs(c.effect_size):.2f} | {c.p_value_corrected:.3f} |",
        )
    lines.append("")
    lines.append(f"Artifact: `{out_path}`")
    lines.append("")
    with Path(summary_path).open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


__all__ = [
    "ANALYSIS_FILENAME",
    "GATE_SEVERITIES",
    "MIGRATION_DECISION_FILENAME",
    "AnalyzeResult",
    "EmptyScoresError",
    "MissingScoresError",
    "analyze_command",
    "run_analyze",
]
