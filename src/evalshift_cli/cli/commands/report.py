"""Implementation of ``evalshift report <run-id>``.

Builds the report payload from the run directory's artefacts, writes
``report.html`` (and ``report.json``), and optionally opens the HTML
file in the user's default browser via ``--open``.
"""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from evalshift_cli.cli.commands._star import print_star_prompt
from evalshift_cli.cli.commands.doctor import CONFIG_FILENAME
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.insights.stage import ensure_insight
from evalshift_cli.models.client import ModelClient
from evalshift_cli.reports.html import write_html
from evalshift_cli.reports.json import build_report_payload, write_report_json
from evalshift_cli.runner.checkpoint import (
    CheckpointError,
    read_state,
    run_dir_for,
)


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Outcome of rendering an HTML + JSON report."""

    run_id: str
    html_path: Path
    json_path: Path


def run_report(
    *,
    run_id: str,
    config_path: Path,
    runs_base: Path,
    insights: bool = True,
    client: ModelClient | None = None,
) -> ReportResult:
    """Render the HTML + JSON report for a fully-analysed run.

    Args:
        run_id: The run to report on.
        config_path: Path to ``evalshift.yaml``. Loaded best-effort.
        runs_base: Base directory for run state.
        insights: Generate (or reuse) the machine-written narrative. ``False``
            for ``--no-insights``.
        client: Model client the narrative is generated with. ``None`` builds a
            real one; the parameter exists so tests never reach a provider.
    """
    run_dir = run_dir_for(run_id, runs_base)
    state = read_state(run_dir)  # validates state.json exists + parses

    # Best-effort config load: drives the per-example "tool match"
    # column. If the user moved the config or the report runs against
    # a foreign run dir, fall back to no tool detection.
    tool_evaluator_names: frozenset[str] = frozenset()
    judge_models: dict[str, str] = {}
    try:
        cfg = load_config(config_path)
    except ConfigError:
        cfg = None
    if cfg is not None:
        # Resolved for the suite the run was launched against, so a
        # per-suite evaluator block reaches the report the same way it
        # reached scoring.
        evaluators = cfg.evaluators_for(state.suite_name)
        tool_evaluator_names = evaluators.tool_evaluator_names
        # Same name rule as evaluate's PairwiseJudgeEvaluator construction,
        # so the judge-family note matches judges to the rows they wrote.
        judge_models = {
            f"llm_judge.{j.criterion_name}": j.judge_model for j in evaluators.llm_judge
        }

    payload = build_report_payload(
        run_dir,
        tool_evaluator_names=tool_evaluator_names,
        judge_models=judge_models,
    )

    # The narrative is prose *around* the payload, never part of it —
    # ``report.json`` stays the computed figures, and a failed generation
    # leaves ``insight`` None rather than failing the report.
    insight = ensure_insight(run_dir, cfg=cfg, enabled=insights, client=client)

    json_path = write_report_json(payload, run_dir)
    html_path = write_html(payload, run_dir, insight=insight)

    return ReportResult(run_id=run_id, html_path=html_path, json_path=json_path)


def report(
    run_id: Annotated[
        str,
        typer.Argument(help="Run id to report on."),
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
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open",
            help="Open the rendered HTML report in your default browser.",
        ),
    ] = False,
    insights: Annotated[
        bool,
        typer.Option(
            "--insights/--no-insights",
            help="Write a plain-language explanation of the run (one extra LLM call).",
        ),
    ] = True,
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
    """Render a single-file HTML report from a fully-analysed run."""
    console = Console()

    try:
        result = run_report(
            run_id=run_id,
            config_path=config_path,
            runs_base=runs_base,
            insights=insights,
        )
    except CheckpointError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}")
        console.print(
            "Hint: run [cyan]evalshift evaluate[/cyan] then "
            "[cyan]evalshift analyze[/cyan] before [cyan]report[/cyan].",
        )
        raise typer.Exit(code=1) from exc

    console.print(f"[green]✓[/green] {result.html_path}")
    console.print(f"[green]✓[/green] {result.json_path}")
    print_star_prompt(console)

    if open_browser:
        webbrowser.open(result.html_path.resolve().as_uri())


__all__ = ["ReportResult", "report", "run_report"]
