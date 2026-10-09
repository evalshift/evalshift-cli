"""Implementation of ``evalshift compare``: the full pipeline in one command.

Runs ``doctor → run → evaluate → analyze → report`` end to end with a
single ``rich.live.Live`` UI: stacked status rows for each stage, an
inline block-bar for the run stage, and a final verdict block printed
after the Live region closes.

Registered under ``compare`` and, permanently, under the former name
``all`` -- see :func:`evalshift_cli.cli.main` for the aliasing. Both names
bind this one function, so there is no second code path to keep in step.

The five stages are driven by the reusable cores in their respective
command modules (``run_evaluate``, ``run_analyze``, ``run_report``);
``doctor`` calls into ``run_checks`` directly. The cost-confirmation
prompt is handled here (outside ``Live``) so the orchestrator can run
unattended.
"""

from __future__ import annotations

import asyncio
import logging
import os
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text

from evalshift_cli.analysis.statistics import ComparisonResult
from evalshift_cli.cli.commands._suites import (
    SUITE_FILENAME,
    AmbiguousSuiteError,
    UnknownSuiteNameError,
    derive_suite_slug,
    resolve_suite_path,
)
from evalshift_cli.cli.commands.analyze import (
    GATE_SEVERITIES,
    EmptyScoresError,
    MissingScoresError,
    format_collapsed_slices,
    run_analyze,
)
from evalshift_cli.cli.commands.doctor import (
    CONFIG_FILENAME,
    EVALUATOR_KEYS_CHECK,
    render_results,
    run_checks,
)
from evalshift_cli.cli.commands.evaluate import (
    NoEvaluatorsError,
    NoPairsError,
    preflight_evaluator_keys,
    run_evaluate,
)
from evalshift_cli.cli.commands.report import run_report
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.config.models import EvaluatorsConfig
from evalshift_cli.evaluators.keys import LLM_JUDGE_KIND, SEMANTIC_EVALUATOR_NAME, EvaluatorKeyGap
from evalshift_cli.evaluators.tool_loader import ToolLoaderError
from evalshift_cli.hosted.push import PushError, push_local_run
from evalshift_cli.models.client import deferred_console_warnings
from evalshift_cli.models.registry import (
    Provider,
    missing_api_keys,
    resolve_model,
)
from evalshift_cli.parsers.base import PromptParseError
from evalshift_cli.runner.orchestrator import (
    COST_CONFIRM_THRESHOLD_USD,
    ProgressEvent,
    RunAborted,
    RunResult,
    preflight_cost,
    run_orchestrator,
)
from evalshift_cli.suite.loader import SuiteError, load_jsonl
from evalshift_cli.utils.templating import SuiteCompatibilityError

# ---------------------------------------------------------------------------
# Status row state
# ---------------------------------------------------------------------------

StageStatus = Literal["pending", "running", "done", "failed"]

LABEL_WIDTH: int = 14
TOTAL_WIDTH: int = 60  # column at which the right-side payload terminates
BAR_WIDTH: int = 10


@dataclass(slots=True)
class StageRow:
    """Mutable state for one row in the Live region."""

    label: str
    status: StageStatus = "pending"
    payload: str = ""
    use_dots: bool = True


@dataclass(slots=True)
class _RunProgress:
    """Live counters from ``ProgressEvent`` callbacks."""

    completed: int = 0
    total: int = 0
    cached: int = 0


_GLYPH: dict[StageStatus, tuple[str, str]] = {
    "pending": (" ", "dim"),
    "running": ("▸", "cyan"),
    "done": ("◇", "green"),
    "failed": ("✗", "red"),
}


def _bar(completed: int, total: int, width: int = BAR_WIDTH) -> str:
    if total <= 0:
        return "▱" * width
    filled = min(width, round(completed / total * width))
    return "▰" * filled + "▱" * (width - filled)


def _row_text(row: StageRow, run_progress: _RunProgress | None = None) -> Text:
    """Render one stage row as a styled ``rich.text.Text``."""
    glyph, glyph_style = _GLYPH[row.status]
    text = Text()
    text.append(f"{glyph} ", style=glyph_style)
    text.append(row.label.ljust(LABEL_WIDTH), style="bold")

    if row.label == "run" and row.status == "running" and run_progress is not None:
        bar = _bar(run_progress.completed, run_progress.total)
        text.append(bar, style="cyan")
        text.append(
            f"  {run_progress.completed}/{run_progress.total}"
            f" · cache {run_progress.cached}/{run_progress.total}",
            style="dim",
        )
        return text

    if row.use_dots and row.payload:
        # Dotted leader filling out to TOTAL_WIDTH, leaving space for payload.
        prefix_len = 2 + LABEL_WIDTH  # "X " + label
        dots_len = max(3, TOTAL_WIDTH - prefix_len - len(row.payload) - 1)
        text.append(" " + "." * dots_len + " ", style="dim")
        text.append(row.payload)
    elif row.payload:
        text.append(row.payload, style="dim")
    return text


def _render_pipeline(rows: list[StageRow], run_progress: _RunProgress | None) -> RenderableType:
    return Group(*(_row_text(r, run_progress) for r in rows))


# ---------------------------------------------------------------------------
# Verdict picker
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Verdict:
    headline: Text
    detail: Text | None
    regression_callout: Text | None


_REGRESSION_SEVERITIES: frozenset[str] = frozenset({"critical", "high"})
_MINOR_SEVERITIES: frozenset[str] = frozenset({"medium", "low"})


def _compose_verdict(comparisons: list[ComparisonResult]) -> _Verdict:
    """Pick a one-line verdict + detail line from analysis results.

    Priority: any critical/high regression → red regression headline.
    Else any improved row → green "significantly better" headline keyed on
    the largest |effect_size| improvement. Otherwise a yellow no-change line.
    Minor (medium/low) regressions are reported as a separate callout.
    """
    regressions = [c for c in comparisons if c.severity in _REGRESSION_SEVERITIES]
    improvements = [c for c in comparisons if c.severity == "improved"]
    minor = [c for c in comparisons if c.severity in _MINOR_SEVERITIES]

    if regressions:
        headline_pick = max(regressions, key=lambda c: abs(c.effect_size))
        headline = Text("✗ candidate regressed", style="bold red")
        detail = _detail_line(headline_pick)
    elif improvements:
        headline_pick = max(improvements, key=lambda c: abs(c.effect_size))
        headline = Text("✓ candidate is significantly better", style="bold green")
        detail = _detail_line(headline_pick)
    else:
        headline = Text("~ no significant change", style="bold yellow")
        detail = None

    callout: Text | None = None
    if minor:
        worst = max(minor, key=lambda c: abs(c.effect_size))
        plural = "s" if len(minor) != 1 else ""
        callout = Text()
        callout.append("⚠ ", style="yellow")
        callout.append(
            f"{len(minor)} sub-metric{plural} regressed "
            f"({worst.evaluator_name}; n.s. q={worst.p_value_corrected:.2f})",
            style="yellow",
        )

    return _Verdict(headline=headline, detail=detail, regression_callout=callout)


def _detail_line(c: ComparisonResult) -> Text:
    detail = Text()
    detail.append("  ", style="")
    detail.append(
        f"Δ {c.delta_avg_score:+.3f} · "
        f"d {c.effect_size:.2f} "
        f"[{c.effect_size_ci_low:.2f}, {c.effect_size_ci_high:.2f}] · "
        f"p {c.p_value:.3f} · "
        f"q {c.p_value_corrected:.3f}",
        style="dim",
    )
    return detail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluator_family_summary(
    evaluators: EvaluatorsConfig,
    skipped: Sequence[EvaluatorKeyGap] = (),
) -> str:
    """Name the evaluator families in an already-resolved set.

    Takes the resolved set rather than the whole config so the pipeline row
    names what *this* suite is scored with — on a heterogeneous project the
    top-level block is not what runs. A family whose model has no key is
    marked ``(no key)``: listed as configured, never as having scored.
    """
    skipped_names = {gap.evaluator_name for gap in skipped}
    families: list[str] = []
    if evaluators.structural:
        families.append("structural")
    if evaluators.semantic is not None:
        families.append(
            "semantic (no key)" if SEMANTIC_EVALUATOR_NAME in skipped_names else "semantic"
        )
    if evaluators.llm_judge:
        all_skipped = all(
            f"{LLM_JUDGE_KIND}.{j.criterion_name}" in skipped_names for j in evaluators.llm_judge
        )
        families.append("judge (no key)" if all_skipped else "judge")
    if evaluators.tool_selection or evaluators.tool_arguments or evaluators.tool_trace_structure:
        families.append("tool-call")
    return " · ".join(families) if families else "(none)"


def _missing_keys(
    models: tuple[str, ...],
    env: os._Environ[str],
) -> list[tuple[str, Provider, tuple[str, ...]]]:
    seen: set[str] = set()
    missing: list[tuple[str, Provider, tuple[str, ...]]] = []
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        keys = missing_api_keys(m, env)
        if keys:
            missing.append((m, resolve_model(m).provider, keys))
    return missing


def _parse_gate(raw: str) -> frozenset[str]:
    if not raw.strip():
        return frozenset()
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    unknown = sorted(set(parts) - GATE_SEVERITIES)
    if unknown:
        allowed = ", ".join(sorted(GATE_SEVERITIES))
        raise ValueError(f"unknown --gate severity {unknown}; allowed: {allowed}")
    return frozenset(parts)


def _print_deferred_warnings(console: Console, records: list[logging.LogRecord]) -> None:
    """Print the warnings deferred during the pipeline, under the block.

    Each line names its origin (``LiteLLM: …``) except our own records, whose
    messages already say what happened. A blank line sets the section off
    from the pipeline block above it.
    """
    if not records:
        return
    console.print()
    for record in records:
        origin = record.name.split(".", 1)[0]
        prefix = "" if origin == "evalshift_cli" else f"{origin}: "
        line = Text()
        line.append("⚠ ", style="yellow")
        line.append(f"{prefix}{record.getMessage()}", style="dim")
        console.print(line)


def _confirm_cost(console: Console, est_usd: float, total_calls: int) -> bool:
    console.print(
        f"This run will make [bold]{total_calls}[/bold] LLM calls "
        f"(at most [bold]${est_usd:.2f}[/bold]).",
    )
    console.print("Continue? [Y/n] ", end="")
    answer = (input().strip() or "y").lower()
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# The ``evalshift compare`` command
# ---------------------------------------------------------------------------


#: The name this command shipped under before it was ``compare``. Kept
#: registered forever: ``evalshift init`` writes ``EVALSHIFT.md`` into user
#: repos telling agents to run it, and that file is never regenerated.
LEGACY_COMMAND_NAME: str = "all"


def _warn_if_legacy_name(ctx: typer.Context) -> None:
    """Print a rename notice when invoked as ``all`` rather than ``compare``.

    Written to stderr so a pipeline parsing stdout is unaffected, and phrased
    as a notice rather than a warning: the alias is supported, not scheduled
    for removal. Says only that the name moved -- what the command actually
    scopes to is the suite-selection error's job, and repeating it here would
    print the same sentence twice.

    Args:
        ctx: The Typer context, whose ``info_name`` is the name typed.
    """
    if ctx.info_name != LEGACY_COMMAND_NAME:
        return
    Console(stderr=True).print(
        "[yellow]![/yellow] [bold]all[/bold] is now [bold]evalshift compare[/bold]. "
        "The old name keeps working.",
    )


def compare_command(
    ctx: typer.Context,
    source: Annotated[
        str | None,
        typer.Option(
            "--from",
            "-f",
            help="Source model id or alias. Overrides config defaults.source_model.",
        ),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option(
            "--to",
            "-t",
            help="Target model id or alias. Overrides config defaults.target_model.",
        ),
    ] = None,
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
    suite_path: Annotated[
        Path | None,
        typer.Option(
            "--suite",
            "-s",
            help=f"Path to the JSONL suite (default: ./{SUITE_FILENAME}).",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    suite_name: Annotated[
        str | None,
        typer.Option(
            "--suite-name",
            help=(
                "Named suite from evalshift.yaml suites: (e.g. a promoted capture). "
                "Required once more than one suite is wired."
            ),
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Continue the most recent in-progress run."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the cost-confirmation prompt.",
        ),
    ] = False,
    gate: Annotated[
        str,
        typer.Option(
            "--gate",
            help="CI gate: comma-separated severities that should fail with exit 1.",
        ),
    ] = "",
    policy_gate: Annotated[
        bool,
        typer.Option(
            "--policy-gate",
            help="CI gate: fail when migration_policy verdict is fail or conditional_pass.",
        ),
    ] = False,
    open_browser: Annotated[
        bool,
        typer.Option("--open", help="Open the rendered HTML report in your browser."),
    ] = False,
    insights: Annotated[
        bool,
        typer.Option(
            "--insights/--no-insights",
            help="Write a plain-language explanation of the run (one extra LLM call).",
        ),
    ] = True,
    push: Annotated[
        bool,
        typer.Option("--push", help="Push the completed run bundle to hosted EvalShift."),
    ] = False,
    runs_base: Annotated[
        Path,
        typer.Option("--runs-base", help="Base directory for run state (advanced).", hidden=True),
    ] = Path(".evalshift") / "runs",
) -> None:
    """Compare two models on one suite: doctor, run, evaluate, analyze, report.

    One invocation compares two models on a single suite. The suite is
    auto-selected when evalshift.yaml wires exactly one; with several, name it
    with --suite-name. To cover every suite, loop over them in the shell (or
    use one CI job per suite).

    Formerly named 'all'. That name still works but is no longer advertised.
    """
    console = Console()
    _warn_if_legacy_name(ctx)

    if not yes and os.environ.get("EVALSHIFT_NONINTERACTIVE", "").strip():
        yes = True

    try:
        gate_severities = _parse_gate(gate)
    except ValueError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    source_model = source or cfg.defaults.source_model
    target_model = target or cfg.defaults.target_model
    if not source_model or not target_model:
        console.print(
            "[red]✗[/red] missing model selection: pass [bold]--from[/bold] / "
            "[bold]--to[/bold] or set [bold]defaults.source_model[/bold] / "
            "[bold]defaults.target_model[/bold] in evalshift.yaml.",
        )
        raise typer.Exit(code=1)

    try:
        suite_path = resolve_suite_path(
            suite_path=suite_path,
            suite_name=suite_name,
            cfg=cfg,
            config_path=config_path,
        )
    except (UnknownSuiteNameError, AmbiguousSuiteError) as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    try:
        suite = load_jsonl(suite_path)
    except SuiteError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    run_slug = derive_suite_slug(suite_name=suite_name, suite_path=suite_path)

    missing = _missing_keys((source_model, target_model), os.environ)
    if missing:
        for model, provider, keys in missing:
            key_str = " or ".join(keys)
            console.print(
                f"[red]✗[/red] missing API key for [bold]{model}[/bold] "
                f"({provider}); export {key_str}.",
            )
        raise typer.Exit(code=1)

    # The judge and embedding models make calls of their own; check them now,
    # before the doctor stage and long before the first arm call is paid for.
    skipped_gaps = preflight_evaluator_keys(console, cfg.evaluators_for(suite_name), os.environ)

    rows = [
        StageRow(label="doctor"),
        StageRow(label="max cost"),
        StageRow(label="run", use_dots=False),
        StageRow(label="evaluate", use_dots=False),
        StageRow(label="analyze", use_dots=False),
    ]
    run_progress = _RunProgress()

    def update(live: Live) -> None:
        live.update(_render_pipeline(rows, run_progress))

    # Air between the shell prompt and the pipeline block (or its failure
    # render) — everything printed from here on belongs to the block.
    console.print()

    # Pre-Live silent preflight: doctor + cost. Both are sub-second; we
    # render them as already-completed rows once Live opens, which keeps
    # the entire pipeline inside a single Live region (no double-render).
    check_results = run_checks(Path.cwd(), os.environ)
    # doctor's evaluator-key rows cover every named suite; this run's own
    # suite-scoped preflight_evaluator_keys has already run and is the
    # authority here, so another suite's keyless judge must not stop it.
    failed_checks = [
        r for r in check_results if r.status == "fail" and r.name != EVALUATOR_KEYS_CHECK
    ]
    if failed_checks:
        first = failed_checks[0]
        extra = f" (+{len(failed_checks) - 1} more)" if len(failed_checks) > 1 else ""
        rows[0].status = "failed"
        rows[0].payload = f"{first.name}: {first.detail}{extra}"
        console.print(_render_pipeline(rows, None))
        raise typer.Exit(code=1)
    rows[0].status = "done"
    rows[0].payload = "ok"

    try:
        plan = preflight_cost(
            config=cfg,
            config_path=config_path,
            suite=suite,
            source_model=source_model,
            target_model=target_model,
        )
    except (PromptParseError, SuiteCompatibilityError, ToolLoaderError) as exc:
        rows[1].status = "failed"
        rows[1].payload = str(exc)
        console.print(_render_pipeline(rows, None))
        raise typer.Exit(code=1) from exc
    rows[1].status = "done"
    rows[1].payload = f"≤ ${plan.estimated_usd:.2f}"

    # Cost-confirmation prompt before Live so the input() doesn't fight
    # with the live region.
    if (
        not resume
        and not yes
        and plan.estimated_usd > COST_CONFIRM_THRESHOLD_USD
        and not _confirm_cost(console, plan.estimated_usd, plan.total_calls)
    ):
        console.print("[yellow]⚠[/yellow] aborted: user declined the cost prompt")
        raise typer.Exit(code=1)

    # Single Live region for stages 3-5; rows 0-1 already show "done".
    # Warnings raised anywhere in stages 3-6 (LiteLLM deprecation notices,
    # insights-retry notes) are deferred and printed as one section directly
    # under the block — also when a stage fails, since a held-back warning
    # may explain the failure printed above it.
    run_result: RunResult | None = None
    with deferred_console_warnings() as deferred_warnings:
        try:
            with Live(
                _render_pipeline(rows, run_progress),
                console=console,
                refresh_per_second=10,
            ) as live:
                # Stage 3: run.
                rows[2].status = "running"
                run_progress.total = plan.total_calls
                update(live)

                def _on_progress(ev: ProgressEvent) -> None:
                    run_progress.completed = ev.completed
                    run_progress.total = ev.total
                    run_progress.cached = ev.cached
                    update(live)

                try:
                    run_result = asyncio.run(
                        run_orchestrator(
                            config=cfg,
                            config_path=config_path,
                            suite=suite,
                            suite_path=suite_path,
                            source_model=source_model,
                            target_model=target_model,
                            runs_base=runs_base,
                            resume=resume,
                            yes=True,  # cost prompt already handled above
                            run_slug=run_slug,
                            suite_name=suite_name,
                            console=console,
                            on_progress=_on_progress,
                        ),
                    )
                except (
                    PromptParseError,
                    SuiteCompatibilityError,
                    RunAborted,
                    ToolLoaderError,
                ) as exc:
                    rows[2].status = "failed"
                    rows[2].payload = str(exc)
                    update(live)
                    raise typer.Exit(code=1) from exc
                rows[2].status = "done"
                rows[2].use_dots = True
                rows[2].payload = (
                    f"{run_result.completed_calls}/{run_result.total_calls}"
                    f" · cache {run_result.cached_calls}/{run_result.total_calls}"
                )
                update(live)

                # Stage 4: evaluate.
                rows[3].status = "running"
                rows[3].payload = _evaluator_family_summary(
                    cfg.evaluators_for(suite_name), skipped_gaps
                )
                update(live)
                try:
                    evaluate_result = run_evaluate(
                        run_id=run_result.run_id,
                        config_path=config_path,
                        runs_base=runs_base,
                        console=console,
                        quiet=True,
                    )
                except (NoEvaluatorsError, NoPairsError) as exc:
                    rows[3].status = "failed"
                    rows[3].payload = str(exc)
                    update(live)
                    raise typer.Exit(code=1) from exc
                except ConfigError as exc:
                    # The preflight above makes this unreachable with the same
                    # environment; kept so a config edited mid-run fails the
                    # row instead of escaping the Live region.
                    rows[3].status = "failed"
                    rows[3].payload = exc.summary
                    update(live)
                    raise typer.Exit(code=1) from exc
                rows[3].status = "done"
                update(live)

                # Stage 5: analyze.
                rows[4].status = "running"
                rows[4].payload = "paired-t · cohen's d · BH-FDR"
                update(live)
                try:
                    analyze_result = run_analyze(
                        run_id=run_result.run_id,
                        config_path=config_path,
                        runs_base=runs_base,
                    )
                except (MissingScoresError, EmptyScoresError) as exc:
                    rows[4].status = "failed"
                    rows[4].payload = str(exc)
                    update(live)
                    raise typer.Exit(code=1) from exc
                rows[4].status = "done"
                update(live)

            # Stage 6: report (silent — not in the Live grid).
            report_result = run_report(
                run_id=run_result.run_id,
                config_path=config_path,
                runs_base=runs_base,
                insights=insights,
            )
        finally:
            # On file consoles Live writes its final frame without a trailing
            # newline (on terminals ``stop()`` adds one), so the next printed
            # newline would terminate the frame's last row instead of making
            # a blank line. Close the frame so both modes render alike.
            if not console.is_terminal:
                console.print()
            _print_deferred_warnings(console, deferred_warnings)

    # Verdict block. The broken-harness finding goes immediately above it,
    # unconditionally: the verdict below is arithmetic over a suite the source
    # model could not satisfy, so it must not be the first thing read. Stage 4
    # scored quietly — this table would have fought with the Live grid — and
    # handed the finding back instead.
    console.print()
    if evaluate_result.harness_check is not None:
        render_results([evaluate_result.harness_check], console)
        console.print()
    collapsed_note = format_collapsed_slices(analyze_result.collapsed_slices)
    if collapsed_note is not None:
        console.print(collapsed_note, style="dim")
        console.print()
    if analyze_result.migration_decision is not None:
        decision = analyze_result.migration_decision
        style = {
            "pass": "bold green",
            "conditional_pass": "bold yellow",
            "fail": "bold red",
            "inconclusive": "bold yellow",
        }.get(decision.verdict, "bold")
        console.print(f"[{style}]Migration verdict: {decision.verdict}[/{style}]")
        if decision.reason is not None:
            console.print(f"  {decision.reason}", style="dim")
        for rec in decision.recommendations:
            # Recommendations are wrapping paragraphs; unseparated they read
            # as one wall of text.
            console.print()
            console.print(f"  {rec}", style="dim")
    else:
        verdict = _compose_verdict(list(analyze_result.comparisons))
        console.print(verdict.headline)
        if verdict.detail is not None:
            console.print(verdict.detail)
        if verdict.regression_callout is not None:
            console.print(verdict.regression_callout)

    console.print()
    console.print(f"[dim]run   [/dim] [bold cyan]{run_result.run_id}[/bold cyan]")
    console.print(f"[dim]suite [/dim] {run_slug}")
    console.print(f"[dim]report[/dim] {report_result.html_path}")

    if push:
        try:
            push_result = push_local_run(
                run_id=run_result.run_id,
                config_path=config_path,
                suite_path=suite_path,
                runs_base=runs_base,
                console=console,
            )
        except PushError as exc:
            console.print(f"[red]✗ hosted push failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        # soft_wrap: same URL `push` prints, same reason it must survive a pipe
        # intact — see the note in cli/commands/push.py.
        console.print(f"[dim]hosted:[/dim] {push_result.view_url}", soft_wrap=True)

    if open_browser:
        webbrowser.open(report_result.html_path.resolve().as_uri())

    if gate_severities:
        offending = [c for c in analyze_result.comparisons if c.severity in gate_severities]
        if offending:
            console.print(
                f"[red]✗ gate failed:[/red] {len(offending)} comparison(s) at "
                f"severity {{{', '.join(sorted(gate_severities))}}}",
            )
            raise typer.Exit(code=1)

    if policy_gate:
        if analyze_result.migration_decision is None:
            console.print("[red]✗ policy gate failed:[/red] no migration_policy configured")
            raise typer.Exit(code=1)
        if analyze_result.migration_decision.verdict in {"fail", "conditional_pass"}:
            console.print(
                f"[red]✗ policy gate failed:[/red] {analyze_result.migration_decision.verdict}",
            )
            raise typer.Exit(code=1)


__all__ = ["LEGACY_COMMAND_NAME", "compare_command"]
