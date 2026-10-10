"""Implementation of ``evalshift run`` — the headline command.

Loads config + suite, dispatches to the async orchestrator, and prints
a human-friendly summary on completion. Friendly errors via Rich for
every failure mode the orchestrator can raise.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from evalshift_cli.cli.commands._suites import (
    SUITE_FILENAME,
    AmbiguousSuiteError,
    UnknownSuiteNameError,
    derive_suite_slug,
    resolve_suite_path,
)
from evalshift_cli.cli.commands.doctor import CONFIG_FILENAME
from evalshift_cli.cli.commands.evaluate import preflight_evaluator_keys
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.evaluators.tool_loader import ToolLoaderError
from evalshift_cli.models.registry import (
    Provider,
    UnknownModelError,
    missing_api_keys,
    resolve_model,
)
from evalshift_cli.parsers.base import PromptParseError
from evalshift_cli.runner.orchestrator import (
    RunAborted,
    RunResult,
    run_orchestrator,
)
from evalshift_cli.suite.loader import SuiteError, load_jsonl
from evalshift_cli.utils.templating import SuiteCompatibilityError

# Back-compat alias: the resolver now lives in ``_suites``. Some callers/tests
# still import ``_resolve_suite_path`` from this module.
_resolve_suite_path = resolve_suite_path


def run(
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
            help="Named suite from evalshift.yaml suites: (e.g. a promoted capture).",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Continue the most recent in-progress run.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help=(
                "Skip the cost-confirmation prompt. Also set automatically "
                "when EVALSHIFT_NONINTERACTIVE is set in the environment."
            ),
        ),
    ] = False,
) -> None:
    """Run paired evaluation on two models against your golden suite."""
    console = Console()

    if not yes and os.environ.get("EVALSHIFT_NONINTERACTIVE", "").strip():
        yes = True

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
        resolved_suite_path = _resolve_suite_path(
            suite_path=suite_path,
            suite_name=suite_name,
            cfg=cfg,
            config_path=config_path,
        )
    except (UnknownSuiteNameError, AmbiguousSuiteError) as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    try:
        suite = load_jsonl(resolved_suite_path)
    except SuiteError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    missing = _missing_api_keys((source_model, target_model), os.environ)
    if missing:
        _print_missing_api_keys(console, missing)
        raise typer.Exit(code=1)

    preflight_evaluator_keys(console, cfg.evaluators_for(suite_name), os.environ)

    try:
        result = asyncio.run(
            run_orchestrator(
                config=cfg,
                config_path=config_path,
                suite=suite,
                suite_path=resolved_suite_path,
                source_model=source_model,
                target_model=target_model,
                resume=resume,
                yes=yes,
                run_slug=derive_suite_slug(suite_name=suite_name, suite_path=resolved_suite_path),
                suite_name=suite_name,
                console=console,
            ),
        )
    except PromptParseError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc
    except SuiteCompatibilityError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc
    except RunAborted as exc:
        console.print(f"[yellow]⚠[/yellow] aborted: {exc}")
        raise typer.Exit(code=1) from exc
    except UnknownModelError as exc:
        # Should be unreachable now that resolve_model is permissive, but
        # keep this as a defensive net so any future strict-path leak
        # surfaces as a friendly error instead of a traceback.
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ToolLoaderError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    _print_summary(console, result)


def _print_summary(console: Console, result: RunResult) -> None:
    body = (
        f"[bold]{result.run_id}[/bold]\n\n"
        f"[dim]calls:[/dim]   "
        f"[green]{result.completed_calls}[/green]/{result.total_calls} completed  "
        f"({result.cached_calls} cached, "
        f"{result.live_calls} live, "
        f"[red]{result.failed_calls} failed[/red])\n"
        f"[dim]cost:[/dim]    ${result.total_cost_usd:.4f}\n"
        f"[dim]outputs:[/dim] {result.run_dir / 'raw.jsonl'}\n\n"
        f"[bold]Next:[/bold] [cyan]evalshift evaluate {result.run_id}[/cyan]"
    )
    console.print(
        Panel(
            body,
            border_style="green",
            title="evalshift run",
            title_align="left",
        ),
    )


def _missing_api_keys(
    models: tuple[str, ...] | list[str],
    env: os._Environ[str],
) -> list[tuple[str, Provider, tuple[str, ...]]]:
    """Return one entry per model whose provider has no recognised key set.

    Models with provider ``"other"`` (we couldn't infer a provider) are
    skipped — we don't know what key they'd need. Duplicates are
    de-duped so the same model isn't reported twice when source ==
    target.
    """
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


def _print_missing_api_keys(
    console: Console,
    missing: list[tuple[str, Provider, tuple[str, ...]]],
) -> None:
    lines = ["[red]✗[/red] missing API key(s) for the requested model(s):", ""]
    for model, provider, keys in missing:
        key_str = " or ".join(f"[bold]{k}[/bold]" for k in keys)
        lines.append(f"    [bold]{model}[/bold] ({provider}) — export {key_str}")
    lines.append("")
    lines.append("Run [cyan]evalshift doctor[/cyan] for a full check.")
    console.print("\n".join(lines))


__all__ = ["run"]
