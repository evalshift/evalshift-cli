"""The GitHub-star request printed after a report is rendered.

``report`` and ``compare`` both end by pointing at a freshly written
``report.html``; that is the moment a person has just got value from the
tool, so it is where the request goes. It is for a person at a terminal
only: CI logs, the GitHub Action's captured output and anything piped stay
free of it.
"""

from __future__ import annotations

from typing import Final

from rich.console import Console

STAR_URL: Final = "https://github.com/evalshift/evalshift-cli"


def print_star_prompt(console: Console) -> None:
    """Ask for a GitHub star, on interactive terminals only.

    Args:
        console: The command's console. Nothing is printed unless it is a
            terminal.
    """
    if not console.is_terminal:
        return
    console.print()
    # soft_wrap: a URL folded across lines stops being clickable.
    console.print(
        f"[bold yellow]★[/bold yellow] Find EvalShift useful? "
        f"[bold]A star on GitHub[/bold] helps: [cyan]{STAR_URL}[/cyan]",
        soft_wrap=True,
    )


__all__ = ["STAR_URL", "print_star_prompt"]
