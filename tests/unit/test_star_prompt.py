"""Tests for the GitHub-star request printed after a report is rendered.

The line is for a person at a terminal. CI logs, the GitHub Action and
anything piped must not carry it, so it keys off ``Console.is_terminal``.
"""

from __future__ import annotations

import io

from rich.console import Console

from evalshift_cli.cli.commands._star import STAR_URL, print_star_prompt


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=terminal, width=80), buf


class TestPrintStarPrompt:
    def test_terminal_gets_the_repo_link(self) -> None:
        console, buf = _console(terminal=True)
        print_star_prompt(console)
        assert STAR_URL in buf.getvalue()

    def test_sits_apart_from_the_lines_above(self) -> None:
        # It follows the run/suite/report block; a blank line keeps it from
        # reading as a fourth row of that block.
        console, buf = _console(terminal=True)
        print_star_prompt(console)
        assert buf.getvalue().startswith("\n")

    def test_piped_output_gets_nothing(self) -> None:
        console, buf = _console(terminal=False)
        print_star_prompt(console)
        assert buf.getvalue() == ""

    def test_link_survives_a_narrow_terminal_on_one_line(self) -> None:
        # A folded URL stops being clickable; the line must soft-wrap instead.
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=40)
        print_star_prompt(console)
        assert STAR_URL in buf.getvalue()

    def test_url_points_at_the_cli_repo(self) -> None:
        assert STAR_URL == "https://github.com/evalshift/evalshift-cli"
