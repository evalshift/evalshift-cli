"""Load and validate ``evalshift.yaml``.

This module is the single entry point for turning a path on disk into a fully
validated :class:`~evalshift_cli.config.models.EvalShiftConfig`. Errors raised
from here are :class:`ConfigError` instances, which carry both a plain-text
representation (used in tests and tracebacks) and a Rich-renderable
representation (used by the CLI for nicer terminal output).

Three failure modes are normalised into ``ConfigError``:

* **Missing file**: the path doesn't exist or isn't a regular file.
* **YAML parse error**: the file isn't valid YAML.
* **Schema violation**: the YAML parses, but doesn't match
  :class:`EvalShiftConfig`. The pydantic errors are flattened into a list of
  ``location``/``message`` pairs so users can fix one thing at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from evalshift_cli.config.models import EvalShiftConfig

ConfigErrorKind = Literal[
    "missing", "not_a_file", "yaml_parse", "not_a_mapping", "schema", "missing_key"
]


#: Panel titles for the kinds that are not a problem with the file itself.
_TITLES: Final[Mapping[ConfigErrorKind, str]] = {"missing_key": "Missing API key"}


@dataclass(frozen=True, slots=True)
class ConfigErrorDetail:
    """A single human-readable problem found in the config file.

    Attributes:
        location: Where in the file the problem is. For schema errors this is
            a dotted field path (e.g. ``prompts.0.content``); for YAML parse
            errors it's a ``line N, column M`` pointer.
        message: Plain-English description of what's wrong and (where
            possible) how to fix it.
    """

    location: str
    message: str


class ConfigError(Exception):
    """Raised when ``evalshift.yaml`` is missing, unparseable, or invalid.

    Catch this in CLI commands to render a friendly error and exit non-zero;
    the loader itself never prints to stderr directly.
    """

    def __init__(
        self,
        path: Path,
        kind: ConfigErrorKind,
        summary: str,
        details: list[ConfigErrorDetail] | None = None,
    ) -> None:
        self.path = path
        self.kind: ConfigErrorKind = kind
        self.summary = summary
        self.details: list[ConfigErrorDetail] = list(details or [])
        super().__init__(self.format_plain())

    def format_plain(self) -> str:
        """Render this error as a multi-line plain-text string."""
        lines = [f"Configuration error in {self.path}:", f"  {self.summary}"]
        if self.details:
            lines.append("")
            for d in self.details:
                lines.append(f"  • [{d.location}] {d.message}")
        return "\n".join(lines)

    def format_rich(self) -> RenderableType:
        """Render this error inside a Rich :class:`Panel` for terminal output.

        The panel is titled by kind: a ``missing_key`` error is raised over a
        valid config, so it reads "Missing API key", not "Invalid config".
        """
        body: list[RenderableType] = [Text(self.summary, style="bold red")]
        if self.details:
            body.append(Text(""))
            for d in self.details:
                line = Text()
                line.append("• ", style="red")
                line.append(d.location, style="bold cyan")
                line.append(": ")
                line.append(d.message)
                body.append(line)
        return Panel(
            Group(*body),
            title=f"[red]{_TITLES.get(self.kind, 'Invalid config')}[/red]: {self.path}",
            title_align="left",
            border_style="red",
        )


def load_config(path: str | Path) -> EvalShiftConfig:
    """Load and validate an ``evalshift.yaml`` file.

    Args:
        path: Path to the YAML file. Accepts either ``str`` or :class:`Path`.

    Returns:
        A fully validated :class:`EvalShiftConfig`.

    Raises:
        ConfigError: If the file is missing, unparseable, or fails schema
            validation. The exception carries a list of all problems found,
            so callers can report several issues at once instead of one per
            run-fix-rerun cycle.
    """
    cfg_path = Path(path)

    if not cfg_path.exists():
        raise ConfigError(
            path=cfg_path,
            kind="missing",
            summary=f"file not found: {cfg_path}",
        )
    if not cfg_path.is_file():
        raise ConfigError(
            path=cfg_path,
            kind="not_a_file",
            summary=f"expected a file, got {cfg_path} (is it a directory?)",
        )

    raw_text = cfg_path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise _yaml_error_to_config_error(cfg_path, exc) from exc

    if data is None:
        raise ConfigError(
            path=cfg_path,
            kind="not_a_mapping",
            summary="config file is empty — expected a mapping with at least 'prompts'",
        )
    if not isinstance(data, dict):
        raise ConfigError(
            path=cfg_path,
            kind="not_a_mapping",
            summary=(f"top-level YAML node must be a mapping (object), got {type(data).__name__}"),
        )

    try:
        return EvalShiftConfig.model_validate(data)
    except ValidationError as exc:
        raise _validation_error_to_config_error(cfg_path, exc) from exc


def _yaml_error_to_config_error(path: Path, exc: yaml.YAMLError) -> ConfigError:
    """Convert a :class:`yaml.YAMLError` into a friendly :class:`ConfigError`."""
    mark = getattr(exc, "problem_mark", None)
    if mark is not None:
        # PyYAML uses 0-indexed line/column internally; humans want 1-indexed.
        location = f"line {mark.line + 1}, column {mark.column + 1}"
        problem = getattr(exc, "problem", None) or "invalid YAML"
        context = getattr(exc, "context", None)
        message = f"{problem}{f' ({context})' if context else ''}"
        return ConfigError(
            path=path,
            kind="yaml_parse",
            summary="failed to parse YAML",
            details=[ConfigErrorDetail(location=location, message=message)],
        )
    return ConfigError(
        path=path,
        kind="yaml_parse",
        summary=f"failed to parse YAML: {exc}",
    )


def _validation_error_to_config_error(path: Path, exc: ValidationError) -> ConfigError:
    """Flatten a pydantic :class:`ValidationError` into a :class:`ConfigError`."""
    details = [
        ConfigErrorDetail(
            location=_format_loc(err["loc"]),
            message=_format_validation_message(err),
        )
        for err in exc.errors()
    ]
    n = len(details)
    summary = f"{n} schema {'problem' if n == 1 else 'problems'} found"
    return ConfigError(path=path, kind="schema", summary=summary, details=details)


def _format_loc(loc: tuple[int | str, ...]) -> str:
    """Render a pydantic ``loc`` tuple as a dotted path."""
    if not loc:
        return "<root>"
    parts: list[str] = []
    for piece in loc:
        if isinstance(piece, int):
            parts.append(f"[{piece}]")
        elif parts:
            parts.append(f".{piece}")
        else:
            parts.append(str(piece))
    return "".join(parts)


def _format_validation_message(err: ErrorDetails) -> str:
    """Render a single pydantic error dict as a single-line message."""
    msg = err.get("msg", "invalid value")
    err_type = err.get("type", "")
    if err_type and err_type not in msg:
        return f"{msg} (type={err_type})"
    return str(msg)


__all__ = ["ConfigError", "ConfigErrorDetail", "ConfigErrorKind", "load_config"]
