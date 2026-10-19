"""Shared suite-path resolution and the managed ``suites:`` region.

The default golden-suite filename and the ``--suite`` / ``--suite-name``
resolution logic live here so every command that loads a suite (``run``,
``all``, ``bundle``, ``validate``, ``push``) shares one implementation and one
error type.

The same module owns the marker-delimited ``suites:`` region that ``init``
scaffolds and ``capture sync`` rewrites: its markers, its YAML rendering, its
parsing, and the derivation of a suite's evaluator block from what that suite's
captures actually contain. Writer and rewriter agree because they are one
implementation.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from evalshift_cli.captures.toolset import EMPTY_TOOLSET_FINGERPRINT
from evalshift_cli.config.loader import load_config
from evalshift_cli.config.models import (
    EvalShiftConfig,
    SuiteEvaluatorsOverride,
    SuiteSource,
    ToolArgumentsEvaluatorConfig,
    ToolSelectionEvaluatorConfig,
)
from evalshift_cli.suite.models import SuiteExample

# The conventional golden-suite filename, used as the default when neither
# ``--suite`` nor ``--suite-name`` is given.
SUITE_FILENAME: Final = "golden.jsonl"

# Marker comments that delimit the ``suites:`` region ``init`` writes and
# ``capture sync`` rewrites. Kept here so the writer and the rewriter agree.
SUITES_MARKER_BEGIN: Final = (
    "# >>> evalshift suites (managed by `evalshift capture sync`) >>>\n"
    "# Regenerated on every sync, evaluators included: each suite is scored with what\n"
    "# its own captures contain, so a tool-free suite gets no tool evaluators. Hand\n"
    "# edits inside this region are overwritten -- set `managed: false` on a suite to\n"
    "# freeze its entry (sync then prints what it would have written instead)."
)
SUITES_MARKER_END: Final = "# <<< evalshift suites <<<"

# Names of the evaluators ``capture sync`` generates. Stable by contract:
# reports key on evaluator names across runs, so regenerating a suite's block
# must not rename what it already wired.
GENERATED_TOOL_SELECTION_NAME: Final = "routing"
GENERATED_TOOL_ARGUMENTS_NAME: Final = "routing_args"


def render_suites_region(body: str) -> str:
    """Wrap ``body`` (the ``suites:`` YAML) in the begin/end marker comments."""
    return f"{SUITES_MARKER_BEGIN}\n{body.rstrip()}\n{SUITES_MARKER_END}"


def inject_suites_block(config_text: str, suites_yaml: str) -> str | None:
    """Replace the marker-delimited suites region in ``config_text``.

    Args:
        config_text: The full ``evalshift.yaml`` text.
        suites_yaml: The new ``suites:`` YAML body (without marker comments).

    Returns:
        The updated config text with the region between the markers replaced,
        or ``None`` if either marker is missing (caller should fall back to
        printing the block for the user to paste).
    """
    begin = config_text.find(SUITES_MARKER_BEGIN)
    end = config_text.find(SUITES_MARKER_END)
    if begin == -1 or end == -1 or end < begin:
        return None
    end_stop = end + len(SUITES_MARKER_END)
    return config_text[:begin] + render_suites_region(suites_yaml) + config_text[end_stop:]


def _offers_tools(example: SuiteExample) -> bool:
    """Whether ``example``'s agent was offered a non-empty toolset.

    Both spellings of a toolset are checked without touching disk: an inline
    ``tools`` list, and the ``toolset_ref`` that ``capture sync`` writes. The
    empty toolset has exactly one possible fingerprint, so "no tools were
    offered" is decidable from the ref alone -- no sidecar read needed.
    """
    if example.tools is not None:
        return bool(example.tools)
    return example.toolset_ref is not None and example.toolset_ref != EMPTY_TOOLSET_FINGERPRINT


def derive_suite_evaluators(examples: Iterable[SuiteExample]) -> SuiteEvaluatorsOverride | None:
    """Derive a suite's tool-evaluator block from what its rows contain.

    A project's suites are rarely homogeneous -- one calls tools, the rest
    answer in prose -- and a tool evaluator pointed at a tool-free suite scores
    an empty denominator, which reads as an inconclusive gate rather than as
    "not applicable here". So the block is generated from evidence:

    * No row was offered a toolset -> ``None``: emit no block at all, and the
      suite inherits the top-level ``evaluators:`` untouched.
    * Any row was offered a toolset -> ``tool_selection``, grading each side
      against the promoted ground truth (``conformance``) *and* the target
      against the source (``divergence: set``), which catches two models
      failing that ground truth in different ways.
    * Any row recorded tool-call arguments -> ``tool_arguments`` scored
      ``against: expected``. No ``strategies:`` block: the default ``auto``
      strategy already grades free text by meaning rather than by bytes.

    ``structural`` is deliberately not derived: nothing in a capture says what
    shape an answer must have.

    Args:
        examples: The suite's rows, as promoted onto disk.

    Returns:
        The per-suite override, or ``None`` when the suite calls no tools.
    """
    rows = list(examples)
    if not any(_offers_tools(row) for row in rows):
        return None
    families: dict[str, Any] = {
        "tool_selection": [
            ToolSelectionEvaluatorConfig(
                name=GENERATED_TOOL_SELECTION_NAME,
                conformance="expected",
                divergence="set",
            ),
        ],
    }
    if any(call.arguments for row in rows for call in row.expected_tools or []):
        families["tool_arguments"] = [
            ToolArgumentsEvaluatorConfig(
                name=GENERATED_TOOL_ARGUMENTS_NAME,
                against="expected",
            ),
        ]
    return SuiteEvaluatorsOverride(**families)


def suite_entry_payload(
    *,
    path: str,
    evaluators: SuiteEvaluatorsOverride | None,
) -> dict[str, Any]:
    """Build the YAML mapping for one generated ``suites:`` entry.

    Dumped with ``exclude_unset`` so only what the generator actually decided
    is written: a default left alone here stays unwritten, and a later change
    to that default still reaches configs generated today.

    Args:
        path: Suite path, relative to the config file's directory.
        evaluators: The suite's derived evaluator block, or ``None`` to leave
            the suite on the top-level ``evaluators:``.

    Returns:
        A plain mapping ready for :func:`render_suites_yaml`.
    """
    fields: dict[str, Any] = {"source": "captured", "path": path}
    if evaluators is not None:
        fields["evaluators"] = evaluators
    return SuiteSource(**fields).model_dump(exclude_unset=True, exclude_none=True)


class _BlockDumper(yaml.SafeDumper):
    """A dumper that indents sequences under their key, as people write YAML."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        """Force block sequences to indent, overriding PyYAML's flush-left default."""
        super().increase_indent(flow=flow, indentless=False)


def render_suites_yaml(entries: Mapping[str, Any]) -> str:
    """Render the ``suites:`` YAML body for the managed region.

    Entries are emitted in name order so regenerating an unchanged set of
    suites reproduces the file byte for byte.

    Args:
        entries: Suite name -> the entry's YAML mapping (from
            :func:`suite_entry_payload`, or carried forward verbatim from a
            previous region by :func:`parse_suites_region`).

    Returns:
        The YAML body, without the marker comments.
    """
    payload = {name: entries[name] for name in sorted(entries)}
    dumped = yaml.dump(
        {"suites": payload},
        Dumper=_BlockDumper,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
        allow_unicode=True,
        width=10_000,
    )
    return dumped.rstrip("\n")


def parse_suites_region(config_text: str) -> dict[str, Any]:
    """Read the entries currently written inside the managed suites region.

    ``capture sync`` regenerates only the suites it just promoted, so the rest
    of the region has to survive the rewrite -- syncing one suite must never
    delete another's entry. Only what is *inside* the markers is read: a
    ``suites:`` block a person wrote elsewhere in the file is not ours to
    carry forward (nor to duplicate into the region).

    Entries come back as raw mappings rather than validated models on purpose:
    an entry this sync is not regenerating is written back exactly as found,
    even if it would fail validation, so a malformed hand edit is never
    silently deleted.

    Args:
        config_text: The full ``evalshift.yaml`` text.

    Returns:
        Suite name -> the entry's YAML mapping. Empty when the markers are
        missing, the region is empty, or its body does not parse.
    """
    begin = config_text.find(SUITES_MARKER_BEGIN)
    end = config_text.find(SUITES_MARKER_END)
    if begin == -1 or end == -1 or end < begin:
        return {}
    body = config_text[begin + len(SUITES_MARKER_BEGIN) : end]
    try:
        loaded = yaml.safe_load(body)
    except yaml.YAMLError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    suites = loaded.get("suites")
    if not isinstance(suites, dict):
        return {}
    return {str(name): entry for name, entry in suites.items()}


def derive_suite_slug(*, suite_name: str | None, suite_path: Path) -> str:
    """Return a short label for a suite, for embedding in a run id.

    Prefers an explicit ``--suite-name``. Otherwise derives one from the
    path: the parent directory name for a conventional ``golden.jsonl``
    layout (``.evalshift/suites/<name>/golden.jsonl``), else the file stem.
    """
    if suite_name:
        return suite_name
    if suite_path.name == SUITE_FILENAME and suite_path.parent.name:
        return suite_path.parent.name
    return suite_path.stem


# Options that select a suite. A hint that appends ``--suite-name`` has to drop
# these first, or the suggested command carries the selection it is correcting.
_SUITE_SELECTION_FLAGS: Final = ("--suite-name", "--suite")


def format_invocation(argv: Sequence[str] | None = None) -> str:
    """Rebuild the current command line as a copy-pasteable ``evalshift ...`` string.

    ``argv[0]`` is replaced with the published entry-point name, so the hint
    reads the way the user types it rather than as an absolute path into a
    virtualenv. Any suite-selection option is dropped: callers append
    ``--suite-name`` to the result, and a command carrying both spellings would
    not do what the hint promises.

    Args:
        argv: Argument vector to render. Defaults to :data:`sys.argv`.

    Returns:
        A single shell-quoted command line, always starting with ``evalshift``.
    """
    args = list(sys.argv if argv is None else argv)
    kept: list[str] = []
    skip_next = False
    for arg in args[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in _SUITE_SELECTION_FLAGS:
            skip_next = True
            continue
        if any(arg.startswith(f"{flag}=") for flag in _SUITE_SELECTION_FLAGS):
            continue
        kept.append(arg)
    return shlex.join(["evalshift", *kept])


def _normalize_invocation(invocation: str) -> str:
    """Re-render a caller-supplied invocation through :func:`format_invocation`.

    The errors below append ``--suite-name`` to whatever they are handed, so a
    stale selection flag has to be stripped here rather than trusted away: the
    hint must hold even when a caller passes a raw command line.
    """
    return format_invocation(shlex.split(invocation))


def _command_word(invocation: str) -> str:
    """Return the subcommand in ``invocation`` (``"compare"`` for ``evalshift compare -y``)."""
    parts = shlex.split(invocation)
    return parts[1] if len(parts) > 1 else ""


# ``all`` chains every pipeline stage over one suite; the name reads as "every
# suite" often enough that the ambiguity error says so outright.
_ALL_MEANS_STAGES: Final = (
    "`all` runs all pipeline stages (doctor \u2192 run \u2192 evaluate \u2192 analyze "
    "\u2192 report) over one suite \u2014 it does not run all suites."
)


def _suite_commands(invocation: str, names: Sequence[str]) -> list[str]:
    """Render one ready-to-run command per suite, in name order."""
    return [f"{invocation} --suite-name {name}" for name in sorted(names)]


def _selection_panel(title: str, summary: str, note: str | None, commands: Sequence[str]) -> Panel:
    """Frame a suite-selection error the way the loaders frame theirs."""
    body: list[RenderableType] = [Text(summary, style="bold red")]
    if note:
        body.extend((Text(""), Text(note, style="dim")))
    if commands:
        body.append(Text(""))
        body.extend(Text(f"    {command}", style="cyan") for command in commands)
    return Panel(
        Group(*body),
        title=f"[red]{title}[/red]",
        title_align="left",
        border_style="red",
    )


class UnknownSuiteNameError(ValueError):
    """Raised when ``--suite-name`` names a suite absent from ``evalshift.yaml``."""

    def __init__(self, *, name: str, known: Sequence[str], invocation: str) -> None:
        self.name = name
        self.known: tuple[str, ...] = tuple(known)
        self.invocation = _normalize_invocation(invocation)
        super().__init__(self.format_plain())

    @property
    def _summary(self) -> str:
        listed = ", ".join(sorted(self.known)) or "(none)"
        return f"unknown --suite-name {self.name!r}. Known suites: {listed}."

    @property
    def _note(self) -> str | None:
        if self.known:
            return None
        return (
            "No suites are wired yet. Define one under suites: in evalshift.yaml "
            "(`evalshift capture sync` writes them), or pass --suite <path>."
        )

    def format_plain(self) -> str:
        """Render this error as a multi-line plain-text string."""
        lines = [self._summary]
        if self._note:
            lines.extend(("", self._note))
        commands = _suite_commands(self.invocation, self.known)
        if commands:
            lines.append("")
            lines.extend(f"    {command}" for command in commands)
        return "\n".join(lines)

    def format_rich(self) -> RenderableType:
        """Render this error inside a Rich :class:`Panel`."""
        return _selection_panel(
            "Unknown suite",
            self._summary,
            self._note,
            _suite_commands(self.invocation, self.known),
        )


class AmbiguousSuiteError(ValueError):
    """Raised when no suite is given but ``evalshift.yaml`` wires more than one."""

    def __init__(self, *, suite_names: Sequence[str], invocation: str) -> None:
        self.suite_names: tuple[str, ...] = tuple(suite_names)
        self.invocation = _normalize_invocation(invocation)
        super().__init__(self.format_plain())

    @property
    def _summary(self) -> str:
        listed = ", ".join(sorted(self.suite_names))
        return (
            f"evalshift.yaml wires {len(self.suite_names)} suites, so there is no "
            f"default to pick: {listed}."
        )

    @property
    def _note(self) -> str | None:
        return _ALL_MEANS_STAGES if _command_word(self.invocation) == "all" else None

    def format_plain(self) -> str:
        """Render this error as a multi-line plain-text string."""
        lines = [self._summary]
        if self._note:
            lines.extend(("", self._note))
        lines.append("")
        lines.extend(
            f"    {command}" for command in _suite_commands(self.invocation, self.suite_names)
        )
        return "\n".join(lines)

    def format_rich(self) -> RenderableType:
        """Render this error inside a Rich :class:`Panel`."""
        return _selection_panel(
            "Pick a suite",
            self._summary,
            self._note,
            _suite_commands(self.invocation, self.suite_names),
        )


def selected_suite_name(
    *,
    suite_path: Path | None,
    suite_name: str | None,
    cfg: EvalShiftConfig,
) -> str | None:
    """The ``suites:`` key :func:`resolve_suite_path` resolves to, or ``None``.

    An explicit ``--suite`` path names no suite. ``--suite-name`` is returned as given (not
    checked; :func:`resolve_suite_path` raises for an unknown one). With neither flag, the
    single wired suite is auto-selected -- the bare ``evalshift compare`` that follows
    ``capture sync``. Several wired suites, or none, select nothing.

    Args:
        suite_path: Explicit ``--suite`` path, or ``None``.
        suite_name: ``--suite-name`` key into ``cfg.suites``, or ``None``.
        cfg: The loaded configuration (provides the ``suites`` mapping).

    Returns:
        The selected suite's name, or ``None`` when the run is not on a named suite.
    """
    if suite_path is not None:
        return None
    if suite_name is not None:
        return suite_name
    # Neither --suite nor --suite-name: prefer the wired suites over the
    # bare-file default so the capture-first flow needs no extra flag.
    if len(cfg.suites) == 1:
        (only,) = cfg.suites
        return only
    return None


def resolve_suite_path(
    *,
    suite_path: Path | None,
    suite_name: str | None,
    cfg: EvalShiftConfig,
    config_path: Path,
) -> Path:
    """Resolve which suite file a command should load.

    Precedence: an explicit ``--suite`` path wins; then ``--suite-name`` (looked
    up in ``cfg.suites`` and resolved relative to the config file's directory).
    When neither is given, a single wired suite is auto-selected so bare
    ``evalshift compare`` works after ``capture sync``; with several wired suites the
    choice is ambiguous and must be named. Falling back to a ``golden.jsonl``
    file in the CWD only happens when no suites are wired at all.

    Args:
        suite_path: Explicit ``--suite`` path, or ``None``.
        suite_name: ``--suite-name`` key into ``cfg.suites``, or ``None``.
        cfg: The loaded configuration (provides the ``suites`` mapping).
        config_path: Path to ``evalshift.yaml``; named-suite paths resolve
            relative to its parent directory.

    Returns:
        The resolved suite file path.

    Raises:
        UnknownSuiteNameError: If ``suite_name`` is not present in ``cfg.suites``.
        AmbiguousSuiteError: If no suite is given but ``cfg.suites`` has >1 entry.
    """
    config_dir = config_path.resolve().parent
    if suite_path is not None:
        return suite_path
    name = selected_suite_name(suite_path=None, suite_name=suite_name, cfg=cfg)
    if name is not None:
        entry = cfg.suites.get(name)
        if entry is None:
            raise UnknownSuiteNameError(
                name=name,
                known=sorted(cfg.suites),
                invocation=format_invocation(),
            )
        return config_dir / entry.path
    if len(cfg.suites) > 1:
        raise AmbiguousSuiteError(
            suite_names=sorted(cfg.suites),
            invocation=format_invocation(),
        )
    return Path(SUITE_FILENAME)


def resolve_suite_override(
    *,
    suite_path: Path | None,
    suite_name: str | None,
    config_path: Path,
) -> Path | None:
    """Resolve an explicit ``--suite`` / ``--suite-name`` override.

    Unlike :func:`resolve_suite_path`, this returns ``None`` when neither
    option is given so the caller can fall back to a context-specific default
    (e.g. ``push`` / ``bundle`` defer to the suite recorded in the run's
    ``state.json``). The config is only loaded when ``--suite-name`` needs it.

    Args:
        suite_path: Explicit ``--suite`` path, or ``None``.
        suite_name: ``--suite-name`` key into ``cfg.suites``, or ``None``.
        config_path: Path to ``evalshift.yaml``; named-suite paths resolve
            relative to its parent directory.

    Returns:
        The resolved suite file path, or ``None`` when no override was given.

    Raises:
        ConfigError: If ``--suite-name`` is given but the config fails to load.
        UnknownSuiteNameError: If ``suite_name`` is not present in ``cfg.suites``.
    """
    if suite_name is None:
        return suite_path
    cfg = load_config(config_path)
    return resolve_suite_path(
        suite_path=suite_path,
        suite_name=suite_name,
        cfg=cfg,
        config_path=config_path,
    )


__all__ = [
    "GENERATED_TOOL_ARGUMENTS_NAME",
    "GENERATED_TOOL_SELECTION_NAME",
    "SUITES_MARKER_BEGIN",
    "SUITES_MARKER_END",
    "SUITE_FILENAME",
    "AmbiguousSuiteError",
    "UnknownSuiteNameError",
    "derive_suite_evaluators",
    "derive_suite_slug",
    "format_invocation",
    "inject_suites_block",
    "parse_suites_region",
    "render_suites_region",
    "render_suites_yaml",
    "resolve_suite_override",
    "resolve_suite_path",
    "selected_suite_name",
    "suite_entry_payload",
]
