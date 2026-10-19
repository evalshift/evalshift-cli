"""Implementation of the ``evalshift capture`` command group.

Lifecycle for captures recorded by the separate ``evalshift-sdk``:

* ``capture list``    — see what the SDK has recorded.
* ``capture promote`` — turn a capture into a golden suite case ``run`` scores.
* ``capture clean``   — prune capture files (never touches promoted suites).
* ``capture diff``    — compare two captures' tool-call traces.

Captures live under ``<base>/captures/<suite>/`` and promoted cases under
``<base>/suites/<suite>/``, where ``base`` follows the SDK convention
(``$EVALSHIFT_DIR`` else ``.evalshift``). ``--base`` overrides for tests/advanced use.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Final

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from evalshift_cli import __version__
from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.captures.promote import (
    PromoteOptions,
    build_conversation_examples,
    build_example_from_capture,
    capture_provenance,
    duplicate_turn_warnings,
    example_content_key,
    iter_promoted_case_files,
    iter_promoted_cases,
    promoted_case_path,
    promoted_suite_dir,
    rebuild_golden_jsonl,
    recover_from_messages,
    retire_promoted_case,
    rewrite_promoted_case,
    write_promoted_case,
)
from evalshift_cli.captures.reader import (
    CaptureError,
    CaptureRecord,
    capture_toolset_refs,
    envelope_toolset_ref,
    find_capture,
    iter_captures,
    load_toolset,
    promoted_capture_ids,
    promoted_toolset_refs,
    suites_root,
    toolsets_root,
)
from evalshift_cli.captures.remote import (
    FetchSummary,
    RemoteStore,
    RemoteStoreError,
    fetch_captures,
    open_store,
    parse_since,
)
from evalshift_cli.captures.repair import (
    RepairSelection,
    reconstructed_content_key,
    repaired_case,
    select_repairs,
    stringified_content,
    stringified_warnings,
)
from evalshift_cli.captures.selection import CandidateGroup, KeptCase, RetireReason, plan_suite
from evalshift_cli.captures.staleness import (
    StalenessRules,
    StaleReason,
    case_instant,
    examples_route,
    is_reviewed,
    parse_instant,
)
from evalshift_cli.captures.store_uri import parse_store_uri
from evalshift_cli.cli.commands._suites import (
    derive_suite_evaluators,
    inject_suites_block,
    parse_suites_region,
    render_suites_yaml,
    suite_entry_payload,
)
from evalshift_cli.cli.commands.doctor import CONFIG_FILENAME
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.config.models import SuiteEvaluatorsOverride
from evalshift_cli.models.matching import describe_models, model_filter_applies
from evalshift_cli.suite.loader import SuiteError, load_jsonl
from evalshift_cli.traces.diff import diff_traces
from evalshift_cli.traces.models import ModelCallEvent, ToolCallEvent
from evalshift_cli.utils.ci_pin import check_ci_pin
from evalshift_cli.utils.duration import format_duration

capture_app = typer.Typer(
    name="capture",
    help="Manage agent captures recorded by the evalshift-sdk.",
    no_args_is_help=True,
    add_completion=False,
)

_BaseOption = Annotated[
    Path | None,
    typer.Option(
        "--base", help="Capture base dir (default: $EVALSHIFT_DIR or .evalshift).", hidden=True
    ),
]
_ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help=f"Path to evalshift.yaml (default: ./{CONFIG_FILENAME}).",
        file_okay=True,
        dir_okay=False,
    ),
]
_OfflineOption = Annotated[
    bool,
    typer.Option(
        "--offline",
        help="Skip fetching from captures.store; use only the captures already on disk.",
    ),
]
_SinceOption = Annotated[
    str | None,
    typer.Option(
        "--since",
        help=(
            "Only fetch captures newer than this from captures.store: a duration "
            "(30m, 24h, 7d) or an ISO date/datetime."
        ),
    ),
]


def _warn_ci_pin(console: Console, config_path: Path) -> None:
    """Warn when a workflow next to ``config_path`` installs an older CLI than this one.

    Advisory only — the config was (or is about to be) written by this CLI,
    and a CI job pinned to an older release would reject any newer keys.
    """
    finding = check_ci_pin(config_path.resolve().parent, __version__)
    if finding is not None:
        console.print(f"[yellow]⚠[/yellow] {escape(finding.message)}")


def _now() -> datetime:
    """The one instant a sync measures capture age from (tests pin it)."""
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


def _warn_unreadable(console: Console) -> Callable[[CaptureError], None]:
    """Build an ``iter_captures`` on_error hook that warns instead of dropping silently."""

    def warn(exc: CaptureError) -> None:
        console.print(
            f"[yellow]⚠[/yellow] skipping unreadable capture [cyan]{exc.path.name}[/cyan]: "
            f"{exc.summary}",
        )

    return warn


def _n_tools(envelope_events: list[Any]) -> int:
    return sum(1 for e in envelope_events if isinstance(e, ToolCallEvent))


def _promoted_capture_ids_or_warn(
    *,
    base: Path | None,
    suite: str | None = None,
    console: Console,
) -> set[str]:
    """``promoted_capture_ids``, degraded to a warning instead of a hard failure.

    ``promoted_capture_ids`` now fails closed (C2) when a promoted-case file
    can't be parsed -- correct for ``capture clean``'s orphan-sidecar sweep,
    which precedes a delete and must never guess. The three call sites here
    are read-only or advisory (a "promoted" column, an unpromoted-sibling
    heads-up, and the promoted-only *filter* that only narrows what ``clean``
    considers -- the sweep itself calls ``promoted_toolset_refs`` directly,
    uncaught, so it still fails closed regardless of what happens here): a
    corrupt case file for some unrelated capture must not crash them
    outright. Degrading to "treat it as not promoted" is the fail-closed
    direction for all three -- it never makes a capture look MORE promoted
    (and therefore more/less eligible for deletion) than the truth.
    """
    try:
        return promoted_capture_ids(base=base, suite=suite)
    except CaptureError as exc:
        console.print(
            f"[yellow]⚠[/yellow] could not fully determine promoted status: "
            f"[cyan]{exc.path.name}[/cyan] failed to parse ({exc.summary}); "
            "treating it as not promoted.",
        )
        return set()


def _promoted_content_owners(
    suite: str,
    *,
    base: Path | None,
    console: Console,
) -> dict[tuple[str, str], str]:
    """Map ``(suite, content key) -> owning case name`` for cases already on disk.

    Seeds ``sync``'s dedup so it spans runs: without this, a capture recorded
    later but sorting earlier by id wins the in-run pass and is promoted, while
    the equivalent capture it duplicates is skipped — leaving the case file an
    earlier run already wrote. Both then end up in ``golden.jsonl``.

    Keyed on the case *name* (not ``from_capture``) because that is what ``sync``
    would overwrite: a case a single ``capture promote --as <name>`` wrote under
    a custom name owns its content, so a later ``sync`` of the same capture is a
    duplicate rather than a refresh.

    Never raises: an unreadable case file is warned about and skipped, so a
    corrupt file degrades dedup rather than failing the sync (the subsequent
    ``rebuild_golden_jsonl`` is what surfaces it as an error).
    """
    suite_dir = promoted_suite_dir(suite, base=base)
    if not suite_dir.is_dir():
        return {}
    try:
        cases = iter_promoted_cases(suite_dir)
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        console.print(
            f"[yellow]⚠[/yellow] could not read promoted cases in [cyan]{suite_dir}[/cyan] "
            f"for dedup: {type(exc).__name__}",
        )
        return {}
    owners: dict[tuple[str, str], str] = {}
    for case in cases:
        owners.setdefault((suite, example_content_key(case.example)), case.name)
    return owners


def _load_promoted(
    suite: str, *, base: Path | None
) -> tuple[list[tuple[Path, PromotedCase]], str | None]:
    """The suite's promoted cases with their files, plus the error's type name on a read failure.

    A suite with no directory yet has no cases. Only the exception's type name is kept: its
    text can carry raw bytes of a case file.
    """
    suite_dir = promoted_suite_dir(suite, base=base)
    if not suite_dir.is_dir():
        return [], None
    try:
        return iter_promoted_case_files(suite_dir), None
    except (ValueError, OSError) as exc:  # pydantic's ValidationError is a ValueError
        return [], type(exc).__name__


def _with_provenance(case: PromotedCase, envelope: CaptureEnvelope) -> PromotedCase:
    """``case`` with an empty ``captured_at`` / ``source_model`` filled from ``envelope``.

    ``envelope`` is the case's own capture. A field already set is kept, and nothing else on
    the case changes; ``case`` itself comes back when there is nothing to fill.
    """
    if case.captured_at and case.source_model:
        return case
    captured_at, source_model = capture_provenance(envelope)
    return case.model_copy(
        update={
            "captured_at": case.captured_at or captured_at,
            "source_model": case.source_model or source_model,
        }
    )


def _backfill_provenance(
    cases: list[tuple[Path, PromotedCase]],
    envelopes: Mapping[str, CaptureEnvelope],
) -> tuple[list[tuple[Path, PromotedCase]], int]:
    """Fill ``captured_at`` / ``source_model`` on cases written before the fields existed.

    Only an empty field is filled, only from the case's own capture, and only while that
    capture is local; nothing else on the case changes (``promoted_at`` included), and each
    case is rewritten in the file it was read from. Returns the cases as they now stand and
    how many files were rewritten.
    """
    out: list[tuple[Path, PromotedCase]] = []
    rewritten = 0
    for path, case in cases:
        envelope = envelopes.get(case.from_capture)
        filled = case if envelope is None else _with_provenance(case, envelope)
        if filled != case:
            rewrite_promoted_case(path, filled)
            rewritten += 1
        out.append((path, filled))
    return out, rewritten


def _suites_with_cases(*, base: Path | None, only: str | None) -> set[str]:
    """Directory names under ``<base>/suites/`` holding at least one case file.

    A name is the suite's sanitised directory name (``team/support`` -> ``team_support``), not
    necessarily the suite name its cases record. ``only`` (a suite name) narrows to its directory.
    """
    root = suites_root(base)
    if not root.is_dir():
        return set()
    only_dir = promoted_suite_dir(only, base=base).name if only is not None else None
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and any(path.glob("*.json"))
        and (only_dir is None or path.name == only_dir)
    }


def _suite_rules(
    suite: str,
    limits: _SyncLimits,
    *,
    now: datetime,
    recorded_models: list[str],
    console: Console,
) -> StalenessRules:
    """One suite's staleness rules, with the model guardrail applied (spec §3)."""
    source = limits.source_model
    if source and not model_filter_applies(recorded_models, source):
        known = [m for m in recorded_models if m]
        if known:
            console.print(
                f"[yellow]⚠[/yellow] {escape(suite)}: none of {len(known)} case(s) and "
                f"capture(s) were recorded on {escape(source)} "
                f"(seen: {escape(describe_models(known))}); not filtering by model — "
                "check defaults.source_model",
                soft_wrap=True,
            )
        source = None
    return StalenessRules(now=now, max_age=limits.max_age, source_model=source)


def _candidate_groups(pending: Sequence[PromotedCase]) -> list[CandidateGroup]:
    """Group pending cases by conversation (a lone case is its own group), turn order kept."""
    turns: dict[str, list[PromotedCase]] = {}
    for case in pending:
        key = (
            f"conversation:{case.conversation_id}"
            if case.conversation_id
            else f"capture:{case.name}"
        )
        turns.setdefault(key, []).append(case)
    return [
        CandidateGroup(
            key=key,
            capture_ids=tuple(c.name for c in cases),
            route=examples_route(c.example for c in cases),
            content_keys=tuple(example_content_key(c.example) for c in cases),
        )
        for key, cases in turns.items()
    ]


def _reason_counts(counts: Counter[str], *, max_age: timedelta | None) -> str:
    """``2 older than 14d, 1 on another model, 3 over max_cases``; zero counts left out."""
    labels = {
        "age": f"older than {format_duration(max_age)}" if max_age is not None else "too old",
        "model": "on another model",
        "history": "missing earlier turns",
        "over_cap": "over max_cases",
    }
    return ", ".join(
        f"{counts[k]} {labels[k]}" for k in ("age", "model", "history", "over_cap") if counts[k]
    )


def _plan_line(
    suite: str,
    *,
    kept: int,
    retired: Sequence[RetireReason] | None,
    added: int,
    eligible: int,
    ineligible: Counter[str],
    max_age: timedelta | None,
) -> str | None:
    """The per-suite summary line (spec §6), from the files sync actually deleted and wrote.

    ``retired`` is ``None`` with ``max_age`` alone (nothing is planned then).
    """
    not_eligible = _reason_counts(ineligible, max_age=max_age)
    if retired is None:
        return f"  {suite}: not eligible: {not_eligible}" if not_eligible else None
    detail = _reason_counts(Counter(retired), max_age=max_age)
    stayed = kept - len(retired)
    return (
        f"  {suite}: {stayed + added} cases — kept {stayed}, retired {len(retired)}"
        + (f" ({detail})" if detail else "")
        + f", added {added} of {eligible} eligible"
        + (f"; not eligible: {not_eligible}" if not_eligible else "")
    )


def _repair_selection(*, base: Path | None, suite: str | None, console: Console) -> RepairSelection:
    """The promoted cases ``capture sync`` treats as repair candidates (``captures/repair.py``).

    Never raises: a suite holding an unreadable case file is warned about and skipped, the
    same degradation :func:`_promoted_content_owners` applies.
    """
    root = suites_root(base)
    if suite is not None:
        suite_dirs = [promoted_suite_dir(suite, base=base)]
    elif root.is_dir():
        suite_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        suite_dirs = []
    cases: list[PromotedCase] = []
    for suite_dir in suite_dirs:
        if not suite_dir.is_dir():
            continue
        try:
            cases.extend(iter_promoted_cases(suite_dir))
        except ValueError as exc:  # pydantic ValidationError subclasses ValueError
            console.print(
                f"[yellow]⚠[/yellow] could not read promoted cases in "
                f"[cyan]{escape(str(suite_dir))}[/cyan] to check for repairs: "
                f"{type(exc).__name__}",
            )
    return select_repairs(cases)


def _warn_repaired_duplicates(
    suite: str, repaired: list[str], *, base: Path | None, console: Console
) -> None:
    """Name each pair a case ``--repair`` just rewrote forms with a case of the same content.

    A plain sync from a fresh clone cannot see a candidate's rebuilt content, so it may already
    have promoted a capture that duplicates it. Which copy to keep is the user's call: this
    only warns, and never deletes or modifies anything.
    """
    try:
        cases = iter_promoted_cases(promoted_suite_dir(suite, base=base))
    except ValueError:  # the golden.jsonl rebuild that follows fails on it anyway
        return
    keyed = [(case.name, example_content_key(case.example)) for case in cases]
    warned: set[frozenset[str]] = set()
    for name, key in keyed:
        if name not in repaired:
            continue
        for other, other_key in keyed:
            pair = frozenset((name, other))
            if other == name or other_key != key or pair in warned:
                continue
            warned.add(pair)
            console.print(
                f"[yellow]⚠[/yellow] {escape(suite)}: repaired case {escape(name)} duplicates "
                f"{escape(other)} — delete one (duplicate examples inflate n)",
            )


def _records_own_history(envelope: CaptureEnvelope) -> bool:
    """Whether the capture's first ``model_call`` yields a history of its own when promoted.

    A turn whose capture doesn't -- no messages list at all, or one with no user turn to split
    on -- gets its history reconstructed from its sibling turns instead.
    """
    first = next((e for e in envelope.trace.events if isinstance(e, ModelCallEvent)), None)
    if first is None:
        return False
    recovered = recover_from_messages(first.input, input_var="input", warnings=[])
    return recovered is not None and recovered.history is not None


def _misses_earlier_turns(envelope: CaptureEnvelope, turns: Mapping[str, set[int]]) -> bool:
    """Whether the turn's history would be rebuilt from earlier turns that are not all local.

    ``turns`` maps each conversation to the turn indices (0-based) on disk. A turn that recorded
    its own messages list needs none of them; a turn without an index can't be checked.
    """
    if envelope.conversation_id is None or envelope.turn_index is None:
        return False
    if _records_own_history(envelope):
        return False
    present = turns.get(envelope.conversation_id, set())
    return any(i not in present for i in range(envelope.turn_index))


def _warn_unrepairable(
    selection: RepairSelection, records: list[CaptureRecord], *, console: Console
) -> None:
    """Name the repair candidates whose capture is neither local nor fetched from the store."""
    available = {r.envelope.capture_id for r in records}
    missing = [c.case.name for c in selection.candidates if c.case.from_capture not in available]
    if not missing:
        return
    shown = ", ".join(missing[:3]) + (", …" if len(missing) > 3 else "")
    console.print(
        f"[yellow]⚠[/yellow] can't repair {len(missing)} case(s) — their capture is no longer "
        f"local or in the store ({escape(shown)}): delete the case file or re-capture.",
    )


#: A top-level ``captures:`` key: at column 0, so neither a comment nor a nested key matches.
_TOP_LEVEL_CAPTURES_KEY = re.compile(rb"^captures\s*:", re.MULTILINE)


def _mentions_captures(config_path: Path) -> bool:
    """Whether the raw config file has a top-level ``captures`` key (``False`` if unreadable).

    Structural rather than a substring test: ``evalshift init`` writes "captures" into its
    comments, and a config that only mentions the word never asked for a store.
    """
    try:
        return _TOP_LEVEL_CAPTURES_KEY.search(config_path.read_bytes()) is not None
    except OSError:
        return False


def _configured_store(config_path: Path, *, console: Console, strict: bool) -> RemoteStore | None:
    """The store ``captures.store`` names, or ``None`` when there is none to fetch from.

    A missing config file means no store. An invalid or unreadable one is a hard error (exit 1,
    one line, no traceback) when ``strict`` (``capture fetch`` has nothing else to do). Otherwise ``sync`` and ``list`` keep working on
    the local mirror the way they did before ``captures.store`` existed: an invalid or
    unreadable file is skipped, with a warning only when it has a top-level ``captures`` key (a
    config that never asked for a store must not change their output). A missing client library
    is always a hard error here: the user asked for a store this command cannot read.
    """
    if not config_path.exists():
        return None
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        if strict:
            console.print(exc.format_rich())
            raise typer.Exit(code=1) from exc
        _warn_store_skipped(config_path, exc.summary, console=console)
        return None
    except (OSError, UnicodeDecodeError) as exc:
        if strict:
            # The exception's type only: its text can carry raw bytes of the file.
            console.print(
                f"[red]✗[/red] could not read {escape(str(config_path))} "
                f"({type(exc).__name__}); capture fetch needs a readable evalshift.yaml.",
                soft_wrap=True,
            )
            raise typer.Exit(code=1) from exc
        _warn_store_skipped(config_path, type(exc).__name__, console=console)
        return None
    if cfg.captures.store is None:
        return None
    try:
        return open_store(parse_store_uri(cfg.captures.store))
    except RemoteStoreError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc


def _warn_store_skipped(config_path: Path, reason: str, *, console: Console) -> None:
    """Warn that captures.store is being skipped, if the file looks like it configures one."""
    if not _mentions_captures(config_path):
        return
    console.print(
        f"[yellow]⚠[/yellow] could not read {escape(str(config_path))} ({escape(reason)}); "
        "skipping captures.store and using the captures already on disk.",
    )


FETCH_BUDGET_PER_CASE: Final = 3
"""With ``captures.max_cases`` set, sync's fetch takes each suite's newest ``3 x max_cases``
captures: spare enough for the planner to balance tool routes, and never the whole bucket."""


@dataclass(frozen=True, slots=True)
class _SyncLimits:
    """``captures.max_age`` / ``captures.max_cases`` and ``defaults.source_model``, for sync.

    ``active`` is the opt-in switch (spec §1): with neither limit set, sync behaves exactly as
    it did before the limits existed, whatever ``defaults.source_model`` says.
    """

    max_age: timedelta | None = None
    max_cases: int | None = None
    source_model: str | None = None

    @property
    def active(self) -> bool:
        return self.max_age is not None or self.max_cases is not None

    @property
    def fetch_budget(self) -> int | None:
        """How many of each suite's newest captures the fetch takes; ``None``: all of them."""
        return FETCH_BUDGET_PER_CASE * self.max_cases if self.max_cases is not None else None

    def fetch_window(self, now: datetime) -> tuple[datetime, str] | None:
        """The cutoff ``max_age`` puts on the fetch, and how its summary names it."""
        if self.max_age is None:
            return None
        return now - self.max_age, f"captures.max_age ({format_duration(self.max_age)})"


def _sync_limits(config_path: Path, *, console: Console, warn: bool) -> _SyncLimits:
    """Read the limits from ``config_path``; none at all when it is missing or unreadable.

    With ``warn``, an unreadable file that has a top-level ``captures`` key gets a line saying
    the limits are dropped. ``capture sync`` passes it online too: the fetch step's own warning
    names only ``captures.store``, so without this line sync would quietly run unbounded.
    """
    if not config_path.exists():
        return _SyncLimits()
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        reason = exc.summary
    except (OSError, UnicodeDecodeError) as exc:
        reason = type(exc).__name__  # its text can carry raw bytes of the file
    else:
        return _SyncLimits(
            max_age=cfg.captures.max_age,
            max_cases=cfg.captures.max_cases,
            source_model=cfg.defaults.source_model,
        )
    if warn and _mentions_captures(config_path):
        console.print(
            f"[yellow]⚠[/yellow] could not read {escape(str(config_path))} ({escape(reason)}); "
            "syncing without captures.max_age / captures.max_cases.",
        )
    return _SyncLimits()


def _resolve_since(since: str | None) -> datetime | None:
    if since is None:
        return None
    try:
        return parse_since(since)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--since") from exc


def _fetch_into_base(
    store: RemoteStore,
    *,
    base: Path | None,
    suite: str | None,
    since: datetime | None,
    console: Console,
    force_ids: frozenset[str] = frozenset(),
    since_label: str = "--since",
    per_suite_limit: int | None = None,
) -> FetchSummary:
    """Mirror ``store`` into ``base`` and print the one-line summary; exit 1 on failure.

    ``force_ids`` are fetched even when promoted or older than ``since`` -- the captures
    ``capture sync --repair`` needs to rebuild promoted cases. ``per_suite_limit`` caps each
    suite at its newest that-many captures (``fetch_captures``).
    """
    skip_ids = _promoted_capture_ids_or_warn(base=base, console=console)
    try:
        summary = fetch_captures(
            store,
            base=base,
            suite=suite,
            since=since,
            skip_ids=skip_ids,
            force_ids=force_ids,
            since_label=since_label,
            per_suite_limit=per_suite_limit,
        )
    except RemoteStoreError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc
    # soft_wrap: one line per fetch, so a long store URI or skip list never splits a count
    # from its label in a CI log someone greps.
    console.print(f"[green]✓[/green] {escape(summary.describe())}", soft_wrap=True)
    return summary


def _maybe_fetch(
    *,
    config_path: Path,
    offline: bool,
    since: str | None,
    suite: str | None,
    base: Path | None,
    console: Console,
    force_ids: frozenset[str] = frozenset(),
    window: tuple[datetime, str] | None = None,
    per_suite_limit: int | None = None,
) -> None:
    """The automatic fetch step ``sync`` and ``list`` run before reading the local mirror.

    ``window`` is the cutoff ``captures.max_age`` implies, and how to name it; an explicit
    ``--since`` wins. ``per_suite_limit`` is the budget ``captures.max_cases`` implies.
    """
    since_dt = _resolve_since(since)
    since_label = "--since"
    if since_dt is None and window is not None:
        since_dt, since_label = window
    store = None if offline else _configured_store(config_path, console=console, strict=False)
    if store is None:
        if since is not None:
            console.print(
                "[red]✗[/red] --since needs captures.store in evalshift.yaml and no --offline: "
                "there is nothing to fetch from.",
            )
            raise typer.Exit(code=1)
        return
    _fetch_into_base(
        store,
        base=base,
        suite=suite,
        since=since_dt,
        since_label=since_label,
        console=console,
        force_ids=force_ids,
        per_suite_limit=per_suite_limit,
    )


@capture_app.command(name="list")
def capture_list(
    suite: Annotated[
        str | None,
        typer.Argument(help="Only list captures for this suite."),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of a table."),
    ] = False,
    config_path: _ConfigOption = Path(CONFIG_FILENAME),
    offline: _OfflineOption = False,
    base: _BaseOption = None,
) -> None:
    """List captures recorded by the SDK.

    Fetches from captures.store first, as capture sync does: only captures within
    captures.max_age and, with captures.max_cases set, each suite's newest 3x
    max_cases.
    """
    # With --json, stdout carries only the JSON: the fetch line and every warning go to stderr.
    console = Console(stderr=as_json)
    limits = _sync_limits(config_path, console=console, warn=False)
    _maybe_fetch(
        config_path=config_path,
        offline=offline,
        since=None,
        suite=suite,
        base=base,
        console=console,
        window=limits.fetch_window(_now()),
        per_suite_limit=limits.fetch_budget,
    )
    records = iter_captures(suite=suite, base=base, on_error=_warn_unreadable(console))
    promoted = _promoted_capture_ids_or_warn(base=base, console=console)

    if as_json:
        rows = [
            {
                "capture_id": r.envelope.capture_id,
                "suite": r.envelope.suite,
                "created_at": r.envelope.created_at,
                "n_tools": _n_tools(r.envelope.trace.events),
                "n_events": len(r.envelope.trace.events),
                "code_version": r.envelope.code_version,
                "input_hash": r.envelope.input_hash,
                "promoted": r.envelope.capture_id in promoted,
            }
            for r in records
        ]
        typer.echo(json.dumps(rows, indent=2))
        return

    if not records:
        where = f" for suite {suite!r}" if suite else ""
        console.print(f"[dim]no captures found{where}.[/dim]")
        return

    # The human table shows the high-value columns; `--json` carries the rest
    # (code_version, input_hash) for tooling.
    table = Table(title="captures", title_justify="left")
    table.add_column("capture_id", style="cyan", no_wrap=True)
    table.add_column("suite", no_wrap=True)
    table.add_column("created_at", style="dim")
    table.add_column("tools", justify="right")
    table.add_column("events", justify="right")
    table.add_column("promoted", justify="center")
    for r in records:
        table.add_row(
            r.envelope.capture_id,
            r.envelope.suite,
            r.envelope.created_at,
            str(_n_tools(r.envelope.trace.events)),
            str(len(r.envelope.trace.events)),
            "✓" if r.envelope.capture_id in promoted else "",
        )
    console.print(table)


@capture_app.command(name="fetch")
def capture_fetch(
    suite: Annotated[
        str | None,
        typer.Option("--suite", help="Only fetch captures for this suite."),
    ] = None,
    since: _SinceOption = None,
    config_path: _ConfigOption = Path(CONFIG_FILENAME),
    base: _BaseOption = None,
) -> None:
    """Mirror new captures from captures.store (evalshift.yaml) into the local directory.

    ``capture sync`` and ``capture list`` run this step automatically; ``fetch`` exists for
    inspecting with ``list``, ``diff`` or ``promote`` by hand before promoting anything.
    """
    console = Console()
    if not config_path.exists():
        console.print(
            f"[red]✗[/red] {escape(str(config_path))} not found; capture fetch needs "
            "captures.store in evalshift.yaml.",
        )
        raise typer.Exit(code=1)
    store = _configured_store(config_path, console=console, strict=True)
    if store is None:
        console.print(
            "[red]✗[/red] no captures.store configured in evalshift.yaml "
            "(for example: captures: {store: s3://bucket/prefix}).",
        )
        raise typer.Exit(code=1)
    _fetch_into_base(store, base=base, suite=suite, since=_resolve_since(since), console=console)


def _check_rounds(rounds: str) -> None:
    """Reject a ``--rounds`` value the promoter has no meaning for."""
    if rounds not in {"first", "all"}:
        raise typer.BadParameter("--rounds must be 'first' or 'all'", param_hint="--rounds")


def _declared_tool_properties(
    envelopes: Iterable[CaptureEnvelope],
    *,
    base: Path | None,
) -> dict[str, frozenset[str]] | None:
    """Declared argument property names per tool, from the recorded toolset.

    Sourced from ``envelopes``' own recorded ``toolset_ref`` sidecar(s) --
    the toolset actually offered during those captures -- rather than any
    config file, so it is accurate even when no ``evalshift.yaml`` exists,
    and never conflates two captures that were offered different toolsets
    sharing a tool name.

    Promotion uses the result only to recognise a capture that recorded a
    wrapper function's parameters instead of the arguments the model passed
    (see :func:`evalshift_cli.captures.promote._unwrap_recorded_arguments`).
    Returns ``None`` when no envelope carries a resolvable ``toolset_ref`` --
    a missing schema must leave the recording alone, not block the
    promotion.

    Args:
        envelopes: The capture(s) being promoted this call. ``capture
            promote`` passes the one capture it names; ``capture sync``
            passes every capture in the batch, so its single shared
            :class:`~evalshift_cli.captures.promote.PromoteOptions` still covers
            every distinct toolset the batch touches.
        base: Capture base dir the sidecar(s) resolve against.
    """
    refs = {ref for e in envelopes if (ref := envelope_toolset_ref(e)) is not None}

    out: dict[str, frozenset[str]] = {}
    # sorted(), not the set's own (hash-seed-dependent) iteration order: two
    # toolsets in the same batch can declare the same tool name with
    # different property sets, and out[spec.name] below is last-write-wins --
    # unsorted iteration made that winner nondeterministic (M1), so two
    # `capture sync` runs over identical captures could unwrap (or not
    # unwrap) the very expected_tools arguments this feeds differently.
    for ref in sorted(refs):
        try:
            specs = load_toolset(ref, base=base)
        except CaptureError:
            continue
        for spec in specs:
            properties = (spec.input_schema.get("properties") or {}).keys()
            out[spec.name] = frozenset(properties)
    return out or None


@capture_app.command(name="promote")
def capture_promote(
    capture_id: Annotated[str, typer.Argument(help="Capture id to promote.")],
    name: Annotated[
        str | None,
        typer.Option("--as", help="Golden case id (default: the capture id)."),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option("--suite", help="Restrict the capture search to this suite."),
    ] = None,
    input_var: Annotated[
        str,
        typer.Option(
            "--input-var",
            help="Template-variable name for a bare-string model input.",
        ),
    ] = "input",
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Extra tag to attach (repeatable)."),
    ] = None,
    strict_args: Annotated[
        bool,
        typer.Option("--strict-args", help="Require exact tool-argument matches."),
    ] = False,
    names_only: Annotated[
        bool,
        typer.Option("--names-only", help="Match tool names only; ignore arguments."),
    ] = False,
    tool_count: Annotated[
        bool,
        typer.Option("--tool-count", help="Also pin the expected tool-call count."),
    ] = False,
    rounds: Annotated[
        str,
        typer.Option(
            "--rounds",
            help=(
                "How many recorded agent rounds to replay: 'first' (default, single-shot, "
                "scored against round 1) or 'all' (teacher-forced — every round is replayed "
                "with the recorded tool results fed back, scored round by round)."
            ),
        ),
    ] = "first",
    allow_errored: Annotated[
        bool,
        typer.Option(
            "--allow-errored",
            help=(
                "Promote even if the turn recorded an error event. The case "
                "still never asserts expected_no_tools."
            ),
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite an existing promoted case."),
    ] = False,
    base: _BaseOption = None,
) -> None:
    """Promote a capture into a golden suite case that 'run' can evaluate.

    Uses messages-aware recovery (see :func:`build_example_from_capture`), but
    — unlike ``capture sync`` — does **not** reconstruct conversation history
    across sibling captures: only the capture's own recorded messages list (if
    any) becomes ``history``. Promote the whole conversation together with
    ``evalshift capture sync`` to get cross-capture reconstruction.
    """
    console = Console()
    _check_rounds(rounds)
    try:
        record = find_capture(capture_id, suite=suite, base=base)
    except CaptureError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    envelope = record.envelope
    opts = PromoteOptions(
        name=name,
        input_var=input_var,
        tags=tuple(tags or ()),
        strict_args=strict_args,
        names_only=names_only,
        tool_count=tool_count,
        allow_errored=allow_errored,
        rounds="all" if rounds == "all" else "first",
        tool_properties=_declared_tool_properties([envelope], base=base),
    )
    built = build_example_from_capture(envelope, opts, base=base)
    for warning in built.warnings:
        console.print(f"[yellow]⚠[/yellow] {warning}")

    # `promote` names one capture explicitly, so a blocked case is a user
    # error worth failing on rather than a line in a sync summary.
    if built.blocked is not None:
        console.print(f"[red]✗[/red] refusing to promote {envelope.capture_id!r}: {built.blocked}")
        # Only an errored turn is rescuable by the flag -- a capture with no
        # usable recorded toolset needs re-capturing (already spelled out in
        # `built.blocked`), so pointing at --allow-errored there is a dead end.
        if built.blocked_reason == "errored":
            console.print("Re-run with [bold]--allow-errored[/bold] to promote it anyway.")
        raise typer.Exit(code=1)

    if envelope.conversation_id is not None:
        promoted = _promoted_capture_ids_or_warn(base=base, suite=envelope.suite, console=console)
        siblings = [
            r
            for r in iter_captures(suite=envelope.suite, base=base)
            if r.envelope.conversation_id == envelope.conversation_id
            and r.envelope.capture_id != envelope.capture_id
            and r.envelope.capture_id not in promoted
        ]
        if siblings:
            console.print(
                f"[yellow]⚠[/yellow] capture belongs to conversation "
                f"{envelope.conversation_id!r} with {len(siblings)} unpromoted sibling "
                "turn(s) — run 'evalshift capture sync' to promote the conversation "
                "together.",
            )

    captured_at, source_model = capture_provenance(envelope)
    case = PromotedCase(
        name=opts.name or envelope.capture_id,
        suite=envelope.suite,
        from_capture=envelope.capture_id,
        promoted_at=_now_iso(),
        source_input_hash=envelope.input_hash,
        code_version=envelope.code_version,
        promotion_source=built.promotion_source,
        cost_usd=built.cost_usd,
        cost_source=built.cost_source,
        conversation_id=built.example.conversation_id,
        turn_index=built.example.turn_index,
        captured_at=captured_at,
        source_model=source_model,
        example=built.example,
    )

    try:
        case_path = write_promoted_case(case, base=base, force=force)
    except FileExistsError as exc:
        console.print(
            f"[red]✗[/red] case {case.name!r} already exists in suite {case.suite!r}. "
            "Re-run with [bold]--force[/bold] to overwrite.",
        )
        raise typer.Exit(code=1) from exc

    golden_path = rebuild_golden_jsonl(case_path.parent)
    try:
        load_jsonl(golden_path)  # round-trip guard: the index must stay loadable.
    except SuiteError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    console.print(f"[green]✓[/green] promoted [cyan]{envelope.capture_id}[/cyan] → {case_path}")
    console.print(f"golden suite: {golden_path}")
    console.print()
    console.print("[bold]Wire it into evalshift.yaml:[/bold]")
    console.print(
        f"  suites:\n    {envelope.suite}:\n      source: captured\n      path: {golden_path}",
    )
    console.print(
        f"\n[bold]Then:[/bold] [cyan]evalshift run --suite-name {envelope.suite}[/cyan]",
    )


@capture_app.command(name="clean")
def capture_clean(
    suite: Annotated[
        str | None,
        typer.Argument(help="Only clean captures for this suite."),
    ] = None,
    promoted_only: Annotated[
        bool,
        typer.Option("--promoted", help="Remove only promoted captures (the default)."),
    ] = False,
    all_captures: Annotated[
        bool,
        typer.Option("--all", help="Remove every capture, promoted or not."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
    base: _BaseOption = None,
) -> None:
    """Delete capture files, then sweep orphaned toolset sidecars.

    Never touches promoted suites. Capture deletion is scoped by ``suite``/
    ``--promoted``/``--all`` as usual; the sweep that follows is not -- a
    toolset sidecar is shared across every suite that used it, so refcounting
    it is global regardless of what this invocation's capture-deletion scope
    was. A sidecar still referenced by a surviving capture (promoted or not:
    an unpromoted capture needs its sidecar to promote later) or by any
    promoted suite example (a tree ``clean`` never touches) is never
    deleted, even with ``--all``.

    Two separate confirmations gate the two separate deletions the sweep can
    add to a run that only asked about capture files: one for the captures
    (now naming sidecars as a possible consequence too, since deleting a
    capture can orphan the sidecar only it referenced), one for the sweep
    itself once it knows exactly which sidecars it would remove -- including
    when zero captures matched (the scope named on the command line, or
    ``suite``, doesn't bound the sweep, so a no-op capture deletion can still
    precede a real sidecar sweep). Declining either aborts before anything is
    deleted from that step; ``--yes`` skips both.
    """
    console = Console()
    records = iter_captures(suite=suite, base=base, on_error=_warn_unreadable(console))

    if not all_captures:
        promoted = _promoted_capture_ids_or_warn(base=base, console=console)
        records = [r for r in records if r.envelope.capture_id in promoted]

    if not records:
        scope = "captures" if all_captures else "promoted captures"
        console.print(f"[dim]nothing to clean — no {scope} found.[/dim]")
    else:
        if not yes:
            scope = "all captures" if all_captures else "promoted captures"
            typer.confirm(
                f"Delete {len(records)} {scope}? "
                "(may also orphan, and later sweep, toolset sidecars they solely reference)",
                abort=True,
            )

        removed = 0
        for r in records:
            try:
                r.path.unlink()
                removed += 1
            except OSError as exc:
                console.print(f"[yellow]⚠[/yellow] could not remove {r.path}: {exc}")
        console.print(f"[green]✓[/green] removed {removed} capture file(s).")

    _sweep_orphan_toolsets(base=base, console=console, yes=yes)


def _sweep_orphan_toolsets(*, base: Path | None, console: Console, yes: bool) -> None:
    """Delete every toolset sidecar no capture or promoted suite example still references.

    Refcounts across BOTH trees a sidecar can be kept alive by --
    :func:`~evalshift_cli.captures.reader.capture_toolset_refs` (``<base>/captures/``,
    every capture regardless of promoted status, re-scanned fresh so it
    reflects whatever this invocation's deletion loop actually left behind)
    and :func:`~evalshift_cli.captures.reader.promoted_toolset_refs`
    (``<base>/suites/``, a tree ``capture clean`` never touches, so every ref
    found there is permanently live no matter what capture-deletion scope
    was requested). This is what stops the destructive path a captures-only
    refcount takes: deleting the last capture that used a toolset must never
    take down the sidecar a promoted ``golden.jsonl`` still depends on.

    Both refcounting calls now fail closed (C2 of the final review): a
    capture or promoted-case file that fails to parse aborts the sweep
    instead of being silently read as "references nothing", which is what
    previously let one corrupt file make every sidecar it touched look
    orphaned and sweep them for good. Naming the file that broke lets the
    operator fix or remove it and re-run rather than losing data to a typo.

    Confirms before deleting anything (unless ``yes``), naming the exact
    sidecar count -- computed after refcounting, so this never runs
    unconfirmed even when the caller's own capture-deletion scope matched
    nothing (the orphan set is global, not scoped to ``suite``/``--all``).
    Deletes only sidecars referenced by neither tree, and reports each one
    (there is no other way for an operator to tell which opaque, hex-named
    files were just removed).
    """
    toolsets_dir = toolsets_root(base)
    if not toolsets_dir.is_dir():
        return

    try:
        live_refs = capture_toolset_refs(base=base) | promoted_toolset_refs(base=base)
    except CaptureError as exc:
        console.print(
            f"[red]✗[/red] refusing to sweep toolset sidecars: "
            f"[cyan]{exc.path.name}[/cyan] failed to parse ({exc.summary}). A refcount that "
            "can't read every capture and promoted case cannot tell a live toolset from an "
            "orphan, so nothing was swept. Fix or remove that file, then re-run.",
        )
        raise typer.Exit(code=1) from exc

    live_hexes = {ref.removeprefix("sha256:") for ref in live_refs}
    orphans = [p for p in sorted(toolsets_dir.glob("*.json")) if p.stem not in live_hexes]
    if not orphans:
        return

    if not yes:
        typer.confirm(
            f"Sweep {len(orphans)} orphaned toolset sidecar(s) no longer referenced by any "
            "capture or promoted suite example?",
            abort=True,
        )

    removed = 0
    for path in orphans:
        try:
            path.unlink()
        except OSError as exc:
            console.print(f"[yellow]⚠[/yellow] could not remove {path}: {exc}")
            continue
        removed += 1
        console.print(f"[green]✓[/green] removed orphaned toolset sidecar {path}")

    if removed:
        console.print(f"[green]✓[/green] swept {removed} orphaned toolset sidecar(s).")


@capture_app.command(name="diff")
def capture_diff(
    capture_a: Annotated[str, typer.Argument(help="First capture id.")],
    capture_b: Annotated[str, typer.Argument(help="Second capture id.")],
    base: _BaseOption = None,
) -> None:
    """Diff the tool-call traces of two captures."""
    console = Console()
    try:
        left = find_capture(capture_a, base=base)
        right = find_capture(capture_b, base=base)
    except CaptureError as exc:
        console.print(exc.format_rich())
        raise typer.Exit(code=1) from exc

    diff = diff_traces(left.envelope.trace, right.envelope.trace)
    console.print(f"[bold]capture diff[/bold] {capture_a} → {capture_b}")
    for item in diff.items:
        source_name = item.source_name or "-"
        target_name = item.target_name or "-"
        suffix = (
            f" {item.field}: {item.source_value!r} -> {item.target_value!r}" if item.field else ""
        )
        console.print(f"{item.kind:14} {source_name:24} {target_name:24} {item.category}{suffix}")
    if not diff.items:
        console.print("no trace differences")


def _build_suite_entries(
    suite_paths: dict[str, Path],
    evaluators: Mapping[str, SuiteEvaluatorsOverride | None],
    *,
    config_path: Path,
    existing: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``suites:`` entries this sync should write.

    Only the suites just promoted are regenerated. Everything else already in
    the managed region is carried forward verbatim, so syncing one suite can
    never delete or rewrite another's entry -- the partition guarantee that
    makes per-suite evaluator blocks safe to regenerate one at a time.

    Args:
        suite_paths: Suite name -> its rebuilt ``golden.jsonl``.
        evaluators: Suite name -> the evaluator block derived from its rows.
        config_path: Path to ``evalshift.yaml``; suite paths are written
            relative to its directory so ``resolve_suite_path`` finds them.
        existing: What the managed region holds today (see
            :func:`parse_suites_region`).

    Returns:
        ``(entries, frozen)`` -- the entries to render, and, for every suite
        pinned with ``managed: false``, the entry sync *would* have written
        (the caller prints it rather than applying it).
    """
    config_dir = config_path.resolve().parent
    entries: dict[str, Any] = dict(existing)
    frozen: dict[str, Any] = {}
    for name in sorted(suite_paths):
        rel = Path(os.path.relpath(suite_paths[name].resolve(), config_dir)).as_posix()
        prior = existing.get(name)
        fresh = _keep_trace_invariants(
            suite_entry_payload(path=rel, evaluators=evaluators.get(name)),
            prior,
        )
        if isinstance(prior, Mapping) and prior.get("managed") is False:
            frozen[name] = fresh  # `entries[name]` already holds the frozen entry.
            continue
        entries[name] = fresh
    return entries, frozen


def _keep_trace_invariants(fresh: dict[str, Any], prior: object) -> dict[str, Any]:
    """Carry a managed entry's hand-written ``trace_invariants`` through a sync.

    Everything else in a managed entry is derived from captures and is
    regenerated. Trace rules are the exception: nobody can derive them, a team
    wrote and owns them, and silently dropping a contract on the next sync is
    exactly the "auto-update the baseline" failure they exist to prevent.

    Args:
        fresh: The entry this sync regenerated from the suite's rows.
        prior: The entry the managed region held before (any shape -- it is
            raw YAML, carried forward unvalidated like every other entry).

    Returns:
        ``fresh``, with the prior ``evaluators.trace_invariants`` value copied in
        verbatim when there was one; otherwise ``fresh`` unchanged.
    """
    if not isinstance(prior, Mapping):
        return fresh
    prior_evaluators = prior.get("evaluators")
    if not isinstance(prior_evaluators, Mapping) or "trace_invariants" not in prior_evaluators:
        return fresh
    evaluators = dict(fresh.get("evaluators") or {})
    evaluators["trace_invariants"] = prior_evaluators["trace_invariants"]
    return {**fresh, "evaluators": evaluators}


@capture_app.command(name="sync")
def capture_sync(
    suite: Annotated[
        str | None,
        typer.Option("--suite", help="Only sync captures for this suite."),
    ] = None,
    input_var: Annotated[
        str,
        typer.Option(
            "--input-var",
            help="Template-variable name for a bare-string model input.",
        ),
    ] = "input",
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Extra tag to attach to every promoted case (repeatable)."),
    ] = None,
    strict_args: Annotated[
        bool,
        typer.Option("--strict-args", help="Require exact tool-argument matches."),
    ] = False,
    names_only: Annotated[
        bool,
        typer.Option("--names-only", help="Match tool names only; ignore arguments."),
    ] = False,
    tool_count: Annotated[
        bool,
        typer.Option("--tool-count", help="Also pin the expected tool-call count."),
    ] = False,
    rounds: Annotated[
        str,
        typer.Option(
            "--rounds",
            help=(
                "How many recorded agent rounds to replay: 'first' (default, single-shot, "
                "scored against round 1) or 'all' (teacher-forced — every round is replayed "
                "with the recorded tool results fed back, scored round by round)."
            ),
        ),
    ] = "first",
    allow_errored: Annotated[
        bool,
        typer.Option(
            "--allow-errored",
            help=(
                "Promote captures whose turn recorded an error event. Off by "
                "default: a turn that failed before the agent acted is not "
                "ground truth."
            ),
        ),
    ] = False,
    config_path: _ConfigOption = Path(CONFIG_FILENAME),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite promoted cases that already exist."),
    ] = False,
    write: Annotated[
        bool,
        typer.Option(
            "--write/--print",
            help="Write the suites: block into evalshift.yaml (default), or only print it.",
        ),
    ] = True,
    keep_duplicates: Annotated[
        bool,
        typer.Option(
            "--keep-duplicates",
            help=(
                "Promote captures whose replayed content (inputs + history) "
                "duplicates an earlier capture in the same suite. Off by "
                "default: duplicate examples inflate n and corrupt the paired "
                "statistics."
            ),
        ),
    ] = False,
    repair: Annotated[
        bool,
        typer.Option(
            "--repair",
            help=(
                "Rebuild the inputs and history of promoted cases an earlier sync built from "
                "stringified message blocks (or left with empty inputs), re-fetching their "
                "captures from captures.store even though they are promoted; every other "
                "field of those cases is kept, and other promoted cases are untouched unless "
                "--force is also given."
            ),
        ),
    ] = False,
    since: _SinceOption = None,
    offline: _OfflineOption = False,
    base: _BaseOption = None,
) -> None:
    """Promote captures into suites and wire the suites: block into evalshift.yaml.

    Without limits, every capture is promoted. With captures.max_cases set,
    each suite keeps at most that many cases, swapping stale ones (older than
    captures.max_age, or recorded on another model than defaults.source_model)
    for fresh captures; reviewed cases are never retired. captures.max_age
    keeps older captures out of a suite. The fetch from captures.store takes
    only captures within captures.max_age and, with captures.max_cases set,
    each suite's newest 3x max_cases.
    """
    console = Console()
    _check_rounds(rounds)
    limits = _sync_limits(config_path, console=console, warn=True)
    now = _now()
    selection = _repair_selection(base=base, suite=suite, console=console)
    _maybe_fetch(
        config_path=config_path,
        offline=offline,
        since=since,
        suite=suite,
        base=base,
        console=console,
        force_ids=selection.refetch_ids if repair else frozenset(),
        window=limits.fetch_window(now),
        per_suite_limit=limits.fetch_budget,
    )
    if not repair:
        for line in stringified_warnings(selection):
            console.print(f"[yellow]⚠[/yellow] {escape(line)}")
    elif not selection.candidates:
        console.print("[dim]no promoted cases need repair.[/dim]")
    records = iter_captures(suite=suite, base=base, on_error=_warn_unreadable(console))
    if repair:
        _warn_unrepairable(selection, records, console=console)
    if not records and not (
        limits.max_cases is not None and _suites_with_cases(base=base, only=suite)
    ):
        where = f" for suite {suite!r}" if suite else ""
        console.print(f"[dim]no captures found{where} under .evalshift/captures/.[/dim]")
        return

    opts = PromoteOptions(
        input_var=input_var,
        tags=tuple(tags or ()),
        strict_args=strict_args,
        names_only=names_only,
        tool_count=tool_count,
        allow_errored=allow_errored,
        rounds="all" if rounds == "all" else "first",
        tool_properties=_declared_tool_properties([r.envelope for r in records], base=base),
    )

    promoted = 0
    wired_generation = 0
    skipped_existing = 0
    skipped_empty = 0
    skipped_duplicates = 0
    skipped_errored = 0
    skipped_no_toolset = 0
    skipped_multi_toolset = 0
    skipped_unreplayable = 0
    skipped_invalid = 0
    repaired = 0
    left_for_repair = 0
    backfilled = 0
    plan_lines: list[str] = []
    backed_by = selection.cases_by_capture()
    suites_seen: set[str] = set()
    # Suites planned only because their directory holds case files: no capture this run.
    directory_only: set[str] = set()
    # (suite, content key) -> the name of the case that owns that content,
    # promoted either by an earlier run (seeded from disk below) or by this one
    # (where the name is always the capture id).
    seen_examples: dict[tuple[str, str], str] = {}

    # Grouped per-suite (not globally) so a conversation_id collision across
    # unrelated suites can never merge their turns together.
    records_by_suite: dict[str, list[CaptureRecord]] = {}
    for record in records:
        envelope = record.envelope
        if not envelope.trace.events:
            console.print(
                f"[yellow]⚠[/yellow] skipping [cyan]{envelope.capture_id}[/cyan] "
                f"(suite {envelope.suite}): no events recorded.",
            )
            skipped_empty += 1
            continue
        suites_seen.add(envelope.suite)
        records_by_suite.setdefault(envelope.suite, []).append(record)

    planned = set(records_by_suite)
    if limits.max_cases is not None:
        # An over-cap suite is trimmed even when no new capture arrived for it. A directory
        # already planned under the suite name its captures record is not planned again.
        captured_dirs = {promoted_suite_dir(name, base=base).name for name in records_by_suite}
        planned |= _suites_with_cases(base=base, only=suite) - captured_dirs
    suites_seen |= planned
    for suite_name in sorted(planned):
        suite_records = records_by_suite.get(suite_name, [])
        # Per-suite, not global: a conversation_id is only unique within the
        # suite that recorded it, same reason the grouping above is per-suite.
        for warning in duplicate_turn_warnings(
            [r.envelope for r in suite_records],
        ):
            console.print(f"[yellow]⚠[/yellow] {suite_name}: {warning}")
        envelopes = {r.envelope.capture_id: r.envelope for r in suite_records}
        kept, unreadable = _load_promoted(suite_name, base=base)
        recorded_names = {c.suite for _, c in kept}
        if not suite_records and len(recorded_names) == 1:
            # A suite planned from its directory alone goes by the name its cases record
            # (``team/support``, not its directory ``team_support``), as its suites entry does.
            (recorded_name,) = recorded_names
            if promoted_suite_dir(recorded_name, base=base).name == suite_name:
                suites_seen.discard(suite_name)
                suites_seen.add(recorded_name)
                suite_name = recorded_name
        if not suite_records:
            directory_only.add(suite_name)
        if unreadable is None:
            kept, rewritten = _backfill_provenance(kept, envelopes)
            backfilled += rewritten
        if unreadable is not None and limits.active:
            console.print(
                f"[yellow]⚠[/yellow] {escape(suite_name)}: could not read its promoted cases "
                f"({unreadable}); left it unchanged.",
            )
            # Not rebuilt either: its golden.jsonl and suites entry stay as they are, and the
            # other suites still sync.
            suites_seen.discard(suite_name)
            continue
        # A kept case is its file: a copied or renamed case file can carry another's name.
        kept_paths = {path for path, _ in kept}
        kept_from = {c.from_capture for _, c in kept}
        stale_by_path: dict[Path, StaleReason | None] = {}
        rules: StalenessRules | None = None
        if limits.active:
            recorded = [c.source_model for _, c in kept] + [
                capture_provenance(r.envelope)[1]
                for r in suite_records
                if r.envelope.capture_id not in kept_from
            ]
            rules = _suite_rules(
                suite_name, limits, now=now, recorded_models=recorded, console=console
            )
            stale_by_path = {
                path: rules.stale_reason(case_instant(c), c.source_model) for path, c in kept
            }
        # Only the planner retires a stale twin, and never a reviewed one; any other stale
        # case keeps its content, or a fresh copy would be promoted beside it.
        replaceable = (
            {
                path
                for path, c in kept
                if stale_by_path.get(path) is not None and not is_reviewed(c.example)
            }
            if limits.max_cases is not None
            else set()
        )
        replaceable_from = {c.from_capture for path, c in kept if path in replaceable}
        ineligible: Counter[str] = Counter()
        pending: list[PromotedCase] = []
        pending_warnings: dict[str, list[str]] = {}
        built_pairs = build_conversation_examples(suite_records, opts, base=base)
        local_ids = {r.envelope.capture_id for r in suite_records}
        local_turns: dict[str, set[int]] = {}
        for r in suite_records:
            if r.envelope.conversation_id is not None and r.envelope.turn_index is not None:
                local_turns.setdefault(r.envelope.conversation_id, set()).add(r.envelope.turn_index)
        # Re-exercising an agent on the same data records duplicate captures.
        # Promoting both would duplicate the example, double n, and inflate
        # every downstream p-value/effect size. Dedup keys on the *built
        # example's* replayed content (inputs + history) — see
        # `example_content_key` for why not the envelope input_hash — and is
        # seeded from the cases already on disk so it spans sync runs.
        if not keep_duplicates:
            owners = _promoted_content_owners(suite_name, base=base, console=console)
            # A replaceable stale case does not claim its content: the same question captured
            # fresh replaces it (spec §5, rule 3).
            replaceable_names = {c.name for path, c in kept if path in replaceable}
            seen_examples.update(
                (key, owner) for key, owner in owners.items() if owner not in replaceable_names
            )
            # ...but a case that stays claims its content even when a replaceable case sharing
            # it was the one recorded as its owner, or the fresh copy would duplicate it.
            for path, case in kept:
                if path not in replaceable:
                    seen_examples.setdefault(
                        (suite_name, example_content_key(case.example)), case.name
                    )
            # A repair candidate's rebuilt content belongs to the candidate, whichever capture
            # is built first: the disk seed above only holds its stringified content, so
            # without this claim a capture replaying the same turn would promote as a new case.
            for record, built in built_pairs:
                backed = backed_by.get(record.envelope.capture_id)
                if backed is not None and built.blocked is None:
                    seen_examples.setdefault(
                        (suite_name, example_content_key(built.example)), backed[0].name
                    )
            # A stringified candidate whose capture is not local (a fresh clone, `capture
            # clean`) is rebuilt offline instead: a capture an earlier sync de-duplicated
            # against it was never promoted, and its correct rebuild no longer matches the
            # stringified content on disk. When the capture is local, its rebuild above wins.
            for candidate in selection.candidates:
                case = candidate.case
                if (
                    case.suite != suite_name
                    or candidate.stringified is None
                    or case.from_capture in local_ids
                ):
                    continue
                key = reconstructed_content_key(case)
                if key is not None:
                    seen_examples.setdefault((suite_name, key), case.name)
        repaired_here: list[str] = []
        for record, built in built_pairs:
            envelope = record.envelope
            backed = backed_by.get(envelope.capture_id)
            if backed is not None:
                # A capture behind a repair candidate is only ever written by --repair, and
                # then only the case's inputs and history -- never as a new case (a custom
                # `--as` name would otherwise get a second copy next to the broken one).
                # A rebuild that changes nothing is no repair (a genuinely input-less case),
                # nor is a blocked one of a case that was never stringified: both take the
                # normal path below, `--force` and the blocked counters included.
                if built.blocked is None:
                    promoted_at = _now_iso()
                    to_repair = [
                        fixed
                        for case in backed
                        if (fixed := repaired_case(case, built.example, promoted_at=promoted_at))
                        is not None
                    ]
                else:
                    to_repair = [c for c in backed if stringified_content(c.example) is not None]
                if to_repair:
                    if not repair:
                        left_for_repair += len(to_repair)
                        continue
                    for warning in built.warnings:
                        console.print(f"[yellow]⚠[/yellow] {envelope.capture_id}: {warning}")
                    if built.blocked is not None:
                        for case in to_repair:
                            console.print(
                                f"[yellow]⚠[/yellow] can't repair "
                                f"[cyan]{escape(case.name)}[/cyan]: {escape(built.blocked)}",
                            )
                    else:
                        # A turn whose own capture yields no history got it from its sibling
                        # turns; with one of them gone, the rebuild would drop it.
                        own_history = _records_own_history(envelope)
                        for case in to_repair:
                            missing = (
                                [] if own_history else selection.missing_turns(case, local_ids)
                            )
                            if missing:
                                console.print(
                                    f"[yellow]⚠[/yellow] can't repair "
                                    f"[cyan]{escape(case.name)}[/cyan]: conversation turn(s) "
                                    f"{escape(', '.join(missing))} are no longer local or in the "
                                    "store, so its history can't be rebuilt",
                                )
                                continue
                            # ``backed`` was loaded before this suite's backfill ran, so the
                            # repaired case would otherwise drop the provenance just written.
                            write_promoted_case(
                                _with_provenance(case, envelope), base=base, force=True
                            )
                            repaired_here.append(case.name)
                    continue
            if built.blocked is not None:
                console.print(
                    f"[yellow]⚠[/yellow] skipped [cyan]{envelope.capture_id}[/cyan]: "
                    f"{built.blocked}",
                )
                # Routed by *why* it was blocked, not lumped under one counter:
                # --allow-errored rescues an errored turn but can never rescue
                # a capture with no usable recorded toolset, one that switched
                # toolsets mid-run, one whose messages carry content a text
                # replay can't carry (unreplayable_content), or one no valid case
                # can be built from (invalid_case), so the summary below must not
                # offer that as a fix for any of them.
                if built.blocked_reason == "no_toolset":
                    skipped_no_toolset += 1
                elif built.blocked_reason == "multi_toolset":
                    skipped_multi_toolset += 1
                elif built.blocked_reason == "unreplayable_content":
                    skipped_unreplayable += 1
                elif built.blocked_reason == "invalid_case":
                    skipped_invalid += 1
                else:
                    skipped_errored += 1
                continue
            if rules is not None and envelope.capture_id not in kept_from:
                captured_at, recorded_model = capture_provenance(envelope)
                reason = rules.stale_reason(parse_instant(captured_at), recorded_model)
                if reason is not None:
                    ineligible[reason] += 1
                    continue
                # Earlier turns outside the fetch window or budget, or promoted elsewhere and so
                # never fetched: the history rebuilt without them would be wrong.
                if _misses_earlier_turns(envelope, local_turns):
                    ineligible["history"] += 1
                    continue
            if limits.max_cases is not None and envelope.capture_id in kept_from and not force:
                skipped_existing += 1
                continue
            content_key = (suite_name, example_content_key(built.example))
            owner = seen_examples.get(content_key)
            # `sync` writes the case under the capture id, so an owner of that
            # name is the very case this capture would refresh — let it through
            # to the already-promoted / --force handling below.
            if not keep_duplicates and owner is not None and owner != envelope.capture_id:
                skipped_duplicates += 1
                continue
            if envelope.capture_id not in replaceable_from:
                # A replaceable stale case refreshed under --force still leaves its content
                # to a fresh twin, whichever capture id sorts first.
                seen_examples.setdefault(content_key, envelope.capture_id)
            captured_at, source_model = capture_provenance(envelope)
            case = PromotedCase(
                name=envelope.capture_id,
                suite=envelope.suite,
                from_capture=envelope.capture_id,
                promoted_at=_now_iso(),
                source_input_hash=envelope.input_hash,
                code_version=envelope.code_version,
                promotion_source=built.promotion_source,
                cost_usd=built.cost_usd,
                cost_source=built.cost_source,
                conversation_id=built.example.conversation_id,
                turn_index=built.example.turn_index,
                captured_at=captured_at,
                source_model=source_model,
                example=built.example,
            )
            if limits.max_cases is not None and envelope.capture_id not in kept_from:
                # A case file already at its path (another capture's case saved under this
                # name) would be overwritten or fail to write: never planned as an addition.
                target = promoted_case_path(case, base=base)
                if target in kept_paths or (target.exists() and not force):
                    skipped_existing += 1
                    continue
                # Its warnings print once it is written: a capture the plan leaves out is quiet.
                pending.append(case)
                pending_warnings[case.name] = built.warnings
                continue
            for warning in built.warnings:
                console.print(f"[yellow]⚠[/yellow] {envelope.capture_id}: {warning}")
            try:
                write_promoted_case(case, base=base, force=force)
                promoted += 1
                if built.example.generation_config:
                    wired_generation += 1
            except FileExistsError:
                skipped_existing += 1
        if repaired_here:
            repaired += len(repaired_here)
            _warn_repaired_duplicates(suite_name, repaired_here, base=base, console=console)
        if ineligible["history"]:
            console.print(
                f"[yellow]⚠[/yellow] {escape(suite_name)}: {ineligible['history']} turn(s) not "
                "promoted: earlier turns of their conversation are not on disk (outside "
                "captures.max_age or the fetch budget, promoted on another machine or removed by "
                "capture clean, sampled out by EVALSHIFT_SAMPLE_RATE, or a turn_index that does "
                "not start at 0). Pass the full messages list to capture.model_call so each turn "
                "carries its own history.",
                soft_wrap=True,
            )
        suite_line: str | None = None
        if limits.max_cases is not None:
            # Planned by file stem, which is unique in the directory, so a retirement deletes
            # the very file the case was read from.
            by_stem = {path.stem: path for path, _ in kept}
            kept_plan = [
                KeptCase(
                    name=path.stem,
                    instant=case_instant(c),
                    stale=stale_by_path.get(path),
                    reviewed=is_reviewed(c.example),
                    route=examples_route([c.example]),
                    content_key=example_content_key(c.example),
                )
                for path, c in kept
            ]
            plan = plan_suite(kept_plan, _candidate_groups(pending), max_cases=limits.max_cases)
            chosen = {cid for group in plan.add for cid in group.capture_ids}
            written: list[PromotedCase] = []
            for case in pending:
                if case.name not in chosen:
                    continue
                try:
                    write_promoted_case(case, base=base, force=force)
                except FileExistsError:
                    skipped_existing += 1
                    continue
                for warning in pending_warnings[case.name]:
                    console.print(f"[yellow]⚠[/yellow] {case.name}: {warning}")
                written.append(case)
                promoted += 1
                if case.example.generation_config:
                    wired_generation += 1
            if len(written) < plan.added_cases:
                # Re-plan around what was written: a stale twin, or a case retired to make
                # room, stays when the capture meant to replace it was not added.
                plan = plan_suite(kept_plan, _candidate_groups(written), max_cases=limits.max_cases)
            retired: list[RetireReason] = []
            for stem, why in plan.retire:
                if retire_promoted_case(by_stem[stem]):
                    retired.append(why)
                else:
                    console.print(
                        f"[yellow]⚠[/yellow] {escape(suite_name)}: could not delete "
                        f"{escape(by_stem[stem].name)}; not counted as retired.",
                        soft_wrap=True,
                    )
            if plan.reviewed_over_cap:
                console.print(
                    f"[yellow]⚠[/yellow] {escape(suite_name)}: {plan.reviewed_over_cap} "
                    f"reviewed case(s) over captures.max_cases {limits.max_cases}; reviewed "
                    "cases are never retired, so it stays over.",
                )
            suite_line = _plan_line(
                suite_name,
                kept=len(kept),
                retired=retired,
                added=len(written),
                eligible=len(pending),
                ineligible=ineligible,
                max_age=limits.max_age,
            )
        elif limits.active:
            suite_line = _plan_line(
                suite_name,
                kept=len(kept),
                retired=None,
                added=0,
                eligible=0,
                ineligible=ineligible,
                max_age=limits.max_age,
            )
        if suite_line is not None:
            plan_lines.append(suite_line)

    suite_paths: dict[str, Path] = {}
    suite_evaluators: dict[str, SuiteEvaluatorsOverride | None] = {}
    for name in sorted(suites_seen):
        suite_dir = promoted_suite_dir(name, base=base)
        if not suite_dir.is_dir():
            continue
        golden_path = rebuild_golden_jsonl(suite_dir)
        try:
            promoted_suite = load_jsonl(golden_path)  # round-trip guard: must stay loadable.
        except SuiteError as exc:
            console.print(exc.format_rich())
            raise typer.Exit(code=1) from exc
        suite_paths[name] = golden_path
        # Derived from the rebuilt suite, not from this run's captures alone:
        # a partial sync must wire the evaluators the *whole* suite needs.
        suite_evaluators[name] = derive_suite_evaluators(promoted_suite.examples)

    summary = f"[green]✓[/green] promoted {promoted} capture(s) into {len(suite_paths)} suite(s)"
    if wired_generation:
        summary += f", wired generation config for {wired_generation} case(s)"
    if backfilled:
        summary += f", backfilled capture time and model on {backfilled} case(s)"
    if skipped_existing:
        summary += f", skipped {skipped_existing} already-promoted (use --force to overwrite)"
    if skipped_empty:
        summary += f", skipped {skipped_empty} with no events"
    if skipped_duplicates:
        summary += (
            f", skipped {skipped_duplicates} duplicate capture(s) (same input; "
            "use --keep-duplicates to keep)"
        )
    if skipped_errored:
        summary += (
            f", skipped {skipped_errored} errored capture(s) "
            "(use --allow-errored to promote anyway)"
        )
    if skipped_no_toolset:
        summary += (
            f", skipped {skipped_no_toolset} capture(s) with no usable recorded toolset "
            "(re-capture with a current evalshift-sdk; --allow-errored does not apply)"
        )
    if skipped_multi_toolset:
        summary += (
            f", skipped {skipped_multi_toolset} capture(s) that switched toolsets mid-run "
            "(not yet promotable as a single case; --allow-errored does not apply)"
        )
    if skipped_unreplayable:
        summary += (
            f", skipped {skipped_unreplayable} capture(s) with non-text content a text replay "
            "can't carry (--allow-errored does not apply)"
        )
    if skipped_invalid:
        summary += (
            f", skipped {skipped_invalid} capture(s) that could not form a valid case "
            "(--allow-errored does not apply)"
        )
    if repair:
        summary += f", repaired {repaired} case(s)"
    if left_for_repair:
        summary += f", left {left_for_repair} case(s) for --repair"
    console.print(summary + ".")
    for line in plan_lines:
        console.print(escape(line), soft_wrap=True)

    if not suite_paths:
        return

    config_exists = config_path.exists()
    config_text = config_path.read_text(encoding="utf-8") if config_exists else ""
    existing = parse_suites_region(config_text)
    # A suite synced only because its directory holds case files is rebuilt, but wired only
    # when it already has an entry: a team that took it out of suites: meant it.
    wire_paths = {
        name: path
        for name, path in suite_paths.items()
        if name not in directory_only or name in existing
    }
    if not wire_paths:
        return
    entries, frozen = _build_suite_entries(
        wire_paths,
        suite_evaluators,
        config_path=config_path,
        existing=existing,
    )
    suites_yaml = render_suites_yaml(entries)
    for name in sorted(frozen):
        console.print(
            f"[yellow]⚠[/yellow] {name}: managed: false — its suites entry is left as it "
            "is. Sync would have written:",
        )
        console.print(render_suites_yaml({name: frozen[name]}))

    if not write:
        console.print()
        console.print("[bold]Add this to evalshift.yaml:[/bold]")
        console.print(suites_yaml)
        _warn_ci_pin(console, config_path)
        return

    if not config_exists:
        console.print(
            f"[yellow]⚠[/yellow] {config_path} not found — run `evalshift init` first, "
            "or paste this block into your config:",
        )
        console.print(suites_yaml)
        _warn_ci_pin(console, config_path)
        return

    updated = inject_suites_block(config_text, suites_yaml)
    if updated is None:
        console.print(
            f"[yellow]⚠[/yellow] no managed suites markers found in {config_path}; "
            "paste this block in yourself:",
        )
        console.print(suites_yaml)
        _warn_ci_pin(console, config_path)
        return

    config_path.write_text(updated, encoding="utf-8")
    console.print(f"[green]✓[/green] wired {len(wire_paths)} suite(s) into {config_path}")
    first = sorted(wire_paths)[0]
    console.print(f"run: [cyan]evalshift compare --suite-name {first} --to <candidate>[/cyan]")
    _warn_ci_pin(console, config_path)


__all__ = ["capture_app"]
