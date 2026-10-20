"""Locate, load, and validate capture files on disk.

Path contract::

    <base>/captures/<suite>/<capture_id>.json                 standalone captures, and every
                                                              capture evalshift-sdk < 0.7.0 wrote
    <base>/captures/<suite>/conv_<16 hex>/<capture_id>.json   a conversation's turns,
                                                              evalshift-sdk >= 0.7.0

``<suite>`` is the SDK's ``_safe_segment`` of the suite name (:func:`capture_suite_dirname`);
``conv_<16 hex>`` is the first 16 hex digits of ``sha256(conversation_id)``. ``base`` resolves
to ``$EVALSHIFT_DIR`` if set, else ``.evalshift`` (CWD relative) — matching the SDK's
``FileSink`` exactly, so the CLI reads from wherever the agent process wrote.

Promoted cases live under ``<base>/suites/<suite>/``. Toolset sidecars — one
content-addressed file per distinct toolset, shared across every capture and
promoted case that used it — live under ``<base>/toolsets/<hex>.json`` (see
:func:`toolset_path`; the fingerprint itself is
:func:`evalshift_cli.captures.toolset.fingerprint_tools`).

Unlike the capture *write* path in the SDK (which fails open — a broken write
is silently dropped), the CLI read path **raises** a structured
:class:`CaptureError`. A user running ``capture promote`` on a corrupt file
wants to be told, not to get a silent no-op.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.captures.toolset import fingerprint_tools
from evalshift_cli.evaluators.tool_models import ToolSpec
from evalshift_cli.traces.models import ModelCallEvent

CAPTURES_DIRNAME = "captures"
SUITES_DIRNAME = "suites"
TOOLSETS_DIRNAME = "toolsets"
_DEFAULT_BASE = ".evalshift"

#: The folder evalshift-sdk (0.7.0 and later) files a conversation's turns under, inside the
#: suite's directory: ``conv_`` + the first 16 hex digits of ``sha256(conversation_id)``.
#: Lowercase only, so two names never collide on a case-insensitive filesystem. Always used
#: with ``fullmatch``.
CONVERSATION_DIR = re.compile(r"conv_[0-9a-f]{16}")


def is_conversation_dir(name: str) -> bool:
    """Whether ``name`` is a conversation folder the SDK writes (``conv_<16 hex>``)."""
    return CONVERSATION_DIR.fullmatch(name) is not None


def capture_suite_dirname(suite: str) -> str:
    """The directory the SDK files ``suite``'s captures under.

    Mirrors ``evalshift.sinks.file._safe_segment`` exactly: separators become ``_``, every
    ``..`` becomes ``_``, leading and trailing dots and spaces are stripped, and an empty result
    is ``_``. So ``team/support`` reads from ``team_support``, where the SDK wrote it, and a
    suite argument can never name a conversation folder.
    """
    cleaned = suite.replace("\\", "/").replace("/", "_")
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.strip(". ")
    return cleaned or "_"


def _capture_files(suite_dir: Path) -> list[Path]:
    """A suite directory's capture files: its flat ones, then each conversation folder's."""
    files = sorted(suite_dir.glob("*.json"))
    for child in sorted(suite_dir.iterdir()):
        if child.is_dir() and is_conversation_dir(child.name):
            files.extend(sorted(child.glob("*.json")))
    return files


# Major schema versions this CLI knows how to read. Bumped to 2 for per-call
# toolset capture (``ModelCallEvent.toolset_ref`` / ``.tools_offered``): a 1.x
# capture predates those fields entirely. There is no migration and no
# dual-major support, so it is refused loudly rather than read as if the
# fields were simply absent.
#
# Deliberately the MAJOR only: an SDK MINOR bump adds optional fields the trace
# models already accept (2.1.0 adds ``ModelCallEvent.requested_tool_calls``),
# so refusing it would strand every install on the older SDK for no reason.
_SUPPORTED_MAJOR = 2

CaptureErrorKind = Literal[
    "missing",
    "not_a_file",
    "json_parse",
    "schema",
    "unsupported_version",
    "fingerprint_mismatch",
]


class CaptureError(Exception):
    """Raised when a capture file is missing, unparseable, or unreadable."""

    def __init__(self, path: Path, kind: CaptureErrorKind, summary: str) -> None:
        self.path = path
        self.kind: CaptureErrorKind = kind
        self.summary = summary
        super().__init__(self.format_plain())

    def format_plain(self) -> str:
        """Render this error as a multi-line plain-text string."""
        return f"Capture error in {self.path}:\n  {self.summary}"

    def format_rich(self) -> RenderableType:
        """Render this error inside a Rich :class:`Panel`."""
        body: list[RenderableType] = [Text(self.summary, style="bold red")]
        return Panel(
            Group(*body),
            title=f"[red]Invalid capture[/red]: {self.path}",
            title_align="left",
            border_style="red",
        )


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    """A capture file paired with its parsed envelope."""

    path: Path
    envelope: CaptureEnvelope


def capture_base() -> Path:
    """Return the capture base dir: ``$EVALSHIFT_DIR`` if set, else ``.evalshift``."""
    env = os.environ.get("EVALSHIFT_DIR")
    if env:
        return Path(env)
    return Path(_DEFAULT_BASE)


def captures_root(base: Path | None = None) -> Path:
    """Return ``<base>/captures``."""
    return (base if base is not None else capture_base()) / CAPTURES_DIRNAME


def suites_root(base: Path | None = None) -> Path:
    """Return ``<base>/suites``."""
    return (base if base is not None else capture_base()) / SUITES_DIRNAME


def toolsets_root(base: Path | None = None) -> Path:
    """Return ``<base>/toolsets``."""
    return (base if base is not None else capture_base()) / TOOLSETS_DIRNAME


def toolset_path(ref: str, *, base: Path | None = None) -> Path:
    """Return the sidecar path for a toolset fingerprint ``ref``.

    ``ref`` is a full fingerprint value (``"sha256:<hex>"``, e.g. a ``toolset_ref`` read off a
    capture or suite example). The ``sha256:`` prefix is stripped for the on-disk filename: that
    prefix marks the *value* as content-addressed and appears only in ``toolset_ref`` fields,
    never in a path -- so the sidecar for ``"sha256:abc123"`` is ``<base>/toolsets/abc123.json``,
    not ``<base>/toolsets/sha256:abc123.json``.
    """
    return toolsets_root(base) / f"{ref.removeprefix('sha256:')}.json"


def load_toolset(
    ref: str,
    *,
    base: Path | None = None,
    cache: MutableMapping[str, list[ToolSpec]] | None = None,
) -> list[ToolSpec]:
    """Resolve a ``toolset_ref`` to the tool specs its sidecar holds.

    Built on :meth:`~evalshift_cli.evaluators.tool_models.ToolSpec.from_dict`, deliberately **not**
    on :func:`evalshift_cli.evaluators.tool_loader.load_tools`: ``load_tools`` raises on an empty
    list, and the empty toolset here is a real, fingerprinted, first-class value -- "this agent
    was offered no tools" -- not an absence to reject.

    Accepts the sidecar's wire shape (``{"tools": [...], ...}`` -- see
    ``PER_CALL_TOOLSET_CAPTURE_PLAN.md`` § Wire format) or a bare list, mirroring
    :func:`evalshift_cli.evaluators.tool_loader.load_tools`'s top-level flexibility. Both the SDK's
    ``ToolsetSink`` shape (``{"schema_version", "fingerprint", "tools"}``) and the checked-in
    example suites' hand-authored shape (``{"tools": [...]}``, no ``schema_version`` or
    ``fingerprint`` field) are accepted equally.

    After parsing, the ``tools`` array is re-fingerprinted via
    :func:`~evalshift_cli.captures.toolset.fingerprint_tools` and checked against ``ref`` --
    content-addressing is only an integrity guarantee if the content is actually checked against
    the address that named it, otherwise it is a naming convention a hand-edit (the checked-in
    example sidecars are committed, hand-editable files) can silently violate. ``ref`` is the
    only fingerprint this check treats as authoritative: a sidecar's own embedded
    ``fingerprint`` field (the SDK shape only -- a hand-authored one has none), if present, is
    never read here. Every other part of the system already trusts ``ref`` as this toolset's
    identity -- cache keys, ``capture_toolset_refs``/``promoted_toolset_refs``' refcounting, a
    suite example's own ``toolset_ref`` field -- so it is the one value worth reconciling
    everything else against, and a third, unread value is simpler to ignore than to reconcile.

    Args:
        ref: A ``toolset_ref`` value, e.g. read off a capture's ``ModelCallEvent`` or a
            promoted :class:`~evalshift_cli.suite.models.SuiteExample`. May carry the ``sha256:``
            prefix or not -- see :func:`toolset_path`.
        base: Capture base dir. ``None`` resolves via :func:`capture_base`.
        cache: When given, resolved toolsets are looked up and stored here by ``ref``, so N
            captures/examples sharing a toolset read, parse, AND VERIFY its sidecar at most
            once per cache (typically scoped to one CLI invocation) -- a cache hit returns
            immediately, before any disk I/O, so it is never re-verified either. ``None`` (the
            default) resolves fresh from disk on every call.

    Raises:
        CaptureError: if the sidecar is missing, not a file, contains invalid JSON, has the
            wrong top-level shape, any tool entry fails :meth:`ToolSpec.from_dict`, or its
            ``tools`` array does not actually fingerprint to ``ref``
            (``kind="fingerprint_mismatch"``).
    """
    if cache is not None and ref in cache:
        return cache[ref]

    path = toolset_path(ref, base=base)
    if not path.exists():
        raise CaptureError(path, "missing", f"toolset sidecar not found: {path}")
    if not path.is_file():
        raise CaptureError(path, "not_a_file", f"expected a file, got {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaptureError(path, "json_parse", f"invalid JSON: {exc.msg}") from exc

    raw_tools = payload.get("tools") if isinstance(payload, dict) else payload
    if not isinstance(raw_tools, list):
        raise CaptureError(
            path,
            "schema",
            "expected a list of tools (or a mapping with a 'tools' key); "
            f"got {type(raw_tools).__name__}",
        )

    tools: list[ToolSpec] = []
    for idx, entry in enumerate(raw_tools):
        if not isinstance(entry, dict):
            raise CaptureError(
                path,
                "schema",
                f"tools[{idx}]: expected a dict, got {type(entry).__name__}",
            )
        try:
            tools.append(ToolSpec.from_dict(entry))
        except (ValueError, KeyError, ValidationError) as exc:
            raise CaptureError(path, "schema", f"tools[{idx}]: {exc}") from exc

    # Integrity check: fingerprint the RAW parsed dicts (`raw_tools`), the exact input
    # `fingerprint_tools` takes everywhere else in this codebase -- never the round-tripped
    # `ToolSpec` objects built above, whose `description` defaults to `""` and would
    # re-canonicalise a tool that omitted the key differently than the sidecar's own bytes do.
    # `fingerprint_tools` does its own single internal sort (by name); nothing here sorts
    # `raw_tools` again or otherwise reorders it first.
    try:
        recomputed = fingerprint_tools(raw_tools)
    except (KeyError, TypeError) as exc:
        # Every entry above already parsed as a valid ToolSpec, which accepts both the
        # canonical {name, description, input_schema} shape and OpenAI's {"function": {...}}
        # shape -- but fingerprint_tools only understands the former (a sidecar is documented
        # to hold pre-normalised tools, never a live provider object). An OpenAI-shape raw
        # entry would parse fine above yet has no top-level "name" key to sort by here. Treat
        # that mismatch as a schema problem rather than letting a bare KeyError/TypeError
        # escape this function's documented "only raises CaptureError" contract.
        raise CaptureError(
            path,
            "schema",
            f"could not fingerprint tools for verification: {exc}",
        ) from exc
    if recomputed != ref:
        raise CaptureError(
            path,
            "fingerprint_mismatch",
            f"sidecar contents do not match the fingerprint that named them -- reached via "
            f"ref {ref!r}, but its tools array actually fingerprints to {recomputed!r}. "
            "Re-run `capture sync` to regenerate this sidecar, or fix the file by hand if "
            "you edited it directly.",
        )

    if cache is not None:
        cache[ref] = tools
    return tools


def _check_supported_version(path: Path, payload: dict[str, object]) -> None:
    raw = payload.get("schema_version")
    if not isinstance(raw, str) or not raw:
        raise CaptureError(path, "schema", "missing or invalid 'schema_version'")
    try:
        major = int(raw.split(".", 1)[0])
    except ValueError as exc:
        raise CaptureError(path, "schema", f"unparseable schema_version {raw!r}") from exc
    if major != _SUPPORTED_MAJOR:
        raise CaptureError(
            path,
            "unsupported_version",
            f"capture schema_version {raw!r} is not supported "
            f"(this CLI reads major version {_SUPPORTED_MAJOR}.x)",
        )


def load_capture(path: str | Path) -> CaptureEnvelope:
    """Load and validate a single capture file.

    Raises:
        CaptureError: if the file is missing, not a file, contains invalid
            JSON, declares an unsupported schema major version, or fails
            envelope/trace validation.
    """
    capture_path = Path(path)
    if not capture_path.exists():
        raise CaptureError(capture_path, "missing", f"file not found: {capture_path}")
    if not capture_path.is_file():
        raise CaptureError(capture_path, "not_a_file", f"expected a file, got {capture_path}")

    try:
        payload = json.loads(capture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaptureError(capture_path, "json_parse", f"invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise CaptureError(capture_path, "schema", "capture must be a JSON object")

    _check_supported_version(capture_path, payload)

    try:
        return CaptureEnvelope.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        msg = str(first.get("msg", "validation failed"))
        summary = f"{loc}: {msg}" if loc else msg
        raise CaptureError(capture_path, "schema", summary) from exc


def iter_captures(
    suite: str | None = None,
    *,
    base: Path | None = None,
    on_error: Callable[[CaptureError], None] | None = None,
) -> list[CaptureRecord]:
    """Return every readable capture, optionally filtered to one ``suite``.

    ``suite`` maps to the directory the SDK writes it to (:func:`capture_suite_dirname`, so
    ``team/support`` reads ``team_support/``). Each suite directory's ``*.json`` files are
    read, then those one level down in its ``conv_<16 hex>/`` conversation folders.

    Capture files that fail to load are skipped; each failure is reported to
    ``on_error`` (if given) so callers can warn instead of silently dropping
    the file. Results are sorted by ``(suite, capture_id)`` for deterministic
    listings.
    """
    root = captures_root(base)
    if not root.is_dir():
        return []

    suite_dirs = (
        [root / capture_suite_dirname(suite)]
        if suite is not None
        else sorted(p for p in root.iterdir() if p.is_dir())
    )

    records: list[CaptureRecord] = []
    for suite_dir in suite_dirs:
        if not suite_dir.is_dir():
            continue
        for path in _capture_files(suite_dir):
            try:
                envelope = load_capture(path)
            except CaptureError as exc:
                if on_error is not None:
                    on_error(exc)
                continue
            records.append(CaptureRecord(path=path, envelope=envelope))
    records.sort(key=lambda r: (r.envelope.suite, r.envelope.capture_id))
    return records


def find_capture(
    capture_id: str,
    *,
    suite: str | None = None,
    base: Path | None = None,
) -> CaptureRecord:
    """Locate one capture by id (the file stem), optionally within a suite.

    Raises:
        CaptureError: ``kind='missing'`` if no capture matches.
    """
    for record in iter_captures(suite=suite, base=base):
        if record.envelope.capture_id == capture_id or record.path.stem == capture_id:
            return record
    where = f" in suite {suite!r}" if suite is not None else ""
    raise CaptureError(
        captures_root(base),
        "missing",
        f"no capture with id {capture_id!r}{where}",
    )


def _iter_promoted_cases(
    *, base: Path | None = None, suite: str | None = None
) -> Iterator[PromotedCase]:
    """Yield every promoted-case file under ``<base>/suites/`` as a parsed :class:`PromotedCase`.

    The shared walk behind :func:`promoted_capture_ids` and
    :func:`promoted_toolset_refs` (M2 of the final review): both used to be
    independent copies of this loop, one swallowing parse failures with a
    bare ``except ... continue`` -- so the fail-closed contract below had to
    be added twice, and correctly, or the two would silently diverge again.
    Sharing it means it can only land once.

    Fails closed (C2): a case file that fails to parse raises immediately
    rather than being skipped. Both callers feed ``capture clean``'s
    orphan-sidecar refcount, which runs immediately before an unlink --
    reading "cannot parse" as "references nothing" there is what let one
    corrupt case file make its suite's sidecar look orphaned and get swept
    out from under a live ``golden.jsonl``. A caller that wants best-effort
    (non-destructive) behaviour instead -- e.g. ``capture list``'s "promoted"
    column -- catches :class:`CaptureError` itself; this generator's job is
    only to never let the failure pass as silence.

    Raises:
        CaptureError: ``kind="schema"``, naming the first case file (in
            sorted order, for determinism) that fails to parse as a
            :class:`PromotedCase`.
    """
    root = suites_root(base)
    if not root.is_dir():
        return

    suite_dirs = (
        [root / suite] if suite is not None else sorted(p for p in root.iterdir() if p.is_dir())
    )
    for suite_dir in suite_dirs:
        if not suite_dir.is_dir():
            continue
        for path in sorted(suite_dir.glob("*.json")):
            try:
                yield PromotedCase.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValidationError, OSError, ValueError) as exc:
                raise CaptureError(
                    path, "schema", f"promoted case file failed to parse: {exc}"
                ) from exc


def promoted_capture_ids(*, base: Path | None = None, suite: str | None = None) -> set[str]:
    """Return the set of ``capture_id`` values already promoted into a suite.

    Reads every promoted-case file under ``<base>/suites/`` (see
    :func:`_iter_promoted_cases`) and collects each case's ``from_capture``.

    Raises:
        CaptureError: if any promoted-case file under the scanned suite
            dir(s) fails to parse (C2) -- see :func:`_iter_promoted_cases`.
            Callers for whom that is too strict (e.g. a read-only listing)
            should catch it and degrade explicitly rather than relying on
            this function to swallow it.
    """
    return {case.from_capture for case in _iter_promoted_cases(base=base, suite=suite)}


def envelope_toolset_ref(envelope: CaptureEnvelope) -> str | None:
    """The ``toolset_ref`` recorded on ``envelope``'s first ``model_call`` event, if any.

    ``None`` when the capture has no ``model_call`` event, or its first one
    predates per-call toolset capture (an old capture, or one an outdated SDK
    never stamped). Public because more than one caller needs "which toolset
    did this capture's first call use" -- ``cli.commands.capture.
    _declared_tool_properties`` (the wrapper-argument-unwrapping heuristic,
    deliberately scoped to one representative toolset). For "every toolset
    this capture used," see :func:`envelope_toolset_refs` instead -- a
    capture that switched toolsets mid-run needs all of them protected, not
    just the first.
    """
    first_model_call = next(
        (e for e in envelope.trace.events if isinstance(e, ModelCallEvent)), None
    )
    return first_model_call.toolset_ref if first_model_call is not None else None


def envelope_toolset_refs(envelope: CaptureEnvelope) -> set[str]:
    """Every distinct ``toolset_ref`` recorded across ``envelope``'s ``model_call`` events.

    The SDK stamps ``toolset_ref`` per call, so a capture whose agent
    switched toolsets mid-run -- the workload per-call toolset capture exists
    for -- legitimately carries more than one distinct ref across its
    ``model_call`` events (C1 of the final review). Used by
    :func:`capture_toolset_refs`, whose refcount must protect every round's
    sidecar, not just the first: :func:`envelope_toolset_ref` (singular)
    looking only at the first call is exactly what let a round-2+ sidecar
    read as unreferenced and get swept.
    """
    return {
        e.toolset_ref
        for e in envelope.trace.events
        if isinstance(e, ModelCallEvent) and e.toolset_ref is not None
    }


def _reraise(exc: CaptureError) -> None:
    """``on_error`` hook for :func:`iter_captures` that re-raises instead of skipping.

    Used by :func:`capture_toolset_refs` to fail closed (C2): a refcount that
    precedes a delete must never read "cannot parse" as "references nothing"
    -- silently skipping an unreadable capture here is what let one corrupt
    file sweep every sidecar its readable siblings still used.
    """
    raise exc


def capture_toolset_refs(*, base: Path | None = None) -> set[str]:
    """Every ``toolset_ref`` recorded by a readable capture under ``<base>/captures/``.

    Spans every suite, unfiltered: the toolsets namespace is global (one
    sidecar shared across every capture/example that used it, regardless of
    suite), so refcounting it must be global too. Promoted and unpromoted
    captures both count -- an unpromoted capture still needs its sidecar to
    promote later, so it is exactly as live a reference as a promoted one.
    Every ``model_call`` event on a capture contributes (see
    :func:`envelope_toolset_refs`), not just the first -- a capture that
    switched toolsets mid-run needs every round's sidecar protected. A
    capture with no recorded ``toolset_ref`` contributes nothing.

    Used by ``capture clean``'s orphan-sidecar sweep (V5) as one half of the
    live set a sidecar must be absent from before it is safe to delete.

    Raises:
        CaptureError: if any capture under ``<base>/captures/`` fails to
            parse (C2) -- unlike :func:`iter_captures`'s own default
            leniency, a refcount that precedes a delete must fail closed
            rather than silently undercounting.
    """
    refs: set[str] = set()
    for record in iter_captures(base=base, on_error=_reraise):
        refs |= envelope_toolset_refs(record.envelope)
    return refs


def promoted_toolset_refs(*, base: Path | None = None) -> set[str]:
    """Every ``toolset_ref`` a promoted suite example under ``<base>/suites/`` still uses.

    Mirrors :func:`promoted_capture_ids`'s walk of the same tree (both share
    :func:`_iter_promoted_cases`), collecting ``example.toolset_ref`` instead
    of ``from_capture``. ``capture clean`` never deletes this tree, so every
    ref found here is permanently live regardless of what a given clean
    invocation is about to remove from ``<base>/captures/`` -- this is what
    keeps a promoted suite's sidecar alive even after every capture that
    produced it is gone. An example carrying inline ``tools`` instead of a
    ``toolset_ref`` contributes nothing, same as a capture with none
    recorded.

    Used by ``capture clean``'s orphan-sidecar sweep (V5) as the other half
    of the live set -- see :func:`capture_toolset_refs`.

    Raises:
        CaptureError: if any promoted-case file fails to parse (C2) -- see
            :func:`_iter_promoted_cases`.
    """
    return {
        case.example.toolset_ref
        for case in _iter_promoted_cases(base=base)
        if case.example.toolset_ref is not None
    }


__all__ = [
    "CAPTURES_DIRNAME",
    "CONVERSATION_DIR",
    "SUITES_DIRNAME",
    "TOOLSETS_DIRNAME",
    "CaptureError",
    "CaptureErrorKind",
    "CaptureRecord",
    "capture_base",
    "capture_suite_dirname",
    "capture_toolset_refs",
    "captures_root",
    "envelope_toolset_ref",
    "envelope_toolset_refs",
    "find_capture",
    "is_conversation_dir",
    "iter_captures",
    "load_capture",
    "load_toolset",
    "promoted_capture_ids",
    "promoted_toolset_refs",
    "suites_root",
    "toolset_path",
    "toolsets_root",
]
