"""Mirror captures from a user-owned object store into the local capture directory.

The SDK's ``ObjectStoreSink`` writes the *local layout* under a bucket prefix --
``captures/<suite>/cap_<hex>.json`` and ``toolsets/<hex>.json`` -- so the CLI never reads a
bucket directly. It fetches what is missing into ``<base>/captures`` and ``<base>/toolsets``
and every other command runs on the local mirror unchanged. Two properties make that cheap:
captures are immutable (a file that exists locally is current) and toolsets are
content-addressed (same name, same bytes), so no ETag or mtime is ever compared.

Keys read from a bucket are untrusted. Only basenames shaped like the SDK's are accepted
(:data:`_CAPTURE_NAME`, :data:`_TOOLSET_NAME`), a suite segment must be one the SDK's
``_safe_segment`` could have produced (:func:`_is_safe_suite_segment` -- so ``Support Agent``
or ``ünïcode`` pass, while ``..``, ``.hidden``, a backslash or a ``:`` do not), and anything
else is counted as malformed and never written. As a second line of defence every destination
must also normalize inside its root (:func:`_absolute`), so no key can place a file outside
``<base>``. Local writes are atomic (temp file beside the target, then :func:`os.replace`) so
an interrupted fetch leaves nothing half-written.

The write side's protocol lives in the SDK (``evalshift.stores``); this module carries the
read side. They share the key layout and URI grammar, not code.
"""

from __future__ import annotations

import importlib.util
import os
import re
import tempfile
from collections.abc import Collection, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from evalshift_cli.captures.reader import captures_root, toolsets_root
from evalshift_cli.captures.store_uri import StoreURI, package_for_module

#: Largest object a fetch will download. Captures are kilobytes; this is a safety stop.
MAX_OBJECT_BYTES = 32 * 1024 * 1024

# Always used with ``fullmatch``: ``$`` also matches before a trailing newline.
_CAPTURE_NAME = re.compile(r"[A-Za-z0-9_-]+\.json")
_TOOLSET_NAME = re.compile(r"[0-9a-f]{64}\.json")
_DURATION = re.compile(r"(\d+)([mhd])")

#: How to fix credentials, per store scheme, for a list or download failure.
_CREDENTIALS_HINT: dict[str, str] = {
    "s3": "check AWS credentials (aws sso login / AWS_PROFILE / the CI OIDC role) and network",
    "gs": "check Google credentials (gcloud auth application-default login / Workload Identity) "
    "and network",
    "az": "check Azure credentials (az login / managed identity) and network",
}

#: Every module a scheme's pip extra provides; ``open_store`` checks all of them.
_REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    "s3": ("boto3",),
    "gs": ("google.cloud.storage",),
    "az": ("azure.storage.blob", "azure.identity"),
}


def _is_safe_suite_segment(segment: str) -> bool:
    """Whether ``segment`` is a suite directory name the SDK could have written.

    The SDK's ``_safe_segment`` turns separators into ``_``, collapses every ``..`` and strips
    leading and trailing dots and spaces, so it never emits an empty name, a path separator,
    ``..`` or an edge dot/space. Anything else came from somewhere other than the SDK and could
    steer a write (``..``, ``.``, an empty segment, a Windows ``\\``), so it is refused. NUL is
    refused too: no filesystem accepts it.

    ``:`` is the one deliberate narrowing against ``_safe_segment``, which keeps it: on Windows
    ``D:`` joins as a drive-relative path and ``a:b`` names a drive or an NTFS alternate data
    stream, either of which lands outside ``<base>``. A suite named with a colon is mirrored
    nowhere and counted as malformed.
    """
    return (
        bool(segment)
        and not any(ch in segment for ch in ("/", "\\", ":", "\x00"))
        and ".." not in segment
        and segment[0] not in ". "
        and segment[-1] not in ". "
    )


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """One listed object.

    Attributes:
        key: Key relative to the store's prefix, e.g. ``captures/<suite>/cap_<hex>.json``.
        last_modified: The provider's timestamp, timezone-aware, or ``None`` if unknown.
        size: Size in bytes from the listing, or ``None`` if the provider did not say.
    """

    key: str
    last_modified: datetime | None
    size: int | None


@runtime_checkable
class RemoteStore(Protocol):
    """The read side of an object store: list keys under a prefix and fetch one object."""

    uri: str

    def list(self, prefix: str) -> Iterator[ObjectInfo]: ...

    def get(self, key: str) -> bytes: ...


class RemoteStoreError(Exception):
    """A remote store could not be opened, listed or read.

    Attributes:
        summary: One sentence saying what failed.
        hint: What to do about it, when there is something to do.
    """

    def __init__(self, summary: str, *, hint: str | None = None) -> None:
        self.summary = summary
        self.hint = hint
        super().__init__(summary if hint is None else f"{summary} ({hint})")

    def format_rich(self) -> RenderableType:
        """Render this error inside a Rich :class:`Panel`."""
        body: list[RenderableType] = [Text(self.summary, style="bold red")]
        if self.hint:
            body.append(Text(self.hint, style="dim"))
        return Panel(
            Group(*body),
            title="[red]Remote capture store[/red]",
            title_align="left",
            border_style="red",
        )


class RemoteStoreUnavailable(RemoteStoreError):  # noqa: N818 — the planned public name.
    """The client library for the store's scheme is not installed."""


def _installed(module: str) -> bool:
    """Whether ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package is missing (no `google` at all)
        return False


def open_store(parsed: StoreURI) -> RemoteStore:
    """Return the adapter for ``parsed`` without building its client yet.

    Args:
        parsed: A URI from :func:`evalshift_cli.captures.store_uri.parse_store_uri`.

    Returns:
        The scheme's adapter; its client is built on the first ``list`` or ``get``.

    Raises:
        RemoteStoreUnavailable: when a client module the scheme needs is not installed. The
            summary names the package that provides the first missing module; the hint names
            the pip command that installs everything the scheme needs.
    """
    for module in _REQUIRED_MODULES[parsed.scheme]:
        if not _installed(module):
            raise RemoteStoreUnavailable(
                f"{parsed.scheme}:// captures store needs {package_for_module(module)}, "
                "which is not installed",
                hint=f"install it: pip install {parsed.packages}",
            )
    if parsed.scheme == "s3":
        from evalshift_cli.captures.stores.s3 import S3Store

        return S3Store(parsed.bucket, parsed.prefix)
    if parsed.scheme == "gs":
        from evalshift_cli.captures.stores.gcs import GCSStore

        return GCSStore(parsed.bucket, parsed.prefix)
    from evalshift_cli.captures.stores.azure import AzureBlobStore

    return AzureBlobStore(parsed.bucket, parsed.container or "", parsed.prefix)


@dataclass
class FetchSummary:
    """What one fetch did, for the one-line console summary."""

    store_uri: str
    captures: int = 0
    toolsets: int = 0
    skipped_existing: int = 0
    skipped_promoted: int = 0
    skipped_old: int = 0
    skipped_large: int = 0
    skipped_malformed: int = 0

    def describe(self) -> str:
        """``fetched N capture(s) and M toolset(s) from <uri> (skipped ...)``."""
        text = (
            f"fetched {self.captures} capture(s) and {self.toolsets} toolset(s) "
            f"from {self.store_uri}"
        )
        skipped = [
            f"{n} {label}"
            for n, label in (
                (self.skipped_existing, "already local"),
                (self.skipped_promoted, "promoted"),
                (self.skipped_old, "older than --since"),
                (self.skipped_large, "over the size cap"),
                (self.skipped_malformed, "malformed"),
            )
            if n
        ]
        return f"{text} (skipped {', '.join(skipped)})" if skipped else text


def parse_since(text: str, *, now: datetime | None = None) -> datetime:
    """Turn a ``--since`` value into an aware UTC instant.

    Accepts a duration (``30m``, ``24h``, ``7d``) relative to ``now`` (UTC now by default) or
    an ISO date / datetime; a naive datetime is taken as UTC.

    Raises:
        ValueError: when ``text`` is neither form.
    """
    reference = now if now is not None else datetime.now(UTC)
    cleaned = text.strip()
    match = _DURATION.fullmatch(cleaned)
    try:
        if match:
            amount, unit = int(match.group(1)), match.group(2)
            delta = {"m": timedelta(minutes=1), "h": timedelta(hours=1), "d": timedelta(days=1)}
            return reference - delta[unit] * amount
        parsed = datetime.fromisoformat(cleaned)
    except (ValueError, OverflowError) as exc:  # OverflowError: a duration reaching before year 1
        raise ValueError(
            f"--since expects a duration like 30m, 24h, 7d or an ISO date/datetime, got {text!r}"
        ) from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def fetch_captures(
    store: RemoteStore,
    *,
    base: Path | None = None,
    suite: str | None = None,
    since: datetime | None = None,
    skip_ids: Collection[str] = frozenset(),
    workers: int = 8,
) -> FetchSummary:
    """Mirror missing toolsets and captures from ``store`` into ``<base>``.

    Toolsets are fetched first so a capture never lands locally before a sidecar that exists
    remotely. A capture is skipped when its file already exists, when its id is in
    ``skip_ids`` (the promoted ids -- so a capture ``capture clean`` removed is not pulled
    back), when ``since`` is set and the object is older, or when it is over
    :data:`MAX_OBJECT_BYTES`. Keys that do not match the SDK's layout are counted as
    malformed and ignored.

    Raises:
        RemoteStoreError: when listing or any download fails. Files already written stay, so a
            re-run resumes; nothing half-written is left behind.
    """
    summary = FetchSummary(store_uri=store.uri)
    skip = set(skip_ids)
    toolsets_dir = _absolute(toolsets_root(base))
    captures_dir = _absolute(captures_root(base))

    toolset_jobs: list[tuple[str, Path]] = []
    for info in _list(store, "toolsets/"):
        if info.key.endswith("/"):
            continue  # a folder marker (S3 console "Create folder", ADLS directory), not an object
        name = info.key.removeprefix("toolsets/")
        if "/" in name or not _TOOLSET_NAME.fullmatch(name):
            summary.skipped_malformed += 1
            continue
        dest = toolsets_dir / name
        if not _absolute(dest.parent).is_relative_to(toolsets_dir):
            summary.skipped_malformed += 1
            continue
        if dest.exists():
            continue  # content-addressed: present means correct; not worth a counter
        if info.size is not None and info.size > MAX_OBJECT_BYTES:
            summary.skipped_large += 1
            continue
        toolset_jobs.append((info.key, dest))

    capture_jobs: list[tuple[str, Path]] = []
    prefix = f"captures/{suite}/" if suite else "captures/"
    for info in _list(store, prefix):
        if info.key.endswith("/"):
            continue  # a folder marker, as above
        rel = info.key.removeprefix("captures/")
        suite_seg, sep, name = rel.partition("/")
        if (
            not sep
            or "/" in name
            or not _is_safe_suite_segment(suite_seg)
            or not _CAPTURE_NAME.fullmatch(name)
        ):
            summary.skipped_malformed += 1
            continue
        dest = captures_dir / suite_seg / name
        if not _absolute(dest.parent).is_relative_to(captures_dir):
            summary.skipped_malformed += 1
            continue
        if dest.exists():
            summary.skipped_existing += 1
            continue
        if name[: -len(".json")] in skip:
            summary.skipped_promoted += 1
            continue
        if since is not None and info.last_modified is not None and info.last_modified < since:
            summary.skipped_old += 1
            continue
        if info.size is not None and info.size > MAX_OBJECT_BYTES:
            summary.skipped_large += 1
            continue
        capture_jobs.append((info.key, dest))

    summary.toolsets, large = _download_all(store, toolset_jobs, workers)
    summary.skipped_large += large
    summary.captures, large = _download_all(store, capture_jobs, workers)
    summary.skipped_large += large
    return summary


def _absolute(path: Path) -> Path:
    """``path`` made absolute and normalized lexically (``..`` collapsed, drive applied).

    Belt and braces behind the key checks: a destination whose parent normalizes outside its
    root is refused. Lexical, so it works before the directory exists and costs no I/O.
    """
    return Path(os.path.abspath(path))


def _fetch_failure_hint(store: RemoteStore) -> str:
    """The hint for a failed list or download: the scheme's credential chain, then --offline.

    ``capture fetch`` has no ``--offline``, so the advice names the commands that do.
    """
    scheme, separator, _ = store.uri.partition("://")
    credentials = _CREDENTIALS_HINT.get(
        scheme if separator else "", "check credentials and network"
    )
    return (
        f"{credentials}, or run `capture sync` / `capture list` with --offline "
        "to use the captures already on disk"
    )


def _list(store: RemoteStore, prefix: str) -> list[ObjectInfo]:
    try:
        return list(store.list(prefix))
    except RemoteStoreError:
        raise
    except Exception as exc:
        raise RemoteStoreError(
            f"could not list {prefix!r} in {store.uri}: {type(exc).__name__}: {exc}",
            hint=_fetch_failure_hint(store),
        ) from exc


def _download_all(
    store: RemoteStore, jobs: list[tuple[str, Path]], workers: int
) -> tuple[int, int]:
    """Download every job; return ``(written, skipped_as_too_large)``."""
    if not jobs:
        return 0, 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
            results = list(pool.map(lambda job: _download(store, job[0], job[1]), jobs))
    except RemoteStoreError:
        raise
    except Exception as exc:
        raise RemoteStoreError(
            f"could not download from {store.uri}: {type(exc).__name__}: {exc}",
            hint=_fetch_failure_hint(store),
        ) from exc
    written = sum(1 for ok in results if ok)
    return written, len(results) - written


def _download(store: RemoteStore, key: str, dest: Path) -> bool:
    """Fetch ``key`` into ``dest`` atomically. ``False`` when the object is over the cap."""
    try:
        data = store.get(key)
    except Exception as exc:
        raise RemoteStoreError(
            f"could not download {key} from {store.uri}: {type(exc).__name__}: {exc}",
            hint=_fetch_failure_hint(store),
        ) from exc
    if len(data) > MAX_OBJECT_BYTES:
        return False
    try:
        _write_atomically(dest, data)
    except OSError as exc:
        # A local failure: the store answered, so credentials and network are not the problem.
        raise RemoteStoreError(
            f"could not write {dest}: {type(exc).__name__}: {exc}",
            hint=f"check free disk space and permissions on {dest.parent}",
        ) from exc
    return True


def _write_atomically(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` via a temp file beside it and :func:`os.replace`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, dest)
    finally:
        with suppress(OSError):
            os.remove(tmp_name)


__all__ = [
    "MAX_OBJECT_BYTES",
    "FetchSummary",
    "ObjectInfo",
    "RemoteStore",
    "RemoteStoreError",
    "RemoteStoreUnavailable",
    "fetch_captures",
    "open_store",
    "parse_since",
]
