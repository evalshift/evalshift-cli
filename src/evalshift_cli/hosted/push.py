"""Hosted push flow for EvalShift run bundles."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import ValidationError
from rich.console import Console

from evalshift_cli.cli.commands._suites import _BlockDumper
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.config.models import MigrationPolicy
from evalshift_cli.hosted.bundle import (
    BUNDLE_FILENAME,
    BundleError,
    build_bundle,
    load_bundle,
    validate_bundle,
)
from evalshift_cli.hosted.client import (
    HostedAccountSuspendedError,
    HostedClient,
    HostedError,
    HostedHTTPError,
    HostedNetworkError,
)
from evalshift_cli.hosted.credentials import CredentialsError, resolve_credentials
from evalshift_cli.runner.checkpoint import (
    PushCheckpoint,
    clear_push_checkpoint,
    read_push_checkpoint,
    run_dir_for,
    write_push_checkpoint,
)

_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
"""Upload statuses worth another attempt. 402 must never appear here: a payment error is a
decision, not a hiccup, and retrying it only burns quota the org does not have."""


_PAYMENT_REQUIRED = 402

_MB = 1024 * 1024

SOFT_LIMIT_BYTES = 50 * _MB
"""Compressed size at which the CLI warns. `BUNDLE_SPEC.md` §Size limits."""

HARD_LIMIT_BYTES = 100 * _MB
"""The server's default rejection threshold, quoted in the warning and enforced there."""

_MISSING_POLICY_WARNING = (
    "this run carries no migration policy; unless this project still has an old "
    "web-app policy, the hosted gate reports inconclusive and never blocks "
    "— add migration_policy to evalshift.yaml"
)
"""Said out loud because the alternative reads as approval: a run with no policy
uploads, renders and reports exactly like a gated one, and the pull request it
belongs to is then never blocked. Hedged because this prints before any network
call: a project that still carries a web-app policy is re-evaluated against that
one server-side, so the gate does still run there."""

_LEGACY_POLICY_HINT = (
    "this project has a policy configured in the web app; move it into evalshift.yaml:"
)
"""The other half of that: projects whose only policy was web-edited would
otherwise read the warning above as wrong. The yaml is the source of truth now,
so the web-app policy is shown as the block that puts it there."""


def _soft_limit_warning(size_bytes: int) -> str | None:
    """The sentence to print when a bundle crosses the soft limit, else ``None``.

    A warning and never a refusal. The hard limit is configurable server-side,
    so a CLI that enforced its own copy would start rejecting runs the hosted
    plan actually accepts the moment the two numbers drifted apart.
    """
    if size_bytes < SOFT_LIMIT_BYTES:
        return None
    return (
        f"bundle is {size_bytes / _MB:.1f} MB compressed, over the "
        f"{SOFT_LIMIT_BYTES / _MB:.0f} MB soft limit; the server's hard limit is "
        f"{HARD_LIMIT_BYTES / _MB:.0f} MB and it rejects anything larger."
    )


class PushError(Exception):
    """Raised when a bundle cannot be pushed to hosted EvalShift."""


def _upgrade_prompt(exc: HostedHTTPError) -> PushError:
    """Turn the server's 402 into the whole upgrade prompt.

    The CLI never inspects entitlements and never decides whether a plan covers a run — it
    repeats the server's sentence and the link the server built, and exits non-zero.
    """
    lines = ["EvalShift: this run needs a paid plan.", f"  {exc}"]
    details = exc.details
    if isinstance(details, dict):
        upgrade_url = details.get("upgrade_url")
        if isinstance(upgrade_url, str) and upgrade_url:
            lines.append(f"  Upgrade: {upgrade_url}")
    return PushError("\n".join(lines))


def _initiate_after_create(
    client: HostedClient, manifest: dict[str, Any], *, size_bytes: int
) -> dict[str, Any]:
    """The ``POST /runs`` retried after auto-creating the project, mapped like the first.

    It runs inside the first attempt's ``except`` clause, so its own errors would skip the
    sibling handlers and surface as a traceback. ``POST /runs`` answers 404 for a missing
    project before any entitlement check, so a 402 for the monthly run or concurrent-run
    limit, seat overage, or a subscription that stopped paying arrives on this retry, not
    on the first call.

    Args:
        client: The hosted client the first attempt used.
        manifest: The bundle manifest sent as the run's create payload.
        size_bytes: Size of the bundle file that will be uploaded.

    Returns:
        The server's ``POST /runs`` response.

    Raises:
        PushError: On any hosted error — the upgrade prompt for a 402, the server's
            message otherwise.
    """
    try:
        return client.initiate_run(manifest, size_bytes=size_bytes)
    except HostedHTTPError as exc:
        if exc.status_code == _PAYMENT_REQUIRED:
            raise _upgrade_prompt(exc) from exc
        raise PushError(str(exc)) from exc
    except (HostedNetworkError, HostedError) as exc:
        raise PushError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class PushResult:
    """Outcome of a hosted push.

    Attributes:
        run_id: The server-minted run id (a UUID). This is the run's only
            address; the CLI's ``r_...`` id lives on solely as the bundle's
            ``manifest.run_id`` idempotency key.
        view_url: The run URL the server built, never one assembled here.
        uploaded: True when this push moved the run from pending to available,
            whether it uploaded the bytes itself or finished an interrupted push
            that had already uploaded them. False when the run was already
            available and there was nothing to do.
        project_created: True when the push auto-created the project first.
    """

    run_id: str
    view_url: str
    uploaded: bool
    project_created: bool = False


def push_local_run(
    *,
    run_id: str,
    config_path: Path,
    suite_path: Path | None = None,
    suite_name: str | None = None,
    runs_base: Path,
    project: str | None = None,
    host: str | None = None,
    token: str | None = None,
    create_project: bool = True,
    console: Console | None = None,
) -> PushResult:
    """Build a missing local bundle if needed, then push it."""
    run_dir = run_dir_for(run_id, runs_base)
    bundle_path = run_dir / BUNDLE_FILENAME
    if not bundle_path.exists():
        build_bundle(
            run_id,
            config_path=config_path,
            suite_path=suite_path,
            suite_name=suite_name,
            runs_base=runs_base,
            project=project,
        )
    return push_bundle(
        bundle_path,
        config_path=config_path,
        project=project,
        host=host,
        token=token,
        create_project=create_project,
        console=console,
        runs_base=runs_base,
    )


def push_bundle(
    bundle_path: Path,
    *,
    config_path: Path | None = None,
    project: str | None = None,
    host: str | None = None,
    token: str | None = None,
    create_project: bool = True,
    console: Console | None = None,
    runs_base: Path | None = None,
) -> PushResult:
    """Push a prebuilt bundle through signed upload and finalize.

    ``runs_base`` locates the run directory the resume checkpoint is written to.
    It is deliberately not derived from ``bundle_path``: ``--bundle`` may point
    anywhere, and generated state belongs under ``.evalshift/``.
    """
    try:
        credentials = resolve_credentials(host=host, token=token)
    except CredentialsError as exc:
        raise PushError(str(exc)) from exc
    client = HostedClient(host=credentials.host, token=credentials.token)
    # Read and validate before anything reaches the network. ``--bundle`` may
    # point at a file this CLI never wrote — an older build, another tool, a
    # hand-edit — and finalize would reject it only after the whole upload.
    try:
        bundle = load_bundle(bundle_path)
        validate_bundle(bundle)
    except BundleError as exc:
        raise PushError(str(exc)) from exc
    manifest = _manifest(bundle)
    # The size of the file actually uploaded, measured on disk rather than
    # carried inside the bundle it describes.
    upload_size_bytes = bundle_path.stat().st_size
    _warn_soft_limit(console, upload_size_bytes)
    # Both of these are facts about the file on disk, so they are said before
    # the network is touched — and, deliberately, before the resume path below
    # returns: a run finalized from a checkpoint lands on the server just as
    # ungated as one uploaded here.
    _warn_missing_policy(console, bundle)
    if project is not None and project != manifest["project_slug"]:
        raise PushError(
            f"--project {project!r} does not match bundle project {manifest['project_slug']!r}; "
            "rebuild the bundle with the desired project",
        )
    client_run_id = str(manifest["run_id"])
    project_slug = str(manifest["project_slug"])
    checkpoint_dir = run_dir_for(client_run_id, runs_base)
    bundle_sha256 = _digest(bundle_path)
    resumed = _resume_push(
        client,
        checkpoint_dir,
        client_run_id=client_run_id,
        project_slug=project_slug,
        bundle_sha256=bundle_sha256,
    )
    if resumed is not None:
        return resumed
    try:
        response = client.initiate_run(manifest, size_bytes=upload_size_bytes)
        project_created = False
    except HostedHTTPError as exc:
        if exc.status_code == _PAYMENT_REQUIRED:
            raise _upgrade_prompt(exc) from exc
        if exc.status_code != 404 or not create_project:
            if exc.status_code == 404:
                raise PushError(
                    "project was not found and auto-creation is disabled or not permitted",
                ) from exc
            raise PushError(str(exc)) from exc
        _auto_create_project(client, project_slug=str(manifest["project_slug"]))
        project_created = True
        response = _initiate_after_create(client, manifest, size_bytes=upload_size_bytes)
    except (HostedNetworkError, HostedError) as exc:
        raise PushError(str(exc)) from exc

    # Above the existing-run return below, so a re-push of an available run
    # still prints it exactly once.
    _print_legacy_policy_hint(console, config_path, response)
    server_run_id = _require_str_field(response, "id", "hosted API did not return a run id")
    view_url = str(response.get("view_url") or "")
    upload_url = response.get("upload_url")
    if upload_url is None:
        if not view_url:
            raise PushError("hosted API did not return a view_url for the existing run")
        return PushResult(
            run_id=server_run_id, view_url=view_url, uploaded=False, project_created=project_created
        )
    if not isinstance(upload_url, str) or not upload_url:
        raise PushError("hosted API returned an invalid upload_url")
    finalize_url = _require_str_field(
        response,
        "finalize_url",
        "hosted API did not return a finalize_url",
    )

    data = bundle_path.read_bytes()
    _put_with_retries(upload_url, data)
    # The bytes are in storage but the run is still invisible: from here until
    # finalize answers, a crash would otherwise lose the server's id and cost a
    # duplicate upload on the next attempt.
    _remember_push(
        checkpoint_dir,
        PushCheckpoint(
            client_run_id=client_run_id,
            server_run_id=server_run_id,
            project_slug=project_slug,
            finalize_url=finalize_url,
            view_url=view_url,
            bundle_sha256=bundle_sha256,
        ),
    )
    final_view_url = _finalize(
        client,
        finalize_url,
        fallback_view_url=view_url,
        checkpoint_dir=checkpoint_dir,
    )
    return PushResult(
        run_id=server_run_id,
        view_url=final_view_url,
        uploaded=True,
        project_created=project_created,
    )


def _resume_push(
    client: HostedClient,
    checkpoint_dir: Path,
    *,
    client_run_id: str,
    project_slug: str,
    bundle_sha256: str,
) -> PushResult | None:
    """Finish a push whose bundle was uploaded but never finalized, else ``None``.

    The checkpoint only exists between the upload and the finalize response, so
    finding one means the server already minted an id for this bundle and
    already holds its bytes. Re-POSTing would just ask for that same id back.
    """
    checkpoint = read_push_checkpoint(checkpoint_dir)
    if checkpoint is None:
        return None
    if not _describes_this_push(
        checkpoint,
        client_run_id=client_run_id,
        project_slug=project_slug,
        bundle_sha256=bundle_sha256,
    ):
        _forget_push(checkpoint_dir)
        return None
    view_url = _finalize(
        client,
        checkpoint.finalize_url,
        fallback_view_url=checkpoint.view_url,
        checkpoint_dir=checkpoint_dir,
    )
    return PushResult(run_id=checkpoint.server_run_id, view_url=view_url, uploaded=True)


def _finalize(
    client: HostedClient,
    finalize_url: str,
    *,
    fallback_view_url: str,
    checkpoint_dir: Path,
) -> str:
    """POST the server's finalize URL and return the run's view URL.

    The checkpoint is dropped either way. A finalize that answered has nothing
    left to resume, and one that failed must not strand every later push on the
    same dead URL — the next push starts over from ``POST /runs``, which is
    idempotent on the bundle's ``client_run_id``.
    """
    try:
        finalized = client.finalize_run(finalize_url)
    except HostedHTTPError as exc:
        _forget_push(checkpoint_dir)
        if exc.status_code == _PAYMENT_REQUIRED:
            raise _upgrade_prompt(exc) from exc
        raise PushError(f"finalize failed: {exc}") from exc
    except HostedError as exc:
        _forget_push(checkpoint_dir)
        raise PushError(f"finalize failed: {exc}") from exc
    _forget_push(checkpoint_dir)
    view_url = str(finalized.get("view_url") or fallback_view_url)
    if not view_url:
        raise PushError("hosted API did not return a view_url after finalize")
    return view_url


def _describes_this_push(
    checkpoint: PushCheckpoint,
    *,
    client_run_id: str,
    project_slug: str,
    bundle_sha256: str,
) -> bool:
    """Whether the checkpoint still describes the bundle about to be pushed.

    Three ways it can stop describing it, all of which mean *start the push
    over* rather than *finalize what the server already holds*:

    * another bundle left it behind — finalizing it would target a run this push
      knows nothing about;
    * the local bundle was rebuilt during the crash window (``analyze``/``report``
      re-run, or ``bundle`` after a config edit) — finalizing would publish the
      *old* bytes and print a URL for content no longer on disk. The server
      cannot catch this: ``manifest.run_id`` is unchanged;
    * the stored id and the stored finalize path disagree — ``push_state.json``
      is plain JSON on disk, and a hand-edited one would otherwise report one
      run's id while publishing another run's bytes.
    """
    return (
        checkpoint.client_run_id == client_run_id
        and checkpoint.project_slug == project_slug
        and checkpoint.bundle_sha256 == bundle_sha256
        and checkpoint.server_run_id in checkpoint.finalize_url
    )


def _digest(bundle_path: Path) -> str:
    """Hex SHA-256 of the bundle file's bytes, streamed rather than buffered."""
    digest = hashlib.sha256()
    with bundle_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_str_field(response: dict[str, Any], key: str, message: str) -> str:
    """Read a required string off the create response — no fallback, no shim."""
    value = response.get(key)
    if not isinstance(value, str) or not value:
        raise PushError(message)
    return value


def _remember_push(checkpoint_dir: Path, checkpoint: PushCheckpoint) -> None:
    """Persist the resume hint. A read-only directory must not fail a live push."""
    try:
        write_push_checkpoint(checkpoint_dir, checkpoint)
    except OSError:
        return


def _forget_push(checkpoint_dir: Path) -> None:
    """Drop the resume hint, best effort."""
    try:
        clear_push_checkpoint(checkpoint_dir)
    except OSError:
        return


def _put_with_retries(
    upload_url: str,
    data: bytes,
    *,
    put: Callable[..., httpx.Response] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 3,
) -> None:
    """PUT bundle bytes, retrying only transient upload failures."""
    putter = put or _httpx_put
    delay = 0.25
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = putter(
                upload_url,
                content=data,
                headers={"Content-Type": "application/gzip"},
            )
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            last_error = exc
            if attempt == max_attempts:
                raise PushError(f"upload failed after {attempt} attempts: {exc}") from exc
            sleep(delay)
            delay *= 2
            continue
        if 200 <= response.status_code < 300:
            return
        if response.status_code in _TRANSIENT_STATUSES and attempt < max_attempts:
            sleep(delay)
            delay *= 2
            continue
        raise PushError(f"upload failed with HTTP {response.status_code}")
    if last_error is not None:
        raise PushError(f"upload failed: {last_error}") from last_error


def _httpx_put(upload_url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
    with httpx.Client(timeout=60.0) as client:
        return client.put(upload_url, content=content, headers=headers)


def _manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict):
        raise PushError("bundle is missing a manifest object")
    return manifest


def _auto_create_project(client: HostedClient, *, project_slug: str) -> None:
    try:
        org_slug, project = project_slug.split("/", 1)
    except ValueError as exc:
        raise PushError(f"invalid project slug {project_slug!r}; expected org/project") from exc
    try:
        projects = client.list_projects(org_slug)
    except HostedAccountSuspendedError as exc:
        # The server's sentence already says what to do; an access hint would mislead.
        raise PushError(str(exc)) from exc
    except HostedHTTPError as exc:
        raise PushError(
            f"cannot auto-create {project_slug!r} at {client.host}: "
            f"{_server_said(exc)}. The org is inaccessible or this token lacks "
            f"permission — check `evalshift whoami` reports this host and an org "
            f"named {org_slug!r}",
        ) from exc
    except (HostedNetworkError, HostedError) as exc:
        raise PushError(str(exc)) from exc
    if any(item.get("slug") == project for item in projects):
        return
    try:
        client.create_project(org_slug, slug=project, name=_name_from_slug(project))
    except HostedAccountSuspendedError as exc:
        # The server's sentence already says what to do; an access hint would mislead.
        raise PushError(str(exc)) from exc
    except HostedHTTPError as exc:
        if exc.status_code == _PAYMENT_REQUIRED:
            # An expired org is read-only: creating a project is refused with the same 402 a
            # push gets, and it deserves the same prompt rather than an access-permission hint.
            raise _upgrade_prompt(exc) from exc
        raise PushError(
            f"cannot auto-create {project_slug!r} at {client.host}: "
            f"{_server_said(exc)}. Creating a project needs owner access to "
            f"{org_slug!r}; a scoped service-account key cannot do it — create the "
            f"project by hand, or push with an owner token",
        ) from exc
    except (HostedNetworkError, HostedError) as exc:
        raise PushError(str(exc)) from exc


def _server_said(exc: HostedHTTPError) -> str:
    """Render a hosted error as ``HTTP <status>: <what the server wrote>``.

    Both auto-create failures used to discard ``exc`` entirely and assert a
    cause ("the org is inaccessible or this token lacks permission"). That is a
    guess: the CLI cannot know, and the one fact that settles it -- the status
    and sentence the server returned -- was the part being thrown away. Pair it
    with the host, since host resolution has four sources (flag, env,
    credentials file, built-in default) and the wrong one produces this same
    failure against a server the user never meant to talk to.
    """
    return f"HTTP {exc.status_code}: {exc}"


def _name_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("-") if part) or slug


def _warn_soft_limit(console: Console | None, size_bytes: int) -> None:
    """Print the soft-limit warning, if this bundle earns one."""
    warning = _soft_limit_warning(size_bytes)
    if console is None or warning is None:
        return
    console.print(f"[yellow]![/yellow] {warning}")


def _warn_missing_policy(console: Console | None, bundle: dict[str, Any]) -> None:
    """Warn when the bundle carries no resolved ``migration_policy``.

    ``decision.policy`` is the only thing the hosted gate has to check a pull
    request against. Without it the run still uploads and still renders, the
    gate reports ``inconclusive`` unless the project still has an old web-app
    policy to fall back on, and nothing blocks the merge — a silence that is
    indistinguishable from a passing gate unless the CLI says so here.
    """
    if console is None:
        return
    decision = bundle.get("decision")
    policy = decision.get("policy") if isinstance(decision, dict) else None
    if policy is not None:
        return
    console.print(f"[yellow]![/yellow] {_MISSING_POLICY_WARNING}")


def _print_legacy_policy_hint(
    console: Console | None,
    config_path: Path | None,
    response: dict[str, Any],
) -> None:
    """Show a web-app policy as the ``evalshift.yaml`` block that adopts it.

    Only for projects that have no ``migration_policy`` in their yaml: once the
    yaml has one it is the source of truth, and the server's copy is history
    nobody needs to act on.

    Everything about ``legacy_project_policy`` is read defensively. It is a
    field of a *response*, so an older server omits it, a newer one may change
    it, and neither is a reason to fail a push whose bundle is already on its
    way to storage.
    """
    if console is None or config_path is None:
        return
    legacy = response.get("legacy_project_policy")
    if not isinstance(legacy, dict):
        return
    try:
        config = load_config(config_path)
    except ConfigError:
        # A config that cannot be read cannot say whether it already has a
        # policy, and guessing here would push the wrong half of the advice.
        return
    if config.migration_policy is not None:
        return
    block = _policy_yaml_block(legacy)
    if block is None:
        return
    console.print(f"[yellow]![/yellow] {_LEGACY_POLICY_HINT}")
    # No markup and no highlighting: this block is meant to be copied into a
    # file verbatim, not rendered.
    console.print(block, markup=False, highlight=False)


def _policy_yaml_block(policy: dict[str, Any]) -> str | None:
    """Render a server-side policy as pasteable YAML, or ``None`` if it cannot be.

    Validated through :class:`MigrationPolicy` first, so what is printed is by
    construction something ``evalshift.yaml`` will load — a server policy the
    CLI's model rejects is a block that would break the file it is pasted into.

    Only the keys the server actually sent survive (``exclude_unset``). A
    web-app policy has six of the ten budgets; writing out the other four
    would pin today's CLI defaults into the user's file as if they had chosen
    them, and they are meant to move with the CLI.
    """
    try:
        parsed = MigrationPolicy.model_validate(policy)
    except ValidationError:
        return None
    fields = parsed.model_dump(mode="json", exclude_unset=True)
    if not fields:
        return None
    # The settings ``_suites.render_suites_yaml`` writes the managed suites
    # region with, so a pasted block matches the rest of the file.
    return yaml.dump(
        {"migration_policy": fields},
        Dumper=_BlockDumper,
        sort_keys=False,
        default_flow_style=False,
        indent=2,
        allow_unicode=True,
        width=10_000,
    ).rstrip("\n")


__all__ = [
    "HARD_LIMIT_BYTES",
    "SOFT_LIMIT_BYTES",
    "HostedHTTPError",
    "PushError",
    "PushResult",
    "_put_with_retries",
    "push_bundle",
    "push_local_run",
]
