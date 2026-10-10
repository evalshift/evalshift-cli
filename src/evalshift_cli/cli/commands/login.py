"""Implementation of ``evalshift login`` for hosted token auth."""

from __future__ import annotations

import socket
import time
import webbrowser
from typing import Annotated

import typer
from rich.console import Console

from evalshift_cli.hosted.client import (
    HostedAccountSuspendedError,
    HostedClient,
    HostedError,
    HostedHTTPError,
)
from evalshift_cli.hosted.credentials import (
    CredentialsError,
    is_insecure_host,
    load_credentials,
    resolve_host,
    save_credentials,
)


def login(
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help="Hosted API token beginning with es_. Paste a personal token, or a "
            "service account key when this shell is CI.",
        ),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option("--host", help="Hosted API base URL. Defaults to $EVALSHIFT_HOST."),
    ] = None,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Print the approval URL instead of opening a browser."),
    ] = False,
    timeout_seconds: Annotated[
        int,
        typer.Option("--timeout", help="Maximum seconds to wait for browser approval."),
    ] = 900,
) -> None:
    """Authenticate this CLI with hosted EvalShift.

    The browser flow issues a personal token: it belongs to you, carries whatever your
    membership allows, and stops working when that membership does. That is the right
    credential for a workstation. Running login again while that token still works
    keeps it rather than minting another; run ``evalshift logout`` first to switch
    accounts.

    It is the wrong credential for CI. A pipeline outlives the person who set it up, so
    mint a service account key instead - hosted web app, Settings then API tokens, under
    Service accounts - scope it to the permissions the job actually needs, and hand it to
    the runner as an encrypted secret rather than running this command there.
    """
    console = Console()
    if timeout_seconds <= 0:
        console.print("[red]✗[/red] --timeout must be greater than 0")
        raise typer.Exit(code=1)
    host = resolve_host(host)
    if is_insecure_host(host):
        console.print(
            "[yellow]![/yellow] host uses plain http against a non-local destination; "
            "your bearer token will transit in cleartext. Prefer https://."
        )
    if token is None:
        if _reuse_stored_credentials(host=host, console=console):
            return
        _login_with_browser_flow(
            host=host,
            no_browser=no_browser,
            timeout_seconds=timeout_seconds,
            console=console,
        )
        return

    _login_with_token(host=host, token=token, console=console)


def _login_with_token(*, host: str, token: str, console: Console) -> None:
    try:
        me = HostedClient(host=host, token=token).me()
        save_credentials(host, token)
    except (CredentialsError, HostedError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    email = me.get("email", "(unknown user)")
    console.print(f"[green]✓[/green] logged in as {email}")


def _reuse_stored_credentials(*, host: str, console: Console) -> bool:
    """Return True when a stored token for ``host`` still works, so no new token is minted.

    Every browser approval mints a fresh personal token on the server, and the CLI can
    only hold one at a time, so re-running ``login`` with a working credential would
    strand the previous token. A rejected token (revoked, or for another account) falls
    through to the browser flow. A suspended account is reported and stops here. Any other
    failure is reported as-is because the browser flow would hit the same wall.
    """
    try:
        stored = load_credentials()
    except CredentialsError:
        return False
    if stored is None or stored.host != host:
        return False
    try:
        me = HostedClient(host=host, token=stored.token).me()
    except HostedAccountSuspendedError as exc:
        # A 403 like the rejected-token case below, but a fresh sign-in cannot fix it.
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except HostedHTTPError as exc:
        if exc.status_code in (401, 403):
            console.print("[yellow]![/yellow] stored token is no longer valid; signing in again")
            return False
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except HostedError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc
    email = me.get("email", "(unknown user)")
    console.print(f"[green]✓[/green] already logged in as {email}")
    console.print("To switch accounts, run `evalshift logout` first, then `evalshift login`.")
    return True


def _login_with_browser_flow(
    *,
    host: str,
    no_browser: bool,
    timeout_seconds: int,
    console: Console,
) -> None:
    try:
        client = HostedClient(host=host)
        started = client.start_cli_device_login(client_name=_client_name())
        device_code = _required_string(started, "device_code")
        user_code = _required_string(started, "user_code")
        verification_url = _required_string(started, "verification_uri_complete")
        expires_in = _positive_int(started.get("expires_in"), default=900)
        interval = _positive_int(started.get("interval"), default=5)
    except (CredentialsError, HostedError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    wait_seconds = min(timeout_seconds, expires_in)
    deadline = time.monotonic() + wait_seconds
    opened = webbrowser.open(verification_url) if not no_browser else False
    if opened:
        console.print("Opening your browser to approve this device...")
    else:
        console.print("To approve, open this link in your browser:")
        console.print(f"  [link={verification_url}]{verification_url}[/link]")
    console.print(f"Confirm this code matches your browser: [bold]{user_code}[/bold]")
    console.print("Waiting for approval...")

    while time.monotonic() < deadline:
        try:
            polled = client.poll_cli_device_login(device_code=device_code)
        except HostedError as exc:
            console.print(f"[red]✗[/red] {exc}")
            raise typer.Exit(code=1) from exc
        status = polled.get("status")
        if status == "pending":
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            continue
        if status != "approved":
            console.print(f"[red]✗[/red] unexpected CLI login status: {status}")
            raise typer.Exit(code=1)
        access_token = _required_string(polled, "access_token")
        try:
            me = HostedClient(host=host, token=access_token).me()
            save_credentials(host, access_token)
        except (CredentialsError, HostedError) as exc:
            console.print(f"[red]✗[/red] {exc}")
            raise typer.Exit(code=1) from exc
        email = me.get("email", "(unknown user)")
        console.print(f"[green]✓[/green] logged in as {email}")
        return

    console.print("[red]✗[/red] CLI login timed out before browser approval")
    raise typer.Exit(code=1)


def _client_name() -> str:
    hostname = socket.gethostname().strip()
    return f"EvalShift CLI {hostname}" if hostname else "EvalShift CLI"


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CredentialsError(f"hosted API did not return {key}")
    return value


def _positive_int(value: object, *, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


__all__ = ["login"]
