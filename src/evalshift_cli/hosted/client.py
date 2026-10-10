"""Small synchronous HTTP client for the hosted backend."""

from __future__ import annotations

from typing import Any

import httpx

from evalshift_cli.hosted.credentials import normalize_host


class HostedError(Exception):
    """Base error for hosted API calls."""


class HostedNetworkError(HostedError):
    """Raised when the hosted API cannot be reached."""


class HostedHTTPError(HostedError):
    """Raised when the hosted API returns a non-2xx response."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str | None = None,
        details: Any = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.details = details
        rendered = _render_detail_errors(details)
        if rendered:
            message = message + "\n" + "\n".join(f"  - {line}" for line in rendered)
        super().__init__(message)


ACCOUNT_SUSPENDED = "account_suspended"


class HostedAccountSuspendedError(HostedHTTPError):
    """The account behind the token is suspended (HTTP 403, code ``account_suspended``).

    The message is the server's sentence and names who to contact, so callers print it as-is.
    It is not an expired token: signing in again cannot help.
    """


class HostedClient:
    """Typed-ish wrapper around the hosted API endpoints used by the CLI."""

    def __init__(self, *, host: str, token: str | None = None, timeout: float = 20.0) -> None:
        self.host = normalize_host(host)
        self.token = token
        self.timeout = timeout

    def start_cli_device_login(self, *, client_name: str) -> dict[str, Any]:
        data = self._request(
            "POST",
            "/auth/cli/device/start",
            json={"client_name": client_name},
        )
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected CLI login start response")
        return data

    def poll_cli_device_login(self, *, device_code: str) -> dict[str, Any]:
        data = self._request(
            "POST",
            "/auth/cli/device/poll",
            json={"device_code": device_code},
        )
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected CLI login poll response")
        return data

    def me(self) -> dict[str, Any]:
        data = self._request("GET", "/me")
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected /me response")
        return data

    def orgs(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/orgs")
        if not isinstance(data, list):
            raise HostedHTTPError(502, "unexpected /orgs response")
        return [item for item in data if isinstance(item, dict)]

    def list_projects(self, org_slug: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/orgs/{org_slug}/projects")
        if not isinstance(data, list):
            raise HostedHTTPError(502, "unexpected project list response")
        return [item for item in data if isinstance(item, dict)]

    def create_project(
        self,
        org_slug: str,
        *,
        slug: str,
        name: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "slug": slug}
        data = self._request("POST", f"/orgs/{org_slug}/projects", json=payload)
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected project create response")
        return data

    def initiate_run(
        self,
        manifest: dict[str, Any],
        *,
        size_bytes: int,
    ) -> dict[str, Any]:
        """Create the run and get a signed upload URL back.

        Args:
            manifest: The bundle's manifest block.
            size_bytes: Compressed size of the bundle file about to be
                uploaded. Request metadata rather than bundle content — a
                field inside the payload cannot report that payload's own
                compressed length. Guards the server's pre-upload 413.

        Keyword-only after ``manifest`` so no existing positional call can
        pass something else where the size belongs.
        """
        payload: dict[str, Any] = {"manifest": manifest, "size_bytes": size_bytes}
        data = self._request("POST", "/runs", json=payload)
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected run create response")
        return data

    def finalize_run(self, finalize_url: str) -> dict[str, Any]:
        """Finalize the run at the path the create response handed back.

        The server owns run addressing: it mints the run id and returns the path
        carrying it. Rebuilding ``/runs/{id}/finalize`` from a base plus an id
        would re-derive an address the CLI does not own.

        It owns the path only, never the origin. Anything but a host-relative
        path is rejected, so the bearer token can only ever be sent to the
        configured host — a response field cannot redirect it elsewhere.

        Args:
            finalize_url: Host-relative ``finalize_url`` from ``POST /runs``,
                e.g. ``/runs/{id}/finalize``.
        """
        if not finalize_url.startswith("/") or finalize_url.startswith("//"):
            raise HostedHTTPError(
                502,
                "hosted API returned an unusable finalize URL "
                f"(expected a host-relative path): {finalize_url!r}",
            )
        data = self._request("POST", finalize_url)
        if not isinstance(data, dict):
            raise HostedHTTPError(502, "unexpected finalize response")
        return data

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Call the hosted API. ``path`` is a host-relative path joined to the
        configured host — there is no way to reach another origin through here."""
        headers = dict(kwargs.pop("headers", {}))
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        headers.setdefault("Accept", "application/json")
        url = f"{self.host}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, headers=headers, **kwargs)
        except httpx.RequestError as exc:
            raise HostedNetworkError(
                f"failed to reach hosted EvalShift at {self.host}: {exc}"
            ) from exc
        if response.status_code >= 400:
            code, message, details = _extract_error(response)
            error_type = (
                HostedAccountSuspendedError if code == ACCOUNT_SUSPENDED else HostedHTTPError
            )
            raise error_type(response.status_code, message, code=code, details=details)
        if not response.content:
            return None
        return response.json()


def _extract_error(response: httpx.Response) -> tuple[str | None, str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return None, response.text or f"hosted API returned HTTP {response.status_code}", None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), str) else None
            message = error.get("message") if isinstance(error.get("message"), str) else None
            details = error.get("details")
            return code, message or f"hosted API returned HTTP {response.status_code}", details
        detail = payload.get("detail")
        if isinstance(detail, str):
            return None, detail, None
    return None, f"hosted API returned HTTP {response.status_code}", None


def _render_detail_errors(details: Any) -> list[str]:
    """Render ``error.details.errors`` entries as human-readable ``loc: msg`` lines."""
    if not isinstance(details, dict):
        return []
    errors = details.get("errors")
    if not isinstance(errors, list):
        return []
    lines: list[str] = []
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        msg = entry.get("msg")
        if not isinstance(msg, str):
            continue
        loc = entry.get("loc")
        if isinstance(loc, list) and loc:
            location = ".".join(str(part) for part in loc)
            line = f"{location}: {msg}"
        else:
            line = msg
        error_type = entry.get("type")
        if isinstance(error_type, str):
            line = f"{line} ({error_type})"
        lines.append(line)
    return lines


__all__ = [
    "HostedClient",
    "HostedError",
    "HostedHTTPError",
    "HostedNetworkError",
]
