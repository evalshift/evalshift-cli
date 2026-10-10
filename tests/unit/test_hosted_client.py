"""Tests for the hosted HTTP client error surfacing."""

from __future__ import annotations

import inspect
import json
from typing import Any

import httpx
import pytest

from evalshift_cli.hosted.client import HostedAccountSuspendedError, HostedClient, HostedHTTPError
from tests.conftest import FakeHostedClient


def _validation_details() -> dict[str, Any]:
    return {
        "errors": [
            {
                "loc": ["body", "manifest", "suite_name"],
                "msg": "Extra inputs are not permitted",
                "type": "extra_forbidden",
            },
            {
                "loc": ["body", "manifest", "examples", 3, "id"],
                "msg": "Field required",
                "type": "missing",
            },
        ]
    }


def test_http_error_str_renders_validation_details() -> None:
    exc = HostedHTTPError(
        422,
        "Request validation failed",
        code="validation_error",
        details=_validation_details(),
    )
    text = str(exc)
    assert "Request validation failed" in text
    assert "body.manifest.suite_name: Extra inputs are not permitted (extra_forbidden)" in text
    assert "body.manifest.examples.3.id: Field required (missing)" in text


def test_http_error_str_without_details_is_plain_message() -> None:
    exc = HostedHTTPError(404, "project not found", code="not_found")
    assert str(exc) == "project not found"
    assert exc.details is None


def test_http_error_preserves_details_attribute() -> None:
    details = _validation_details()
    exc = HostedHTTPError(422, "Request validation failed", details=details)
    assert exc.details == details


@pytest.mark.parametrize(
    "details",
    [
        "not a dict",
        {"errors": "not a list"},
        {"errors": [{"msg": 1, "loc": None}]},
        {"other": "shape"},
    ],
)
def test_http_error_str_tolerates_malformed_details(details: Any) -> None:
    exc = HostedHTTPError(422, "Request validation failed", details=details)
    assert str(exc).startswith("Request validation failed")


def test_http_error_str_renders_entry_without_type() -> None:
    exc = HostedHTTPError(
        422,
        "Request validation failed",
        details={"errors": [{"loc": ["body", "manifest"], "msg": "bad manifest"}]},
    )
    assert "body.manifest: bad manifest" in str(exc)


def _client_with_response(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> HostedClient:
    transport = httpx.MockTransport(lambda request: response)
    original_client = httpx.Client

    def patched_client(**kwargs: Any) -> httpx.Client:
        return original_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)
    return HostedClient(host="https://api.evalshift.dev", token="tok")


def _recording_client(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> tuple[HostedClient, list[httpx.Request]]:
    """A client whose transport records every request it was handed."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def patched_client(**kwargs: Any) -> httpx.Client:
        return original_client(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched_client)
    return HostedClient(host="https://api.test", token="es_x"), seen


def test_initiate_run_sends_size_bytes_at_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``size_bytes`` is request metadata; the manifest must not carry it."""
    client, seen = _recording_client(
        monkeypatch,
        httpx.Response(200, json={"run_id": "r_1", "view_url": "https://app.test/r_1"}),
    )

    client.initiate_run({"run_id": "r_1"}, size_bytes=4096)

    assert len(seen) == 1
    request = seen[0]
    assert request.url.path == "/runs"
    body = json.loads(request.content)
    assert body["size_bytes"] == 4096
    assert "size_bytes" not in body["manifest"]


def test_the_client_has_no_thresholds_parameter_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``thresholds`` is gone from the config, so nothing may still offer to send it.

    Checked on the signatures and not only on one payload: the key was
    optional on both calls, so a reintroduced parameter that no caller happens
    to pass would leave every payload assertion green while handing the field
    a way back onto the wire.
    """
    for method in (HostedClient.create_project, HostedClient.initiate_run):
        assert "thresholds" not in inspect.signature(method).parameters

    client, seen = _recording_client(
        monkeypatch,
        httpx.Response(200, json={"run_id": "r_1", "view_url": "https://app.test/r_1"}),
    )

    client.initiate_run({"run_id": "r_1"}, size_bytes=17)

    assert "thresholds" not in json.loads(seen[0].content)


def test_request_raises_error_with_details_from_422_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client_with_response(
        monkeypatch,
        httpx.Response(
            422,
            json={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": _validation_details(),
                }
            },
        ),
    )
    with pytest.raises(HostedHTTPError) as excinfo:
        client.initiate_run({"run_id": "r1"}, size_bytes=1024)
    exc = excinfo.value
    assert exc.status_code == 422
    assert exc.code == "validation_error"
    assert exc.details == _validation_details()
    assert "body.manifest.suite_name: Extra inputs are not permitted (extra_forbidden)" in str(exc)


def test_request_surfaces_the_402_envelope_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``details`` is the whole upgrade prompt — the CLI must not reshape it."""
    details = {
        "feature": "runs_per_month",
        "limit": 50,
        "used": 50,
        "tier": "free",
        "status": "active",
        "resets_at": "2026-08-01",
        "upgrade_url": "https://app.evalshift.dev/app/acme/settings/billing",
    }
    client = _client_with_response(
        monkeypatch,
        httpx.Response(
            402,
            json={
                "error": {
                    "code": "payment_required",
                    "message": "Monthly run limit reached on the Free plan.",
                    "details": details,
                }
            },
        ),
    )
    with pytest.raises(HostedHTTPError) as excinfo:
        client.initiate_run({"run_id": "r1"}, size_bytes=1024)
    exc = excinfo.value
    assert exc.status_code == 402
    assert exc.code == "payment_required"
    assert exc.details == details
    # No ``details.errors`` here, so the message stays exactly what the server said.
    assert str(exc) == "Monthly run limit reached on the Free plan."


def test_request_error_without_details_keeps_current_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client_with_response(
        monkeypatch,
        httpx.Response(
            404,
            json={"error": {"code": "not_found", "message": "project not found"}},
        ),
    )
    with pytest.raises(HostedHTTPError) as excinfo:
        client.me()
    assert str(excinfo.value) == "project not found"
    assert excinfo.value.details is None


_FINALIZE_PATH = "/runs/3f7a1c2e-8b40-4d61-9a2f-6c5e0d7b1a93/finalize"


def test_finalize_run_posts_the_host_relative_path_to_the_configured_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /runs`` returns a host-relative ``finalize_url``; it joins to the host."""
    client, seen = _recording_client(
        monkeypatch,
        httpx.Response(200, json={"id": "3f7a1c2e", "view_url": "https://app.test/r"}),
    )

    client.finalize_run(_FINALIZE_PATH)

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == f"https://api.test{_FINALIZE_PATH}"
    assert request.headers["Authorization"] == "Bearer es_x"


@pytest.mark.parametrize(
    "finalize_url",
    [
        f"https://evil.example{_FINALIZE_PATH}",
        f"http://evil.example{_FINALIZE_PATH}",
        f"ftp://evil.example{_FINALIZE_PATH}",
        f"//evil.example{_FINALIZE_PATH}",
        f".evil.example{_FINALIZE_PATH}",
        "runs/3f7a1c2e/finalize",
        "",
    ],
)
def test_finalize_run_rejects_anything_that_is_not_a_host_relative_path(
    monkeypatch: pytest.MonkeyPatch,
    finalize_url: str,
) -> None:
    """A finalize URL that could retarget the bearer token is a hard error.

    The token only ever reaches the configured host because the CLI refuses to
    treat a response field as an origin — no request is issued at all.
    """
    client, seen = _recording_client(
        monkeypatch,
        httpx.Response(200, json={"id": "3f7a1c2e"}),
    )

    with pytest.raises(HostedHTTPError, match="unusable finalize URL"):
        client.finalize_run(finalize_url)

    assert seen == []
    assert not any("authorization" in request.headers for request in seen)


@pytest.mark.parametrize(
    "path",
    [
        "/runs/1/finalize",
        # Host-relative but shaped like an origin: joined onto a host that
        # already carries a scheme, a leading ``//`` is just path, not an
        # authority. ``finalize_run`` rejects it anyway.
        "//evil.example/runs/1/finalize",
        "/runs/1/finalize?next=https://evil.example",
    ],
)
def test_request_joins_every_path_onto_the_configured_host(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    """``_request`` has one origin — the configured host — and no way past it.

    Its second argument is a host-relative path, never a URL: it is concatenated
    onto ``self.host``, so a bearer-carrying request always lands under the host
    prefix and no response field can ship the token to a host the CLI did not
    choose. The assertion is on that whole prefix, ``host + "/"``, and not on the
    parsed host: ``api.test`` is a *prefix* of ``api.test.evil.example``, so a
    host-shaped check would wave a real origin escape through.

    That the prefix cannot be broken depends on the path being host-relative,
    which is exactly what ``finalize_run`` enforces on the one value the server
    supplies — see
    :func:`test_finalize_run_rejects_anything_that_is_not_a_host_relative_path`.
    """
    client, seen = _recording_client(monkeypatch, httpx.Response(200, json={}))

    client._request("POST", path)

    assert len(seen) == 1
    assert str(seen[0].url).startswith(f"{client.host}/")


def test_fake_hosted_client_mirrors_hosted_client() -> None:
    """The push tests' fake is bound to this client's signatures, not to a memory of them.

    Around twenty push tests drive :class:`FakeHostedClient` instead of the real
    :class:`HostedClient`, and ``mypy.ini`` type-checks ``src/evalshift`` only —
    so nothing else notices when the two drift. Without this check a rename or a
    changed keyword on the real client leaves every one of those tests green
    while ``push_bundle`` can no longer call it, which is precisely how a
    finalize guard that rejected the server's real value once shipped
    suite-green.
    """
    mirrored = sorted(
        name
        for name, member in vars(FakeHostedClient).items()
        if not name.startswith("_") and inspect.isfunction(member)
    )
    assert mirrored, "the fake stands in for no method at all"

    for name in mirrored:
        real = getattr(HostedClient, name, None)
        assert real is not None, f"HostedClient has no {name}(); the fake is stale"
        assert inspect.signature(getattr(FakeHostedClient, name)) == inspect.signature(real), (
            f"FakeHostedClient.{name}() no longer matches HostedClient.{name}()"
        )


SUSPENDED_MESSAGE = "Your account has been suspended. Contact support@evalshift.dev."


def test_account_suspended_maps_to_its_own_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_response(
        monkeypatch,
        httpx.Response(
            403,
            json={
                "error": {
                    "code": "account_suspended",
                    "message": SUSPENDED_MESSAGE,
                    "details": {"support_email": "support@evalshift.dev"},
                }
            },
        ),
    )

    with pytest.raises(HostedAccountSuspendedError) as excinfo:
        client.me()

    assert isinstance(excinfo.value, HostedHTTPError)
    assert excinfo.value.status_code == 403
    assert str(excinfo.value) == SUSPENDED_MESSAGE


def test_other_403s_stay_plain_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_response(
        monkeypatch,
        httpx.Response(403, json={"error": {"code": "forbidden", "message": "nope"}}),
    )

    with pytest.raises(HostedHTTPError) as excinfo:
        client.me()

    assert not isinstance(excinfo.value, HostedAccountSuspendedError)
