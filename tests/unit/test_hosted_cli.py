"""Phase 5 hosted CLI tests."""

from __future__ import annotations

import gzip
import json
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from evalshift_cli.cli.main import app
from evalshift_cli.hosted.bundle import BUNDLE_FILENAME, BundleError, build_bundle
from evalshift_cli.hosted.client import HostedNetworkError
from evalshift_cli.hosted.credentials import (
    CredentialsError,
    load_credentials,
    resolve_credentials,
    resolve_host,
    save_credentials,
)
from evalshift_cli.hosted.push import (
    _TRANSIENT_STATUSES,
    HostedHTTPError,
    PushError,
    PushResult,
    _put_with_retries,
    push_bundle,
    push_local_run,
)
from tests.conftest import FakeHostedClient as _SharedFakeClient
from tests.conftest import write_completed_run as _write_completed_run
from tests.conftest import write_project_files as _write_project_files

runner = CliRunner()


def _read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)
    return data


def test_credentials_file_uses_0600_and_resolution_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "credentials"

    save_credentials("https://api.evalshift.test/", "es_secret", path=path)

    mode = stat.S_IMODE(path.stat().st_mode)
    stored = load_credentials(path=path)
    assert stored is not None
    assert mode == 0o600
    assert stored.host == "https://api.evalshift.test"
    assert stored.token == "es_secret"
    monkeypatch.setenv("EVALSHIFT_HOST", "https://env.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_env")
    assert resolve_credentials(path=path).token == "es_env"
    assert resolve_credentials(host="https://flag.test", token="es_flag", path=path).host == (
        "https://flag.test"
    )


def test_resolve_credentials_requires_token_when_none_available(tmp_path: Path) -> None:
    with pytest.raises(CredentialsError, match="evalshift login"):
        resolve_credentials(path=tmp_path / "missing")


def test_resolve_host_precedence_flag_then_env_then_default() -> None:
    env = {"EVALSHIFT_HOST": "https://env.evalshift.test"}
    assert resolve_host("https://flag.test/", env=env) == "https://flag.test"
    assert resolve_host(None, env=env) == "https://env.evalshift.test"
    assert resolve_host(None, env={}) == "https://api.evalshift.dev"


def test_login_help_distinguishes_personal_tokens_from_service_account_keys() -> None:
    result = runner.invoke(app, ["login", "--help"])

    assert result.exit_code == 0, result.output
    # Rich re-wraps help to the terminal width, so compare on normalized whitespace.
    help_text = " ".join(result.output.split())
    assert "personal token" in help_text
    assert "service account" in help_text
    assert "CI" in help_text


def test_login_writes_credentials_without_printing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))

    def fake_me(self: Any) -> dict[str, Any]:
        return {"email": "dev@example.com"}

    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient.me", fake_me)

    result = runner.invoke(
        app,
        ["login", "--token", "es_plaintext", "--host", "https://api.evalshift.test"],
    )

    assert result.exit_code == 0, result.output
    assert "dev@example.com" in result.output
    assert "es_plaintext" not in result.output
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.token == "es_plaintext"


def test_login_device_flow_opens_browser_polls_and_saves_credentials_without_printing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))
    opened: list[str] = []

    class FakeHostedClient:
        def __init__(self, *, host: str, token: str | None = None, timeout: float = 20.0) -> None:
            self.host = host.rstrip("/")
            self.token = token

        def start_cli_device_login(self, *, client_name: str) -> dict[str, Any]:
            assert client_name.startswith("EvalShift CLI")
            return {
                "device_code": "device-secret",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://api.evalshift.test/auth/cli/approve",
                "verification_uri_complete": (
                    "https://api.evalshift.test/auth/cli/approve?user_code=ABCD-EFGH"
                ),
                "expires_in": 900,
                "interval": 0,
            }

        def poll_cli_device_login(self, *, device_code: str) -> dict[str, Any]:
            assert device_code == "device-secret"
            return {
                "status": "approved",
                "access_token": "es_device_plaintext",
                "user": {"email": "dev@example.com"},
            }

        def me(self) -> dict[str, Any]:
            assert self.token == "es_device_plaintext"
            return {"email": "dev@example.com"}

    def fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", FakeHostedClient)
    monkeypatch.setattr("evalshift_cli.cli.commands.login.webbrowser.open", fake_open)

    result = runner.invoke(app, ["login", "--host", "https://api.evalshift.test"])

    assert result.exit_code == 0, result.output
    assert opened == ["https://api.evalshift.test/auth/cli/approve?user_code=ABCD-EFGH"]
    assert "Opening your browser" in result.output
    assert "ABCD-EFGH" in result.output
    assert "dev@example.com" in result.output
    assert "es_device_plaintext" not in result.output
    assert "https://api.evalshift.test" not in result.output
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.host == "https://api.evalshift.test"
    assert stored.token == "es_device_plaintext"


def test_login_honors_env_host_when_no_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")

    seen_hosts: list[str] = []

    class FakeHostedClient:
        def __init__(self, *, host: str, token: str | None = None, timeout: float = 20.0) -> None:
            seen_hosts.append(host)
            self.host = host.rstrip("/")
            self.token = token

        def me(self) -> dict[str, Any]:
            return {"email": "dev@example.com"}

    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", FakeHostedClient)

    result = runner.invoke(app, ["login", "--token", "es_plaintext"])

    assert result.exit_code == 0, result.output
    assert seen_hosts == ["https://api.evalshift.test"]
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.host == "https://api.evalshift.test"


def test_login_device_flow_no_browser_does_not_open_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))

    class FakeHostedClient:
        def __init__(self, *, host: str, token: str | None = None, timeout: float = 20.0) -> None:
            self.token = token

        def start_cli_device_login(self, *, client_name: str) -> dict[str, Any]:
            return {
                "device_code": "device-secret",
                "user_code": "WXYZ-1234",
                "verification_uri": "https://api.evalshift.test/auth/cli/approve",
                "verification_uri_complete": (
                    "https://api.evalshift.test/auth/cli/approve?user_code=WXYZ-1234"
                ),
                "expires_in": 900,
                "interval": 0,
            }

        def poll_cli_device_login(self, *, device_code: str) -> dict[str, Any]:
            return {"status": "approved", "access_token": "es_no_browser"}

        def me(self) -> dict[str, Any]:
            assert self.token == "es_no_browser"
            return {"email": "dev@example.com"}

    def fail_open(url: str) -> None:
        raise AssertionError(f"browser should not open {url}")

    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", FakeHostedClient)
    monkeypatch.setattr("evalshift_cli.cli.commands.login.webbrowser.open", fail_open)

    result = runner.invoke(
        app,
        ["login", "--host", "https://api.evalshift.test", "--no-browser"],
    )

    assert result.exit_code == 0, result.output
    assert "https://api.evalshift.test/auth/cli/approve?user_code=WXYZ-1234" in result.output
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.token == "es_no_browser"


def test_login_device_flow_denied_does_not_save_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))

    class FakeHostedClient:
        def __init__(self, *, host: str, token: str | None = None, timeout: float = 20.0) -> None:
            pass

        def start_cli_device_login(self, *, client_name: str) -> dict[str, Any]:
            return {
                "device_code": "device-secret",
                "user_code": "DENY-0001",
                "verification_uri": "https://api.evalshift.test/auth/cli/approve",
                "verification_uri_complete": (
                    "https://api.evalshift.test/auth/cli/approve?user_code=DENY-0001"
                ),
                "expires_in": 900,
                "interval": 0,
            }

        def poll_cli_device_login(self, *, device_code: str) -> dict[str, Any]:
            raise HostedHTTPError(400, "CLI login was denied")

    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", FakeHostedClient)
    monkeypatch.setattr("evalshift_cli.cli.commands.login.webbrowser.open", lambda _url: None)

    result = runner.invoke(app, ["login", "--host", "https://api.evalshift.test"])

    assert result.exit_code == 1
    assert "denied" in result.output
    assert load_credentials(path=credentials_path) is None


def _device_flow_client(
    *,
    me_for_stored: Any = None,
    minted_token: str = "es_device_plaintext",
) -> tuple[type, list[str]]:
    """Return a fake HostedClient plus a log of device-flow calls it received.

    ``me_for_stored`` is what ``me()`` does for the pre-existing token: a dict is
    returned, an exception instance is raised.
    """
    calls: list[str] = []

    class FakeHostedClient:
        def __init__(self, *, host: str, token: str | None = None, timeout: float = 20.0) -> None:
            self.host = host.rstrip("/")
            self.token = token

        def start_cli_device_login(self, *, client_name: str) -> dict[str, Any]:
            calls.append("start")
            return {
                "device_code": "device-secret",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://api.evalshift.test/auth/cli/approve",
                "verification_uri_complete": (
                    "https://api.evalshift.test/auth/cli/approve?user_code=ABCD-EFGH"
                ),
                "expires_in": 900,
                "interval": 0,
            }

        def poll_cli_device_login(self, *, device_code: str) -> dict[str, Any]:
            calls.append("poll")
            return {"status": "approved", "access_token": minted_token}

        def me(self) -> dict[str, Any]:
            if self.token == minted_token:
                return {"email": "dev@example.com"}
            if isinstance(me_for_stored, Exception):
                raise me_for_stored
            assert isinstance(me_for_stored, dict)
            return me_for_stored

    return FakeHostedClient, calls


def test_login_reuses_valid_stored_credentials_instead_of_minting_a_new_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))
    save_credentials("https://api.evalshift.test", "es_existing", path=credentials_path)
    fake_client, calls = _device_flow_client(me_for_stored={"email": "dev@example.com"})
    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", fake_client)
    monkeypatch.setattr(
        "evalshift_cli.cli.commands.login.webbrowser.open",
        lambda _url: pytest.fail("browser flow must not start when already logged in"),
    )

    result = runner.invoke(app, ["login", "--host", "https://api.evalshift.test"])

    assert result.exit_code == 0, result.output
    assert calls == []
    assert "already logged in as dev@example.com" in result.output
    assert "evalshift logout" in result.output
    assert "es_existing" not in result.output
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.token == "es_existing"


def test_login_reauthenticates_when_stored_token_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))
    save_credentials("https://api.evalshift.test", "es_revoked", path=credentials_path)
    fake_client, calls = _device_flow_client(
        me_for_stored=HostedHTTPError(401, "invalid or revoked token"),
    )
    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", fake_client)
    monkeypatch.setattr("evalshift_cli.cli.commands.login.webbrowser.open", lambda _url: True)

    result = runner.invoke(app, ["login", "--host", "https://api.evalshift.test"])

    assert result.exit_code == 0, result.output
    assert calls == ["start", "poll"]
    assert "no longer valid" in result.output
    assert "logged in as dev@example.com" in result.output
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.token == "es_device_plaintext"


def test_login_ignores_stored_credentials_for_a_different_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))
    save_credentials("https://other.evalshift.test", "es_other_host", path=credentials_path)
    fake_client, calls = _device_flow_client(
        me_for_stored=AssertionError("stored token for another host must not be probed"),
    )
    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", fake_client)
    monkeypatch.setattr("evalshift_cli.cli.commands.login.webbrowser.open", lambda _url: True)

    result = runner.invoke(app, ["login", "--host", "https://api.evalshift.test"])

    assert result.exit_code == 0, result.output
    assert calls == ["start", "poll"]
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.host == "https://api.evalshift.test"
    assert stored.token == "es_device_plaintext"


def test_login_fails_when_stored_token_cannot_be_verified_for_other_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))
    save_credentials("https://api.evalshift.test", "es_existing", path=credentials_path)
    fake_client, calls = _device_flow_client(
        me_for_stored=HostedNetworkError("could not reach https://api.evalshift.test"),
    )
    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient", fake_client)
    monkeypatch.setattr(
        "evalshift_cli.cli.commands.login.webbrowser.open",
        lambda _url: pytest.fail("browser flow must not start on a network failure"),
    )

    result = runner.invoke(app, ["login", "--host", "https://api.evalshift.test"])

    assert result.exit_code == 1
    assert calls == []
    assert "could not reach" in result.output
    stored = load_credentials(path=credentials_path)
    assert stored is not None
    assert stored.token == "es_existing"


def test_logout_removes_stored_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials_path = tmp_path / "credentials"
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(credentials_path))
    save_credentials("https://api.evalshift.test", "es_secret", path=credentials_path)

    result = runner.invoke(app, ["logout"])
    again = runner.invoke(app, ["logout"])

    assert result.exit_code == 0, result.output
    assert again.exit_code == 0, again.output
    assert not credentials_path.exists()


def test_whoami_prints_user_and_visible_org_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")

    def fake_me(self: Any) -> dict[str, Any]:
        return {"email": "dev@example.com"}

    def fake_orgs(self: Any) -> list[dict[str, Any]]:
        return [{"slug": "acme", "name": "Acme", "role": "owner"}]

    monkeypatch.setattr("evalshift_cli.cli.commands.whoami.HostedClient.me", fake_me)
    monkeypatch.setattr("evalshift_cli.cli.commands.whoami.HostedClient.orgs", fake_orgs)

    result = runner.invoke(app, ["whoami"])

    assert result.exit_code == 0, result.output
    assert "https://api.evalshift.test" in result.output
    assert "dev@example.com" in result.output
    assert "acme" in result.output
    assert "owner" in result.output


def test_bundle_builder_writes_valid_gzip_bundle_with_deterministic_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/model-swap")

    first = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
    )
    second = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
        output=tmp_path / "second_bundle.json.gz",
    )

    assert first.path.name == BUNDLE_FILENAME
    data = _read_bundle(first.path)
    again = _read_bundle(second.path)
    assert data["manifest"]["project_slug"] == "acme/model-migration"
    assert data["manifest"]["git_sha"] == "a" * 40
    assert data["manifest"]["branch"] == "feature/model-swap"
    assert first.size_bytes == first.path.stat().st_size
    assert data["manifest"]["eval_config_hash"] == again["manifest"]["eval_config_hash"]
    assert data["manifest"]["dataset_hash"] == again["manifest"]["dataset_hash"]
    # tmp_path/golden.jsonl: the conventional-layout rule slugs by parent dir name.
    assert data["manifest"]["suite_name"] == tmp_path.name
    assert data["examples"][0]["passed"] is True
    assert data["aggregate"]["total"] == 1


def test_bundle_manifest_prefers_explicit_suite_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    result = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        suite_name="checkout",
        runs_base=tmp_path / ".evalshift" / "runs",
    )

    data = _read_bundle(result.path)
    assert data["manifest"]["suite_name"] == "checkout"


def test_bundle_builder_reports_missing_artifacts(tmp_path: Path) -> None:
    """``report.html`` is deliberately absent from this list — see the bundle spec."""
    _write_project_files(tmp_path)
    run_dir = _write_completed_run(tmp_path)
    (run_dir / "analysis.json").unlink()

    with pytest.raises(BundleError, match=r"analysis\.json"):
        build_bundle(
            "r_20260516_abcdef",
            config_path=tmp_path / "evalshift.yaml",
            suite_path=tmp_path / "golden.jsonl",
            runs_base=tmp_path / ".evalshift" / "runs",
        )


def test_bundle_carries_analysis_and_decision_without_a_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_files(tmp_path)  # no migration_policy configured
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    result = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
    )
    data = _read_bundle(result.path)

    analysis = data["analysis"]
    assert "comparisons" in analysis
    assert "slice_aggregates" in analysis

    decision = data["decision"]
    assert decision["verdict"] == "inconclusive"
    assert decision["reason"] == "no migration_policy configured"
    assert decision["budget_results"] == []
    assert decision["overall"]["n_records"] == 1

    aggregate = data["aggregate"]
    assert aggregate["cost_usd_source"] == 0.01
    assert aggregate["cost_usd_target"] == 0.02
    assert aggregate["cost_usd_delta"] == pytest.approx(0.01)
    assert aggregate["latency_ms_source_p50"] == 100.0
    assert aggregate["latency_ms_target_p50"] == 110.0


def test_bundle_decision_uses_policy_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    # Enable a migration policy (all defaults) so the decision path runs budgets.
    (tmp_path / "evalshift.yaml").write_text(
        """
        version: 1
        project: acme/model-migration
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini/gemini-2.5-flash
          target_model: gemini/gemini-3.1-flash-lite-preview
        evaluators:
          structural:
            - type: length
              min_chars: 1
        migration_policy:
          max_cost_increase: 0.5
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    result = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
    )
    decision = _read_bundle(result.path)["decision"]

    assert decision["verdict"] in {"pass", "conditional_pass", "fail", "inconclusive"}
    assert decision["reason"] is None
    # Default policy evaluates all seven budgets at overall scope.
    assert len(decision["budget_results"]) == 7


# The server mints the run id; the bundle's r_... id is only an idempotency key.
_SERVER_RUN_ID = "3f7a1c2e-8b40-4d61-9a2f-6c5e0d7b1a93"
_SERVER_VIEW_URL = f"https://app.test/runs/{_SERVER_RUN_ID}"
_SERVER_FINALIZE_URL = f"/runs/{_SERVER_RUN_ID}/finalize"


def _fake_client(**kwargs: Any) -> _SharedFakeClient:
    """The shared fake, wired to the server id and view URL this module scripts.

    Imported under an alias because the ``login`` tests below define their own
    local ``FakeHostedClient`` for a different slice of the client.
    """
    kwargs.setdefault("finalize_response", {"id": _SERVER_RUN_ID, "view_url": _SERVER_VIEW_URL})
    return _SharedFakeClient(**kwargs)


def _build_bundle_for_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/model-swap")
    result = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
    )
    return result.path


def _push(bundle_path: Path, tmp_path: Path, **kwargs: Any) -> PushResult:
    """``push_bundle`` against this project's config and run directory.

    ``runs_base`` is always passed: it is where ``push_state.json`` goes, and
    defaulting it would put the checkpoint under the *current* directory rather
    than the temporary project.
    """
    return push_bundle(
        bundle_path,
        config_path=tmp_path / "evalshift.yaml",
        runs_base=tmp_path / ".evalshift" / "runs",
        **kwargs,
    )


def test_push_treats_available_run_as_idempotent_without_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[
            {
                "id": _SERVER_RUN_ID,
                "client_run_id": "r_20260516_abcdef",
                "status": "available",
                "upload_url": None,
                "finalize_url": _SERVER_FINALIZE_URL,
                "view_url": _SERVER_VIEW_URL,
            },
        ],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    result = _push(bundle_path, tmp_path, create_project=False)

    assert result.uploaded is False
    assert result.run_id == _SERVER_RUN_ID
    assert result.view_url == _SERVER_VIEW_URL


def test_push_sends_the_uploaded_files_size_on_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server sizes its pre-upload 413 guard on this, so it must be the file."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[
            {
                "id": _SERVER_RUN_ID,
                "client_run_id": "r_20260516_abcdef",
                "status": "available",
                "upload_url": None,
                "finalize_url": _SERVER_FINALIZE_URL,
                "view_url": _SERVER_VIEW_URL,
            },
        ],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    _push(bundle_path, tmp_path, create_project=False)

    assert fake.initiate_sizes == [bundle_path.stat().st_size]


def test_bundle_defaults_suite_path_from_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_bundle`` reads ``state.suite_path`` when ``--suite`` is omitted."""
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)  # records state.suite_path = <tmp>/golden.jsonl
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    result = build_bundle(
        "r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=None,  # defer to the run's recorded suite
        runs_base=tmp_path / ".evalshift" / "runs",
    )

    bundle = _read_bundle(result.path)
    assert bundle["dataset_snapshot"]["suite_path"] == str(tmp_path / "golden.jsonl")
    assert bundle["manifest"]["suite_name"] == tmp_path.name


def test_push_local_run_defaults_suite_path_from_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting ``--suite`` builds the bundle from the run's recorded suite."""
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)

    captured: dict[str, Any] = {}

    def fake_push_bundle(bundle_path: Path, **_: Any) -> PushResult:
        captured["bundle_path"] = bundle_path
        return PushResult(run_id="r_20260516_abcdef", view_url="https://app.test/x", uploaded=True)

    monkeypatch.setattr("evalshift_cli.hosted.push.push_bundle", fake_push_bundle)

    result = push_local_run(
        run_id="r_20260516_abcdef",
        config_path=tmp_path / "evalshift.yaml",
        suite_path=None,
        runs_base=tmp_path / ".evalshift" / "runs",
    )

    assert result.uploaded is True
    assert captured["bundle_path"].exists()


def test_push_command_omits_suite_defers_to_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``evalshift push <run>`` without ``--suite`` must not require a cwd suite."""
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)

    captured: dict[str, Any] = {}

    def fake_push_local_run(**kwargs: Any) -> PushResult:
        captured.update(kwargs)
        return PushResult(run_id="r_20260516_abcdef", view_url="https://app.test/x", uploaded=True)

    monkeypatch.setattr("evalshift_cli.cli.commands.push.push_local_run", fake_push_local_run)

    result = runner.invoke(
        app,
        [
            "push",
            "r_20260516_abcdef",
            "--config",
            str(tmp_path / "evalshift.yaml"),
            "--runs-base",
            str(tmp_path / ".evalshift" / "runs"),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured["suite_path"] is None  # deferred to the run's recorded suite


def test_push_command_resolves_named_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--suite-name`` resolves against ``suites:`` relative to the config directory."""
    (tmp_path / "evalshift.yaml").write_text(
        """
        version: 1
        project: acme/model-migration
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini/gemini-2.5-flash
          target_model: gemini/gemini-3.1-flash-lite-preview
        evaluators:
          structural:
            - type: length
              min_chars: 1
        suites:
          captured:
            source: captured
            path: captures/promoted.jsonl
        """,
        encoding="utf-8",
    )
    _write_completed_run(tmp_path)

    captured: dict[str, Any] = {}

    def fake_push_local_run(**kwargs: Any) -> PushResult:
        captured.update(kwargs)
        return PushResult(run_id="r_20260516_abcdef", view_url="https://app.test/x", uploaded=True)

    monkeypatch.setattr("evalshift_cli.cli.commands.push.push_local_run", fake_push_local_run)

    result = runner.invoke(
        app,
        [
            "push",
            "r_20260516_abcdef",
            "--config",
            str(tmp_path / "evalshift.yaml"),
            "--suite-name",
            "captured",
            "--runs-base",
            str(tmp_path / ".evalshift" / "runs"),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured["suite_path"] == tmp_path / "captures" / "promoted.jsonl"


def test_push_command_unknown_suite_name_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)

    def fail_push_local_run(**_: Any) -> PushResult:
        raise AssertionError("push_local_run should not be reached on a bad --suite-name")

    monkeypatch.setattr("evalshift_cli.cli.commands.push.push_local_run", fail_push_local_run)

    result = runner.invoke(
        app,
        [
            "push",
            "r_20260516_abcdef",
            "--config",
            str(tmp_path / "evalshift.yaml"),
            "--suite-name",
            "nope",
            "--runs-base",
            str(tmp_path / ".evalshift" / "runs"),
        ],
    )

    assert result.exit_code == 1
    assert "unknown --suite-name" in result.output


PIPED_CONSOLE_COLUMNS = "80"
"""Rich's fallback width whenever it cannot measure a terminal — i.e. any pipe.

Pinned via ``COLUMNS`` rather than trusted: under ``pytest -s`` rich can still
measure the developer's own terminal through fd 1, and a wide one would let a
wrapping bug pass unnoticed.
"""

LONG_VIEW_URL = (
    "https://www.evalshift.dev/app/acme-analytics/model-migration"
    "/runs/0f1d2c3b-4a59-4c8e-9b1f-2d3e4f5a6b7c"
)
"""A realistic hosted run URL: 102 characters, well past the 80-column fold.

Server-minted run ids are UUIDs, which is what pushed the common case over the
fold width in the first place.
"""


def test_push_prints_the_run_url_on_one_unbroken_line_when_stdout_is_a_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``push`` prints only the run URL, and CI reads it back off the pipe.

    Rich folds at 80 columns when stdout is not a terminal, so an unguarded
    ``console.print`` splits a 102-character URL mid-UUID into two lines that
    both still look like output — the first one even still starts with
    ``https://``. Consumers scanning for a URL line then take half of one and
    fail much later with a confusing 404.
    """
    assert len(LONG_VIEW_URL) > int(PIPED_CONSOLE_COLUMNS)
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("COLUMNS", PIPED_CONSOLE_COLUMNS)

    def fake_push_local_run(**_: Any) -> PushResult:
        return PushResult(run_id="r_20260516_abcdef", view_url=LONG_VIEW_URL, uploaded=True)

    monkeypatch.setattr("evalshift_cli.cli.commands.push.push_local_run", fake_push_local_run)

    result = runner.invoke(
        app,
        [
            "push",
            "r_20260516_abcdef",
            "--config",
            str(tmp_path / "evalshift.yaml"),
            "--runs-base",
            str(tmp_path / ".evalshift" / "runs"),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert LONG_VIEW_URL in [line.strip() for line in result.output.splitlines()]


def test_put_retries_transient_failures_only() -> None:
    attempts = 0

    def fake_put(url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        return httpx.Response(200)

    _put_with_retries(
        "https://storage.test/upload",
        b"bundle",
        put=fake_put,
        sleep=lambda _: None,
    )

    assert attempts == 3


def test_push_auto_creates_missing_project_when_org_is_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[
            {"raise_404": True},
            {
                "id": _SERVER_RUN_ID,
                "client_run_id": "r_20260516_abcdef",
                "status": "available",
                "upload_url": None,
                "finalize_url": _SERVER_FINALIZE_URL,
                "view_url": _SERVER_VIEW_URL,
            },
        ],
        projects=[],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    result = _push(bundle_path, tmp_path, create_project=True)

    assert result.project_created is True
    assert fake.created_project == {
        "org_slug": "acme",
        "slug": "model-migration",
        "name": "Model Migration",
    }


def test_push_does_not_auto_create_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(responses=[{"raise_404": True}], projects=[])
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError, match="project was not found"):
        _push(bundle_path, tmp_path, create_project=False)


def test_auto_create_failure_names_the_host_and_the_servers_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed org lookup must say WHICH server refused, and what it said.

    The host is resolved from a flag, an env var, the credentials file, or a
    built-in default, so "the org is inaccessible" is unanswerable on its own:
    the same slug is reachable on one host and absent on another, and the user
    cannot tell which one the push even talked to.
    """
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[{"raise_404": True}],
        list_projects_error=HostedHTTPError(403, "forbidden: org not visible to this token"),
        host="http://localhost:8080",
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "http://localhost:8080")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=True)

    text = str(excinfo.value)
    assert "acme/model-migration" in text
    assert "http://localhost:8080" in text
    assert "403" in text
    assert "forbidden: org not visible to this token" in text


def test_create_project_failure_names_the_host_and_the_servers_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same for the create call: the org was visible, the create was refused."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[{"raise_404": True}],
        projects=[],
        create_project_error=HostedHTTPError(403, "project:create is owner-only"),
        host="http://localhost:8080",
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "http://localhost:8080")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=True)

    text = str(excinfo.value)
    assert "acme/model-migration" in text
    assert "http://localhost:8080" in text
    assert "403" in text
    assert "project:create is owner-only" in text
    assert "owner access" in text


def _payment_required(
    message: str = "Monthly run limit reached on the Free plan (50 of 50 runs used).",
    *,
    upgrade_url: str | None = "https://app.test/app/acme/settings/billing",
) -> HostedHTTPError:
    """The server's 402 envelope, verbatim (``app/entitlements/service.py``)."""
    details: dict[str, Any] = {
        "feature": "runs_per_month",
        "limit": 50,
        "used": 50,
        "tier": "free",
        "status": "active",
        "resets_at": "2026-08-01",
    }
    if upgrade_url is not None:
        details["upgrade_url"] = upgrade_url
    return HostedHTTPError(402, message, code="payment_required", details=details)


def test_push_renders_the_upgrade_prompt_on_402(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must print the server's message and upgrade URL, and exit non-zero —
    never retry, and never claim the run succeeded."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(responses=[{"raise_error": _payment_required()}], projects=[])
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=True)

    text = str(excinfo.value)
    assert "this run needs a paid plan" in text
    assert "Monthly run limit reached on the Free plan (50 of 50 runs used)." in text
    assert "Upgrade: https://app.test/app/acme/settings/billing" in text
    # A payment error is never retried, and never routed into project auto-creation.
    assert fake.initiate_calls == 1
    assert fake.created_project is None


def test_push_upgrade_prompt_omits_the_link_when_the_server_sends_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 402 without ``details.upgrade_url`` still renders the server's sentence."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[{"raise_error": _payment_required("Seat limit reached.", upgrade_url=None)}],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=False)

    text = str(excinfo.value)
    assert "this run needs a paid plan" in text
    assert "Seat limit reached." in text
    assert "Upgrade:" not in text


def test_push_renders_the_upgrade_prompt_when_finalize_returns_402(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Storage quota is charged at finalize, so the same prompt must render there."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[
            {
                "id": _SERVER_RUN_ID,
                "client_run_id": "r_20260516_abcdef",
                "status": "pending_upload",
                "upload_url": "https://storage.test/upload",
                "finalize_url": _SERVER_FINALIZE_URL,
                "view_url": _SERVER_VIEW_URL,
            },
        ],
        finalize_error=_payment_required("Storage limit reached on the Free plan."),
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)
    monkeypatch.setattr("evalshift_cli.hosted.push._put_with_retries", lambda *a, **k: None)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=False)

    text = str(excinfo.value)
    assert "this run needs a paid plan" in text
    assert "Storage limit reached on the Free plan." in text
    assert "Upgrade: https://app.test/app/acme/settings/billing" in text


def test_push_command_exits_non_zero_and_prints_the_upgrade_prompt_on_402(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through ``evalshift push``: one clear prompt, exit code 1, no traceback."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(responses=[{"raise_error": _payment_required()}])
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    result = runner.invoke(
        app,
        [
            "push",
            "--bundle",
            str(bundle_path),
            "--config",
            str(tmp_path / "evalshift.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert "needs a paid plan" in result.output
    assert "https://app.test/app/acme/settings/billing" in result.output
    assert "Traceback" not in result.output


_TRIAL_ENDED = (
    "The free trial for this organization ended on 2026-11-02. Subscribe to Pro to push "
    "runs and keep the CI gate — existing runs stay readable."
)
_RUN_LIMIT = "Monthly run limit reached on the Pro plan (4,000 of 4,000 runs used)."


def test_a_402_on_the_retry_after_auto_create_renders_the_upgrade_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /runs`` for a missing project answers 404 before any entitlement check, so a
    402 for being over the monthly run or concurrent-run limit, or for a subscription that
    stopped paying, arrives on the *retried* call after auto-create. That retry runs inside
    the first attempt's ``except`` clause, so its 402 used to skip every handler and reach
    the user as a traceback."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[{"raise_404": True}, {"raise_error": _payment_required(_RUN_LIMIT)}],
        projects=[],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=True)

    text = str(excinfo.value)
    assert "this run needs a paid plan" in text
    assert _RUN_LIMIT in text
    assert "Upgrade: https://app.test/app/acme/settings/billing" in text
    assert fake.created_project is not None
    assert fake.initiate_calls == 2


def test_any_other_error_on_the_retry_is_a_push_error_not_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[
            {"raise_404": True},
            {"raise_error": HostedHTTPError(500, "database is down", code="internal_error")},
        ],
        projects=[],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=True)

    assert "database is down" in str(excinfo.value)


def test_a_402_while_auto_creating_the_project_renders_the_upgrade_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v6: an organization whose trial has ended is refused at ``POST /orgs/{slug}/projects``
    with a 402. That is a plan problem, not the "needs owner access" hint the generic create
    failure prints."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[{"raise_404": True}],
        projects=[],
        create_project_error=_payment_required(_TRIAL_ENDED),
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=True)

    text = str(excinfo.value)
    assert "this run needs a paid plan" in text
    assert _TRIAL_ENDED in text
    assert "owner access" not in text


@pytest.mark.parametrize("failing_call", ["list_projects", "create_project"])
def test_a_network_error_while_auto_creating_the_project_is_a_push_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_call: str,
) -> None:
    """Auto-create runs inside the first attempt's ``except`` clause, so a connection
    failure there skips the sibling handlers unless it is mapped where it happens."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[{"raise_404": True}],
        projects=[],
        **{f"{failing_call}_error": HostedNetworkError("could not reach api.evalshift.test")},
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    with pytest.raises(PushError) as excinfo:
        _push(bundle_path, tmp_path, create_project=True)

    assert "could not reach api.evalshift.test" in str(excinfo.value)


def test_push_command_prints_the_upgrade_prompt_when_the_retry_after_auto_create_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: exit 1, the server's sentence, and no escaped exception."""
    bundle_path = _build_bundle_for_push(tmp_path, monkeypatch)
    fake = _fake_client(
        responses=[{"raise_404": True}, {"raise_error": _payment_required(_RUN_LIMIT)}],
        projects=[],
    )
    monkeypatch.setenv("EVALSHIFT_HOST", "https://api.evalshift.test")
    monkeypatch.setenv("EVALSHIFT_TOKEN", "es_secret")
    monkeypatch.setattr("evalshift_cli.hosted.push.HostedClient", lambda **_: fake)

    result = runner.invoke(
        app,
        ["push", "--bundle", str(bundle_path), "--config", str(tmp_path / "evalshift.yaml")],
    )

    # CliRunner catches an escaped exception into ``result.exception`` (exit 1, no printed
    # traceback), so the exit code alone cannot tell a handled refusal from a crash.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert "needs a paid plan" in result.output
    assert _RUN_LIMIT in result.output


def test_transient_retry_statuses_never_include_payment_required() -> None:
    """Guard: a future edit to the retry set must not start retrying a payment error."""
    assert 402 not in _TRANSIENT_STATUSES
    assert 429 in _TRANSIENT_STATUSES


def test_put_retries_rate_limited_uploads() -> None:
    attempts = 0

    def fake_put(url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return httpx.Response(429)
        return httpx.Response(200)

    _put_with_retries(
        "https://storage.test/upload",
        b"bundle",
        put=fake_put,
        sleep=lambda _: None,
    )

    assert attempts == 2


def test_put_retries_connection_errors_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def fake_put(url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("connection refused")

    with pytest.raises(PushError, match="upload failed after 3 attempts"):
        _put_with_retries(
            "https://storage.test/upload",
            b"bundle",
            put=fake_put,
            sleep=lambda _: None,
        )
    assert attempts == 3


def test_bundle_builder_rejects_invalid_bundle_via_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force a manifest field to a value the schema rejects and assert BundleError."""
    _write_project_files(tmp_path)
    _write_completed_run(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")  # not a 40-char hex sha
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/model-swap")

    with pytest.raises(BundleError):
        build_bundle(
            "r_20260516_abcdef",
            config_path=tmp_path / "evalshift.yaml",
            suite_path=tmp_path / "golden.jsonl",
            runs_base=tmp_path / ".evalshift" / "runs",
        )


def test_login_warns_on_plain_http_to_non_local_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALSHIFT_CREDENTIALS_PATH", str(tmp_path / "credentials"))

    def fake_me(self: Any) -> dict[str, Any]:
        return {"email": "dev@example.com"}

    monkeypatch.setattr("evalshift_cli.cli.commands.login.HostedClient.me", fake_me)

    result = runner.invoke(
        app,
        ["login", "--token", "es_plaintext", "--host", "http://prod.evalshift.example.com"],
    )

    assert result.exit_code == 0, result.output
    assert "cleartext" in result.output
    assert "es_plaintext" not in result.output
