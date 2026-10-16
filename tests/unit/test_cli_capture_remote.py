"""``capture fetch`` and the automatic fetch step in ``capture sync`` / ``capture list``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalshift_cli.captures.remote import RemoteStore, RemoteStoreUnavailable
from evalshift_cli.captures.store_uri import StoreURI
from evalshift_cli.cli.commands import capture as capture_module
from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config
from evalshift_cli.cli.main import app
from evalshift_cli.suite.loader import load_jsonl
from tests.unit.fake_remote_store import FakeRemoteStore, populated_store

runner = CliRunner()


def _invoke(args: list[str], base: Path) -> Any:
    return runner.invoke(app, ["capture", *args, "--base", str(base)])


def _write_config(path: Path, *, store: str | None = "s3://acme-evals/support-agent") -> Path:
    text = render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
    if store is not None:
        text += f"\ncaptures:\n  store: {store}\n"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> FakeRemoteStore:
    """Install a populated fake as the store every URI opens to."""
    store = populated_store("cap_1", "cap_2")
    opened: list[StoreURI] = []

    def _open(parsed: StoreURI) -> RemoteStore:
        opened.append(parsed)
        return store

    monkeypatch.setattr(capture_module, "open_store", _open)
    store.opened = opened  # type: ignore[attr-defined]
    return store


# --- fetch ---------------------------------------------------------------------------------


def test_fetch_mirrors_store_into_base(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["fetch", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "fetched 2 capture(s) and 1 toolset(s) from s3://fake/prefix" in result.stdout
    assert (tmp_path / "captures" / "alpha" / "cap_1.json").exists()
    assert remote.opened[0].bucket == "acme-evals"  # type: ignore[attr-defined]


def test_fetch_without_store_configured_errors(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml", store=None)
    result = _invoke(["fetch", "--config", str(config)], tmp_path)
    assert result.exit_code == 1
    assert "no captures.store configured" in result.stdout
    assert remote.gets == []


def test_fetch_without_config_file_errors(tmp_path: Path, remote: FakeRemoteStore) -> None:
    result = _invoke(["fetch", "--config", str(tmp_path / "missing.yaml")], tmp_path)
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_fetch_invalid_config_prints_config_error(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml", store="ftp://nope/x")
    result = _invoke(["fetch", "--config", str(config)], tmp_path)
    assert result.exit_code == 1
    assert "Invalid config" in result.stdout


def test_fetch_since_and_suite_are_passed_through(
    tmp_path: Path, remote: FakeRemoteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _fake_fetch(store: RemoteStore, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return capture_module.FetchSummary(store_uri=store.uri)

    monkeypatch.setattr(capture_module, "fetch_captures", _fake_fetch)
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(
        ["fetch", "--suite", "alpha", "--since", "24h", "--config", str(config)], tmp_path
    )
    assert result.exit_code == 0, result.stdout
    assert seen["suite"] == "alpha"
    assert seen["since"] is not None
    assert seen["base"] == tmp_path


def test_fetch_undecodable_config_exits_1_without_traceback(
    tmp_path: Path, remote: FakeRemoteStore
) -> None:
    config = tmp_path / "evalshift[x].yaml"
    config.write_bytes(b"captures:\n  store: s3://b/\xff\xfeSECRET\n")
    result = _invoke(["fetch", "--config", str(config)], tmp_path)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"could not read {config}" in result.output.replace("\n", "")
    assert "UnicodeDecodeError" in result.output
    assert "SECRET" not in result.output  # never the raw config text
    assert remote.gets == []


def test_fetch_unreadable_config_exits_1_without_traceback(
    tmp_path: Path, remote: FakeRemoteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _unreadable(path: Path) -> Any:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(capture_module, "load_config", _unreadable)
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["fetch", "--config", str(config)], tmp_path)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "could not read" in result.output and "PermissionError" in result.output


@pytest.mark.parametrize("since", ["yesterday", "99999999d"])
def test_fetch_rejects_bad_since(tmp_path: Path, remote: FakeRemoteStore, since: str) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["fetch", "--since", since, "--config", str(config)], tmp_path)
    assert result.exit_code == 2
    assert "--since expects" in result.output


# --- sync ----------------------------------------------------------------------------------


def test_sync_fetches_first_then_promotes(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "fetched 2 capture(s)" in result.stdout
    assert "promoted 2 capture(s)" in result.stdout
    golden = tmp_path / "suites" / "alpha" / "golden.jsonl"
    assert len(load_jsonl(golden).ids()) == 2


def test_sync_offline_skips_fetch(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["sync", "--offline", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "fetched" not in result.stdout
    assert remote.gets == []


def test_sync_without_store_is_unchanged(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml", store=None)
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "fetched" not in result.stdout
    assert remote.gets == []


def test_sync_since_without_store_errors(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml", store=None)
    result = _invoke(["sync", "--since", "24h", "--config", str(config)], tmp_path)
    assert result.exit_code == 1
    assert "--since needs captures.store" in result.stdout
    assert not (tmp_path / "suites").exists()


def test_sync_fetch_failure_stops_before_promotion(tmp_path: Path, remote: FakeRemoteStore) -> None:
    remote.fail_keys.add("captures/alpha/cap_2.json")
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 1
    assert "Remote capture store" in result.stdout
    assert "--offline" in result.stdout
    assert not (tmp_path / "suites").exists()


def test_sync_does_not_refetch_promoted_then_cleaned_captures(
    tmp_path: Path, remote: FakeRemoteStore
) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    assert _invoke(["sync", "--config", str(config)], tmp_path).exit_code == 0
    cleaned = _invoke(["clean", "--yes"], tmp_path)
    assert cleaned.exit_code == 0, cleaned.stdout
    assert not (tmp_path / "captures" / "alpha" / "cap_1.json").exists()
    remote.gets.clear()
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "fetched 0 capture(s)" in result.stdout
    assert "2 promoted" in result.stdout
    assert all(not key.startswith("captures/") for key in remote.gets)


def test_sync_with_unreadable_config_warns_and_continues_locally(
    tmp_path: Path, remote: FakeRemoteStore
) -> None:
    config = tmp_path / "evalshift.yaml"
    # Invalid (prompts needs one entry), but it asks for a store, so skipping it is news.
    config.write_text("prompts: []\ncaptures:\n  store: s3://b/p\n", encoding="utf-8")
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "skipping captures.store" in result.stdout
    assert remote.gets == []


@pytest.mark.parametrize("command", ["sync", "list"])
def test_invalid_config_without_captures_stays_silent(
    tmp_path: Path, remote: FakeRemoteStore, command: str
) -> None:
    config = tmp_path / "evalshift.yaml"
    config.write_text("prompts: []\n", encoding="utf-8")  # invalid, never mentions a store
    result = _invoke([command, "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.output
    assert "skipping captures.store" not in result.output
    assert "could not read" not in result.output
    assert remote.gets == []


@pytest.mark.parametrize("command", ["sync", "list"])
def test_invalid_init_config_with_captures_only_in_comments_stays_silent(
    tmp_path: Path, remote: FakeRemoteStore, command: str
) -> None:
    # `evalshift init` output mentions "captures" in comments; that is not a store request.
    # Its top-level `captures:` block is stripped so only the comments remain.
    config = tmp_path / "evalshift.yaml"
    text = render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
    config.write_text(text + "\ntypo_key: 1\n", encoding="utf-8")
    assert b"captures" in config.read_bytes()
    result = _invoke([command, "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.output
    assert "skipping captures.store" not in result.output
    assert "could not read" not in result.output
    assert remote.gets == []


@pytest.mark.parametrize("command", ["sync", "list"])
def test_invalid_init_config_with_top_level_captures_warns(
    tmp_path: Path, remote: FakeRemoteStore, command: str
) -> None:
    config = tmp_path / "evalshift.yaml"
    config.write_text(
        render_minimal_config(profile="model-upgrade")
        + "\ntypo_key: 1\ncaptures:\n  store: s3://b/p\n",
        encoding="utf-8",
    )
    result = _invoke([command, "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.output
    assert "skipping captures.store" in result.output
    assert remote.gets == []


def test_list_with_undecodable_config_stays_local_without_traceback(
    tmp_path: Path, remote: FakeRemoteStore
) -> None:
    config = tmp_path / "evalshift.yaml"
    config.write_bytes(b"prompts:\n  - \xff\xfe\x80 not utf-8\n")
    result = _invoke(["list", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "no captures found" in result.stdout
    assert "could not read" not in result.output
    assert remote.gets == []


def test_bracketed_store_uri_prints_literally(tmp_path: Path, remote: FakeRemoteStore) -> None:
    remote.uri = "s3://acme/team[/x]"
    config = _write_config(tmp_path / "evalshift.yaml")
    fetched = _invoke(["fetch", "--config", str(config)], tmp_path)
    assert fetched.exit_code == 0, fetched.output
    assert "from s3://acme/team[/x]" in fetched.stdout
    remote.uri = "s3://acme/[bold]team"
    synced = _invoke(["sync", "--config", str(config)], tmp_path)
    assert synced.exit_code == 0, synced.output
    assert "from s3://acme/[bold]team" in synced.stdout


# --- list ----------------------------------------------------------------------------------


def test_list_fetches_first_when_store_configured(tmp_path: Path, remote: FakeRemoteStore) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["list", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "fetched 2 capture(s)" in result.stdout
    assert "cap_1" in result.stdout and "cap_2" in result.stdout


def test_list_offline_and_without_config_stay_local(
    tmp_path: Path, remote: FakeRemoteStore
) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    offline = _invoke(["list", "--offline", "--config", str(config)], tmp_path)
    assert offline.exit_code == 0 and "no captures found" in offline.stdout
    missing = _invoke(["list", "--config", str(tmp_path / "nope.yaml")], tmp_path)
    assert missing.exit_code == 0 and "no captures found" in missing.stdout
    assert remote.gets == []


def test_list_json_output_is_not_polluted_by_fetch_line(
    tmp_path: Path, remote: FakeRemoteStore
) -> None:
    import json

    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke(["list", "--json", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)  # the whole of stdout is the JSON payload
    assert {r["capture_id"] for r in rows} == {"cap_1", "cap_2"}
    assert "fetched 2 capture(s)" in result.stderr


# --- missing client extra -------------------------------------------------------------------


@pytest.fixture
def missing_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every store URI fail to open as if its pip extra were not installed."""

    def _open(parsed: StoreURI) -> RemoteStore:
        raise RemoteStoreUnavailable(
            "s3:// captures store needs boto3, which is not installed",
            hint="install it: pip install boto3",
        )

    monkeypatch.setattr(capture_module, "open_store", _open)


@pytest.mark.parametrize("command", ["fetch", "sync", "list"])
def test_missing_library_exits_1_naming_the_package(
    tmp_path: Path, missing_library: None, command: str
) -> None:
    config = _write_config(tmp_path / "evalshift.yaml")
    result = _invoke([command, "--config", str(config)], tmp_path)
    assert result.exit_code == 1
    assert "Remote capture store" in result.stdout
    assert "pip install boto3" in result.stdout
    assert not (tmp_path / "suites").exists()


# --- captures limits -----------------------------------------------------------------------

_NOW = datetime(2026, 6, 20, tzinfo=UTC)


def _with_max_age(config: Path) -> Path:
    config.write_text(config.read_text(encoding="utf-8") + "  max_age: 14d\n", encoding="utf-8")
    return config


def test_sync_fetches_only_inside_max_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: FakeRemoteStore
) -> None:
    remote.modified["captures/alpha/cap_1.json"] = datetime(2026, 5, 1, tzinfo=UTC)
    config = _with_max_age(_write_config(tmp_path / "evalshift.yaml"))
    monkeypatch.setattr(capture_module, "_now", lambda: _NOW)

    result = _invoke(["sync", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    flat = " ".join(result.stdout.split())
    assert "fetched 1 capture(s)" in flat
    assert "1 older than captures.max_age (14d)" in flat
    assert not (tmp_path / "captures" / "alpha" / "cap_1.json").exists()


def test_an_explicit_since_overrides_max_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: FakeRemoteStore
) -> None:
    remote.modified["captures/alpha/cap_1.json"] = datetime(2026, 5, 1, tzinfo=UTC)
    config = _with_max_age(_write_config(tmp_path / "evalshift.yaml"))
    monkeypatch.setattr(capture_module, "_now", lambda: _NOW)

    result = _invoke(["sync", "--since", "2026-04-01", "--config", str(config)], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "fetched 2 capture(s)" in result.stdout


def test_max_age_without_a_store_is_not_a_fetch_error(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "evalshift.yaml", store=None)
    config.write_text(
        config.read_text(encoding="utf-8") + "\ncaptures:\n  max_age: 14d\n", encoding="utf-8"
    )
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout


def test_offline_sync_says_it_ignores_limits_it_cannot_read(tmp_path: Path) -> None:
    config = tmp_path / "evalshift.yaml"
    config.write_text("captures:\n  max_age: 2w\n", encoding="utf-8")
    result = _invoke(["sync", "--offline", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.stdout
    assert "syncing without captures.max_age / captures.max_cases" in " ".join(
        result.stdout.split()
    )


def test_online_sync_says_it_ignores_limits_it_cannot_read(
    tmp_path: Path, remote: FakeRemoteStore
) -> None:
    # An `evalshift init` config with a top-level `captures:` block, made invalid by a typo:
    # the fetch step's warning names only captures.store, so the dropped limits get their own.
    config = tmp_path / "evalshift.yaml"
    config.write_text(
        render_minimal_config(profile="model-upgrade") + "\ntypo_key: 1\n", encoding="utf-8"
    )
    assert capture_module._mentions_captures(config)
    result = _invoke(["sync", "--config", str(config)], tmp_path)
    assert result.exit_code == 0, result.output
    flat = " ".join(result.stdout.split())
    assert (
        f"could not read {config} (1 schema problem found); "
        "syncing without captures.max_age / captures.max_cases."
    ) in flat
    assert "skipping captures.store" in flat
    assert remote.gets == []
