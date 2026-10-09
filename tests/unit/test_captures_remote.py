"""Incremental fetch from a RemoteStore into the local capture directory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evalshift_cli.captures import remote
from evalshift_cli.captures.remote import (
    MAX_OBJECT_BYTES,
    FetchSummary,
    RemoteStore,
    RemoteStoreError,
    _is_safe_suite_segment,
    fetch_captures,
    parse_since,
)
from tests.unit.fake_remote_store import (
    TOOLSET_HEX,
    TOOLSET_KEY,
    TOOLSET_PAYLOAD,
    FakeRemoteStore,
    capture_payload,
    populated_store,
)


def _local_captures(base: Path) -> set[str]:
    root = base / "captures"
    return {str(p.relative_to(root)) for p in root.rglob("*.json")} if root.exists() else set()


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeRemoteStore(), RemoteStore)


def test_fetch_downloads_toolsets_then_captures(tmp_path: Path) -> None:
    store = populated_store("cap_1", "cap_2")
    summary = fetch_captures(store, base=tmp_path)
    assert summary.captures == 2
    assert summary.toolsets == 1
    assert (tmp_path / "toolsets" / f"{TOOLSET_HEX}.json").read_bytes() == TOOLSET_PAYLOAD
    assert _local_captures(tmp_path) == {"alpha/cap_1.json", "alpha/cap_2.json"}
    assert store.gets[0] == TOOLSET_KEY  # sidecars land before the captures that need them


def test_fetch_skips_objects_already_local(tmp_path: Path) -> None:
    store = populated_store("cap_1", "cap_2")
    fetch_captures(store, base=tmp_path)
    store.gets.clear()
    summary = fetch_captures(store, base=tmp_path)
    assert summary.captures == 0
    assert summary.skipped_existing == 2
    assert store.gets == []  # nothing re-downloaded, toolset included


def test_fetch_skips_promoted_ids(tmp_path: Path) -> None:
    store = populated_store("cap_1", "cap_2")
    summary = fetch_captures(store, base=tmp_path, skip_ids={"cap_1"})
    assert summary.captures == 1
    assert summary.skipped_promoted == 1
    assert _local_captures(tmp_path) == {"alpha/cap_2.json"}


def test_fetch_filters_by_suite(tmp_path: Path) -> None:
    store = populated_store("cap_a")
    store.objects["captures/beta/cap_b.json"] = capture_payload("cap_b", "beta")
    summary = fetch_captures(store, base=tmp_path, suite="beta")
    assert summary.captures == 1
    assert _local_captures(tmp_path) == {"beta/cap_b.json"}


def test_fetch_since_filters_on_last_modified(tmp_path: Path) -> None:
    now = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    store = populated_store("cap_old", "cap_new", "cap_unknown")
    store.modified["captures/alpha/cap_old.json"] = now - timedelta(days=3)
    store.modified["captures/alpha/cap_new.json"] = now - timedelta(hours=1)
    store.modified["captures/alpha/cap_unknown.json"] = None
    summary = fetch_captures(store, base=tmp_path, since=now - timedelta(days=1))
    assert summary.skipped_old == 1
    # Unknown last_modified is never filtered out.
    assert _local_captures(tmp_path) == {"alpha/cap_new.json", "alpha/cap_unknown.json"}


def test_fetch_rejects_malformed_and_traversal_keys(tmp_path: Path) -> None:
    store = FakeRemoteStore(
        {
            "captures/../../escape.json": b"{}",
            "captures/alpha/evil.sh": b"#!",
            "captures/alpha/nested/cap_x.json": b"{}",
            "captures/alpha": b"{}",
            "captures/.hidden/cap_y.json": b"{}",
            "toolsets/not-a-hash.json": b"{}",
            "toolsets/sub/" + "ab" * 32 + ".json": b"{}",
            "README.md": b"ignored: not under captures/ or toolsets/",
        }
    )
    summary = fetch_captures(store, base=tmp_path)
    assert summary.captures == 0
    assert summary.toolsets == 0
    assert summary.skipped_malformed == 7
    assert not (tmp_path / "captures").exists()
    assert not (tmp_path / "toolsets").exists()
    assert not (tmp_path.parent / "escape.json").exists()
    assert store.gets == []


@pytest.mark.parametrize(
    "key",
    [
        "captures/../cap_z.json",  # would land in <base>/cap_z.json, outside captures/
        "captures/./cap_z.json",
        "captures//cap_z.json",  # empty suite: would land directly in captures/
        "captures/a\\b/cap_z.json",  # a path separator on Windows
        "captures/a..b/cap_z.json",
        "captures/alpha./cap_z.json",
        "captures/ alpha/cap_z.json",
    ],
)
def test_fetch_rejects_suite_segments_the_sdk_never_writes(tmp_path: Path, key: str) -> None:
    # The capture basename is valid here, so only the suite-segment check stands in the way.
    store = FakeRemoteStore({key: capture_payload("cap_z")})
    summary = fetch_captures(store, base=tmp_path)
    assert summary.skipped_malformed == 1
    assert summary.captures == 0
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    assert not (tmp_path.parent / "cap_z.json").exists()
    assert store.gets == []


def test_fetch_accepts_every_suite_name_the_sdk_writes(tmp_path: Path) -> None:
    # The SDK names suite directories with ``_safe_segment``, which keeps spaces, leading
    # underscores and non-ASCII letters; all of them must mirror to the same local directory.
    suites = ("Support Agent", "_internal", "ünïcode")
    store = FakeRemoteStore(
        {
            f"captures/{suite}/cap_{i}.json": capture_payload(
                f"cap_{i}", suite, model_input=f"input {i}"
            )
            for i, suite in enumerate(suites)
        }
    )
    summary = fetch_captures(store, base=tmp_path)
    assert summary.captures == 3
    assert summary.skipped_malformed == 0
    assert _local_captures(tmp_path) == {f"{suite}/cap_{i}.json" for i, suite in enumerate(suites)}


@pytest.mark.parametrize(
    ("segment", "safe"),
    [
        ("alpha", True),
        ("Support Agent", True),
        ("_internal", True),
        ("ünïcode", True),
        ("v1.2", True),
        ("", False),
        ("a/b", False),
        ("a\\b", False),
        ("a\x00b", False),
        ("..", False),
        ("a..b", False),
        (".hidden", False),
        ("alpha.", False),
        (" alpha", False),
        ("alpha ", False),
    ],
)
def test_is_safe_suite_segment(segment: str, safe: bool) -> None:
    assert _is_safe_suite_segment(segment) is safe


def test_fetch_skips_too_large_objects(tmp_path: Path) -> None:
    store = populated_store("cap_big", "cap_ok")
    store.sizes["captures/alpha/cap_big.json"] = MAX_OBJECT_BYTES + 1
    summary = fetch_captures(store, base=tmp_path)
    assert summary.skipped_large == 1
    assert _local_captures(tmp_path) == {"alpha/cap_ok.json"}


def test_fetch_checks_size_after_download_when_listing_has_none(tmp_path: Path) -> None:
    store = populated_store("cap_1")
    store.sizes["captures/alpha/cap_1.json"] = None
    store.objects["captures/alpha/cap_1.json"] = b"x" * (MAX_OBJECT_BYTES + 1)
    summary = fetch_captures(store, base=tmp_path)
    assert summary.skipped_large == 1
    assert _local_captures(tmp_path) == set()


def test_fetch_failure_leaves_no_temp_files(tmp_path: Path) -> None:
    store = populated_store("cap_1", "cap_2")
    store.fail_keys.add("captures/alpha/cap_2.json")
    with pytest.raises(RemoteStoreError) as info:
        fetch_captures(store, base=tmp_path, workers=1)
    assert "cap_2.json" in info.value.summary
    assert info.value.hint is not None and "--offline" in info.value.hint
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix == ".tmp"]
    assert leftovers == []
    # Already-fetched files stay: a re-run resumes.
    assert (tmp_path / "toolsets" / f"{TOOLSET_HEX}.json").exists()


def test_fetch_write_failure_leaves_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The get succeeds and the temp file is written; the rename into place then fails.
    def failing_replace(src: str, dst: str) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(remote.os, "replace", failing_replace)
    store = populated_store("cap_1")
    with pytest.raises(RemoteStoreError) as info:
        fetch_captures(store, base=tmp_path, workers=1)
    assert "simulated rename failure" in info.value.summary
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_summary_describe() -> None:
    full = FetchSummary(
        store_uri="s3://fake/prefix", captures=2, toolsets=1, skipped_existing=3, skipped_promoted=1
    )
    assert full.describe() == (
        "fetched 2 capture(s) and 1 toolset(s) from s3://fake/prefix "
        "(skipped 3 already local, 1 promoted)"
    )
    assert FetchSummary(store_uri="gs://b").describe() == (
        "fetched 0 capture(s) and 0 toolset(s) from gs://b"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30m", timedelta(minutes=30)),
        ("24h", timedelta(hours=24)),
        ("7d", timedelta(days=7)),
        (" 2d ", timedelta(days=2)),
    ],
)
def test_parse_since_durations(text: str, expected: timedelta) -> None:
    now = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    assert parse_since(text, now=now) == now - expected


def test_parse_since_iso_date_and_datetime() -> None:
    assert parse_since("2026-10-01") == datetime(2026, 10, 1, tzinfo=UTC)
    assert parse_since("2026-10-01T08:30:00+02:00") == datetime(2026, 10, 1, 6, 30, tzinfo=UTC)
    assert parse_since("2026-10-01T08:30:00").tzinfo is UTC


@pytest.mark.parametrize("text", ["yesterday", "24", "h", "7w", ""])
def test_parse_since_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError, match="--since expects"):
        parse_since(text)
