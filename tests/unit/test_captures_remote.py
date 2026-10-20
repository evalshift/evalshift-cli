"""Incremental fetch from a RemoteStore into the local capture directory."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evalshift_cli.captures import remote
from evalshift_cli.captures.remote import (
    MAX_OBJECT_BYTES,
    FetchSummary,
    ObjectInfo,
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
        "captures/D:/cap_z.json",  # a drive-relative path on Windows
        "captures/a:b/cap_z.json",  # a drive or NTFS alternate data stream on Windows
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


def test_fetch_containment_check_rejects_escaping_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With the suite predicate disabled, the containment check alone must stop a key whose
    # destination resolves outside ``<base>/captures``.
    monkeypatch.setattr(remote, "_is_safe_suite_segment", lambda segment: True)
    base = tmp_path / "base"
    store = FakeRemoteStore({"captures/../cap_z.json": capture_payload("cap_z")})
    summary = fetch_captures(store, base=base)
    assert summary.skipped_malformed == 1
    assert summary.captures == 0
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
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
        ("D:", False),
        ("a:b", False),
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


def test_fetch_counts_keys_with_a_trailing_newline_as_malformed(tmp_path: Path) -> None:
    store = FakeRemoteStore(
        {
            "captures/alpha/cap_1.json\n": capture_payload("cap_1"),
            f"toolsets/{TOOLSET_HEX}.json\n": TOOLSET_PAYLOAD,
        }
    )
    summary = fetch_captures(store, base=tmp_path)
    assert summary.skipped_malformed == 2
    assert summary.captures == summary.toolsets == 0
    assert list(tmp_path.rglob("*")) == []
    assert store.gets == []


def test_fetch_skips_folder_marker_keys_silently(tmp_path: Path) -> None:
    # S3 console "Create folder" and ADLS directory entries list as zero-byte "<dir>/" keys.
    store = populated_store("cap_1")
    store.objects.update(
        {"captures/": b"", "captures/alpha/": b"", "toolsets/": b"", "captures/beta/": b""}
    )
    summary = fetch_captures(store, base=tmp_path)
    assert summary.skipped_malformed == 0
    assert summary.captures == 1 and summary.toolsets == 1
    assert _local_captures(tmp_path) == {"alpha/cap_1.json"}
    assert not any(key.endswith("/") for key in store.gets)


class _ListFailingStore(FakeRemoteStore):
    def list(self, prefix: str) -> Iterator[ObjectInfo]:
        raise ConnectionError("simulated listing failure")


@pytest.mark.parametrize(
    ("uri", "credentials"),
    [
        ("s3://b/p", "check AWS credentials (aws sso login / AWS_PROFILE / the CI OIDC role)"),
        (
            "gs://b/p",
            "check Google credentials (gcloud auth application-default login / Workload Identity)",
        ),
        ("az://a/c/p", "check Azure credentials (az login / managed identity)"),
        ("mem://x", "check credentials"),
    ],
)
@pytest.mark.parametrize("failure", ["list", "get"])
def test_fetch_failure_hint_names_the_scheme_credentials(
    tmp_path: Path, uri: str, credentials: str, failure: str
) -> None:
    if failure == "list":
        store: FakeRemoteStore = _ListFailingStore()
    else:
        store = populated_store("cap_1")
        store.fail_keys.add("captures/alpha/cap_1.json")
    store.uri = uri
    with pytest.raises(RemoteStoreError) as info:
        fetch_captures(store, base=tmp_path, workers=1)
    assert info.value.hint == (
        f"{credentials} and network, or run `capture sync` / `capture list` with --offline "
        "to use the captures already on disk"
    )


def test_fetch_local_write_failure_names_the_path_not_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(remote.tempfile, "mkstemp", failing_mkstemp)
    store = populated_store("cap_1")
    with pytest.raises(RemoteStoreError) as info:
        fetch_captures(store, base=tmp_path, workers=1)
    toolsets_dir = tmp_path / "toolsets"
    assert str(toolsets_dir / f"{TOOLSET_HEX}.json") in info.value.summary
    assert "No space left on device" in info.value.summary
    assert info.value.hint == f"check free disk space and permissions on {toolsets_dir}"
    assert "credentials" not in str(info.value)


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


@pytest.mark.parametrize("text", ["99999999d", "9999999999999h", "99999999999999999999m"])
def test_parse_since_rejects_out_of_range_durations(text: str) -> None:
    with pytest.raises(ValueError, match="--since expects"):
        parse_since(text)


def test_force_ids_bypass_the_promoted_skip_and_since(tmp_path: Path) -> None:
    now = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    store = populated_store("cap_forced", "cap_old", "cap_promoted")
    store.modified["captures/alpha/cap_forced.json"] = now - timedelta(days=30)
    store.modified["captures/alpha/cap_old.json"] = now - timedelta(days=30)
    summary = fetch_captures(
        store,
        base=tmp_path,
        skip_ids={"cap_forced", "cap_promoted"},
        force_ids={"cap_forced"},
        since=now - timedelta(days=1),
    )
    assert _local_captures(tmp_path) == {"alpha/cap_forced.json"}
    assert summary.skipped_promoted == 1  # cap_promoted
    assert summary.skipped_old == 1  # cap_old


def test_force_ids_still_skip_captures_already_local(tmp_path: Path) -> None:
    store = populated_store("cap_1")
    fetch_captures(store, base=tmp_path)
    store.gets.clear()
    summary = fetch_captures(store, base=tmp_path, force_ids={"cap_1"})
    assert summary.skipped_existing == 1
    assert store.gets == []


def test_skipped_old_names_the_window_it_was_measured_against() -> None:
    summary = FetchSummary(
        store_uri="s3://b/p", skipped_old=3, since_label="captures.max_age (14d)"
    )
    assert "3 older than captures.max_age (14d)" in summary.describe()


def test_skipped_old_defaults_to_the_since_flag() -> None:
    assert "1 older than --since" in FetchSummary(store_uri="s3://b/p", skipped_old=1).describe()


# --- per-suite budget (captures.max_cases) ---------------------------------------------------

_T = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _aged(store: FakeRemoteStore, *hours: tuple[str, float], suite: str = "alpha") -> None:
    """Stamp each capture's last_modified that many hours before ``_T``."""
    for cid, h in hours:
        store.modified[f"captures/{suite}/{cid}.json"] = _T - timedelta(hours=h)


def test_per_suite_limit_downloads_each_suites_newest_captures(tmp_path: Path) -> None:
    store = populated_store("cap_new", "cap_mid", "cap_old")
    _aged(store, ("cap_new", 1), ("cap_mid", 2), ("cap_old", 3))
    store.objects["captures/beta/cap_b.json"] = capture_payload("cap_b", "beta")

    summary = fetch_captures(store, base=tmp_path, per_suite_limit=2)

    assert _local_captures(tmp_path) == {
        "alpha/cap_new.json",
        "alpha/cap_mid.json",
        "beta/cap_b.json",
    }
    assert summary.skipped_budget == 1


def test_per_suite_limit_counts_local_and_promoted_captures_so_a_rerun_fetches_nothing(
    tmp_path: Path,
) -> None:
    store = populated_store("cap_new", "cap_mid", "cap_old")
    _aged(store, ("cap_new", 1), ("cap_mid", 2), ("cap_old", 3))
    fetch_captures(store, base=tmp_path, skip_ids={"cap_new"}, per_suite_limit=2)
    assert _local_captures(tmp_path) == {"alpha/cap_mid.json"}
    store.gets.clear()

    summary = fetch_captures(store, base=tmp_path, skip_ids={"cap_new"}, per_suite_limit=2)

    assert store.gets == []
    assert summary.skipped_budget == 1


def test_captures_outside_the_window_take_no_budget(tmp_path: Path) -> None:
    store = populated_store("cap_new", "cap_mid", "cap_old")
    _aged(store, ("cap_new", 1), ("cap_mid", 2), ("cap_old", 48))

    summary = fetch_captures(store, base=tmp_path, since=_T - timedelta(days=1), per_suite_limit=2)

    assert summary.captures == 2
    assert summary.skipped_old == 1
    assert summary.skipped_budget == 0


def test_forced_ids_are_fetched_beyond_the_budget(tmp_path: Path) -> None:
    store = populated_store("cap_new", "cap_old")
    _aged(store, ("cap_new", 1), ("cap_old", 2))

    fetch_captures(store, base=tmp_path, per_suite_limit=1, force_ids={"cap_old"})

    assert _local_captures(tmp_path) == {"alpha/cap_new.json", "alpha/cap_old.json"}


def test_an_unknown_last_modified_counts_as_oldest_for_the_budget(tmp_path: Path) -> None:
    store = populated_store("cap_known", "cap_unknown")
    _aged(store, ("cap_known", 100))
    store.modified["captures/alpha/cap_unknown.json"] = None

    fetch_captures(store, base=tmp_path, per_suite_limit=1)

    assert _local_captures(tmp_path) == {"alpha/cap_known.json"}


def test_equal_last_modified_breaks_ties_by_key(tmp_path: Path) -> None:
    store = populated_store("cap_b", "cap_a", "cap_c")

    fetch_captures(store, base=tmp_path, per_suite_limit=2)

    assert _local_captures(tmp_path) == {"alpha/cap_a.json", "alpha/cap_b.json"}


def test_skipped_budget_names_the_limit() -> None:
    summary = FetchSummary(store_uri="s3://b/p", skipped_budget=4, per_suite_limit=600)
    assert "4 beyond each suite's newest 600" in summary.describe()


def test_a_suite_filter_lists_the_directory_the_sdk_wrote(tmp_path: Path) -> None:
    store = populated_store("cap_1", suite="team_support")

    summary = fetch_captures(store, base=tmp_path, suite="team/support")

    assert summary.captures == 1


# --- conversation folders (evalshift-sdk >= 0.7.0) --------------------------------------------

#: The folder evalshift-sdk writes for conversation_id "c1".
CONV = "conv_d0f631ca1ddba8db"


def _turns(
    store: FakeRemoteStore, *turns: tuple[str, float], suite: str = "alpha", folder: str = CONV
) -> None:
    """One conversation's turns in ``folder``, each written that many hours before ``_T``."""
    for cid, hours in turns:
        key = f"captures/{suite}/{folder}/{cid}.json"
        store.objects[key] = capture_payload(
            cid, suite, model_input=f"turn {cid}", conversation_id="c1"
        )
        store.modified[key] = _T - timedelta(hours=hours)


def test_fetch_mirrors_a_conversation_folder(tmp_path: Path) -> None:
    store = populated_store()
    _turns(store, ("cap_t0", 2), ("cap_t1", 1))

    summary = fetch_captures(store, base=tmp_path)

    assert summary.captures == 2
    assert _local_captures(tmp_path) == {f"alpha/{CONV}/cap_t0.json", f"alpha/{CONV}/cap_t1.json"}


@pytest.mark.parametrize(
    "folder",
    ["nested", "conv_D0F631CA1DDBA8DB", "conv_d0f631ca", f"{CONV}/deeper"],
)
def test_fetch_counts_folders_the_sdk_never_writes_as_malformed(
    tmp_path: Path, folder: str
) -> None:
    store = populated_store()
    store.objects[f"captures/alpha/{folder}/cap_x.json"] = capture_payload("cap_x")

    summary = fetch_captures(store, base=tmp_path)

    assert summary.skipped_malformed == 1
    assert _local_captures(tmp_path) == set()


def test_a_new_turn_brings_its_promoted_turns_along_whatever_their_age(tmp_path: Path) -> None:
    # A fresh checkout: turns 0-1 are promoted (committed case files) and turn 0 is outside the
    # window, but turn 2 rebuilds its history from both.
    store = populated_store()
    _turns(store, ("cap_t0", 72), ("cap_t1", 2), ("cap_t2", 1))

    summary = fetch_captures(
        store, base=tmp_path, skip_ids={"cap_t0", "cap_t1"}, since=_T - timedelta(days=1)
    )

    assert _local_captures(tmp_path) == {
        f"alpha/{CONV}/{cid}.json" for cid in ("cap_t0", "cap_t1", "cap_t2")
    }
    assert summary.skipped_promoted == 0
    assert summary.skipped_old == 0


def test_the_budget_counts_a_conversations_promoted_turns_so_they_come_along(
    tmp_path: Path,
) -> None:
    # As above, under a budget: the promoted turn outside the window counts toward it too, so
    # the conversation fits whole and the new turn's history is not dropped as beyond budget.
    store = populated_store()
    _turns(store, ("cap_t0", 72), ("cap_t1", 2), ("cap_t2", 1))

    summary = fetch_captures(
        store,
        base=tmp_path,
        skip_ids={"cap_t0", "cap_t1"},
        since=_T - timedelta(days=1),
        per_suite_limit=3,
    )

    assert _local_captures(tmp_path) == {
        f"alpha/{CONV}/{cid}.json" for cid in ("cap_t0", "cap_t1", "cap_t2")
    }
    assert summary.skipped_budget == 0


def test_an_old_unpromoted_turn_stays_behind(tmp_path: Path) -> None:
    store = populated_store()
    _turns(store, ("cap_t0", 72), ("cap_t1", 1))

    summary = fetch_captures(store, base=tmp_path, since=_T - timedelta(days=1))

    assert _local_captures(tmp_path) == {f"alpha/{CONV}/cap_t1.json"}
    assert summary.skipped_old == 1


def test_the_budgets_leading_turns_stay_inside_the_window(tmp_path: Path) -> None:
    # A long conversation: three turns days old, three in the window, a budget of two.
    store = populated_store()
    _turns(
        store,
        ("cap_t0", 100),
        ("cap_t1", 90),
        ("cap_t2", 80),
        ("cap_t3", 3),
        ("cap_t4", 2),
        ("cap_t5", 1),
    )

    summary = fetch_captures(
        store, base=tmp_path, since=_T - timedelta(hours=10), per_suite_limit=2
    )

    assert _local_captures(tmp_path) == {f"alpha/{CONV}/cap_t3.json", f"alpha/{CONV}/cap_t4.json"}
    assert summary.skipped_old == 3
    assert summary.skipped_budget == 1


def test_a_conversation_with_nothing_new_downloads_nothing(tmp_path: Path) -> None:
    store = populated_store()
    _turns(store, ("cap_t0", 2), ("cap_t1", 1))

    summary = fetch_captures(store, base=tmp_path, skip_ids={"cap_t0", "cap_t1"})

    assert _local_captures(tmp_path) == set()
    assert summary.skipped_promoted == 2


def test_a_conversation_entirely_outside_the_window_is_skipped(tmp_path: Path) -> None:
    store = populated_store()
    _turns(store, ("cap_t0", 50), ("cap_t1", 49))

    summary = fetch_captures(store, base=tmp_path, since=_T - timedelta(days=1))

    assert summary.skipped_old == 2
    assert _local_captures(tmp_path) == set()


def test_the_budget_takes_whole_conversations_newest_first(tmp_path: Path) -> None:
    store = populated_store("cap_solo")
    _aged(store, ("cap_solo", 1))
    _turns(store, ("cap_a0", 3), ("cap_a1", 2), folder="conv_" + "a" * 16)
    _turns(store, ("cap_b0", 30), ("cap_b1", 29), ("cap_b2", 28), folder="conv_" + "b" * 16)

    summary = fetch_captures(store, base=tmp_path, per_suite_limit=4)

    # solo (1) + conversation a (2) fit; b (3) does not, so the 1 left takes b's first turn.
    assert _local_captures(tmp_path) == {
        "alpha/cap_solo.json",
        f"alpha/conv_{'a' * 16}/cap_a0.json",
        f"alpha/conv_{'a' * 16}/cap_a1.json",
        f"alpha/conv_{'b' * 16}/cap_b0.json",
    }
    assert summary.skipped_budget == 2


def test_a_conversation_larger_than_the_budget_gives_its_earliest_turns(tmp_path: Path) -> None:
    store = populated_store()
    _turns(store, ("cap_t0", 4), ("cap_t1", 3), ("cap_t2", 2), ("cap_t3", 1))

    summary = fetch_captures(store, base=tmp_path, per_suite_limit=2)

    assert _local_captures(tmp_path) == {f"alpha/{CONV}/cap_t0.json", f"alpha/{CONV}/cap_t1.json"}
    assert summary.skipped_budget == 2


def test_the_leftover_passes_over_a_conversation_whose_leading_turns_are_promoted(
    tmp_path: Path,
) -> None:
    # Neither conversation fits a budget of 2. a ranks first, but its two leading turns are
    # promoted, so spending the leftover there fetches nothing new; b's leading turns get it.
    conv_a, conv_b = "conv_" + "a" * 16, "conv_" + "b" * 16
    store = populated_store()
    _turns(store, ("cap_a0", 10), ("cap_a1", 9), ("cap_a2", 1), folder=conv_a)
    _turns(store, ("cap_b0", 8), ("cap_b1", 7), ("cap_b2", 5), folder=conv_b)

    summary = fetch_captures(store, base=tmp_path, skip_ids={"cap_a0", "cap_a1"}, per_suite_limit=2)

    assert _local_captures(tmp_path) == {
        f"alpha/{conv_b}/cap_b0.json",
        f"alpha/{conv_b}/cap_b1.json",
    }
    assert summary.skipped_budget == 4


def test_the_leftover_goes_unused_when_it_would_fetch_promoted_turns_only(
    tmp_path: Path,
) -> None:
    # After `capture clean`, the promoted turns are gone locally; the new turn's history does
    # not fit, so the leftover must not re-download the same promoted turns on every sync.
    store = populated_store()
    _turns(store, ("cap_t0", 3), ("cap_t1", 2), ("cap_t2", 1))

    summary = fetch_captures(store, base=tmp_path, skip_ids={"cap_t0", "cap_t1"}, per_suite_limit=2)

    assert [key for key in store.gets if key.startswith("captures/")] == []
    assert summary.skipped_budget == 3


def test_a_conversation_budget_fetches_nothing_on_a_rerun(tmp_path: Path) -> None:
    store = populated_store()
    _turns(store, ("cap_t0", 4), ("cap_t1", 3), ("cap_t2", 2))
    fetch_captures(store, base=tmp_path, per_suite_limit=2)
    store.gets.clear()

    fetch_captures(store, base=tmp_path, per_suite_limit=2)

    assert store.gets == []
