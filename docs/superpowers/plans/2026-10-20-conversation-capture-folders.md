# Conversation Capture Folders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the conversation, not the turn, the unit of capture storage end to end. evalshift-sdk files a conversation's turns in one folder and samples whole conversations. The evalshift CLI reads both the old flat layout and the new folders, fetches and cleans whole conversations, and keeps every existing bucket working.

**Architecture:** The SDK keeps writing one immutable file per agent run: no read-modify-write and no reads of the bucket. The only new local cost is GC scanning each conversation folder (one `scandir` per folder, the same one `stat` per file as today). A capture with a `conversation_id` goes to `captures/<suite>/conv_<16 hex of sha256(conversation_id)>/cap_<hex>.json`, both on disk and in a bucket. A standalone capture stays at `captures/<suite>/cap_<hex>.json`. The CLI treats a `conv_*` folder (or a flat file) as one unit when it fetches, budgets and cleans. Promotion keeps keying on envelope fields, so nothing downstream of the reader changes. **The CLI ships first** (it reads both layouts before any SDK writes the new one); the SDK release requires it.

**Tech Stack:** CLI: Python 3.11+, Typer, Rich, pydantic, pytest, ruff, `mypy --strict`. SDK: Python 3.10+, stdlib only in core (D-deps), pytest, ruff, mypy (via `uv run`).

**Spec:** None. The user asked to skip the spec on 2026-10-20. The **Decisions** section below is binding: an executor resolves conflicts against it, then against this plan's task text.

## Decisions (binding)

- **D1. Layout contract.** A capture whose envelope has `conversation_id is not None` lives at `captures/<safe_suite>/<conversation_dir>/<capture_id>.json`; every other capture at `captures/<safe_suite>/<capture_id>.json`. "Under `<base>/`" locally and "under the store prefix" in a bucket are the same relative layout (the bucket stays a mirror of `.evalshift/`). Toolset sidecars stay at `toolsets/<hex>.json`.
- **D2. Folder name.** `conversation_dir(cid) = "conv_" + sha256(str(cid).encode("utf-8")).hexdigest()[:16]` (`str()` so a non-string id can't drop the capture). It is lowercase hex, fixed length, and safe on every filesystem. It never echoes the raw id (testApp's ids carry `:`, and real ids can carry user ids). Readers match `conv_[0-9a-f]{16}` with `fullmatch` and ignore or reject anything else. Golden values, pinned in **both** repos' tests: `"c1"` → `conv_d0f631ca1ddba8db`, `"planner:session-1"` → `conv_e1961a13f5bd6642`, `""` → `conv_e3b0c44298fc1c14`.
- **D3. Escape hatch.** `EVALSHIFT_CAPTURE_LAYOUT=flat` makes the SDK write the old flat layout. It is also a `layout="flat"` argument on `FileSink` / `ObjectStoreSink` (the argument wins over the env var). Unset, empty, `conversation` or any unrecognised value means `conversation`. The escape hatch serves as a production rollback lever and as a bridge for tooling that globs the flat layout.
- **D4. Sampling.** With `sample_rate` set, `capture.agent_session` / `agent_session_async` given a `conversation_id` draw `int(sha256(("evalshift-sample:" + cid).encode("utf-8")).hexdigest()[:16], 16) / 2**64`, compared with `< rate`. Every turn of a conversation gets the same answer in every process. The `@capture.agent` decorator keeps `random.random()` even with a `conversation_id`: that id is fixed when the function is decorated (`api.py:728-730`), so it names no conversation, and hashing it would capture every run or none, forever. Runs without a `conversation_id` and the LangChain handler (which has none) keep `random.random()` too. Golden draws: `"c1"` → `0.5987745500564885`, `"planner:session-1"` → `0.49559010417828814`.
- **D5. Local GC (disk stays bounded).** `max_captures` / `capture_ttl` work on **units**: a flat `*.json` file, or a `conv_*` folder whose time is its newest file's mtime.
  - The count rule walks units newest first. The newest unit keeps its newest `max_captures` files: all of them for a normal conversation, and only the most recent turns of a never-ending one (a static decorator id, or a user id used as conversation id). Older units are kept whole while the running count stays ≤ `max_captures`; the first that doesn't fit is evicted with every older one.
  - The TTL rule evicts **every file** older than the TTL, as today. An idle conversation therefore goes whole, while a live one loses only turns older than the TTL.
  - A folder that ends up empty is removed.
  - A suite with only flat files behaves exactly as today.
  - The trade-off: a conversation longer than `max_captures`, or older than `capture_ttl`, loses its oldest turns locally, and the CLI then reports later rebuilt turns as missing earlier turns. That is the price of a bound; apps should pass the full messages list.
- **D6. Dedup unchanged.** `input_hash` already folds in `conversation_id` and `turn_index` (SDK `serialize.py:221-239`), so dedup never drops a distinct turn. No dedup change. Docs say so.
- **D7. CLI reader.** `iter_captures` reads `<suite>/*.json` and then `<suite>/conv_<16 hex>/*.json` (one level, pattern-matched, never `rglob`). A suite argument maps to the SDK's directory name (`team/support` → `team_support`), so a suite argument can never address a conversation folder.
- **D8. Fetch units.** A bucket key is valid when it is `captures/<suite>/<name>.json` or `captures/<suite>/conv_<16 hex>/<name>.json`; anything else is malformed. A unit is one conversation folder or one flat key.
  - A unit is **wanted** when it holds at least one capture that is not promoted and is inside `since`.
  - From a wanted unit, the fetch downloads every key inside `since`, plus its **promoted** keys at any age, so a new turn's history can be rebuilt on a fresh checkout. That download is bounded by the suite, which holds the promoted turns. Unpromoted keys outside `since` stay behind (`skipped_old`), as flat keys always have.
  - With `per_suite_limit`, a unit counts its keys inside `since` plus its promoted keys. Units with a key inside `since` are ranked newest first by their newest key (unknown `last_modified` counts as oldest), ties broken by unit name. Each unit is taken whole while its count fits the budget; one too big for what's left is skipped for the next.
  - Once no whole unit fits, the budget that's left takes the earliest-written **counted** keys of the first skipped unit. That is never an out-of-window unpromoted key.
  - Local and promoted captures count toward the budget. Forced ids (`--repair`) bypass the window, the promoted skip and the budget.
  - The wanted-unit rule runs with or without the limits, but it only changes anything for folder captures: a flat key is a unit of one, so flat buckets fetch exactly as today.
- **D9. `capture clean`.** Default (`--promoted`): a conversation, grouped by `(envelope.suite, envelope.conversation_id)` whatever its folder, is deleted only when every one of its local turns is promoted; otherwise all of its turns stay, and sync says how many promoted ones it kept. `--all` deletes everything. Conversation folders a run empties are removed.
- **D10. Release order.** CLI first, then the SDK:
  1. CLI changes merge to `main` and ship in **1.6.0** (with the unreleased #57 and #58).
  2. SDK changes ship in **0.7.0**, whose CHANGELOG states "requires evalshift CLI 1.6.0 or later".
  3. Website PRs follow each release.

## Global Constraints

- CLI: every module starts with `from __future__ import annotations`; public functions and classes get Google-style docstrings; `mypy --strict src/evalshift_cli` stays clean.
- CLI: no new CLI flags and no new `evalshift.yaml` keys. Config models keep `extra="forbid"`.
- CLI: never writes to or deletes from a bucket.
- CLI: with neither `captures.max_age` nor `captures.max_cases` set, `capture sync` promotes exactly as before.
- CLI: same captures give the same suite on any machine, and a re-sync with nothing new changes no file.
- SDK: core stays stdlib-only (D-deps). Every capture-path fault fails open and never raises into the host app (`safety.guard` / `safety.fail_open`).
- SDK: Python floor `>=3.10` (no 3.11-only syntax).
- Both repos: Conventional Commits, each message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Both repos: user-visible changes update `CHANGELOG.md` under `## [Unreleased]`, `DOCS.md`, `llms-full.txt` and the matching docs page in the same task.
- The user merges PRs (`gh pr merge` is blocked). Release PRs are separate.
- Harness: run scripts directly (no `source`, no `bash script.sh`, no `cd a && b` chains). Use absolute paths and `uv run --directory <repo>`. Edit files with the Edit tool.

## Compatibility matrix (what each combination does)

| SDK writes | CLI reads | Result |
| --- | --- | --- |
| ≤ 0.6.0 (flat) | ≤ 1.5.x | Today's behaviour. |
| ≤ 0.6.0 (flat) | ≥ 1.6.0 | Works: flat files are units of one, and conversations group by envelope as today. |
| ≥ 0.7.0 (folders) | ≥ 1.6.0 | Works: the target state. |
| ≥ 0.7.0 (folders) | ≤ 1.5.x | **Broken, silently:** nested captures are invisible locally, the fetch counts them as malformed, and `capture clean`'s sidecar sweep can delete toolsets only nested captures use. Mitigation: the SDK CHANGELOG/README require CLI ≥ 1.6.0, and `EVALSHIFT_CAPTURE_LAYOUT=flat` restores the old layout. |
| ≥ 0.7.0 with `EVALSHIFT_CAPTURE_LAYOUT=flat` | any | Same as SDK ≤ 0.6.0, except sampling is per conversation. |
| Mixed bucket (upgrade mid-conversation) | ≥ 1.6.0 | Works locally: promotion groups by `envelope.conversation_id` across both layouts. On a fresh checkout, though, the conversation's promoted *flat* turns are not fetched (each is a unit of one with nothing new), so its folder turns that need them for history are reported as missing earlier turns. |

## Review Focus

1. **A conversation split across layouts** (SDK upgraded mid-conversation: turn 0 flat, turn 1 in a folder): it should sync as one conversation with turn 1's history rebuilt from turn 0. Pinned by Task A1's `test_a_conversation_split_across_layouts_syncs_as_one`.
2. **A fresh checkout whose conversation has promoted turns plus one new turn**: the new turn should be promoted with its full history, not skipped as "missing earlier turns". Pinned by Task A4's integration test.
3. **A folder or key that only looks like a conversation folder** (`nested/`, uppercase hex, 8 or 17 hex digits, two levels deep): it should be ignored locally and counted as malformed remotely, never read or written. Pinned by Task A1's and Task A2's parametrized tests.
4. **A conversation that never ends** (a static decorator `conversation_id`, or a user id as conversation id): local disk should stay bounded at `max_captures` (its oldest turns go), and sampling should not freeze it at all-or-nothing. Pinned by Task B3's `test_a_conversation_larger_than_the_cap_keeps_only_its_newest_turns` and Task B2's `test_the_decorators_static_conversation_id_samples_at_random`.
5. **`capture clean` on a half-promoted conversation**: its promoted turns should stay while any turn is unpromoted, and an emptied folder should disappear. Pinned by Task A3's tests.

---

## File Structure

**CLI (`/home/lukas/repos/evalshift/evalshift-cli/.claude/worktrees/bounded-capture-suites`, branch `feat/conversation-folders`):**
- `src/evalshift_cli/captures/reader.py`: adds `CONVERSATION_DIR`, `is_conversation_dir`, `capture_suite_dirname` and `_capture_files`; `iter_captures` reads conversation folders. Path-contract docstring.
- `src/evalshift_cli/captures/remote.py`: nested-key validation, `_ListedCapture`, unit-based `_budgeted_keys` (replaces `_newest_per_suite`), and the wanted-unit rule.
- `src/evalshift_cli/cli/commands/capture.py`: `_cleanable`, `_remove_empty_conversation_dirs`, `capture_clean` wiring, and the missing-earlier-turns warning text.
- Tests: `tests/unit/test_captures_reader.py`, `tests/unit/test_captures_remote.py`, `tests/unit/test_cli_capture.py`, `tests/integration/test_capture_lifecycle.py`.
- Docs: `CHANGELOG.md`, `DOCS.md`, `llms-full.txt`, `docs/configuration.md`, `docs/conversations.md`, `docs/sdk.md`, docstrings in `captures/__init__.py`, `config/models.py`.

**SDK (`/home/lukas/repos/evalshift/evalshift-sdk-wt-conversation-folders`, branch `feat/conversation-folders` from `origin/main`):**
- Create `src/evalshift/sinks/_layout.py`: `CaptureLayout`, `LAYOUT_ENV`, `conversation_dir`, `is_conversation_dir`, `resolve_layout`.
- `src/evalshift/sinks/file.py`: `capture_relpath`; `FileSink(base, *, layout=None)`; `FileSink.suite_dir(envelope)`.
- `src/evalshift/sinks/object_store.py`: `ObjectStoreSink(..., layout=None)`, key via `capture_relpath`.
- `src/evalshift/hygiene/sample.py`: `conversation_draw`; `should_capture(rate, *, rng=None, key=None)`.
- `src/evalshift/config.py`: `should_capture_now(conversation_id=None)`.
- `src/evalshift/capture/api.py`: 4 call sites pass `conversation_id`.
- `src/evalshift/hygiene/gc.py`: unit-based `evict`.
- `src/evalshift/sinks/hygiene.py`: GC on the suite dir.
- Tests: create `tests/test_conversation_layout.py`; modify `tests/conftest.py`, `tests/test_hygiene_gc.py`, `tests/test_hygiene_sink.py`, `tests/test_capture_sampling.py`.
- Docs: `CHANGELOG.md`, `README.md`, `DOCS.md`, `llms-full.txt`, `llms.txt`, `docs/SCHEMA.md`, `docs/DECISIONS.md`, `examples/support_agent/README.md`, `src/evalshift/hygiene/__init__.py` docstring.

---

## Part A — evalshift CLI (ships first, in 1.6.0)

All Part A commands run from the CLI worktree root `/home/lukas/repos/evalshift/evalshift-cli/.claude/worktrees/bounded-capture-suites` with `.venv/bin/pytest -q -p no:cacheprovider --no-cov <paths>` for targeted runs.

### Task A1: The reader sees conversation folders

**Files:**
- Modify: `src/evalshift_cli/captures/reader.py` (module docstring; constants after `_DEFAULT_BASE`; `iter_captures` at ~`305-339`)
- Modify: `src/evalshift_cli/captures/remote.py` (fetch `prefix` line only, ~`313`)
- Test: `tests/unit/test_captures_reader.py`, `tests/unit/test_cli_capture.py`, `tests/unit/test_captures_remote.py`

**Interfaces:**
- Produces: `reader.CONVERSATION_DIR: re.Pattern[str]`; `reader.is_conversation_dir(name: str) -> bool`; `reader.capture_suite_dirname(suite: str) -> str`. Tasks A2 and A3 import these.
- Produces: the `tests/unit/test_cli_capture.py::_write_capture(..., conversation_dir: str | None = None)` parameter. Task A3 uses it.

- [ ] **Step 1: Write the failing reader tests** (append to `tests/unit/test_captures_reader.py`; it already imports `json`, `Path`, `Any`, `pytest`, `iter_captures`, `find_capture`; add `capture_suite_dirname` to the `evalshift_cli.captures.reader` import)

```python
#: The folder evalshift-sdk >= 0.7.0 writes for conversation_id "c1" (pinned in
#: evalshift-sdk tests/test_conversation_layout.py too).
CONV = "conv_d0f631ca1ddba8db"


def _write_nested(base: Path, payload: dict[str, Any], folder: str = CONV) -> Path:
    conv_dir = base / "captures" / payload["suite"] / folder
    conv_dir.mkdir(parents=True, exist_ok=True)
    path = conv_dir / f"{payload['capture_id']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_iter_captures_reads_conversation_folders_beside_flat_captures(tmp_path: Path) -> None:
    _write_capture(tmp_path, _capture_dict(capture_id="cap_flat", suite="alpha"))
    nested = _write_nested(tmp_path, _capture_dict(capture_id="cap_turn", suite="alpha"))

    records = iter_captures(suite="alpha", base=tmp_path)

    assert {r.envelope.capture_id: r.path for r in records} == {
        "cap_flat": tmp_path / "captures" / "alpha" / "cap_flat.json",
        "cap_turn": nested,
    }
    assert [r.envelope.capture_id for r in iter_captures(base=tmp_path)] == ["cap_flat", "cap_turn"]


@pytest.mark.parametrize(
    "folder",
    ["nested", "conv_D0F631CA1DDBA8DB", "conv_d0f631ca", "conv_d0f631ca1ddba8db0"],
)
def test_iter_captures_ignores_folders_the_sdk_never_writes(tmp_path: Path, folder: str) -> None:
    _write_nested(tmp_path, _capture_dict(capture_id="cap_x", suite="alpha"), folder=folder)
    assert iter_captures(base=tmp_path) == []


def test_find_capture_finds_a_turn_in_a_conversation_folder(tmp_path: Path) -> None:
    path = _write_nested(tmp_path, _capture_dict(capture_id="cap_turn", suite="alpha"))
    assert find_capture("cap_turn", base=tmp_path).path == path


def test_a_suite_name_reads_the_directory_the_sdk_wrote_it_to(tmp_path: Path) -> None:
    suite_dir = tmp_path / "captures" / "team_support"
    suite_dir.mkdir(parents=True)
    payload = _capture_dict(capture_id="cap_1", suite="team/support")
    (suite_dir / "cap_1.json").write_text(json.dumps(payload), encoding="utf-8")

    for name in ("team/support", "team_support"):
        assert [r.envelope.capture_id for r in iter_captures(suite=name, base=tmp_path)] == ["cap_1"]


def test_a_suite_argument_cannot_address_a_conversation_folder(tmp_path: Path) -> None:
    _write_nested(tmp_path, _capture_dict(capture_id="cap_turn", suite="alpha"))
    assert iter_captures(suite=f"alpha/{CONV}", base=tmp_path) == []


@pytest.mark.parametrize(
    ("suite", "dirname"),
    [
        ("alpha", "alpha"),
        ("team/support", "team_support"),
        ("a\\b", "a_b"),
        ("../x", "__x"),
        (" .hidden. ", "hidden"),
        ("..", "_"),
        ("", "_"),
    ],
)
def test_capture_suite_dirname_matches_the_sdks_safe_segment(suite: str, dirname: str) -> None:
    assert capture_suite_dirname(suite) == dirname
```

- [ ] **Step 2: Write the failing CLI tests.** In `tests/unit/test_cli_capture.py`, first add `conversation_dir: str | None = None,` as the last keyword parameter of `_write_capture`, and replace its `suite_dir = base / "captures" / suite` line with:

```python
    suite_dir = base / "captures" / suite
    if conversation_dir is not None:
        suite_dir = suite_dir / conversation_dir
```

Then append:

```python
#: The folder evalshift-sdk >= 0.7.0 writes for conversation_id "c1".
CONV = "conv_d0f631ca1ddba8db"


def test_clean_keeps_a_sidecar_only_a_conversation_folder_capture_uses(tmp_path: Path) -> None:
    _write_capture(
        tmp_path, capture_id="cap_t0", conversation_id="c1", turn_index=0, conversation_dir=CONV
    )

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "toolsets" / f"{_TOOLSET_REF.removeprefix('sha256:')}.json").exists()


def test_a_conversation_split_across_layouts_syncs_as_one(tmp_path: Path) -> None:
    # The SDK was upgraded mid-conversation: turn 0 is flat, turn 1 sits in its folder.
    _write_min_config(tmp_path / "evalshift.yaml")
    _write_capture(
        tmp_path, capture_id="cap_t0", conversation_id="c1", turn_index=0, model_input="turn zero"
    )
    _write_capture(
        tmp_path,
        capture_id="cap_t1",
        conversation_id="c1",
        turn_index=1,
        model_input="turn one",
        conversation_dir=CONV,
    )

    result = runner.invoke(
        app,
        [
            "capture", "sync", "--offline",
            "--config", str(tmp_path / "evalshift.yaml"), "--base", str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    case = json.loads((tmp_path / "suites" / "support_agent" / "cap_t1.json").read_text())
    assert [m["content"] for m in case["example"]["history"]][:1] == ["turn zero"]
```

Append to `tests/unit/test_captures_remote.py` (the fetch's `--suite` prefix uses the same mapping):

```python
def test_a_suite_filter_lists_the_directory_the_sdk_wrote(tmp_path: Path) -> None:
    store = populated_store("cap_1", suite="team_support")

    summary = fetch_captures(store, base=tmp_path, suite="team/support")

    assert summary.captures == 1
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/pytest -q -p no:cacheprovider --no-cov tests/unit/test_captures_reader.py tests/unit/test_cli_capture.py tests/unit/test_captures_remote.py -k "conversation or suite_name or suite_argument or dirname or find_capture_finds or suite_filter"`
Expected: FAIL (`ImportError: cannot import name 'capture_suite_dirname'`, and the nested captures are not found).

- [ ] **Step 4: Implement in `reader.py`.** Replace the module docstring's path-contract block with:

```python
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
"""
```

Keep every paragraph that followed the old contract block (promoted cases, toolset sidecars, the raising read path). Then add `import re` to the imports and these definitions after `_DEFAULT_BASE = ".evalshift"`:

```python
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
```

In `iter_captures`, replace `[root / suite] if suite is not None else ...` with `[root / capture_suite_dirname(suite)] if suite is not None else ...`. Replace `for path in sorted(suite_dir.glob("*.json")):` with `for path in _capture_files(suite_dir):`. Add `"CONVERSATION_DIR", "capture_suite_dirname", "is_conversation_dir"` to `__all__` if the module defines one.

- [ ] **Step 5: Use the same suite mapping in the fetch.** In `remote.py`, add `capture_suite_dirname` to the `from evalshift_cli.captures.reader import ...` line, and change the capture listing prefix to:

```python
    prefix = f"captures/{capture_suite_dirname(suite)}/" if suite else "captures/"
```

- [ ] **Step 6: Run the tests to verify they pass, then the capture test files**

Run: `.venv/bin/pytest -q -p no:cacheprovider --no-cov tests/unit/test_captures_reader.py tests/unit/test_cli_capture.py tests/unit/test_cli_capture_limits.py tests/unit/test_cli_capture_remote.py tests/unit/test_captures_remote.py`
Expected: PASS. If `test_a_suite_named_with_a_slash_*` in `test_cli_capture_limits.py` now reads more captures than before, that is the bug this task fixes. Update its expectation and say why in the commit message.

- [ ] **Step 7: Commit**

```bash
git add src/evalshift_cli/captures/reader.py src/evalshift_cli/captures/remote.py tests/unit/test_captures_reader.py tests/unit/test_cli_capture.py tests/unit/test_captures_remote.py
git commit -m "feat(capture): read the conversation folders evalshift-sdk 0.7.0 writes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task A2: The fetch moves whole conversations

**Files:**
- Modify: `src/evalshift_cli/captures/remote.py` (capture section of `fetch_captures` ~`313-367`; replace `_newest_per_suite` ~`371-392`; module docstring lines 3-5 and the `fetch_captures` docstring)
- Test: `tests/unit/test_captures_remote.py`

**Interfaces:**
- Consumes: `reader.is_conversation_dir`, `reader.capture_suite_dirname` (Task A1).
- Produces: `fetch_captures` keeps its signature and `FetchSummary` keeps every field and label (sync, list and docs depend on the exact strings). `_newest_per_suite` is deleted and replaced by `_budgeted_keys(units, *, since, limit) -> frozenset[str]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_captures_remote.py` after the per-suite budget section; `_T`, `_aged`, `populated_store`, `capture_payload`, `_local_captures`, `FakeRemoteStore` already exist there)

```python
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
        ("cap_t0", 100), ("cap_t1", 90), ("cap_t2", 80), ("cap_t3", 3), ("cap_t4", 2), ("cap_t5", 1),
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


def test_a_conversation_budget_fetches_nothing_on_a_rerun(tmp_path: Path) -> None:
    store = populated_store()
    _turns(store, ("cap_t0", 4), ("cap_t1", 3), ("cap_t2", 2))
    fetch_captures(store, base=tmp_path, per_suite_limit=2)
    store.gets.clear()

    fetch_captures(store, base=tmp_path, per_suite_limit=2)

    assert store.gets == []

```

Guard tests that pass before and after this task (expected, not a mistake): all four cases of `test_fetch_counts_folders_the_sdk_never_writes_as_malformed`, and `test_a_conversation_budget_fetches_nothing_on_a_rerun`. Every other new test fails first.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q -p no:cacheprovider --no-cov tests/unit/test_captures_remote.py -k "conversation or folder or budget_takes_whole or new_turn or nothing_new or outside_the_window or stays_behind or leading_turns"`
Expected: FAIL. The nested keys are counted as malformed, so nothing is downloaded.

- [ ] **Step 3: Implement.** In `remote.py`:
  - add `is_conversation_dir` to the `reader` import;
  - add `Mapping` to the `collections.abc` import;
  - replace lines 3-5 of the module docstring with: `The SDK's ``ObjectStoreSink`` writes the *local layout* under a bucket prefix -- ``captures/<suite>/cap_<hex>.json``, ``captures/<suite>/conv_<16 hex>/cap_<hex>.json`` for a conversation's turns (evalshift-sdk 0.7.0 and later) and ``toolsets/<hex>.json`` -- so the CLI never reads a`;
  - then add above `fetch_captures`:

```python
@dataclass(frozen=True, slots=True)
class _ListedCapture:
    """A capture key that passed the layout checks, and where it lands locally.

    ``unit`` is what the fetch moves together: ``<suite>/<conv folder>`` for a conversation's
    turns, the key itself for a flat capture.
    """

    info: ObjectInfo
    suite: str
    unit: str
    dest: Path
    capture_id: str
```

Replace the capture section of `fetch_captures`, from `capture_jobs: list[tuple[str, Path]] = []` through the `capture_jobs.append(...)` loop, with:

```python
    capture_jobs: list[tuple[str, Path]] = []
    prefix = f"captures/{capture_suite_dirname(suite)}/" if suite else "captures/"
    listed: list[_ListedCapture] = []
    for info in _list(store, prefix):
        if info.key.endswith("/"):
            continue  # a folder marker, as above
        parts = info.key.removeprefix("captures/").split("/")
        conversation: str | None = None
        if len(parts) == 2:
            suite_seg, name = parts
        elif len(parts) == 3 and is_conversation_dir(parts[1]):
            suite_seg, conversation, name = parts
        else:
            summary.skipped_malformed += 1
            continue
        if not _is_safe_suite_segment(suite_seg) or not _CAPTURE_NAME.fullmatch(name):
            summary.skipped_malformed += 1
            continue
        folder = captures_dir / suite_seg
        if conversation is not None:
            folder = folder / conversation
        dest = folder / name
        if not _absolute(dest.parent).is_relative_to(captures_dir):
            summary.skipped_malformed += 1
            continue
        listed.append(
            _ListedCapture(
                info=info,
                suite=suite_seg,
                unit=info.key if conversation is None else f"{suite_seg}/{conversation}",
                dest=dest,
                capture_id=name[: -len(".json")],
            )
        )

    units: dict[str, list[_ListedCapture]] = {}
    for item in listed:
        units.setdefault(item.unit, []).append(item)
    # A unit is fetched for a capture that is new (not promoted) and inside the window. It then
    # brings its promoted turns too, whatever their age -- a new turn's history is rebuilt from
    # them on a fresh checkout, and the suite bounds how many there are. Unpromoted turns
    # outside the window stay behind, as flat captures always have.
    wanted = {
        unit
        for unit, items in units.items()
        if any(i.capture_id not in skip and _inside(i.info, since) for i in items)
    }
    in_budget = (
        _budgeted_keys(units, since=since, promoted=skip, limit=per_suite_limit)
        if per_suite_limit is not None
        else None
    )
    for item in listed:
        info, dest = item.info, item.dest
        if dest.exists():
            summary.skipped_existing += 1
            continue
        if item.capture_id not in force:
            promoted = item.capture_id in skip
            if item.unit not in wanted:
                if promoted:
                    summary.skipped_promoted += 1
                else:
                    summary.skipped_old += 1
                continue
            if not promoted and not _inside(info, since):
                summary.skipped_old += 1
                continue
            if in_budget is not None and info.key not in in_budget:
                summary.skipped_budget += 1
                continue
        if info.size is not None and info.size > MAX_OBJECT_BYTES:
            summary.skipped_large += 1
            continue
        capture_jobs.append((info.key, dest))
```

Replace `_newest_per_suite` entirely with:

```python
def _inside(info: ObjectInfo, since: datetime | None) -> bool:
    """Whether ``info`` was written inside the ``since`` window; an unknown time always is."""
    return since is None or info.last_modified is None or info.last_modified >= since


def _budgeted_keys(
    units: Mapping[str, Sequence[_ListedCapture]],
    *,
    since: datetime | None,
    promoted: Collection[str],
    limit: int,
) -> frozenset[str]:
    """The keys each suite's budget of ``limit`` captures takes, a conversation at a time.

    A unit is a conversation folder, or one flat capture; it counts its captures inside
    ``since`` plus its promoted ones (the turns a fetch of it brings). Units with a capture
    inside ``since`` are ranked newest first by their newest one (an unknown ``last_modified``
    counts as oldest), ties by unit name, and taken whole while they fit; one too big for what
    is left is skipped for the next. Once no whole unit fits, the budget left takes the
    earliest-written counted captures of the first one skipped -- leading turns, never an old
    unpromoted one. Captures already local or promoted count, so a re-run with nothing new
    fetches nothing.
    """

    def stamp(item: _ListedCapture) -> float:
        modified = item.info.last_modified
        return modified.timestamp() if modified is not None else float("-inf")

    by_suite: dict[str, list[tuple[str, Sequence[_ListedCapture]]]] = {}
    for unit, items in units.items():
        if not any(_inside(i.info, since) for i in items):
            continue
        counted = [i for i in items if _inside(i.info, since) or i.capture_id in promoted]
        by_suite.setdefault(items[0].suite, []).append((unit, counted))
    taken: set[str] = set()
    for ranked in by_suite.values():
        ranked.sort(key=lambda pair: (-max(stamp(i) for i in pair[1]), pair[0]))
        left = limit
        skipped: Sequence[_ListedCapture] | None = None
        for _, items in ranked:
            if len(items) <= left:
                taken.update(i.info.key for i in items)
                left -= len(items)
            elif skipped is None:
                skipped = items
        if left > 0 and skipped is not None:
            earliest = sorted(skipped, key=lambda i: (stamp(i), i.info.key))
            taken.update(i.info.key for i in earliest[:left])
    return frozenset(taken)
```

Update the `fetch_captures` docstring paragraph that starts "``per_suite_limit`` caps the download". Replace it with the D8 rule in two sentences: whole conversations newest first, then the leading in-window turns of the first one that doesn't fit; and a new turn brings its conversation's promoted turns along, whatever their age.

- [ ] **Step 4: Run the remote tests, including every existing budget test**

Run: `.venv/bin/pytest -q -p no:cacheprovider --no-cov tests/unit/test_captures_remote.py tests/unit/test_cli_capture_remote.py tests/unit/test_cli_capture_limits.py`
Expected: PASS. The #58 budget tests (`test_per_suite_limit_*`, `test_equal_last_modified_breaks_ties_by_key`, `test_an_unknown_last_modified_counts_as_oldest_for_the_budget`, `test_captures_outside_the_window_take_no_budget`, `test_forced_ids_are_fetched_beyond_the_budget`) must pass **unchanged**: flat keys are units of one. `test_fetch_rejects_malformed_and_traversal_keys` must still count its `captures/alpha/nested/cap_x.json` as malformed.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/captures/remote.py tests/unit/test_captures_remote.py
git commit -m "feat(capture): fetch and budget whole conversation folders" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task A3: `capture clean` keeps conversations whole

**Files:**
- Modify: `src/evalshift_cli/cli/commands/capture.py` (`capture_clean` ~`1066-1133`; two helpers above it; imports)
- Test: `tests/unit/test_cli_capture.py`

**Interfaces:**
- Consumes: `reader.is_conversation_dir` (Task A1); `_write_capture(..., conversation_dir=...)` and `CONV` (Task A1's test additions).
- Produces: `_cleanable(records: Sequence[CaptureRecord], promoted: Collection[str]) -> tuple[list[CaptureRecord], int]`; `_remove_empty_conversation_dirs(paths: Iterable[Path]) -> None`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_cli_capture.py`)

```python
@pytest.mark.parametrize("folder", [None, CONV])
def test_clean_keeps_a_conversation_whole_while_a_turn_is_unpromoted(
    tmp_path: Path, folder: str | None
) -> None:
    for turn in range(2):
        _write_capture(
            tmp_path,
            capture_id=f"cap_t{turn}",
            conversation_id="c1",
            turn_index=turn,
            model_input=f"turn {turn}",
            conversation_dir=folder,
        )
    assert _invoke(["promote", "cap_t0", "--as", "t0"], tmp_path).exit_code == 0

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    suite_dir = tmp_path / "captures" / "support_agent"
    where = suite_dir / folder if folder else suite_dir
    assert (where / "cap_t0.json").exists()
    assert (where / "cap_t1.json").exists()
    said = " ".join(result.stdout.split())
    assert "kept 1 promoted capture(s)" in said
    assert "nothing to clean" not in said


def test_clean_removes_a_fully_promoted_conversation_folder(tmp_path: Path) -> None:
    for turn in range(2):
        _write_capture(
            tmp_path,
            capture_id=f"cap_t{turn}",
            conversation_id="c1",
            turn_index=turn,
            model_input=f"turn {turn}",
            conversation_dir=CONV,
        )
        assert _invoke(["promote", f"cap_t{turn}", "--as", f"t{turn}"], tmp_path).exit_code == 0

    result = _invoke(["clean", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "captures" / "support_agent" / CONV).exists()
    assert (tmp_path / "suites" / "support_agent" / "t1.json").exists()


def test_clean_all_removes_conversation_folders(tmp_path: Path) -> None:
    _write_capture(
        tmp_path, capture_id="cap_t0", conversation_id="c1", turn_index=0, conversation_dir=CONV
    )

    result = _invoke(["clean", "--all", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "captures" / "support_agent" / CONV).exists()


def test_clean_leaves_a_folder_it_did_not_empty(tmp_path: Path) -> None:
    _write_capture(
        tmp_path, capture_id="cap_t0", conversation_id="c1", turn_index=0, conversation_dir=CONV
    )
    (tmp_path / "captures" / "support_agent" / CONV / "notes.txt").write_text("keep me")

    result = _invoke(["clean", "--all", "--yes"], tmp_path)

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "captures" / "support_agent" / CONV / "notes.txt").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q -p no:cacheprovider --no-cov tests/unit/test_cli_capture.py -k "clean_keeps_a_conversation or clean_removes_a_fully or clean_all_removes_conversation or folder_it_did_not"`
Expected: FAIL, except `test_clean_leaves_a_folder_it_did_not_empty`, which passes before and after: it guards the `rmdir`. Today `clean` deletes `cap_t0` and never removes folders.

- [ ] **Step 3: Implement.** In `capture.py`:
  - add `is_conversation_dir` and `CaptureRecord` to the `evalshift_cli.captures.reader` import if they are missing;
  - add `Collection` to the `collections.abc` import;
  - add `from contextlib import suppress` (keep the module's import order; ruff `I` sorts it);
  - add above `capture_clean`:

```python
def _cleanable(
    records: Sequence[CaptureRecord], promoted: Collection[str]
) -> tuple[list[CaptureRecord], int]:
    """The promoted captures ``capture clean`` deletes, and how many promoted ones it keeps.

    A conversation is cleaned whole, whichever folder its turns sit in: while any of its local
    turns is unpromoted, its promoted turns stay, because a later turn without its own messages
    list rebuilds its history from them. A capture outside any conversation goes on its own.
    """
    units: dict[tuple[str, ...], list[CaptureRecord]] = {}
    for record in records:
        envelope = record.envelope
        key = (
            ("conversation", envelope.suite, envelope.conversation_id)
            if envelope.conversation_id is not None
            else ("capture", str(record.path))
        )
        units.setdefault(key, []).append(record)
    doomed: list[CaptureRecord] = []
    kept = 0
    for unit in units.values():
        done = [r for r in unit if r.envelope.capture_id in promoted]
        if len(done) == len(unit):
            doomed.extend(unit)
        else:
            kept += len(done)
    return sorted(doomed, key=lambda r: r.path), kept


def _remove_empty_conversation_dirs(paths: Iterable[Path]) -> None:
    """Remove the conversation folders ``paths`` emptied; a folder still holding files stays."""
    for folder in sorted({p.parent for p in paths if is_conversation_dir(p.parent.name)}):
        with suppress(OSError):
            folder.rmdir()
```

In `capture_clean`, replace

```python
    if not all_captures:
        promoted = _promoted_capture_ids_or_warn(base=base, console=console)
        records = [r for r in records if r.envelope.capture_id in promoted]
```

with

```python
    if not all_captures:
        promoted = _promoted_capture_ids_or_warn(base=base, console=console)
        records, held = _cleanable(records, promoted)
        if held:
            console.print(
                f"[dim]kept {held} promoted capture(s): their conversation still has unpromoted "
                "turns, whose history is rebuilt from them.[/dim]"
            )
```

Guard the existing "nothing to clean" line so it doesn't contradict that message. Initialise `held = 0` before the `if not all_captures:` block, and change `if not records:` to:

```python
    if not records and not held:
        scope = "captures" if all_captures else "promoted captures"
        console.print(f"[dim]nothing to clean — no {scope} found.[/dim]")
    elif records:
```

(the old `else:` branch becomes this `elif records:`), and the deletion loop with

```python
        removed = 0
        gone: list[Path] = []
        for r in records:
            try:
                r.path.unlink()
            except OSError as exc:
                console.print(f"[yellow]⚠[/yellow] could not remove {r.path}: {exc}")
                continue
            removed += 1
            gone.append(r.path)
        _remove_empty_conversation_dirs(gone)
        console.print(f"[green]✓[/green] removed {removed} capture file(s).")
```

Add one sentence to the `capture_clean` docstring: "A conversation's promoted turns stay while any of its turns is unpromoted; folders it empties are removed."

- [ ] **Step 4: Run the clean tests and the whole capture CLI file**

Run: `.venv/bin/pytest -q -p no:cacheprovider --no-cov tests/unit/test_cli_capture.py`
Expected: PASS. The existing `test_clean_*` tests use standalone captures, so they are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/cli/commands/capture.py tests/unit/test_cli_capture.py
git commit -m "feat(capture): clean keeps a conversation whole while a turn is unpromoted" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task A4: A fresh checkout rebuilds a new turn's history (integration)

**Files:**
- Test: `tests/integration/test_capture_lifecycle.py`

**Interfaces:**
- Consumes: `_Bucket`, `_config`, `_sync`, `_said`, `_cases`, `_day` from that file; `capture_payload(..., conversation_id=, turn_index=)` from `tests/unit/fake_remote_store.py`.

- [ ] **Step 1: Write the test** (append to `tests/integration/test_capture_lifecycle.py`)

```python
#: The folder evalshift-sdk >= 0.7.0 writes for conversation_id "planner:session-1" (pinned in
#: evalshift-sdk tests/test_conversation_layout.py too).
CONV = "conv_e1961a13f5bd6642"


def _turn(bucket: _Bucket, turn: int, when: datetime) -> None:
    key = f"captures/planner/{CONV}/cap_t{turn}.json"
    bucket.store.objects[key] = capture_payload(
        f"cap_t{turn}",
        "planner",
        model_input=f"step {turn}",
        created_at=when.isoformat(),
        model_id="claude-sonnet-5-20260301",
        conversation_id="planner:session-1",
        turn_index=turn,
    )
    bucket.store.modified[key] = when


def test_a_fresh_checkout_rebuilds_a_new_turns_history_from_promoted_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = _Bucket(monkeypatch)
    _config(tmp_path / "evalshift.yaml")
    _turn(bucket, 0, _day(10))
    _turn(bucket, 1, _day(11))
    _sync(tmp_path, monkeypatch, _day(12))
    assert _cases(tmp_path) == {"cap_t0", "cap_t1"}

    # A teammate's fresh checkout: the committed case files, no local captures.
    shutil.rmtree(tmp_path / "captures")
    _turn(bucket, 2, _day(12))
    result = _sync(tmp_path, monkeypatch, _day(13))

    assert _cases(tmp_path) == {"cap_t0", "cap_t1", "cap_t2"}
    assert "missing earlier turns" not in _said(result)
    case = json.loads((tmp_path / "suites" / "planner" / "cap_t2.json").read_text())
    assert [m["content"] for m in case["example"]["history"] if m["role"] == "user"] == [
        "step 0",
        "step 1",
    ]
```

Add `import json` to the file's imports. `shutil` is already imported.

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest -q -p no:cacheprovider --no-cov -m integration tests/integration/test_capture_lifecycle.py`
Expected: PASS once Tasks A1-A2 are in. To check the test is real, revert Task A2's `wanted` rule locally (make every key obey the promoted skip): the test must FAIL with `cap_t2` missing and "missing earlier turns" in the output. Restore the rule afterwards; do not commit the revert.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_capture_lifecycle.py
git commit -m "test(capture): a fresh checkout rebuilds a new turn's history from its folder" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task A5: CLI docs, warning text and changelog

**Files:**
- Modify: `CHANGELOG.md`, `DOCS.md`, `llms-full.txt`, `docs/configuration.md`, `docs/conversations.md`, `docs/sdk.md`, `src/evalshift_cli/captures/__init__.py` (docstring lines 5, 11), `src/evalshift_cli/cli/commands/capture.py` (module docstring line 10; the "turn(s) not promoted" warning), `src/evalshift_cli/config/models.py` (~line 845 docstring)
- Test: none new. `tests/unit/test_cli_capture_limits.py::test_a_turn_whose_earlier_turns_are_not_local_is_not_eligible` asserts the warning's stable substrings, which this task keeps.

- [ ] **Step 1: Warning text.** In `capture.py`, replace the body of the `if ineligible["history"]:` warning with:

```python
            console.print(
                f"[yellow]⚠[/yellow] {escape(suite_name)}: {ineligible['history']} turn(s) not "
                "promoted: earlier turns of their conversation are not on disk (outside "
                "captures.max_age or the fetch budget, removed by capture clean, written flat -- "
                "evalshift-sdk older than 0.7.0 or EVALSHIFT_CAPTURE_LAYOUT=flat -- and promoted on "
                "another machine or sampled out, evicted by the SDK's max_captures / capture_ttl, "
                "or a turn_index that does not start at 0). Pass the full messages list to "
                "capture.model_call so each turn carries its own history.",
                soft_wrap=True,
            )
```

- [ ] **Step 2: CHANGELOG.** Under `## [Unreleased]` → `### Added`, append:

```markdown
- **Conversation folders.** evalshift-sdk 0.7.0 files a conversation's turns
  in one folder, `captures/<suite>/conv_<16 hex>/`, on disk and in a bucket.
  The CLI reads it beside the flat layout, which older SDKs and standalone
  captures keep using, so existing buckets need no migration.
  `capture sync`'s and `capture list`'s fetch moves whole conversations: the
  `max_cases` budget counts captures but takes a conversation at a time,
  newest first, and a conversation with a new turn brings its promoted turns
  along, whatever their age, so a fresh checkout rebuilds the new turn's
  history instead of reporting it as missing earlier turns.
```

Under `### Changed`, append:

```markdown
- `capture clean` (the default `--promoted` scope) keeps a conversation's
  promoted captures while any of its turns is unpromoted, since a later turn
  rebuilds its history from them, and says how many it kept. It removes the
  conversation folders it empties.
- `capture list <suite>`, `capture sync --suite` and `capture fetch --suite`
  read the directory the SDK writes the suite to, so `team/support` finds
  `captures/team_support/`.
```

- [ ] **Step 3: User docs.** Apply each change. The quoted text is what is there today; find it with `grep -n`.
  - `docs/configuration.md`, "Under the prefix the layout is exactly the local one — `captures/<suite>/cap_<hex>.json` and `toolsets/<hex>.json`": replace with "Under the prefix the layout is exactly the local one — `captures/<suite>/cap_<hex>.json`, `captures/<suite>/conv_<16 hex>/cap_<hex>.json` for a conversation's turns (evalshift-sdk 0.7.0 and later), and `toolsets/<hex>.json`".
  - `docs/configuration.md` "Fetch window", the `max_cases` bullet: replace its first sentence with "**`max_cases`** downloads at most `3 × max_cases` captures per suite per sync inside that window, a conversation at a time: conversation folders and lone captures newest first by their newest capture (ties by name), each taken whole while it fits; once none fits, what is left takes the earliest turns of the next one." Keep the rest of the bullet.
  - In the same section, after that bullet, add the paragraph: "A conversation folder with a new turn inside the window also brings its promoted turns, whatever their age, so a fresh checkout rebuilds the new turn's history; its unpromoted turns outside the window stay behind. Turns written flat (an evalshift-sdk older than 0.7.0, or `EVALSHIFT_CAPTURE_LAYOUT=flat`) are fetched one at a time, as before."
  - `docs/configuration.md` plan step 1: replace "Its later turns can enter on a later sync while the earlier turns' captures are still local; after `capture clean` or on a fresh checkout, a later turn that does not record its own messages list is missing earlier turns (below)." with "Its later turns can enter on a later sync: the fetch brings a conversation folder's promoted turns along, and `capture clean` keeps them while any turn is unpromoted (a fully promoted conversation is cleaned, so a turn added after that is missing earlier turns unless it carries its own messages list)."
  - `docs/configuration.md` missing-earlier-turns paragraph: replace the cause list with "They are not when the window or the budget cut them off, when `capture clean` removed them, when they were written flat (an evalshift-sdk older than 0.7.0, or `EVALSHIFT_CAPTURE_LAYOUT=flat`) and promoted on another machine (a fresh checkout never fetches promoted flat captures) or dropped by a per-turn sample rate, when the SDK's `max_captures` / `capture_ttl` evicted them, or when the app numbers `turn_index` from 1 instead of 0." Also update the quoted warning to match Step 1.
  - `docs/configuration.md` "Capture lifecycle", `.evalshift/captures/<suite>/<capture_id>.json`: append "(a conversation's turns in `.evalshift/captures/<suite>/conv_<16 hex>/`, from evalshift-sdk 0.7.0)". In the `capture clean` paragraph, add the D9 sentence from the CHANGELOG.
  - `DOCS.md`, "JSON capture files under `.evalshift/captures/<suite>/cap_<hex>.json`": the same parenthetical. In the bounded-suites paragraph, update the fetch sentence the same way as the `max_cases` bullet. In the `capture clean` sentence, add the D9 rule.
  - `docs/conversations.md`: in the "one capture file per turn" sentences (lines 4 and 14), add "— evalshift-sdk 0.7.0 and later groups them in the conversation's `conv_<16 hex>/` folder". Replace the paragraph's "so after `capture clean` or on a fresh checkout, where promoted turns are not local, only turns carrying their own messages list can still enter" with "the fetch brings a conversation folder's promoted turns along and `capture clean` keeps them while any turn is unpromoted, so this mostly catches turns written flat, cleaned, or evicted by the SDK".
  - `docs/sdk.md`: in the layout diagram (lines 16-17) add the `conv_<16 hex>/` line. The "layout is the interface" paragraph (24-26) names both layouts and the release rule: "evalshift-sdk 0.7.0 needs evalshift CLI 1.6.0 or later". Lines 106-109: the SDK writes a conversation's turns in one folder; `EVALSHIFT_SAMPLE_RATE` samples whole conversations from 0.7.0; `max_captures` keeps the newest 200 captures per suite, evicting whole conversations; `EVALSHIFT_CAPTURE_LAYOUT=flat` restores the old layout.
  - `llms-full.txt`: lines ~33 and ~53 (layout), ~360-370 (fetch rules: valid key shapes, units, a wanted unit brings its promoted turns at any age while unpromoted out-of-window turns stay behind, budget a conversation at a time with in-window leading turns), and ~614-619 (clean). Use the same facts as above in the file's terse style.
  - Docstrings: `captures/__init__.py` lines 5 and 11, `capture.py` line 10 and `config/models.py` ~845 name the two layouts.

- [ ] **Step 4: Verify docs and code together**

Run: `.venv/bin/pre-commit run --all-files` then `make ci`
Expected: both pass. The docs drift tests (`-k "doc or llms or help"`) are part of `make ci`.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md DOCS.md llms-full.txt docs/configuration.md docs/conversations.md docs/sdk.md src/evalshift_cli/captures/__init__.py src/evalshift_cli/cli/commands/capture.py src/evalshift_cli/config/models.py
git commit -m "docs: conversation folders, whole-conversation fetch and clean" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task A6: CLI verification and PR

- [ ] **Step 1: Full gate.** Run `make ci` and `.venv/bin/pre-commit run --all-files`. Expected: all pass, coverage ≥ 90%. Record the pass/skip counts.
- [ ] **Step 2: Offline smoke on testApp data, in a scratch copy only.** testApp itself is never written.
  - Copy testApp's `evalshift.yaml` and `.evalshift/` into `<scratchpad>/conv-smoke/`.
  - Run `.venv/bin/evalshift capture sync --offline --config <scratch>/evalshift.yaml --base <scratch>/.evalshift` twice.
  - Expected: the first run changes nothing versus today (no nested captures exist), and the second leaves every suite file byte-identical (`md5sum -c`).
  - Then move converter's ten `conversation_id` captures into `captures/converter/conv_<sha256("converter:…")[:16]>/` with the Python one-liner below and run `capture sync --offline` again. Expected: the same converter suite, byte-identical.

```bash
python3 -c "import hashlib,json,pathlib,sys; d=pathlib.Path(sys.argv[1]); [p.rename((d/('conv_'+hashlib.sha256(json.loads(p.read_text())['conversation_id'].encode()).hexdigest()[:16])).joinpath(p.name)) if (d/('conv_'+hashlib.sha256(json.loads(p.read_text())['conversation_id'].encode()).hexdigest()[:16])).mkdir(exist_ok=True) is None else None for p in sorted(d.glob('*.json')) if json.loads(p.read_text()).get('conversation_id')]" <scratch>/.evalshift/captures/converter
```

  - Delete the scratch copy afterwards.
- [ ] **Step 3: Code review.** Dispatch one reviewer (model: opus) over `git diff origin/main...HEAD`. Use the Review Focus list and the compatibility matrix as its checklist. Fix every Critical or Important finding, re-run Step 1, and commit.
- [ ] **Step 4: Push and open the PR** against `main`: title `feat(capture): read conversation folders; fetch and clean whole conversations`. The body states that it is CLI-first and safe alone, gives the compatibility matrix, the test plan with counts, and the D10 release order, and ends with the 🤖 line. The user merges it.

---

## Part B — evalshift-sdk (after Part A merges; ships as 0.7.0 after CLI 1.6.0 is on PyPI)

Part B runs in an evalshift-sdk worktree. If this session cannot create one (worktree-isolated harness), ask the user to run:
`! git -C ~/repos/evalshift/evalshift-sdk fetch origin && git -C ~/repos/evalshift/evalshift-sdk worktree add ~/repos/evalshift/evalshift-sdk-wt-conversation-folders -b feat/conversation-folders origin/main`

Then run `uv sync --locked --directory /home/lukas/repos/evalshift/evalshift-sdk-wt-conversation-folders` once. Targeted test runs: `uv run --directory /home/lukas/repos/evalshift/evalshift-sdk-wt-conversation-folders pytest -q <paths>`. The full gate is that repo's CI: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, `uv run pytest`.

### Task B1: Conversation folders in both sinks, with a flat escape hatch

**Files:**
- Create: `src/evalshift/sinks/_layout.py`
- Modify: `src/evalshift/sinks/file.py` (`capture_relpath`, `FileSink.__init__`, `FileSink.write`, new `FileSink.suite_dir`, module docstring)
- Modify: `src/evalshift/sinks/object_store.py` (`ObjectStoreSink.__init__` gains `layout`, `write` key, docstring lines 9-12)
- Modify: `tests/conftest.py` (`read_captures` reads conversation folders)
- Create: `tests/test_conversation_layout.py`

**Interfaces:**
- Produces: `evalshift.sinks._layout.conversation_dir(conversation_id: str) -> str`; `is_conversation_dir(name: str) -> bool`; `resolve_layout(explicit: CaptureLayout | None = None) -> CaptureLayout`; `CaptureLayout = Literal["conversation", "flat"]`; `LAYOUT_ENV = "EVALSHIFT_CAPTURE_LAYOUT"`.
- Produces: `evalshift.sinks.file.capture_relpath(envelope: CaptureEnvelope, *, layout: CaptureLayout) -> str`; `FileSink(base=None, *, layout=None)`; `FileSink.suite_dir(envelope) -> Path`. Task B3 uses `suite_dir` and `is_conversation_dir`.
- Produces: `ObjectStoreSink(store, *, background=True, queue_size=1000, flush_timeout=10.0, layout=None)`.

- [ ] **Step 1: Write the failing tests**, `tests/test_conversation_layout.py`:

```python
"""Conversation folders: where a capture lands, on disk and in a store (0.7.0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, record_model_call
from evalshift.capture.span import SpanTree
from evalshift.sinks._layout import conversation_dir, is_conversation_dir, resolve_layout
from evalshift.sinks.file import FileSink
from evalshift.sinks.object_store import ObjectStoreSink
from evalshift.stores.memory import MemoryStore
from evalshift.trace.models import CaptureEnvelope
from evalshift.trace.serialize import build_capture
from tests.conftest import CaptureReader


def _envelope(conversation_id: str | None = "c1") -> CaptureEnvelope:
    return build_capture(
        SpanTree(),
        suite="support_agent",
        agent_input="hi",
        capture_id="cap_abc",
        conversation_id=conversation_id,
        turn_index=0 if conversation_id is not None else None,
    )


@pytest.mark.parametrize(
    ("conversation_id", "folder"),
    [
        # Pinned in evalshift-cli's tests too: both sides read and write these exact names.
        ("c1", "conv_d0f631ca1ddba8db"),
        ("planner:session-1", "conv_e1961a13f5bd6642"),
        ("", "conv_e3b0c44298fc1c14"),
    ],
)
def test_conversation_dir_is_a_stable_hash_of_the_id(conversation_id: str, folder: str) -> None:
    assert conversation_dir(conversation_id) == folder
    assert is_conversation_dir(folder)


@pytest.mark.parametrize(
    "name", ["nested", "conv_D0F631CA1DDBA8DB", "conv_d0f631ca", "conv_d0f631ca1ddba8db0"]
)
def test_only_the_sdks_own_folder_names_count(name: str) -> None:
    assert not is_conversation_dir(name)


def test_file_sink_files_a_conversation_turn_in_its_folder(tmp_path: Path) -> None:
    path = FileSink(base=tmp_path).write(_envelope())
    folder = tmp_path / "captures" / "support_agent" / "conv_d0f631ca1ddba8db"
    assert path == folder / "cap_abc.json"
    assert path is not None
    assert path.exists()


def test_file_sink_keeps_a_standalone_capture_flat(tmp_path: Path) -> None:
    path = FileSink(base=tmp_path).write(_envelope(None))
    assert path == tmp_path / "captures" / "support_agent" / "cap_abc.json"


def test_object_store_keys_mirror_the_local_layout() -> None:
    store = MemoryStore()
    sink = ObjectStoreSink(store, background=False)
    sink.write(_envelope())
    sink.write(_envelope(None))
    assert set(store.objects) == {
        "captures/support_agent/conv_d0f631ca1ddba8db/cap_abc.json",
        "captures/support_agent/cap_abc.json",
    }


def test_flat_layout_env_restores_the_old_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE_LAYOUT", " FLAT ")
    assert FileSink(base=tmp_path).write(_envelope()) == (
        tmp_path / "captures" / "support_agent" / "cap_abc.json"
    )
    store = MemoryStore()
    ObjectStoreSink(store, background=False).write(_envelope())
    assert set(store.objects) == {"captures/support_agent/cap_abc.json"}


def test_an_explicit_layout_wins_over_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE_LAYOUT", "flat")
    path = FileSink(base=tmp_path, layout="conversation").write(_envelope())
    assert path is not None
    assert path.parent.name == "conv_d0f631ca1ddba8db"


@pytest.mark.parametrize("raw", [None, "", "conversation", "Conversation", "nested", "0"])
def test_anything_but_flat_means_conversation(
    raw: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if raw is None:
        monkeypatch.delenv("EVALSHIFT_CAPTURE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("EVALSHIFT_CAPTURE_LAYOUT", raw)
    assert resolve_layout() == "conversation"


def test_a_session_turn_lands_in_its_conversation_folder(
    capturing: Path, read_captures: CaptureReader
) -> None:
    for turn in range(2):
        with capture.agent_session(
            suite="planner",
            redact=False,
            agent_input=f"turn {turn}",
            conversation_id="planner:session-1",
            turn_index=turn,
            tools=[],
        ):
            record_model_call(model_id="claude-opus-4-8", tools=[])
    folder = capturing / "captures" / "planner" / "conv_e1961a13f5bd6642"
    assert len(list(folder.glob("*.json"))) == 2
    assert sorted(c["turn_index"] for c in read_captures("planner")) == [0, 1]
```

In `tests/conftest.py`, add `monkeypatch.delenv("EVALSHIFT_CAPTURE_LAYOUT", raising=False)` to the autouse `_isolate_config` fixture, so a developer's shell setting can't change test results. Then replace the `read_captures` body's last line with one that reads conversation folders too. Otherwise a sample-out test asserting `== []` would pass even when captures went to a folder:

```python
        paths = sorted(suite_dir.glob("*.json")) + sorted(suite_dir.glob("conv_*/*.json"))
        return [json.loads(p.read_text(encoding="utf-8")) for p in paths]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory /home/lukas/repos/evalshift/evalshift-sdk-wt-conversation-folders pytest -q tests/test_conversation_layout.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'evalshift.sinks._layout'`.

- [ ] **Step 3: Create `src/evalshift/sinks/_layout.py`**

```python
"""Where a capture lives under ``captures/``: in its conversation's folder, or flat.

A capture with a ``conversation_id`` goes to ``<safe_suite>/conv_<16 hex>/<capture_id>.json``,
where the hex is the start of ``sha256(conversation_id)``: fixed length, lowercase, safe on every
filesystem, and never the raw id. Every other capture goes to ``<safe_suite>/<capture_id>.json``.
The layout is the same under ``<base>/captures/`` and under a store's prefix, so a bucket stays a
mirror of ``.evalshift/``. ``EVALSHIFT_CAPTURE_LAYOUT=flat`` (or ``layout="flat"`` on a sink)
writes every capture flat, as before 0.7.0. The evalshift CLI reads both layouts from 1.6.0.

Stdlib only (D-deps).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Literal

CaptureLayout = Literal["conversation", "flat"]

#: Env var selecting the layout; only ``flat`` (any case, stripped) changes it.
LAYOUT_ENV = "EVALSHIFT_CAPTURE_LAYOUT"

_CONVERSATION_DIR = re.compile(r"conv_[0-9a-f]{16}")


def conversation_dir(conversation_id: str) -> str:
    """The folder a conversation's turns live in: ``conv_`` + 16 hex digits of its sha256.

    ``str()`` first: the envelope types the id as ``str``, but nothing enforces it, and a stray
    int must not turn into a dropped capture.
    """
    return "conv_" + hashlib.sha256(str(conversation_id).encode("utf-8")).hexdigest()[:16]


def is_conversation_dir(name: str) -> bool:
    """Whether ``name`` is a folder :func:`conversation_dir` produces."""
    return _CONVERSATION_DIR.fullmatch(name) is not None


def resolve_layout(explicit: CaptureLayout | None = None) -> CaptureLayout:
    """The layout to write: ``explicit`` if given, else ``flat`` when the env var says so.

    Anything other than ``flat`` -- unset, empty, ``conversation`` or a typo -- means
    ``conversation``, the default, so a bad value can never stop captures.
    """
    if explicit is not None:
        return explicit
    return "flat" if os.environ.get(LAYOUT_ENV, "").strip().lower() == "flat" else "conversation"


__all__ = [
    "LAYOUT_ENV",
    "CaptureLayout",
    "conversation_dir",
    "is_conversation_dir",
    "resolve_layout",
]
```

- [ ] **Step 4: Wire `file.py`.** Add the imports `from evalshift.sinks._layout import CaptureLayout, conversation_dir, resolve_layout`, then add after `_safe_segment`:

```python
def capture_relpath(envelope: CaptureEnvelope, *, layout: CaptureLayout) -> str:
    """``<safe_suite>/[conv_<16 hex>/]<capture_id>.json`` -- the same under ``captures/`` on
    disk and in a store (see :mod:`evalshift.sinks._layout`)."""
    suite = _safe_segment(envelope.suite)
    name = capture_filename(envelope.capture_id)
    if layout == "conversation" and envelope.conversation_id is not None:
        return f"{suite}/{conversation_dir(envelope.conversation_id)}/{name}"
    return f"{suite}/{name}"
```

Change `FileSink.__init__` to `def __init__(self, base: str | os.PathLike[str] | None = None, *, layout: CaptureLayout | None = None) -> None:` and store `self._layout = layout`. Add:

```python
    def suite_dir(self, envelope: CaptureEnvelope) -> Path:
        """``<base>/captures/<safe_suite>`` -- what GC caps, whichever folder a write landed in."""
        return (self._resolve_base() / "captures" / _safe_segment(envelope.suite)).absolute()
```

In `write`, replace the `target_dir` / `path` lines with:

```python
        path = (
            self._resolve_base()
            / "captures"
            / capture_relpath(envelope, layout=resolve_layout(self._layout))
        ).absolute()
        target_dir = path.parent
```

Keep the `mkdir` / `write_text` / `OSError` block as it is. Update the module and class docstrings: "Path layout is `<base>/captures/<suite>/<capture_id>.json`, or `<base>/captures/<suite>/conv_<16 hex>/<capture_id>.json` for a conversation's turns (see `_layout`)."

- [ ] **Step 5: Wire `object_store.py`.** Import `capture_relpath` from `evalshift.sinks.file` and `CaptureLayout, resolve_layout` from `evalshift.sinks._layout`. Add `layout: CaptureLayout | None = None` as the last keyword parameter of `__init__` (store `self._layout = layout`). In `write` use:

```python
        key = f"captures/{capture_relpath(envelope, layout=resolve_layout(self._layout))}"
```

Then remove the now-unused imports `from evalshift.sinks.file import _safe_segment` (line 52) and `capture_filename` from `from evalshift.trace.serialize import capture_filename, dumps` (line 56; keep `dumps`). Otherwise ruff fails with F401. Update the docstring at lines 9-12 and the `write` docstring to name both shapes.

- [ ] **Step 6: Run the new tests and the sink tests**

Run: `uv run --directory <sdk-wt> pytest -q tests/test_conversation_layout.py tests/test_filesink.py tests/test_object_store_sink.py tests/test_object_store_sink_background.py tests/test_capture_object_store_routing.py tests/test_config_sink_env.py tests/test_capture_agent.py tests/test_capture_async.py tests/conformance`
Expected: PASS. Existing flat-path assertions use standalone envelopes, so they pass unchanged. If any existing test builds an envelope **with** a `conversation_id` and asserts a flat path, update the assertion to the folder path and name the test in the commit body.

- [ ] **Step 7: Commit**

```bash
git add src/evalshift/sinks/_layout.py src/evalshift/sinks/file.py src/evalshift/sinks/object_store.py tests/conftest.py tests/test_conversation_layout.py
# (object_store.py's unused imports removed in Step 5)
git commit -m "feat(sinks): file a conversation's turns in its own folder" -m "EVALSHIFT_CAPTURE_LAYOUT=flat (or layout=\"flat\") keeps the 0.6 layout. Needs evalshift CLI 1.6.0 or later to read." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task B2: Sample whole conversations

**Files:**
- Modify: `src/evalshift/hygiene/sample.py`, `src/evalshift/config.py` (`should_capture_now`), `src/evalshift/capture/api.py` (lines ~473, ~528, ~848, ~915)
- Test: `tests/test_hygiene_sample.py`, `tests/test_capture_sampling.py`

**Interfaces:**
- Produces: `sample.conversation_draw(key: str) -> float`; `sample.should_capture(rate, *, rng=None, key: str | None = None) -> bool`; `config.should_capture_now(conversation_id: str | None = None) -> bool`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_hygiene_sample.py`, add `import pytest` to the top imports (the file has none) and `conversation_draw` to its existing `from evalshift.hygiene.sample import ...` line; do not import mid-file (ruff E402). Then append:

```python
@pytest.mark.parametrize(
    ("key", "draw"),
    [("c1", 0.5987745500564885), ("planner:session-1", 0.49559010417828814)],
)
def test_a_conversations_draw_is_a_stable_hash(key: str, draw: float) -> None:
    assert conversation_draw(key) == draw


def test_a_conversation_key_ignores_the_random_draw() -> None:
    assert should_capture(0.59, rng=lambda: 0.0, key="c1") is False  # 0.5988 >= 0.59
    assert should_capture(0.60, rng=lambda: 0.99, key="c1") is True  # 0.5988 < 0.60


def test_rate_bounds_still_win_over_a_key() -> None:
    assert should_capture(None, key="c1") is True
    assert should_capture(1.0, key="c1") is True
    assert should_capture(0.0, key="c1") is False
```

(If the file already imports `should_capture` or `pytest`, merge the imports rather than duplicating them.) Append to `tests/test_capture_sampling.py`:

```python
def test_a_conversation_is_sampled_whole(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rate 0.55: "planner:session-1" draws 0.4956 (kept), "c1" draws 0.5988 (dropped) -- on every
    # turn, whatever random.random says.
    configure(sample_rate=0.55)
    draws = iter([0.0, 0.99, 0.0, 0.99, 0.0, 0.99])
    monkeypatch.setattr("random.random", lambda: next(draws))
    for conversation in ("planner:session-1", "c1"):
        for turn in range(3):
            with capture.agent_session(
                suite="s",
                redact=False,
                agent_input=f"{conversation} {turn}",
                conversation_id=conversation,
                turn_index=turn,
                tools=[],
            ):
                record_model_call(model_id="m", tools=[])
    kept = read_captures("s")
    assert {c["conversation_id"] for c in kept} == {"planner:session-1"}
    assert sorted(c["turn_index"] for c in kept) == [0, 1, 2]


def test_the_decorators_static_conversation_id_samples_at_random(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The decorator's id is fixed at decoration time: hashing it would keep every run or none.
    configure(sample_rate=0.55)
    draws = iter([0.0, 0.99, 0.0, 0.99])
    monkeypatch.setattr("random.random", lambda: next(draws))

    @capture.agent(suite="s", redact=False, tools=[], conversation_id="c1")
    def run(question: str) -> str:
        record_model_call(model_id="m", tools=[])
        return question

    for i in range(4):
        run(f"q{i}")
    assert len(read_captures("s")) == 2  # the two runs whose random draw was 0.0
```

Before relying on that decorator call, check `capture.agent`'s real keyword names in `api.py:728-730` and `tests/test_capture_agent.py::test_agent_decorator_accepts_static_conversation_identity`, and match them exactly.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory <sdk-wt> pytest -q tests/test_hygiene_sample.py tests/test_capture_sampling.py`
Expected: FAIL (`ImportError: cannot import name 'conversation_draw'`; the e2e test keeps turns from both conversations).

- [ ] **Step 3: Implement.** In `sample.py`, add `import hashlib`, update the module docstring ("…once, at agent entry. With a conversation id the draw is a hash of it, so a conversation is captured whole or not at all, in every process."), and replace `should_capture` with:

```python
def conversation_draw(key: str) -> float:
    """A stable draw in ``[0, 1)`` for ``key``: 64 bits of ``sha256("evalshift-sample:" + key)``.

    Every turn of a conversation, in every process, gets the same draw, so a sample rate keeps or
    drops a conversation whole.
    """
    digest = hashlib.sha256(f"evalshift-sample:{key}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64


def should_capture(
    rate: float | None, *, rng: Callable[[], float] | None = None, key: str | None = None
) -> bool:
    """True iff this run should be captured under ``rate`` (a probability in ``[0, 1]``).

    ``None`` or ``rate >= 1`` always captures; ``rate <= 0`` never captures. Otherwise, with a
    ``key`` (a conversation id) capture iff :func:`conversation_draw` of it falls below ``rate``;
    without one, iff a uniform draw ``rng()`` does (strict ``<``, so ``rate == draw`` drops).

    ``rng`` defaults to :func:`random.random`, resolved at call time so tests can monkeypatch it.
    """
    if rate is None or rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    if key is not None:
        return conversation_draw(key) < rate
    draw = rng if rng is not None else random.random
    return draw() < rate
```

Add `"conversation_draw"` to `__all__`. In `config.py`:

```python
def should_capture_now(conversation_id: str | None = None) -> bool:
    """Sampling decision for one agent run (read at agent entry). Fail-open: capture on fault.

    A configured ``sample_rate`` decides the draw -- per conversation when ``conversation_id`` is
    known, so its turns are kept or dropped together; a fault in the draw defaults to
    **capturing** so a sampling bug never silently disables all telemetry.
    """
    decision = safety.guard(
        "sample decision",
        lambda: should_capture(_CONFIG.sample_rate, key=conversation_id),
    )
    return decision is not False
```

In `api.py`, change only the two session calls, in `agent_session` (~848) and `agent_session_async` (~915), to `config.should_capture_now(conversation_id)`. Leave `_run_agent` (~473) and `_run_agent_async` (~528), the decorator paths, calling `config.should_capture_now()` with the comment `# the decorator's conversation_id is static: sampled per run (D4)`. Leave `adapters/langchain.py:307` as it is and add the comment `# no conversation id here: sampled per run`.

- [ ] **Step 4: Run the sampling tests and the fail-open tests**

Run: `uv run --directory <sdk-wt> pytest -q tests/test_hygiene_sample.py tests/test_capture_sampling.py tests/test_configure.py tests/test_capture_failopen.py tests/test_capture_async_failopen.py tests/adapters`
Expected: PASS. Existing tests patch `random.random` and use no `conversation_id`, so they pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift/hygiene/sample.py src/evalshift/config.py src/evalshift/capture/api.py src/evalshift/adapters/langchain.py tests/test_hygiene_sample.py tests/test_capture_sampling.py
git commit -m "feat(sampling): an agent_session conversation is kept or dropped whole" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task B3: Local GC evicts whole conversations

**Files:**
- Modify: `src/evalshift/hygiene/gc.py` (`evict`, module docstring), `src/evalshift/sinks/hygiene.py` (the `evict(path.parent, ...)` call)
- Test: `tests/test_hygiene_gc.py`, `tests/test_hygiene_sink.py`

**Interfaces:**
- Consumes: `evalshift.sinks._layout.is_conversation_dir`, `FileSink.suite_dir` (Task B1).
- Produces: `gc.evict(directory, *, max_captures, capture_ttl_seconds, now_ts) -> list[Path]` with the same signature. `directory` is now always a suite dir.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_hygiene_gc.py` (`_make` exists there):

```python
CONV_A = "conv_" + "a" * 16
CONV_B = "conv_" + "b" * 16


def _turns(directory: Path, folder: str, *mtimes: float) -> list[Path]:
    conv = directory / folder
    conv.mkdir(parents=True, exist_ok=True)
    return [_make(conv, f"cap_{folder[-2:]}_{i}.json", mtime=m) for i, m in enumerate(mtimes)]


def test_an_older_conversation_is_evicted_whole(tmp_path: Path) -> None:
    old = _turns(tmp_path, CONV_A, 100.0, 101.0, 102.0)
    new = _turns(tmp_path, CONV_B, 200.0, 201.0)
    removed = gc.evict(tmp_path, max_captures=4, capture_ttl_seconds=None, now_ts=1000.0)
    assert set(removed) == set(old)  # 2 + 3 > 4: the older conversation goes, not single turns
    assert not (tmp_path / CONV_A).exists()
    assert all(p.exists() for p in new)


def test_a_conversation_larger_than_the_cap_keeps_only_its_newest_turns(tmp_path: Path) -> None:
    # A conversation that never ends (a static decorator id, a user id): disk stays bounded.
    live = _turns(tmp_path, CONV_A, 300.0, 301.0, 302.0)
    older = _make(tmp_path, "cap_flat.json", mtime=100.0)
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [older, live[0]]
    assert live[1].exists() and live[2].exists()


def test_ttl_evicts_every_old_file_so_an_idle_conversation_goes_whole(tmp_path: Path) -> None:
    idle = _turns(tmp_path, CONV_A, 100.0, 110.0)
    active = _turns(tmp_path, CONV_B, 100.0, 990.0)  # first turn old, latest recent
    removed = gc.evict(tmp_path, max_captures=None, capture_ttl_seconds=500.0, now_ts=1000.0)
    assert set(removed) == {*idle, active[0]}
    assert not (tmp_path / CONV_A).exists()
    assert active[1].exists()


def test_flat_and_conversation_units_share_one_cap_oldest_first(tmp_path: Path) -> None:
    flat_old = _make(tmp_path, "cap_old.json", mtime=50.0)
    conv = _turns(tmp_path, CONV_A, 100.0, 150.0)
    flat_new = _make(tmp_path, "cap_new.json", mtime=200.0)
    removed = gc.evict(tmp_path, max_captures=3, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [flat_old]
    assert flat_new.exists() and all(p.exists() for p in conv)


def test_a_folder_that_is_not_a_conversation_is_never_touched(tmp_path: Path) -> None:
    other = tmp_path / "nested"
    other.mkdir()
    keep = _make(other, "cap_x.json", mtime=1.0)
    _make(tmp_path, "cap_new.json", mtime=200.0)
    gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=1.0, now_ts=1000.0)
    assert keep.exists()
```

Append to `tests/test_hygiene_sink.py` (use that file's existing envelope helper if there is one, else `build_capture(SpanTree(), ...)` as in Task B1):

```python
def test_gc_caps_the_suite_across_conversation_folders(tmp_path: Path) -> None:
    sink = HygieneSink(FileSink(base=tmp_path), dedup=False, max_captures=2, capture_ttl=None)
    for i, conversation in enumerate(("c1", "c2", "c3")):
        path = sink.write(
            build_capture(
                SpanTree(),
                suite="s",
                agent_input=f"q{i}",
                capture_id=f"cap_{i}",
                conversation_id=conversation,
                turn_index=0,
            )
        )
        assert path is not None
        # Deterministic mtimes, oldest first: the next write's GC sees this one as older.
        os.utime(path, (100.0 + i, 100.0 + i))
    suite_dir = tmp_path / "captures" / "s"
    assert sorted(p.name for p in suite_dir.glob("conv_*/*.json")) == ["cap_1.json", "cap_2.json"]
    assert len(list(suite_dir.iterdir())) == 2  # the evicted conversation's folder is gone
```

(Add `import os`, `HygieneSink`, `FileSink`, `SpanTree` and `build_capture` imports if the file lacks them.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --directory <sdk-wt> pytest -q tests/test_hygiene_gc.py tests/test_hygiene_sink.py`
Expected: FAIL. `evict` never looks inside folders, and the sink runs GC on the conversation folder. `test_a_folder_that_is_not_a_conversation_is_never_touched` passes before and after: it is a guard.

- [ ] **Step 3: Implement `gc.py`.** Replace the module docstring's first paragraph with: ":func:`evict` runs after a successful disk write, on the **suite** directory just written to. It caps it by **unit**: a flat `*.json` capture, or a whole conversation folder (`conv_<16 hex>/`, see `evalshift.sinks._layout`) timed by its newest file. The rest of the docstring stays, adapted from files to units. Then replace `evict` with:

```python
@dataclass(frozen=True, slots=True)
class _Unit:
    """What GC counts together: one flat capture, or a conversation folder.

    ``files`` are ``(mtime, path)`` pairs, oldest first; the unit's time is its newest file's.
    """

    files: tuple[tuple[float, Path], ...]
    folder: Path | None

    @property
    def newest(self) -> float:
        return self.files[-1][0]


def _scan(directory: Path) -> list[_Unit]:
    units: list[_Unit] = []
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith(".json"):
            units.append(_Unit(((entry.stat().st_mtime, Path(entry.path)),), None))
        elif entry.is_dir() and is_conversation_dir(entry.name):
            files = sorted(
                (turn.stat().st_mtime, Path(turn.path))
                for turn in os.scandir(entry.path)
                if turn.is_file() and turn.name.endswith(".json")
            )
            if files:
                units.append(_Unit(tuple(files), Path(entry.path)))
    return sorted(units, key=lambda u: (u.newest, u.files[0][1]))  # oldest first


def evict(
    directory: Path,
    *,
    max_captures: int | None,
    capture_ttl_seconds: float | None,
    now_ts: float,
) -> list[Path]:
    """Evict aged / overflow captures from the suite ``directory``; return the files removed.

    Count rule, newest unit first: the newest unit keeps its newest ``max_captures`` files (all
    of a normal conversation; the latest turns of one that never ends), then older units are
    kept whole while the count stays within ``max_captures`` -- the first that does not fit
    goes, with everything older. TTL rule: every file older than ``capture_ttl_seconds`` goes,
    so an idle conversation goes whole and a live one loses only its old turns. A folder left
    empty is removed. No-op (returns ``[]``) when neither threshold is set or the directory is
    unreadable; never raises into the caller.
    """
    if max_captures is None and capture_ttl_seconds is None:
        return []
    try:
        units = _scan(directory)
    except OSError:
        logger.debug("evalshift: gc scan of %s failed (skipped)", directory, exc_info=True)
        return []

    doomed: set[Path] = set()
    if capture_ttl_seconds is not None:
        doomed.update(
            path
            for unit in units
            for mtime, path in unit.files
            if now_ts - mtime > capture_ttl_seconds
        )
    if max_captures is not None:
        kept = 0
        full = False
        for index in range(len(units) - 1, -1, -1):  # newest first
            files = units[index].files
            if index == len(units) - 1:
                cut = max(len(files) - max_captures, 0)  # never-ending: keep the newest turns
                doomed.update(path for _, path in files[:cut])
                kept = len(files) - cut
            elif not full and kept + len(files) <= max_captures:
                kept += len(files)
            else:
                full = True
                doomed.update(path for _, path in files)

    removed: list[Path] = []
    for unit in units:  # oldest-first, for a stable removal order
        hit = False
        for _, path in unit.files:
            if path not in doomed:
                continue
            hit = True
            try:
                path.unlink(missing_ok=True)
                removed.append(path)
            except OSError:
                logger.debug("evalshift: gc unlink of %s failed (skipped)", path, exc_info=True)
        if hit and unit.folder is not None:
            try:
                unit.folder.rmdir()  # fails, harmlessly, while turns remain
            except OSError:
                logger.debug("evalshift: gc kept %s (not empty)", unit.folder)
    return removed
```

Add the imports `from dataclasses import dataclass` and `from evalshift.sinks._layout import is_conversation_dir`. `_layout` is stdlib-only and imports nothing from `hygiene`, so there is no cycle; confirm with `uv run python -c "import evalshift"`.

- [ ] **Step 4: Point the sink's GC at the suite dir.** In `sinks/hygiene.py`, replace `capture_gc.evict(path.parent, ...)` with:

```python
                suite_dir = getattr(self._base, "suite_dir", None)
                if callable(suite_dir):
                    directory = suite_dir(envelope)
                elif is_conversation_dir(path.parent.name):
                    directory = path.parent.parent  # a custom sink that wrote a folder
                else:
                    directory = path.parent
                capture_gc.evict(
                    directory,
                    max_captures=self._max_captures,
                    capture_ttl_seconds=self._capture_ttl,
                    now_ts=self._now(),
                )
```

A custom sink without `suite_dir` gets the suite as the parent of a conversation folder, or its write's parent otherwise (today's behaviour). Add `from evalshift.sinks._layout import is_conversation_dir` to `sinks/hygiene.py`.

- [ ] **Step 5: Run the GC tests, the hygiene e2e tests and the fail-open tests**

Run: `uv run --directory <sdk-wt> pytest -q tests/test_hygiene_gc.py tests/test_hygiene_sink.py tests/test_capture_hygiene_e2e.py tests/test_capture_failopen.py tests/test_capture_async_failopen.py`
Expected: PASS. Every existing flat-only GC test must pass unchanged: flat files are units of one, so the old keep-newest-N result holds. `test_hygiene_gc.py:78-100` ("does not recurse") must still pass. If its subfolder name happens to match `conv_[0-9a-f]{16}`, rename the folder in the test, not the rule.

- [ ] **Step 6: Commit**

```bash
git add src/evalshift/hygiene/gc.py src/evalshift/sinks/hygiene.py tests/test_hygiene_gc.py tests/test_hygiene_sink.py
git commit -m "feat(gc): evict whole conversations, never single turns" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task B4: SDK docs and changelog

**Files:**
- Modify: `CHANGELOG.md`, `README.md` (:6, :85-105), `DOCS.md` (:111, :197, :584, :593, :610-614, :668, :866, :1076), `llms-full.txt` (:14, :17, :31, :55, :231, :247, :420-422, :447, :452), `llms.txt` (:3, :11), `docs/SCHEMA.md` (:54-56, :108-119), `docs/DECISIONS.md`, `examples/support_agent/README.md` (:25), `src/evalshift/hygiene/__init__.py` (the stale "off by default" docstring at :3-4)

- [ ] **Step 1: CHANGELOG** under `## [Unreleased]`:

```markdown
### Changed

- **Breaking: conversation folders.** A capture with a `conversation_id` is
  written to `captures/<suite>/conv_<16 hex>/cap_<hex>.json`, on disk and in
  an object store. The hex is the start of `sha256(conversation_id)`.
  Standalone captures stay at `captures/<suite>/cap_<hex>.json`. **Upgrade
  the evalshift CLI to 1.6.0 or later first**: older CLIs do not see the
  folders, and their `capture clean` can sweep toolset sidecars that only
  folder captures use. Code that globs `captures/<suite>/*.json` must also
  read `captures/<suite>/conv_*/*.json`. Set `EVALSHIFT_CAPTURE_LAYOUT=flat`
  (or pass `layout="flat"` to `FileSink` / `ObjectStoreSink`) to keep the 0.6
  layout.
- **Sampling keeps conversations whole.** With `EVALSHIFT_SAMPLE_RATE` set,
  a `capture.agent_session` given a `conversation_id` is sampled by a hash of
  that id, so every turn of a conversation is captured or none is, in every
  process. The `@capture.agent` decorator (whose `conversation_id` is fixed
  when it is applied), runs without an id, and the LangChain handler still
  draw at random.
- **`max_captures` / `capture_ttl` count conversations.** Older
  conversations are evicted whole rather than turn by turn. The newest one
  keeps its newest `max_captures` turns, so a conversation that never ends
  (a static decorator id) stays bounded. `capture_ttl` still evicts every
  capture older than it, and folders left empty are removed. Suites with
  only standalone captures behave as before. Dedup is unchanged: its hash
  already includes the conversation and turn.
```

- [ ] **Step 2: Reference docs.** At each listed location, replace the flat-only layout statement with the D1 layout and add the D3 escape hatch. Restate sampling (D4: "decided once per agent invocation; per conversation when a `conversation_id` is given") and GC (D5: "per suite dir, by unit: a capture or a whole conversation folder"). Every changed sentence must say the same thing in every file.
  - `docs/DECISIONS.md`: add an entry **D-conv-layout (2026-10-20)** with four paragraphs:
    - the problem: per-turn files scatter conversations, causing sampling gaps, GC gaps and budget cuts in the CLI;
    - the decision: D1-D6, and why not one mutable file per conversation (object stores can't append, turns of one conversation land on different processes, read-modify-write races, cost grows with the square of conversation length);
    - the trade-offs: the D5 bound on never-ending conversations; the decorator's static id staying random (D4); and the rare race where GC or `capture clean` removes an emptied folder just as another process writes a new turn into it, so that one capture is dropped, fail-open;
    - the compatibility rule: D10 and the escape hatch. Amend D-6, D-6a, the toolset GC note and D-stores in place, each with a pointer to the new entry.
  - Fix `src/evalshift/hygiene/__init__.py`'s stale "off by default" (dedup and `max_captures=200` are on by default).
- [ ] **Step 3: Gate.** Run `uv run --directory <sdk-wt> ruff check`, `ruff format --check`, `mypy`, `pytest`. Expected: all pass.
- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md DOCS.md llms-full.txt llms.txt docs/SCHEMA.md docs/DECISIONS.md examples/support_agent/README.md src/evalshift/hygiene/__init__.py
git commit -m "docs: conversation folders, per-conversation sampling and GC" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task B5: SDK verification and PR

- [ ] **Step 1:** Run the full SDK gate (`ruff check`, `ruff format --check`, `mypy`, `pytest`). Record the counts.
- [ ] **Step 2: Code review.** Dispatch one reviewer (model: opus) over the SDK branch diff. Its checklist: the Review Focus items 1-5 and the fail-open rule (no new code path can raise into a host). Also check that `_layout.py` imports nothing from `config`, `capture` or `hygiene` (no import cycle). Fix every Critical or Important finding.
- [ ] **Step 3:** Push and open the SDK PR, titled `feat: conversation folders, per-conversation sampling and GC`. The body names the CLI ≥ 1.6.0 requirement in its first line and links the CLI PR. Do not merge before CLI 1.6.0 is on PyPI.

---

## Part C — Integration and rollout

### Task C1: Cross-repo smoke (SDK branch writes, CLI branch reads)

No committed code. Both branches must be checked out.

- [ ] **Step 1: A scratch environment with both branches.** Use the session scratchpad, never either repo's venv:

```bash
uv venv <scratch>/xrepo --python 3.11
uv pip install --python <scratch>/xrepo/bin/python -e /home/lukas/repos/evalshift/evalshift-cli/.claude/worktrees/bounded-capture-suites -e /home/lukas/repos/evalshift/evalshift-sdk-wt-conversation-folders
```

- [ ] **Step 2: Write `<scratch>/xrepo_smoke.py`** (outside both repos):

```python
"""SDK branch writes a 3-turn conversation and a standalone run; CLI branch syncs them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

base = Path(tempfile.mkdtemp(prefix="xrepo-"))
os.environ["EVALSHIFT_CAPTURE"] = "1"
os.environ["EVALSHIFT_DIR"] = str(base)

from evalshift import capture, record_model_call  # noqa: E402

for turn, text in enumerate(["plan my monday", "move lunch to 1pm", "add a gym slot"]):
    with capture.agent_session(
        suite="planner",
        redact=False,
        agent_input=text,
        conversation_id="planner:session-1",
        turn_index=turn,
        tools=[],
    ):
        record_model_call(model_id="m", tools=[], input=text, output=f"ok {turn}")
with capture.agent_session(suite="planner", redact=False, agent_input="standalone", tools=[]):
    record_model_call(model_id="m", tools=[], input="standalone", output="ok")

folder = base / "captures" / "planner" / "conv_e1961a13f5bd6642"
assert len(list(folder.glob("*.json"))) == 3, list(base.rglob("*.json"))
assert len(list((base / "captures" / "planner").glob("*.json"))) == 1

from evalshift_cli.cli.commands.init import CAPTURES_SCAFFOLD, render_minimal_config  # noqa: E402

# The CLI tests' own minimal config, without the capture limits: unbounded sync promotes all four.
config = render_minimal_config(profile="model-upgrade").replace(CAPTURES_SCAFFOLD + "\n", "")
(base / "evalshift.yaml").write_text(config, encoding="utf-8")
evalshift = Path(sys.executable).with_name("evalshift")
out = subprocess.run(
    [str(evalshift), "capture", "sync", "--offline", "--config", str(base / "evalshift.yaml"),
     "--base", str(base)],
    capture_output=True, text=True, check=True,
)
print(out.stdout)
cases = {p.stem: json.loads(p.read_text()) for p in (base / "suites" / "planner").glob("cap_*.json")}
assert len(cases) == 4, sorted(cases)
last = max(cases.values(), key=lambda c: -1 if c.get("turn_index") is None else c["turn_index"])
assert [m["content"] for m in last["example"]["history"] if m["role"] == "user"] == [
    "plan my monday",
    "move lunch to 1pm",
]
print("OK", base)
```

- [ ] **Step 3: Run it** with `<scratch>/xrepo/bin/python <scratch>/xrepo_smoke.py`. Expected: `OK <path>`.
- [ ] **Step 4: Bucket shape.** Write `<scratch>/xrepo_bucket.py` and run it with `<scratch>/xrepo/bin/python <scratch>/xrepo_bucket.py`; the harness refuses an env-prefixed command. Expected: `OK`.

```python
"""The SDK's store keys, fetched by the CLI, rebuild exactly the SDK's FileSink tree."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# The CLI's tests package (for FakeRemoteStore); set here, not via PYTHONPATH.
sys.path.insert(0, "/home/lukas/repos/evalshift/evalshift-cli/.claude/worktrees/bounded-capture-suites")

from evalshift.capture.span import SpanTree  # noqa: E402
from evalshift.sinks.file import FileSink
from evalshift.sinks.object_store import ObjectStoreSink
from evalshift.stores.memory import MemoryStore
from evalshift.trace.serialize import build_capture
from evalshift_cli.captures.remote import fetch_captures
from tests.unit.fake_remote_store import FakeRemoteStore

envelopes = [
    build_capture(
        SpanTree(),
        suite=suite,
        agent_input=f"q{i}",
        capture_id=f"cap_{i}",
        conversation_id=conversation,
        turn_index=None if conversation is None else i,
    )
    for i, (suite, conversation) in enumerate(
        [
            ("planner", "planner:session-1"),
            ("planner", "planner:session-1"),
            ("planner", None),
            ("team/support", "c1"),
        ]
    )
]
store = MemoryStore()
remote_sink = ObjectStoreSink(store, background=False)
local = Path(tempfile.mkdtemp(prefix="xrepo-local-"))
local_sink = FileSink(base=local)
for envelope in envelopes:
    remote_sink.write(envelope)
    local_sink.write(envelope)

mirror = Path(tempfile.mkdtemp(prefix="xrepo-mirror-"))
summary = fetch_captures(FakeRemoteStore(dict(store.objects)), base=mirror)


def tree(root: Path) -> list[Path]:
    return sorted(p.relative_to(root) for p in (root / "captures").rglob("*.json"))


assert summary.skipped_malformed == 0, summary
assert tree(local) == tree(mirror), (tree(local), tree(mirror))
print("OK", tree(mirror))
```
- [ ] **Step 5:** Delete `<scratch>/xrepo*` and the temp dirs. Report the outputs in the PR threads.

### Task C2: Release and rollout (checklist; each step is a separate, user-merged PR)

- [ ] **CLI 1.6.0.**
  - After the Part A PR merges: a `chore/release-1.6.0` PR bumping `pyproject.toml`'s version and dating the `[Unreleased]` CHANGELOG section. It covers #57, #58 and Part A.
  - After the user merges it: tag `v1.6.0`, push the tag, and watch `release.yml` publish to PyPI.
  - Website PR in evalshift-client (in its own worktree, per memory: the main checkout is shared):
    - bump `src/lib/version.ts` `CLI_VERSION` and the `HostedSetup.tsx` pin;
    - add a `Changelog.tsx` line;
    - regenerate `public/cli-llms-full.txt` from `git show origin/main:llms-full.txt`;
    - update `Captures.tsx:85` and `Configuration.tsx:1351/1365`.
- [ ] **evalshift-action:** after CLI 1.6.0 is on PyPI, a PR bumping the default `evalshift-version` (`action.yml:39`, plus README.md:182, DOCS.md:11/:260 and llms-full.txt:82) to `1.6.0`. CI users who adopt the `captures:` keys from #57 need it. The user merges it.
- [ ] **SDK 0.7.0** (only after CLI 1.6.0 is on PyPI).
  - The user merges the Part B PR. Then open a `chore/release-0.7.0` PR bumping **both** `pyproject.toml` and `__version__` in `src/evalshift/__init__.py:32`. After the user merges it, tag `v0.7.0` and watch the release.
  - Before tagging, check the SDK's PyPI trusted publisher names the lowercase `evalshift/...` org (memory: it broke after the org move).
  - Website PR:
    - `SDK_VERSION`;
    - `public/sdk-llms-full.txt` (header `version: 0.7.0`; `scripts/llmsFullSync.test.ts` enforces it);
    - `SdkConfig.tsx` (:9, :12, :18, :78, :138-147, :271, :294-295, :320-343), `Sdk.tsx:22`, `GettingStarted.tsx:24`, `Workflow.tsx:19,36`, `SdkCapture.tsx:264,276`;
    - the blog post `build-a-golden-suite-from-production-traffic.md:17,63-66`, and `glossary/data/terms.ts:33`.
- [ ] **testApp (the user's app; propose, don't push):**
  - `tests/test_capture.py:30` globs `captures/<suite>/cap_*.json` and must also read `conv_*/cap_*.json` once testApp moves to SDK 0.7.0;
  - bump `.github/workflows/evalshift.yml`'s `evalshift-version` to `1.6.0` before committing the `captures.max_age` / `max_cases` keys;
  - bump the SDK to 0.7.0 last.
- [ ] **Rollback lever:** if anything goes wrong after SDK 0.7.0, set `EVALSHIFT_CAPTURE_LAYOUT=flat` on the agent. CLI 1.6.0 reads both layouts, so nothing already written needs moving.
