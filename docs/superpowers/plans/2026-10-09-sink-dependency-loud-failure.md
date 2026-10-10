# Sink Dependency Loud Failure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A misconfigured `EVALSHIFT_SINK` (client library missing, URI malformed) stops the agent at startup with one exception that says `pip install boto3`, instead of logging a warning and writing captures to local disk; the CLI names packages the same way and `doctor` stops failing CI over it.

**Architecture:** The SDK records the problem when its process-wide config is built at import, raises it from `config.require_sink_ready()` at every explicit touch point (the `capture.*` decorators, the client wrappers, the LangChain handler, `configure()` without a sink) while `EVALSHIFT_CAPTURE` is truthy, and makes `is_capture_enabled()` return `False` (one warning) as a backstop if the gate turns on later. The CLI changes only messages and the `doctor` row status. Both repos keep the store-URI grammar module as the shared, verbatim home of the scheme → package tables.

**Tech Stack:** Python 3.10+ (SDK, stdlib-only, `uv run pytest` / `ruff` / `mypy`), Python 3.11+ (CLI, `pytest` / `ruff` / `mypy --strict` via `.venv`), Typer, Rich, pytest `monkeypatch` + `caplog`.

**Spec:** `docs/superpowers/plans/../specs/2026-10-09-sink-dependency-loud-failure-design.md` (this repo: `evalshift-cli/docs/superpowers/specs/2026-10-09-sink-dependency-loud-failure-design.md`)

## Global Constraints

- SDK stays stdlib-only: `dependencies = []` in `evalshift-sdk/pyproject.toml`; cloud clients are imported lazily and only checked with `importlib.util.find_spec`.
- The extras `[s3]`, `[gcs]`, `[azure]` stay in both `pyproject.toml` files; nothing user-facing (messages, docs) mentions them again.
- No raised or logged text ever contains the raw `EVALSHIFT_SINK` / `captures.store` value; grammar errors are raised `from None`.
- Nothing raises, logs or changes behaviour unless `EVALSHIFT_CAPTURE` is truthy (`1`/`true`/`yes`/`on`, case-insensitive). Bare `import evalshift` never raises.
- Scheme → module → package tables are identical in `evalshift-sdk/src/evalshift/stores/uri.py` and `evalshift-cli/src/evalshift_cli/captures/store_uri.py`.
- Messages: SDK missing library → `EVALSHIFT_SINK points at an object store (s3://), but boto3 is not installed. Run: pip install boto3, or unset EVALSHIFT_SINK to capture to local disk.`; CLI → summary `s3:// captures store needs boto3, which is not installed`, hint `install it: pip install boto3`. Azure names both packages: `pip install azure-storage-blob azure-identity`.
- Gates: SDK `uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest`; CLI `ruff check . && ruff format --check . && mypy --strict src/evalshift_cli && pytest` with `PATH=$PWD/.venv/bin:$PATH` (the commit and push hooks need it).
- Conventional Commits; every commit message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Run `git` with `-C <repo>` or from inside the repo; never run two repos' git commands in parallel shells.
- Versions: SDK 0.6.0, CLI 1.3.1, bumped in the release task only (not in feature commits).

## Review Focus

1. `EVALSHIFT_SINK=ftp://x` with `EVALSHIFT_CAPTURE` unset: decorating, wrapping, `configure()` and `is_capture_enabled()` must neither raise nor log — Task 4 `test_gate_off_never_raises_or_warns`.
2. A malformed value carrying a secret (`az://acct?sig=SECRET`, an Azure connection string): the raised exception's full formatted traceback must not contain it — Task 3 `test_recorded_error_never_leaks_a_secret_bearing_value`.
3. The CLI process (`evalshift doctor` imports `evalshift`) with both env vars exported in the shell and no boto3: must still exit normally — Task 5 `test_import_with_a_broken_sink_exits_zero_and_logs_nothing`.
4. Azure with `azure-storage-blob` installed but not `azure-identity`: the message names both packages in one `pip install` — Task 1 `test_open_store_azure_names_both_packages_when_identity_is_missing`.
5. Gate turned on after every touch point ran (env set mid-process): no capture file on disk, one warning, agent output unchanged — Task 3 `test_gate_turned_on_late_drops_captures_instead_of_writing_to_disk`.

---

# Part A — evalshift-sdk (0.6.0)

Work in `/home/lukas/repos/evalshift/evalshift-sdk` on branch `feat/sink-loud-failure` from `main` (`git -C /home/lukas/repos/evalshift/evalshift-sdk checkout -b feat/sink-loud-failure main`). All paths below are relative to that repo.

### Task 1: Package tables and `MissingStoreDependencyError` in the URI grammar

**Files:**
- Modify: `src/evalshift/stores/uri.py` (lines 29, 50-53, 96-115, 118-140, 143-150)
- Modify: `tests/test_stores_adapters.py:18,141-156`
- Grep-and-fix: every `MissingExtraError` in `src/`, `tests/`, `DOCS.md`, `llms.txt`, `llms-full.txt` (`git grep -n MissingExtraError`)

**Interfaces:**
- Produces: `_PACKAGE_FOR_MODULE: dict[str, str]`, `_PACKAGES_FOR_SCHEME: dict[str, str]`, `StoreURI.packages -> str`, `class MissingStoreDependencyError(ImportError)` with attributes `scheme`, `module`, `package`, `packages`, `extra`, and `require_store_modules(scheme: Scheme) -> None`. `open_store(uri)` raises `MissingStoreDependencyError` (was `MissingExtraError`).

- [ ] **Step 1: Write the failing tests**

Replace `test_open_store_missing_extra_names_the_pip_extra` and `test_open_store_azure_needs_azure_identity_too` in `tests/test_stores_adapters.py` (the file already imports `from evalshift.stores import uri as uri_module` and patches `uri_module._installed`; keep that) with:

```python
def test_open_store_missing_library_names_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uri_module, "_installed", lambda module: False)
    with pytest.raises(MissingStoreDependencyError) as info:
        open_store("gs://b/p")
    err = info.value
    assert str(err) == (
        "gs:// store needs google-cloud-storage, which is not installed; "
        "run: pip install google-cloud-storage"
    )
    assert (err.scheme, err.module, err.package, err.packages, err.extra) == (
        "gs", "google.cloud.storage", "google-cloud-storage", "google-cloud-storage", "gcs"
    )
    assert "[gcs]" not in str(err)


def test_open_store_azure_names_both_packages_when_identity_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # azure-storage-blob alone is not enough: DefaultAzureCredential lives in azure-identity,
    # and the first put would fail in the background. One pip command installs both.
    monkeypatch.setattr(uri_module, "_installed", lambda module: module != "azure.identity")
    with pytest.raises(MissingStoreDependencyError) as info:
        open_store("az://a/c/p")
    assert info.value.package == "azure-identity"
    assert str(info.value).endswith("run: pip install azure-storage-blob azure-identity")


def test_require_store_modules_is_a_no_op_when_everything_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uri_module, "_installed", lambda module: True)
    require_store_modules("az")


def test_store_uri_packages_property() -> None:
    assert parse_store_uri("s3://b/p").packages == "boto3"
    assert parse_store_uri("gs://b/p").packages == "google-cloud-storage"
    assert parse_store_uri("az://a/c/p").packages == "azure-storage-blob azure-identity"
```

Update the import at line 18 to `from evalshift.stores.uri import MissingStoreDependencyError, open_store, parse_store_uri, require_store_modules`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/lukas/repos/evalshift/evalshift-sdk && uv run pytest tests/test_stores_adapters.py -q`
Expected: ImportError on `MissingStoreDependencyError`.

- [ ] **Step 3: Implement**

In `src/evalshift/stores/uri.py`, after `_EXTRA_FOR_SCHEME` (line 29) add:

```python
#: The pip distribution that provides each client module. Shared verbatim with the CLI.
_PACKAGE_FOR_MODULE: dict[str, str] = {
    "boto3": "boto3",
    "google.cloud.storage": "google-cloud-storage",
    "azure.storage.blob": "azure-storage-blob",
    "azure.identity": "azure-identity",
}

#: The ``pip install`` argument that installs everything a scheme needs, in one command.
#: Named in every message instead of the extras: a user who installed the SDK the normal way
#: has no reason to know what an extra is. Shared verbatim with the CLI.
_PACKAGES_FOR_SCHEME: dict[str, str] = {
    "s3": "boto3",
    "gs": "google-cloud-storage",
    "az": "azure-storage-blob azure-identity",
}
```

Add to `StoreURI` after the `extra` property:

```python
    @property
    def packages(self) -> str:
        """The ``pip install`` argument that installs this scheme's client library (or libraries)."""
        return _PACKAGES_FOR_SCHEME[self.scheme]
```

Replace `class MissingExtraError` (lines 107-115) with:

```python
class MissingStoreDependencyError(ImportError):
    """The client library for a store scheme is not installed.

    Attributes:
        scheme: ``"s3"``, ``"gs"`` or ``"az"``.
        module: The first module that could not be found.
        package: The pip distribution that provides ``module``.
        packages: The ``pip install`` argument that installs everything the scheme needs.
        extra: The pip extra that pins the same packages, for callers that prefer it.

    The message names ``package`` and ``packages`` only -- never an extra, never a URI.
    """

    def __init__(self, scheme: Scheme, module: str) -> None:
        self.scheme = scheme
        self.module = module
        self.package = _PACKAGE_FOR_MODULE[module]
        self.packages = _PACKAGES_FOR_SCHEME[scheme]
        self.extra = _EXTRA_FOR_SCHEME[scheme]
        super().__init__(
            f"{scheme}:// store needs {self.package}, which is not installed; "
            f"run: pip install {self.packages}"
        )
```

Replace the module-check loop at the top of `open_store` with a call to a new helper placed right after `_installed`:

```python
def require_store_modules(scheme: Scheme) -> None:
    """Raise :class:`MissingStoreDependencyError` unless every module ``scheme`` needs is importable.

    Uses ``importlib.util.find_spec`` only, so nothing is imported and nothing touches the
    network: safe at process start and at store construction.
    """
    for module in _REQUIRED_MODULES[scheme]:
        if not _installed(module):
            raise MissingStoreDependencyError(scheme, module)
```

and in `open_store`:

```python
    parsed = parse_store_uri(uri)
    require_store_modules(parsed.scheme)
```

Update `open_store`'s docstring (`MissingStoreDependencyError: when a module the scheme's adapter needs is not installed; it names the first missing one and the package to install.`), the `_REQUIRED_MODULES` comment (`-- both ship in the ``[azure]`` extra` → `-- installed together by ``pip install azure-storage-blob azure-identity``), and `__all__`: remove `"MissingExtraError"`, add `"MissingStoreDependencyError"` and `"require_store_modules"`. Run `git grep -n MissingExtraError` and rename every remaining reference in `src/` and `tests/` (docs are rewritten in Task 6; leave them for now).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_stores_adapters.py tests/test_stores_uri.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift/stores/uri.py tests/test_stores_adapters.py
git commit -m "feat(stores): name the package to install, not the extra

MissingExtraError -> MissingStoreDependencyError; its message and new
.package/.packages attributes say 'pip install boto3' (both Azure packages
in one command). The scheme tables are shared verbatim with the CLI.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 2: Store constructors check their library at construction

**Files:**
- Modify: `src/evalshift/stores/s3.py:1-9,18-31`, `src/evalshift/stores/gcs.py` (docstring, `__init__`, line 29), `src/evalshift/stores/azure.py` (docstring, `__init__`, lines 31-32)
- Test: `tests/test_stores_adapters.py`

**Interfaces:**
- Consumes: `require_store_modules` from Task 1.
- Produces: `S3Store(bucket, prefix="", *, client=None)`, `GCSStore(...)`, `AzureBlobStore(account, container, prefix="", *, client=None)` raise `MissingStoreDependencyError` at construction when `client is None` and the library is absent. Behaviour with an injected `client` is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stores_adapters.py`:

```python
@pytest.mark.parametrize(
    ("build", "package"),
    [
        (lambda: S3Store("b", "p"), "boto3"),
        (lambda: GCSStore("b", "p"), "google-cloud-storage"),
        (lambda: AzureBlobStore("a", "c", "p"), "azure-storage-blob azure-identity"),
    ],
)
def test_store_without_a_client_requires_its_library_at_construction(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[], object], package: str
) -> None:
    # The client is still built lazily on the first put (a background thread, where a raise
    # would only be a logged failure); the *presence* of the library is checked here, where
    # user code constructs the store and a raise is loud.
    monkeypatch.setattr(uri_module, "_installed", lambda module: False)
    with pytest.raises(MissingStoreDependencyError, match=rf"pip install {package}$"):
        build()


def test_store_with_an_injected_client_needs_no_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uri_module, "_installed", lambda module: False)
    S3Store("b", "p", client=object())
    GCSStore("b", "p", client=object())
    AzureBlobStore("a", "c", "p", client=object())
```

Add `from collections.abc import Callable` to the test imports if missing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_stores_adapters.py -q -k "at_construction or injected_client"`
Expected: the first test FAILS (no exception raised).

- [ ] **Step 3: Implement**

`src/evalshift/stores/s3.py`: change the module docstring's first line to `"""Amazon S3 (and S3-compatible) ``ObjectStore``. Needs ``boto3`` (``pip install boto3``).` and add the import and check:

```python
from typing import Any

from evalshift.stores.uri import require_store_modules


class S3Store:
    """Put objects under ``s3://<bucket>/<prefix>``."""

    def __init__(self, bucket: str, prefix: str = "", *, client: Any | None = None) -> None:
        if client is None:
            require_store_modules("s3")  # loud here, in user code, not on the first background put
        self.bucket = bucket
```

Change the lazy-import comment to `import boto3  # lazy: optional, presence checked in __init__`.

`src/evalshift/stores/gcs.py`: same shape with `require_store_modules("gs")`, docstring `Needs ``google-cloud-storage`` (``pip install google-cloud-storage``)`, comment on line 29 `# lazy: optional, presence checked in __init__`.

`src/evalshift/stores/azure.py`: same shape with `require_store_modules("az")`, docstring `Needs ``azure-storage-blob`` and ``azure-identity`` (``pip install azure-storage-blob azure-identity``)`, comment on line 31 `# lazy: optional, presence checked in __init__`.

The import `evalshift.stores.uri` from the three store modules creates no cycle: `uri.py` imports the store modules only inside `open_store`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_stores_adapters.py tests/test_object_store_sink.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift/stores/s3.py src/evalshift/stores/gcs.py src/evalshift/stores/azure.py tests/test_stores_adapters.py
git commit -m "feat(stores): a store built without a client checks its library at construction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 3: `SinkConfigurationError`, recorded at import, raised by `require_sink_ready()`

**Files:**
- Modify: `src/evalshift/config.py:36,108-179,181-221,328-339`
- Modify: `src/evalshift/__init__.py:19,34-50`
- Modify: `tests/test_config_sink_env.py` (replace lines 1-2, 18, 51-133)

**Interfaces:**
- Consumes: `MissingStoreDependencyError` (Task 1).
- Produces: `class SinkConfigurationError(RuntimeError)` exported from `evalshift` and `evalshift.config`; `_Config.sink_error: SinkConfigurationError | None`; `require_sink_ready() -> None` (raises when `_gate_on()` and an error is recorded); `is_capture_enabled()` now `_gate_on() and not _sink_blocked()`; `configure(sink=...)` clears `sink_error`, `configure()` without `sink` calls `require_sink_ready()`; `reset_config()` resets the once-per-process warning flag.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config_sink_env.py`: change the module docstring to `"""``EVALSHIFT_SINK`` selects an ObjectStoreSink at config construction; an unusable value is recorded there and raised by ``require_sink_ready()`` once capture is on. ``configure(sink=...)`` still beats the env."""`, change line 18 to `from evalshift.stores.uri import MissingStoreDependencyError`, add `import traceback` and `from evalshift import SinkConfigurationError, capture` and `from evalshift.config import require_sink_ready, is_capture_enabled` to the imports, delete `test_env_sink_bad_grammar_warns_once_and_falls_back`, `test_env_sink_missing_extra_warns_and_falls_back`, `test_env_sink_unexpected_error_warns_once_and_falls_back` and `test_env_sink_warning_does_not_leak_secret_bearing_value`, and add:

```python
def _recorded() -> SinkConfigurationError:
    err = cfg._CONFIG.sink_error
    assert isinstance(err, SinkConfigurationError)
    return err


def test_bad_grammar_is_recorded_at_reset_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EVALSHIFT_SINK", "ftp://bucket/prefix")
    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        reset_config()  # must not raise: this runs at `import evalshift`
    err = _recorded()
    assert str(err) == (
        "EVALSHIFT_SINK is not a valid store URI. Accepted forms: s3://<bucket>/<prefix>, "
        "gs://<bucket>/<prefix>, az://<account>/<container>/<prefix>. "
        "Credentials never go in the URI."
    )
    assert not caplog.records
    assert cfg._CONFIG.sink is None


def test_missing_library_is_recorded_naming_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(uri: str) -> object:
        raise MissingStoreDependencyError("s3", "boto3")

    monkeypatch.setattr(cfg, "open_store", _raise)
    monkeypatch.setenv("EVALSHIFT_SINK", "s3://bucket/prefix")
    reset_config()
    assert str(_recorded()) == (
        "EVALSHIFT_SINK points at an object store (s3://), but boto3 is not installed. "
        "Run: pip install boto3, or unset EVALSHIFT_SINK to capture to local disk."
    )


def test_missing_azure_library_names_both_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(uri: str) -> object:
        raise MissingStoreDependencyError("az", "azure.identity")

    monkeypatch.setattr(cfg, "open_store", _raise)
    monkeypatch.setenv("EVALSHIFT_SINK", "az://acct/c/p")
    reset_config()
    assert "Run: pip install azure-storage-blob azure-identity," in str(_recorded())


def test_unexpected_open_error_is_recorded_without_its_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(uri: str) -> object:
        raise RuntimeError(f"boom {uri}")

    monkeypatch.setattr(cfg, "open_store", _raise)
    monkeypatch.setenv("EVALSHIFT_SINK", "s3://bucket/SECRETPREFIX")
    reset_config()
    err = _recorded()
    assert str(err) == (
        "EVALSHIFT_SINK could not be opened (RuntimeError). Unset it to capture to local disk."
    )
    assert "SECRETPREFIX" not in "".join(traceback.format_exception(err))


@pytest.mark.parametrize(
    ("value", "secret"),
    [
        ("az://acct/c?sv=1&sig=SECRETSIG", "SECRETSIG"),
        (
            "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=SECRETKEY==;"
            "EndpointSuffix=core.windows.net",
            "SECRETKEY",
        ),
        ("az://acct#sig=SECRETSIG", "SECRETSIG"),
    ],
)
def test_recorded_error_never_leaks_a_secret_bearing_value(
    monkeypatch: pytest.MonkeyPatch, value: str, secret: str
) -> None:
    # The parser's grammar errors may quote the value they reject; the recorded error must
    # carry a fixed message and no chained cause or context that reaches a traceback.
    monkeypatch.setenv("EVALSHIFT_SINK", value)
    reset_config()
    err = _recorded()
    assert secret not in "".join(traceback.format_exception(err))
    assert err.__cause__ is None and err.__suppress_context__


def test_require_sink_ready_raises_only_with_the_gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_SINK", "ftp://bucket/prefix")
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    reset_config()
    require_sink_ready()  # gate off: nothing
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    with pytest.raises(SinkConfigurationError, match="EVALSHIFT_SINK is not a valid store URI"):
        require_sink_ready()
    with pytest.raises(SinkConfigurationError):
        require_sink_ready()  # raising twice must work (the stored instance is reused)


def test_configure_with_a_sink_clears_the_error_and_without_one_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALSHIFT_SINK", "ftp://bucket/prefix")
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    reset_config()
    with pytest.raises(SinkConfigurationError):
        configure(dedup=False)
    assert cfg._CONFIG.dedup is True  # the raise comes first; nothing was merged
    memory = MemorySink()
    configure(sink=memory)
    assert cfg._CONFIG.sink_error is None
    require_sink_ready()
    assert is_capture_enabled() is True
    assert _unwrap(active_sink()) is memory


def test_is_capture_enabled_is_false_while_the_sink_is_blocked(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EVALSHIFT_SINK", "ftp://bucket/prefix")
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    reset_config()
    with caplog.at_level(logging.WARNING, logger="evalshift"):
        assert is_capture_enabled() is False
        assert is_capture_enabled() is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage().startswith("evalshift: capture disabled: EVALSHIFT_SINK")


def test_gate_turned_on_late_drops_captures_instead_of_writing_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path))
    monkeypatch.setenv("EVALSHIFT_SINK", "ftp://bucket/prefix")
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    reset_config()

    @capture.agent(suite="s", redact=False, tools=[])  # gate off: decorating must not raise
    def run(q: str) -> str:
        return q.upper()

    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    with caplog.at_level(logging.WARNING, logger="evalshift"):
        assert run("hi") == "HI"
        assert run("again") == "AGAIN"
    assert not list(tmp_path.rglob("*.json"))
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
```

Add `from pathlib import Path` to the imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_sink_env.py -q`
Expected: ImportError on `SinkConfigurationError`.

- [ ] **Step 3: Implement `config.py`**

Line 36: `from evalshift.stores.uri import STORE_URI_FORMS, MissingStoreDependencyError, open_store`.

Before `_env_sink` (line 108) add:

```python
class SinkConfigurationError(RuntimeError):
    """``EVALSHIFT_SINK`` names a store the SDK cannot build.

    Recorded when the process-wide config is built (at ``import evalshift``) and raised by
    :func:`require_sink_ready` at the first explicit touch -- a ``capture.*`` decorator, a client
    wrapper, the LangChain handler, or :func:`configure` without an explicit ``sink`` -- while
    ``EVALSHIFT_CAPTURE`` is on. Never raised on bare import, never when capture is off. The
    message names the variable, the scheme and the fix; never the value, which may carry a
    pasted credential.
    """
```

Replace `_env_sink` (lines 108-151) with:

```python
def _env_sink() -> Sink | None:
    """Build the sink ``EVALSHIFT_SINK`` names, or ``None`` for the default ``FileSink``.

    Blank counts as unset. The store's client is built lazily, so this never touches the network
    or the credential chain.

    Raises:
        SinkConfigurationError: when the value is set but unusable. The text never includes the
            raw value, directly or through a chained cause: several of the parser's grammar
            errors quote the URI they reject, so every branch raises ``from None`` with a fixed
            message naming only the variable, the scheme, the accepted forms or the package to
            install.
    """
    raw = os.environ.get(SINK_ENV, "").strip()
    if not raw:
        return None
    try:
        return ObjectStoreSink(open_store(raw))
    except MissingStoreDependencyError as exc:
        raise SinkConfigurationError(
            f"{SINK_ENV} points at an object store ({exc.scheme}://), but {exc.package} is not "
            f"installed. Run: pip install {exc.packages}, or unset {SINK_ENV} to capture to "
            "local disk."
        ) from None
    except ValueError:
        raise SinkConfigurationError(
            f"{SINK_ENV} is not a valid store URI. Accepted forms: {STORE_URI_FORMS}. "
            "Credentials never go in the URI."
        ) from None
    except Exception as exc:
        # Not a grammar or missing-library error, so its text is unknown and may quote the value.
        raise SinkConfigurationError(
            f"{SINK_ENV} could not be opened ({type(exc).__name__}). "
            "Unset it to capture to local disk."
        ) from None
```

In `_Config` replace the `sink` field with two plain fields and add `__post_init__`; update the class docstring's first sentence about `sink` to: *``sink`` defaults from ``EVALSHIFT_SINK`` (an object-store URI selects an :class:`~evalshift.sinks.object_store.ObjectStoreSink`; unset or blank leaves it ``None``, i.e. the default ``FileSink``; an unusable value leaves it ``None`` and records ``sink_error`` instead); an explicit ``configure(sink=...)`` still wins and clears ``sink_error``.*

```python
    sink: Sink | None = None
    sink_error: SinkConfigurationError | None = None
    ...  # the hygiene fields stay exactly as they are

    def __post_init__(self) -> None:
        if self.sink is not None or self.sink_error is not None:
            return
        try:
            self.sink = _env_sink()
        except SinkConfigurationError as exc:
            self.sink_error = exc  # raised later by require_sink_ready(); import stays safe
        except Exception:  # pragma: no cover - _env_sink converts everything; belt and braces
            safety.logger.debug("evalshift: sink env failed (swallowed)", exc_info=True)
```

Replace `is_capture_enabled` (lines 184-186) with:

```python
#: Whether the once-per-process "capture disabled" warning has been logged. Reset by reset_config().
_SINK_BLOCK_WARNED = False


def _gate_on() -> bool:
    """The raw gate: ``EVALSHIFT_CAPTURE`` is set to a truthy value."""
    return os.environ.get(CAPTURE_ENV, "").strip().lower() in _TRUTHY


def _sink_blocked() -> bool:
    """True while a recorded ``EVALSHIFT_SINK`` error must stop captures; warns once per process."""
    global _SINK_BLOCK_WARNED
    if _CONFIG.sink_error is None:
        return False
    if not _SINK_BLOCK_WARNED:
        _SINK_BLOCK_WARNED = True
        safety.logger.warning("evalshift: capture disabled: %s", _CONFIG.sink_error)
    return True


def is_capture_enabled() -> bool:
    """True iff ``EVALSHIFT_CAPTURE`` is truthy and no unusable ``EVALSHIFT_SINK`` blocks capture.

    A sink the user asked for is never silently replaced by local disk: with the gate on and a
    recorded sink error this is ``False`` (one ``WARNING`` per process), so every entry point
    behaves as if capture were off. Normally the error is raised earlier, at startup, by
    :func:`require_sink_ready`; this is the backstop for a gate that turned on after that.
    """
    return _gate_on() and not _sink_blocked()


def require_sink_ready() -> None:
    """Raise the recorded ``EVALSHIFT_SINK`` error if capture is on; a no-op otherwise.

    Every explicit touch point calls this -- the ``capture.*`` decorators, the client wrappers,
    the LangChain handler and :func:`configure` without a ``sink`` -- so in a real agent the
    raise lands at process start, in the deploy logs, before any traffic.
    """
    if _CONFIG.sink_error is not None and _gate_on():
        raise _CONFIG.sink_error.with_traceback(None)
```

In `configure`, replace the `sink` branch with:

```python
    if sink is not _UNSET:
        _CONFIG.sink = sink
        _CONFIG.sink_error = None  # an explicit sink replaces whatever EVALSHIFT_SINK named
    else:
        require_sink_ready()
```

(keep it as the first statement of the function so a raise merges nothing). In `reset_config`, add `global _CONFIG, _SINK_BLOCK_WARNED` and `_SINK_BLOCK_WARNED = False` before `dedup.reset_registry()`. Add `"SinkConfigurationError"` and `"require_sink_ready"` to `__all__`. In `src/evalshift/__init__.py` change line 19 to `from evalshift.config import SinkConfigurationError, configure, flush_captures` and add `"SinkConfigurationError"` to `__all__` (alphabetical, after `"SCHEMA_VERSION"`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config_sink_env.py tests/test_config.py -q && uv run mypy`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift/config.py src/evalshift/__init__.py tests/test_config_sink_env.py
git commit -m "feat(config): an unusable EVALSHIFT_SINK is raised, not quietly replaced by disk

The error is recorded when the config is built at import and raised by
require_sink_ready() once EVALSHIFT_CAPTURE is on; configure(sink=...)
clears it. If the gate turns on after that, is_capture_enabled() is False
with one warning, so nothing is ever written to disk in its place.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 4: The touch points call `require_sink_ready()`

**Files:**
- Modify: `src/evalshift/capture/api.py:771` (`agent`, right after `redactor = resolve_redactor(redact)`), `:842` (`agent_session`, same spot), `:908` (`agent_session_async`, same spot), `:989-995` (`tool`, first statement of the method body)
- Modify: `src/evalshift/adapters/openai.py:316-334`, `src/evalshift/adapters/anthropic.py:311-…`, `src/evalshift/adapters/genai.py:410-…` (each `wrap_*` body, before `return cast(C, ClientProxy(...))`)
- Modify: `src/evalshift/adapters/langchain.py:281` (after `super().__init__()`)
- Create: `tests/test_sink_touch_points.py`

**Interfaces:**
- Consumes: `config.require_sink_ready()` (Task 3).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sink_touch_points.py`:

```python
"""Every explicit touch point raises the recorded ``EVALSHIFT_SINK`` error while capture is on."""

from __future__ import annotations

import asyncio
import importlib
import logging

import pytest

from evalshift import SinkConfigurationError, capture, configure
from evalshift.config import is_capture_enabled, require_sink_ready, reset_config


@pytest.fixture
def broken_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALSHIFT_SINK", "ftp://bucket/prefix")
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    reset_config()  # records, never raises


def test_agent_decorator_raises_at_decoration(broken_sink: None) -> None:
    with pytest.raises(SinkConfigurationError, match="EVALSHIFT_SINK is not a valid store URI"):
        capture.agent(suite="s", redact=False, tools=[])


def test_tool_decorator_raises_at_decoration(broken_sink: None) -> None:
    with pytest.raises(SinkConfigurationError):

        @capture.tool
        def lookup() -> None: ...

    with pytest.raises(SinkConfigurationError):
        capture.tool(name="lookup")


def test_agent_session_raises_on_entry(broken_sink: None) -> None:
    with pytest.raises(SinkConfigurationError):
        with capture.agent_session(suite="s", redact=False, tools=[]):
            pass


def test_agent_session_async_raises_on_entry(broken_sink: None) -> None:
    async def run() -> None:
        async with capture.agent_session_async(suite="s", redact=False, tools=[]):
            pass

    with pytest.raises(SinkConfigurationError):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("evalshift.adapters.openai", "wrap_openai"),
        ("evalshift.adapters.anthropic", "wrap_anthropic"),
        ("evalshift.adapters.genai", "wrap_genai"),
    ],
)
def test_client_wrappers_raise_at_wrap_time(broken_sink: None, module: str, name: str) -> None:
    wrap = getattr(importlib.import_module(module), name)
    with pytest.raises(SinkConfigurationError):
        wrap(object())


def test_langchain_handler_raises_at_construction(broken_sink: None) -> None:
    pytest.importorskip("langchain_core")
    from evalshift.adapters.langchain import EvalShiftCallbackHandler

    with pytest.raises(SinkConfigurationError):
        EvalShiftCallbackHandler(suite="s", redact=False, tools=[])


def test_gate_off_never_raises_or_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EVALSHIFT_SINK", "ftp://bucket/prefix")
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    reset_config()
    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        require_sink_ready()
        capture.agent(suite="s", redact=False, tools=[])
        capture.tool(name="x")
        with capture.agent_session(suite="s", redact=False, tools=[]) as tree:
            assert tree is None
        from evalshift.adapters.openai import wrap_openai

        wrap_openai(object())
        configure(dedup=False)
        assert is_capture_enabled() is False
    assert not caplog.records
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sink_touch_points.py -q`
Expected: every `raises` test FAILS with "DID NOT RAISE"; `test_gate_off_never_raises_or_warns` passes already.

- [ ] **Step 3: Implement**

`src/evalshift/capture/api.py` — in `agent`, after `redactor = resolve_redactor(redact)  # eager: ...` add:

```python
        config.require_sink_ready()  # eager too: a sink the SDK cannot build fails at startup
```

In `agent_session` and `agent_session_async`, after their `redactor = resolve_redactor(redact)` line add the same call (it runs on `with` entry, before the gate check). In `tool`, make it the first statement of the method body:

```python
    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
        """Decorator that records a tool span (no-op when no agent session is active).

        Supports both ``@capture.tool`` and ``@capture.tool(name="...")``. Like :meth:`agent`, an
        unusable ``EVALSHIFT_SINK`` raises here, at decoration time, when capture is on.
        """
        config.require_sink_ready()
```

`src/evalshift/adapters/openai.py`: add `from evalshift import config` to the imports and, in `wrap_openai`, before the `return`:

```python
    config.require_sink_ready()  # a sink the SDK cannot build fails here, at startup
    return cast(C, ClientProxy(client, _OVERRIDES))
```

Same two-line change in `wrap_anthropic` (`anthropic.py`, which already imports `safety` from `evalshift`: extend to `from evalshift import config, safety`) and `wrap_genai` (`genai.py`). In `langchain.py`'s `EvalShiftCallbackHandler.__init__`, after `super().__init__()` add `config.require_sink_ready()` (the module already imports `config`). Add one sentence to each wrapper's docstring: *Raises :class:`~evalshift.SinkConfigurationError` when capture is on and ``EVALSHIFT_SINK`` cannot be built.*

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sink_touch_points.py tests/test_capture_agent.py tests/test_capture_tool.py tests/test_capture_gate.py tests/adapters -q && uv run mypy && uv run ruff check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift/capture/api.py src/evalshift/adapters/openai.py src/evalshift/adapters/anthropic.py src/evalshift/adapters/genai.py src/evalshift/adapters/langchain.py tests/test_sink_touch_points.py
git commit -m "feat(capture): decorators, wrappers and the LangChain handler raise on an unusable sink

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 5: Process-level proof: bare import is safe, decoration is loud

**Files:**
- Modify: `tests/test_object_store_sink_subprocess.py` (append; reuse `_run_child` and its `child_env` construction)

- [ ] **Step 1: Write the tests**

```python
def test_import_with_a_broken_sink_exits_zero_and_logs_nothing() -> None:
    # The CLI imports the SDK, and a developer may have both env vars exported in a shell:
    # `import evalshift` must stay silent and successful whatever EVALSHIFT_SINK holds.
    result = _run_child(
        "import evalshift",
        env={"EVALSHIFT_SINK": "ftp://bucket/prefix", "EVALSHIFT_CAPTURE": "1"},
    )
    assert "EVALSHIFT_SINK" not in result.stderr


def test_decorating_with_a_broken_sink_fails_the_process_naming_the_fix() -> None:
    child_env = {k: v for k, v in os.environ.items() if k != "EVALSHIFT_SINK"}
    child_env.update({"EVALSHIFT_SINK": "ftp://bucket/prefix", "EVALSHIFT_CAPTURE": "1"})
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from evalshift import capture\ncapture.agent(suite='s', redact=False, tools=[])",
        ],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "SinkConfigurationError" in result.stderr
    assert "Accepted forms: s3://<bucket>/<prefix>" in result.stderr
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_object_store_sink_subprocess.py -q`
Expected: PASS (Tasks 3–4 are in place). If the first test fails because `_run_child` prepends a `pythonpath`, pass `pythonpath=Path("src")` the way the file's existing tests do.

- [ ] **Step 3: Commit**

```bash
git add tests/test_object_store_sink_subprocess.py
git commit -m "test(config): bare import survives a broken EVALSHIFT_SINK; decoration does not

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 6: SDK docs, decision record and changelog

**Files:**
- Modify: `README.md:50-56,112-121`, `DOCS.md:52-60,210-216,566,649-663,1086-1100` plus every `MissingExtraError` hit, `llms.txt:8`, `llms-full.txt:7-10,55,259-263`, `docs/DECISIONS.md:526-533,545-549` (+ new sub-decision), `CHANGELOG.md:8`

Enumerate first: `git grep -n -e '\[s3\]' -e '\[gcs\]' -e '\[azure\]' -e 'falls back to local disk' -e 'local disk instead' -e 'keeps writing to local disk' -e MissingExtraError -e 'pip extra' -- README.md DOCS.md llms.txt llms-full.txt docs/` and fix every hit; the list below is what that grep returns today.

- [ ] **Step 1: README.md**

Replace lines 50-56 with:

```markdown
Optional object-store sinks (`EVALSHIFT_SINK=s3://…` / `gs://…` / `az://…`, see below) use the
provider's own client library — install it like any other package:

```bash
pip install boto3                              # s3:// — Amazon S3 and S3-compatible stores
pip install google-cloud-storage               # gs://
pip install azure-storage-blob azure-identity  # az://
```
```

In the "Hosts whose disk does not outlive them" section replace `are accepted; install the matching extra above.` with `are accepted; install the provider's client library (above).` and replace the sentence `Uploads run on a bounded background thread and never raise into the agent; an invalid `EVALSHIFT_SINK` or a missing extra logs one warning and falls back to local disk.` with:

```markdown
Uploads run on a bounded background thread and never raise into the agent. Configuration is the
one thing that does raise: if `EVALSHIFT_SINK` is set but the library is missing or the value is
malformed, the first `capture.*` decorator, client wrapper or `configure()` call raises
`SinkConfigurationError` — at startup, naming the `pip install` to run — instead of quietly
writing to local disk. With `EVALSHIFT_CAPTURE` off nothing happens at all.
```

- [ ] **Step 2: DOCS.md**

Lines 55-60 (install block): same replacement as the README's. Line 52 keeps "Every adapter module is import-guarded".

Lines 210-216: change the lead-in to `Three deliberate exceptions to fail-open:` and add a row:

```markdown
| Sink configuration (`EVALSHIFT_SINK`) | **Raises** `SinkConfigurationError` at the first explicit touch while capture is on — never on bare import, never with capture off | A bucket you asked for and did not get is the loss this feature exists to prevent; a crash in the deploy log beats captures quietly written to a disk that dies with the task |
```

Line 566 (env table row): replace the description cell with:

```markdown
Ship captures to an object store you own: `s3://<bucket>/<prefix>`, `gs://<bucket>/<prefix>` or `az://<account>/<container>/<prefix>`. Needs the provider's client library (`pip install boto3` / `google-cloud-storage` / `azure-storage-blob azure-identity`). An unusable value — library missing, malformed URI — is recorded here and raised as `SinkConfigurationError` by the first `capture.*` decorator, client wrapper, LangChain handler or `configure()` call while `EVALSHIFT_CAPTURE` is on; the message names the variable, the scheme and the fix, never the value. With capture off nothing happens.
```

Lines 659-661 (URI table): rename the `Extra` column to `Install` and put `pip install boto3`, `pip install google-cloud-storage`, `pip install azure-storage-blob azure-identity` in it. Line 663: replace from `If `EVALSHIFT_SINK` is invalid` to the end of the paragraph with:

```markdown
If `EVALSHIFT_SINK` is invalid, its library is missing, or the store cannot be opened for any other reason, the SDK records the problem at import and raises it as `SinkConfigurationError` the first time your code asks it to capture — applying a `capture.*` decorator, wrapping a client, building the LangChain handler or calling `configure()` without a `sink` — while `EVALSHIFT_CAPTURE` is on. In a real agent that is process start. Bare `import evalshift` never raises, and with capture off nothing happens. Should the gate turn on only after those ran, captures are dropped with one `WARNING` rather than written to local disk: a store you asked for is never silently replaced. The message names the variable, the scheme and the package to install (`Run: pip install boto3`), never the value.
```

Lines 1086-1100 (API reference): after the `ObjectStoreSink` block add:

```markdown
`evalshift.SinkConfigurationError` (`RuntimeError`) — raised by `evalshift.config.require_sink_ready()`, which every touch point calls; `evalshift.stores.uri.MissingStoreDependencyError` (`ImportError`; `.scheme`, `.module`, `.package`, `.packages`, `.extra`) — raised by `open_store()`, `require_store_modules(scheme)` and a store constructed without a `client` when its library is absent.
```

Replace every remaining `MissingExtraError` with `MissingStoreDependencyError` and every `pip install "evalshift-sdk[<extra>]"`-style phrase with the package form.

- [ ] **Step 3: llms.txt and llms-full.txt**

`llms.txt:8`: add `SinkConfigurationError` to the top-level public API list. `llms-full.txt:7-10`: replace the three store extras lines with `Object stores: pip install boto3 (s3://) | google-cloud-storage (gs://) | azure-storage-blob azure-identity (az://) -- plain packages, no extra needed`. Line 55: rewrite the `EVALSHIFT_SINK` row to `| EVALSHIFT_SINK | unset (disk) | Ship captures to s3://<bucket>/<prefix>, gs://<bucket>/<prefix> or az://<account>/<container>/<prefix>; needs the provider's client library (pip install boto3 / google-cloud-storage / azure-storage-blob azure-identity). Unusable value (library missing, bad URI) -> recorded at import, raised as SinkConfigurationError by the first capture.* decorator / client wrapper / LangChain handler / configure() while EVALSHIFT_CAPTURE is on; never on bare import, never with capture off; if the gate turns on later, captures are dropped with ONE warning (never written to disk instead). Message names the var, scheme and package, never the value. | Config construction |`. Lines 259-263: `MissingExtraError (ImportError, .extra) naming pip install "evalshift-sdk[<extra>]"` → `MissingStoreDependencyError (ImportError; .scheme/.module/.package/.packages/.extra) naming "pip install <packages>"`. Add after it: `evalshift.config.require_sink_ready() -> None: raises the recorded SinkConfigurationError iff EVALSHIFT_CAPTURE is on. evalshift.SinkConfigurationError(RuntimeError).`

- [ ] **Step 4: docs/DECISIONS.md**

In D-stores, replace the sentence ending `(or, for a missing client library, the pip extra).` with `accepted forms and the package to install -- never the value.`, replace `Extras `[s3]`, `[gcs]`, `[azure]`;` with `Client libraries are plain packages (`pip install boto3` / `google-cloud-storage` / `azure-storage-blob azure-identity`); the extras of the same names stay in pyproject as tested floors but are never advertised;`, and replace the bullet `**Warning-level logging, deliberately**: ... and an invalid or unusable `EVALSHIFT_SINK`, log at `WARNING`` so it no longer lists the invalid sink. Then append a sub-decision:

```markdown
### D-stores-b — an unusable `EVALSHIFT_SINK` raises at startup; it is never replaced by disk
A warning on a green deploy is read by nobody, and a sink the user asked for and did not get is
the loss D-stores exists to prevent. So configuration is a third carve-out from fail-open (with
redaction and the read side): `_Config` records a `SinkConfigurationError` at import and
`config.require_sink_ready()` raises it at the first explicit touch -- `capture.agent` /
`agent_session` / `tool`, `wrap_openai` / `wrap_anthropic` / `wrap_genai`,
`EvalShiftCallbackHandler()`, `configure()` without a `sink` -- iff `EVALSHIFT_CAPTURE` is on.
Bare `import evalshift` never raises (the CLI imports the SDK; a shell may export the vars), and
with capture off nothing happens. Backstop: `is_capture_enabled()` is `False` (one `WARNING`)
while an error is recorded, so a gate that turns on late drops captures rather than writing them
to local disk. Messages name packages, not extras (`pip install boto3`), because a user who
installed the SDK normally does not know what an extra is, and `"evalshift-sdk[s3]"` needs quotes
in zsh. Runtime faults (puts, queue, exit flush) stay fail-open. `MissingExtraError` became
`MissingStoreDependencyError`, no alias (0.5.0 was one day old).
```

- [ ] **Step 5: CHANGELOG.md**

Under `## [Unreleased]`:

```markdown
### Changed

- **An unusable `EVALSHIFT_SINK` now raises instead of falling back to local disk.** When
  `EVALSHIFT_CAPTURE` is on and the value names a store the SDK cannot build — its client
  library is not installed, or the URI is malformed — the first `capture.*` decorator, client
  wrapper (`wrap_openai` / `wrap_anthropic` / `wrap_genai`), `EvalShiftCallbackHandler()` or
  `configure()` without a `sink` raises `SinkConfigurationError` (new, exported from
  `evalshift`), naming the variable, the scheme and the fix. In a real agent that is process
  start. Bare `import evalshift` never raises, and with capture off nothing happens. If the gate
  turns on only after that, captures are dropped with one `WARNING` rather than written to disk.
  Previously the SDK logged one warning and wrote to local disk, which on the ephemeral hosts
  this feature exists for lost every capture behind a green deploy.
- Every message names the package to install, not a pip extra: `pip install boto3`,
  `google-cloud-storage`, `azure-storage-blob azure-identity` (both Azure packages in one
  command). The `[s3]` / `[gcs]` / `[azure]` extras still exist as tested version floors but
  are no longer documented. `evalshift.stores.uri.MissingExtraError` is renamed
  `MissingStoreDependencyError` (still an `ImportError`; new `.package` / `.packages`), and a
  store constructed without a `client` (`S3Store(...)` etc.) raises it at construction instead
  of failing its first background upload.
```

- [ ] **Step 6: Verify and commit**

Run: `git grep -n -e '\[s3\]' -e '\[gcs\]' -e '\[azure\]' -e MissingExtraError -e 'falls back to local disk' -e 'local disk instead' -- README.md DOCS.md llms.txt llms-full.txt docs/` — expected: only `pyproject.toml`-free, historical `CHANGELOG.md` 0.5.0 lines remain (none of these files should match). Then `uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest -q` — all green.

```bash
git add README.md DOCS.md llms.txt llms-full.txt docs/DECISIONS.md CHANGELOG.md
git commit -m "docs(sdk): the sink raises on misconfiguration; install the package, not the extra

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Push and open the PR against `main` (`gh pr create -R evalshift/evalshift-sdk`), body ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

---

# Part B — evalshift-cli (1.3.1)

Work in `/home/lukas/repos/evalshift/evalshift-cli` on branch `fix/store-dependency-messages` from `main`. Export `PATH=$PWD/.venv/bin:$PATH` first (the hooks need it). Paths relative to that repo.

### Task 7: Package tables in `store_uri.py`; `open_store` names the package

**Files:**
- Modify: `src/evalshift_cli/captures/store_uri.py:30,50-53`
- Modify: `src/evalshift_cli/captures/remote.py:155-173`
- Modify: `tests/unit/test_captures_stores.py:207-220`, `tests/unit/test_cli_capture_remote.py:322-346`, `tests/unit/test_captures_store_uri.py` (append)

**Interfaces:**
- Produces: `_PACKAGE_FOR_MODULE`, `_PACKAGES_FOR_SCHEME` (verbatim copies of the SDK's), `StoreURI.packages -> str`, `package_for_module(module: str) -> str`. `RemoteStoreUnavailable.summary == "<scheme>:// captures store needs <package>, which is not installed"`, `.hint == "install it: pip install <packages>"`.

- [ ] **Step 1: Write the failing tests**

Replace `test_open_store_missing_extra_names_pip_extra` and `test_open_store_azure_needs_identity_too` in `tests/unit/test_captures_stores.py` with:

```python
def test_open_store_missing_library_names_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote_module, "_installed", lambda module: False)
    with pytest.raises(RemoteStoreUnavailable) as info:
        open_store(parse_store_uri("gs://b/p"))
    assert info.value.summary == (
        "gs:// captures store needs google-cloud-storage, which is not installed"
    )
    assert info.value.hint == "install it: pip install google-cloud-storage"


def test_open_store_azure_names_both_packages_when_identity_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote_module, "_installed", lambda module: module != "azure.identity")
    with pytest.raises(RemoteStoreUnavailable) as info:
        open_store(parse_store_uri("az://a/c/p"))
    assert info.value.summary == "az:// captures store needs azure-identity, which is not installed"
    assert info.value.hint == "install it: pip install azure-storage-blob azure-identity"
```

Append to `tests/unit/test_captures_store_uri.py`:

```python
def test_packages_names_every_library_the_scheme_needs() -> None:
    assert parse_store_uri("s3://b/p").packages == "boto3"
    assert parse_store_uri("gs://b/p").packages == "google-cloud-storage"
    assert parse_store_uri("az://a/c/p").packages == "azure-storage-blob azure-identity"
```

In `tests/unit/test_cli_capture_remote.py` change the `missing_extra` fixture's raise to

```python
        raise RemoteStoreUnavailable(
            "s3:// captures store needs boto3, which is not installed",
            hint="install it: pip install boto3",
        )
```

rename it `missing_library` (and the fixture parameter in the test below it), rename the test to `test_missing_library_exits_1_naming_the_package`, and change its assertion to `assert "pip install boto3" in result.stdout`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_captures_stores.py tests/unit/test_captures_store_uri.py tests/unit/test_cli_capture_remote.py -q --no-cov`
Expected: FAIL on the new strings and on `packages`.

- [ ] **Step 3: Implement**

`src/evalshift_cli/captures/store_uri.py` after line 30:

```python
#: The pip distribution that provides each client module. Shared verbatim with the SDK.
_PACKAGE_FOR_MODULE: dict[str, str] = {
    "boto3": "boto3",
    "google.cloud.storage": "google-cloud-storage",
    "azure.storage.blob": "azure-storage-blob",
    "azure.identity": "azure-identity",
}

#: The ``pip install`` argument that installs everything a scheme needs, in one command.
#: Named in every message instead of the extras: a user who installed the CLI the normal way
#: has no reason to know what an extra is. Shared verbatim with the SDK.
_PACKAGES_FOR_SCHEME: dict[str, str] = {
    "s3": "boto3",
    "gs": "google-cloud-storage",
    "az": "azure-storage-blob azure-identity",
}


def package_for_module(module: str) -> str:
    """The pip distribution that provides ``module`` (one of the client modules above)."""
    return _PACKAGE_FOR_MODULE[module]
```

and in `StoreURI`:

```python
    @property
    def packages(self) -> str:
        """The ``pip install`` argument that installs this scheme's client library (or libraries)."""
        return _PACKAGES_FOR_SCHEME[self.scheme]
```

`src/evalshift_cli/captures/remote.py`: import `package_for_module` from `evalshift_cli.captures.store_uri` (next to the existing `StoreURI` import) and replace the raise in `open_store` with:

```python
    for module in _REQUIRED_MODULES[parsed.scheme]:
        if not _installed(module):
            raise RemoteStoreUnavailable(
                f"{parsed.scheme}:// captures store needs {package_for_module(module)}, "
                "which is not installed",
                hint=f"install it: pip install {parsed.packages}",
            )
```

Update the docstring's `Raises:` line to `RemoteStoreUnavailable: when a client module the scheme needs is not installed. The summary names the package that provides the first missing module; the hint names the pip command that installs everything the scheme needs.`

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_captures_stores.py tests/unit/test_captures_store_uri.py tests/unit/test_cli_capture_remote.py -q --no-cov && mypy --strict src/evalshift_cli && ruff check .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/captures/store_uri.py src/evalshift_cli/captures/remote.py tests/unit/test_captures_stores.py tests/unit/test_captures_store_uri.py tests/unit/test_cli_capture_remote.py
git commit -m "fix(captures): a missing store library is reported as the package to install

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 8: `doctor`'s `captures.store` row warns instead of failing

**Files:**
- Modify: `src/evalshift_cli/cli/commands/doctor.py:8-16,252-275,528-534`
- Modify: `src/evalshift_cli/cli/commands/capture.py:244-246` (docstring only)
- Modify: `tests/unit/test_doctor.py:810-821` (+ any exit-code test: `grep -n 'captures.store\|missing_extra' tests/unit/test_doctor.py`)

- [ ] **Step 1: Write the failing test**

Replace `test_missing_extra_fails` in `tests/unit/test_doctor.py` with:

```python
    def test_missing_library_warns_without_failing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A run on a committed suite never touches the bucket (the GitHub Action installs the
        # bare package), so a missing client library must not fail doctor -- or compare, which
        # exits on any doctor failure. The capture commands that do need it exit 1 themselves.
        self._config(tmp_path, "gs://b/p")

        def _raise(parsed: object) -> object:
            raise RemoteStoreUnavailable(
                "gs:// captures store needs google-cloud-storage, which is not installed",
                hint="install it: pip install google-cloud-storage",
            )

        monkeypatch.setattr(doctor_module, "open_store", _raise)
        results = run_checks(cwd=tmp_path, env=_empty_env())
        row = _by_name(results, "captures.store")
        assert row.status == "warn"
        assert "pip install google-cloud-storage" in row.detail
        assert not any(r.status == "fail" for r in results)
```

If the grep finds a command-level test asserting `exit_code == 1` for this case, change it to `== 0`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_doctor.py -q --no-cov -k missing_library`
Expected: FAIL (`'fail' == 'warn'`).

- [ ] **Step 3: Implement**

`doctor.py` lines 252-275: change the status to `"warn"` and the docstring to:

```python
    """One row for ``captures.store`` when the config names one: library installed, bucket listable.

    No config, an invalid config (the ``evalshift.yaml`` row already reports that) or no
    ``captures.store`` produce no row. A missing client library is a ``warn`` naming the package
    to install: a run on a committed suite never touches the bucket (CI installs the bare
    package, and ``compare`` exits on any doctor failure), and the capture commands that do
    need it exit 1 themselves. A listing that raises is a ``warn`` too: doctor runs on laptops
    without cloud credentials, and that must not fail the command.
    """
```

Lines 8-16 (module docstring): drop the `, or its ``captures.store`` needs a client extra that isn't installed` clause so exit 1 is `an ``evalshift.yaml`` exists in the cwd but doesn't validate`. Line 532: `an install hint such as ``pip install google-cloud-storage`` or a user-written store URI`. `capture.py:245`: `A missing client library is always a hard error here: the user asked for a store this command cannot read.`

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_doctor.py tests/unit/test_compare_command.py -q --no-cov && mypy --strict src/evalshift_cli && ruff check .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evalshift_cli/cli/commands/doctor.py src/evalshift_cli/cli/commands/capture.py tests/unit/test_doctor.py
git commit -m "fix(doctor): a missing store library is a warning, not a failure

It failed compare -- and so every GitHub Action run -- for a config that
names captures.store, although a run on a committed suite never reads the
bucket. The capture commands that need the library still exit 1.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 9: CLI docs, docs-currency tripwire and changelog

**Files:**
- Modify: `DOCS.md:220,872`, `llms-full.txt:155-158,176-180,322-326,740-745,1308-1311`, `docs/configuration.md:1044-1046`, `docs/sdk.md:124-125`, `CHANGELOG.md:8`, `tests/unit/test_docs_currency.py` (append)

- [ ] **Step 1: Write the failing tripwire**

Append to `tests/unit/test_docs_currency.py`:

```python
#: Install-story phrases retired on 2026-10-09: every message and doc names the package to
#: install (`pip install boto3`), never a pip extra, and `doctor` no longer fails over it.
RETIRED_INSTALL_TERMS: tuple[str, ...] = (
    "evalshift[s3]",
    "evalshift[gcs]",
    "evalshift[azure]",
    "client extra",
    "optional dependency",
    "whose extra is missing",
)

_INSTALL_COPY_FILES: tuple[str, ...] = (*PROSE_FILES, "docs/sdk.md")


@pytest.mark.parametrize("term", RETIRED_INSTALL_TERMS)
def test_copy_names_packages_not_extras(term: str) -> None:
    offenders = [name for name in _INSTALL_COPY_FILES if term in _whitespace_normalized(name)]
    assert offenders == [], f"{term!r} still appears in {offenders}"
```

Run: `pytest tests/unit/test_docs_currency.py -q --no-cov -k packages_not_extras` — expected: FAIL naming `DOCS.md`, `llms-full.txt`, `docs/configuration.md`, `docs/sdk.md`.

- [ ] **Step 2: Rewrite the sites**

`DOCS.md:220`: delete ` — or when its `captures.store` needs a client extra that is not installed` and change ``fail` (exit 1) naming the `pip install "evalshift[<extra>]"` when the client extra is missing` to ``warn` naming the `pip install <package>` to run when the client library is missing (the capture commands that need it exit 1 themselves)`.

`DOCS.md:872`: `Exit 1 only on an invalid existing config.` and `(`warn` naming the package to install when the client library is missing, `warn` when the bucket cannot be listed)`.

`llms-full.txt:155-158`: `Env/config check. Exit 1 ONLY when an existing evalshift.yaml fails validation (the row shows the problem count + "run `evalshift validate` for details"; validate prints each problem); missing API keys and a captures.store whose client library is missing are soft warnings (exit 0).`

`llms-full.txt:176-180`: `captures.store: warn if the client library is missing (detail names "pip install <package>"; exit 0 -- compare and CI are not failed over a bucket a committed-suite run never reads), warn if the bucket cannot be listed (exit 0), ok "<uri> reachable"; no row without a store (or when evalshift.yaml is invalid -- its own row reports that).`

`llms-full.txt:322-326`: `Missing library -> exit 1 naming pip install boto3 | google-cloud-storage | azure-storage-blob azure-identity (az:// needs both; one command installs them).`

`llms-full.txt:740-745`: replace `Extras: evalshift[s3] (boto3; AWS_ENDPOINT_URL covers MinIO/R2/B2), [gcs] (google-cloud-storage), [azure] (azure-storage-blob + azure-identity).` with `Client libraries are plain packages: pip install boto3 (s3://; AWS_ENDPOINT_URL covers MinIO/R2/B2), google-cloud-storage (gs://), azure-storage-blob azure-identity (az://).`

`llms-full.txt:1308-1311`: `doctor exits 1 only on invalid existing config;`.

`docs/configuration.md:1044-1046` Install column: `pip install boto3`, `pip install google-cloud-storage`, `pip install azure-storage-blob azure-identity`.

`docs/sdk.md:124-125`: `and install the provider's client library (`pip install boto3`, `google-cloud-storage` or `azure-storage-blob azure-identity`); `list` and `sync` then fetch new captures first.`

Then `git grep -n -e '\[s3\]' -e '\[gcs\]' -e '\[azure\]' -e 'client extra' -e 'optional dependency' -- README.md DOCS.md llms-full.txt docs/ src/` and fix anything the list above missed (the `CHANGELOG.md` 1.3.0 entry stays as history).

- [ ] **Step 3: CHANGELOG.md**

Under `## [Unreleased]`:

```markdown
### Fixed

- `doctor`'s `captures.store` row no longer fails the command — and with it `compare` and
  every GitHub Action run — when the store's client library is not installed; it warns, naming
  the package to install. A run on a committed suite never reads the bucket; `capture fetch`,
  `list` and `sync`, which do, still exit 1.

### Changed

- Every store message names the package to install instead of a pip extra: `pip install boto3`,
  `google-cloud-storage`, `azure-storage-blob azure-identity` (both Azure packages in one
  command). The `[s3]` / `[gcs]` / `[azure]` extras still exist but are no longer documented.
```

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/unit/test_docs_currency.py -q --no-cov && pre-commit run --all-files` — all green.

```bash
git add DOCS.md llms-full.txt docs/configuration.md docs/sdk.md CHANGELOG.md tests/unit/test_docs_currency.py
git commit -m "docs: install the store library as a plain package; doctor warns, never fails, over it

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Push and open the PR against `main` (`gh pr create -R evalshift/evalshift-cli`), body ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

---

# Part C — release and the website

### Task 10: Release SDK 0.6.0 and CLI 1.3.1 (after both PRs merge)

Follow the existing `chore(release): X` pattern, one PR per repo:

- **SDK 0.6.0**: `pyproject.toml:7`, `src/evalshift/__init__.py:32`, `llms-full.txt:4` (`version: 0.6.0`) and `:298` (`__version__ = "0.6.0"`), `DOCS.md:5` and `:1142`, `llms.txt:7`; move the `Unreleased` block under `## [0.6.0] - <date>`. `uv run pytest` green. Merge, tag `v0.6.0`, confirm PyPI shows 0.6.0.
- **CLI 1.3.1**: `pyproject.toml:7`, `DOCS.md:15`, `llms-full.txt:4`; changelog header `## [1.3.1] - <date>`; `pip install -e ".[dev]"` into `.venv` so `test_reference_header_version_matches_the_package` sees 1.3.1; `make ci` green. Merge, tag `v1.3.1`. The action's `bump-cli-pin.yml` opens the pin PR on its own; merge it.

### Task 11: Website follow-up (after Task 10)

In `/home/lukas/repos/evalshift/evalshift-client` on a branch from `main`:

- `src/pages/docs/pages/SdkConfig.tsx`: `SINK_ENV_CODE`'s first line → `pip install boto3   # s3:// — or google-cloud-storage (gs://), azure-storage-blob azure-identity (az://)`; the `EVALSHIFT_SINK` env bullet's `Needs the matching extra; an invalid value or a missing extra logs one warning and falls back to local disk.` → `Needs the provider's client library. An unusable value raises SinkConfigurationError the first time your code asks the SDK to capture — at startup — while capture is on; never on bare import.`; the `ObjectStoreSink` table's `Extra` column → `Install` with the three `pip install` strings; the paragraph starting `If EVALSHIFT_SINK is invalid, its extra is missing` → the DOCS.md wording from Task 6 Step 2 (the paragraph beginning "If `EVALSHIFT_SINK` is invalid, its library is missing").
- `src/pages/docs/pages/Captures.tsx`: the `Extras.` bullet → `Client library. Install the provider's own package — pip install boto3, google-cloud-storage, or azure-storage-blob azure-identity; a missing one exits 1 naming it.`
- `src/pages/docs/pages/Configuration.tsx`: URI table `Install` column → the three `pip install` strings.
- `src/pages/docs/pages/CliCommands.tsx` `doctor` row and `src/pages/docs/pages/Cli.tsx` doctor bullet: `fail`/`exit 1` for the missing library → `warn`; exit 1 only on an invalid config.
- `src/pages/docs/pages/Changelog.tsx`: add `1.3.1` and `0.6.0` lines; `src/lib/version.ts`: `CLI_VERSION = "1.3.1"`, `SDK_VERSION = "0.6.0"`; `HostedSetup.tsx` pin `1.3.1`.
- `npm run sync:llms`; `npm run lint && npm run typecheck && npx vitest run && npm run build` (the `scripts/llmsFullSync.test.ts` tripwire must pass). Commit, push, PR.
