# Object-store sink: fail loudly on a missing client library, name the package

**Date:** 2026-10-09 · **Status:** draft for maintainer review · **Repos:** evalshift-sdk, evalshift-cli (behaviour); evalshift-client (docs only); evalshift-action (nothing)

## Problem

SDK 0.5.0 / CLI 1.3.0 shipped the object-store sink behind pip extras (`evalshift-sdk[s3]`,
`[gcs]`, `[azure]`; `evalshift[s3]` …). Two things are wrong with how that fails:

1. **The SDK fails quietly.** `EVALSHIFT_SINK=s3://…` with no boto3 installed logs one
   `WARNING` at import and writes captures to local disk. On the ephemeral hosts the feature
   exists for, that is the exact loss it was meant to prevent, and a green deploy hides it.
2. **The fix is phrased in packaging jargon.** The error and every doc say "install the `[s3]`
   extra". A user who installed `evalshift-sdk` the normal way does not know what an extra is,
   and `pip install "evalshift-sdk[s3]"` needs quotes in zsh.

Side effect on the CLI: `doctor` returns `fail` for `captures.store` when the library is
missing, `compare` exits 1 on any `fail`, and the GitHub Action installs the bare package, so
every repo that sets `captures.store` has a red CI (reproduced 2026-10-09).

## Decision

- **Misconfiguration raises; runtime faults stay fail-open.** A sink the SDK cannot build from
  `EVALSHIFT_SINK` — client library missing, URI malformed, store failing to open — is refused
  at the agent's startup with one exception naming the fix. Network errors, failed puts, full
  queues and exit-flush timeouts keep today's fail-open behaviour. This is a documented carve-out
  from "fail-open is the core promise", in the same table as redaction and the read side.
- **Raise at the first explicit touch, never on bare import.** `import evalshift` stays safe
  (the CLI imports it; a developer may have the env vars exported in a shell). The raise happens
  where the user's own code first asks the SDK to capture.
- **Name the library, not the extra.** Every message and every doc says `pip install boto3`
  (`google-cloud-storage`, `azure-storage-blob azure-identity`). The extras stay in
  `pyproject.toml` as tested version floors; nothing user-facing mentions them.
- **The CLI keeps exit 1 on the capture commands, but `doctor` only warns.** A CI run on a
  committed suite never touches the bucket.

Not doing: bundling the cloud SDKs into the base package (three heavy dependency trees into
every agent; "stdlib-only" is a documented promise), an extras input on the Action (unneeded
once `doctor` warns), vendoring a client.

## SDK (evalshift-sdk 0.6.0)

### Startup validation

`_Config` is built at import (`config.py:181`) and `_env_sink()` runs then. Change `_env_sink()`
so a misconfigured `EVALSHIFT_SINK` is **recorded, not swallowed**: `_Config` gains
`sink_error: SinkConfigurationError | None`; `sink` stays `None` (the `FileSink` default is not
used while an error is recorded — see "Hot path"). Nothing is raised at import. The current
`WARNING` lines go away; the error carries the same text.

A blank or unset `EVALSHIFT_SINK` records nothing (unchanged). Recording is independent of the
gate (as building the env sink is today), but **nothing below happens unless `EVALSHIFT_CAPTURE`
is truthy**: with capture off, a broken `EVALSHIFT_SINK` never raises, never logs, never changes
behaviour.

### Touch points that raise

When `EVALSHIFT_CAPTURE` is truthy **and** `sink_error` is recorded, these raise it:

| Touch point | When it runs in a real agent |
|---|---|
| `capture.agent(...)`, `capture.agent_session(...)`, `capture.tool(...)` applied as decorators | the user's module import, i.e. process startup |
| `wrap_openai(...)`, `wrap_anthropic(...)`, `wrap_genai(...)` | client construction, startup |
| `EvalShiftCallbackHandler(...)` | handler construction, startup |
| `configure(...)` called **without** an explicit `sink` | startup |

`configure(sink=<explicit sink>)` clears `sink_error`: the user replaced the env sink with
something that exists. The gate is read live, so a process started with capture off never raises.

The check is one function in `config.py`, `require_sink_ready()`, that every touch point calls;
it is a no-op when the gate is off or nothing is recorded.

### Hot path

If a capture is attempted while `sink_error` is recorded (possible only when the gate turned on
after every touch point ran), the SDK **behaves as if the gate were off** for that invocation —
no span, no sidecar, no disk write — and logs the error text at `WARNING` once per process.
Rule: a sink the user asked for is never silently replaced by local disk.

### The explicit-sink path

`S3Store(...)`, `GCSStore(...)`, `AzureBlobStore(...)` still import their client lazily on the
first put, but a store constructed **without** an injected `client` now checks that its library
is importable (`importlib.util.find_spec`, no import, no network) and raises
`MissingStoreDependencyError` at construction — in user code, at startup — rather than failing
its first background upload. The error is renamed from `MissingExtraError` (same base
`ImportError`, same module `evalshift.stores.uri`) and its text names the pip package(s), not the
extra. `open_store()` and the new `require_store_modules(scheme)` raise the same. No alias for
the old name: 0.5.0 shipped today and the rename is listed under *Changed* in the changelog.

### Messages

All raised and logged text follows the existing rule: never the raw env value. Scheme, variable
name, package names and accepted forms only. A grammar error is raised `from None` so the
parser's quoted-URI message cannot appear as a chained cause.

| Case | `SinkConfigurationError` text |
|---|---|
| library missing | `EVALSHIFT_SINK points at an object store (s3://), but boto3 is not installed. Run: pip install boto3, or unset EVALSHIFT_SINK to capture to local disk.` |
| malformed URI | `EVALSHIFT_SINK is not a valid store URI. Accepted forms: s3://<bucket>/<prefix>, gs://<bucket>/<prefix>, az://<account>/<container>/<prefix>. Credentials never go in the URI.` |
| store failed to open | `EVALSHIFT_SINK could not be opened (<ExceptionType>). Unset it to capture to local disk.` |

Scheme → pip packages, kept **verbatim in both repos'** `uri.py` next to `_REQUIRED_MODULES`:

| scheme | required modules | `pip install` |
|---|---|---|
| `s3` | `boto3` | `boto3` |
| `gs` | `google.cloud.storage` | `google-cloud-storage` |
| `az` | `azure.storage.blob`, `azure.identity` | `azure-storage-blob azure-identity` |

The Azure message names both packages in one command even if only one is missing.

### Public API changes

- New `evalshift.SinkConfigurationError` (subclass of `RuntimeError`), exported at top level.
- `evalshift.stores.uri.MissingExtraError` → `MissingStoreDependencyError`.
- `StoreURI.extra` stays (the extras still exist) but gains `StoreURI.packages` → the
  `pip install` argument string above; messages use `packages`.
- Fail-open table in DOCS.md gains a row: *Sink configuration — raises `SinkConfigurationError`
  at startup — a bucket you asked for and did not get is the loss this feature exists to
  prevent.* DECISIONS.md D-stores: the "Warning-level logging" and "Adapters … Extras" bullets
  are rewritten accordingly (new sub-decision D-stores-b).

## CLI (evalshift 1.3.1)

- `RemoteStoreUnavailable` text: summary `s3:// captures store needs boto3, which is not
  installed`, hint `install it: pip install boto3` (packages from the shared table).
- `doctor`'s `captures.store` row: library missing → `warn` with that text; bucket not
  listable → `warn` (unchanged); listable → `ok`. `doctor` exits 1 only on an invalid existing
  config again. Docstrings and DOCS.md:872 updated.
- `capture fetch` / `list` / `sync`: unchanged — exit 1 with the new text when the library is
  missing (the user asked for a store the command cannot read).
- Consequence: `compare` and the Action no longer fail on a config that names a store.

## Docs

Every surface that says "extra", "`[s3]`" or "falls back to local disk" changes. Enumerate with
`git grep` at implementation time; known sites today:

- **sdk:** README.md, DOCS.md (install, env table, `ObjectStoreSink` section, fail-open table,
  troubleshooting), llms.txt, llms-full.txt, docs/DECISIONS.md, CHANGELOG.md.
- **cli:** docs/configuration.md (`captures` URI table "Install" column), docs/sdk.md, DOCS.md
  (`doctor` paragraph, capture section, exit-code sentence), llms-full.txt, CHANGELOG.md,
  `doctor.py` / `remote.py` / `uri.py` docstrings.
- **client (after both releases):** SdkConfig.tsx, Captures.tsx, Configuration.tsx, Changelog.tsx,
  version.ts, then `npm run sync:llms`.
- **action:** nothing (its docs never mention the extras; the troubleshooting entry in PR #27 is
  about pin skew and stays true).

The 2026-10-09 docs PRs (sdk #14, cli #42, action #27, server #36, client #55) are merged and
describe the extras-based install story of 0.5.0 / 1.3.0 accurately; the 0.6.0 and 1.3.1 PRs
rewrite those passages on top, and the site follows the releases with the follow-up above.

## Release order

SDK 0.6.0 and CLI 1.3.1 are independent; no server or action change. Ship both, then the client
follow-up, then `npm run sync:llms`.

## Testing

- **SDK** `tests/test_config_sink_env.py` is rewritten: bare import with a broken sink does not
  raise; each touch point raises `SinkConfigurationError` with the gate on and not with it off;
  `configure(sink=MemorySink())` clears the error; the secret-bearing-value test keeps asserting
  the value never appears in the exception text or the chained cause; the hot-path rule (gate
  turned on late → no file written, one `WARNING`); the Azure message names both packages;
  `MissingStoreDependencyError` from `S3Store()` names `pip install boto3`. The subprocess test
  gains a case proving `import evalshift` exits 0 with a broken `EVALSHIFT_SINK`.
- **CLI** `doctor` test: missing library → `warn`, exit 0; capture-command tests assert the new
  hint text; the docs-currency test gains `"[s3]"`-style extras and `"optional dependency"` as
  retired terms across the prose files.

## Judgement calls to confirm

1. Hot-path rule drops the capture rather than writing to disk when the gate turns on late.
   Alternative: keep today's disk fallback with the warning for that one case.
2. `MissingExtraError` is renamed without an alias.
3. `SinkConfigurationError` subclasses `RuntimeError`, not `ImportError`, because the malformed
   URI case is not an import problem; the library-missing case keeps `ImportError` semantics
   through `MissingStoreDependencyError` on the explicit path only.
