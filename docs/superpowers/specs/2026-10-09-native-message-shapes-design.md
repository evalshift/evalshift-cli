# Capture promotion: read every wrapped provider's native message shape, and repair what it broke

**Date:** 2026-10-09 · **Status:** draft for maintainer review · **Repos:** evalshift-cli (behaviour + docs); evalshift-sdk, evalshift-server, evalshift-client, evalshift-action (nothing)

## Problem

`capture sync` (and `capture promote`) turn a capture's first `model_call.input` into a case's
`inputs` + `history`. That code — `_recover_inputs` / `_recover_inputs_from_messages` /
`_history_message` in `captures/promote.py` — only understands one message shape: `content` a
string, assistant `tool_calls: [{id, name, arguments: dict}]`, results as `role: "tool"` with a
`tool_call_id`. Of the SDK's four adapters, only `wrap_genai` records that shape
(`adapters/genai.py:179`). The others record the provider's own wire shape, by design
(`wrap_anthropic` folds `system` in and keeps "block-list message content" as-is). Nothing
checks the two sides agree, and `ChatMessage.content: str` (`suite/models.py:76`) accepts
anything `str()` produces.

Reproduced end to end on 2026-10-09 (SDK 0.5.0 wrappers with scripted responses → CLI 1.3.0
`capture sync`):

| Recorded shape | What promotion does today |
|---|---|
| Claude, follow-up turn (assistant `tool_use` / `thinking` blocks, user `tool_result` blocks) | `content=str(list)` (`promote.py:980`): history holds Python-repr strings (`"[{'id': 'toolu_1', … 'type': 'tool_use'}]"`); no `tool_calls`; results become `user` messages; thinking blocks *and their signatures* end up in the text. **No warning; `validate` passes.** |
| Claude, first turn with a prompt-cached `system` (`[{"type": "text", …, "cache_control": …}]`) or text-block user content | The system message / the current turn's `inputs` are repr strings. No warning. |
| OpenAI Chat, assistant turn appended via `message.model_dump()` (`content: None`, nested `tool_calls[].function`) | `HistoryToolCall(name="")` fails validation and **the whole `capture sync` aborts with a traceback, exit 1.** |
| OpenAI Chat, `model_dump(exclude_none=True)` (no `content` key) | `_looks_like_messages_list` rejects the list → "could not recover structured inputs", `inputs` empty, history reconstructed from sibling turns without the tool round. |
| OpenAI Responses (`function_call` / `function_call_output` items have no `role`) | Same as above: empty `inputs`. |
| LangChain (roles recorded as `message.type`: `human`, `ai`) | No `user` message found → empty `inputs`, even on a first turn. |
| Gemini | Correct. |

The suite schema is fine — `run` already replays `tool_calls` / `tool` messages through LiteLLM
in every provider's own form (`docs/conversations.md`, "Teacher-forced replay semantics"). The
bug is entirely in reading the capture.

**Remote stores make repair harder.** With `captures.store` set, `capture sync` mirrors the
bucket's capture JSON byte-for-byte into `.evalshift/captures/` and promotes from there
(`_maybe_fetch` → `fetch_captures`), so a reader-side fix reaches bucket captures unchanged.
But a promoted case file (`.evalshift/suites/<suite>/<name>.json`) holds the *derived* example,
not the raw capture, and `fetch_captures` deliberately skips every promoted capture id
(`captures/remote.py:313`, so `capture clean` is not undone). On a fresh clone, in CI, or after
`capture clean`, `sync --force` therefore cannot regenerate a broken case: its raw capture never
comes back down.

## Decision

- **Normalize on the read side, in one new module, by shape.** `captures/native_messages.py`
  translates a recorded messages list into the shape promotion already expects, keyed on each
  message's *shape* (block-list content, `tool_calls[].function`, Responses item `type`, role
  aliases) — never on the model id or adapter, which lie behind gateways (LiteLLM, OpenRouter),
  LangChain, and hand-written `record_model_call`. Already-canonical input (Gemini, hand-built
  lists) passes through unchanged. Captures already sitting in buckets are fixed on the next sync;
  an SDK-side fix would only reach captures recorded after each app redeploys.
- **Content is a string by construction.** No code path `str()`s message content any more;
  `_history_message` raises `TypeError` on anything else, so a repr string cannot reach a suite.
- **Thinking is dropped silently.** The Claude API itself ignores prior turns' thinking, and a
  signature is meaningless to a different candidate model.
- **Non-text content blocks the case.** An image, document, audio part, server-tool result, or
  any block / item type the normalizer does not know makes the capture *blocked* with a new
  reason, `unreplayable_content`. A text replay of a turn whose model saw an image is not ground
  truth. `--allow-errored` does not rescue it. Unknown types block rather than drop, so a future
  block type fails loudly instead of silently thinning the context.
- **Broken suites announce themselves, and `--repair` fixes them.** `capture sync` and `validate`
  detect this bug's signature in promoted cases and say how many and what to run;
  `capture sync --repair` re-fetches exactly the captures it needs (past the promoted-id skip, for
  S3 / GCS / Azure stores) and rewrites only the affected cases.

Not doing: changing the SDK adapters (raw recording stays the faithful record; one normalizer, not
four); recording LangChain `tool_calls` / `tool_call_id` (SDK follow-up, see Limitations);
multimodal `ChatMessage.content`; recovering Responses history chained through
`previous_response_id` / `conversation=` (never recorded); auto-repair on a plain sync (sync does
not overwrite promoted cases unasked); warnings in `run` / `compare`.

## The normalizer — `evalshift_cli/captures/native_messages.py`

```python
@dataclass(frozen=True, slots=True)
class NormalizedMessages:
    messages: list[dict[str, Any]]  # canonical messages, see below
    unreplayable: list[str]         # e.g. "recorded message 3 (user) carries an `image` block"
    warnings: list[str]             # non-fatal notes for the operator

def normalize_messages(raw: list[Any]) -> NormalizedMessages | None: ...
```

Returns `None` when `raw` is not a messages list — any element that is not a dict, or a dict with
neither a `role` nor a recognised Responses item `type`. The caller then falls back to today's
"could not recover structured inputs" warning. Pure: no I/O, no globals.

**Canonical message** (what comes out, and what passes through untouched):
`{"role": str, "content": str}`, plus on assistant messages optionally
`"tool_calls": [{"id": str | None, "name": str, "arguments": dict}]`, plus on tool messages
optionally `"tool_call_id": str`. Roles the normalizer does not alias are passed through as-is, so
`_recover_inputs_from_messages` keeps its existing "dropped N history message(s) with unrecognised
role(s)" warning. Positions in messages are 1-based indexes into the *recorded* list.

### Roles

| Recorded | Canonical |
|---|---|
| `user`, `human` | `user` |
| `assistant`, `ai`, `model` | `assistant` |
| `system`, `developer` | `system` |
| `tool`, `function` | `tool` |

(`developer` is the system role OpenAI's reasoning models use; today it is dropped as
unrecognised.)

### Content

`content` may be a string (kept), `None` / absent (→ `""`), a single block dict (treated as a
one-element list), or a list. List elements are processed in order:

| Element | Becomes |
|---|---|
| a bare string (LangChain allows these) | text |
| `type` ∈ `text`, `input_text`, `output_text` | its `text` |
| `type == "refusal"` | its `refusal` |
| `type` ∈ `thinking`, `redacted_thinking` | dropped, no warning |
| `type == "tool_use"` (assistant) | a tool call `{id, name, arguments: input or {}}` |
| `type == "tool_result"` (user) | a separate `tool` message — see below |
| anything else (`image`, `document`, `image_url`, `input_image`, `input_file`, `input_audio`, `file`, `server_tool_use`, `web_search_tool_result`, unknown) | **unreplayable**: `"recorded message N (<role>) carries a `<type>` block"` |

Text pieces are joined with `"\n"` (the Gemini adapter's join). Block keys other than the ones
read here — `cache_control`, `citations`, `signature`, … — are ignored. An OpenAI Chat assistant
message whose content comes out empty but which carries a string `refusal` key uses that as its
content. Other message keys (`name`, `audio`, `annotations`, the legacy `function_call`, …) are
ignored.

**Claude `tool_result`.** Each becomes `{"role": "tool", "tool_call_id": tool_use_id, "content":
…}`, emitted in block order *before* any text that shared its user message (which becomes one
following `user` message; a user message of only results emits no user message). Its `content` is
a string (kept), absent (`""`), or a block list processed by the same table — so an image inside a
result is unreplayable. `is_error` is not carried: the replay shape has no error flag (LiteLLM
drops it the same way).

**Assistant `tool_calls` already present on the message:**

| Entry | Becomes |
|---|---|
| canonical `{id, name, arguments: dict}` | unchanged |
| OpenAI Chat `{id, type: "function", function: {name, arguments}}` | `{id, name: function.name, arguments}`; a string `arguments` is `json.loads`-ed; a dict is kept |
| `arguments` string that is not valid JSON, or decodes to a non-object | `arguments: {}` + warning: ``recorded message N: tool call `search` (call_1) had arguments that are not a JSON object; replayed with {}`` |

Tool calls from `tool_use` blocks are appended after any already on the message.

### OpenAI Responses items

Elements of the recorded list that carry a `type`:

| Item | Becomes |
|---|---|
| `{"type": "message", "role": …, "content": …}` (or a plain `{role, content}`) | a message, content per the table above |
| `function_call` `{call_id, name, arguments}` | a tool call `{id: call_id, name, arguments}` (arguments as for OpenAI Chat) appended to the last normalized message when that is an assistant message; otherwise a new `{"role": "assistant", "content": ""}` carrying it |
| `function_call_output` `{call_id, output}` | `{"role": "tool", "tool_call_id": call_id, "content": …}`; `output` a string is kept, a part list goes through the content table |
| `reasoning` | dropped, no warning |
| any other `type` (`web_search_call`, `file_search_call`, `computer_call`, `computer_call_output`, `image_generation_call`, `code_interpreter_call`, `mcp_call`, `mcp_list_tools`, `custom_tool_call`, `item_reference`, unknown) | **unreplayable**: `"recorded item N carries a `<type>` item"` |

### Invariants (tested)

- Output messages always have a string `content`.
- `normalize_messages(canonical) == canonical` for any already-canonical list.
- `normalize_messages` applied to its own output returns it unchanged.
- An unreplayable element never raises; it is reported and skipped, so the rest still normalizes
  (the case is blocked, but conversation grouping still sees a well-formed example).

## Wiring it into promotion — `captures/promote.py`

- `_recover_inputs`: when the first `model_call.input` is a list, call `normalize_messages`. On a
  result, extend `warnings` with its warnings and recover from `result.messages` via
  `_recover_inputs_from_messages` (unchanged pairing logic: id-less results still pair in order
  with open calls, unpaired ones still get synthetic ids). `None` → today's warning.
  `_looks_like_messages_list` goes away. The dict and string paths are unchanged.
- `RecoveredInput` gains `unreplayable: list[str]` (default empty).
- `_history_message`: `content` must be a `str` — anything else raises `TypeError` (an internal
  invariant: the normalizer guarantees it). Tool-call entries are read in canonical form only.
- `build_example_from_capture`: a new branch after the multi-toolset check and before the errored
  check — when `recovered.unreplayable` is non-empty, `blocked =` ``f"capture {id!r} carries
  content a text replay can't carry — {first}{' (+N more)'}. A replay without it is not ground
  truth; re-capture without it or hand-write the case."``, `blocked_reason =
  "unreplayable_content"`. It takes precedence over `errored` because `--allow-errored` cannot
  rescue it.
- `blocked_reason`'s `Literal` (on `BuiltExample` and the local in `build_example_from_capture`)
  gains `"unreplayable_content"`; the `BuiltExample` docstring documents it.

`cli/commands/capture.py`:

- `capture sync` routes the new reason to its own counter — not the `else` branch, which today
  counts everything unrecognised as `errored` and would point at `--allow-errored`. Summary clause:
  `skipped N capture(s) with non-text content a text replay can't carry (--allow-errored does not
  apply)`.
- `capture promote` already prints `built.blocked` and only offers `--allow-errored` when
  `blocked_reason == "errored"`; it needs a test, not a change.

## Detecting broken cases — `evalshift_cli/captures/repair.py`

```python
def stringified_content(example: SuiteExample) -> str | None: ...
def is_repair_candidate(example: SuiteExample) -> bool: ...
```

- `stringified_content` returns where the bug's signature is — `"history message 3"` or
  `"inputs['input']"` — or `None`. The signature: a history `content` or a string value in
  `inputs` that, after `lstrip()`, starts with `[`, evaluates with `ast.literal_eval` (safe; any
  exception means "no"), and yields a non-empty list whose every element is a dict with a string
  `"type"`. That is exactly what `str()` of a block list produces; a user message that merely
  contains JSON text does not match unless it is itself a list of typed blocks.
- `is_repair_candidate` is `stringified_content(...) is not None` **or** `inputs == {}` (what the
  OpenAI `exclude_none`, Responses, and LangChain failures wrote). Empty inputs are *not* part of
  the warning signature: a genuinely input-less case would nag on every sync, and `validate`
  already fails a case missing a prompt's template variables.

**Warnings.**

- `capture sync`, per suite, from the promoted cases it already reads for dedup
  (`_promoted_content_owners`): `⚠ support: 4 promoted case(s) carry message blocks an earlier
  sync stringified (e.g. cap_…); run evalshift capture sync --repair to rebuild them.`
- `validate`, over the loaded suite's examples, as an advisory after the green check (exit code
  unchanged, like the CI-pin and judge-family advisories): `⚠ 4 example(s) carry message blocks an
  earlier capture sync stringified (e.g. cap_…); run evalshift capture sync --repair.`

**A plain sync leaves candidates alone.** A capture whose id is the `from_capture` of a repair
candidate is not promoted by a plain `sync` (it is counted with the already-promoted). Without
this, a case promoted under a custom name (`capture promote --as`) would get a second, correct
copy next to the broken one, because its rebuilt content no longer matches the stored content
that dedup compares against.

## `capture sync --repair`

A normal sync plus three things, in this order:

1. **Select.** Read the promoted cases (honouring `--suite`); the candidates are those
   `is_repair_candidate` accepts. The *refetch set* is the candidates' `from_capture` ids plus the
   `from_capture` of every promoted case sharing a `(suite, conversation_id)` with a candidate —
   a turn whose history is reconstructed from sibling turns needs them local.
2. **Fetch.** When a store is configured and `--offline` is not set, `fetch_captures` takes a new
   `force_ids` collection: ids in it bypass the promoted-id skip **and** `--since`; size cap,
   already-local, and malformed-key rules still apply. `_maybe_fetch` / `_fetch_into_base` pass the
   refetch set through. Toolset sidecars are fetched as today (they are never skipped as promoted).
3. **Rewrite.** In the promotion loop, a capture that is the `from_capture` of a candidate is
   written under **that case's existing name** (custom names are kept) with `force=True`,
   bypassing the dedup owner check (the owner is the case itself) — but only when the rebuilt
   example differs from the stored one. Every other promoted case keeps today's no-overwrite rule;
   siblings fetched only for context are not rewritten.

Reporting, added to the summary: `repaired N case(s)`. Then, as warnings:

- candidates whose capture is neither local nor in the store: `can't repair N case(s) — their
  capture is no longer local or in the store (cap_…): delete the case file or re-capture`;
- candidates whose rebuild is blocked (e.g. the turn carries an image): the blocked message, and
  the case is left as it is;
- candidates that rebuilt identically (a genuinely input-less case): nothing.

With no candidates: `no promoted cases need repair`, then a normal sync. `--repair --force` is
`--force` plus the refetch set. `--repair --offline` repairs from local captures only. The
`--help` text says a repaired case's hand edits are overwritten. Trace invariants live in
`evalshift.yaml`'s managed region and are carried forward as on any sync.

## Docs

- `docs/conversations.md`:
  - replace "its `history` is recovered **verbatim**" / "verbatim recovery reproduces exactly what
    the model saw" with: translated from the provider's own message shape into the replay shape,
    text and tool calls preserved; a per-shape table (Claude Messages, OpenAI Chat, OpenAI
    Responses, Gemini, LangChain);
  - thinking / reasoning is dropped; a turn with non-text content is skipped with the reason;
  - a "Repairing suites promoted before this fix" section: the warning, `capture sync --repair`,
    the remote-store re-fetch, the can't-repair cases;
  - Limitations: Responses conversations chained through `previous_response_id` or
    `conversation=` record only the new input, so their history cannot be recovered;
    LangChain over OpenAI loses tool rounds because the LangChain adapter records no `tool_calls`
    or `tool_call_id` (LangChain over Claude keeps them — the blocks are in `content`).
- `docs/configuration.md` and `docs/sdk.md`: `--repair` next to the other `capture sync` flags
  they document; the new skip reason wherever the existing ones are listed.
- `CHANGELOG.md`: `Fixed` (the shapes, the crash) and `Added` (`--repair`, the warnings).
- `tests/unit/test_docs_currency.py` must stay green; nothing in evalshift-client mirrors these
  pages.

## Testing

- **Normalizer unit tests** (`tests/unit/test_native_messages.py`), table-driven, one row per rule
  above: each role alias; string / `None` / absent / single-dict / list content; every text block
  type; thinking and `redacted_thinking` dropped; `tool_use` → `tool_calls`; `tool_result`
  split-out with string, block-list, and image content; text after results; cached system blocks;
  OpenAI nested `tool_calls` with JSON-string, dict, invalid-JSON, and non-object arguments;
  `content: None` and absent; `developer`; `refusal`; Responses `message` / plain item /
  `function_call` merging into a preceding assistant message and opening a new one /
  `function_call_output` / `reasoning` / an unreplayable item type; mixed LangChain roles with
  Claude blocks; a non-dict element → `None`. Plus the three invariants (canonical passthrough,
  idempotence, string content) as parametrized checks over every row's output.
- **Fixture captures** generated by the real SDK 0.5.0 `wrap_anthropic` and `wrap_openai` with
  scripted responses (two-turn tool-using conversations; OpenAI in both `model_dump()` and
  `exclude_none` variants), committed under `tests/fixtures/captures/native_shapes/` with the
  generating script beside them, so they can be regenerated against a future SDK.
- **Sync end to end:** the Claude conversation's second turn has `tool_calls` on the assistant
  message, a `tool` message keyed `toolu_1`, no repr string anywhere, and `validate` passes with
  no advisory; both OpenAI variants promote with recovered `inputs` (and the `model_dump()` one no
  longer crashes); a capture with an `image` block is skipped and counted under the new reason;
  `capture promote` on it refuses without suggesting `--allow-errored`.
- **Repair end to end** (`tests/unit/fake_remote_store.py`):
  - fresh clone — only broken case files locally, raw captures only in the store: `--repair`
    fetches exactly the refetch set, rewrites the broken cases, leaves an unflagged hand-edited
    case byte-identical, and fetches but does not rewrite a conversation sibling;
  - a custom-named (`--as`) broken case is rewritten under its own name, with no second case;
  - a candidate missing from the store produces the can't-repair warning;
  - `--since` does not hide a candidate; `--offline` repairs from local captures only;
  - a plain sync prints the warning and writes nothing for candidates.
- **Detection unit tests:** the signature matches repr'd block lists in history and in `inputs`;
  does not match JSON text, a repr'd list of untyped dicts, or ordinary prose starting with `[`;
  never raises on malformed input.
- **Guard rail:** `_history_message` raises `TypeError` on list content.

## Rollout

One evalshift-cli PR from a branch off `main`; ships in the next CLI release (minor — new flag).
Users repair by upgrading the CLI they run `capture sync` with and running `capture sync
--repair`, then committing the regenerated `golden.jsonl`; the Action only reads the committed
suite, so it needs nothing. Follow-up, separate spec: have the SDK's LangChain adapter record
`tool_calls` and `tool_call_id`.
