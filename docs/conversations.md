# Multi-turn conversations

EvalShift can evaluate multi-turn agent conversations, not just single-shot
prompts. A conversation is captured as one file per turn by the
`evalshift-sdk`; `evalshift capture sync` links the turns back together into
golden suite examples that carry a **teacher-forced** conversation prefix,
and `run` replays each turn against a candidate model with that prefix sent
verbatim.

## How the SDK records turns

Schema `1.1.0` of the capture envelope (the `evalshift-sdk` package, a
separate install from this CLI) added three optional fields so a multi-turn
conversation — one capture file per turn — can be linked back together:

| Field               | Type          | Meaning |
| ------------------- | ------------- | ------- |
| `conversation_id`   | `str \| None` | Shared identifier for every capture belonging to the same conversation. `None` for a standalone (single-turn) capture. |
| `turn_index`        | `int \| None` | This capture's position in the conversation (0-based). |
| `parent_capture_id` | `str \| None` | The `capture_id` of the immediately preceding turn's capture, if any. |

Record one capture per turn, passing fresh values each time:

```python
from evalshift import capture, record_model_call

# One capture per turn — pass fresh conversation_id/turn_index/parent_capture_id each time.
with capture.agent_session(
    suite="support_agent",
    redact=True,  # required — mask PII before it reaches disk
    tools=[],  # required — the toolset offered this turn
    agent_input=messages,  # the full messages list — see below
    conversation_id="conv_abc123",
    turn_index=2,
    parent_capture_id="cap_prev_turn_id",
):
    record_model_call(model_id="claude-opus-4-8", tools=None, input=messages, output=reply)
```

(`capture.agent_session_async(...)` is the `async with` equivalent, for async
agent code.)

### The messages-list convention for `model_call.input`

`capture.model_call(input=...)` (and `record_model_call(input=...)`) accept
`Any` — the SDK doesn't validate the shape. The **convention** for a
multi-turn agent is to pass the complete per-turn context as a list of
role-tagged messages, not just the latest message in isolation:

```python
messages = [
    {"role": "system", "content": "You are a scheduling assistant."},
    {"role": "user", "content": "Can we move my appointment?"},
    {"role": "assistant", "content": "Sure — what time works?"},
    {"role": "user", "content": "1pm"},  # the current turn's user message
]
```

i.e. the system prompt, every prior turn, and the current user message. This
is what lets the CLI recover both the conversation history and the
current-turn input from a single capture file, without needing the sibling
turn captures — see `capture promote`/`capture sync` below.

## Promotion: `capture sync`

`evalshift capture sync` groups captures by `conversation_id`, orders them by
`turn_index`, and builds one `SuiteExample` per turn:

* If a turn's own capture recorded a full messages list (the convention
  above), its `history` is recovered from that list: everything before the
  last `user` message becomes `history`, and the last `user` message becomes
  the current turn's `inputs`. The list is first translated from the
  provider's own message shape into the replay shape (see
  [Provider message shapes](#provider-message-shapes)), so its text and tool
  calls are kept exactly, in the form `run` replays.
* If a turn's capture only recorded a bare string or dict (no messages
  list), its `history` is **reconstructed** from the group's prior turns:
  each prior turn contributes a `user` message (its recovered current-turn
  text) and an `assistant` message (its recorded final output). If any prior
  turn recovered a `system` message from its own recorded messages list, the
  reconstructed history is seeded with it — the **earliest** turn's system
  prompt wins, and `capture sync` warns if later turns recorded a different
  one.

Both paths produce a `SuiteExample.history` list that `run` will replay
ahead of the current turn, but they are not equivalent: recovery from the
turn's own messages list reproduces the text and tool calls the model saw,
while reconstruction is an approximation stitched from sibling captures —
assistant replies come from recorded final outputs (intermediate tool
exchanges are absent), and the system prompt is only recovered when at least
one turn in the conversation recorded a full messages list. Prefer the
messages-list convention above when you can.

With `captures.max_cases` set, a conversation's new turns enter a suite
together when they fit; one larger than the suite's free slots is skipped
for the next one. Once no whole conversation fits, the slots left take the
leading turns of the next conversation, never a middle or trailing slice, so
each turn's earlier turns are in the suite with it. Under the limits, a turn
that recorded no messages list of its own is promoted only when every
earlier turn of its conversation is local, since its history is rebuilt from
them (see [Bounded suites](configuration.md#bounded-suites)).

```bash
evalshift capture sync                              # promote captures,
                                                      #   wire suites: into evalshift.yaml
evalshift run --suite-name support_agent --yes       # score a candidate model against it
```

`evalshift capture promote` (promoting a **single** capture id) does *not*
do this cross-capture reconstruction — it only recovers `history` from that
one capture's own messages list, if any. Promoting a multi-turn conversation
one capture at a time with `promote` produces examples with gaps in their
history; use `capture sync` to promote a whole conversation together. If you
promote a single mid-conversation capture and unpromoted sibling turns exist,
`capture promote` prints a warning pointing you at `capture sync`.

## Provider message shapes

The SDK's client wrappers record each model call's messages the way that
provider takes them, and `capture sync` / `capture promote` translate them into
the replay shape before splitting off `history`:

| Recorded by | What promotion reads |
| --- | --- |
| `wrap_anthropic` (Claude Messages) | `text` blocks become the message text; `tool_use` blocks become the assistant's `tool_calls`; each `tool_result` block becomes a `tool` message keyed by its `tool_use_id`; a `system` prompt sent as blocks (prompt caching) becomes its text. |
| `wrap_openai`, Chat Completions | `tool_calls[].function` becomes `{id, name, arguments}` with the JSON-string arguments parsed; `content: null` or no `content` key is an empty string; `developer` is read as `system` and merges into a leading system prompt, while a system or `developer` message after the first turn is dropped with a warning (a replay carries one leading system prompt); a `refusal` becomes the assistant's text. |
| `wrap_openai`, Responses | `message` items become messages; `function_call` items become the assistant's `tool_calls`; `function_call_output` items become `tool` messages keyed by `call_id`; `instructions=` and a `developer` message after it merge into one leading system prompt, and a system or `developer` message after the first turn is dropped with a warning. |
| `wrap_genai` (Gemini) | Already the replay shape. |
| LangChain callback | `human` / `ai` are read as `user` / `assistant`; block-list content (Claude through LangChain) is read as above. |

Thinking (`thinking`, `redacted_thinking`, Responses `reasoning`) is dropped:
the Claude API ignores earlier turns' thinking itself, and a signature means
nothing to a different candidate model.

A turn whose recorded messages carry anything a text replay can't carry — an
image, a document, audio, a server-tool result such as `web_search_tool_result`,
or a block or item type promotion doesn't know — is **not promoted**.
`capture sync` skips it, naming the block, and counts it as
`non-text content a text replay can't carry`; `capture promote` refuses it.
`--allow-errored` does not apply: a replay without the image is not ground
truth.

### Repairing suites promoted before this

Earlier CLI releases passed Claude's blocks through `str()`, so a promoted
follow-up turn's history held Python-repr strings such as
`"[{'id': 'toolu_1', … 'type': 'tool_use'}]"`, and OpenAI, Responses and
LangChain turns could come out with empty `inputs`. `capture sync` and
`validate` now warn when a suite still holds stringified blocks:

```text
⚠ support: 4 promoted case(s) carry message blocks an earlier sync stringified (e.g. cap_…); run evalshift capture sync --repair to rebuild them.
```

`evalshift capture sync --repair` rebuilds those cases, and any promoted case
left with empty `inputs`, from their captures. It replaces only each affected
case's `inputs` and `history` — the case's name, tags, `expected_*` and any
other hand edits stay, and so does the name of its single input variable (a
case promoted with `--input-var question` stays `question`; a case left with
empty `inputs` takes this run's `--input-var`) — and leaves every other
promoted case alone unless you also pass `--force`. Those two fields come back
exactly as the capture recorded them, so any hand edit you made to a rebuilt
case's `inputs` or `history` — a redaction, say — is replaced too: review the
diff before you commit it. `--repair` is a full sync: it also promotes any
new captures, exactly like a plain sync, so run it
with the flags you normally sync with (`--input-var`, `--tag`, `--strict-args`,
…). With `captures.store` set it re-downloads the capture behind each of those
cases, plus every other promoted turn of the same conversation, even though
they are promoted and even past `--since`. A case whose capture is gone from
both `.evalshift/captures/` and the store is named (`can't repair N case(s)`):
delete it or re-capture. A case with stringified blocks whose rebuild is now
refused — its turn carries an image, say — is named too
(`can't repair <case>: …`) and left as it is; a case left with empty `inputs`
whose rebuild is refused is left as it is too, but reported like any other
refused capture (`skipped <capture>: …`, counted in the summary). A turn
whose capture recorded no messages list, or one with no user turn to split
on, has its `history` reconstructed from its sibling turns (see
[Promotion](#promotion-capture-sync)); when any other
promoted turn of its conversation has a capture that is gone from both
places, `--repair` leaves the case as it is rather than rebuild it with a
shorter history, and names the missing turns:

```text
⚠ can't repair cap_t1: conversation turn(s) cap_t0 are no longer local or in the store, so its history can't be rebuilt
```

Without `--repair`, sync — `--force` included — leaves every case with
stringified blocks untouched, and every case left with empty `inputs` that a
rebuild would change (`left N case(s) for --repair`). A case whose capture
genuinely has no inputs rebuilds unchanged, so it is not a repair: sync treats
it like any other promoted case, and `--force` refreshes it.

A plain sync still recognises a capture that replays the same turn as a case
with stringified blocks when that case's own capture is not local (a fresh
clone, or after `capture clean`): the strings evaluate back to the recorded
blocks, so sync rebuilds the case's content offline and skips the capture as
a duplicate. A case left with empty `inputs` has nothing to rebuild from
without its capture, though, so run `--repair` before a plain sync on such
suites — otherwise the plain sync may promote a capture that replays the same
turn as a second case. If a repaired case now replays the same content as
another promoted case in its suite, `--repair` names the pair and deletes
nothing; delete one yourself:

```text
⚠ support: repaired case cap_1 duplicates cap_2 — delete one (duplicate examples inflate n)
```

Commit the regenerated `golden.jsonl` afterwards; CI reads the committed
suite.

## Teacher-forced replay semantics

When `run` dispatches an example whose `history` is not `None`, it sends the
recorded history prefix **verbatim**, followed by the current turn's
rendered prompt as the final `user` message — via
`ModelClient.complete_messages()` (or `complete_messages_with_tools()` for
agent-style prompts) instead of the plain single-string `complete()` path.
The candidate model never generates its own intermediate turns; each turn is
scored independently against the *same* recorded prefix the source
conversation actually took, and only the **current turn's** output is
compared between source and target.

Recorded **tool calls and tool results replay too**. An `assistant` turn's
`tool_calls` are sent in the OpenAI wire shape
(`tool_calls[].function.arguments` as a JSON string) and each `tool` message
is keyed by the `tool_call_id` it answers; LiteLLM translates that into each
provider's own form. Without this, a candidate model was asked to continue an
agent conversation with every tool call and result deleted from its context,
then scored against what a model *with* that context did.

This is deliberate: replaying the literal recorded prefix keeps every turn's
comparison a clean paired measurement (the two models see byte-identical
context up to the turn being scored), which is what the statistical pipeline
(paired tests, Cohen's d, BH correction — see `docs/methodology.md`) assumes.
Re-driving whole conversations — letting the candidate model's own replies
feed into the *next* turn's prompt — is not supported yet. It would mean two
models drift onto genuinely different conversations after turn 0, which
breaks the paired-comparison contract this tool is built on. It may come
later as an explicitly different (and differently-analyzed) mode.

## `SuiteExample` field reference

Three fields on a golden suite row carry conversation state (see
`docs/configuration.md` for the full suite schema):

| Field             | Type                    | Meaning |
| ----------------- | ----------------------- | ------- |
| `history`         | list of `{role, content, ...}` or `null` | Conversation prefix replayed verbatim before the current turn. `null` means single-turn (no message-mode dispatch). An empty list still triggers message-mode dispatch with no prefix. `role` is one of `system`, `user`, `assistant`, `tool`; at most one `system` message is allowed, and if present it must be first. |
| `history[].tool_calls` | list of `{id, name, arguments}` or absent | Tool calls the turn emitted. Only valid on an `assistant` message. |
| `history[].tool_call_id` | string or absent    | The `tool_calls[].id` this result answers. **Required** on a `tool` message, forbidden elsewhere — a result that can't be paired with its call is rejected at load time, not at dispatch. |
| `conversation_id` | string or `null`        | Id of the recorded conversation this example's turn came from. Provenance only — not used by dispatch. |
| `turn_index`      | integer or `null`       | Zero-based position of this turn within its conversation. Surfaced in the HTML report as a `turn N` badge. |

Single-turn suites (no `history`/`conversation_id`/`turn_index` fields at
all) parse and run exactly as before — these fields are purely additive.

## Limitations

* **A tool result with no recorded id is paired by order.** `tool` messages
  are replayed, but only if they can be paired with the call they answer.
  Some providers put no ids on the wire (Gemini's function responses, for
  one), so when the recording carried no `tool_call_id`, promotion keys the
  result to the next still-unanswered `tool_calls[].id` of the preceding
  assistant turn — the same by-order pairing the provider itself uses — and
  warns. A call that also had no id gets `call_p<N>_<i>` on both sides. Only
  a result with no preceding call to answer (after a later user turn, say)
  gets a positional `_pos<N>` id, which no provider can match; record the
  provider's call id if you hit that.
* **Reconstructed history is an approximation.** When a turn's capture only
  recorded a bare string, its prefix is stitched from sibling captures:
  assistant replies are the recorded final outputs (any intermediate tool
  calls and results are absent), and the system prompt is seeded only if
  some turn in the conversation recorded a full messages list — the earliest
  turn's system prompt is used, with a warning if later turns disagree. A
  conversation where *no* turn recorded a messages list replays with no
  system prompt at all.
* **No full-conversation re-drive.** As above — each turn is scored against
  the recorded prefix, not a prefix generated by the candidate model's own
  prior replies.
* **`capture promote` on a single capture doesn't reconstruct cross-capture
  history.** Use `capture sync` (or promote every turn of a conversation in
  one `capture sync` pass) to get recovered/reconstructed history spanning
  multiple capture files. A lone `capture promote` only ever looks at that
  one capture's own recorded messages list.
* **Responses conversations chained server-side lose their history.** An
  OpenAI Responses call that continues a conversation through
  `previous_response_id` or `conversation=` sends only the new input, so that
  is all the capture holds; the earlier turns can't be recovered.
* **LangChain over OpenAI loses tool rounds.** The SDK's LangChain callback
  records each message's role and content but not `tool_calls` or
  `tool_call_id`, and an OpenAI-backed `AIMessage` keeps its calls only
  there, so its tool results replay unpaired. LangChain over Claude keeps
  them: the `tool_use` blocks are in the content.
