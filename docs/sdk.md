# Capture SDK

The [evalshift-sdk](https://github.com/evalshift/evalshift-sdk) is a separate
package (`pip install evalshift-sdk`, import name `evalshift`) that you install
**inside your agent process**. It records what your agent actually did — model
calls, tool calls, the final output — as JSON capture files on disk. The CLI
promotes those captures into golden suites.

This page covers the CLI side of that contract. The full SDK guide lives in the
SDK repo: [DOCS.md](https://github.com/evalshift/evalshift-sdk/blob/main/DOCS.md),
dense LLM reference at <https://www.evalshift.dev/sdk-llms-full.txt>.

## The contract

```
your agent (evalshift-sdk)  →  .evalshift/captures/<suite>/cap_<hex>.json
                            →  or, EVALSHIFT_SINK=s3://… : <prefix>/captures/<suite>/cap_<hex>.json
                            →  evalshift capture sync       (fetches from captures.store first)
                            →  .evalshift/suites/<suite>/golden.jsonl + evalshift.yaml
```

- **The layout is the interface.** The SDK never imports or calls the CLI, and the CLI reads
  captures from disk without calling SDK code (`doctor` only checks which package the
  `evalshift` import name resolves to). The same layout — `captures/<suite>/cap_<hex>.json`
  plus `toolsets/<hex>.json` — is what the SDK writes to local disk *or* under a prefix in an
  object store you own; the CLI mirrors the latter into `.evalshift/` and reads it unchanged.
- **One environment or two.** The CLI imports as `evalshift_cli` and depends on
  the SDK, so `pip install evalshift` gives you both and `import evalshift` is
  always the SDK. A production agent that only records captures installs
  `evalshift-sdk` alone.
- **Off by default.** Nothing is recorded unless `EVALSHIFT_CAPTURE=1` is set,
  so the instrumentation is safe to leave in production code permanently.
- **No network by default.** The SDK writes local files unless `EVALSHIFT_SINK` names a bucket
  you own (`s3://`, `gs://`, `az://`, SDK 0.5.0+), and then it writes only there — never to
  EvalShift. Python 3.10+, stdlib-only; the cloud clients are optional extras.

## 1. Instrument the agent

Three primitives cover most agents: `@capture.agent` marks the agent boundary,
`@capture.tool` records a tool call, `record_model_call` records a completed
model call.

```python
from evalshift import capture, record_model_call


@capture.tool(name="issue_refund")
def issue_refund(order_id: str) -> dict:
    return {"status": "refunded", "order_id": order_id}


@capture.agent(suite="support_agent", redact=True, tools=[])
def handle_ticket(query: str) -> str:
    response = client.messages.create(model="claude-sonnet-5", messages=messages)
    record_model_call(
        model_id="claude-sonnet-5", tools=None, input=messages, output=response.text
    )  # tools=None inherits the session's toolset
    ...
```

`redact` is a **required** keyword on `@capture.agent` — and on
`capture.agent_session`, `capture.agent_session_async` and
`EvalShiftCallbackHandler` — as of SDK 0.3.0. Captures hold the inside of a run
(tool arguments and results, model input and output), so every capture point has
to state its masking policy in the call itself. `True` applies the SDK's
`default_redactor` (emails, `sk-…`, `AKIA…`, `Bearer …`), `False` records
verbatim, and a `(value) -> value` callable does something custom; any other
value — `None` included — raises `TypeError`. Details:
[REDACTION.md](https://github.com/evalshift/evalshift-sdk/blob/main/docs/REDACTION.md).

`tools` is required on the same entry points: the toolset the agent was offered,
or `[]` if it never calls tools. Omitting it is a `TypeError` too.
`record_model_call` and `capture.model_call` require `tools=` as well; pass
`None` to inherit the session's toolset.

**Provider client wrappers (SDK 0.4.0+).** If the agent calls OpenAI, Anthropic
or Google GenAI directly, wrap the client once instead of calling
`record_model_call` by hand — `wrap_openai(OpenAI())`, `wrap_anthropic(Anthropic())`,
`wrap_genai(genai.Client())` from `evalshift.adapters.<provider>`, installed with
`pip install "evalshift-sdk[openai]"` / `[anthropic]` / `[google-genai]`. Every
intercepted call — sync, async and streaming — records one `model_call` with
the model id, the tools offered, the calls the model **requested**
(`requested_tool_calls`), input, output, token usage, latency and the tool-use
generation settings (`tool_choice`, `parallel_tool_calls`, a tool's `strict`
flag). `wrap_openai` with a `base_url` covers OpenAI-compatible servers
(DeepSeek, Ollama, vLLM, Groq, OpenRouter). If you keep `record_model_call`, pass
`requested_tool_calls=extract_requested_tool_calls(response)` on every model
call: `capture sync` then scores against what the model *asked for* rather than
what the app executed (`promotion_source: requested`), and `run` replays the
case under the same tool-use constraints the source ran under. See
[Agent migrations](agents.md#requested-calls-are-the-ground-truth-when-they-were-captured).

Pass the **messages list** (not a bare string) as the model-call input where you
can: `capture sync` recovers conversation history verbatim from a messages list,
and only approximates it when all it has is a bare string. See
[Multi-turn conversations](conversations.md).

## 2. Record captures

```bash
EVALSHIFT_CAPTURE=1 python your_agent.py
```

Each sampled invocation writes one file to
`.evalshift/captures/<suite>/cap_<hex>.json`. The directory stays bounded on its
own — identical-input runs are de-duplicated and each suite dir keeps the 200
newest captures. `EVALSHIFT_MAX_CAPTURES`, `EVALSHIFT_DEDUP`,
`EVALSHIFT_CAPTURE_TTL` and `EVALSHIFT_SAMPLE_RATE` tune that; they are read by
the SDK, in your agent's process.

```bash
# on a host whose disk does not outlive it (Fargate, Lambda, pods):
EVALSHIFT_CAPTURE=1 EVALSHIFT_SINK=s3://acme-evals/support-agent python your_agent.py
```

The SDK ships each capture (and its toolset sidecar) to the bucket on a background thread,
fail-open; see the SDK docs for the `SIGTERM` / Lambda flush note. Bucket retention is a
lifecycle rule; `EVALSHIFT_MAX_CAPTURES` applies to local disk only.

## 3. Promote captures into suites

Run the CLI from the directory holding `.evalshift/` (or set `EVALSHIFT_DIR`).
If the captures live in a bucket, name it once in `evalshift.yaml`
(`captures: {store: s3://acme-evals/support-agent}`) and install `evalshift[s3]` (or `[gcs]` /
`[azure]`); `list` and `sync` then fetch new captures first. `evalshift capture fetch` does
only that step.

```bash
evalshift capture list                  # what was recorded (--json for machine output)
evalshift capture fetch                 # mirror new captures from captures.store
evalshift capture sync                  # promote ALL captures → suites + wire evalshift.yaml
evalshift capture promote cap_ab12 --as refund_case_1
evalshift capture diff cap_ab12 cap_cd34
evalshift capture clean                 # delete already-promoted capture files
```

`capture sync` groups captures (by `conversation_id`/`turn_index` for
multi-turn), turns each into a suite example — first model input → `inputs`,
recorded tool calls → `expected_tools`, final output → `expected`, messages list
→ `history` — skips duplicate content across runs, writes
`.evalshift/suites/<suite>/golden.jsonl`, and rewrites the managed `suites:`
block in `evalshift.yaml`.

Strictness knobs for the derived tool expectations: `--strict-args`,
`--names-only`, `--tool-count`, and `--rounds first|all` — whether only the
first agent round or every recorded round becomes ground truth; `all` also
carries the recorded tool results so `run` replays later rounds teacher-forced
(see [Agent rounds](agents.md#agent-rounds-and-what-a-replay-can-reproduce)).
`--tag` adds slice tags, `--print` previews the config block without writing,
`--keep-duplicates` disables dedup.

Full behaviour: [Configuration](configuration.md) and the
[Capturing from production](https://github.com/evalshift/evalshift-cli/blob/main/DOCS.md#capturing-from-production)
section of DOCS.md.

## 4. Evaluate a candidate against real behaviour

```bash
evalshift compare --suite-name <suite> --to <candidate-model>
```

That is a normal EvalShift run — the only difference is that the suite came
from production traffic rather than hand-written examples.

## Related

- [`examples/capture-first/`](https://github.com/evalshift/evalshift-cli/tree/main/examples/capture-first)
  — every step above checked in: the instrumented agent, the captures and
  toolset sidecar it wrote, the promoted suite, and the `evalshift.yaml` whose
  managed `suites:` block `capture sync` filled in.
- [Getting started](getting-started.md) — the capture-first `evalshift init` flow.
- [Multi-turn conversations](conversations.md) — how history is recovered.
- **Provider clients** need no per-call code: `wrap_openai` / `wrap_anthropic` /
  `wrap_genai` (SDK 0.4.0+, extras `[openai]`, `[anthropic]`, `[google-genai]`)
  proxy a client you already built and record every call, streams included.
- **LangChain** needs no decorators: the SDK ships an `EvalShiftCallbackHandler`
  (`pip install "evalshift-sdk[langchain]"`) that records a chain or agent run
  from the `callbacks=[...]` list, into the same captures as step 1.
- [Agent traces](traces.md) — for agents the SDK cannot instrument at all
  (another language, or a framework with no adapter): import full timelines with
  `evalshift traces import`.
