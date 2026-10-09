"""Agent-instruction wiring for ``evalshift init``.

Coding agents (Claude Code, Codex, Gemini CLI, Cursor, Copilot) drive the
``evalshift`` CLI far better when the project tells them the command map, the
canonical pipeline flow, and which commands cost money or delete data. ``init``
handles that with zero extra user setup:

1. It writes a standalone :data:`AGENT_INSTRUCTIONS_FILENAME` (``EVALSHIFT.md``)
   at the project root — a tool-owned guide, refreshed on every ``init``.
2. It scans for the agent-context files each vendor reads
   (:data:`AGENT_CONTEXT_FILES`) and appends an idempotent, marker-delimited
   pointer to ``EVALSHIFT.md`` into each one that exists — creating
   :data:`DEFAULT_AGENT_CONTEXT_FILE` (``AGENTS.md``, the cross-vendor standard)
   when none are present.

The pointer block is delimited by :data:`POINTER_MARKER_BEGIN` /
:data:`POINTER_MARKER_END` so re-running ``init`` rewrites the block in place
instead of appending duplicates, and never touches the human-authored rest of
the file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from rich.console import Console

AGENT_INSTRUCTIONS_FILENAME: Final = "EVALSHIFT.md"

POINTER_MARKER_BEGIN: Final = "<!-- evalshift:begin -->"
POINTER_MARKER_END: Final = "<!-- evalshift:end -->"

# Hosted llms.txt documentation, served by the EvalShift web app. Linked from
# the pointer block so coding agents can pull full reference docs on demand —
# but only when the task actually touches EvalShift, hence the guard sentence
# in ``_render_pointer_block``.
CLI_DOCS_URL: Final = "https://www.evalshift.dev/cli-llms-full.txt"
SDK_DOCS_URL: Final = "https://www.evalshift.dev/sdk-llms-full.txt"
CI_DOCS_URL: Final = "https://www.evalshift.dev/ci-llms-full.txt"

# Agent-context files we wire a pointer into when they already exist, in scan
# order. Each vendor reads its own filename; AGENTS.md is the cross-vendor
# standard and doubles as the fallback host below.
AGENT_CONTEXT_FILES: Final[tuple[str, ...]] = (
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".cursorrules",
    ".github/copilot-instructions.md",
)

# Created as the pointer host when none of AGENT_CONTEXT_FILES exist.
DEFAULT_AGENT_CONTEXT_FILE: Final = "AGENTS.md"

# The standalone guide. Tool-owned: refreshed verbatim on every ``init`` so it
# tracks the current CLI. Authoritative flags always come from ``--help``; this
# is the orientation layer that tells an agent *which* command and *when*.
AGENT_INSTRUCTIONS: Final = f"""\
# EvalShift — CLI guide for coding agents

EvalShift is a local-first CLI for **safe LLM model migrations**. It runs the
same prompts on two models against a golden suite, scores the outputs
(structural / semantic / LLM-judge / tool-call evaluators), and reports paired
statistics so you can tell whether a migration regressed.

> This file is the orientation layer. Run `evalshift --help` and
> `evalshift <command> --help` for the authoritative, current flags.

## Documentation

Full reference documentation (llms.txt format):

- EvalShift CLI: {CLI_DOCS_URL}
- EvalShift SDK: {SDK_DOCS_URL}
- EvalShift GitHub Action (CI): {CI_DOCS_URL}

Fetch these **only** when the task involves the EvalShift CLI, the EvalShift
SDK, or EvalShift in CI (e.g. `evalshift.yaml`, `evalshift` commands, golden
suites, SDK instrumentation/captures, or the EvalShift GitHub Action in a
workflow). Do not fetch them for unrelated work.

## Setup

- Requires Python 3.11+ and provider API keys in the environment
  (e.g. `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `DEEPSEEK_API_KEY`) matching the models set in `evalshift.yaml`.
- Config lives in `evalshift.yaml` at the project root. Run `evalshift doctor`
  to validate config + keys before a paid run.
- Generated data lives under `.evalshift/` and is gitignored — never commit it.

## The pipeline (canonical flow)

```
init  ->  doctor  ->  run  ->  evaluate  ->  analyze  ->  report
```

Each stage writes one artifact under `.evalshift/runs/<run-id>/` and the next
stage reads it; stages are independently re-runnable.

- `evalshift compare` drives doctor -> run -> evaluate -> analyze -> report end
  to end, comparing two models on ONE suite. (`all` is the former name, still
  accepted; it never meant "every suite".)

## Recipes

**I have captures and want a report (and to push it to hosted):**

```
evalshift capture sync     # promote captures into a suite AND wire evalshift.yaml
evalshift compare          # doctor -> run -> evaluate -> analyze -> report (report.html)
evalshift push <run-id>    # optional: upload the run to hosted EvalShift
```

Bare `evalshift compare` auto-selects the suite when `evalshift.yaml` wires
exactly one (the common case after `capture sync`). It never runs every suite:
with several wired it prints a ready-to-run command per suite, and you add
`--suite-name <name>`. To cover them all, loop over the names.

`capture sync` **is** the backfill — do not do it by hand. It renders each
capture's model input, attaches `expected_tools` / `expected_no_tools` from the
recorded tool calls, and writes the `suites:` block into `evalshift.yaml`. Tune
matching with its flags (`--input-var`, `--strict-args`, `--names-only`,
`--tool-count`); see `evalshift capture sync --help`. After it runs, go straight
to `evalshift compare` — you do **not** need to write case inputs, edit the
`suites:` block, or touch any capture file yourself.

## Rules for agents

- **Prefer `evalshift compare`.** It runs doctor -> run -> evaluate -> analyze ->
  report in one go, for one suite. Only drop to individual stages (`run`,
  `evaluate`, ...) when re-running a single stage after a fix — never
  reimplement the pipeline by hand.
- **Never create, edit, or backfill anything under `.evalshift/`** (captures,
  runs, cache, artifacts). It is generated state the CLI owns and reads. To turn
  captures into eval cases, run `evalshift capture sync` — not manual JSON/YAML
  edits. If a case input looks empty, that is expected: `capture sync` fills it.
- **Let commands own `evalshift.yaml`'s `suites:` block** — `capture sync`
  writes it. Editing evaluators / policy / prompts by hand is fine and expected.
- Run `evalshift <command> --help` before guessing flags.

## Commands

| Command | What it does |
| --- | --- |
| `evalshift init` | Scaffold a minimal `evalshift.yaml` (this guide's writer). |
| `evalshift doctor` | Validate local config + API keys. |
| `evalshift run` | Dispatch prompt x example x model calls -> `raw.jsonl`. |
| `evalshift evaluate` | Score `raw.jsonl` -> `scores.jsonl`. |
| `evalshift analyze` | Paired stats per (prompt, evaluator, slice) -> `analysis.json`. |
| `evalshift report` | Render the single-file HTML report. |
| `evalshift compare` | Run doctor -> run -> evaluate -> analyze -> report end to end on one suite (formerly `all`). |
| `evalshift inspect` | Inspect a run / artifact. |
| `evalshift capture ...` | `list` / `fetch` / `promote` / `clean` / `diff` / `sync` captures into suites. |
| `evalshift diff case` | Diff a single case between runs. |
| `evalshift replay case` | Replay a single case. |
| `evalshift traces import` | Import external traces. |
| `evalshift login` / `logout` / `whoami` | Hosted-server auth. |
| `evalshift bundle` / `push` | Build + upload a run to hosted EvalShift. |
| `evalshift runs clean` | Prune old local run directories. |
| `evalshift cache clear` | Clear the local model-response cache. |

## Hosted flow (optional)

`evalshift login` -> `evalshift bundle` -> `evalshift push` uploads a run to the
hosted server for diffing, gating, and sharing. Needs `EVALSHIFT_TOKEN` (or an
interactive `login`).

## Safety — confirm with the human before these

- **Costs money:** `evalshift run` and `evalshift compare` always call real
  models. They prompt above $10 unless `--yes` is passed; do **not** pass
  `--yes` on a paid run without explicit approval.
- **Deletes local data:** `evalshift runs clean`, `evalshift cache clear`.
- **Publishes externally:** `evalshift push` uploads a run to the hosted server.
"""


def _render_pointer_block(*, claude_import: bool) -> str:
    """Render the marker-delimited pointer block for one agent-context file.

    ``claude_import`` emits Claude Code's ``@path`` import form (which inlines
    the file into context) instead of a plain prose reference.
    """
    ref = f"./{AGENT_INSTRUCTIONS_FILENAME}"
    reference = f"@{ref}" if claude_import else ref
    return (
        f"{POINTER_MARKER_BEGIN}\n"
        f"## EvalShift CLI\n\n"
        f"This project uses the EvalShift CLI. Run `evalshift --help`; the full "
        f"agent guide is {reference}\n\n"
        f"Full reference documentation (llms.txt format):\n\n"
        f"- EvalShift CLI: {CLI_DOCS_URL}\n"
        f"- EvalShift SDK: {SDK_DOCS_URL}\n"
        f"- EvalShift GitHub Action (CI): {CI_DOCS_URL}\n\n"
        f"Only fetch these documents when the task involves the EvalShift CLI, "
        f"the EvalShift SDK, or EvalShift in CI (e.g. `evalshift.yaml`, "
        f"`evalshift` commands, golden suites, SDK instrumentation/captures, "
        f"or the EvalShift GitHub Action in a workflow). Do not fetch them "
        f"for unrelated work.\n"
        f"{POINTER_MARKER_END}\n"
    )


_BLOCK_RE: Final = re.compile(
    re.escape(POINTER_MARKER_BEGIN) + r".*?" + re.escape(POINTER_MARKER_END) + r"\n?",
    re.DOTALL,
)


def _strip_pointer_block(text: str) -> tuple[str, bool]:
    """Remove any existing pointer block; return ``(new_text, had_block)``."""
    new_text, count = _BLOCK_RE.subn("", text)
    return new_text, count > 0


def _upsert_pointer(path: Path, block: str) -> str:
    """Create, link, or refresh the pointer block in ``path``.

    Returns the action taken: ``"created"`` (new file), ``"updated"`` (block
    already present -> rewritten), or ``"linked"`` (existing file -> block
    appended).
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block, encoding="utf-8")
        return "created"

    original = path.read_text(encoding="utf-8")
    stripped, had_block = _strip_pointer_block(original)
    base = stripped.rstrip("\n")
    new_text = f"{base}\n\n{block}" if base else block
    path.write_text(new_text, encoding="utf-8")
    return "updated" if had_block else "linked"


def wire_agent_instructions(*, target: Path, console: Console) -> None:
    """Write ``EVALSHIFT.md`` and wire a pointer into each agent-context file.

    Idempotent: the standalone guide is refreshed verbatim, and each pointer
    block is rewritten in place rather than duplicated. When no agent-context
    file exists, ``AGENTS.md`` is created as the host.
    """
    instructions_path = target / AGENT_INSTRUCTIONS_FILENAME
    instructions_path.write_text(AGENT_INSTRUCTIONS, encoding="utf-8")
    console.print(f"[green]✓[/green] wrote {instructions_path}")

    existing = [name for name in AGENT_CONTEXT_FILES if (target / name).exists()]
    hosts = existing or [DEFAULT_AGENT_CONTEXT_FILE]

    for name in hosts:
        path = target / name
        block = _render_pointer_block(claude_import=(name == "CLAUDE.md"))
        action = _upsert_pointer(path, block)
        verb = {"created": "created", "updated": "refreshed pointer in", "linked": "linked into"}[
            action
        ]
        console.print(f"[green]✓[/green] {verb} {path}")


__all__ = [
    "AGENT_CONTEXT_FILES",
    "AGENT_INSTRUCTIONS",
    "AGENT_INSTRUCTIONS_FILENAME",
    "CI_DOCS_URL",
    "CLI_DOCS_URL",
    "DEFAULT_AGENT_CONTEXT_FILE",
    "POINTER_MARKER_BEGIN",
    "POINTER_MARKER_END",
    "SDK_DOCS_URL",
    "wire_agent_instructions",
]
