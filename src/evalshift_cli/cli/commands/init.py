"""Implementation of ``evalshift init``.

``init`` writes a single file — a minimal, capture-ready ``evalshift.yaml`` —
so a real project starts from a clean config instead of scaffolded example
data. The intended flow is:

.. code-block:: shell

    evalshift init                 # writes evalshift.yaml
    # instrument your agent with the evalshift-sdk, exercise it to record captures
    evalshift capture sync         # promote captures into suites + wire them in
    evalshift compare --suite-name <suite> --to <candidate>

``evalshift.yaml`` is never overwritten unless ``--force`` is passed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Final

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt

from evalshift_cli import __version__
from evalshift_cli.cli.commands._agents import (
    AGENT_INSTRUCTIONS_FILENAME,
    wire_agent_instructions,
)
from evalshift_cli.cli.commands._scaffold import (
    CI_WORKFLOW_PATH,
    INIT_PROFILE_POLICIES,
    PROVIDER_API_KEY_ENVS,
    ProfileOption,
    render_ci_workflow,
    write_scaffold_files,
)
from evalshift_cli.cli.commands._suites import render_suites_region
from evalshift_cli.cli.commands.doctor import CONFIG_FILENAME
from evalshift_cli.models.registry import PROVIDER_ENV_VARS, missing_api_keys, resolve_model
from evalshift_cli.utils.ci_pin import check_ci_pin

PROVIDERS: Final = ("gemini", "openai", "anthropic", "deepseek")

# Per-provider model ids written into the scaffold. The judge is a strong
# model from the same provider so a fresh project needs exactly one API key;
# the scaffold comments nudge users toward a cross-family judge once the
# target model is known.
_PROVIDER_MODELS: Final[dict[str, dict[str, str]]] = {
    "gemini": {
        "source_model": "gemini-3.1-flash-lite-preview",
        "target_hint": "gemini-3.1-pro-preview",
        "judge_model": "gemini-3.1-pro-preview",
        "embedding_model": "gemini/gemini-embedding-001",
    },
    "openai": {
        "source_model": "gpt-5.4-mini",
        "target_hint": "gpt-5.6-luna",
        "judge_model": "gpt-5.6-luna",
        "embedding_model": "openai/text-embedding-3-small",
    },
    "anthropic": {
        "source_model": "claude-sonnet-5",
        "target_hint": "claude-opus-4-8",
        "judge_model": "claude-opus-4-8",
        "embedding_model": "",  # no Anthropic embedding endpoint
    },
    "deepseek": {
        "source_model": "deepseek-flash",
        "target_hint": "deepseek-v4-pro",
        "judge_model": "deepseek-v4-pro",
        "embedding_model": "",  # no DeepSeek embedding endpoint
    },
}

_SEMANTIC_BLOCK: Final = """\
  # Embedding-based drift score between source and target outputs. Advisory
  # (blocking: false): it reports and ranks drift but never fails a run by
  # itself — cosine distance can't tell "reworded" from "wrong".
{borrowed_note}  semantic:
    embedding_model: {embedding_model}
    # Cosine similarity below which a target output is flagged as a
    # semantic regression. Defaults to 0.9; lower it to tolerate more drift.
    min_similarity: 0.9
    blocking: false"""

_BORROWED_EMBEDDING_NOTE: Final = """\
  # {provider_label} has no embeddings endpoint, so this uses {embedding_label}'s and
  # needs {key}. Without it, `evalshift compare` skips this evaluator and says so.
"""

# Tried in order when the provider has no embeddings endpoint of its own; the
# first key already exported decides, so a user with OpenAI or Gemini set up
# gets a working semantic evaluator with nothing to edit.
_BORROWED_EMBEDDINGS: Final[tuple[tuple[str, str], ...]] = (
    ("OPENAI_API_KEY", "openai/text-embedding-3-small"),
    ("GEMINI_API_KEY", "gemini/gemini-embedding-001"),
    ("GOOGLE_API_KEY", "gemini/gemini-embedding-001"),
)
_DEFAULT_BORROWED_EMBEDDING: Final = "openai/text-embedding-3-small"
_PROVIDER_LABELS: Final[dict[str, str]] = {"anthropic": "Anthropic", "deepseek": "DeepSeek"}
_EMBEDDING_LABELS: Final[dict[str, str]] = {"openai": "OpenAI", "google": "Gemini"}

# Body of the minimal config, up to (but excluding) the suites region and the
# migration_policy block. ``render_minimal_config`` appends those.
_MINIMAL_YAML_BODY: Final = """\
# EvalShift configuration. See https://github.com/evalshift/evalshift-cli for docs.
#
# `init` writes only this file, set up for the capture-first flow: instrument
# your agent with the evalshift-sdk, exercise it to record captures, then run
# `evalshift capture sync` to promote them into suites and wire them in below.
version: 1

# Hosted only (https://www.evalshift.dev): the project runs are pushed to, in
# `org/project` form — lowercase letters, digits and hyphens on both sides.
# Local runs never need it and nothing leaves this machine without an explicit
# `evalshift push` / `evalshift compare --push`; `push --project` overrides it for
# one invocation. Uncomment and set it to save typing that flag.
# project: your-org/your-project

prompts:
  # Passthrough replay: a promoted capture's example is
  # {{"input": "<full rendered prompt>"}}, so we echo it back verbatim.
  # Extracting a prompt from your source instead? Use detection: python_string
  # with `path:` + `variable:` (see docs/configuration.md).
  - id: replay
    detection: manual
    content: "{{input}}"
    variables: [input]

# Defaults applied to every run; CLI flags override these.
defaults:
  # Your CURRENT production model — the baseline you migrate FROM.
  source_model: {source_model}
  # The candidate you migrate TO. Set it here or pass --to <model> per run.
  # target_model: {target_hint}
  concurrency: 4
  max_cost_usd: 50.0
  # Completion length cap for every model call. Raise it (or set a
  # per-prompt prompts[].max_tokens) if outputs are being truncated;
  # truncated calls are excluded from the regression stats.
  max_tokens: 4096

evaluators:
{semantic_block}
  # Pairwise LLM judge. Advisory (blocking: false) by default: at the small
  # suite sizes fresh captures start with, judge noise would gate the verdict.
  # Flip to true once your suite is large enough that you trust its calls.
  # The judge sees the two outputs as anonymous "A" and "B" — keep the
  # criterion symmetric (never mention source/target) and keep an explicit
  # tie instruction, or the judge degenerates to picking a side at random.
  # When you set target_model, prefer a judge from a third model family —
  # judging its own family's outputs biases the result.
  llm_judge:
    - criterion_name: equivalence
      criterion_prompt: >
        Which output is more complete and correct? Prefer valid JSON over
        fenced or malformed JSON, no dropped or invented fields or entity
        ids, and conclusions grounded in the input. Answer "tie" when both
        are equivalent in substance and differ only in wording.
      judge_model: {judge_model}
      blocking: false
  # Nothing to uncomment for agent/tool migrations: `evalshift capture sync`
  # reads what each suite's captures actually contain and writes that suite's
  # own tool_selection / tool_arguments block into the managed suites: region
  # below — so a tool-free suite is never scored on an empty tool denominator.
  # See docs/agents.md.
"""


def scaffold_embedding_model(provider: str, env: Mapping[str, str]) -> str:
    """Pick the embedding model ``init`` writes for ``provider``.

    The provider's own when it has one; otherwise the first of OpenAI's or
    Gemini's whose key is already set, else OpenAI's — written active either
    way, so ``compare`` names the missing key instead of the YAML hiding it.

    Args:
        provider: One of :data:`PROVIDERS`.
        env: Environment mapping, typically ``os.environ``.
    """
    own = _PROVIDER_MODELS[provider]["embedding_model"]
    if own:
        return own
    for key, model in _BORROWED_EMBEDDINGS:
        if env.get(key):
            return model
    return _DEFAULT_BORROWED_EMBEDDING


def _borrowed_embedding_key(provider: str, embedding_model: str) -> str | None:
    """Primary env var of a borrowed embedding provider, or ``None`` when it is the provider's own."""
    if _PROVIDER_MODELS[provider]["embedding_model"]:
        return None
    return PROVIDER_ENV_VARS[resolve_model(embedding_model).provider][0]


def render_minimal_config(
    *, profile: str, provider: str = "gemini", env: Mapping[str, str] | None = None
) -> str:
    """Render the minimal ``evalshift.yaml`` for a migration profile + provider.

    Args:
        profile: One of the migration-profile names in ``INIT_PROFILE_POLICIES``;
            selects the ``migration_policy`` block.
        provider: One of :data:`PROVIDERS`; selects the model ids the
            scaffold writes (source, judge, embedding).
        env: Environment used to pick a borrowed embedding model; defaults to os.environ.
    """
    models = _PROVIDER_MODELS[provider]
    embedding_model = scaffold_embedding_model(provider, os.environ if env is None else env)
    borrowed_key = _borrowed_embedding_key(provider, embedding_model)
    borrowed_note = (
        _BORROWED_EMBEDDING_NOTE.format(
            provider_label=_PROVIDER_LABELS[provider],
            embedding_label=_EMBEDDING_LABELS[resolve_model(embedding_model).provider],
            key=borrowed_key,
        )
        if borrowed_key is not None
        else ""
    )
    semantic_block = _SEMANTIC_BLOCK.format(
        embedding_model=embedding_model, borrowed_note=borrowed_note
    )
    body = _MINIMAL_YAML_BODY.format(
        source_model=models["source_model"],
        target_hint=models["target_hint"],
        judge_model=models["judge_model"],
        semantic_block=semantic_block,
    )
    header = f"# migration_profile: {profile}\n"
    policy = INIT_PROFILE_POLICIES[profile]
    # The managed region goes last: it is the only part of the file a command
    # rewrites, and the only part that grows without bound (one entry per
    # suite, each with its own evaluator block). At the tail, a `capture sync`
    # diff stays confined to the tail and everything a person hand-edits above
    # it keeps stable line numbers.
    suites = render_suites_region("suites: {}")
    return header + body.rstrip() + "\n\n" + policy.rstrip() + "\n\n" + suites + "\n"


def init(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite an existing evalshift.yaml. Off by default to protect work.",
        ),
    ] = False,
    directory: Annotated[
        Path,
        typer.Option(
            "--directory",
            "-d",
            help="Target directory (default: current working directory).",
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("."),
    ci: Annotated[
        bool,
        typer.Option(
            "--ci",
            help=(
                f"Also scaffold {CI_WORKFLOW_PATH} — a GitHub Actions "
                "workflow that discovers every committed suite, evaluates "
                "each on every PR, pushes the runs to hosted EvalShift, and "
                "gates merges on the migration policy via a single "
                "'evalshift gate' check. Setup steps are documented in the "
                "file itself."
            ),
        ),
    ] = False,
    wire_agents: Annotated[
        bool,
        typer.Option(
            "--wire-agents/--no-wire-agents",
            help=(
                f"Write {AGENT_INSTRUCTIONS_FILENAME} and point existing agent "
                "files (AGENTS.md, CLAUDE.md, GEMINI.md, .cursorrules, "
                ".github/copilot-instructions.md) at it so coding agents can "
                "drive the CLI. Creates AGENTS.md if none exist. Idempotent."
            ),
        ),
    ] = True,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help=(
                "LLM provider the scaffold's model ids target: "
                f"{', '.join(PROVIDERS)}. Prompted interactively when omitted "
                "on a TTY; defaults to gemini otherwise."
            ),
        ),
    ] = None,
    profile: ProfileOption = "model-upgrade",
) -> None:
    """Scaffold a minimal, capture-ready ``evalshift.yaml``."""
    console = Console()
    if provider is not None and provider not in PROVIDERS:
        console.print(
            f"[red]✗[/red] unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}.",
        )
        raise typer.Exit(code=2)
    if provider is None:
        if sys.stdin.isatty():
            provider = Prompt.ask(
                "Which LLM provider does your project use?",
                choices=list(PROVIDERS),
                default="gemini",
                console=console,
            )
        else:
            provider = "gemini"
            console.print(
                "[dim]no --provider given; defaulting to gemini "
                f"(choices: {', '.join(PROVIDERS)}).[/dim]",
            )

    target = directory.resolve()
    target.mkdir(parents=True, exist_ok=True)
    embedding_model = scaffold_embedding_model(provider, os.environ)
    embedding_key = _borrowed_embedding_key(provider, embedding_model)

    files: dict[str, str] = {
        CONFIG_FILENAME: render_minimal_config(profile=profile, provider=provider),
    }
    if ci:
        files[CI_WORKFLOW_PATH] = render_ci_workflow(
            provider=provider, version=__version__, embedding_api_key=embedding_key
        )

    write_scaffold_files(target=target, files=files, force=force, console=console)

    if wire_agents:
        wire_agent_instructions(target=target, console=console)

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print(
        "  1. Instrument your agent with the [cyan]evalshift-sdk[/cyan] and exercise it"
        " to record captures under [cyan].evalshift/captures/[/cyan].",
    )
    console.print(
        "  2. [cyan]evalshift capture sync[/cyan]  - promote captures into suites"
        " and wire them into evalshift.yaml.",
    )
    console.print(
        "  3. [cyan]evalshift compare --suite-name <suite> --to <candidate>[/cyan]"
        "  - run the migration end to end.",
    )
    if embedding_key is not None and missing_api_keys(embedding_model, os.environ):
        console.print(
            f"     [bold]semantic[/bold] uses [cyan]{embedding_model}[/cyan] — export "
            f"[bold]{embedding_key}[/bold] to enable it; compare skips it until then.",
        )
    if wire_agents:
        console.print()
        console.print(
            f"  Coding agents: [cyan]{AGENT_INSTRUCTIONS_FILENAME}[/cyan] holds the CLI"
            " guide; your agent files now point at it. Off? [cyan]--no-wire-agents[/cyan].",
        )
    if ci:
        console.print()
        console.print(
            f"  CI: commit [cyan]{CI_WORKFLOW_PATH}[/cyan], add "
            f"[bold]{PROVIDER_API_KEY_ENVS[provider]}[/bold]"
            + (f", [bold]{embedding_key}[/bold] (for semantic)" if embedding_key else "")
            + " and "
            "[bold]EVALSHIFT_TOKEN[/bold] as repo secrets, and require the "
            "[bold]evalshift gate[/bold] check in branch protection — the "
            "full checklist is documented at the top of the workflow file.",
        )
    else:
        # Re-initialising next to a hand-written or older scaffolded workflow:
        # the config just written may carry keys that CI's pinned CLI rejects.
        finding = check_ci_pin(target, __version__)
        if finding is not None:
            console.print()
            console.print(f"[yellow]⚠[/yellow] {escape(finding.message)}")


__all__ = ["init", "render_minimal_config"]
