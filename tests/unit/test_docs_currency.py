"""The docs must advertise the current command name, not a hidden alias.

`evalshift all` became `evalshift compare` in 1.0.0. The old name stays
registered forever -- scaffolded EVALSHIFT.md files in user repos reference it,
and removing it would itself be breaking -- but it is `hidden=True` and prints a
rename notice. Nine doc sites still told readers to type it, so the docs taught
a name that `evalshift --help` does not list.

Prose that *describes* the alias ("formerly `all`", "`all` -> `compare` in
1.0.0") is correct and deliberately not matched here: the assertion is on the
exact string `all --push`, which only ever appeared as advertised usage.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import evalshift_cli
from evalshift_cli.cli.commands._agents import AGENT_INSTRUCTIONS
from evalshift_cli.cli.commands.capture import capture_app
from evalshift_cli.cli.commands.init import PROVIDERS
from evalshift_cli.models.registry import PROVIDER_ENV_VARS

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every file that documents the CLI in prose, relative to the repo root.
PROSE_FILES: tuple[str, ...] = (
    "README.md",
    "DOCS.md",
    "llms-full.txt",
    "docs/faq.md",
    "docs/hosted.md",
    "docs/configuration.md",
    "docs/index.md",
    "AGENTS.md",
)


@pytest.mark.parametrize("name", PROSE_FILES)
def test_prose_advertises_compare_not_the_hidden_alias(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "all --push" not in text, (
        f"{name} advertises the hidden `all` alias; write `compare --push`"
    )


#: Files that describe *which providers work*, as opposed to naming three as
#: examples. Each must name LiteLLM, because LiteLLM is the actual boundary:
#: `models/registry.py` says so in its module docstring, and `docs/faq.md`
#: already answers "which models?" with "Anything LiteLLM supports."
PROVIDER_SCOPE_FILES: tuple[str, ...] = (
    "README.md",
    "docs/index.md",
    "docs/faq.md",
    "docs/hosted.md",
    "docs/getting-started.md",
)


@pytest.mark.parametrize("name", PROVIDER_SCOPE_FILES)
def test_provider_scope_is_not_capped_at_three(name: str) -> None:
    """The curated registry has three entries; the CLI calls far more than three.

    `Provider` is a Literal of three names plus "other" because those three have
    pricing tables and env-var mappings worth curating. Every call still goes
    through `litellm.acompletion`, and `resolve_model` never raises -- an
    unregistered id is dispatched with a prefix-inferred provider. Prose that
    lists the three without naming LiteLLM reads as a compatibility list and
    undersells the tool.

    Two assertions, because either alone is insufficient:

    - The presence check (`"LiteLLM" in text`) states the boundary, but
      `docs/faq.md` already contained the string "LiteLLM" at an unrelated
      answer (line 50, "what models does EvalShift support?") before this
      file's other answer (the "send my prompts" one, lines 5-8) was fixed.
      That means presence alone would stay green even if the lines 5-8 fix
      were fully reverted -- the test would guard nothing for this file.
    - The absence check catches exactly that revert: it fails if the
      three-brand phrasing reappears anywhere in the file,
      whitespace-normalised so it survives the line break in
      `docs/index.md`. The regex has two alternatives because the phrasing
      shows up two ways in the wild: the plain list ("Anthropic, OpenAI,
      Google") and the Oxford-comma form ("Anthropic, OpenAI, and Google",
      `docs/getting-started.md`'s old wording). Both require a comma right
      after "OpenAI", which is what keeps this from tripping on README's
      "Anthropic, OpenAI and Google ids additionally get a curated..."
      sentence -- it has no comma before "and Google", only the (correct)
      word "and".
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "LiteLLM" in text, f"{name} scopes providers without naming LiteLLM"

    normalized = " ".join(text.split())
    assert not re.search(r"Anthropic, OpenAI, and Google|Anthropic, OpenAI, Google", normalized), (
        f"{name} still prints the three-brand list as the compatibility boundary"
    )


@pytest.mark.parametrize("name", PROSE_FILES)
def test_prose_quotes_the_rendered_placeholder(name: str) -> None:
    """Docs must quote the config `init` writes, not the format template.

    `_MINIMAL_YAML_BODY` in `cli/commands/init.py` is passed through
    `str.format`, so its literal `{{input}}` is a brace escape that renders as
    `{input}` on disk -- which is what `test_init.py` asserts the loaded config
    contains. Three doc sites copied the escaped source form verbatim, which
    reads as instructions to write a placeholder `templating.py` will never
    expand.
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert "{{input}}" not in text, (
        f"{name} quotes the escaped `{{{{input}}}}`; `init` writes `{{input}}`"
    )


#: Files that tell a user which env var authenticates which provider. A
#: provider the registry can authenticate but these files never name is a
#: provider whose users are told nothing — the state DeepSeek support shipped
#: into on 2026-09-30.
KEY_TABLE_FILES: tuple[str, ...] = ("DOCS.md", "llms-full.txt", "docs/getting-started.md")

#: Files that spell out `init --provider`'s choices.
INIT_PROVIDER_FILES: tuple[str, ...] = ("DOCS.md", "llms-full.txt")


@pytest.mark.parametrize("name", KEY_TABLE_FILES)
@pytest.mark.parametrize("env_var", sorted(aliases[0] for aliases in PROVIDER_ENV_VARS.values()))
def test_key_docs_name_every_registry_provider(name: str, env_var: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert env_var in text, f"{name} never tells users about {env_var}"


@pytest.mark.parametrize("name", INIT_PROVIDER_FILES)
def test_docs_list_every_init_provider(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")

    assert f"--provider {'|'.join(PROVIDERS)}" in text, (
        f"{name} lists `init --provider` choices that differ from init.PROVIDERS"
    )


#: The two reference files that print the CLI version in their header.
#: `DOCS.md` writes `**version:** X`, `llms-full.txt` writes `version: X`.
VERSION_HEADER_FILES: tuple[str, ...] = ("DOCS.md", "llms-full.txt")


@pytest.mark.parametrize("name", VERSION_HEADER_FILES)
def test_reference_header_version_matches_the_package(name: str) -> None:
    """The documented version is the released one, not the one before it.

    `evalshift_cli.__version__` is the installed metadata of `pyproject.toml`'s
    `version`. The two headers drifted apart once already: `llms-full.txt` said
    1.1.0 while `DOCS.md` still said 1.0.1.
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    match = re.search(r"version:(?:\*\*)? (\S+)", text)

    assert match is not None, f"{name} has no `version:` line in its header"
    assert match.group(1) == evalshift_cli.__version__, (
        f"{name} says version {match.group(1)}; the package is {evalshift_cli.__version__}"
    )


def _yaml_bearing_docs() -> list[str]:
    """Every checked-in file a reader might copy an `evalshift.yaml` block from."""
    found = [
        *PROSE_FILES,
        *(str(p.relative_to(REPO_ROOT)) for p in sorted((REPO_ROOT / "docs").glob("*.md"))),
        *(str(p.relative_to(REPO_ROOT)) for p in sorted((REPO_ROOT / "examples").rglob("*.md"))),
        *(
            str(p.relative_to(REPO_ROOT))
            for p in sorted((REPO_ROOT / "examples").rglob("evalshift.yaml"))
        ),
    ]
    return sorted(set(found))


#: Top-level `evalshift.yaml` keys that were removed. A config that still sets
#: one fails to load, so a doc that shows one hands the reader a broken config.
REMOVED_TOP_LEVEL_KEYS: tuple[str, ...] = ("thresholds", "slices")


@pytest.mark.parametrize("name", _yaml_bearing_docs())
def test_no_doc_shows_a_removed_top_level_key(name: str) -> None:
    """A top-level YAML key starts in column 0; a nested one never does.

    That keeps `migration_policy.slices` -- indented under its parent, and very
    much alive -- out of the match, while catching a copied-in `slices:` or
    `thresholds:` block wherever it appears.
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    pattern = re.compile(rf"^(?:{'|'.join(REMOVED_TOP_LEVEL_KEYS)}):", re.MULTILINE)

    hits = [m.group(0) for m in pattern.finditer(text)]
    assert not hits, f"{name} shows removed top-level key(s) {hits}; that config would not load"


#: Exact substrings that described the v5 hosted plans. v6 (2026-10) has no hosted Free or
#: Team plan; the CLI itself is still free. Retiring another plan claim? Append it here.
RETIRED_PLAN_TERMS: tuple[str, ...] = ("Free 1, Pro 5, Team 10", "on the Free plan")

_PLAN_COPY_FILES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *PROSE_FILES,
            "docs/hosted.md",
            "docs/github-action.md",
            "src/evalshift_cli/cli/commands/_scaffold.py",
        )
    )
)


def _whitespace_normalized(name: str) -> str:
    """The file's text with every whitespace run collapsed to one space, so a retired
    phrase hard-wrapped across lines (as Markdown and code comments are) still matches."""
    return " ".join((REPO_ROOT / name).read_text(encoding="utf-8").split())


@pytest.mark.parametrize("term", RETIRED_PLAN_TERMS)
def test_copy_does_not_describe_the_retired_hosted_plans(term: str) -> None:
    offenders = [name for name in _PLAN_COPY_FILES if term in _whitespace_normalized(name)]
    assert offenders == [], f"{term!r} still appears in {offenders}"


def _registered_capture_subcommands() -> list[str]:
    """Every subcommand `evalshift capture --help` lists, straight from the Typer group."""
    return sorted(str(cmd.name) for cmd in capture_app.registered_commands)


@pytest.mark.parametrize("name", _registered_capture_subcommands())
def test_scaffolded_agent_guide_names_every_capture_subcommand(name: str) -> None:
    """`init` writes EVALSHIFT.md into user repos, and its `evalshift capture ...`
    row is the one place a coding agent learns which subcommands exist.

    `capture fetch` shipped in 1.3.0 without being added to that row, so every
    scaffolded guide told agents the group had five subcommands. The row is
    matched by its leading cell so a bare word like `list` elsewhere in the
    guide cannot satisfy the check.
    """
    row = next(
        line
        for line in AGENT_INSTRUCTIONS.splitlines()
        if line.startswith("| `evalshift capture ...`")
    )

    assert f"`{name}`" in row, f"EVALSHIFT.md's capture row does not name `{name}`"


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
