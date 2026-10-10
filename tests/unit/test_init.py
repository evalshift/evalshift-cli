"""Tests for ``evalshift init`` (:mod:`evalshift_cli.cli.commands.init`).

``init`` writes a single, minimal, capture-ready ``evalshift.yaml`` — no demo
data. The invariant we care about: the file parses cleanly via
:func:`load_config`, is wired for the capture-first flow (a passthrough prompt
and output evaluators), and carries the marker block that ``capture sync``
rewrites.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import evalshift_cli
from evalshift_cli.analysis.policy import BUDGET_LABELS
from evalshift_cli.cli.commands._agents import (
    AGENT_INSTRUCTIONS_FILENAME,
    DEFAULT_AGENT_CONTEXT_FILE,
    POINTER_MARKER_BEGIN,
)
from evalshift_cli.cli.commands._scaffold import (
    CI_WORKFLOW_PATH,
    INIT_PROFILE_POLICIES,
    PROVIDER_API_KEY_ENVS,
)
from evalshift_cli.cli.commands._suites import (
    SUITE_FILENAME,
    SUITES_MARKER_BEGIN,
    SUITES_MARKER_END,
)
from evalshift_cli.cli.commands.doctor import CONFIG_FILENAME
from evalshift_cli.cli.commands.init import _PROVIDER_MODELS, PROVIDERS, render_minimal_config
from evalshift_cli.cli.main import app
from evalshift_cli.config.loader import load_config
from evalshift_cli.models.registry import PROVIDER_ENV_VARS, resolve_model

runner = CliRunner()

# Demo-scaffold file names `init` must never write (see
# TestInitHappy.test_writes_only_the_config below). `_scaffold.py` used to
# define these and a deleted `demo` command used to write them; T7 removed
# both, so these are now purely local vocabulary for the negative
# assertion rather than production constants.
FIXTURES_FILENAME = "fixtures.jsonl"
PROMPTS_FILENAME = "prompts.py"
TOOLS_FILENAME = "tools.yaml"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every doc that reprints the `migration_policy` block `init` writes. Each shows
# the same budgets in a different shape (block YAML in the two references, flow
# YAML in the LLM digest), so the check below is a substring match rather than a
# YAML parse.
POLICY_DOC_FILENAMES = ("docs/configuration.md", "llms-full.txt", "DOCS.md")


@pytest.fixture
def in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run inside ``tmp_path`` with no provider keys set."""
    monkeypatch.chdir(tmp_path)
    for keys in PROVIDER_ENV_VARS.values():
        for key in keys:
            monkeypatch.delenv(key, raising=False)
    return tmp_path


class TestInitHappy:
    def test_writes_only_the_config(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        assert (in_tmp / CONFIG_FILENAME).is_file()
        # init must NOT scaffold any demo data.
        for name in (PROMPTS_FILENAME, TOOLS_FILENAME, SUITE_FILENAME, FIXTURES_FILENAME):
            assert not (in_tmp / name).exists(), f"init unexpectedly wrote {name}"

    def test_written_config_parses_via_load_config(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert len(cfg.prompts) == 1
        prompt = cfg.prompts[0]
        assert prompt.id == "replay"
        assert prompt.detection == "manual"
        assert prompt.content == "{input}"
        assert prompt.variables == ["input"]
        # Empty suites block — capture sync fills it.
        assert cfg.suites == {}

    def test_config_wires_capture_first_evaluators(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == "gemini/gemini-embedding-001"
        assert len(cfg.evaluators.llm_judge) == 1
        assert cfg.evaluators.llm_judge[0].judge_model == "gemini-3.1-pro-preview"
        # No tool evaluators in the minimal (non-agent) scaffold.
        assert not cfg.evaluators.tool_selection
        assert not cfg.evaluators.tool_arguments

    def test_config_defers_tool_evaluators_to_capture_sync(self, in_tmp: Path) -> None:
        """No commented-out tool block to uncomment: sync wires them per suite."""
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# tool_selection:" not in body
        assert "# tool_arguments:" not in body
        assert "capture sync" in body

    def test_config_documents_the_project_field(self, in_tmp: Path) -> None:
        """The hosted slug is opt-in, but the scaffold has to name it.

        Nothing in a local run needs ``project:``, so init must not invent one --
        but a user who never sees the key does not learn it exists until a push
        fails for the lack of it.
        """
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# project: your-org/your-project" in body
        assert "push" in body
        # Commented, not set: init must not guess a slug on the user's behalf.
        assert load_config(in_tmp / CONFIG_FILENAME).project is None

    def test_commented_project_slug_parses_once_uncommented(self, in_tmp: Path) -> None:
        """The placeholder must satisfy the ``org/project`` pattern.

        A scaffolded example that fails validation the moment it is uncommented
        teaches the wrong shape and fails at the user's first push.
        """
        runner.invoke(app, ["init"])
        path = in_tmp / CONFIG_FILENAME
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "# project: your-org/your-project",
                "project: your-org/your-project",
            ),
            encoding="utf-8",
        )
        assert load_config(path).project == "your-org/your-project"

    def test_config_carries_suites_markers(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert SUITES_MARKER_BEGIN in body
        assert SUITES_MARKER_END in body
        assert "suites: {}" in body

    def test_suites_region_is_last_in_the_file(self, in_tmp: Path) -> None:
        """The managed region goes at the tail, after ``migration_policy``.

        It is the only part of the file a command rewrites, and the only part
        that grows without bound -- one entry per suite, each carrying its own
        evaluator block. Last means a sync's diff stays confined to the tail and
        the hand-edited config above it keeps stable line numbers.
        """
        runner.invoke(app, ["init"])
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert body.index("migration_policy:") < body.index(SUITES_MARKER_BEGIN)
        assert body.rstrip().endswith(SUITES_MARKER_END)

    def test_profile_adds_migration_policy(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--profile", "cost-reduction"])
        assert result.exit_code == 0, result.stdout
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# migration_profile: cost-reduction" in body
        assert "migration_policy:" in body
        assert "max_cost_increase: 0.05" in body

    def test_default_profile_budgets_leave_room_for_a_first_migration(self, in_tmp: Path) -> None:
        """A fresh suite should report its regressions, not fail on two of them."""
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# migration_profile: model-upgrade" in body
        for line in (
            "max_overall_regression_rate: 0.30",
            "max_critical_regressions: 1",
            "min_equivalence_rate: 0.75",
            "max_tool_argument_drift: 0.20",
            "max_tool_divergence: 0.20",
            "max_invariant_violations: 0",
            "max_cost_increase: 0.30",
            "max_latency_increase: 0.30",
        ):
            assert f"  {line}" in body

    @pytest.mark.parametrize("profile", sorted(INIT_PROFILE_POLICIES))
    def test_every_profile_scaffolds_every_budget(self, profile: str) -> None:
        """A budget a profile leaves out is one its reader never learns exists.

        ``BUDGET_LABELS`` names every budget the policy emits a row for, the
        CLI-only ones (``max_tool_divergence``, ``max_invariant_violations``)
        included. ``tool_argument_drift_floor`` and ``fail_on_dropped_params``
        are not budgets, so they are absent from both by design.
        """
        block = INIT_PROFILE_POLICIES[profile]
        missing = [name for name in BUDGET_LABELS if f"\n  {name}: " not in block]
        assert not missing, f"profile {profile!r} omits budget(s) {missing}"

    def test_prints_capture_first_next_steps(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert "evalshift capture sync" in result.stdout


class TestInitPolicyDocsMatchTheScaffold:
    """Every doc that publishes the init budgets must publish the *current* ones.

    The test above pins ``init``'s output to :data:`INIT_PROFILE_POLICIES`; this
    one pins the docs to the same constant. Without it the reference pages drift
    (they carried a 0.50/2.0 cost/latency budget the CLI stopped writing), and a
    reader copies numbers no scaffold ever produced.
    """

    @staticmethod
    def _policy_pairs() -> list[str]:
        """The ``key: value`` pairs of the default profile, in scaffold order."""
        return [
            line.strip()
            for line in INIT_PROFILE_POLICIES["model-upgrade"].splitlines()
            if line.startswith(" ") and line.strip()
        ]

    def test_every_budget_is_parsed(self) -> None:
        """Guard the parser itself: an empty list would pass every doc check."""
        pairs = self._policy_pairs()
        assert len(pairs) == 8, pairs
        assert all(pair.count(": ") == 1 for pair in pairs), pairs

    @pytest.mark.parametrize("doc_name", POLICY_DOC_FILENAMES)
    def test_doc_publishes_the_scaffolded_budgets(self, doc_name: str) -> None:
        text = (REPO_ROOT / doc_name).read_text(encoding="utf-8")
        # A trailing-digit guard so `max_critical_regressions: 1` is not
        # satisfied by a published `: 10`, nor `0.30` by `0.300`.
        missing = [
            pair
            for pair in self._policy_pairs()
            if not re.search(f"{re.escape(pair)}(?![0-9])", text)
        ]
        assert not missing, (
            f"{doc_name} does not publish INIT_PROFILE_POLICIES['model-upgrade'] "
            f"verbatim; missing: {missing}"
        )


class TestInitStrongDefaults:
    """The generated config must be honest-by-default (see 2026-07-21 spec)."""

    def test_judge_criterion_is_symmetric(self, in_tmp: Path) -> None:
        # The judge sees anonymized A/B outputs — a criterion phrased in
        # TARGET/SOURCE terms is unanswerable and degenerates to position bias.
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        criterion = cfg.evaluators.llm_judge[0].criterion_prompt
        assert "TARGET" not in criterion
        assert "SOURCE" not in criterion
        assert "tie" in criterion.lower()

    def test_semantic_and_judge_are_advisory(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.blocking is False
        assert cfg.evaluators.llm_judge[0].blocking is False


class TestInitProvider:
    def test_default_is_gemini(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == "gemini/gemini-embedding-001"
        assert "gemini" in cfg.defaults.source_model

    def test_openai_provider_writes_openai_ids(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "openai"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "gpt-5.4-mini"
        assert cfg.evaluators.llm_judge[0].judge_model == "gpt-5.6-luna"
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == "openai/text-embedding-3-small"
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "gemini-3.1" not in body

    def test_unknown_provider_rejected(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "grok"])
        assert result.exit_code != 0

    def test_every_provider_config_round_trips(self, in_tmp: Path) -> None:
        for provider in PROVIDERS:
            for f in in_tmp.iterdir():
                if f.is_file():
                    f.unlink()
            result = runner.invoke(app, ["init", "--provider", provider, "--force"])
            assert result.exit_code == 0, f"{provider}: {result.stdout}"
            cfg = load_config(in_tmp / CONFIG_FILENAME)
            assert cfg.prompts[0].id == "replay"

    @pytest.mark.parametrize(
        ("env_key", "expected"),
        [
            (None, "openai/text-embedding-3-small"),
            ("OPENAI_API_KEY", "openai/text-embedding-3-small"),
            ("GEMINI_API_KEY", "gemini/gemini-embedding-001"),
            ("GOOGLE_API_KEY", "gemini/gemini-embedding-001"),
        ],
    )
    @pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
    def test_providers_without_embeddings_borrow_one_and_keep_semantic_on(
        self,
        in_tmp: Path,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        env_key: str | None,
        expected: str,
    ) -> None:
        if env_key is not None:
            monkeypatch.setenv(env_key, "k")
        result = runner.invoke(app, ["init", "--provider", provider])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.evaluators.semantic is not None
        assert cfg.evaluators.semantic.embedding_model == expected
        assert cfg.evaluators.semantic.blocking is False
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "# semantic:" not in body
        label = {"anthropic": "Anthropic", "deepseek": "DeepSeek"}[provider]
        assert f"{label} has no embeddings endpoint" in body

    def test_anthropic_keeps_its_own_ids(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "anthropic"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "claude-sonnet-5"
        assert cfg.evaluators.llm_judge[0].judge_model == "claude-opus-4-8"

    def test_deepseek_keeps_its_own_ids(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "deepseek"])
        assert result.exit_code == 0, result.stdout
        cfg = load_config(in_tmp / CONFIG_FILENAME)
        assert cfg.defaults.source_model == "deepseek-flash"
        assert cfg.evaluators.llm_judge[0].judge_model == "deepseek-v4-pro"
        body = (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")
        assert "Anthropic has no" not in body

    def test_borrowed_embedding_without_its_key_is_named_in_next_steps(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "anthropic"])
        flat = " ".join(result.stdout.split())
        assert (
            "semantic uses openai/text-embedding-3-small — export OPENAI_API_KEY to "
            "enable it; compare skips it until then."
        ) in flat

    def test_borrowed_embedding_with_its_key_needs_no_next_step(
        self, in_tmp: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        result = runner.invoke(app, ["init", "--provider", "anthropic"])
        assert "skips it until then" not in result.stdout

    def test_own_embedding_needs_no_next_step(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--provider", "gemini"])
        assert "skips it until then" not in result.stdout

    def test_render_is_pure_given_env(self) -> None:
        text = render_minimal_config(
            profile="model-upgrade", provider="anthropic", env={"GEMINI_API_KEY": "k"}
        )
        assert yaml.safe_load(text)["evaluators"]["semantic"]["embedding_model"] == (
            "gemini/gemini-embedding-001"
        )

    def test_every_init_provider_key_is_the_registry_key(self) -> None:
        # init and the run pre-check must agree on which env var authenticates
        # a scaffold's models, or `init` tells the user to export the wrong one.
        for provider in PROVIDERS:
            source = _PROVIDER_MODELS[provider]["source_model"]
            registry_provider = resolve_model(source).provider
            assert PROVIDER_API_KEY_ENVS[provider] == PROVIDER_ENV_VARS[registry_provider][0]


class TestInitAgentWiring:
    """``init`` wires coding-agent instructions by default."""

    def test_writes_guide_and_creates_agents_md(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        assert (in_tmp / AGENT_INSTRUCTIONS_FILENAME).is_file()
        host = in_tmp / DEFAULT_AGENT_CONTEXT_FILE
        assert host.is_file()
        assert POINTER_MARKER_BEGIN in host.read_text(encoding="utf-8")

    def test_wires_existing_claude_md(self, in_tmp: Path) -> None:
        (in_tmp / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
        runner.invoke(app, ["init"])
        body = (in_tmp / "CLAUDE.md").read_text(encoding="utf-8")
        assert "# rules" in body
        assert f"@./{AGENT_INSTRUCTIONS_FILENAME}" in body
        # No fallback AGENTS.md when a context file already exists.
        assert not (in_tmp / "AGENTS.md").exists()

    def test_no_wire_agents_skips_everything(self, in_tmp: Path) -> None:
        result = runner.invoke(app, ["init", "--no-wire-agents"])
        assert result.exit_code == 0, result.stdout
        assert not (in_tmp / AGENT_INSTRUCTIONS_FILENAME).exists()
        assert not (in_tmp / DEFAULT_AGENT_CONTEXT_FILE).exists()


class TestInitConflicts:
    def test_refuses_to_overwrite_by_default(self, in_tmp: Path) -> None:
        (in_tmp / CONFIG_FILENAME).write_text("# user's existing config\n", encoding="utf-8")
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "Refusing to overwrite" in result.stdout
        assert (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8") == (
            "# user's existing config\n"
        )

    def test_force_overwrites(self, in_tmp: Path) -> None:
        (in_tmp / CONFIG_FILENAME).write_text("# old\n", encoding="utf-8")
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0, result.stdout
        assert "version: 1" in (in_tmp / CONFIG_FILENAME).read_text(encoding="utf-8")


class TestInitCI:
    """``--ci`` drops a production-shaped GitHub Actions workflow.

    The invariants: the file is valid YAML; suites are discovered dynamically
    (so ``capture sync`` adding one needs no workflow edit) and an empty
    project stays green; a single join job exists for branch protection
    (per-suite matrix job names are dynamic and the ``evalshift/regression``
    commit status is last-write-wins across suites); the CLI version in CI is
    pinned to the CLI that scaffolded the config (``extra="forbid"`` config
    from a newer scaffold fails loudly on an older CLI); and the provider key
    matches ``--provider``.
    """

    def _workflow(self, in_tmp: Path, *args: str) -> tuple[str, dict[str, object]]:
        result = runner.invoke(app, ["init", "--ci", *args])
        assert result.exit_code == 0, result.stdout
        body = (in_tmp / CI_WORKFLOW_PATH).read_text(encoding="utf-8")
        return body, yaml.safe_load(body)

    def test_no_ci_flag_skips_workflow(self, in_tmp: Path) -> None:
        runner.invoke(app, ["init"])
        assert not (in_tmp / CI_WORKFLOW_PATH).exists()

    def test_ci_flag_writes_valid_workflow(self, in_tmp: Path) -> None:
        body, wf = self._workflow(in_tmp)
        assert "evalshift/evalshift-action@v0" in body
        assert "EVALSHIFT_TOKEN" in body
        assert "EVALSHIFT_NONINTERACTIVE" in body
        jobs = wf["jobs"]
        assert isinstance(jobs, dict)
        assert set(jobs) == {"discover", "evalshift", "gate"}

    def test_gate_job_joins_the_dynamic_matrix(self, in_tmp: Path) -> None:
        _, wf = self._workflow(in_tmp)
        jobs = wf["jobs"]
        assert isinstance(jobs, dict)
        gate = jobs["gate"]
        assert gate["needs"] == ["discover", "evalshift"]
        # Must run even when the matrix is skipped/failed, else branch
        # protection sees "expected" forever on fork PRs and empty projects.
        assert "always()" in gate["if"]

    def test_empty_suite_dir_is_skipped_not_failed(self, in_tmp: Path) -> None:
        # A fresh `init --ci` project has no suites until the first
        # `capture sync`; an unguarded glob would matrix over the literal
        # pattern and fail every CI run until then.
        body, wf = self._workflow(in_tmp)
        assert "shopt -s nullglob" in body
        jobs = wf["jobs"]
        assert isinstance(jobs, dict)
        assert "!= '[]'" in jobs["evalshift"]["if"]

    def test_gates_on_hosted_policy_verdict(self, in_tmp: Path) -> None:
        # `policy` gates on the verdict computed against the migration_policy
        # block init itself writes into evalshift.yaml — the two scaffolds gate
        # on one contract.
        body, _ = self._workflow(in_tmp)
        assert "fail-on: policy" in body
        assert "fail-on: regression" not in body

    def test_token_scope_names_policy_read(self, in_tmp: Path) -> None:
        # The scaffolded workflow defaults to `fail-on: policy`, which reads
        # the hosted policy-check endpoint (`policy:read`). A key scoped to
        # only `run:create` + `run:read` gets a 403 on that check and the
        # action silently falls back to `fail-on: regression` -- so the
        # scaffolded key guidance must name all three scopes together.
        body, _ = self._workflow(in_tmp)
        assert "run:create + run:read + policy:read" in body

    def test_pins_the_scaffolding_cli_version(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp)
        assert f'evalshift-version: "{evalshift_cli.__version__}"' in body

    def test_selects_each_suite_by_name_not_by_path(self, in_tmp: Path) -> None:
        # `suite-name` keys into `suites:`, which is where a captured suite's own
        # tool evaluators live; `suite` is a bare path and loses them. A
        # tool-calling suite scored with the top-level semantic/judge evaluators
        # produces no rows at all, so the run dies at analyze with an empty
        # scores.jsonl and nothing pointing at the selection as the cause.
        _, wf = self._workflow(in_tmp)
        jobs = wf["jobs"]
        assert isinstance(jobs, dict)
        steps = jobs["evalshift"]["steps"]
        (action_step,) = [
            step for step in steps if str(step.get("uses", "")).startswith("evalshift/")
        ]
        assert action_step["with"]["suite-name"] == "${{ matrix.suite }}"
        assert "suite" not in action_step["with"]

    def test_discovers_suite_names_that_the_config_can_be_keyed_by(self, in_tmp: Path) -> None:
        # The matrix carries `suites:` keys, not paths: `capture sync` promotes a
        # suite to `.evalshift/suites/<name>/golden.jsonl` and wires it under that
        # same `<name>`, so the directory name is the key.
        body, _ = self._workflow(in_tmp)
        assert 'names+=("$(basename "$(dirname "$f")")")' in body

    def test_provider_key_matches_provider(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp, "--provider", "anthropic")
        assert "ANTHROPIC_API_KEY" in body
        assert "GEMINI_API_KEY" not in body

    def test_deepseek_workflow_uses_the_deepseek_key(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp, "--provider", "deepseek")
        assert "DEEPSEEK_API_KEY" in body
        assert "GEMINI_API_KEY" not in body

    def test_borrowed_embedding_key_is_wired_as_a_secret(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp, "--provider", "anthropic")
        assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in body
        assert "__EMBEDDING_API_KEY__" not in body

    def test_own_embedding_adds_no_second_secret(self, in_tmp: Path) -> None:
        body, _ = self._workflow(in_tmp, "--provider", "gemini")
        assert "OPENAI_API_KEY" not in body
        assert "__EMBEDDING_API_KEY__" not in body

    def test_main_baseline_runs_are_never_cancelled(self, in_tmp: Path) -> None:
        # Push runs on main produce the base-branch baselines PRs diff
        # against; cancel-in-progress must be scoped to pull requests.
        _, wf = self._workflow(in_tmp)
        concurrency = wf["concurrency"]
        assert isinstance(concurrency, dict)
        assert "pull_request" in str(concurrency["cancel-in-progress"])

    def test_documents_the_setup_it_needs(self, in_tmp: Path) -> None:
        # The file is the documentation: committing suites past the
        # `.evalshift/` ignore, the required check, and the secrets must all
        # be explained in place, not in a repo the user has to go find.
        body, _ = self._workflow(in_tmp)
        assert "!.evalshift/suites/" in body
        assert "evalshift gate" in body
        assert "capture sync" in body


class TestInitDirectoryFlag:
    def test_writes_into_specified_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "fresh"
        result = runner.invoke(app, ["init", "--directory", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / CONFIG_FILENAME).is_file()
        # The cwd itself should be untouched.
        assert not (tmp_path / CONFIG_FILENAME).exists()

    def test_creates_directory_if_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "nested" / "deep"
        assert not target.exists()
        result = runner.invoke(app, ["init", "-d", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / CONFIG_FILENAME).is_file()


class TestInitCiPin:
    """``init`` warns about an old workflow it did not write; ``--ci`` never warns about its own."""

    @staticmethod
    def _stale_workflow(root: Path) -> None:
        path = root / CI_WORKFLOW_PATH
        path.parent.mkdir(parents=True)
        path.write_text(
            "on: push\njobs:\n  evalshift:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - uses: evalshift/evalshift-action@v0\n"
            '        with:\n          evalshift-version: "0.0.1"\n',
            encoding="utf-8",
        )

    def test_ci_flag_does_not_warn_about_the_workflow_it_wrote(
        self, in_tmp: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("evalshift_cli.cli.commands.init.__version__", "1.2.3")
        result = runner.invoke(app, ["init", "--ci"])
        assert result.exit_code == 0, result.stdout
        assert "CI installs" not in result.stdout

    def test_ci_flag_does_not_warn_when_overwriting_a_stale_workflow(
        self, in_tmp: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("evalshift_cli.cli.commands.init.__version__", "1.2.3")
        self._stale_workflow(in_tmp)
        result = runner.invoke(app, ["init", "--ci", "--force"])
        assert result.exit_code == 0, result.stdout
        assert "CI installs" not in result.stdout

    def test_plain_init_warns_next_to_a_stale_workflow(
        self, in_tmp: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("evalshift_cli.cli.commands.init.__version__", "1.2.3")
        self._stale_workflow(in_tmp)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0, result.stdout
        assert "CI installs evalshift 0.0.1" in result.stdout
        assert 'evalshift-version: "1.2.3"' in result.stdout
