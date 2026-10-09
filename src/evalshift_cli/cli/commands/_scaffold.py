"""Scaffolding content and helpers for ``evalshift init``.

``init`` writes a single, minimal ``evalshift.yaml``. The migration-profile
policy blocks, the optional CI workflow template, and the file-writing
helper live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal

import typer
from rich.console import Console

CI_WORKFLOW_PATH: Final = ".github/workflows/evalshift.yml"

MigrationProfile = Literal[
    "model-upgrade",
    "cost-reduction",
    "local-model",
    "quantization",
    "provider-switch",
]

# Shared typer option for the migration profile (used by ``init``).
ProfileOption = Annotated[
    MigrationProfile,
    typer.Option(
        "--profile",
        help="Migration profile used to scaffold default regression budgets.",
    ),
]

PROFILE_POLICIES: Final[dict[str, str]] = {
    "model-upgrade": """\
migration_policy:
  max_overall_regression_rate: 0.03
  max_critical_regressions: 0
  min_equivalence_rate: 0.95
  max_tool_argument_drift: 0.01
  max_tool_divergence: 0.03
  max_invariant_violations: 0
  max_cost_increase: 0.20
  max_latency_increase: 0.30
""",
    "cost-reduction": """\
migration_policy:
  max_overall_regression_rate: 0.02
  max_critical_regressions: 0
  min_equivalence_rate: 0.97
  max_tool_argument_drift: 0.01
  max_tool_divergence: 0.02
  max_invariant_violations: 0
  max_cost_increase: 0.05
  max_latency_increase: 0.30
""",
    "local-model": """\
migration_policy:
  max_overall_regression_rate: 0.05
  max_critical_regressions: 0
  min_equivalence_rate: 0.90
  max_tool_argument_drift: 0.02
  max_tool_divergence: 0.05
  max_invariant_violations: 0
  max_cost_increase: 0.00
  max_latency_increase: 0.50
""",
    "quantization": """\
migration_policy:
  max_overall_regression_rate: 0.02
  max_critical_regressions: 0
  min_equivalence_rate: 0.97
  max_tool_argument_drift: 0.005
  max_tool_divergence: 0.02
  max_invariant_violations: 0
  max_cost_increase: 0.00
  max_latency_increase: 0.20
""",
    "provider-switch": """\
migration_policy:
  max_overall_regression_rate: 0.03
  max_critical_regressions: 0
  min_equivalence_rate: 0.95
  max_tool_argument_drift: 0.01
  max_tool_divergence: 0.03
  max_invariant_violations: 0
  max_cost_increase: 0.20
  max_latency_increase: 0.40
""",
}

# ``init`` scaffolds the starting point of a real migration, so its default
# model-upgrade budgets are looser than the other profiles: a first run on a
# fresh suite should report the regressions it found, not fail the gate on a
# couple of reworded tool arguments. These mirror the ``MigrationPolicy``
# field defaults, so a config that omits the block behaves the same as one
# that keeps it.
INIT_PROFILE_POLICIES: Final[dict[str, str]] = {
    **PROFILE_POLICIES,
    "model-upgrade": """\
migration_policy:
  max_overall_regression_rate: 0.30
  max_critical_regressions: 1
  min_equivalence_rate: 0.75
  max_tool_argument_drift: 0.20
  max_tool_divergence: 0.20
  max_invariant_violations: 0
  max_cost_increase: 0.30
  max_latency_increase: 0.30
""",
}


# Env var each provider's key must arrive under — litellm reads them by name.
PROVIDER_API_KEY_ENVS: Final[dict[str, str]] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# The `__PROVIDER_API_KEY__` / `__EVALSHIFT_VERSION__` sentinels are replaced
# by ``render_ci_workflow`` — plain ``str.format`` would fight the workflow's
# own ``${{ }}`` expressions and ``string.Template`` its shell ``$vars``.
CI_WORKFLOW_TEMPLATE: Final = """\
# EvalShift eval-regression gate — scaffolded by `evalshift init --ci`.
#
# On every PR (and push to main) this replays each captured suite against the
# source/target models in evalshift.yaml, pushes the run to hosted EvalShift
# (https://www.evalshift.dev), and posts a PR comment + commit status with any
# regression against the base-branch baseline. Push runs on main are what
# create those baselines, so keep the `push:` trigger.
#
# Setup checklist — everything this file needs, in one place:
#
#   1. Repo secrets (Settings -> Secrets and variables -> Actions):
#        EVALSHIFT_TOKEN       hosted EvalShift service-account key (es_...).
#                              Mint it in the web app (org Settings -> API
#                              tokens -> Service accounts) scoped to
#                              run:create + run:read + policy:read. Not a
#                              personal token.
#        __PROVIDER_API_KEY__      key for the provider your evalshift.yaml models
#                              use. Add further keys here (and under `env:`
#                              below) if your judge/embedding models live in
#                              another family.
#      Until EVALSHIFT_TOKEN is set, runs no-op green with a notice — this
#      workflow never fails just because setup isn't finished.
#
#   2. Commit your suites. `evalshift capture sync` writes them under
#      `.evalshift/suites/<name>/golden.jsonl`, but `.evalshift/` is usually
#      gitignored wholesale. Keep the runtime data ignored and un-ignore just
#      the suites (+ their toolset sidecars) in .gitignore:
#          .evalshift/*
#          !.evalshift/suites/
#          !.evalshift/toolsets/
#      With no suites committed, this workflow skips (green) with a notice.
#
#   3. Branch protection: require the single check "evalshift gate" (the join
#      job below) — NOT the per-suite jobs (their names are dynamic) and NOT
#      the `evalshift/regression` commit status (with several suites, every
#      suite writes that same status, so the last one to finish overwrites the
#      rest; the gate job is the only surface that sees every suite).
#
# How suites get evaluated: the action evaluates ONE suite per invocation, so
# the `discover` job lists `.evalshift/suites/*/golden.jsonl` and fans out a
# matrix job per suite, selecting each by the name it is wired under in
# evalshift.yaml. A suite added by `capture sync` is picked up on the next run —
# no workflow edit.
#
# How the verdict is decided: `fail-on: policy` asks hosted EvalShift for the
# verdict the run itself computed against the `migration_policy` limits in
# evalshift.yaml — the same budgets `init` scaffolded, carried in the pushed
# bundle rather than re-scored server-side. Other modes: `regression` (fail on
# any regressed example), `any-slice-regression`, `never`. If the policy check
# is unreachable the action falls back to plain regression gating and says so.

name: evalshift

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

permissions:
  contents: read # actions/checkout
  pull-requests: write # PR comment
  issues: write # comment upsert uses the issues API
  statuses: write # evalshift/regression commit status

# Supersede in-flight runs when a PR gets a new push — but never cancel runs
# on main: those produce the base-branch baselines every future PR diffs
# against, and two quick merges would otherwise eat the first baseline.
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  discover:
    name: discover suites
    runs-on: ubuntu-latest
    outputs:
      suites: ${{ steps.list.outputs.suites }}
    steps:
      - uses: actions/checkout@v7
      - name: List suites under .evalshift/suites
        id: list
        # Directory names, which are the `suites:` keys the eval job selects by:
        # `capture sync` promotes a suite to `.evalshift/suites/<name>/golden.jsonl`
        # and wires it under that same `<name>`.
        # nullglob: with no suites yet (fresh project, suites not committed)
        # the glob must yield an empty list — not the literal pattern, which
        # would matrix a job over a suite named "*" and fail every run.
        run: |
          shopt -s nullglob
          names=()
          for f in .evalshift/suites/*/golden.jsonl; do
            names+=("$(basename "$(dirname "$f")")")
          done
          if [ "${#names[@]}" -eq 0 ]; then
            echo "::notice::No suites under .evalshift/suites/ — run 'evalshift capture sync' and commit the suites (see the checklist at the top of this workflow)."
            suites='[]'
          else
            suites=$(printf '%s\\n' "${names[@]}" | jq -R . | jq -sc .)
          fi
          echo "Discovered suites: $suites"
          echo "suites=$suites" >> "$GITHUB_OUTPUT"

  evalshift:
    name: eval ${{ matrix.suite }}
    runs-on: ubuntu-latest
    needs: discover
    # Skip when there is nothing to evaluate, and on PRs from forks (fork PRs
    # cannot read repo secrets). Both skips turn green via the gate job.
    if: ${{ needs.discover.outputs.suites != '[]' && (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository) }}
    strategy:
      # One suite regressing must not cancel the others' results.
      fail-fast: false
      # Hosted EvalShift refuses uploads past your org's concurrent-run
      # limit (Pro: 5; Enterprise: by contract). Raise this toward it
      # to evaluate suites in parallel.
      max-parallel: 1
      matrix:
        suite: ${{ fromJSON(needs.discover.outputs.suites) }}
    env:
      HAS_EVALSHIFT_TOKEN: ${{ secrets.EVALSHIFT_TOKEN != '' }}
      EVALSHIFT_NONINTERACTIVE: "1"
      __PROVIDER_API_KEY__: ${{ secrets.__PROVIDER_API_KEY__ }}
      __EMBEDDING_API_KEY__: ${{ secrets.__EMBEDDING_API_KEY__ }}
    steps:
      - uses: actions/checkout@v7

      - name: Note missing EVALSHIFT_TOKEN secret
        if: ${{ env.HAS_EVALSHIFT_TOKEN != 'true' }}
        run: |
          echo "::notice::EVALSHIFT_TOKEN not set — add it under Settings -> Secrets -> Actions to enable the hosted regression gate."

      - name: Run evalshift on ${{ matrix.suite }}
        if: ${{ env.HAS_EVALSHIFT_TOKEN == 'true' }}
        uses: evalshift/evalshift-action@v0
        with:
          token: ${{ secrets.EVALSHIFT_TOKEN }}
          config: evalshift.yaml
          # By NAME, not by path: `suite-name` keys into the `suites:` block in
          # evalshift.yaml, which is where each captured suite's own evaluators
          # live (`capture sync` writes a tool-calling suite its tool evaluators
          # there). `suite: <path>` loads the same rows but resolves no name, so
          # the suite is scored with the top-level `evaluators:` instead --
          # silently, and for a tool-calling suite that means no scored rows at
          # all and a run that dies at analyze with an empty scores.jsonl.
          suite-name: ${{ matrix.suite }}
          # Pin CI to the CLI version that scaffolded this project; the
          # action's own default can lag, and evalshift.yaml from a newer CLI
          # fails loudly on an older one. Bump when you upgrade locally.
          evalshift-version: "__EVALSHIFT_VERSION__"
          fail-on: policy
          # The PR comment's marker is one constant, so with several suites
          # every job would overwrite one comment with just its own summary —
          # let only the first matrix job comment. The full per-suite verdict
          # is this workflow's job list; the merge gate is the gate job.
          comment: ${{ strategy.job-index == 0 }}

  # The one check to require in branch protection. Collapses the dynamic
  # matrix into a single stable-named verdict: fails if any suite failed,
  # passes when evaluation was skipped (fork PR, no suites, no token yet).
  gate:
    name: evalshift gate
    runs-on: ubuntu-latest
    needs: [discover, evalshift]
    if: ${{ always() }}
    steps:
      - name: Collapse suite results into one verdict
        env:
          DISCOVER_RESULT: ${{ needs.discover.result }}
          EVAL_RESULT: ${{ needs.evalshift.result }}
        run: |
          echo "discover: $DISCOVER_RESULT, evalshift: $EVAL_RESULT"
          if [ "$DISCOVER_RESULT" != "success" ]; then
            echo "::error::suite discovery failed"
            exit 1
          fi
          case "$EVAL_RESULT" in
            success) echo "all suites passed the gate" ;;
            skipped) echo "::notice::evaluation skipped (fork PR, or no suites committed yet) — gate passes" ;;
            *)
              echo "::error::one or more suites failed the eval gate — see the eval jobs above"
              exit 1
              ;;
          esac
"""


# Present only when ``init`` borrowed another provider's embedding model; an
# unset secret arrives as "" and the CLI then skips semantic with a warning.
_EMBEDDING_KEY_LINE: Final = "      __EMBEDDING_API_KEY__: ${{ secrets.__EMBEDDING_API_KEY__ }}\n"


def render_ci_workflow(*, provider: str, version: str, embedding_api_key: str | None = None) -> str:
    """Render the ``--ci`` GitHub Actions workflow for a provider.

    Args:
        provider: Key of :data:`PROVIDER_API_KEY_ENVS`; selects the provider
            API key the workflow wires through as a secret.
        version: CLI version to pin via the action's ``evalshift-version``
            input, normally :data:`evalshift_cli.__version__`.
        embedding_api_key: The env var of a borrowed embedding provider, wired
            through as a second secret; ``None`` when the provider embeds itself.
    """
    template = (
        CI_WORKFLOW_TEMPLATE.replace(_EMBEDDING_KEY_LINE, "")
        if embedding_api_key is None
        else CI_WORKFLOW_TEMPLATE.replace("__EMBEDDING_API_KEY__", embedding_api_key)
    )
    return template.replace("__PROVIDER_API_KEY__", PROVIDER_API_KEY_ENVS[provider]).replace(
        "__EVALSHIFT_VERSION__", version
    )


def write_scaffold_files(
    *,
    target: Path,
    files: dict[str, str],
    force: bool,
    console: Console,
) -> None:
    """Write ``files`` under ``target``; refuse to clobber unless ``force``.

    Raises ``typer.Exit(1)`` (after printing guidance) if any target file
    already exists and ``force`` is not set.
    """
    if not force:
        existing = [name for name in files if (target / name).exists()]
        if existing:
            console.print(
                "[red]Refusing to overwrite existing files:[/red] " + ", ".join(existing),
            )
            console.print(
                "Re-run with [bold]--force[/bold] to overwrite, or move/delete them first.",
            )
            raise typer.Exit(code=1)

    for name, body in files.items():
        out_path = target / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
        console.print(f"[green]✓[/green] wrote {out_path}")


__all__ = [
    "CI_WORKFLOW_PATH",
    "CI_WORKFLOW_TEMPLATE",
    "PROFILE_POLICIES",
    "MigrationProfile",
    "ProfileOption",
    "write_scaffold_files",
]
