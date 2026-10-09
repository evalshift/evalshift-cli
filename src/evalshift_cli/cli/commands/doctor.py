"""Implementation of ``evalshift doctor``.

The doctor command does a fast environmental sanity check so users discover
problems (missing API keys, an invalid ``evalshift.yaml``, the wrong Python
version, …) before they kick off a paid run.

Exit codes:
    * **0** — every check passed, or any failures were merely informational
      (e.g. an unset API key, or no config in this directory yet).
    * **1** — at least one **hard** failure was reported (currently: an
      ``evalshift.yaml`` exists in the cwd but doesn't validate, or its
      ``captures.store`` needs a client extra that isn't installed).

Soft failures (missing API keys, no config yet) are surfaced visually with
a yellow ``✗`` so users see them, but they never fail the command — this
keeps ``doctor`` useful as a fresh-install smoke test before users have set
up their environment.

Not every misconfiguration is visible before the run. :func:`run_checks` is
the pre-flight half; :func:`source_conformance_check` is the post-scoring
half, reported in the same :class:`CheckResult` vocabulary because it is the
same kind of finding — *your setup is wrong*, not *your target model is* —
and ``evaluate`` renders it with this module's own :func:`render_results`.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Final, Literal

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from evalshift_cli import __version__
from evalshift_cli.captures.remote import RemoteStoreError, open_store
from evalshift_cli.captures.store_uri import parse_store_uri
from evalshift_cli.captures.toolset import EMPTY_TOOLSET_FINGERPRINT, fingerprint_tools
from evalshift_cli.cli.commands._suites import SUITE_FILENAME
from evalshift_cli.config.loader import ConfigError, load_config
from evalshift_cli.config.models import EvalShiftConfig
from evalshift_cli.evaluators import tool_selection
from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import BROKEN_HARNESS_CAUSES
from evalshift_cli.models.family import (
    configured_judge_models,
    describe_overlap,
    judge_family_overlaps,
)
from evalshift_cli.models.registry import PROVIDER_ENV_VARS
from evalshift_cli.suite.models import SuiteExample
from evalshift_cli.utils.ci_pin import check_ci_pin, find_action_pins

CheckStatus = Literal["ok", "warn", "fail"]

CONFIG_FILENAME: Final = "evalshift.yaml"
# Row name for the judge-family check (one row per overlapping judge).
JUDGE_FAMILY_CHECK: Final = "judge family"
# The capture SDK: a declared dependency of this package, and the owner of the
# import name ``evalshift`` (this package imports as ``evalshift_cli``).
SDK_DISTRIBUTION: Final = "evalshift-sdk"
SDK_IMPORT_NAME: Final = "evalshift"
# Attributes only the SDK's package exposes. An evalshift CLI from before the
# import rename also imported as ``evalshift`` and has neither, and nor does a
# stray ``evalshift/`` directory on ``sys.path``.
_SDK_MARKERS: Final = ("capture", "SCHEMA_VERSION")
# One entry per provider, primary env var first then accepted aliases. Sourced
# from the model registry so doctor and the client agree on what authenticates.
PROVIDER_KEYS: Final[tuple[tuple[str, ...], ...]] = tuple(PROVIDER_ENV_VARS.values())

# Glyph + Rich style for each check status. The yellow ``✗`` for warnings is
# borrowed from the PDF spec's example output: missing API keys show up as
# ``✗`` but don't fail the command.
_GLYPHS: Final[dict[CheckStatus, tuple[str, str]]] = {
    "ok": ("✓", "green"),
    "warn": ("✗", "yellow"),
    "fail": ("✗", "red"),
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    """A single line in the doctor report.

    Attributes:
        name: Short label (e.g. ``"ANTHROPIC_API_KEY"`` or
            ``"evalshift.yaml"``).
        status: ``"ok"`` (passes), ``"warn"`` (informational, doesn't fail
            the command), or ``"fail"`` (hard failure, exits non-zero).
        detail: Free-form one-line explanation rendered next to the status.
    """

    name: str
    status: CheckStatus
    detail: str


def run_checks(cwd: Path, env: Mapping[str, str]) -> list[CheckResult]:
    """Run every doctor check and return the list of results.

    Pure of side effects beyond reading ``cwd`` and ``env``, which makes it
    trivial to test by monkeypatching either.

    Args:
        cwd: Directory to look in for ``evalshift.yaml``.
        env: Environment-variable mapping (typically ``os.environ``).

    Returns:
        One :class:`CheckResult` per row in the doctor table, in display order.
    """
    results = [_python_check(), _sdk_check()]
    results.extend(_api_key_check(env, aliases) for aliases in PROVIDER_KEYS)
    results.append(_config_check(cwd))
    results.extend(_captures_store_check(cwd))
    results.extend(_tool_consistency_checks(cwd))
    results.extend(_judge_family_checks(cwd))
    results.extend(_ci_pin_check(cwd))
    return results


def _python_check() -> CheckResult:
    v = sys.version_info
    return CheckResult(
        name=f"Python {v.major}.{v.minor}.{v.micro}",
        status="ok",
        detail=f"EvalShift {__version__}",
    )


def _module_location(module: ModuleType) -> str:
    """Where ``module`` was imported from: its package directory, for the shadowing message."""
    file = getattr(module, "__file__", None)
    if file:
        return str(Path(file).parent)
    paths = list(getattr(module, "__path__", []))  # a namespace package has no __file__
    return ", ".join(str(p) for p in paths) if paths else "an unknown location"


def _sdk_check(
    *,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
    dist_version: Callable[[str], str] = version,
) -> CheckResult:
    """Report which package the ``evalshift`` import name resolves to.

    The CLI depends on :data:`SDK_DISTRIBUTION` so that one install serves both
    instrumenting an agent and running evaluations, and so that ``import
    evalshift`` is always the SDK. This row confirms that from inside the
    interpreter that will run the agent: ``ok`` when the SDK imports and carries
    its markers (:data:`_SDK_MARKERS`), ``warn`` when it is missing, when the
    import fails, or when something else answers to the name — an evalshift
    CLI from before the import rename, or a local ``evalshift/`` directory.

    Never ``fail``: the CLI itself does not need the SDK to run.

    Args:
        import_module: Importer to use; injectable for tests.
        dist_version: Distribution-version lookup; injectable for tests.
    """
    try:
        installed: str | None = dist_version(SDK_DISTRIBUTION)
    except PackageNotFoundError:
        installed = None
    try:
        module = import_module(SDK_IMPORT_NAME)
    except Exception as exc:
        if installed is None:
            return CheckResult(
                name=SDK_DISTRIBUTION,
                status="warn",
                detail=(
                    f"not installed (`from {SDK_IMPORT_NAME} import capture` fails in this "
                    f"environment; run `pip install {SDK_DISTRIBUTION}`)"
                ),
            )
        return CheckResult(
            name=SDK_DISTRIBUTION,
            status="warn",
            detail=f"{installed} is installed but `import {SDK_IMPORT_NAME}` failed: {exc}",
        )
    if not all(hasattr(module, marker) for marker in _SDK_MARKERS):
        return CheckResult(
            name=SDK_DISTRIBUTION,
            status="warn",
            detail=(
                f"`import {SDK_IMPORT_NAME}` resolves to {_module_location(module)}, not the "
                f"SDK — an older evalshift CLI's leftover files or a local "
                f"{SDK_IMPORT_NAME}/ directory shadow it; remove them"
            ),
        )
    shown = installed or getattr(module, "__version__", None) or "unknown version"
    return CheckResult(
        name=SDK_DISTRIBUTION,
        status="ok",
        detail=f"{shown} (import name `{SDK_IMPORT_NAME}`)",
    )


def _api_key_check(env: Mapping[str, str], aliases: tuple[str, ...]) -> CheckResult:
    """Check one provider's API key, shown under its primary env-var name.

    ``aliases`` is the provider's env vars in preference order (primary first).
    The key counts as set if any alias is present; a non-primary hit is noted so
    the user knows which one carried them.
    """
    primary = aliases[0]
    for alias in aliases:
        if env.get(alias):
            detail = "set" if alias == primary else f"set via {alias}"
            return CheckResult(name=primary, status="ok", detail=detail)
    return CheckResult(
        name=primary,
        status="warn",
        detail="not set (calls to this provider will fail)",
    )


def _config_check(cwd: Path) -> CheckResult:
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return CheckResult(
            name=CONFIG_FILENAME,
            status="warn",
            detail=f"not found in {cwd} (run `evalshift init` to create one)",
        )
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        # The row has room for the summary ("1 schema problem found") but not
        # the per-field reasons, so point at the command that prints them.
        return CheckResult(
            name=CONFIG_FILENAME,
            status="fail",
            detail=f"{exc.summary} — run `evalshift validate` for details",
        )
    n = len(cfg.prompts)
    return CheckResult(
        name=CONFIG_FILENAME,
        status="ok",
        detail=f"valid ({n} prompt{'s' if n != 1 else ''})",
    )


def _captures_store_check(cwd: Path) -> list[CheckResult]:
    """One row for ``captures.store`` when the config names one: extra installed, bucket listable.

    No config, an invalid config (the ``evalshift.yaml`` row already reports that) or no
    ``captures.store`` produce no row. A missing client extra is a ``fail`` -- it is a local,
    deterministic problem with a one-line fix. A listing that raises is a ``warn``: doctor runs
    on laptops without cloud credentials, and that must not fail the command.
    """
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return []
    try:
        cfg = load_config(cfg_path)
    except ConfigError:
        return []
    uri = cfg.captures.store
    if uri is None:
        return []
    try:
        store = open_store(parse_store_uri(uri))
    except RemoteStoreError as exc:
        detail = exc.summary if exc.hint is None else f"{exc.summary} — {exc.hint}"
        return [CheckResult(name="captures.store", status="fail", detail=detail)]
    try:
        next(iter(store.list("captures/")), None)
    except Exception as exc:
        return [
            CheckResult(
                name="captures.store",
                status="warn",
                detail=f"{uri} not reachable: {type(exc).__name__}: {exc}",
            )
        ]
    return [CheckResult(name="captures.store", status="ok", detail=f"{uri} reachable")]


def _named_suite_paths(cwd: Path, cfg: EvalShiftConfig) -> list[tuple[str, Path]]:
    """Return ``(name, path)`` for every golden-suite file the toolset check should inspect.

    Projects created by ``evalshift init`` + ``evalshift capture sync`` keep
    their suites under ``.evalshift/suites/<name>/golden.jsonl`` and wire them
    into the managed ``suites:`` block, so there is no ``./golden.jsonl`` to
    find. Configured suites therefore win, named after their ``suites:`` key;
    the flat ``./golden.jsonl`` layout is the fallback for configs that wire
    no suites at all, named after :data:`~evalshift_cli.cli.commands._suites.SUITE_FILENAME`.

    Args:
        cwd: Project directory holding ``evalshift.yaml``; suite paths in the
            ``suites:`` block are relative to it.
        cfg: The loaded configuration.

    Returns:
        Existing suite files with their names, in config order (empty when
        none exist).
    """
    if cfg.suites:
        candidates = [(name, cwd / entry.path) for name, entry in cfg.suites.items()]
    else:
        candidates = [(SUITE_FILENAME, cwd / SUITE_FILENAME)]
    return [(name, path) for name, path in candidates if path.exists()]


def _example_toolset_fingerprint(example: SuiteExample) -> str:
    """Content-address one example's toolset to a ``sha256:`` string, inline or ref alike.

    A ``toolset_ref`` already *is* that fingerprint (verbatim, content-addressed
    at capture time), so it is used as-is with no sidecar I/O. Inline ``tools``
    are fingerprinted fresh via :func:`~evalshift_cli.captures.toolset.fingerprint_tools`
    -- the same algorithm, so an inline example and a ``toolset_ref`` example
    naming the identical toolset always compare equal.
    """
    if example.toolset_ref is not None:
        return example.toolset_ref
    return fingerprint_tools([t.to_anthropic() for t in example.tools or []])


def _describe_toolset_fingerprint(fingerprint: str) -> str:
    """Human-readable label for one toolset fingerprint in a doctor report line."""
    if fingerprint == EMPTY_TOOLSET_FINGERPRINT:
        return "no tools"
    short = fingerprint.removeprefix("sha256:")[:12]
    return f"toolset sha256:{short}…"


def _suite_toolset_check(name: str, examples: list[SuiteExample]) -> CheckResult:
    """One report row for ``name``: the toolset its examples share, or a differing-toolset flag."""
    fingerprints = [_example_toolset_fingerprint(ex) for ex in examples]
    distinct = sorted(set(fingerprints))
    n = len(examples)
    plural = "s" if n != 1 else ""
    if len(distinct) == 1:
        return CheckResult(
            name=f"toolset: {name}",
            status="ok",
            detail=f"{n} example{plural} share one toolset ({_describe_toolset_fingerprint(distinct[0])})",
        )
    return CheckResult(
        name=f"toolset: {name}",
        status="warn",
        detail=(
            f"{n} example{plural} carry {len(distinct)} different toolsets — legal (the "
            "runner dispatches each example's own toolset, not one shared per prompt), but "
            "confirm the split is intentional."
        ),
    )


def _tool_consistency_checks(cwd: Path) -> list[CheckResult]:
    """v0.3 — report the toolset each configured suite carries.

    One ``CheckResult`` per suite (see :func:`_named_suite_paths`):

    * ``ok`` when every example in the suite shares one toolset fingerprint
      (:func:`_example_toolset_fingerprint`) — including "every example
      shares the empty toolset", the truthful signal for an agent that
      genuinely never sees a tool.
    * ``warn`` when a suite's examples carry more than one distinct
      fingerprint. Legal — different examples may legitimately dispatch
      different toolsets — but also the shape a wiring mistake takes, so it
      is surfaced rather than silently accepted.

    Skipped silently when the config or a suite file can't be loaded, and for
    a suite with zero examples — those cases already surface elsewhere in the
    doctor output (or have nothing to report).
    """
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return []
    try:
        cfg = load_config(cfg_path)
    except ConfigError:
        return []

    from evalshift_cli.suite.loader import load_jsonl

    out: list[CheckResult] = []
    for name, suite_path in _named_suite_paths(cwd, cfg):
        try:
            suite = load_jsonl(suite_path)
        except Exception:
            continue
        if not suite.examples:
            continue
        out.append(_suite_toolset_check(name, suite.examples))
    return out


def _judge_family_checks(cwd: Path) -> list[CheckResult]:
    """Report every configured judge that shares a model family with an arm.

    One :data:`JUDGE_FAMILY_CHECK` row per overlapping judge (``warn``), or
    a single ``ok`` row when judges are configured and none overlaps. Never
    ``fail``: a same-family judge is a bias to know about, not a broken
    setup — ``init`` deliberately scaffolds one so a first run needs a
    single API key.

    Silent when there is no loadable config, no ``llm_judge`` evaluator, or
    no ``defaults.source_model`` / ``target_model`` to compare against
    (``doctor`` takes no ``--from`` / ``--to``, and guessing the arms would
    make the row noise).
    """
    cfg_path = cwd / CONFIG_FILENAME
    if not cfg_path.exists():
        return []
    try:
        cfg = load_config(cfg_path)
    except ConfigError:
        return []
    judges = configured_judge_models(cfg)
    source = cfg.defaults.source_model
    target = cfg.defaults.target_model
    if not judges or source is None or target is None:
        return []
    overlaps = judge_family_overlaps(judge_models=judges, source_model=source, target_model=target)
    if not overlaps:
        n = len(judges)
        return [
            CheckResult(
                name=JUDGE_FAMILY_CHECK,
                status="ok",
                detail=f"{n} judge model{'s' if n != 1 else ''} from a third family",
            ),
        ]
    return [
        CheckResult(name=JUDGE_FAMILY_CHECK, status="warn", detail=describe_overlap(o))
        for o in overlaps
    ]


def _ci_pin_check(cwd: Path) -> list[CheckResult]:
    """Report whether CI installs a CLI at least as new as this one.

    One ``ci pin`` row when a workflow under ``cwd/.github/workflows`` uses
    the EvalShift action: ``warn`` with the finding from
    :func:`~evalshift_cli.utils.ci_pin.check_ci_pin`, else ``ok`` naming the pin.
    No row at all when no workflow uses the action.
    """
    pins = find_action_pins(cwd)
    if not pins:
        return []
    finding = check_ci_pin(cwd, __version__)
    if finding is not None:
        return [CheckResult(name="ci pin", status="warn", detail=finding.message)]
    versions = sorted({pin.version for pin in pins if pin.version is not None})
    detail = f"pinned to {', '.join(versions)}" if versions else "not pinned (version unknown)"
    return [CheckResult(name="ci pin", status="ok", detail=detail)]


#: The fraction of a suite's conformance rows the source model has to fail
#: before the run is called a broken harness rather than a migration finding.
#: Half is where the ground truth stops describing the source model at all:
#: the expectations were recorded *from* that model, so it is the one side
#: that should satisfy them, and a coin-flip rate is not sampling noise —
#: it is a different setup. Below half the rows stay a finding, reported as
#: ``TOOL_GROUND_TRUTH_MISS`` counts and excluded from the policy rates, but
#: not an accusation.
BROKEN_HARNESS_RATE: Final = 0.5

#: …and the smallest suite on which that accusation stands on its own data.
#: A 100% failure rate over three rows has a 95% Wilson lower bound of 0.44 —
#: under half — so a three-example smoke suite cannot support the sentence
#: this check prints. At four rows the bound is 0.51 and it can. Sub-threshold
#: suites stay silent rather than guessing.
BROKEN_HARNESS_MIN_ROWS: Final = 4


def source_conformance_check(records: Sequence[EvalRecord]) -> CheckResult | None:
    """Report a suite whose ground truth the *source* model cannot satisfy.

    The conformance axis grades each side absolutely against the expectations
    the suite recorded, and on a captured suite those expectations came from
    the source model. The source is therefore the one side that should always
    conform; when it does not, the comparison never reached the migration —
    it measured the harness, and every rate the run publishes describes that.

    Counted on the **source side alone**, over conformance rows the evaluator
    actually scored:

    * not ``analysis.policy.is_shared_ground_truth_miss``, which is the right
      selector for *excluding* a row from the rates but needs ``delta == 0``
      and a ``TOOL_GROUND_TRUTH_MISS`` tag that requires *both* sides to miss.
      A suite the source fails and the target happens to satisfy is invisible
      to it (``0.0 / 1.0`` is a positive delta and carries no tag) and is
      exactly as misconfigured;
    * errored rows are dropped: their neutral ``0.5`` is a measurement that
      broke, not an expectation that was missed.

    Args:
        records: Every row the scoring stage produced for the run.

    Returns:
        A ``fail``-status :class:`CheckResult` when the source failed at least
        :data:`BROKEN_HARNESS_RATE` of at least :data:`BROKEN_HARNESS_MIN_ROWS`
        conformance rows, else ``None``.
    """
    rows = [r for r in records if r.error is None and r.kind == tool_selection.KIND_CONFORMANCE]
    if len(rows) < BROKEN_HARNESS_MIN_ROWS:
        return None
    failed = sum(1 for r in rows if r.source_score < 1.0)
    rate = failed / len(rows)
    if rate < BROKEN_HARNESS_RATE:
        return None
    return CheckResult(
        name="broken eval harness",
        status="fail",
        detail=(
            f"the source model failed the recorded ground truth on {failed} of "
            f"{len(rows)} tool-selection conformance rows ({rate:.0%}). "
            f"{BROKEN_HARNESS_CAUSES} This run measured the eval harness, not "
            "the migration: no verdict it reports describes the target model "
            "until the suite is re-captured against the agent under test, or "
            "the evaluator's conformance axis is set to off."
        ),
    )


def render_results(results: list[CheckResult], console: Console) -> None:
    """Render the check results as a Rich table.

    Names and details are plain text, escaped so a bracketed value -- an install hint such as
    ``pip install "evalshift[gcs]"`` or a user-written store URI -- is shown, not parsed as markup.
    """
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("status", no_wrap=True)
    table.add_column("name", style="bold")
    table.add_column("detail", overflow="fold")
    for r in results:
        glyph, style = _GLYPHS[r.status]
        table.add_row(f"[{style}]{glyph}[/{style}]", escape(r.name), escape(r.detail))
    console.print(table)


def doctor() -> None:
    """Check environment and configuration; exit 1 on hard failures."""
    cwd = Path.cwd()
    results = run_checks(cwd=cwd, env=os.environ)
    render_results(results, Console())
    if any(r.status == "fail" for r in results):
        raise typer.Exit(code=1)


__all__ = [
    "BROKEN_HARNESS_MIN_ROWS",
    "BROKEN_HARNESS_RATE",
    "CONFIG_FILENAME",
    "JUDGE_FAMILY_CHECK",
    "PROVIDER_KEYS",
    "SDK_DISTRIBUTION",
    "SDK_IMPORT_NAME",
    "CheckResult",
    "CheckStatus",
    "doctor",
    "render_results",
    "run_checks",
    "source_conformance_check",
]
