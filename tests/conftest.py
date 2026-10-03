"""Shared pytest fixtures.

Besides the console-styling guard below, this module owns the *completed run*
fixture — a temporary project plus a finished run directory that both the hosted
CLI tests and the bundle-shape tests build bundles from — and the single
:class:`FakeHostedClient` the push tests drive. Both live here so the suites
cannot drift apart about what a run on disk looks like or what the hosted client
answers.
"""

from __future__ import annotations

import json
import os

# Force plain, unstyled CLI output for the whole suite.
#
# Rich styles output when it detects a terminal, and GitHub Actions sets
# ``GITHUB_ACTIONS``, which Rich treats as one. With styling on, Rich emits ANSI
# codes around each dash-delimited segment of an option name, so ``--suite-name``
# renders as ``-`` + ``-suite`` + ``-name`` and the literal substring never
# appears in stdout. Tests asserting on CLI output then pass locally (no TTY)
# and fail in CI.
#
# ``TERM=dumb`` is what actually disables it. ``NO_COLOR`` is not enough — it
# drops colour but keeps bold, so the option name still splits. ``FORCE_COLOR``
# would override the whole thing, so both are cleared. This must run before any
# Rich console is constructed.
os.environ["TERM"] = "dumb"
os.environ.pop("NO_COLOR", None)
os.environ.pop("FORCE_COLOR", None)

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.hosted.bundle import BundleBuildResult, build_bundle
from evalshift_cli.hosted.client import HostedError, HostedHTTPError
from evalshift_cli.runner.checkpoint import append_call, write_state
from evalshift_cli.runner.models import Call, RunModels, RunState

DEFAULT_RUN_ID = "r_20260516_abcdef"


def write_project_files(root: Path) -> None:
    """Write a minimal `evalshift.yaml` plus a two-example golden suite."""
    (root / "evalshift.yaml").write_text(
        """
        version: 1
        project: acme/model-migration
        prompts:
          - id: greet
            detection: manual
            content: "Hello {name}"
            variables: [name]
        defaults:
          source_model: gemini/gemini-2.5-flash
          target_model: gemini/gemini-3.1-flash-lite-preview
        evaluators:
          structural:
            - type: length
              min_chars: 1
        """,
        encoding="utf-8",
    )
    (root / "golden.jsonl").write_text(
        json.dumps({"id": "ex1", "inputs": {"name": "Ada"}, "tags": ["security"], "tools": []})
        + "\n"
        + json.dumps(
            {"id": "ex2", "inputs": {"name": "Grace"}, "tags": ["checkout", "cart"], "tools": []}
        )
        + "\n",
        encoding="utf-8",
    )


def _score(run_id: str, example_id: str, *, delta: float) -> str:
    return (
        EvalRecord(
            run_id=run_id,
            prompt_id="greet",
            example_id=example_id,
            evaluator_name="length",
            source_score=1.0,
            target_score=1.0 + delta,
            delta=delta,
        ).model_dump_json()
        + "\n"
    )


def write_completed_run(
    root: Path,
    run_id: str = DEFAULT_RUN_ID,
    *,
    with_cached_pair: bool = False,
) -> Path:
    """Write a completed run directory under ``root/.evalshift/runs/<run_id>``.

    Args:
        root: Project root that already has ``write_project_files`` applied.
        run_id: Run identifier to write.
        with_cached_pair: Also record a second, cache-replayed example pair.
            Cache hits carry ``latency_ms=0``, which is what makes the bundle's
            ``latency_comparable`` flag observable.

    Returns:
        The run directory that was written.
    """
    runs_base = root / ".evalshift" / "runs"
    run_dir = runs_base / run_id
    pairs = 2 if with_cached_pair else 1
    state = RunState(
        run_id=run_id,
        status="completed",
        config_hash="local-config-hash",
        started_at=datetime(2026, 5, 16, 12, 0, tzinfo=UTC),
        last_checkpoint_at=datetime(2026, 5, 16, 12, 1, 14, tzinfo=UTC),
        models=RunModels(
            source="gemini/gemini-2.5-flash", target="gemini/gemini-3.1-flash-lite-preview"
        ),
        prompt_ids=["greet"],
        suite_path=str(root / "golden.jsonl"),
        total_evaluations=2 * pairs,
        completed_evaluations=2 * pairs,
    )
    write_state(run_dir, state)
    append_call(
        run_dir,
        Call(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex1",
            model_id="gemini/gemini-2.5-flash",
            role="source",
            text="hello ada",
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.01,
            latency_ms=100,
        ),
    )
    append_call(
        run_dir,
        Call(
            run_id=run_id,
            prompt_id="greet",
            example_id="ex1",
            model_id="gemini/gemini-3.1-flash-lite-preview",
            role="target",
            text="hello ada!",
            input_tokens=10,
            output_tokens=3,
            cost_usd=0.02,
            latency_ms=110,
        ),
    )
    scores = _score(run_id, "ex1", delta=0.0)
    if with_cached_pair:
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex2",
                model_id="gemini/gemini-2.5-flash",
                role="source",
                text="hello grace",
                input_tokens=12,
                output_tokens=4,
                cost_usd=0.03,
                latency_ms=0,
                cached=True,
            ),
        )
        append_call(
            run_dir,
            Call(
                run_id=run_id,
                prompt_id="greet",
                example_id="ex2",
                model_id="gemini/gemini-3.1-flash-lite-preview",
                role="target",
                text="",
                input_tokens=12,
                output_tokens=6,
                cost_usd=0.05,
                latency_ms=0,
                cached=True,
                finish_reason="stop",
            ),
        )
        scores += _score(run_id, "ex2", delta=-0.25)
    (run_dir / "scores.jsonl").write_text(scores, encoding="utf-8")
    (run_dir / "analysis.json").write_text('{"comparisons": []}', encoding="utf-8")
    (run_dir / "report.html").write_text(
        "<!doctype html><title>EvalShift</title>", encoding="utf-8"
    )
    # The cache envelope ``evalshift report`` leaves behind, not the bundle
    # block: ``config_hash`` is cache metadata and must be stripped on the way
    # into a bundle, which is exactly what the bundle-shape suite asserts.
    (run_dir / "insights.json").write_text(
        json.dumps(
            {
                "config_hash": "sha256:" + "b" * 64,
                "insight": {
                    "model": "gemini/gemini-3.1-flash-lite-preview",
                    "generated_at": "2026-05-16T12:02:00Z",
                    "verdict_summary": "PASS under the configured policy.",
                    "advisory_summary": "No advisory evaluator ran.",
                    "economics_summary": "The target costs more per call.",
                    "recommendation": "Safe to migrate under the configured policy.",
                    "findings": [
                        {
                            "kind": "negative",
                            "title": "Target returns no text on the cached pair",
                            "detail": "The second example's target output is empty.",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


@dataclass
class FakeHostedClient:
    """The stand-in for :class:`HostedClient` that every push test drives.

    One fake, not one per test module: it is checked against the real client's
    signatures by ``test_fake_hosted_client_mirrors_hosted_client`` in
    ``tests/unit/test_hosted_client.py``, and a second copy would be a second
    thing to keep in step. Add a method here only when the push flow calls it.

    Attributes:
        responses: ``POST /runs`` responses, popped in order. An entry carrying
            ``raise_404`` or ``raise_error`` raises instead of returning, so a
            test can script the server's error paths.
        finalize_response: What ``finalize_run`` returns — each module binds it
            to the server id and view URL that module's fixtures use.
        projects: What ``list_projects`` reports back.
        created_project: The last ``create_project`` payload, or ``None``.
        finalize_error: Raised by ``finalize_run`` instead of answering.
        list_projects_error: Raised by ``list_projects`` instead of answering.
        create_project_error: Raised by ``create_project`` instead of answering.
        host: The base URL the real client would have been built with -- error
            messages quote it, so the fake has to carry one.
        on_finalize: Called inside ``finalize_run``, before it answers or
            raises — the only moment the push checkpoint is on disk.
        initiate_calls: How many times ``POST /runs`` was called.
        initiate_sizes: The ``size_bytes`` of each ``POST /runs``.
        finalize_urls: Every URL ``finalize_run`` was handed, in order.
    """

    responses: list[dict[str, Any]] = field(default_factory=list)
    finalize_response: dict[str, Any] = field(default_factory=dict)
    projects: list[dict[str, Any]] | None = None
    created_project: dict[str, Any] | None = None
    finalize_error: HostedHTTPError | None = None
    list_projects_error: HostedError | None = None
    create_project_error: HostedError | None = None
    host: str = "https://api.evalshift.test"
    on_finalize: Callable[[], None] | None = None
    initiate_calls: int = 0
    initiate_sizes: list[int] = field(default_factory=list)
    finalize_urls: list[str] = field(default_factory=list)

    def initiate_run(
        self,
        manifest: dict[str, Any],
        *,
        size_bytes: int,
    ) -> dict[str, Any]:
        self.initiate_calls += 1
        self.initiate_sizes.append(size_bytes)
        response = self.responses.pop(0)
        if response.get("raise_404"):
            raise HostedHTTPError(404, "project not found", code="not_found")
        error = response.get("raise_error")
        if isinstance(error, HostedHTTPError):
            raise error
        return response

    def finalize_run(self, finalize_url: str) -> dict[str, Any]:
        self.finalize_urls.append(finalize_url)
        if self.on_finalize is not None:
            self.on_finalize()
        if self.finalize_error is not None:
            raise self.finalize_error
        return dict(self.finalize_response)

    def list_projects(self, org_slug: str) -> list[dict[str, Any]]:
        if self.list_projects_error is not None:
            raise self.list_projects_error
        return list(self.projects or [])

    def create_project(
        self,
        org_slug: str,
        *,
        slug: str,
        name: str,
    ) -> dict[str, Any]:
        if self.create_project_error is not None:
            raise self.create_project_error
        self.created_project = {
            "org_slug": org_slug,
            "slug": slug,
            "name": name,
        }
        return self.created_project


@dataclass(frozen=True, slots=True)
class RunFixture:
    """A completed run on disk plus the arguments needed to bundle it."""

    run_id: str
    root: Path
    run_dir: Path
    config: Path
    suite: Path
    runs_base: Path

    def build(self, **overrides: Any) -> BundleBuildResult:
        """Build the bundle for this run, overriding any ``build_bundle`` kwarg."""
        kwargs: dict[str, Any] = {
            "config_path": self.config,
            "suite_path": self.suite,
            "runs_base": self.runs_base,
        }
        kwargs.update(overrides)
        return build_bundle(self.run_id, **kwargs)


@pytest.fixture(autouse=True)
def _isolated_response_cache(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the default on-disk response cache at a fresh per-test file.

    Command-level tests (``run``, ``compare``, ``evaluate`` and the integration
    pipelines) open the cache without passing a URL, which resolves to
    ``~/.evalshift/cache.db``. Unredirected, they wrote the developer's real
    cache, and a row left there by one test run could serve the next one.
    ``tmp_path_factory`` rather than ``tmp_path``, so tests that inspect their
    own ``tmp_path`` never find a stray ``cache.db`` in it.
    """
    monkeypatch.setattr(
        "evalshift_cli.cache.schema.DEFAULT_CACHE_PATH",
        tmp_path_factory.mktemp("response-cache") / "cache.db",
    )


@pytest.fixture
def run_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunFixture:
    """A completed two-example run, one live pair and one cache-replayed pair.

    Git metadata is pinned through the environment so bundle bytes do not
    depend on the checkout the suite happens to run in.
    """
    write_project_files(tmp_path)
    run_dir = write_completed_run(tmp_path, with_cached_pair=True)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/model-swap")
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    return RunFixture(
        run_id=DEFAULT_RUN_ID,
        root=tmp_path,
        run_dir=run_dir,
        config=tmp_path / "evalshift.yaml",
        suite=tmp_path / "golden.jsonl",
        runs_base=tmp_path / ".evalshift" / "runs",
    )


@pytest.fixture
def built_bundle_path(run_fixture: RunFixture) -> Path:
    """Path to a freshly built bundle for :func:`run_fixture`."""
    return run_fixture.build().path
