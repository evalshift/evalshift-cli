"""Pydantic models for run state and per-call records.

Two complementary types live here:

* :class:`RunState` — top-level state of an in-flight run. Persisted to
  ``state.json`` per the PDF §5.4 schema, used by the resume logic, and
  rendered in Rich progress UIs.
* :class:`Call` — one row of ``raw.jsonl``: one *example* replayed
  against one model, which is a single LLM call unless the example asks
  for a teacher-forced multi-round replay, in which case it is the merged
  result of one call per replayed round (see
  :meth:`~evalshift_cli.suite.models.SuiteExample.rounds_to_replay`). The
  Phase 5 evaluators consume this stream and pair up the ``role="source"``
  and ``role="target"`` rows for each ``(prompt_id, example_id)`` to
  produce evaluations.

We deliberately store one row per (prompt, example, role) — not per pair,
and not per round — because:

1. Resume logic stays simple — every crash leaves a coherent prefix of
   ``raw.jsonl``; we just skip what's already there.
2. The Phase 5 evaluators don't gain much from a pre-paired row — they
   need to walk every example anyway.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evalshift_cli.evaluators.tool_models import ToolTrace

CallRole = Literal["source", "target"]
RunStatus = Literal["in_progress", "completed", "failed"]


class _StrictModel(BaseModel):
    """Forbid extras + validate on assignment.

    Mirrors the same-named bases in :mod:`evalshift_cli.config.models` and
    :mod:`evalshift_cli.suite.models`. We keep these in parallel rather than
    extracting a single shared base because the diverging defaults are
    likely (e.g. config wants strict, but ``Call`` may eventually allow
    extra provider-specific metadata).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunModels(_StrictModel):
    """Source/target model pair for a run.

    Stored under the ``models`` key in ``state.json`` for symmetry with
    the PDF §5.4 schema example.
    """

    source: str = Field(min_length=1, description="Canonical id of the source model.")
    target: str = Field(min_length=1, description="Canonical id of the target model.")


class UnmeasuredPair(_StrictModel):
    """One (prompt, example) pair an evaluator was handed but did not score.

    Attributes:
        prompt_id: Prompt the pair was produced under.
        example_id: Suite example the pair belongs to.
    """

    prompt_id: str
    example_id: str


class EvaluatorCoverage(_StrictModel):
    """What one evaluator *axis* was asked to score, and what it scored.

    One entry per ``(evaluator_name, kind)``, not per evaluator:
    ``tool_selection`` scores conformance and divergence independently and
    writes a row for each, so a single tally could report ``recorded``
    above ``attempted`` and would hide an axis that measured nothing behind
    one that measured everything.

    An evaluator that measures nothing on a pair writes no row, so
    ``scores.jsonl`` no longer carries any trace of the attempt. Without
    this, an evaluator that measured 2 of 10 pairs would report a rate over
    a denominator of 2 with the other 8 invisible, and one that measured
    *none* would vanish from the analysis entirely — taking its
    ``severity: insufficient`` verdict with it, which is the only thing
    standing between a no-measurement run and a silent pass.

    Attributes:
        evaluator_name: The user-chosen name, matching ``EvalRecord``'s.
        kind: The axis's type slug (see :class:`EvalRecord.kind`). Together
            with ``evaluator_name`` it is what the analysis layer groups
            comparisons by, so coverage lines up with the rows it accounts
            for.
        attempted: Pairs this axis was handed.
        recorded: Pairs that produced a row in ``scores.jsonl``, errored
            rows included — an error is a broken measurement, not an absent
            one. ``attempted - recorded`` is the not-applicable count.
        unmeasured: The attempted pairs that produced no row, named
            individually so the analysis layer can attribute them to the
            same slices a real row would have landed in. Empty on a healthy
            run, so this costs nothing until something is wrong.
        blocking: The evaluator's config ``blocking`` flag. Every scored row
            carries it, but an axis that measured *nothing* has no rows —
            this is then the only place the flag survives, and without it
            the policy layer named silent ``blocking: false`` evaluators as
            blind gates. Defaults ``True`` so a ``state.json`` written
            before the field existed reads as gating — over-warning is the
            conservative side.
    """

    evaluator_name: str
    kind: str = ""
    attempted: int = Field(ge=0)
    recorded: int = Field(ge=0)
    unmeasured: list[UnmeasuredPair] = Field(default_factory=list)
    blocking: bool = True


class SkippedEvaluator(_StrictModel):
    """An evaluator the evaluate stage did not run because its model had no key.

    Written by ``evaluate`` for an *advisory* ``semantic`` / ``llm_judge``
    entry whose judge or embedding model had no API key — a blocking one is
    refused instead. Recorded, not just printed, because ``analyze``, the
    report and the bundle run later and each must still say what was missing.

    Attributes:
        evaluator_name: The name its records would have carried.
        kind: ``"semantic"`` or ``"llm_judge"``.
        label: How messages name it.
        model: The judge or embedding model id as written in config.
        env_vars: The env vars that would have satisfied it, primary first.
        note: A side effect of the skip (e.g. the ``tool_arguments``
            fallback), or ``""``.
    """

    evaluator_name: str
    kind: str
    label: str
    model: str
    env_vars: list[str]
    note: str = ""


class RunState(_StrictModel):
    """Top-level state of an in-flight run.

    Persisted at ``.evalshift/runs/<run_id>/state.json`` after every
    checkpoint. The Phase 4 orchestrator writes it; ``--resume`` reads
    it.

    Attributes:
        run_id: Stable identifier (``r_YYYYMMDD_<6hex>``).
        status: ``"in_progress"`` while running, ``"completed"`` on
            clean exit, ``"failed"`` on unrecoverable error.
        config_hash: SHA-256 of the canonicalised ``evalshift.yaml`` +
            suite path. Resume aborts if this changes between attempts.
        started_at: UTC timestamp the run was first kicked off.
        last_checkpoint_at: UTC of the most recent state write, or
            ``None`` before the first checkpoint.
        models: Source and target canonical model ids.
        prompt_ids: Stable list of prompt ids in the run.
        suite_path: Original path to the JSONL suite, kept verbatim
            so reports can quote it back.
        suite_name: The ``suites:`` key the run was launched with
            (``--suite-name``), or ``None`` for a raw ``--suite <path>`` run.
            Recorded because evaluate/report/bundle all run *after* the CLI
            invocation is gone and each needs to resolve the same per-suite
            evaluator set via
            :meth:`~evalshift_cli.config.models.EvalShiftConfig.evaluators_for` —
            the run must remember which suite it was, not just where the file
            sat.
        total_evaluations: Total ``Call`` rows implied by the run shape
            (``len(prompts) * len(examples) * 2`` models). One row per
            example per role, so a teacher-forced multi-round example
            counts once here however many rounds it replays — the *cost*
            estimate counts rounds (see
            :func:`evalshift_cli.utils.cost.estimate_run_cost`), the
            progress denominator does not.
        completed_evaluations: Calls completed so far. Drives the
            progress bar and the resume "skip" filter.
        non_deterministic_models: Canonical ids of models in this run that
            do not honour ``temperature``, so their outputs vary between
            identical calls. Recorded at run start rather than recomputed
            later: a bundle is immutable, and a LiteLLM upgrade between the
            run and a subsequent ``evalshift report`` must not rewrite what
            was true when the calls were made. Empty for every run where
            both arms sample deterministically.
        dropped_params: Canonical model id → the sorted generation parameters
            the suite's captures asked for that LiteLLM says the model does
            not accept. ``models/client.py`` sets ``drop_params=True``, so
            those calls still succeed — with the constraint missing. Recorded
            at run start for the same reason as ``non_deterministic_models``:
            the answer must be the one that was true when the calls were made.
            Models that honour everything are absent, so an empty dict means
            every recorded constraint reached both arms.
        evaluator_coverage: Per-evaluator attempted-vs-recorded counts,
            written by the ``evaluate`` stage rather than the orchestrator —
            it is the one piece of run-level state only scoring knows. Empty
            until the run has been scored. See :class:`EvaluatorCoverage`.
        samples_per_example: ``defaults.samples_per_example`` at run start —
            how many :class:`Call` rows each ``(prompt, example, role)`` has.
            Recorded so evaluate and report can read the run's shape without
            the config that launched it. ``1`` for every run made before the
            field existed.
        skipped_evaluators: Advisory ``semantic`` / ``llm_judge`` entries the
            ``evaluate`` stage skipped because their model had no API key.
            Empty when every evaluator ran. See :class:`SkippedEvaluator`.
    """

    run_id: str = Field(min_length=1)
    status: RunStatus = "in_progress"
    config_hash: str = Field(min_length=1)
    started_at: datetime
    last_checkpoint_at: datetime | None = None
    models: RunModels
    prompt_ids: list[str] = Field(min_length=1)
    suite_path: str
    # Defaulted so state.json files written before this field existed still
    # load under extra="forbid" and --resume keeps working across upgrades.
    suite_name: str | None = None
    total_evaluations: int = Field(ge=0)
    completed_evaluations: int = Field(default=0, ge=0)
    # Defaulted so state.json files written before this field existed still
    # load under extra="forbid" and --resume keeps working across upgrades.
    non_deterministic_models: list[str] = Field(default_factory=list)
    # Defaulted for the same reason as the field above: state.json files
    # written before it existed must still load under extra="forbid".
    dropped_params: dict[str, list[str]] = Field(default_factory=dict)
    # Written by `evalshift evaluate`, which rewrites state.json once scoring
    # finishes. Defaulted because every state.json is written by the
    # orchestrator first, long before any evaluator has run.
    evaluator_coverage: list[EvaluatorCoverage] = Field(default_factory=list)
    # Defaulted so state.json files written before this field existed still
    # load under extra="forbid".
    samples_per_example: int = Field(default=1, ge=1)
    # Written by `evalshift evaluate` alongside evaluator_coverage. Defaulted
    # so state.json files written before this field existed still load under
    # extra="forbid".
    skipped_evaluators: list[SkippedEvaluator] = Field(default_factory=list)


class Call(_StrictModel):
    """One row of ``raw.jsonl`` — one example replayed against one model.

    Usually that is a single completed LLM call. An example carrying
    ``tool_result_fixtures`` is replayed teacher-forced over several rounds
    (:meth:`~evalshift_cli.suite.models.SuiteExample.rounds_to_replay`) and
    still produces exactly one row: tokens, cost and latency are summed over
    the rounds, ``text`` is the last round's answer, and ``trace`` is the
    merged multi-round :class:`~evalshift_cli.evaluators.tool_models.ToolTrace`.

    Successful calls have ``error=None`` and a populated ``text``.
    Failed calls record the exception message in ``error`` and leave
    ``text`` empty. Either way the call counts as "done" for resume
    purposes — we don't retry failed calls automatically across resumes,
    because the most common cause of a failure is a deterministic
    config/key issue that wouldn't be fixed by re-running.

    Attributes:
        run_id: Owning run.
        prompt_id: Prompt this call belongs to.
        example_id: Suite example whose inputs were rendered.
        model_id: Canonical id of the model that was called.
        role: ``"source"`` or ``"target"``.
        sample_index: Which repeat of this ``(prompt, example, role)`` the
            row is, ``0``-based (see ``defaults.samples_per_example``).
            Always ``0`` on a single-sample run, and on every row written
            before the field existed. Part of the resume key and of the
            evaluate stage's pairing key: sample *i* of the source is scored
            against sample *i* of the target.
        text: The model's response text — the *last* round's, for a
            multi-round replay. Empty string on error.
        input_tokens / output_tokens: From the provider response, summed
            over replayed rounds.
        cost_usd: Per-call cost (``litellm.completion_cost``), summed over
            replayed rounds; ``0.0`` when the model isn't priced.
        latency_ms: Wall time of the provider call, summed over replayed
            rounds. A round served from the cache contributes the latency
            recorded when it originally ran live, not ``0`` — so whenever
            :attr:`latency_replayed` is true the figure is not a measurement
            of this run.
        cached: ``True`` if the response came from the local cache — for a
            multi-round replay, only when every round did (each round is
            its own cache entry). ``cached`` means "nothing was spent on this
            run"; a row with any live round is not cached.
        cached_rounds: How many of the row's provider rounds were served
            from the cache: ``0`` for a fully live row (and for every row
            written before the field existed), the round count for a fully
            cached one, in between for a multi-round replay that re-sent only
            some rounds. Anything above ``0`` means ``latency_ms`` mixes in
            replayed latencies — see :attr:`latency_replayed`.
        error: ``None`` on success; the stringified error on failure. A
            multi-round replay that fails part-way names the round it died
            in (``"round 2/3: <error>"``) and records no ``trace`` — a
            partially replayed example is an unmeasured example.
        finish_reason: The provider's normalised stop reason — for a
            multi-round replay the last round's, unless any earlier round
            was ``"length"``, which wins. ``"length"``
            means the output was truncated by the ``max_tokens`` cap;
            :meth:`truncated` reports this. Truncated calls are excluded
            from the paired regression statistics (via the evaluator's
            error path) so a cut-off output can't manufacture a false
            regression.
    """

    run_id: str
    prompt_id: str
    example_id: str
    model_id: str
    role: CallRole
    # Defaulted so pre-existing raw.jsonl lines still validate on resume.
    sample_index: int = Field(default=0, ge=0)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    # Defaulted so pre-existing raw.jsonl lines still validate on resume.
    cached_rounds: int = Field(default=0, ge=0)
    error: str | None = None
    # v0.2 — populated only for tool-aware calls; ``None`` for plain text
    # ones. The orchestrator switches between ``ModelClient.complete`` and
    # ``complete_with_tools`` per call, based on whether the dispatched
    # example's own resolved toolset (see
    # ``runner.orchestrator.resolve_example_tools``) is non-empty — not on
    # anything the owning prompt declares.
    trace: ToolTrace | None = None
    # Optional (defaulted) so pre-existing raw.jsonl lines still validate
    # on resume. ``"length"`` flags a token-cap truncation.
    finish_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        """True if this call has no error attached."""
        return self.error is None

    @property
    def truncated(self) -> bool:
        """True when the provider cut the output off at the token cap."""
        return self.finish_reason == "length"

    @property
    def latency_replayed(self) -> bool:
        """True when any part of ``latency_ms`` was replayed from the cache.

        Such a row's latency is not a measurement of this run, so it stays out
        of live latency statistics and makes a latency delta incomparable. A
        partly cached multi-round row is not :attr:`cached` (it spent money)
        but is latency-replayed. ``cached`` is checked too because rows
        written before ``cached_rounds`` existed read it back as ``0``.
        """
        return self.cached or self.cached_rounds > 0


def representative_calls(calls: Iterable[Call]) -> list[Call]:
    """The one call per ``(prompt, example, role)`` a consumer should display.

    Every consumer that shows *an* output per example — the report's example
    rows, the bundle, the insights facts, ``inspect`` — reads sample ``0``,
    in the order the rows were written. On a single-sample run this is the
    whole list, unchanged. Totals (cost, tokens, call counts) must keep
    summing over every row; this is for the places that would otherwise
    overwrite one sample with another in a dict keyed by role.

    Args:
        calls: Rows of ``raw.jsonl``, in any order.

    Returns:
        The rows whose ``sample_index`` is ``0``.
    """
    return [c for c in calls if c.sample_index == 0]


__all__ = [
    "Call",
    "CallRole",
    "EvaluatorCoverage",
    "RunModels",
    "RunState",
    "RunStatus",
    "SkippedEvaluator",
    "UnmeasuredPair",
    "representative_calls",
]
