"""Pydantic models for capture files and promoted golden cases.

A *capture* is the JSON envelope the ``evalshift-sdk`` writes per agent
invocation. We mirror only what the CLI needs to read it back:

* The envelope is **forward-compatible** (``extra='ignore'``): future SDK
  releases may add envelope keys, and an older CLI should still read the
  fields it knows. Version gating happens in the reader, not here.
* The inner ``trace`` reuses the strict :class:`~evalshift_cli.traces.models.AgentTrace`
  unchanged — the whole point of the SDK contract is that a capture's trace
  is already a CLI-valid ``AgentTrace``.

A *promoted case* (:class:`PromotedCase`) is the canonical, auditable artifact
``capture promote`` writes to ``<base>/suites/<suite>/<name>.json``. It wraps a
:class:`~evalshift_cli.suite.models.SuiteExample` with provenance the strict
``SuiteExample`` can't carry inline (``extra='forbid'``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evalshift_cli.suite.models import SuiteExample
from evalshift_cli.traces.models import AgentTrace


class CaptureEnvelope(BaseModel):
    """The on-disk capture envelope written by ``evalshift-sdk``.

    Forward-compatible by design: unknown top-level keys from a newer SDK are
    ignored rather than rejected, so a capture written by SDK 1.1 still loads
    under a CLI that only knows 1.0 fields.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(min_length=1)
    capture_id: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    input_hash: str = ""
    code_version: str = ""
    created_at: str = ""
    trace: AgentTrace
    # Multi-turn provenance — optional so captures from older SDK releases
    # (which predate conversation tracking) still parse.
    conversation_id: str | None = None
    turn_index: int | None = None
    parent_capture_id: str | None = None


class PromotedCase(BaseModel):
    """A capture promoted into a golden suite case.

    The canonical, human-auditable form of a promoted case. ``capture promote``
    writes one of these per case; ``golden.jsonl`` (the run-facing index) is
    regenerated from the ``example`` field of every case in the suite dir.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    from_capture: str = Field(min_length=1)
    promoted_at: str = ""
    source_input_hash: str = ""
    code_version: str = ""
    # Which yardstick the tool-call ground truth was taken from:
    # ``requested`` — the calls the model itself asked for, recorded on the
    # capture's ``model_call`` events; ``executed`` — the ``tool_call`` events
    # the app recorded while actually running tools. Reports surface it so a
    # reader knows whether a row measures the model's own output or the app's
    # post-processed version of it. Defaults to ``executed`` because every
    # promoted-case file written before captures carried requested calls was
    # sourced that way, and must keep parsing unchanged.
    promotion_source: Literal["requested", "executed"] = "executed"
    # What the recorded run cost, summed over its ``model_call`` events, and
    # where the figure came from: ``recorded`` -- the app's own
    # instrumentation set ``cost_usd``; ``estimated`` -- at least one event
    # recorded tokens but no cost, and the CLI priced it from litellm's table
    # at promotion (the SDK never prices anything). ``None`` with ``0.0``
    # means nothing was priced: no tokens, or a model litellm does not price
    # (local / self-hosted -- the normal case, not a failure). Provenance of
    # the capture, deliberately not on ``example``: a replay does not
    # reproduce it. Defaults keep pre-existing case files parsing.
    cost_usd: float = Field(default=0.0, ge=0.0)
    cost_source: Literal["recorded", "estimated"] | None = None
    # When the source capture was recorded (its ``created_at``, verbatim) and the model its
    # first ``model_call`` ran on -- the call a replay re-sends. With captures.max_age /
    # captures.max_cases set, ``capture sync`` uses both to tell a stale case from a fresh
    # one. Empty on case files written before they existed; sync backfills them while the
    # capture is still local.
    captured_at: str = ""
    source_model: str = ""
    # Multi-turn provenance — optional so existing promoted-case JSON files
    # (written before conversation tracking) still parse.
    conversation_id: str | None = None
    turn_index: int | None = None
    example: SuiteExample


__all__ = ["CaptureEnvelope", "PromotedCase"]
