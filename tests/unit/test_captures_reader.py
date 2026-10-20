"""Tests for the capture-file reader (``evalshift_cli.captures.reader``).

The reader consumes capture files written by the separate ``evalshift-sdk``
package. The on-disk contract (frozen at SDK schema 1.0.0) is:

    <base>/captures/<suite>/<capture_id>.json

where ``base`` is ``EVALSHIFT_DIR`` if set, else ``.evalshift``. Each file is
a JSON envelope wrapping a CLI-valid ``AgentTrace``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.captures.reader import (
    CaptureError,
    capture_base,
    capture_suite_dirname,
    capture_toolset_refs,
    captures_root,
    find_capture,
    iter_captures,
    load_capture,
    load_toolset,
    promoted_capture_ids,
    promoted_toolset_refs,
    toolset_path,
    toolsets_root,
)
from evalshift_cli.captures.toolset import fingerprint_tools
from evalshift_cli.evaluators.tool_models import ToolSpec
from evalshift_cli.suite.models import SuiteExample
from evalshift_cli.traces.models import ModelCallEvent

_TOOLSET_REF = "sha256:" + "ab" * 32


def _write_toolset(base: Path, ref: str, tools: list[dict[str, Any]]) -> Path:
    path = toolset_path(ref, base=base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "fingerprint": ref, "tools": tools}),
        encoding="utf-8",
    )
    return path


def _model_call(
    index: int,
    *,
    model_input: Any = "hi",
    toolset_ref: str | None = _TOOLSET_REF,
    tools_offered: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "model_call",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:00+00:00",
        "metadata": {"evalshift": {"span_id": "m1", "start_ts": 1.0, "end_ts": 1.5}},
        "model_id": "claude-opus-4-8",
        "input": model_input,
        "output": "yo",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.01,
        "latency_ms": 500,
        "toolset_ref": toolset_ref,
        "tools_offered": tools_offered if tools_offered is not None else ["search"],
    }


def _tool_call(name: str, index: int, *, call_id: str = "c1") -> dict[str, Any]:
    return {
        "type": "tool_call",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:02+00:00",
        "metadata": {"evalshift": {"span_id": call_id, "start_ts": 2.0, "end_ts": 2.2}},
        "name": name,
        "arguments": {"q": "x"},
        "call_id": call_id,
        "parent_call_id": None,
    }


def _final(text: str, index: int) -> dict[str, Any]:
    return {
        "type": "final_output",
        "sequence_index": index,
        "timestamp": "2026-06-16T12:00:03+00:00",
        "metadata": {},
        "text": text,
    }


def _capture_dict(
    *,
    capture_id: str = "cap_abc",
    suite: str = "support_agent",
    schema_version: str = "2.0.0",
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if events is None:
        events = [_model_call(0), _tool_call("search", 1), _final("done", 2)]
    return {
        "schema_version": schema_version,
        "capture_id": capture_id,
        "suite": suite,
        "input_hash": "deadbeef",
        "code_version": "",
        "created_at": "2026-06-16T12:00:00+00:00",
        "trace": {
            "run_id": capture_id,
            "prompt_id": suite,
            "example_id": capture_id,
            "role": "source",
            "events": events,
        },
    }


def _write_capture(base: Path, payload: dict[str, Any]) -> Path:
    suite_dir = base / "captures" / payload["suite"]
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / f"{payload['capture_id']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_capture_base_prefers_env_over_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EVALSHIFT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert capture_base() == Path(".evalshift")

    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path / "custom"))
    assert capture_base() == tmp_path / "custom"


def test_captures_root_is_base_captures(tmp_path: Path) -> None:
    assert captures_root(base=tmp_path) == tmp_path / "captures"


def test_toolsets_root_is_base_toolsets(tmp_path: Path) -> None:
    assert toolsets_root(base=tmp_path) == tmp_path / "toolsets"


def test_toolset_path_strips_sha256_prefix(tmp_path: Path) -> None:
    ref = "sha256:abc123"
    assert toolset_path(ref, base=tmp_path) == tmp_path / "toolsets" / "abc123.json"


def test_toolset_path_leaves_a_ref_without_the_prefix_unchanged(tmp_path: Path) -> None:
    """Every real ``toolset_ref`` carries the ``sha256:`` prefix (V6); a bare hex string is not
    a case callers are expected to pass. ``removeprefix`` is a no-op rather than a validator, so
    this pins that deliberate choice instead of leaving it to be discovered by accident.
    """
    assert toolset_path("abc123", base=tmp_path) == tmp_path / "toolsets" / "abc123.json"


def test_load_capture_parses_envelope_and_trace(tmp_path: Path) -> None:
    path = _write_capture(tmp_path, _capture_dict())

    envelope = load_capture(path)

    assert isinstance(envelope, CaptureEnvelope)
    assert envelope.capture_id == "cap_abc"
    assert envelope.suite == "support_agent"
    assert envelope.schema_version == "2.0.0"
    assert envelope.trace.prompt_id == "support_agent"
    assert [e.type for e in envelope.trace.events] == ["model_call", "tool_call", "final_output"]


def test_load_capture_tolerates_unknown_envelope_keys(tmp_path: Path) -> None:
    payload = _capture_dict()
    payload["future_field"] = {"added": "in 1.1"}
    path = _write_capture(tmp_path, payload)

    envelope = load_capture(path)

    assert envelope.capture_id == "cap_abc"


def test_load_capture_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(CaptureError) as exc:
        load_capture(path)
    assert exc.value.kind == "json_parse"


def test_load_capture_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CaptureError) as exc:
        load_capture(tmp_path / "nope.json")
    assert exc.value.kind == "missing"


def test_load_capture_rejects_unsupported_major_version(tmp_path: Path) -> None:
    """A 1.x capture predates per-call toolset capture; there is no dual-major support."""
    path = _write_capture(tmp_path, _capture_dict(schema_version="1.1.0"))

    with pytest.raises(CaptureError) as exc:
        load_capture(path)
    assert exc.value.kind == "unsupported_version"


def test_load_capture_accepts_a_newer_minor_schema_version(tmp_path: Path) -> None:
    """The reader gates on the MAJOR only, so an SDK MINOR bump must keep loading.

    The SDK bumps its capture schema to ``2.1.0`` to add
    ``ModelCallEvent.requested_tool_calls``. Nothing in a 2.1.0 capture is
    unreadable by a 2.0.0-era CLI, so refusing it would strand every install
    on the older SDK for no reason.
    """
    path = _write_capture(tmp_path, _capture_dict(schema_version="2.1.0"))

    envelope = load_capture(path)

    assert envelope.schema_version == "2.1.0"


@pytest.mark.parametrize("version", ["1.0.0", "3.0.0"])
def test_load_capture_rejects_other_major_schema_versions(tmp_path: Path, version: str) -> None:
    """Only the 2.x major is readable — either side of it is refused loudly."""
    path = _write_capture(tmp_path, _capture_dict(schema_version=version))

    with pytest.raises(CaptureError) as exc:
        load_capture(path)
    assert exc.value.kind == "unsupported_version"


def test_load_capture_round_trips_requested_tool_calls(tmp_path: Path) -> None:
    """A 2.1.0 capture's model-requested calls survive the whole read path."""
    event = _model_call(0)
    event["requested_tool_calls"] = [
        {"name": "search", "arguments": {"q": "x"}, "call_id": "c1"},
        {"name": "issue_refund", "arguments": {}, "call_id": None},
    ]
    path = _write_capture(tmp_path, _capture_dict(schema_version="2.1.0", events=[event]))

    envelope = load_capture(path)

    model_call = envelope.trace.events[0]
    assert isinstance(model_call, ModelCallEvent)
    assert model_call.requested_tool_calls is not None
    assert [c.name for c in model_call.requested_tool_calls] == ["search", "issue_refund"]
    assert model_call.requested_tool_calls[0].arguments == {"q": "x"}


def test_load_capture_round_trips_toolset_fields(tmp_path: Path) -> None:
    """A 2.0.0 capture's toolset_ref and tools_offered survive load_capture unchanged."""
    event = _model_call(
        0,
        toolset_ref="sha256:" + "cd" * 32,
        tools_offered=["search", "issue_refund"],
    )
    path = _write_capture(tmp_path, _capture_dict(events=[event]))

    envelope = load_capture(path)

    model_call = envelope.trace.events[0]
    assert isinstance(model_call, ModelCallEvent)
    assert model_call.toolset_ref == "sha256:" + "cd" * 32
    assert model_call.tools_offered == ["search", "issue_refund"]


def test_load_capture_accepts_a_model_call_missing_toolset_fields(tmp_path: Path) -> None:
    """Optional at the model level: a capture predating toolset capture still parses.

    Presence is only *required* at promotion (a separate concern, enforced
    elsewhere with an error that can name the capture) -- the reader's job is
    just to accept the shape.
    """
    event = {
        "type": "model_call",
        "sequence_index": 0,
        "timestamp": "2026-06-16T12:00:00+00:00",
        "metadata": {},
        "model_id": "claude-opus-4-8",
        "input": "hi",
        "output": "yo",
        # No toolset_ref / tools_offered at all -- not merely null.
    }
    path = _write_capture(tmp_path, _capture_dict(events=[event]))

    envelope = load_capture(path)

    model_call = envelope.trace.events[0]
    assert isinstance(model_call, ModelCallEvent)
    assert model_call.toolset_ref is None
    assert model_call.tools_offered is None


def test_load_capture_rejects_schema_violation(tmp_path: Path) -> None:
    # tool_result with no preceding tool_call violates the AgentTrace validator.
    bad_event = {
        "type": "tool_result",
        "sequence_index": 0,
        "timestamp": "2026-06-16T12:00:02+00:00",
        "metadata": {},
        "name": "search",
        "call_id": "orphan",
        "result": {"hits": 1},
        "error": None,
    }
    path = _write_capture(tmp_path, _capture_dict(events=[bad_event]))

    with pytest.raises(CaptureError) as exc:
        load_capture(path)
    assert exc.value.kind == "schema"


def test_iter_captures_filters_by_suite(tmp_path: Path) -> None:
    _write_capture(tmp_path, _capture_dict(capture_id="cap_1", suite="alpha"))
    _write_capture(tmp_path, _capture_dict(capture_id="cap_2", suite="alpha"))
    _write_capture(tmp_path, _capture_dict(capture_id="cap_3", suite="beta"))

    alpha = iter_captures(suite="alpha", base=tmp_path)
    assert sorted(r.envelope.capture_id for r in alpha) == ["cap_1", "cap_2"]

    everything = iter_captures(base=tmp_path)
    assert sorted(r.envelope.capture_id for r in everything) == ["cap_1", "cap_2", "cap_3"]


def test_iter_captures_empty_when_no_captures_dir(tmp_path: Path) -> None:
    assert iter_captures(base=tmp_path) == []


def test_iter_captures_reports_unreadable_files(tmp_path: Path) -> None:
    _write_capture(tmp_path, _capture_dict(capture_id="cap_good", suite="alpha"))
    bad = tmp_path / "captures" / "alpha" / "cap_bad.json"
    bad.write_text("{not json", encoding="utf-8")

    seen: list[CaptureError] = []
    records = iter_captures(base=tmp_path, on_error=seen.append)

    assert [r.envelope.capture_id for r in records] == ["cap_good"]
    assert [e.kind for e in seen] == ["json_parse"]
    assert seen[0].path == bad


def test_iter_captures_default_still_skips_silently(tmp_path: Path) -> None:
    (tmp_path / "captures" / "alpha").mkdir(parents=True)
    (tmp_path / "captures" / "alpha" / "cap_bad.json").write_text("{not json", encoding="utf-8")

    assert iter_captures(base=tmp_path) == []


def test_find_capture_locates_by_id(tmp_path: Path) -> None:
    _write_capture(tmp_path, _capture_dict(capture_id="cap_xyz", suite="alpha"))

    record = find_capture("cap_xyz", base=tmp_path)
    assert record.envelope.suite == "alpha"
    assert record.path.name == "cap_xyz.json"


def test_find_capture_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(CaptureError) as exc:
        find_capture("cap_missing", base=tmp_path)
    assert exc.value.kind == "missing"


# ---------------------------------------------------------------------------
# load_toolset — the dedicated sidecar reader (V7: built on ToolSpec.from_dict,
# never on evaluators.tool_loader.load_tools, which rejects an empty list).
# ---------------------------------------------------------------------------


def test_load_toolset_resolves_a_populated_sidecar(tmp_path: Path) -> None:
    tools_raw = [
        {"name": "search_orders", "description": "Look up orders.", "input_schema": {}},
        {"name": "issue_refund", "description": "Issue a refund.", "input_schema": {}},
    ]
    ref = fingerprint_tools(tools_raw)
    _write_toolset(tmp_path, ref, tools_raw)

    tools = load_toolset(ref, base=tmp_path)

    assert [t.name for t in tools] == ["search_orders", "issue_refund"]
    assert all(isinstance(t, ToolSpec) for t in tools)


def test_load_toolset_resolves_the_empty_toolset_without_error(tmp_path: Path) -> None:
    """The empty toolset is a first-class value -- unlike ``load_tools``, this must not raise."""
    ref = fingerprint_tools([])
    _write_toolset(tmp_path, ref, [])

    assert load_toolset(ref, base=tmp_path) == []


def test_load_toolset_accepts_a_tool_with_an_empty_description(tmp_path: Path) -> None:
    """A recorded tool with no description fingerprints fine in the SDK (V7 reconciliation)."""
    tools_raw = [{"name": "search_orders", "description": "", "input_schema": {}}]
    ref = fingerprint_tools(tools_raw)
    _write_toolset(tmp_path, ref, tools_raw)

    tools = load_toolset(ref, base=tmp_path)

    assert tools == [ToolSpec(name="search_orders", description="", input_schema={})]


def test_load_toolset_round_trips_a_strict_tool_back_to_its_own_ref(tmp_path: Path) -> None:
    """``strict`` must survive ``from_dict`` -> ``to_anthropic`` unchanged.

    The orchestrator content-addresses a resolved toolset by hashing
    ``[t.to_anthropic() for t in tools]``; if ``to_anthropic`` dropped
    ``strict`` the recomputed fingerprint would stop matching the sidecar's
    own ref and every strict toolset would silently miss the cache.
    """
    tools_raw = [
        {"name": "issue_refund", "description": "Refund.", "input_schema": {}, "strict": True},
        {"name": "search_orders", "description": "Look up.", "input_schema": {}},
    ]
    ref = fingerprint_tools(tools_raw)
    _write_toolset(tmp_path, ref, tools_raw)

    tools = load_toolset(ref, base=tmp_path)

    assert [t.strict for t in tools] == [True, False]
    assert fingerprint_tools([t.to_anthropic() for t in tools]) == ref


def test_load_toolset_accepts_a_bare_list_without_the_tools_wrapper(tmp_path: Path) -> None:
    tools_raw = [{"name": "a", "description": "d", "input_schema": {}}]
    ref = fingerprint_tools(tools_raw)
    path = toolset_path(ref, base=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tools_raw), encoding="utf-8")

    tools = load_toolset(ref, base=tmp_path)

    assert [t.name for t in tools] == ["a"]


def test_load_toolset_raises_when_sidecar_is_missing(tmp_path: Path) -> None:
    with pytest.raises(CaptureError) as exc:
        load_toolset(_TOOLSET_REF, base=tmp_path)
    assert exc.value.kind == "missing"
    assert exc.value.path == toolset_path(_TOOLSET_REF, base=tmp_path)


def test_load_toolset_raises_on_invalid_json(tmp_path: Path) -> None:
    path = toolset_path(_TOOLSET_REF, base=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(CaptureError) as exc:
        load_toolset(_TOOLSET_REF, base=tmp_path)
    assert exc.value.kind == "json_parse"


def test_load_toolset_raises_on_wrong_top_level_shape(tmp_path: Path) -> None:
    path = toolset_path(_TOOLSET_REF, base=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tools": "not a list"}), encoding="utf-8")

    with pytest.raises(CaptureError) as exc:
        load_toolset(_TOOLSET_REF, base=tmp_path)
    assert exc.value.kind == "schema"


def test_load_toolset_raises_naming_the_malformed_entry(tmp_path: Path) -> None:
    _write_toolset(tmp_path, _TOOLSET_REF, [{"description": "no name field"}])

    with pytest.raises(CaptureError) as exc:
        load_toolset(_TOOLSET_REF, base=tmp_path)
    assert exc.value.kind == "schema"
    assert "tools[0]" in exc.value.summary


def test_load_toolset_caches_by_ref(tmp_path: Path) -> None:
    """A second resolution with the same cache must not re-read a since-changed sidecar.

    The rewrite below no longer matches ``ref`` (V9's fingerprint check --
    see ``TestLoadToolsetVerifiesFingerprint``): if the cache didn't
    short-circuit and this were re-read, it would now raise
    ``CaptureError``, not just return stale content. That makes this an even
    stronger proof that verification never re-checks a cache hit.
    """
    tools_v1 = [{"name": "a", "description": "d", "input_schema": {}}]
    ref = fingerprint_tools(tools_v1)
    _write_toolset(tmp_path, ref, tools_v1)
    cache: dict[str, list[ToolSpec]] = {}

    first = load_toolset(ref, base=tmp_path, cache=cache)
    _write_toolset(tmp_path, ref, [{"name": "b", "description": "d", "input_schema": {}}])
    second = load_toolset(ref, base=tmp_path, cache=cache)

    assert first == second == [ToolSpec(name="a", description="d", input_schema={})]


def test_load_toolset_cache_reads_the_sidecar_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert the read count directly, not just cache-hit behaviour."""
    tools_raw = [{"name": "a", "description": "d", "input_schema": {}}]
    ref = fingerprint_tools(tools_raw)
    _write_toolset(tmp_path, ref, tools_raw)
    cache: dict[str, list[ToolSpec]] = {}
    reads = 0
    real_read_text = Path.read_text

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal reads
        if self == toolset_path(ref, base=tmp_path):
            reads += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    for _ in range(3):
        load_toolset(ref, base=tmp_path, cache=cache)

    assert reads == 1


def test_load_toolset_without_a_cache_re_reads_every_call(tmp_path: Path) -> None:
    """Caching is opt-in: no ``cache`` means every call resolves fresh from disk.

    Pre-V9 this proved it by rewriting the SAME ref's sidecar with different
    content between two reads and observing the change. V9's fingerprint
    check makes that technique nonsensical -- new content at an unchanged
    ref is now exactly the tampering scenario that must be REJECTED, not
    observed -- so this proves "no hidden memoization" a different way:
    deleting the sidecar between calls must surface as a fresh ``missing``
    error, never a stale cached result.
    """
    tools_raw = [{"name": "a", "description": "d", "input_schema": {}}]
    ref = fingerprint_tools(tools_raw)
    _write_toolset(tmp_path, ref, tools_raw)

    first = load_toolset(ref, base=tmp_path)
    assert [t.name for t in first] == ["a"]

    toolset_path(ref, base=tmp_path).unlink()
    with pytest.raises(CaptureError) as exc:
        load_toolset(ref, base=tmp_path)
    assert exc.value.kind == "missing"


# ---------------------------------------------------------------------------
# load_toolset — fingerprint verification (hardening pass, Fix 2): content-
# addressing is only an integrity guarantee if the content is checked
# against the address that named it. Before this, `load_toolset` trusted the
# ref/filename correspondence as a pure naming convention -- editing a
# committed sidecar's `tools` array by hand (e.g.
# examples/agent/toolsets/*.json, a committed, hand-editable file) silently
# produced a toolset whose ref no longer matched what it actually contained,
# with nothing to notice. That's the original per-call-toolset-capture bug
# again, triggered by an ordinary edit instead of anything exotic.
# ---------------------------------------------------------------------------


class TestLoadToolsetVerifiesFingerprint:
    def test_tampered_sidecar_is_rejected(self, tmp_path: Path) -> None:
        """The whole point: edit a sidecar's contents, keep its filename
        (hence its ref) unchanged -- exactly what a hand-edit of a committed
        example sidecar does -- and this must now be caught."""
        tools = [{"name": "search_orders", "description": "Look up orders.", "input_schema": {}}]
        ref = fingerprint_tools(tools)
        path = _write_toolset(tmp_path, ref, tools)

        # Tamper: add a tool after the fact without renaming the file / updating ref --
        # this is what hand-editing examples/agent/toolsets/*.json looks like.
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["tools"].append(
            {"name": "issue_refund", "description": "Issue a refund.", "input_schema": {}}
        )
        path.write_text(json.dumps(tampered), encoding="utf-8")

        with pytest.raises(CaptureError) as exc:
            load_toolset(ref, base=tmp_path)
        assert exc.value.kind == "fingerprint_mismatch"

    def test_error_names_the_file_the_ref_and_the_actual_fingerprint(self, tmp_path: Path) -> None:
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        real_fingerprint = fingerprint_tools(tools)
        wrong_ref = "sha256:" + "00" * 32
        path = _write_toolset(tmp_path, wrong_ref, tools)

        with pytest.raises(CaptureError) as exc:
            load_toolset(wrong_ref, base=tmp_path)

        message = str(exc.value)
        assert str(path) in message
        assert wrong_ref in message
        assert real_fingerprint in message

    def test_matching_sidecar_resolves_cleanly(self, tmp_path: Path) -> None:
        """Positive control: a genuinely self-consistent sidecar is unaffected."""
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        ref = fingerprint_tools(tools)
        _write_toolset(tmp_path, ref, tools)

        resolved = load_toolset(ref, base=tmp_path)
        assert [t.name for t in resolved] == ["a"]

    def test_verification_applies_to_the_sdk_shape(self, tmp_path: Path) -> None:
        """The SDK's ``ToolsetSink`` shape: ``{schema_version, fingerprint, tools}``."""
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        ref = fingerprint_tools(tools)
        path = toolset_path(ref, base=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": "1.0.0", "fingerprint": ref, "tools": tools}),
            encoding="utf-8",
        )

        assert [t.name for t in load_toolset(ref, base=tmp_path)] == ["a"]

    def test_verification_applies_to_the_bare_tools_only_shape(self, tmp_path: Path) -> None:
        """The checked-in example sidecars' shape: ``{"tools": [...]}`` only --
        no ``schema_version``, no embedded ``fingerprint`` field."""
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        ref = fingerprint_tools(tools)
        path = toolset_path(ref, base=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tools": tools}), encoding="utf-8")

        assert [t.name for t in load_toolset(ref, base=tmp_path)] == ["a"]

    def test_verification_applies_to_the_bare_list_shape(self, tmp_path: Path) -> None:
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        ref = fingerprint_tools(tools)
        path = toolset_path(ref, base=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tools), encoding="utf-8")

        assert [t.name for t in load_toolset(ref, base=tmp_path)] == ["a"]

    def test_embedded_fingerprint_field_disagreeing_with_ref_is_not_consulted(
        self, tmp_path: Path
    ) -> None:
        """The deliberate three-way-disagreement decision (see the hardening
        report): only ``ref`` -- what every other part of the system already
        trusts as this toolset's identity (cache keys, refcounting, suite
        examples' own ``toolset_ref`` field) -- is checked against the
        recomputed fingerprint. A sidecar's own embedded ``fingerprint``
        field, when present (the SDK shape only), is never read for
        verification. So a sidecar whose ``tools`` genuinely matches ``ref``
        resolves cleanly even when its OWN embedded field is stale or wrong.
        """
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        ref = fingerprint_tools(tools)  # what the filename/ref says
        stale_embedded = "sha256:" + "11" * 32  # agrees with neither ref nor tools
        path = toolset_path(ref, base=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": "1.0.0", "fingerprint": stale_embedded, "tools": tools}),
            encoding="utf-8",
        )

        resolved = load_toolset(ref, base=tmp_path)
        assert [t.name for t in resolved] == ["a"]

    def test_embedded_fingerprint_field_agreeing_with_stale_ref_is_still_rejected(
        self, tmp_path: Path
    ) -> None:
        """The other half of the same decision: an embedded field cannot
        rescue a sidecar whose ``tools`` doesn't match ``ref``, even if that
        embedded field happens to (incorrectly) agree with the content --
        ``ref`` alone is authoritative."""
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        real_fingerprint = fingerprint_tools(tools)
        wrong_ref = "sha256:" + "22" * 32
        path = toolset_path(wrong_ref, base=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"schema_version": "1.0.0", "fingerprint": real_fingerprint, "tools": tools}
            ),
            encoding="utf-8",
        )

        with pytest.raises(CaptureError) as exc:
            load_toolset(wrong_ref, base=tmp_path)
        assert exc.value.kind == "fingerprint_mismatch"

    def test_a_raw_entry_fingerprint_tools_cannot_hash_degrades_to_a_clean_capture_error(
        self, tmp_path: Path
    ) -> None:
        """An OpenAI/function-shape raw entry parses fine via ``ToolSpec.from_dict``
        (which accepts both provider shapes) but has no top-level ``"name"`` key for
        ``fingerprint_tools`` to sort by -- a sidecar is documented to hold only
        pre-normalised {name, description, input_schema} tools, so this isn't a
        real, supported sidecar shape, but ``load_toolset`` must still degrade to a
        clean ``CaptureError`` here rather than let a bare ``KeyError`` escape its
        documented "only raises CaptureError" contract."""
        openai_shape_tools = [
            {
                "type": "function",
                "function": {
                    "name": "a",
                    "description": "d",
                    "parameters": {},
                },
            },
        ]
        path = toolset_path(_TOOLSET_REF, base=tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tools": openai_shape_tools}), encoding="utf-8")

        with pytest.raises(CaptureError) as exc:
            load_toolset(_TOOLSET_REF, base=tmp_path)
        assert exc.value.kind == "schema"
        assert "fingerprint" in exc.value.summary

    def test_verification_does_not_re_read_or_re_verify_a_cached_ref(self, tmp_path: Path) -> None:
        """The per-run resolution cache must not be defeated by verification:
        once a ref is cached, later calls for it never touch disk again, so
        a sidecar that's tampered with after the first resolution has no
        effect on later calls for the same ref."""
        tools = [{"name": "a", "description": "d", "input_schema": {}}]
        ref = fingerprint_tools(tools)
        path = _write_toolset(tmp_path, ref, tools)
        cache: dict[str, list[ToolSpec]] = {}
        load_toolset(ref, base=tmp_path, cache=cache)

        # Tamper with the sidecar post-cache: if this got re-read and
        # re-verified, it would now raise.
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["tools"].append({"name": "b", "description": "d", "input_schema": {}})
        path.write_text(json.dumps(tampered), encoding="utf-8")

        resolved = load_toolset(ref, base=tmp_path, cache=cache)
        assert [t.name for t in resolved] == ["a"]


class TestLoadToolsetVerificationDoesNotFalselyReject:
    """The reviewer-flagged real risk: legitimately odd-but-valid schemas
    must verify cleanly, never caught as false positives by the new check.
    """

    def test_tool_with_no_description_key_at_all(self, tmp_path: Path) -> None:
        """Not ``description=""`` -- the key is absent from the raw JSON
        entirely. The fingerprint must be recomputed over the raw dict as
        parsed, never over a round-tripped :class:`ToolSpec` (whose
        ``description`` defaults to ``""`` and would re-canonicalise this
        tool differently than the sidecar's own bytes do)."""
        tools = [{"name": "a", "input_schema": {}}]
        ref = fingerprint_tools(tools)
        _write_toolset(tmp_path, ref, tools)

        resolved = load_toolset(ref, base=tmp_path)
        assert resolved == [ToolSpec(name="a", description="", input_schema={})]

    def test_non_ascii_text(self, tmp_path: Path) -> None:
        tools = [
            {
                "name": "検索",
                "description": "顧客の注文履歴を検索する — søk etter ordre 🔍",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        ref = fingerprint_tools(tools)
        _write_toolset(tmp_path, ref, tools)

        resolved = load_toolset(ref, base=tmp_path)
        assert resolved[0].name == "検索"

    def test_deeply_nested_input_schema(self, tmp_path: Path) -> None:
        tools = [
            {
                "name": "a",
                "description": "d",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "and": {
                                        "type": "array",
                                        "items": {"type": "object", "properties": {}},
                                    },
                                },
                            },
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        ]
        ref = fingerprint_tools(tools)
        _write_toolset(tmp_path, ref, tools)

        resolved = load_toolset(ref, base=tmp_path)
        assert resolved[0].input_schema["properties"]["filters"]["type"] == "array"

    def test_unusual_but_valid_input_schema_shapes(self, tmp_path: Path) -> None:
        """``oneOf``/``enum``/empty-schema flavours -- unusual, still valid JSON Schema."""
        tools = [
            {
                "name": "a",
                "description": "d",
                "input_schema": {
                    "oneOf": [
                        {"type": "string", "enum": ["x", "y"]},
                        {"type": "null"},
                    ],
                },
            },
            {"name": "b", "description": "", "input_schema": {}},
        ]
        ref = fingerprint_tools(tools)
        _write_toolset(tmp_path, ref, tools)

        resolved = load_toolset(ref, base=tmp_path)
        assert [t.name for t in resolved] == ["a", "b"]

    def test_the_real_committed_agent_example_sidecar_verifies(self) -> None:
        """Regression pin for the exact sidecar this hardening pass must not
        break: the committed, hand-editable ``examples/agent/toolsets/*.json``
        -- resolved for real, off the actual checked-in file, base included.

        A direct path from this test file's own location, never a broad
        rglob, so this can't wander into ``.claude/worktrees/`` (a stale full
        copy of the repo) -- same guard ``test_suite_loader.py``'s
        ``TestLoadJsonlCheckedInExamples`` and ``test_orchestrator.py``'s
        ``TestOrchestratorShippedExamples`` use for the same reason.
        """
        agent_dir = Path(__file__).resolve().parents[2] / "examples" / "agent"
        ref = "sha256:ad3f07832877239336d594c7c1626197b39b9eb4c178842304c8754e94323d67"

        resolved = load_toolset(ref, base=agent_dir)

        assert len(resolved) == 6
        assert "notify_security_team" in [t.name for t in resolved]


# ---------------------------------------------------------------------------
# capture_toolset_refs / promoted_toolset_refs — refcounting for the
# orphan-sidecar sweep in ``capture clean`` (V5)
# ---------------------------------------------------------------------------

_REF_A = "sha256:" + "aa" * 32
_REF_B = "sha256:" + "bb" * 32


def _write_promoted_case(
    base: Path,
    *,
    suite: str,
    name: str,
    toolset_ref: str | None = _REF_A,
    tools: list[ToolSpec] | None = None,
) -> Path:
    """Write a promoted-case file directly under ``<base>/suites/<suite>/``.

    Bypasses ``write_promoted_case``/promotion entirely -- this file only
    needs a valid ``PromotedCase`` on disk for the reader to walk, not a
    real capture behind it.
    """
    example = SuiteExample(
        id=name,
        inputs={},
        toolset_ref=toolset_ref,
        tools=tools if toolset_ref is None else None,
    )
    case = PromotedCase(
        name=name,
        suite=suite,
        from_capture=f"cap_{name}",
        promoted_at="2026-06-16T12:00:00+00:00",
        source_input_hash="hash",
        code_version="v1",
        example=example,
    )
    suite_dir = base / "suites" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / f"{name}.json"
    path.write_text(case.model_dump_json(), encoding="utf-8")
    return path


class TestCaptureToolsetRefs:
    """Every ``toolset_ref`` a readable capture under ``<base>/captures/`` still uses."""

    def test_empty_when_no_captures_dir(self, tmp_path: Path) -> None:
        assert capture_toolset_refs(base=tmp_path) == set()

    def test_collects_the_ref_from_a_capture(self, tmp_path: Path) -> None:
        _write_capture(
            tmp_path,
            _capture_dict(capture_id="cap_1", events=[_model_call(0, toolset_ref=_REF_A)]),
        )
        assert capture_toolset_refs(base=tmp_path) == {_REF_A}

    def test_spans_every_suite_unfiltered(self, tmp_path: Path) -> None:
        """The toolsets namespace is global, so refcounting must not stop at one suite."""
        _write_capture(
            tmp_path,
            _capture_dict(
                capture_id="cap_1", suite="suite_a", events=[_model_call(0, toolset_ref=_REF_A)]
            ),
        )
        _write_capture(
            tmp_path,
            _capture_dict(
                capture_id="cap_2", suite="suite_b", events=[_model_call(0, toolset_ref=_REF_B)]
            ),
        )
        assert capture_toolset_refs(base=tmp_path) == {_REF_A, _REF_B}

    def test_capture_with_no_toolset_ref_contributes_nothing(self, tmp_path: Path) -> None:
        """A capture predating per-call toolset capture has no ref to protect."""
        _write_capture(
            tmp_path,
            _capture_dict(capture_id="cap_1", events=[_model_call(0, toolset_ref=None)]),
        )
        assert capture_toolset_refs(base=tmp_path) == set()

    def test_collects_refs_from_every_model_call_not_just_the_first(self, tmp_path: Path) -> None:
        """C1: the SDK stamps toolset_ref per call, so a capture that switched
        toolsets mid-run -- the exact workload this feature was built for --
        legitimately carries more than one distinct ref across its model_call
        events. Every one of them is a live reference the sweep must protect,
        not just the first round's.
        """
        _write_capture(
            tmp_path,
            _capture_dict(
                capture_id="cap_1",
                events=[
                    _model_call(0, toolset_ref=_REF_A),
                    _tool_call("search", 1),
                    _model_call(2, toolset_ref=_REF_B),
                    _tool_call("archive", 3),
                ],
            ),
        )
        assert capture_toolset_refs(base=tmp_path) == {_REF_A, _REF_B}

    def test_unreadable_capture_fails_closed(self, tmp_path: Path) -> None:
        """C2: a refcount preceding a delete must fail closed. Reading "cannot
        parse" as "references nothing" is what let one corrupt capture file
        sweep every sidecar its readable siblings still used.
        """
        suite_dir = tmp_path / "captures" / "s"
        suite_dir.mkdir(parents=True)
        (suite_dir / "broken.json").write_text("not json", encoding="utf-8")
        with pytest.raises(CaptureError) as exc:
            capture_toolset_refs(base=tmp_path)
        assert exc.value.path.name == "broken.json"

    def test_unresolvable_schema_version_also_fails_closed(self, tmp_path: Path) -> None:
        """The exact scenario the review verified: a capture stamped with a
        schema_version this CLI can't read must abort the count too, not just
        a JSON-parse failure -- both are "cannot parse", handled identically.
        """
        _write_capture(tmp_path, _capture_dict(capture_id="cap_future", schema_version="3.0.0"))
        with pytest.raises(CaptureError):
            capture_toolset_refs(base=tmp_path)


class TestPromotedToolsetRefs:
    """Every ``toolset_ref`` a promoted suite example under ``<base>/suites/`` still uses."""

    def test_empty_when_no_suites_dir(self, tmp_path: Path) -> None:
        assert promoted_toolset_refs(base=tmp_path) == set()

    def test_collects_the_ref_from_a_promoted_case(self, tmp_path: Path) -> None:
        _write_promoted_case(tmp_path, suite="s", name="case1", toolset_ref=_REF_A)
        assert promoted_toolset_refs(base=tmp_path) == {_REF_A}

    def test_spans_every_suite(self, tmp_path: Path) -> None:
        _write_promoted_case(tmp_path, suite="suite_a", name="case1", toolset_ref=_REF_A)
        _write_promoted_case(tmp_path, suite="suite_b", name="case2", toolset_ref=_REF_B)
        assert promoted_toolset_refs(base=tmp_path) == {_REF_A, _REF_B}

    def test_inline_tools_example_contributes_nothing(self, tmp_path: Path) -> None:
        """A hand-authored case inlining ``tools`` has no sidecar ref to protect."""
        tool = ToolSpec(name="search", description="d", input_schema={})
        _write_promoted_case(tmp_path, suite="s", name="case1", toolset_ref=None, tools=[tool])
        assert promoted_toolset_refs(base=tmp_path) == set()

    def test_golden_jsonl_itself_is_not_double_counted_or_misread(self, tmp_path: Path) -> None:
        """``golden.jsonl`` sits beside the case files but must not be parsed as one."""
        _write_promoted_case(tmp_path, suite="s", name="case1", toolset_ref=_REF_A)
        (tmp_path / "suites" / "s" / "golden.jsonl").write_text("not a PromotedCase\n")
        assert promoted_toolset_refs(base=tmp_path) == {_REF_A}

    def test_unreadable_case_file_fails_closed(self, tmp_path: Path) -> None:
        """C2: same fail-closed contract as capture_toolset_refs (M2 -- both
        share the reader's private promoted-case walk)."""
        suite_dir = tmp_path / "suites" / "s"
        suite_dir.mkdir(parents=True)
        (suite_dir / "broken.json").write_text("not json", encoding="utf-8")
        with pytest.raises(CaptureError) as exc:
            promoted_toolset_refs(base=tmp_path)
        assert exc.value.path.name == "broken.json"


class TestPromotedCaptureIds:
    """Every ``from_capture`` a promoted suite example under ``<base>/suites/`` names.

    Shares its walk of ``<base>/suites/`` with ``promoted_toolset_refs`` (M2) --
    covered here only for what that sharing changed: the fail-closed contract
    (C2). ``promoted_capture_ids`` itself is otherwise exercised indirectly via
    ``capture list``/``capture clean`` in ``test_cli_capture.py``.
    """

    def test_collects_from_capture_of_every_promoted_case(self, tmp_path: Path) -> None:
        _write_promoted_case(tmp_path, suite="s", name="case1", toolset_ref=_REF_A)
        assert promoted_capture_ids(base=tmp_path) == {"cap_case1"}

    def test_unreadable_case_file_fails_closed(self, tmp_path: Path) -> None:
        """C2/M2: must fail closed identically to promoted_toolset_refs -- both
        are the shared generator's two callers."""
        suite_dir = tmp_path / "suites" / "s"
        suite_dir.mkdir(parents=True)
        (suite_dir / "broken.json").write_text("not json", encoding="utf-8")
        with pytest.raises(CaptureError) as exc:
            promoted_capture_ids(base=tmp_path)
        assert exc.value.path.name == "broken.json"


#: The folder evalshift-sdk >= 0.7.0 writes for conversation_id "c1" (pinned in
#: evalshift-sdk tests/test_conversation_layout.py too).
CONV = "conv_d0f631ca1ddba8db"


def _write_nested(base: Path, payload: dict[str, Any], folder: str = CONV) -> Path:
    conv_dir = base / "captures" / payload["suite"] / folder
    conv_dir.mkdir(parents=True, exist_ok=True)
    path = conv_dir / f"{payload['capture_id']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_iter_captures_reads_conversation_folders_beside_flat_captures(tmp_path: Path) -> None:
    _write_capture(tmp_path, _capture_dict(capture_id="cap_flat", suite="alpha"))
    nested = _write_nested(tmp_path, _capture_dict(capture_id="cap_turn", suite="alpha"))

    records = iter_captures(suite="alpha", base=tmp_path)

    assert {r.envelope.capture_id: r.path for r in records} == {
        "cap_flat": tmp_path / "captures" / "alpha" / "cap_flat.json",
        "cap_turn": nested,
    }
    assert [r.envelope.capture_id for r in iter_captures(base=tmp_path)] == ["cap_flat", "cap_turn"]


@pytest.mark.parametrize(
    "folder",
    ["nested", "conv_D0F631CA1DDBA8DB", "conv_d0f631ca", "conv_d0f631ca1ddba8db0"],
)
def test_iter_captures_ignores_folders_the_sdk_never_writes(tmp_path: Path, folder: str) -> None:
    _write_nested(tmp_path, _capture_dict(capture_id="cap_x", suite="alpha"), folder=folder)
    assert iter_captures(base=tmp_path) == []


def test_find_capture_finds_a_turn_in_a_conversation_folder(tmp_path: Path) -> None:
    path = _write_nested(tmp_path, _capture_dict(capture_id="cap_turn", suite="alpha"))
    assert find_capture("cap_turn", base=tmp_path).path == path


def test_a_suite_name_reads_the_directory_the_sdk_wrote_it_to(tmp_path: Path) -> None:
    suite_dir = tmp_path / "captures" / "team_support"
    suite_dir.mkdir(parents=True)
    payload = _capture_dict(capture_id="cap_1", suite="team/support")
    (suite_dir / "cap_1.json").write_text(json.dumps(payload), encoding="utf-8")

    for name in ("team/support", "team_support"):
        assert [r.envelope.capture_id for r in iter_captures(suite=name, base=tmp_path)] == [
            "cap_1"
        ]


def test_a_suite_argument_cannot_address_a_conversation_folder(tmp_path: Path) -> None:
    _write_nested(tmp_path, _capture_dict(capture_id="cap_turn", suite="alpha"))
    assert iter_captures(suite=f"alpha/{CONV}", base=tmp_path) == []


@pytest.mark.parametrize(
    ("suite", "dirname"),
    [
        ("alpha", "alpha"),
        ("team/support", "team_support"),
        ("a\\b", "a_b"),
        ("../x", "__x"),
        (" .hidden. ", "hidden"),
        ("..", "_"),
        ("", "_"),
    ],
)
def test_capture_suite_dirname_matches_the_sdks_safe_segment(suite: str, dirname: str) -> None:
    assert capture_suite_dirname(suite) == dirname
