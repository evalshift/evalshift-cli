"""Unit tests for :mod:`evalshift_cli.captures.models`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evalshift_cli.captures.models import CaptureEnvelope, PromotedCase
from evalshift_cli.suite.models import SuiteExample


def _trace_payload() -> dict[str, Any]:
    return {
        "run_id": "r1",
        "prompt_id": "p1",
        "example_id": "cap_abc",
        "role": "source",
        "events": [
            {
                "type": "final_output",
                "sequence_index": 0,
                "timestamp": "2026-06-16T12:00:00+00:00",
                "metadata": {},
                "text": "hello",
            },
        ],
    }


def _envelope_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "2.0.0",
        "capture_id": "cap_abc",
        "suite": "support_agent",
        "input_hash": "hash123",
        "code_version": "v1",
        "created_at": "2026-06-16T12:00:00+00:00",
        "trace": _trace_payload(),
    }
    payload.update(overrides)
    return payload


class TestCaptureEnvelope:
    def test_parses_with_new_provenance_fields(self) -> None:
        env = CaptureEnvelope.model_validate(
            _envelope_payload(
                conversation_id="conv_1",
                turn_index=2,
                parent_capture_id="cap_parent",
            ),
        )
        assert env.conversation_id == "conv_1"
        assert env.turn_index == 2
        assert env.parent_capture_id == "cap_parent"

    def test_parses_without_new_provenance_fields(self) -> None:
        """Old SDK captures lack these keys entirely."""
        env = CaptureEnvelope.model_validate(_envelope_payload())
        assert env.conversation_id is None
        assert env.turn_index is None
        assert env.parent_capture_id is None

    def test_unknown_keys_ignored(self) -> None:
        env = CaptureEnvelope.model_validate(_envelope_payload(future_field="x"))
        assert env.capture_id == "cap_abc"


class TestPromotedCase:
    def _case_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": "case_1",
            "suite": "support_agent",
            "from_capture": "cap_abc",
            "promoted_at": "2026-06-16T12:00:00+00:00",
            "source_input_hash": "hash123",
            "code_version": "v1",
            "example": {"id": "ex1", "inputs": {"q": "hi"}, "tools": []},
        }
        payload.update(overrides)
        return payload

    def test_parses_with_provenance(self) -> None:
        case = PromotedCase.model_validate(
            self._case_payload(conversation_id="conv_1", turn_index=3),
        )
        assert case.conversation_id == "conv_1"
        assert case.turn_index == 3
        assert isinstance(case.example, SuiteExample)

    def test_parses_without_provenance(self) -> None:
        """Existing promoted-case JSON files lack these keys."""
        case = PromotedCase.model_validate(self._case_payload())
        assert case.conversation_id is None
        assert case.turn_index is None

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PromotedCase.model_validate(self._case_payload(rogue_key=True))

    def test_promotion_source_defaults_to_executed(self) -> None:
        """Every promoted-case file written before the field is executed-sourced."""
        case = PromotedCase.model_validate(self._case_payload())
        assert case.promotion_source == "executed"

    def test_promotion_source_round_trips(self) -> None:
        case = PromotedCase.model_validate(self._case_payload(promotion_source="requested"))
        assert case.promotion_source == "requested"

    def test_promotion_source_rejects_an_unknown_yardstick(self) -> None:
        with pytest.raises(ValidationError, match="promotion_source"):
            PromotedCase.model_validate(self._case_payload(promotion_source="guessed"))


def test_a_case_written_before_capture_provenance_still_parses() -> None:
    case = PromotedCase.model_validate(
        {"name": "c", "suite": "s", "from_capture": "cap_1", "example": {"id": "c", "tools": []}}
    )
    assert case.captured_at == ""
    assert case.source_model == ""
