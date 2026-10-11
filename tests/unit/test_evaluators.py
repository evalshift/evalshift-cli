"""Tests for the evaluators package."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from evalshift_cli.cache.store import CacheStore
from evalshift_cli.config.models import (
    ToolArgumentsEvaluatorConfig,
    ToolSelectionEvaluatorConfig,
    ToolTraceStructureEvaluatorConfig,
)
from evalshift_cli.evaluators import semantic as semantic_module
from evalshift_cli.evaluators.base import (
    EvalRecord,
    Evaluator,
    EvaluatorError,
    PairedScore,
)
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION
from evalshift_cli.evaluators.llm_judge import (
    MAX_TRANSCRIPT_CHARS,
    PairwiseJudgeEvaluator,
    _format_transcript,
    _parse_verdict,
)
from evalshift_cli.evaluators.semantic import CosineSimilarityEvaluator, _cosine
from evalshift_cli.evaluators.structural import (
    JsonSchemaEvaluator,
    LengthEvaluator,
    RegexEvaluator,
)
from evalshift_cli.evaluators.tool_arguments import ToolArgumentsEvaluator
from evalshift_cli.evaluators.tool_models import ToolCall, ToolTrace
from evalshift_cli.evaluators.tool_selection import ToolSelectionEvaluator
from evalshift_cli.evaluators.tool_trace_structure import ToolTraceStructureEvaluator
from evalshift_cli.models.client import CompletionResult, ModelClient
from evalshift_cli.suite.models import ExpectedToolCall
from tests.scoring_fixtures import (
    PROMPT_ID,
    RUN_ID,
    TOOL_ARGUMENTS_NAME,
    empty_output_pair,
    load_pairs,
    records_of,
)
from tests.unit.suite_examples import suite_example

# In-memory SQLite keeps the cache round-trip tests fast and hermetic.
IN_MEMORY_DB = "sqlite+aiosqlite:///:memory:"

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class TestPairedScore:
    def test_delta_is_target_minus_source(self) -> None:
        ps = PairedScore(source_score=0.2, target_score=0.9)
        assert ps.delta == pytest.approx(0.7)

    def test_score_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PairedScore(source_score=1.5, target_score=0.5)


class TestEvalRecord:
    def test_round_trip(self) -> None:
        rec = EvalRecord(
            run_id="r1",
            prompt_id="p",
            example_id="e",
            evaluator_name="x",
            source_score=0.8,
            target_score=0.6,
            delta=-0.2,
        )
        assert EvalRecord.model_validate_json(rec.model_dump_json()) == rec


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


class TestJsonSchemaEvaluator:
    def _schema(self, tmp_path: Path) -> Path:
        schema = {
            "type": "object",
            "required": ["greeting"],
            "properties": {"greeting": {"type": "string"}},
        }
        path = tmp_path / "schema.json"
        path.write_text(json.dumps(schema), encoding="utf-8")
        return path

    async def test_valid_outputs(self, tmp_path: Path) -> None:
        ev = JsonSchemaEvaluator(schema_path=self._schema(tmp_path))
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='{"greeting": "hello"}',
            target_output='{"greeting": "hi"}',
        )
        assert score.source_score == 1.0
        assert score.target_score == 1.0

    async def test_invalid_target_only(self, tmp_path: Path) -> None:
        ev = JsonSchemaEvaluator(schema_path=self._schema(tmp_path))
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='{"greeting": "hello"}',
            target_output='{"missing_field": true}',
        )
        assert score.source_score == 1.0
        assert score.target_score == 0.0
        assert score.delta == pytest.approx(-1.0)

    async def test_unparseable_json_scores_zero(self, tmp_path: Path) -> None:
        ev = JsonSchemaEvaluator(schema_path=self._schema(tmp_path))
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="not json",
            target_output='{"greeting": "ok"}',
        )
        assert score.source_score == 0.0
        assert score.target_score == 1.0

    def test_missing_schema_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluatorError):
            JsonSchemaEvaluator(schema_path=tmp_path / "nope.json")

    async def test_accepts_history_kwarg_without_behavior_change(self, tmp_path: Path) -> None:
        ev = JsonSchemaEvaluator(schema_path=self._schema(tmp_path))
        history = [{"role": "user", "content": "hi"}]
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='{"greeting": "hello"}',
            target_output='{"greeting": "hi"}',
            history=history,
        )
        assert score.source_score == 1.0
        assert score.target_score == 1.0


class TestRegexEvaluator:
    async def test_match_vs_no_match(self) -> None:
        ev = RegexEvaluator(pattern=r"^summary:")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="summary: ok",
            target_output="random text",
        )
        assert score.source_score == 1.0
        assert score.target_score == 0.0

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(EvaluatorError):
            RegexEvaluator(pattern="(unclosed")

    async def test_accepts_history_kwarg_without_behavior_change(self) -> None:
        ev = RegexEvaluator(pattern=r"^summary:")
        history = [{"role": "user", "content": "hi"}]
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="summary: ok",
            target_output="random text",
            history=history,
        )
        assert score.source_score == 1.0
        assert score.target_score == 0.0


class TestLengthEvaluator:
    async def test_within_bounds(self) -> None:
        ev = LengthEvaluator(min_chars=5, max_chars=20)
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="hello world",
            target_output="ok",  # too short
        )
        assert score.source_score == 1.0
        assert 0.0 <= score.target_score < 1.0

    async def test_too_long_decays(self) -> None:
        ev = LengthEvaluator(max_chars=10)
        # 30 chars is way past the boundary; decay drives score toward 0.
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="x" * 5,
            target_output="x" * 30,
        )
        assert score.source_score == 1.0
        assert score.target_score < 0.5

    def test_no_bounds_raises(self) -> None:
        with pytest.raises(EvaluatorError):
            LengthEvaluator()

    async def test_accepts_history_kwarg_without_behavior_change(self) -> None:
        ev = LengthEvaluator(min_chars=5, max_chars=20)
        history = [{"role": "user", "content": "hi"}]
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="hello world",
            target_output="ok",
            history=history,
        )
        assert score.source_score == 1.0
        assert 0.0 <= score.target_score < 1.0


# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------


class TestCosineHelper:
    def test_identical_vectors(self) -> None:
        assert _cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _cosine([1.0], [1.0, 2.0]) == 0.0


class TestCosineEvaluator:
    async def test_high_similarity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock embeddings: identical → similarity 1.
        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            return type(
                "R",
                (),
                {
                    "data": [{"embedding": [1.0, 0.0, 0.0]}],
                },
            )()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="a",
            target_output="b",
        )
        assert score.source_score == 1.0
        assert score.target_score == pytest.approx(1.0)
        assert score.delta == pytest.approx(0.0)

    async def test_min_similarity_threshold_controls_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Near-identical outputs: cosine ~0.98, below 1.0 but above 0.9.
        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            vec = [1.0, 0.0] if input[0] == "src" else [0.98, 0.198997]
            return type("R", (), {"data": [{"embedding": vec}]})()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)

        # Default 0.9 → drift this small is NOT flagged.
        lenient = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await lenient.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.target_score == pytest.approx(0.98, abs=1e-3)
        assert score.metadata["failure_categories"] == []

        # min_similarity=1.0 → any deviation IS flagged.
        strict = CosineSimilarityEvaluator(embedding_model="x/y", min_similarity=1.0)
        score = await strict.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.metadata["failure_categories"] == [SEMANTIC_REGRESSION]

    async def test_accepts_history_kwarg_without_behavior_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            return type("R", (), {"data": [{"embedding": [1.0, 0.0, 0.0]}]})()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        history = [{"role": "user", "content": "hi"}]
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="a",
            target_output="b",
            history=history,
        )
        assert score.source_score == 1.0
        assert score.target_score == pytest.approx(1.0)

    async def test_identical_outputs_are_embedded_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A target that matches the source verbatim is a guaranteed 1.0 —
        # embedding the same text twice is a wasted provider call.
        embedded: list[str] = []

        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            embedded.append(input[0])
            return type("R", (), {"data": [{"embedding": [1.0, 0.0]}]})()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="same",
            target_output="same",
        )
        assert embedded == ["same"]
        assert score.target_score == pytest.approx(1.0)

    async def test_embeddings_are_cached_across_scores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Embedding calls are pure functions of (model, text) — a second
        # score over the same outputs must not hit the provider again.
        embedded: list[str] = []

        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            embedded.append(input[0])
            vec = [1.0, 0.0] if input[0] == "src" else [0.98, 0.198997]
            return type("R", (), {"data": [{"embedding": vec}]})()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)
        cache = await CacheStore.open(IN_MEMORY_DB)
        try:
            ev = CosineSimilarityEvaluator(embedding_model="x/y", cache=cache)
            kwargs: dict[str, Any] = {
                "prompt_id": "p",
                "example_id": "e",
                "input_vars": {},
                "source_output": "src",
                "target_output": "tgt",
            }
            first = await ev.score(**kwargs)
            second = await ev.score(**kwargs)
        finally:
            await cache.close()

        assert sorted(embedded) == ["src", "tgt"]
        assert second.target_score == pytest.approx(first.target_score)

    async def test_cache_is_keyed_by_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embedded: list[tuple[str, str]] = []

        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            embedded.append((model, input[0]))
            return type("R", (), {"data": [{"embedding": [1.0, 0.0]}]})()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)
        cache = await CacheStore.open(IN_MEMORY_DB)
        kwargs: dict[str, Any] = {
            "prompt_id": "p",
            "example_id": "e",
            "input_vars": {},
            "source_output": "src",
            "target_output": "src",
        }
        try:
            await CosineSimilarityEvaluator(embedding_model="x/y", cache=cache).score(**kwargs)
            await CosineSimilarityEvaluator(embedding_model="x/z", cache=cache).score(**kwargs)
        finally:
            await cache.close()

        assert {model for model, _ in embedded} == {"x/y", "x/z"}

    async def test_reports_nothing_when_both_outputs_are_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Tool-only turn: nothing to embed, and a provider 400s on an
        # empty input. Non-applicable, not broken — so no score at all.
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the provider must not be called")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="",
            target_output="   ",
        )
        assert score is None

    async def test_scores_zero_when_target_went_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A target that went silent where the source answered IS a
        # regression — and one an embedding provider can't measure, since
        # it 400s on empty input. Similarity to nothing is 0.0 by
        # definition, so no provider call is made at all.
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the provider must not be called")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="When should I schedule that for you?",
            target_output="",
        )
        assert score is not None
        assert score.source_score == 1.0
        assert score.target_score == 0.0
        assert score.metadata["raw_cosine"] == 0.0
        assert score.metadata["empty_side"] == "target"
        assert score.metadata["failure_categories"] == [SEMANTIC_REGRESSION]
        assert "no text" in score.explanation

    async def test_scores_zero_when_source_was_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The mirror image — a target that answered where the source was
        # silent — is the same total divergence, scored the same way.
        # Whitespace-only counts as silent, matching the both-empty guard.
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the provider must not be called")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="   ",
            target_output="Sure, done!",
        )
        assert score is not None
        assert score.source_score == 1.0
        assert score.target_score == 0.0
        assert score.metadata["empty_side"] == "source"
        assert score.metadata["failure_categories"] == [SEMANTIC_REGRESSION]

    async def test_empty_side_respects_zero_min_similarity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The empty-side score goes through the same threshold check as a
        # measured one: min_similarity=0.0 means nothing is ever flagged.
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the provider must not be called")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        ev = CosineSimilarityEvaluator(embedding_model="x/y", min_similarity=0.0)
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="Done.",
            target_output="",
        )
        assert score is not None
        assert score.metadata["failure_categories"] == []

    async def test_embedding_failure_raises_evaluator_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A failed embedding call must raise (the harness records the
        # error and excludes the row) — never neutral-score, which would
        # count a broken measurement as "equivalent".
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("network down")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        ev = CosineSimilarityEvaluator()
        with pytest.raises(EvaluatorError, match="embedding failed"):
            await ev.score(
                prompt_id="p",
                example_id="e",
                input_vars={},
                source_output="a",
                target_output="b",
            )


class TestCosineEvaluatorFormatNormalization:
    """Presentation (code fences, JSON layout) is not meaning.

    One model fencing pretty-printed JSON where the other returns it
    compact used to drag cosine from ~0.97 to ~0.73 — formatting drowned
    the content signal and flagged every example of a run.
    """

    @staticmethod
    def _recording_embed(
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[str]:
        embedded: list[str] = []

        async def fake_embed(model: str, input: list[str]) -> Any:  # noqa: A002
            embedded.append(input[0])
            vec = [1.0, 0.0] if len(embedded) == 1 else [0.0, 1.0]
            return type("R", (), {"data": [{"embedding": vec}]})()

        monkeypatch.setattr(semantic_module.litellm, "aembedding", fake_embed)
        return embedded

    async def test_fenced_and_compact_json_with_same_content_score_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embedded = self._recording_embed(monkeypatch)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='```json\n{\n  "b": 2,\n  "a": "x"\n}\n```',
            target_output='{"a":"x","b":2}',
        )
        assert score is not None
        assert score.target_score == pytest.approx(1.0)
        assert embedded == ['{"a": "x", "b": 2}']

    async def test_differing_json_is_embedded_in_canonical_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embedded = self._recording_embed(monkeypatch)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='```json\n{"summary": "café", "category": "bug"}\n```\n',
            target_output='{"category":"billing","summary":"café"}',
        )
        assert sorted(embedded) == [
            '{"category": "billing", "summary": "café"}',
            '{"category": "bug", "summary": "café"}',
        ]

    async def test_fenced_non_json_has_only_the_fence_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embedded = self._recording_embed(monkeypatch)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="```python\ndef f():\n    return 1\n```",
            target_output="def f():\n    return 1",
        )
        assert score is not None
        assert score.target_score == pytest.approx(1.0)
        assert embedded == ["def f():\n    return 1"]

    async def test_prose_with_an_inner_fence_is_embedded_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only a fence wrapping the whole output is presentation; a fence
        # inside prose is part of the answer.
        embedded = self._recording_embed(monkeypatch)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        source = "Run this:\n```bash\nls\n```"
        await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output=source,
            target_output="Run ls.",
        )
        assert source in embedded

    async def test_empty_fence_counts_as_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An empty fenced block carries no text; embedding "" would 400.
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the provider must not be called")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        ev = CosineSimilarityEvaluator(embedding_model="x/y")
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output='{"a": 1}',
            target_output="```json\n```",
        )
        assert score is not None
        assert score.target_score == 0.0
        assert score.metadata["empty_side"] == "target"


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_clean_json(self) -> None:
        assert _parse_verdict('{"winner": "A", "reason": "x"}') == "A"

    def test_b_winner(self) -> None:
        assert _parse_verdict('{"winner": "B", "reason": "y"}') == "B"

    def test_tie(self) -> None:
        assert _parse_verdict('{"winner": "tie", "reason": "z"}') == "tie"

    def test_lowercase_winner(self) -> None:
        assert _parse_verdict('{"winner": "a"}') == "A"

    def test_json_inside_prose(self) -> None:
        text = 'Here is my verdict: {"winner": "A", "reason": "..."} done.'
        assert _parse_verdict(text) == "A"

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            _parse_verdict("just prose")

    def test_invalid_winner_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid winner"):
            _parse_verdict('{"winner": "C"}')


class TestFormatTranscript:
    def test_renders_roles_and_current_input_as_final_user_line(self) -> None:
        history = [
            {"role": "system", "content": "You are a scheduling assistant."},
            {"role": "user", "content": "create meeting"},
            {"role": "assistant", "content": "What time?"},
        ]
        text = _format_transcript(history, "1pm")
        assert text == (
            "Conversation context (both outputs respond to the final user message):\n"
            '"""\n'
            "[system] You are a scheduling assistant.\n"
            "[user] create meeting\n"
            "[assistant] What time?\n"
            "[user] 1pm\n"
            '"""'
        )

    def test_no_system_message(self) -> None:
        history = [
            {"role": "user", "content": "create meeting"},
            {"role": "assistant", "content": "What time?"},
        ]
        text = _format_transcript(history, "1pm")
        assert "[system]" not in text
        assert text.splitlines()[2] == "[user] create meeting"
        assert text.splitlines()[-2] == "[user] 1pm"

    def test_empty_history_still_appends_current_input(self) -> None:
        text = _format_transcript([], "hello")
        assert text.splitlines()[-2] == "[user] hello"

    def test_renders_tool_calls_and_tool_results(self) -> None:
        history = [
            {"role": "user", "content": "list my projects"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "get_projects", "arguments": {"status": "active"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"projects": []}'},
        ]
        text = _format_transcript(history, "yes")
        lines = text.splitlines()
        assert lines[3] == '[assistant] → get_projects({"status": "active"})'
        assert lines[4] == '[tool get_projects] {"projects": []}'

    def test_renders_assistant_text_alongside_its_tool_calls(self) -> None:
        history = [
            {
                "role": "assistant",
                "content": "Archiving those now.",
                "tool_calls": [{"id": "c1", "name": "archive_project", "arguments": {}}],
            },
        ]
        text = _format_transcript(history, "ok")
        assert text.splitlines()[2] == "[assistant] Archiving those now. → archive_project({})"

    def test_truncates_long_transcript_keeping_system_head_and_tail(self) -> None:
        system_msg = {"role": "system", "content": "SYS-" + "s" * 50}
        # Build enough turns to blow well past MAX_TRANSCRIPT_CHARS.
        turns = [system_msg]
        for i in range(200):
            turns.append({"role": "user", "content": f"user turn {i} " + "x" * 30})
            turns.append({"role": "assistant", "content": f"assistant turn {i} " + "y" * 30})
        text = _format_transcript(turns, "final question")

        assert len(text) < MAX_TRANSCRIPT_CHARS
        assert "SYS-" in text  # system head preserved
        assert "[... truncated ...]" in text
        # Tail preserved: the current input line must survive truncation.
        assert text.rstrip().splitlines()[-2] == "[user] final question"
        # An early-but-not-first turn should have been dropped.
        assert "user turn 0 " not in text

    def test_truncation_clips_a_current_input_longer_than_the_whole_budget(self) -> None:
        # Pathological case: the current-input line alone would blow the
        # cap even with everything else dropped. Must still respect the cap.
        history = [{"role": "system", "content": "sys " * 5}, {"role": "user", "content": "hi"}]
        text = _format_transcript(history, "Q" * 5000)
        assert len(text) < MAX_TRANSCRIPT_CHARS

    def test_truncation_total_stays_under_cap_even_with_no_system_message(self) -> None:
        turns = []
        for i in range(300):
            turns.append({"role": "user", "content": f"turn {i} " + "z" * 40})
            turns.append({"role": "assistant", "content": f"reply {i} " + "w" * 40})
        text = _format_transcript(turns, "final question")
        assert len(text) < MAX_TRANSCRIPT_CHARS
        assert "[... truncated ...]" in text
        assert text.rstrip().splitlines()[-2] == "[user] final question"


class TestPairwiseJudge:
    def _client_returning(self, text: str) -> ModelClient:
        client = ModelClient()
        client.complete = AsyncMock(  # type: ignore[method-assign]
            return_value=CompletionResult(
                text=text,
                model_id="x",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            ),
        )
        return client

    async def test_judge_survives_temperature_value_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end through the real ModelClient: the first dispatch 400s
        # the temperature value, the client adapts, the judge gets its
        # verdict. Before the adaptation existed this raised EvaluatorError
        # and a blocking judge poisoned the gate.
        from types import SimpleNamespace

        from evalshift_cli.models import client as client_module

        bad_request = type("BadRequestError", (Exception,), {})
        calls: list[dict[str, Any]] = []

        async def fake_acompletion(**kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise bad_request("'temperature' does not support 0.0 with this model.")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"winner": "A"}'),
                        finish_reason="stop",
                    ),
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        monkeypatch.setattr(client_module.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(
            client_module.litellm,
            "completion_cost",
            lambda completion_response=None, **_: 0.0,
        )
        rng = random.Random(0)
        rng.random = lambda: 0.0  # type: ignore[method-assign]  # target shown as A
        ev = PairwiseJudgeEvaluator(
            criterion_name="equivalence",
            criterion_prompt="Which is better?",
            judge_model="gpt-4o",
            client=ModelClient(),
            rng=rng,
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score is not None
        assert score.target_score == 1.0
        assert len(calls) == 2
        assert "temperature" not in calls[1]

    async def test_target_wins_when_target_is_a_and_a_wins(self) -> None:
        # Force target_is_a = True via a deterministic RNG.
        rng = random.Random(0)
        # random.Random(0).random() == 0.84... but we need it < 0.5 for target=A.
        # Easier to override below the bias by patching random() itself:
        rng.random = lambda: 0.0  # type: ignore[method-assign]
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning('{"winner": "A"}'),
            rng=rng,
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.target_score == 1.0
        assert score.source_score == 0.0

    async def test_target_loses_when_target_is_a_and_b_wins(self) -> None:
        rng = random.Random()
        rng.random = lambda: 0.0  # target=A  # type: ignore[method-assign]
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning('{"winner": "B"}'),
            rng=rng,
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.target_score == 0.0
        assert score.source_score == 1.0

    async def test_tie_yields_half_half(self) -> None:
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning('{"winner": "tie"}'),
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="src",
            target_output="tgt",
        )
        assert score.source_score == 0.5
        assert score.target_score == 0.5
        assert score.delta == 0.0

    async def test_reports_nothing_when_both_outputs_are_empty(self) -> None:
        # Both models answered with tool calls, not prose: there is nothing
        # for a judge to compare, so it must not be asked — and the tie it
        # used to fabricate was indistinguishable from a judged one.
        client = ModelClient()
        client.complete = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("the judge must not be called"),
        )
        ev = PairwiseJudgeEvaluator(criterion_name="x", criterion_prompt="x", client=client)
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="",
            target_output="   ",
        )
        assert score is None

    async def test_still_judges_when_only_one_side_is_empty(self) -> None:
        """A target that went silent where the source answered IS a regression."""
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning('{"winner": "A"}'),
        )
        score = await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={},
            source_output="Done.",
            target_output="",
        )
        assert score is not None
        assert score.metadata["verdict"] in {"A", "B", "tie"}

    async def test_malformed_response_raises_evaluator_error(self) -> None:
        # A judge that can't produce a verdict must raise (the harness
        # records the error and excludes the row) — never neutral-score,
        # which silently counts a broken judge as "tie".
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="x",
            client=self._client_returning("just prose, no JSON anywhere"),
        )
        with pytest.raises(EvaluatorError, match="judge call failed"):
            await ev.score(
                prompt_id="p",
                example_id="e",
                input_vars={},
                source_output="src",
                target_output="tgt",
            )

    async def test_api_failure_raises_evaluator_error(self) -> None:
        client = ModelClient()
        client.complete = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("temperature not supported"),
        )
        ev = PairwiseJudgeEvaluator(criterion_name="x", criterion_prompt="x", client=client)
        with pytest.raises(EvaluatorError, match="judge call failed"):
            await ev.score(
                prompt_id="p",
                example_id="e",
                input_vars={},
                source_output="src",
                target_output="tgt",
            )

    async def test_prompt_includes_transcript_when_history_passed(self) -> None:
        client = self._client_returning('{"winner": "tie"}')
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="is this a good reply?",
            client=client,
        )
        history = [
            {"role": "system", "content": "You are a scheduling assistant."},
            {"role": "user", "content": "create meeting"},
            {"role": "assistant", "content": "What time?"},
        ]
        await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={"user_message": "1pm"},
            source_output="src",
            target_output="tgt",
            history=history,
        )
        sent_prompt = client.complete.call_args.kwargs["prompt"]  # type: ignore[attr-defined]
        assert "Conversation context" in sent_prompt
        assert "[system] You are a scheduling assistant." in sent_prompt
        assert "[user] create meeting" in sent_prompt
        assert "[assistant] What time?" in sent_prompt
        assert "[user] 1pm" in sent_prompt
        # The context section sits between the role preamble and the criterion.
        preamble_idx = sent_prompt.index("impartial judge")
        context_idx = sent_prompt.index("Conversation context")
        criterion_idx = sent_prompt.index("Criterion:")
        assert preamble_idx < context_idx < criterion_idx

    async def test_prompt_has_no_context_section_and_no_stray_blank_lines_when_history_none(
        self,
    ) -> None:
        client = self._client_returning('{"winner": "tie"}')
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="is this a good reply?",
            client=client,
        )
        await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={"user_message": "hi"},
            source_output="src",
            target_output="tgt",
            history=None,
        )
        sent_prompt = client.complete.call_args.kwargs["prompt"]  # type: ignore[attr-defined]
        assert "Conversation context" not in sent_prompt
        # No blank-line artifacts from an empty context_section substitution.
        assert "\n\n\n" not in sent_prompt
        expected_start = (
            "You are an impartial judge comparing two AI assistant outputs "
            "against a criterion.\n\nCriterion: is this a good reply?\n"
        )
        assert sent_prompt.startswith(expected_start)

    async def test_current_input_derived_from_single_input_var(self) -> None:
        client = self._client_returning('{"winner": "tie"}')
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="c",
            client=client,
        )
        await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={"user_message": "what time works?"},
            source_output="src",
            target_output="tgt",
            history=[{"role": "user", "content": "hi"}],
        )
        sent_prompt = client.complete.call_args.kwargs["prompt"]  # type: ignore[attr-defined]
        assert "[user] what time works?" in sent_prompt

    async def test_current_input_derived_from_multiple_input_vars_as_json(self) -> None:
        client = self._client_returning('{"winner": "tie"}')
        ev = PairwiseJudgeEvaluator(
            criterion_name="x",
            criterion_prompt="c",
            client=client,
        )
        await ev.score(
            prompt_id="p",
            example_id="e",
            input_vars={"a": "1", "b": "2"},
            source_output="src",
            target_output="tgt",
            history=[{"role": "user", "content": "hi"}],
        )
        sent_prompt = client.complete.call_args.kwargs["prompt"]  # type: ignore[attr-defined]
        assert json.dumps({"a": "1", "b": "2"}) in sent_prompt


class TestPairwiseJudgeCache:
    def _client_returning(self, text: str) -> ModelClient:
        client = ModelClient()
        client.complete = AsyncMock(  # type: ignore[method-assign]
            return_value=CompletionResult(
                text=text,
                model_id="x",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                latency_ms=1,
            ),
        )
        return client

    def _kwargs(self) -> dict[str, Any]:
        return {
            "prompt_id": "p",
            "example_id": "e",
            "input_vars": {},
            "source_output": "src",
            "target_output": "tgt",
        }

    async def test_second_score_is_served_from_cache(self) -> None:
        client = self._client_returning('{"winner": "A"}')
        cache = await CacheStore.open(IN_MEMORY_DB)
        try:
            ev = PairwiseJudgeEvaluator(
                criterion_name="x",
                criterion_prompt="c",
                client=client,
                cache=cache,
            )
            first = await ev.score(**self._kwargs())
            second = await ev.score(**self._kwargs())
        finally:
            await cache.close()

        assert client.complete.await_count == 1  # type: ignore[attr-defined]
        assert second.target_score == first.target_score
        assert second.source_score == first.source_score
        # The randomized A/B orientation is part of the recorded verdict, so
        # a replayed score reports exactly what the live one did.
        assert second.metadata == first.metadata

    async def test_cache_hit_survives_flipped_ab_orientation(self) -> None:
        # The judge randomizes which output is shown as A. That must not
        # leak into the cache key, or ~half of all re-runs would miss.
        flip = iter([0.0, 0.9])
        rng = random.Random()
        rng.random = lambda: next(flip)  # type: ignore[method-assign]
        client = self._client_returning('{"winner": "A"}')
        cache = await CacheStore.open(IN_MEMORY_DB)
        try:
            ev = PairwiseJudgeEvaluator(
                criterion_name="x",
                criterion_prompt="c",
                client=client,
                rng=rng,
                cache=cache,
            )
            first = await ev.score(**self._kwargs())
            second = await ev.score(**self._kwargs())
        finally:
            await cache.close()

        assert client.complete.await_count == 1  # type: ignore[attr-defined]
        assert second.target_score == first.target_score

    async def test_different_criterion_does_not_share_cache_entries(self) -> None:
        client = self._client_returning('{"winner": "A"}')
        cache = await CacheStore.open(IN_MEMORY_DB)
        try:
            for criterion in ("brevity", "accuracy"):
                await PairwiseJudgeEvaluator(
                    criterion_name=criterion,
                    criterion_prompt=f"is it {criterion}?",
                    client=client,
                    cache=cache,
                ).score(**self._kwargs())
        finally:
            await cache.close()

        assert client.complete.await_count == 2  # type: ignore[attr-defined]

    async def test_no_cache_means_every_score_calls_the_judge(self) -> None:
        client = self._client_returning('{"winner": "A"}')
        ev = PairwiseJudgeEvaluator(criterion_name="x", criterion_prompt="c", client=client)
        await ev.score(**self._kwargs())
        await ev.score(**self._kwargs())
        assert client.complete.await_count == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_every_evaluator_satisfies_protocol(self, tmp_path: Path) -> None:
        schema_path = tmp_path / "s.json"
        schema_path.write_text('{"type": "object"}', encoding="utf-8")
        evaluators = [
            JsonSchemaEvaluator(schema_path=schema_path),
            RegexEvaluator(pattern="x"),
            LengthEvaluator(min_chars=1),
            CosineSimilarityEvaluator(),
            PairwiseJudgeEvaluator(criterion_name="x", criterion_prompt="x"),
        ]
        for ev in evaluators:
            assert isinstance(ev, Evaluator)


# ---------------------------------------------------------------------------
# Evaluator kind — the stable slug the analysis layer selects rows on
# ---------------------------------------------------------------------------


class TestEvaluatorKind:
    """Every evaluator advertises a type slug independent of its user name.

    ``analysis/policy.py`` used to select rows by an ``evaluator_name``
    prefix, so renaming an evaluator silently unhooked it from its policy
    budget. The slug is what makes the selection rename-proof.
    """

    def test_text_evaluators_advertise_their_kind(self, tmp_path: Path) -> None:
        schema_path = tmp_path / "s.json"
        schema_path.write_text('{"type": "object"}', encoding="utf-8")
        assert JsonSchemaEvaluator(schema_path=schema_path).kind == "structural"
        assert RegexEvaluator(pattern="x").kind == "structural"
        assert LengthEvaluator(min_chars=1).kind == "structural"
        assert CosineSimilarityEvaluator(name="anything").kind == "semantic"
        assert PairwiseJudgeEvaluator(criterion_name="x", criterion_prompt="x").kind == "llm_judge"

    @pytest.mark.asyncio
    async def test_tool_arguments_records_carry_its_kind(self) -> None:
        ev = ToolArgumentsEvaluator(ToolArgumentsEvaluatorConfig(name="anything_at_all"))
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=suite_example(id="x"),
            source_trace=ToolTrace(
                calls=[ToolCall(tool_name="a", arguments={"k": 1}, sequence_index=0)]
            ),
            target_trace=ToolTrace(
                calls=[ToolCall(tool_name="a", arguments={"k": 1}, sequence_index=0)],
            ),
        )
        assert record.evaluator_name == "anything_at_all"
        assert record.kind == "tool_arguments"

    @pytest.mark.asyncio
    async def test_tool_selection_records_carry_a_kind_per_axis(self) -> None:
        """One slug per axis, not one per evaluator.

        The policy layer selects rows on ``kind``, and conformance and
        divergence are different measurements against different baselines —
        a shared slug would put them in one budget and one comparison.
        """
        ev = ToolSelectionEvaluator(ToolSelectionEvaluatorConfig(name="whatever"))
        records = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=suite_example(id="x", expected_tools=[ExpectedToolCall(tool_name="a")]),
            source_trace=ToolTrace(calls=[ToolCall(tool_name="a", sequence_index=0)]),
            target_trace=ToolTrace(calls=[ToolCall(tool_name="a", sequence_index=0)]),
        )
        assert [r.kind for r in records] == [
            "tool_selection.conformance",
            "tool_selection.divergence",
        ]

    @pytest.mark.asyncio
    async def test_tool_trace_structure_records_carry_its_kind(self) -> None:
        ev = ToolTraceStructureEvaluator(ToolTraceStructureEvaluatorConfig(name="whatever"))
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=suite_example(id="x"),
            source_trace=ToolTrace(calls=[ToolCall(tool_name="a", sequence_index=0)]),
            target_trace=ToolTrace(calls=[ToolCall(tool_name="a", sequence_index=0)]),
        )
        assert record.kind == "tool_trace_structure"


class TestToolArgumentsOptionalFields:
    """A field present on one side only is a partial match, not a total one.

    ``get_projects(status="active")`` vs ``get_projects()`` scored 0.0 because
    the presence check ran before any strategy and ``_score_call_args`` takes
    the *union* of both sides' keys. ``status`` is optional in the declared
    schema, so both calls are valid — scoring that as total failure turns every
    tool with optional parameters into a false regression at ``blocking: true``.
    """

    @staticmethod
    async def _score(
        ev: ToolArgumentsEvaluator,
        *,
        source_args: dict[str, Any],
        target_args: dict[str, Any],
    ) -> float:
        record = await ev.score_pair(
            run_id="r",
            prompt_id="p",
            example=suite_example(id="x"),
            source_trace=ToolTrace(
                calls=[ToolCall(tool_name="get_projects", arguments=source_args, sequence_index=0)],
            ),
            target_trace=ToolTrace(
                calls=[ToolCall(tool_name="get_projects", arguments=target_args, sequence_index=0)],
            ),
        )
        return record.target_score

    async def test_a_field_on_one_side_only_is_not_a_total_failure(self) -> None:
        ev = ToolArgumentsEvaluator(ToolArgumentsEvaluatorConfig(name="args"))
        score = await self._score(ev, source_args={"status": "active"}, target_args={})
        assert score == pytest.approx(0.5)

    async def test_the_direction_of_the_omission_does_not_matter(self) -> None:
        ev = ToolArgumentsEvaluator(ToolArgumentsEvaluatorConfig(name="args"))
        score = await self._score(ev, source_args={}, target_args={"status": "active"})
        assert score == pytest.approx(0.5)

    async def test_strict_mode_restores_zero_for_a_missing_field(self) -> None:
        ev = ToolArgumentsEvaluator(
            ToolArgumentsEvaluatorConfig(name="args", optional_fields_scored="strict"),
        )
        score = await self._score(ev, source_args={"status": "active"}, target_args={})
        assert score == pytest.approx(0.0)

    async def test_both_sides_absent_is_still_a_full_match(self) -> None:
        ev = ToolArgumentsEvaluator(ToolArgumentsEvaluatorConfig(name="args"))
        assert await self._score(ev, source_args={}, target_args={}) == pytest.approx(1.0)

    async def test_a_wrong_value_is_still_a_total_failure_under_exact(self) -> None:
        """Leniency applies to *presence*, never to a value both sides passed."""
        ev = ToolArgumentsEvaluator(
            ToolArgumentsEvaluatorConfig(name="args", default_strategy="exact"),
        )
        score = await self._score(
            ev,
            source_args={"status": "active"},
            target_args={"status": "archived"},
        )
        assert score == pytest.approx(0.0)

    async def test_a_wrong_value_is_still_penalised_under_auto(self) -> None:
        """``auto`` grades short enum-like values by string similarity, so the
        penalty is partial rather than total until the schema dispatch lands
        and pins ``enum`` fields back to ``exact``."""
        ev = ToolArgumentsEvaluator(ToolArgumentsEvaluatorConfig(name="args"))
        score = await self._score(
            ev,
            source_args={"status": "active"},
            target_args={"status": "archived"},
        )
        assert score < 1.0

    async def test_an_explicit_null_on_both_sides_is_a_full_match(self) -> None:
        ev = ToolArgumentsEvaluator(ToolArgumentsEvaluatorConfig(name="args"))
        score = await self._score(ev, source_args={"status": None}, target_args={"status": None})
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Absence must be representable
# ---------------------------------------------------------------------------


class TestNothingMeasuredEmitsNothing:
    """A text evaluator handed two empty outputs must report *nothing*.

    Both halves of every pair in ``r_20260820_project_insights_143a5f`` are
    ``""`` — the models answered with a tool call and no prose. There is
    nothing to embed and nothing for a judge to read, and both evaluators
    correctly decline to call a provider. What they could not do was *say
    so*: ``score`` returned a non-optional :class:`PairedScore`, so they
    invented one — ``semantic`` invented perfect similarity and ``llm_judge``
    invented a tie, and both landed in ``scores.jsonl``, the report, and the
    hosted bundle as maximum scores over a comparison that never happened.
    The ``metadata["skipped"]`` flag was a workaround for a return type that
    could not express absence, and it only protected the one consumer that
    checked it.
    # S1: ``score`` returns ``PairedScore | None`` and ``evaluate.py`` writes
    # no record for ``None``.
    """

    async def test_semantic_emits_no_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the provider must not be called")

        monkeypatch.setattr(semantic_module.litellm, "aembedding", boom)
        source_output, target_output = empty_output_pair()
        ev = CosineSimilarityEvaluator(embedding_model="gemini/gemini-embedding-001")
        assert (
            await ev.score(
                prompt_id=PROMPT_ID,
                example_id="cap_16bd3ed793874ecea123cf7c7c2b3593",
                input_vars={},
                source_output=source_output,
                target_output=target_output,
            )
            is None
        )

    async def test_llm_judge_emits_no_score(self) -> None:
        client = ModelClient()
        client.complete = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError("the judge must not be called"),
        )
        source_output, target_output = empty_output_pair()
        ev = PairwiseJudgeEvaluator(
            criterion_name="equivalence",
            criterion_prompt="which is more complete and correct?",
            client=client,
        )
        assert (
            await ev.score(
                prompt_id=PROMPT_ID,
                example_id="cap_16bd3ed793874ecea123cf7c7c2b3593",
                input_vars={},
                source_output=source_output,
                target_output=target_output,
            )
            is None
        )

    async def test_tool_arguments_emits_no_record_without_expected_arguments(self) -> None:
        """The run's ``routing_args`` rows: ``against: expected``, no ground truth."""
        ev = ToolArgumentsEvaluator(
            ToolArgumentsEvaluatorConfig(name=TOOL_ARGUMENTS_NAME, against="expected"),
        )
        pair = load_pairs()[0]
        assert (
            records_of(
                await ev.score_pair(
                    run_id=RUN_ID,
                    prompt_id=PROMPT_ID,
                    example=pair.example,
                    source_trace=pair.source_trace,
                    target_trace=pair.target_trace,
                )
            )
            == []
        )
