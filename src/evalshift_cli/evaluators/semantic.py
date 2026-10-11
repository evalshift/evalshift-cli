"""Semantic-similarity evaluator using LLM-provider embeddings.

We treat similarity as a *target preservation* score: source is 1.0 by
definition (it preserves itself), and target gets its cosine similarity
to source. ``delta = target - source`` is then in [-2, 0] and any
negative number indicates drift.

This deliberately doesn't make symmetric "both halves same score" pairs
because that would always produce delta=0 and defeat the regression
detection.

Both outputs are reduced to their meaning before embedding (see
:func:`_meaning_text`): a code fence wrapping the whole output is dropped
and JSON is re-serialized canonically. Presentation otherwise dominates
the embedding — a fenced, pretty-printed object against the same object
compact scores ~0.73, enough to flag every example of a run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any

import litellm

from evalshift_cli.cache.store import CacheStore, cache_key
from evalshift_cli.evaluators.base import EvaluatorError, PairedScore
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION

log = logging.getLogger(__name__)

#: Marker written into the cache key's ``inputs`` so embedding entries can
#: never collide with completion entries for the same model + text.
_EMBEDDING_INPUTS: dict[str, Any] = {"kind": "embedding"}

#: A markdown code fence wrapping the *entire* output (optional info
#: string, e.g. ``json``). A fence inside prose is part of the answer.
_WHOLE_FENCE = re.compile(r"\A\s*```[^\n`]*\n(.*?)\n?```\s*\Z", re.DOTALL)


class CosineSimilarityEvaluator:
    """Cosine similarity between source and target output embeddings."""

    #: Stable evaluator-type slug the analysis layer selects policy rows
    #: on — never the user-chosen ``name``, which is free to change.
    kind = "semantic"

    def __init__(
        self,
        *,
        embedding_model: str = "text-embedding-3-small",
        min_similarity: float = 0.9,
        name: str = "semantic.cosine",
        cache: CacheStore | None = None,
    ) -> None:
        """Build the evaluator.

        Args:
            embedding_model: LiteLLM embedding model id.
            min_similarity: Cosine similarity below which the target is
                flagged as a semantic regression.
            name: Evaluator name stamped onto each record.
            cache: Optional store for embedding reuse. An embedding is a
                pure function of (model, text), so caching makes re-running
                ``evaluate`` over the same run essentially free. ``None``
                disables caching.
        """
        self.name = name
        self.embedding_model = embedding_model
        self.min_similarity = min_similarity
        self.cache = cache

    async def score(
        self,
        *,
        prompt_id: str,
        example_id: str,
        input_vars: dict[str, Any],
        source_output: str,
        target_output: str,
        history: list[dict[str, str]] | None = None,
    ) -> PairedScore | None:
        # Everything below — the silence guards, the identical shortcut and
        # the embeddings — sees meaning only, never fences or JSON layout.
        source_output = _meaning_text(source_output)
        target_output = _meaning_text(target_output)
        source_silent = not source_output
        target_silent = not target_output
        if source_silent and target_silent:
            # Tool-only turn: both models answered with calls, not prose.
            # There is nothing to embed, and an embedding provider will 400
            # on an empty input. Nothing was measured, so nothing is
            # reported — a score here would be invented, and the invented
            # 1.0/1.0 this used to return read as perfect similarity.
            #
            # One empty side deliberately still scores: a target that went
            # silent where the source answered is exactly the regression
            # this evaluator exists to catch.
            return None

        if source_silent or target_silent:
            # One side has text and the other doesn't: similarity to
            # nothing is 0.0 by definition, and the empty side can't be
            # embedded anyway (the provider 400s on empty input) — so no
            # call is made at all. The score still passes through the
            # normal min_similarity gate below.
            empty_side = "source" if source_silent else "target"
            explanation = (
                "The target answered with text where the source produced no text."
                if source_silent
                else "The target produced no text where the source answered."
            )
            return PairedScore(
                source_score=1.0,  # source preserves itself by definition
                target_score=0.0,
                explanation=explanation,
                metadata={
                    "raw_cosine": 0.0,
                    "empty_side": empty_side,
                    "failure_categories": (
                        [SEMANTIC_REGRESSION] if self.min_similarity > 0.0 else []
                    ),
                },
            )

        try:
            if source_output == target_output:
                # Identical text is a guaranteed 1.0; one embedding is enough.
                source_emb = target_emb = await self._embed(source_output)
            else:
                # The two sides are independent calls; overlapping them
                # halves the wall-clock cost of every semantic score.
                source_emb, target_emb = await asyncio.gather(
                    self._embed(source_output),
                    self._embed(target_output),
                )
        except Exception as exc:
            # A failed embedding call is a broken measurement, not a score.
            # Raise so the harness records the error and the analysis layer
            # excludes the row — a 0/0 score here would read as "both sides
            # equally bad" and poison the regression metrics.
            log.warning("embedding failed for %s/%s: %s", prompt_id, example_id, exc)
            raise EvaluatorError(
                f"embedding failed for {prompt_id}/{example_id}: {exc}",
            ) from exc

        sim = _cosine(source_emb, target_emb)
        # Clamp to [0,1] so the pydantic constraint holds; cosine can
        # be slightly negative or above 1 due to float drift.
        sim_clamped = max(0.0, min(1.0, sim))
        return PairedScore(
            source_score=1.0,  # source preserves itself by definition
            target_score=sim_clamped,
            metadata={
                "raw_cosine": sim,
                "failure_categories": (
                    [SEMANTIC_REGRESSION] if sim_clamped < self.min_similarity else []
                ),
            },
        )

    async def _embed(self, text: str) -> list[float]:
        key = self._cache_key(text)
        if self.cache is not None:
            hit = await self.cache.get(key)
            if hit is not None:
                cached = _decode_embedding(hit.response_text)
                if cached is not None:
                    return cached

        response = await litellm.aembedding(
            model=self.embedding_model,
            input=[text],
        )
        # LiteLLM returns either dict-like or object-like depending on
        # provider; handle both.
        data = response.data if hasattr(response, "data") else response["data"]
        first = data[0]
        embedding = [
            float(x) for x in (first["embedding"] if isinstance(first, dict) else first.embedding)
        ]

        if self.cache is not None:
            await self.cache.put(
                key,
                model_id=self.embedding_model,
                prompt_text=text,
                inputs=_EMBEDDING_INPUTS,
                response_text=json.dumps(embedding),
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0,
            )
        return embedding

    def _cache_key(self, text: str) -> str:
        """Key an embedding by (model, text) — nothing else affects it.

        ``temperature``/``max_tokens`` are meaningless for embeddings but
        the shared :func:`cache_key` takes them, so we pin them to zero.
        The ``inputs`` marker keeps embedding entries from ever colliding
        with a completion cached under the same model + text.
        """
        return cache_key(
            model_id=self.embedding_model,
            prompt_text=text,
            inputs=_EMBEDDING_INPUTS,
            temperature=0.0,
            max_tokens=0,
        )


def _meaning_text(text: str) -> str:
    """Strip presentation from an output, keeping what it says.

    Drops a code fence wrapping the whole output, then, if what remains is
    a JSON object or array, re-serializes it with sorted keys and default
    separators — layout and key order are not meaning. Anything else is
    returned stripped of surrounding whitespace.

    Args:
        text: A raw model output.

    Returns:
        The text to embed; empty when the output carries no text.
    """
    stripped = text.strip()
    fenced = _WHOLE_FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if not isinstance(decoded, dict | list):
        return stripped
    return json.dumps(decoded, sort_keys=True, ensure_ascii=False)


def _decode_embedding(payload: str) -> list[float] | None:
    """Decode a cached embedding, treating anything unreadable as a miss.

    A corrupt or shape-shifted cache row must never fail a run — the cache
    is a disposable optimisation, so we fall through to a live call.
    """
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list) or not decoded:
        return None
    try:
        return [float(x) for x in decoded]
    except (TypeError, ValueError):
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, returning 0 for any degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


__all__ = ["CosineSimilarityEvaluator"]
