"""Whether two model ids name the same model, for capture-suite staleness.

A capture records whatever id the provider echoed back -- often a dated snapshot
(``claude-sonnet-5-20260301``) or a provider-prefixed id -- while ``defaults.source_model``
is usually the bare alias. Matching errs toward "same": a false match keeps a stale case a
little longer, a false mismatch would retire good ones.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from evalshift_cli.models.registry import resolve_model

_SNAPSHOT_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")


def bare_model_name(model_id: str) -> str:
    """``model_id`` lower-cased, without its provider prefix or a trailing snapshot date."""
    name = model_id.strip().lower().rsplit("/", 1)[-1]
    return _SNAPSHOT_SUFFIX.sub("", name)


def same_model(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` share a bare name or resolve to one registry id."""
    if not a.strip() or not b.strip():
        return False
    if bare_model_name(a) == bare_model_name(b):
        return True
    return resolve_model(a).id == resolve_model(b).id


def model_filter_applies(recorded: Iterable[str], source_model: str | None) -> bool:
    """Whether filtering one suite by ``source_model`` is safe (the spec §3 guardrail).

    False without a source model, when nothing recorded a model, and when not one recorded
    model matches it -- a wrong or unrecognised ``defaults.source_model``, or a migration whose
    first new-model capture has not arrived yet, must never empty or freeze a suite.
    """
    if not source_model:
        return False
    return any(same_model(model, source_model) for model in recorded if model)


def describe_models(recorded: Iterable[str], *, limit: int = 3) -> str:
    """``a ×3, c ×2, b ×1, +1 more``: the most-recorded ids first, at most ``limit`` of them."""
    counts = Counter(model for model in recorded if model)
    shown = ", ".join(f"{model} ×{n}" for model, n in counts.most_common(limit))
    rest = len(counts) - limit
    return f"{shown}, +{rest} more" if rest > 0 else shown
