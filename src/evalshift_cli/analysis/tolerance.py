"""Which score drops an evaluator's own threshold already forgives.

``semantic`` pins the source at 1.0, so every target that words anything
differently scores below it: a negative delta is the evaluator's *normal*
output, not evidence of a regression. ``min_similarity`` is where the user
drew the line, and the evaluator records which side of it a row fell on as a
``SEMANTIC_REGRESSION`` failure category at scoring time.

The policy budgets and the paired significance test must read that one line.
When only the budgets did, nine rows at 0.90-0.98 passed every budget and still
failed the run as a "critical" regression — nine small, consistent drops have
almost no variance, so the test was certain of a difference nobody had asked
it to find.
"""

from __future__ import annotations

from evalshift_cli.evaluators.base import EvalRecord
from evalshift_cli.evaluators.failures import SEMANTIC_REGRESSION

#: The ``EvalRecord.kind`` slug the semantic evaluator stamps.
SEMANTIC_KIND = "semantic"

#: Records checkpointed before ``EvalRecord.kind`` existed carry only the name
#: prefix the evaluator was registered under.
_LEGACY_SEMANTIC_PREFIX = "semantic."


def is_semantic(record: EvalRecord) -> bool:
    """Whether ``record`` came from a ``semantic`` evaluator.

    Selected by kind, so a user who renames the evaluator keeps the rule;
    falls back to the legacy name prefix only for rows written without one.
    """
    if record.kind:
        return record.kind == SEMANTIC_KIND
    return record.evaluator_name.startswith(_LEGACY_SEMANTIC_PREFIX)


def is_semantic_regression(record: EvalRecord) -> bool:
    """Whether a semantic record breached ``min_similarity``.

    Reads the ``SEMANTIC_REGRESSION`` flag the evaluator wrote at scoring time
    rather than re-deriving it from ``delta``: the threshold lives in config,
    and the evaluator is the one place that knew it.
    """
    categories = record.metadata.get("failure_categories", [])
    return isinstance(categories, list) and SEMANTIC_REGRESSION in categories


def within_tolerance(record: EvalRecord) -> bool:
    """Whether ``record``'s drop is inside its evaluator's own threshold.

    True only for a semantic row that stayed above ``min_similarity``. Such a
    row counts as *equivalent* in every policy rate and as *no change* in the
    paired test; every other row is taken at its delta.
    """
    return is_semantic(record) and not is_semantic_regression(record)


__all__ = [
    "SEMANTIC_KIND",
    "is_semantic",
    "is_semantic_regression",
    "within_tolerance",
]
