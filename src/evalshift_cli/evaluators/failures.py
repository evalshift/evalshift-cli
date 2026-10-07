"""Stable failure-category labels, and the prose explaining them, surfaced in
reports and policy output."""

from __future__ import annotations

FORMAT_FAILURE: str = "FORMAT_FAILURE"
SEMANTIC_REGRESSION: str = "SEMANTIC_REGRESSION"
TOOL_SELECTION_DRIFT: str = "TOOL_SELECTION_DRIFT"
#: Both models violated the same recorded ground truth on one example. That is
#: not a migration finding — the source model failing ground truth captured
#: from that same source model is proof of a broken harness (wrong toolset
#: attached, wrong prompt, suite promoted from a different agent), so it gets
#: its own category rather than sharing ``TOOL_SELECTION_DRIFT``.
TOOL_GROUND_TRUTH_MISS: str = "TOOL_GROUND_TRUTH_MISS"
ARGUMENT_VALUE_DRIFT: str = "ARGUMENT_VALUE_DRIFT"
TOOL_TRACE_STRUCTURE_DRIFT: str = "TOOL_TRACE_STRUCTURE_DRIFT"
TOOL_ORDER_DRIFT: str = "TOOL_ORDER_DRIFT"
DANGEROUS_ACTION_DRIFT: str = "DANGEROUS_ACTION_DRIFT"
MISSING_VERIFICATION_STEP: str = "MISSING_VERIFICATION_STEP"
UNNECESSARY_TOOL_CALL: str = "UNNECESSARY_TOOL_CALL"
REFUSAL_REGRESSION: str = "REFUSAL_REGRESSION"
#: The target broke a rule the team wrote down in ``trace_invariants`` --
#: judged against the rule, never against the source, so a rule both models
#: broke still carries it.
INVARIANT_VIOLATION: str = "INVARIANT_VIOLATION"

#: Why a model failing ground truth recorded from itself indicts the harness.
#: Defined once and imported by every surface that reports it — ``evaluate``
#: says it the moment the rate is known, ``analyze`` says it again as a
#: recommendation once the rows leave the rates — because a second wording of
#: one fact is a second thing to keep true.
BROKEN_HARNESS_CAUSES: str = (
    "Ground truth the source model itself fails usually means the suite was "
    "captured against a different toolset, prompt or agent."
)

#: Plain-language display names, one per machine label above. The machine
#: labels are the stable grouping keys (JSON payloads, the hosted diff); these
#: are what a reader is shown wherever a label reaches prose or HTML. Keep the
#: two sets in lockstep — ``test_failures.py`` walks ``__all__`` to enforce it.
CATEGORY_LABELS: dict[str, str] = {
    FORMAT_FAILURE: "Broken output format",
    SEMANTIC_REGRESSION: "Meaning of the answer changed",
    TOOL_SELECTION_DRIFT: "Different tools chosen",
    TOOL_GROUND_TRUTH_MISS: "Both models missed the recorded tools",
    ARGUMENT_VALUE_DRIFT: "Tool arguments changed",
    TOOL_TRACE_STRUCTURE_DRIFT: "Tool-call sequence restructured",
    TOOL_ORDER_DRIFT: "Tools called in a different order",
    DANGEROUS_ACTION_DRIFT: "New risky tool action",
    MISSING_VERIFICATION_STEP: "Verification step skipped",
    UNNECESSARY_TOOL_CALL: "Unnecessary extra tool call",
    REFUSAL_REGRESSION: "New refusal",
    INVARIANT_VIOLATION: "Broke a hand-written trace rule",
}


def category_label(category: str) -> str:
    """The display name for ``category``, humanised when unknown.

    Custom evaluators may stamp categories this table has never heard of;
    echoing their raw identifier back at the reader is exactly what the
    display layer exists to avoid, so an unknown one is rewritten as words.

    Args:
        category: A machine label, e.g. ``TOOL_SELECTION_DRIFT``.

    Returns:
        The mapped display name, or the identifier with underscores turned
        into spaces and sentence casing applied.
    """
    label = CATEGORY_LABELS.get(category)
    if label is not None:
        return label
    return category.replace("_", " ").strip().capitalize()


__all__ = [
    "ARGUMENT_VALUE_DRIFT",
    "BROKEN_HARNESS_CAUSES",
    "CATEGORY_LABELS",
    "DANGEROUS_ACTION_DRIFT",
    "FORMAT_FAILURE",
    "INVARIANT_VIOLATION",
    "MISSING_VERIFICATION_STEP",
    "REFUSAL_REGRESSION",
    "SEMANTIC_REGRESSION",
    "TOOL_GROUND_TRUTH_MISS",
    "TOOL_ORDER_DRIFT",
    "TOOL_SELECTION_DRIFT",
    "TOOL_TRACE_STRUCTURE_DRIFT",
    "UNNECESSARY_TOOL_CALL",
    "category_label",
]
