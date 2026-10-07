"""Render :class:`ReportData` as a single self-contained HTML file.

Every asset (CSS, fonts, glyphs) is inlined so the report works offline
and can be emailed/zipped without breaking. We deliberately ship no
JavaScript in the MVP — static HTML is more reliable to render and
easier to skim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from evalshift_cli.analysis.policy import BUDGET_LABELS, BUDGET_MEANINGS
from evalshift_cli.analysis.statistics import AXIS_NOTE_PREFIX, UNMEASURED_NOTE_PREFIX
from evalshift_cli.evaluators.failures import category_label
from evalshift_cli.evaluators.tool_selection import KIND_CONFORMANCE, KIND_DIVERGENCE
from evalshift_cli.insights.models import Insight
from evalshift_cli.reports.json import (
    SHARED_GROUND_TRUTH_NOTE_PREFIX,
    ReportData,
    TopRegression,
)

REPORT_HTML_FILENAME: str = "report.html"

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Icon per ``InsightFinding.kind``. Matches the hosted design's ✓ / ✕ / !.
_INSIGHT_GLYPHS: dict[str, str] = {
    "positive": "✓",
    "negative": "✕",
    "warning": "!",
}


_SEVERITY_GLYPHS: dict[str, str] = {
    "critical": "✗",
    "high": "✗",
    "medium": "⚠",
    "low": "⚠",
    "improved": "↑",
    "none": "✓",
    "insufficient": "?",
}

# Plain-language verdict shown in the executive summary. Each entry is a
# (headline, explanation) pair aimed at readers who don't parse Cohen's d.
_SEVERITY_VERDICTS: dict[str, tuple[str, str]] = {
    "critical": ("Regressed — critical", "Target scores clearly lower than source."),
    "high": ("Regressed — high", "Target scores notably lower than source."),
    "medium": ("Regressed — medium", "Target scores somewhat lower than source."),
    "low": ("Minor change", "Small dip — likely tolerable."),
    "improved": ("Improved", "Target scores higher than source."),
    "none": ("Equivalent", "No meaningful difference between models."),
    "insufficient": ("Not enough data", "Too few samples to judge."),
}

# `insufficient` covers two very different situations: a real but small
# sample, and a comparison where every row was non-applicable. Only the
# second one means the evaluator never ran.
_UNMEASURED_VERDICT: tuple[str, str] = (
    "Nothing measured",
    "This evaluator found no comparable pair — its result is unknown, not equal.",
)

# A conformance axis on which every pair scored the same non-zero-delta miss.
# The delta really is zero, so the severity really is `none` — but "no
# meaningful difference between models" is a statement about the migration,
# and this row is a statement about the suite.
_SHARED_MISS_VERDICT: tuple[str, str] = (
    "Ground truth missed by both",
    "Both models missed the suite's recorded tool calls on every pair — this "
    "measures the suite, not the migration.",
)

# Small paired samples make effect sizes and p-values unstable; flag them.
_SMALL_SAMPLE_THRESHOLD: int = 10


def _verdict(severity: str, notes: Sequence[str] | None = None) -> tuple[str, str]:
    """Return the (headline, explanation) pair for a severity bucket.

    Two notes override the bucket, because in both cases the number is right
    and the plain-language reading of it is not: an evaluator that compared
    nothing is unknown rather than equal, and an axis both models failed
    identically is a broken suite rather than a safe migration.
    """
    if any(note.startswith(SHARED_GROUND_TRUTH_NOTE_PREFIX) for note in notes or ()):
        return _SHARED_MISS_VERDICT
    if severity == "insufficient" and any(
        note.startswith(UNMEASURED_NOTE_PREFIX) for note in notes or ()
    ):
        return _UNMEASURED_VERDICT
    return _SEVERITY_VERDICTS.get(severity, ("Unknown", ""))


def _verdict_glyph(severity: str, notes: Sequence[str] | None = None) -> str:
    """The glyph beside a verdict headline.

    A shared ground-truth miss scores ``none``, whose glyph is ✓. Leaving it
    there would put a tick on the row that says the suite is broken.
    """
    if any(note.startswith(SHARED_GROUND_TRUTH_NOTE_PREFIX) for note in notes or ()):
        return "!"
    return _SEVERITY_GLYPHS.get(severity, "?")


def _confidence_label(severity: str, p_corrected: float) -> str:
    """Map a corrected p-value to a plain-language confidence word.

    Returns an em dash when the comparison had too little data to judge.
    """
    if severity == "insufficient":
        return "—"
    if p_corrected < 0.01:
        return "Certain"
    if p_corrected < 0.05:
        return "Likely"
    return "Unclear"


# Friendly display names for the built-in evaluators. Falls back to a
# humanised "Family: metric" for anything not listed (e.g. custom criteria).
_EVALUATOR_LABELS: dict[str, str] = {
    "semantic.cosine": "Semantic similarity",
    "structural.json_schema": "Valid structure",
    "structural.regex": "Pattern match",
    "structural.length": "Output length",
}
_EVALUATOR_FAMILIES: dict[str, str] = {
    "llm_judge": "LLM judge",
    "tool_selection": "Tool selection",
    "tool_arguments": "Tool arguments",
    "tool_trace_structure": "Tool sequence",
    "semantic": "Semantic",
    "structural": "Structure",
    "trace_invariants": "Trace rules",
}

# Axis labels, keyed on the record's ``kind`` slug and never on the
# evaluator's *name*. The name is whatever the user typed in evalshift.yaml —
# on the run this work came from, both axes were called `routing` — so a
# name-keyed map cannot tell two axes apart, which is precisely how a
# critical divergence regression and a ✓ conformance row came to render as
# two identical lines.
_AXIS_LABELS: dict[str, str] = {
    KIND_CONFORMANCE: "conformance",
    KIND_DIVERGENCE: "divergence",
}

# What each axis compares, spelled out — the two answer different questions
# against different baselines, and a reader who assumes otherwise reads a
# zero delta on conformance as a safe migration.
_AXIS_BLURBS: dict[str, str] = {
    KIND_CONFORMANCE: "each side graded against the suite's recorded tool calls",
    KIND_DIVERGENCE: "the target graded against what the source did",
}

# Statistical tests, spelled out for the row tooltip.
_TEST_LABELS: dict[str, str] = {
    "paired_t": "paired t-test",
    "wilcoxon": "Wilcoxon signed-rank test",
}


def _axis_of(notes: Sequence[str] | None) -> str:
    """The axis slug a comparison names in ``notes``, or ``""``.

    ``analyze`` writes the note only where one evaluator contributed several
    axes to the same slice — the only case where ``(prompt, evaluator,
    slice)`` stops identifying a row.
    """
    for note in notes or ():
        if note.startswith(AXIS_NOTE_PREFIX):
            return note[len(AXIS_NOTE_PREFIX) :].strip()
    return ""


def _axis_blurb(kind: str) -> str:
    """One phrase saying what an axis compares, or ``""`` for a plain evaluator."""
    return _AXIS_BLURBS.get(kind, "")


def _evaluator_label(name: str, kind: str = "") -> str:
    """Return a human-readable name for an evaluator id like ``semantic.cosine``.

    ``kind`` names the axis when one evaluator scores more than one. It is
    appended rather than substituted: the user-chosen name is what they
    configured and the axis is what was measured, and a reader needs both.
    """
    label = _base_evaluator_label(name)
    axis = _AXIS_LABELS.get(kind, "")
    return f"{label} — {axis}" if axis else label


def _base_evaluator_label(name: str) -> str:
    if name in _EVALUATOR_LABELS:
        return _EVALUATOR_LABELS[name]
    head, _, tail = name.partition(".")
    family = _EVALUATOR_FAMILIES.get(head, head.replace("_", " ").capitalize())
    if tail:
        return f"{family}: {tail.replace('_', ' ')}"
    return family


def _test_label(test: str) -> str:
    """Spell out a statistical test id for tooltips."""
    return _TEST_LABELS.get(test, test)


def _kind_label(kind: str) -> str:
    """A record-kind slug as words, e.g. ``tool_selection.divergence``.

    Used where a regression card has no failure category to show and falls
    back to naming what was scored — which must still read as words, not as
    the record's internal slug.
    """
    family, _, _ = kind.partition(".")
    label = _EVALUATOR_FAMILIES.get(family, family.replace("_", " ").capitalize())
    axis = _AXIS_LABELS.get(kind, "")
    return f"{label} — {axis}" if axis else label


def _regression_reason(regression: TopRegression) -> str:
    """Explain, in one plain sentence, why an example was flagged.

    Prefers the evaluator's own written rationale (e.g. an llm_judge verdict).
    Falls back to an explanation derived from the record — its axis first,
    then its evaluator family — when the evaluator scores numerically and
    writes no prose.

    Dispatch is on ``kind``, not on the evaluator's name. The name is the
    user's: on the run this work came from both tool axes were called
    ``routing``, which matched no family prefix, so every tool regression
    rendered its scores and no reason at all.
    """
    text = regression.explanation.strip()
    if text:
        return text
    if regression.kind == KIND_DIVERGENCE:
        return _tool_change_sentence(regression)
    if regression.kind == KIND_CONFORMANCE:
        return (
            "The target missed the suite's recorded tool calls where the source "
            "matched them." + _tool_change_suffix(regression)
        )
    name = regression.evaluator_name
    if name.startswith("semantic"):
        raw = regression.metadata.get("raw_cosine")
        if isinstance(raw, (int, float)):
            return (
                f"The target reworded the content enough to drift from the "
                f"source's meaning ({raw * 100:.0f}% similar; 100% = identical)."
            )
        return "The target drifted from the source's meaning."
    if name.startswith("tool"):
        return "The target used different tools than the source."
    if name.startswith("structural"):
        return "The target's output no longer matches the expected structure."
    return ""


def _tool_change_sentence(regression: TopRegression) -> str:
    """Name both sides' tools — the whole content of a divergence finding."""
    change = regression.tool_change
    if change is None:
        return "The target called different tools than the source."
    return (
        f"The source called {_tool_list(change.source_names)}; "
        f"the target called {_tool_list(change.target_names)}."
    )


def _tool_change_suffix(regression: TopRegression) -> str:
    change = regression.tool_change
    if change is None:
        return ""
    return (
        f" Source called {_tool_list(change.source_names)}, "
        f"target called {_tool_list(change.target_names)}."
    )


def _tool_list(names: Sequence[str]) -> str:
    """Render tool names for prose, or say plainly that there were none."""
    return ", ".join(names) if names else "no tools"


def _latency(ms: float, live_calls: int) -> str:
    """Format a latency in human units.

    Returns an em dash when there were no live calls (all cache hits), since a
    latency of zero would misleadingly read as instantaneous. Sub-second values
    stay in milliseconds; anything ≥1s renders as seconds with one decimal.
    """
    if live_calls == 0:
        return "—"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000.0:.1f} s"


def _effect_magnitude(effect_size: float) -> str:
    """Describe a Cohen's d magnitude in words (Cohen's conventions)."""
    d = abs(effect_size)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    if d < 1.3:
        return "large"
    return "very large"


def _display_timestamp(started_at: str) -> str:
    """Render the run's ISO start time as ``YYYY-MM-DD HH:MM:SS UTC``.

    The header sets a whole run in context in one line; a microsecond-precise
    offset-suffixed ISO string spends a third of that line on digits nobody
    reads. Anything unparseable falls through unchanged rather than being
    dropped — a malformed timestamp is worth seeing.
    """
    try:
        moment = datetime.fromisoformat(started_at)
    except ValueError:
        return started_at
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC")


def _suite_name(suite_path: str) -> str:
    """The short suite label shown in the header pill.

    Suites live at ``.evalshift/suites/<name>/golden.jsonl``, so the directory
    is the name the user typed and the file name is boilerplate. Anything laid
    out differently falls back to the file's stem.
    """
    if not suite_path:
        return ""
    path = PurePosixPath(suite_path.replace("\\", "/"))
    if path.parent.name and path.parent.parent.name == "suites":
        return path.parent.name
    return path.stem


def _pct_delta(source: float, target: float) -> float | None:
    """Target relative to source, in percent, or ``None`` when unmeasurable.

    A zero source is not a 0% or an infinite change — it is a baseline that
    never happened, and the panels render an em dash for it rather than a
    number the reader would take at face value.
    """
    if source <= 0:
        return None
    return (target - source) / source * 100.0


def _run_latency_delta_pct(totals: Mapping[str, Mapping[str, float]]) -> float | None:
    """Run-level latency change, or ``None`` unless both roles measured some live.

    The means cover only calls measured fully live on this run (cache hits and
    rows with any cached round are excluded), so a role with none averages
    0.0 — unmeasured, not instant. :func:`_pct_delta` already refuses a zero
    source; a zero target would otherwise read as a -100% change.
    """
    if not (totals["source"]["live_calls"] and totals["target"]["live_calls"]):
        return None
    return _pct_delta(totals["source"]["latency_ms_avg"], totals["target"]["latency_ms_avg"])


def _run_role_totals(sections: Sequence[Any]) -> dict[str, dict[str, float]]:
    """Roll every prompt's economics up into one source/target pair.

    Only the figures the header panels quote — spend and mean live latency.
    A run-level p95 is deliberately absent: percentiles do not merge, and
    inventing one from per-prompt p95s would put a fabricated number beside
    measured ones.
    """
    totals: dict[str, dict[str, float]] = {
        role: {"cost": 0.0, "latency_ms_sum": 0.0, "live_calls": 0.0}
        for role in ("source", "target")
    }
    for section in sections:
        for role in ("source", "target"):
            econ = getattr(section.economics, role)
            bucket = totals[role]
            bucket["cost"] += econ.total_cost_usd
            bucket["latency_ms_sum"] += econ.latency_ms_avg * econ.live_calls
            bucket["live_calls"] += econ.live_calls
    for bucket in totals.values():
        live = bucket["live_calls"]
        bucket["latency_ms_avg"] = bucket["latency_ms_sum"] / live if live else 0.0
    return totals


def _budget_tally(budgets: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    """Return ``(passed, total, unmeasured)`` over a decision's budgets.

    A budget counted over an empty sample passes by default and measures
    nothing, so it is excluded from the passed count and reported separately —
    the same rule ``BudgetResult.measured`` applies for the narrative. The
    property itself does not survive ``to_dict()``, so it is re-derived here
    from the two fields that do.
    """
    passed = 0
    unmeasured = 0
    for budget in budgets:
        measured = bool(budget.get("conclusive", True)) or not budget.get("passed", False)
        if not measured:
            unmeasured += 1
        elif budget.get("passed", False):
            passed += 1
    return passed, len(budgets), unmeasured


def _avg_score_delta(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Mean per-prompt score change across the run, or ``None`` with no rows."""
    deltas = [float(row["delta_avg_score"]) for row in rows if "delta_avg_score" in row]
    return sum(deltas) / len(deltas) if deltas else None


def _headline_comparison(sections: Sequence[Any]) -> Mapping[str, Any] | None:
    """The run's worst aggregate comparison — the one the panels quote.

    Ordered exactly as the narrative orders it (:mod:`evalshift_cli.insights.facts`)
    so the effect size in a hero panel and the effect size in the prose beside
    it are always the same row: most negative effect first, corrected p-value
    breaking ties.
    """
    tested: list[Mapping[str, Any]] = [
        row
        for section in sections
        for row in section.aggregate_rows
        if row.get("test") not in (None, "skipped")
    ]
    if not tested:
        return None
    return min(tested, key=lambda r: (r["effect_size"], r["p_value_corrected"]))


def _budget_limit(budgets: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    """The named budget from a decision, or ``None`` when it was not configured."""
    for budget in budgets:
        if budget.get("name") == name and budget.get("scope", "overall") == "overall":
            return budget
    return None


def render_html(report: ReportData, *, insight: Insight | None = None) -> str:
    """Return the full HTML document as a string.

    Args:
        report: The computed payload. Every figure on the page comes from here.
        insight: The machine-written narrative, when one was generated. It is
            passed separately rather than carried on ``report`` because
            ``report.json`` is the computed payload and prose is not part of
            it. Model output, so the template renders it escaped.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
        keep_trailing_newline=True,
    )
    env.globals["severity_glyph"] = lambda sev: _SEVERITY_GLYPHS.get(sev, "?")
    env.globals["verdict"] = _verdict
    env.globals["verdict_glyph"] = _verdict_glyph
    env.globals["confidence_label"] = _confidence_label
    env.globals["effect_magnitude"] = _effect_magnitude
    env.globals["small_sample_threshold"] = _SMALL_SAMPLE_THRESHOLD
    env.globals["latency"] = _latency
    env.globals["evaluator_label"] = _evaluator_label
    env.globals["axis_of"] = _axis_of
    env.globals["axis_blurb"] = _axis_blurb
    env.globals["test_label"] = _test_label
    env.globals["regression_reason"] = _regression_reason
    env.globals["tool_list"] = _tool_list
    env.globals["insight_glyph"] = lambda kind: _INSIGHT_GLYPHS.get(kind, "·")
    env.globals["budget_limit"] = _budget_limit
    env.globals["category_label"] = category_label
    env.globals["kind_label"] = _kind_label
    env.globals["budget_labels"] = BUDGET_LABELS
    env.globals["budget_meanings"] = BUDGET_MEANINGS
    template = env.get_template("report.html.j2")
    css = (_TEMPLATE_DIR / "report.css").read_text(encoding="utf-8")

    budgets = (report.migration_decision or {}).get("budget_results", [])
    budgets_passed, budgets_total, budgets_unmeasured = _budget_tally(budgets)
    role_totals = _run_role_totals(report.prompt_sections)

    return template.render(
        run_id=report.run_id,
        started_at=_display_timestamp(report.started_at),
        source_model=report.source_model,
        target_model=report.target_model,
        suite_path=report.suite_path,
        n_examples=report.n_examples,
        n_calls=report.n_calls,
        cached_calls=report.cached_calls,
        failed_calls=report.failed_calls,
        truncated_calls=report.truncated_calls,
        total_cost_usd=report.total_cost_usd,
        executive_summary=report.executive_summary,
        migration_decision=report.migration_decision,
        prompt_sections=report.prompt_sections,
        methodology_notes=report.methodology_notes,
        non_deterministic_models=report.non_deterministic_models,
        dropped_params=report.dropped_params,
        samples_per_example=report.samples_per_example,
        judge_family_overlap=report.judge_family_overlap,
        invariant_violations=report.invariant_violations,
        suite_name=_suite_name(report.suite_path),
        source_cost_usd=role_totals["source"]["cost"],
        target_cost_usd=role_totals["target"]["cost"],
        cost_delta_pct=_pct_delta(role_totals["source"]["cost"], role_totals["target"]["cost"]),
        latency_delta_pct=_run_latency_delta_pct(role_totals),
        avg_score_delta=_avg_score_delta(report.executive_summary),
        headline=_headline_comparison(report.prompt_sections),
        budgets_passed=budgets_passed,
        budgets_total=budgets_total,
        budgets_unmeasured=budgets_unmeasured,
        insight=insight,
        insight_generated_at=(
            insight.generated_at.strftime("%Y-%m-%d %H:%M UTC") if insight is not None else None
        ),
        css=css,
    )


def write_html(report: ReportData, run_dir: Path, *, insight: Insight | None = None) -> Path:
    """Write the HTML report into the run directory and return its path."""
    out_path = run_dir / REPORT_HTML_FILENAME
    out_path.write_text(render_html(report, insight=insight), encoding="utf-8")
    return out_path


__all__ = ["REPORT_HTML_FILENAME", "render_html", "write_html"]
