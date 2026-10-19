"""How fresh a committed capture suite is, for ``doctor`` and ``compare`` (spec §7).

Reads the promoted case files committed beside ``golden.jsonl`` -- the only place a case's
capture time and model live -- so a team that only runs CI still learns its suite went stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from evalshift_cli.captures.models import PromotedCase
from evalshift_cli.captures.staleness import case_instant, is_reviewed
from evalshift_cli.models.matching import describe_models, model_filter_applies, same_model
from evalshift_cli.utils.duration import format_duration

FindingKind = Literal["unreadable", "stale_age", "stale_model", "no_source_model_case", "over_cap"]


@dataclass(frozen=True, slots=True)
class FreshnessFinding:
    """One thing that is stale about a suite, worded for a doctor row or a compare line."""

    kind: FindingKind
    message: str


@dataclass(frozen=True, slots=True)
class SuiteFreshness:
    """A suite's case count, capture-time span and findings."""

    suite: str
    cases: int
    newest: datetime | None
    oldest: datetime | None
    findings: tuple[FreshnessFinding, ...]

    def describe(self, now: datetime) -> str:
        """``alpha: 200 cases, captured 2d–12d ago`` (the span left out when unknown)."""  # noqa: RUF002
        text = f"{self.suite}: {self.cases} cases"
        if self.newest is not None and self.oldest is not None:
            text += f", captured {_age(now, self.newest)}–{_age(now, self.oldest)} ago"  # noqa: RUF001 — an age span
        return text


def _age(now: datetime, then: datetime) -> str:
    return f"{max(0, (now - then).days)}d"


def suite_freshness(
    suite: str,
    suite_dir: Path,
    *,
    max_age: timedelta | None,
    max_cases: int | None,
    source_model: str | None,
    now: datetime,
) -> SuiteFreshness | None:
    """Read ``suite_dir``'s case files and report what is stale; ``None`` when there are none.

    Never raises: an unreadable case file becomes one ``unreadable`` finding and is left out
    of every count.

    Args:
        suite: The suite's name, which every finding message starts with.
        suite_dir: The directory holding the suite's promoted case files.
        max_age: ``captures.max_age``; ``None`` turns the age finding off.
        max_cases: ``captures.max_cases``; ``None`` turns the over-cap finding off.
        source_model: The model cases should have been recorded on; ``None`` turns the
            model findings off.
        now: The instant ages are measured from.

    Returns:
        The suite's freshness, or ``None`` when ``suite_dir`` holds no case files.
    """
    paths = sorted(suite_dir.glob("*.json")) if suite_dir.is_dir() else []
    cases: list[PromotedCase] = []
    unreadable = 0
    for path in paths:
        try:
            cases.append(PromotedCase.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):  # ValidationError and UnicodeDecodeError are ValueErrors
            unreadable += 1
    if not cases and not unreadable:
        return None
    instants = [i for i in (case_instant(c) for c in cases) if i is not None]
    newest = max(instants, default=None)
    oldest = min(instants, default=None)
    findings: list[FreshnessFinding] = []
    if unreadable:
        findings.append(
            FreshnessFinding(
                "unreadable",
                f"{suite}: {unreadable} case file(s) could not be read; the counts leave them out",
            )
        )
    if max_age is not None and newest is not None and newest < now - max_age:
        findings.append(
            FreshnessFinding(
                "stale_age",
                f"{suite}: newest case captured {_age(now, newest)} ago "
                f"(captures.max_age {format_duration(max_age)}); "
                "refresh with evalshift capture sync",
            )
        )
    models = [c.source_model for c in cases]
    if source_model and any(models):
        if model_filter_applies(models, source_model):
            other = sum(1 for m in models if m and not same_model(m, source_model))
            if other:
                # Sync retires a stale case only to make room under max_cases.
                advice = (
                    "evalshift capture sync replaces them as new captures arrive"
                    if max_cases is not None
                    else "set captures.max_cases for evalshift capture sync to replace them"
                )
                findings.append(
                    FreshnessFinding(
                        "stale_model",
                        f"{suite}: {other} of {len(cases)} cases recorded on another model "
                        f"than {source_model}; {advice}",
                    )
                )
        else:
            findings.append(
                FreshnessFinding(
                    "no_source_model_case",
                    f"{suite}: no case recorded on {source_model} "
                    f"(seen: {describe_models(models)}); check defaults.source_model, "
                    "or record new captures",
                )
            )
    if max_cases is not None and len(cases) > max_cases:
        reviewed = sum(1 for c in cases if is_reviewed(c.example))
        # Reviewed cases never leave, so a trim stops at them.
        trim = (
            f"evalshift capture sync trims it only down to its {reviewed} reviewed case(s)"
            if reviewed > max_cases
            else "evalshift capture sync trims it"
        )
        findings.append(
            FreshnessFinding(
                "over_cap",
                f"{suite}: {len(cases)} cases, over captures.max_cases {max_cases}; {trim}",
            )
        )
    return SuiteFreshness(
        suite=suite, cases=len(cases), newest=newest, oldest=oldest, findings=tuple(findings)
    )
