"""Prompt a model for the run narrative, then refuse anything it derived.

The model is handed every figure pre-rendered by :mod:`evalshift_cli.insights.facts`
and told to copy them verbatim, so :func:`validate_numbers` can hold it to a
permit-list on the way out: a numeric token that was not supplied as a fact
means the model calculated something, and in a report that gates merges a
derived number is a defect even when it happens to be right.

Three checks on the way out — invented figures, echoed identifiers, and on a
FAIL a held budget blamed for the verdict (:func:`validate_blame`). Two
attempts, then :func:`evalshift_cli.insights.templates.fallback_insight`. The
prose is never load-bearing enough to fail a run over — every headline figure on
the report is rendered from data, and this module only writes the sentences
around them.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from evalshift_cli.analysis.policy import BUDGET_LABELS
from evalshift_cli.evaluators.failures import CATEGORY_LABELS, category_label
from evalshift_cli.insights.facts import BUDGET_WITHIN, Facts, RegressionSample
from evalshift_cli.insights.models import (
    INSIGHT_KINDS,
    MAX_FINDING_DETAIL_CHARS,
    MAX_FINDING_TITLE_CHARS,
    MAX_FINDINGS,
    MAX_MODEL_CHARS,
    MAX_SUMMARY_CHARS,
    NUMERIC_TOKEN_RE,
    Insight,
    InsightFinding,
    InsightKind,
    clamp_text,
)
from evalshift_cli.insights.templates import fallback_insight
from evalshift_cli.models.client import ModelClient

log = logging.getLogger(__name__)

#: Generations attempted before the templated prose ships instead. The second
#: attempt is worth paying for only because it is told what was wrong with the
#: first; a third adds cost without adding information.
MAX_ATTEMPTS: int = 2

#: The four prose fields the server requires, in the order they are prompted for.
PROSE_FIELDS: tuple[str, ...] = (
    "verdict_summary",
    "advisory_summary",
    "economics_summary",
    "recommendation",
)

#: Leading characters stripped before a token is re-checked against the
#: permit-list. Signs only — deliberately narrower than what
#: ``facts._allowed_numbers`` strips.
#:
#: The facts insert both the rendered figure and its fully bare numeral, so a
#: model writing ``102`` out of ``+102%`` already matches exactly. What is left
#: to absorb here is sign notation: the facts render negatives with U+2212 and a
#: model will write an ASCII hyphen. Stripping ``%`` or ``$`` as well would go
#: further and be wrong — the permit-list contains every integer up to the
#: example count so "15 of 21" is writable, and that would silently admit any
#: percentage in that range. ``+7%`` is exactly the mistyped figure this rejects.
_SIGN_CHARS = "+−-"

#: Models fence JSON even when told not to. Greedy by design: it spans the
#: outermost braces of a fenced or prose-wrapped object.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

#: A budget's limit as prose names it — "budget", "policy", "their limits",
#: "the +200% cost limit": an optional the/its/their, at most two words or
#: figures (none with a comma), then a limit noun — limit, budget, ceiling,
#: threshold, maximum, allowance or policy, singular or plural. Shared by the
#: blame words ("over budget") and the holding words ("within policy"), so
#: the two can never disagree about what a limit is.
_LIMIT_PHRASE = (
    r"(?:(?:the|its|their) )?(?:[^\s,]+ ){0,2}?"
    r"(?:(?:limit|budget|ceiling|threshold|maximum|allowance)s?|polic(?:y|ies))"
)

#: Words that lay a verdict at a budget's door. A budget marked "within" that
#: shares a clause with one of these, un-negated, is a cause the model
#: invented: a shipped narrative said a run "failed to meet the budget for
#: Trace-rule violations" when that budget held and only latency broke.
#: "over"/"above"/"beyond" accuse only with a limit phrase ("over its +200%
#: limit", "over budget"): "+102% over the source" measures against the
#: source model, not against a budget.
_BLAME_RE = re.compile(
    r"\b(?:breach\w*|exceed\w*|fail\w*|violat\w*|address\w*|miss(?:ed|es|ing)?|"
    r"unmet|not met|broke|broken|"
    r"(?:over|above|beyond) " + _LIMIT_PHRASE + r")\b",
    re.IGNORECASE,
)

#: Phrases that say a budget held, measured against its limit. One after a
#: held label, in its clause, with no blame word between and not itself
#: negated, describes that label rather than accusing it: "Despite Cost
#: increase staying within its limit, the run failed on Latency increase."
#: Only limit-relative phrases count: "within", "inside", "under" or "below"
#: followed by a limit phrase ("within policy", "under their limits"), or a
#: bare "held"/"holds"/"passed" with nothing but punctuation after it to the
#: clause end ("Trace-rule violations held."). A verb or preposition alone
#: says nothing held — "remained above the limit", "held at +250%, exceeding
#: its limit", "under the target model breached its ceiling", "within hours,
#: broke its ceiling" all accuse.
_HOLDING_RE = re.compile(
    r"\b(?:(?:within|inside|under|below) " + _LIMIT_PHRASE + r"|"
    r"(?:held|holds|passed)(?=[^\w\x00]*$))\b",
    re.IGNORECASE,
)

#: Stands in for every located label in the masked clause. One character, so
#: the negation window measures what it did when labels were blanked out, and
#: neither a word character nor whitespace, so no pattern above can match it.
_MASK = "\x00"

#: A held label right after a negation — with only other labels, "or", "and",
#: "nor" and commas between — may be named to be cleared: "not Cost increase
#: or Trace-rule violations". Matched against the masked text before the label.
_NEGATED_MENTION_RE = re.compile(
    r"\b(?:not|no|never|without)\b(?:[\s,\x00]|\b(?:or|and|nor)\b)*$",
    re.IGNORECASE,
)

#: ...but only when the name stands alone. After the label, and after any
#: labels chained to it by "or"/"and"/"nor"/commas, must come a comma, the
#: clause end or a coordinator. Otherwise the label modifies the noun after it
#: and the negation is that noun's: "no Cost increase headroom left" denies the
#: headroom, not the budget's blame. Matched against the masked text after the
#: label; the chain is possessive so it cannot back off a trailing noun.
_STANDS_ALONE_RE = re.compile(
    r"(?:(?:\s*,\s*(?:(?:or|and|nor)\s+)?|\s+(?:or|and|nor)\s+)\x00)*+"
    r"(?:\s*,|\s+(?:or|and|nor|but)\b|[^\w\x00]*$)",
    re.IGNORECASE,
)

#: What may join two labels that share one holding word: "Cost increase and
#: Trace-rule violations all stayed within budget". Anything else between them
#: ("Cost increase as Latency increase stayed within") gives the holding word
#: to the nearer label.
_COORDINATION_RE = re.compile(r"(?:[\s,\x00]|\b(?:or|and|nor)\b)*", re.IGNORECASE)

#: A blame word shortly after one of these is a denial ("was not breached",
#: "no breach of"), not an accusation. Bounded so a "not" three clauses back
#: cannot excuse an accusation.
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without)\b[^.;,]{0,24}$", re.IGNORECASE)

#: Clause boundaries. A contrast word starts a new claim, so "latency was
#: exceeded while Trace-rule violations stayed within budget" blames only
#: latency. A bare "and" is deliberately not one: "exceeded latency and failed
#: to meet Trace-rule violations" is the shipped sentence, and both halves
#: accuse; nor is a bare comma. ", with" is, because the retry hint invites
#: "…exceeded its ceiling, with Cost increase within policy", and so are a
#: parenthetical ", however," and a dash.
_CLAUSE_RE = re.compile(
    r"[.;:!?]\s+|\s(?:while|but|whereas|although|though|however)\s|,\s*(?:and|but)\s"
    r"|,\s*with\s|,\s*however,?\s|\s*[\u2014\u2013]\s*",
    re.IGNORECASE,
)

#: Straight and curly quotes, folded out of labels and prose alike before the
#: blame scan matches one against the other.
_QUOTE_RE = re.compile(r"[\"'\u2018\u2019\u201c\u201d]")

_INSTRUCTION = """\
You are writing the explanation section of a model-migration report. The reader
is an engineer deciding whether to switch their production model to the target.

Rules:
- Be concrete and direct. No marketing tone, no hedging, no filler.
- EVERY figure you write must be copied verbatim from the FACTS block below.
  Never calculate, round, infer or restate a number that is not in FACTS.
- Write plain English. FACTS keys (like cost_delta_pct) address figures inside
  this prompt and must never appear in your prose — refer to each figure by
  what it means ("cost per call fell 62%", never "a cost_delta_pct of -62%").
  The same goes for any snake_case or SCREAMING_CASE identifier. Evaluator
  names listed in FACTS are the user's own and may be quoted as-is.
- A fact reading "not measured" means nothing was compared. Say so plainly.
  Never call the run equivalent, consistent, unchanged, safe or free of
  regressions on the strength of a figure that is absent — an unmeasured rate
  is unknown, not zero and not perfect.
- A gate named under "unmeasured_budgets" or "unmeasured_evaluators" was handed
  no comparable row. It did not pass; it was blind. While a "coverage_basis"
  fact is present, never write that all budgets passed, that every constraint
  or requirement is met, or that the migration is safe — name the blind gates
  and say their silence is unknown.
- Every "budget" line ends with its status. Only a budget whose status is
  "breached" — the same ones listed under "failed_budgets" — may be named as
  why the run failed or as something to fix. A budget marked "within" held:
  never write that it was breached, exceeded, failed, missed, violated or
  must be addressed. When "verdict_reason" is present it is the decision's
  own explanation of the verdict, and your verdict summary must agree with it.
- Findings describe BEHAVIORAL changes visible in the sampled outputs — what the
  target does differently from the source. They are not restatements of the
  statistics; the report already renders those.
- If the evidence does not support a finding, return fewer findings. Do not pad.
- Caps: at most {max_findings} findings, at most {max_summary} characters per
  summary and per finding detail, at most {max_title} characters per finding title.

Reply with strict JSON only, no prose outside it, matching this schema exactly:

{{
  "verdict_summary": "<why the run reached this verdict>",
  "advisory_summary": "<what the non-blocking signal shows>",
  "economics_summary": "<cost and latency, in the reader's terms>",
  "findings": [
    {{"kind": "positive" | "negative" | "warning", "title": "<short label>",
      "detail": "<what changed, with an example if one is shown below>"}}
  ],
  "recommendation": "<what to do next>"
}}
"""


class GenerationError(Exception):
    """A generation that cannot be shipped as it stands.

    Attributes:
        tokens: Offending tokens — invented figures or echoed internal
            identifiers — fed back into the retry so the second attempt can
            correct them. Empty for every other kind of rejection.
        misblamed: Display labels of budgets that held but were blamed for
            the verdict — fed back the same way, under their own hint.
    """

    def __init__(
        self,
        message: str,
        *,
        tokens: list[str] | None = None,
        misblamed: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.tokens = tokens or []
        self.misblamed = misblamed or []


#: FACTS field names that are not ``rendered`` keys but appear in the prompt
#: and were observed echoed into shipped prose. All identifier-shaped: every
#: entry contains an underscore, so none can collide with an English word.
_FACTS_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "source_model",
        "target_model",
        "worst_evaluator",
        "rates_basis",
        "coverage_basis",
        "failed_budgets",
        "verdict_reason",
        "blocking_evaluators",
        "unmeasured_budgets",
        "unmeasured_evaluators",
        "failure_category",
    }
)


def banned_identifiers(facts: Facts) -> tuple[str, ...]:
    """The internal tokens ``facts`` puts in reach of the model's prose.

    The prompt addresses figures by identifier-shaped keys, and a model told
    to copy tokens verbatim will copy those too — a shipped report read "a
    cost_delta_pct of −61.7%". The set is every underscore-carrying rendered
    key, the FACTS field names, the ``evalshift.yaml`` budget fields and the
    machine failure-category labels. User-chosen evaluator names are absent
    deliberately: they are the only handle the narrative has for those gates.
    """
    tokens = {key for key in facts.rendered if "_" in key}
    tokens |= _FACTS_FIELD_NAMES
    tokens |= set(BUDGET_LABELS)
    tokens |= set(CATEGORY_LABELS)
    return tuple(sorted(tokens))


def validate_identifiers(text: str, banned: tuple[str, ...]) -> list[str]:
    """Return the banned internal identifiers that appear in ``text``.

    Plain substring match: every banned token carries an underscore or is
    SCREAMING_CASE, so none occurs in ordinary prose by accident.
    """
    return [token for token in banned if token in text]


def validate_numbers(text: str, allowed: frozenset[str]) -> list[str]:
    """Return numeric tokens in ``text`` that were not supplied as facts.

    The model is given every figure pre-rendered and told to copy it verbatim, so
    a token outside ``allowed`` means it calculated something. In a report that
    gates merges, a derived number is a defect even when it happens to be right.

    Args:
        text: One prose field of a generation.
        allowed: ``Facts.allowed_numbers`` — every figure the narrative may
            legitimately contain, plus each one's bare-numeral form.

    Returns:
        The offending tokens in the order they appear. Empty means valid.
    """
    return [token for token in NUMERIC_TOKEN_RE.findall(text) if not _is_allowed(token, allowed)]


def validate_blame(text: str, within: Sequence[str], *, others: Sequence[str] = ()) -> list[str]:
    """Return the budgets in ``within`` that ``text`` blames for the verdict.

    A budget that held may be mentioned — "Trace-rule violations stayed within
    budget" is accurate — but not accused: it may not share a clause with a
    blame word unless that word is negated, or unless each of its mentions in
    that clause is excused. A mention is excused by a negation right before a
    name that stands alone ("not Cost increase or Trace-rule violations", but
    not "no Cost increase headroom"), or by a limit-relative holding phrase
    after it that describes it ("Cost increase stayed within its limit" beside
    "the run failed on Latency increase", but not "remained above the limit"
    or "within hours, broke its ceiling")
    — see :func:`_excused`. A plain-English accusation carries no digit for
    :func:`validate_numbers` and no identifier for :func:`validate_identifiers`,
    so this is the only check that sees it.

    A safety net, not a parser: it catches the shipped phrasing and its close
    variants. A false match costs the retry — the one retry, shared with
    invented-number rejections — and a repeated one ships the templated prose,
    which has no findings. Two misses are accepted: a holding phrase excuses
    the label before it whatever follows, so "Cost increase stayed within
    budget at first, then exceeded it" and "Address Cost increase so it stays
    within budget" pass — telling their blame word's subject from the one in
    "staying within its limit, the run failed on Latency increase" takes a
    parser. For the same reason "Cost increase below the target's ceiling
    failed the run" and "Cost increase, below the ceiling on 3 prompts, failed
    on the rest" pass. Labels are matched case-insensitively with hyphens and runs of
    whitespace folded to one space.
    Every label — held or not — is located longest first and masked before the
    scan: "'checkout' slice cost increase" contains "cost increase", and
    "Trace-rule violations" contains a blame word, and neither may accuse a
    held budget by containment.

    Args:
        text: One prose field of a generation.
        within: Display labels of the budgets whose status is
            :data:`~evalshift_cli.insights.facts.BUDGET_WITHIN`.
        others: Display labels of the gating budgets that did not hold. Never
            reported, only located: a span one of them claims cannot also
            match a shorter held label, and its words are not blame words.

    Returns:
        The wrongly blamed labels, each once, in ``within`` order. Empty means
        valid.
    """
    held = set(within)
    # Longest first, so a position a longer label covers cannot also match a
    # shorter one it contains. ``sorted`` is stable: on a tie, held labels win.
    needles = sorted(
        ((label, _normalise(label)) for label in (*within, *others)),
        key=lambda pair: -len(pair[1]),
    )
    blamed: set[str] = set()
    for clause in (_normalise(c) for c in _CLAUSE_RE.split(text)):
        spans = _locate_labels(clause, needles)
        if not any(label in held for _, _, label in spans):
            continue
        # Every located label comes out before the scan, not just the one
        # under test: "Trace-rule violations" would otherwise read as a
        # violation of whichever budget shares its clause.
        rest, cursor = "", 0
        marks: list[tuple[int, str]] = []
        for start, end, label in sorted(spans):
            rest += clause[cursor:start]
            marks.append((len(rest), label))
            rest += _MASK
            cursor = end
        rest += clause[cursor:]
        blames = [
            m.start()
            for m in _BLAME_RE.finditer(rest)
            if not _NEGATION_RE.search(rest[: m.start()])
        ]
        if not blames:
            continue
        positions = [position for position, _ in marks]
        blamed.update(
            label
            for position, label in marks
            if label in held and not _excused(rest, position, blames, positions)
        )
    return [label for label in within if label in blamed]


def _excused(rest: str, position: int, blames: list[int], marks: list[int]) -> bool:
    """Whether the masked label at ``position`` is mentioned rather than accused.

    Two ways out, judged per occurrence. A negation right before it ("not
    Cost increase or Trace-rule violations") names it to clear it — provided
    the name stands alone: it, and any labels chained to it, must be followed
    by a comma, the clause end or "or"/"and"/"nor"/"but" ("no Cost increase
    headroom left" negates the headroom and still accuses). Or a holding
    phrase after it ("Cost increase stayed within its limit") describes it —
    provided the phrase is limit-relative (see :data:`_HOLDING_RE`: within,
    inside, under or below need a limit phrase after them, so "remained above
    the limit" and "within hours" are not), it is not negated, no un-negated
    blame word
    stands between the two ("exceeded its limit and stayed within budget"
    still accuses), and no other label does unless joined by
    "and"/"or"/"nor"/a comma ("as Latency increase stayed within" describes
    latency).

    Args:
        rest: The clause with every located label replaced by :data:`_MASK`.
        position: Where the label under judgement sits in ``rest``.
        blames: Start of each un-negated blame word in ``rest``.
        marks: Where every located label sits in ``rest``, in order.
    """
    if _NEGATED_MENTION_RE.search(rest[:position]) and _STANDS_ALONE_RE.match(rest, position + 1):
        return True
    for holding in _HOLDING_RE.finditer(rest, position + 1):
        at = holding.start()
        if _NEGATION_RE.search(rest[:at]):
            continue
        if any(position < blame < at for blame in blames):
            continue
        between = [mark for mark in marks if position < mark < at]
        if between and not _COORDINATION_RE.fullmatch(rest, position + 1, between[-1]):
            continue
        return True
    return False


def _locate_labels(clause: str, needles: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    """Claim non-overlapping spans of ``clause`` for ``needles``, in the order given.

    Returns:
        One ``(start, end, label)`` per claimed span.
    """
    spans: list[tuple[int, int, str]] = []
    for label, needle in needles:
        start = clause.find(needle)
        while start != -1:
            end = start + len(needle)
            if all(end <= s or start >= e for s, e, _ in spans):
                spans.append((start, end, label))
            start = clause.find(needle, start + 1)
    return spans


def _normalise(text: str) -> str:
    """Lower-case, quotes dropped, hyphens and whitespace runs folded, for label matching.

    Quotes go because a slice label reaches the model as "'checkout' slice
    cost increase" and comes back bare or curly-quoted; left in, the held
    "cost increase" inside it would claim the span.
    """
    return re.sub(r"[-\s]+", " ", _QUOTE_RE.sub("", text.lower()))


async def generate_insight(facts: Facts, *, model: str, client: ModelClient) -> Insight:
    """Write the run's narrative, or fall back to deterministic prose.

    Args:
        facts: The pre-rendered figures and regression samples to write from.
        model: Model id for the generating call. Recorded on the result as
            provenance, so a reader can tell the prose is machine-written.
        client: The shared model client. Its errors propagate — the caller
            (``run_report``) is what decides a failed narrative is non-fatal.

    Returns:
        The generated :class:`~evalshift_cli.insights.models.Insight`, clamped to the
        server's limits, or the templated fallback after
        :data:`MAX_ATTEMPTS` unusable generations.
    """
    invented: list[str] = []
    misblamed: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = await client.complete(
            model=model,
            prompt=build_prompt(facts, invented=invented, misblamed=misblamed),
            temperature=0.0,
        )
        log.debug(
            "insights generation attempt %d: %d output tokens, $%.6f",
            attempt,
            response.output_tokens,
            response.cost_usd,
        )
        try:
            return _build_insight(response.text, facts=facts, model=model)
        except GenerationError as exc:
            log.warning("insights generation attempt %d rejected: %s", attempt, exc)
            invented = exc.tokens
            misblamed = exc.misblamed
    log.warning(
        "insights generation produced nothing usable in %d attempts; using templated prose",
        MAX_ATTEMPTS,
    )
    return fallback_insight(facts)


def build_prompt(
    facts: Facts,
    *,
    invented: list[str] | None = None,
    misblamed: list[str] | None = None,
) -> str:
    """Assemble the instruction, the FACTS block and the regression samples.

    Args:
        facts: The run's rendered figures and worst regressions.
        invented: Offending tokens from a previous attempt — derived figures
            or echoed internal identifiers. Present only on a retry, where
            naming them is the whole reason the second call is worth paying
            for.
        misblamed: Budgets that held but a previous attempt blamed for the
            verdict. Named separately from ``invented``: the label itself is
            allowed in prose, the accusation is not.
    """
    sections = [
        _INSTRUCTION.format(
            max_findings=MAX_FINDINGS,
            max_summary=MAX_SUMMARY_CHARS,
            max_title=MAX_FINDING_TITLE_CHARS,
        ),
        _facts_block(facts),
        _samples_block(facts.regression_samples),
    ]
    if invented:
        sections.append(
            "Your previous answer contained tokens that must not appear: "
            + ", ".join(dict.fromkeys(invented))
            + ". Figures must be copied verbatim from FACTS; internal "
            "identifiers must be rewritten as plain language."
        )
    if misblamed:
        # A FAIL from severity or dropped parameters breached no budget, and
        # its FACTS block has no failed_budgets line to point at.
        breached = (
            "Only the budgets under failed_budgets were breached"
            if facts.failed_budgets
            else "No budget was breached"
        )
        sections.append(
            "Your previous answer blamed budgets that held: "
            + ", ".join(dict.fromkeys(misblamed))
            + f". {breached}; a budget marked within may be described as within "
            "policy or left out, never as a cause of the verdict or as something "
            "to address."
        )
    return "\n\n".join(section for section in sections if section)


# ---------------------------------------------------------------------------
# Prompt sections
# ---------------------------------------------------------------------------


def _facts_block(facts: Facts) -> str:
    lines = [f"{key}: {value}" for key, value in facts.rendered.items()]
    lines.append(f"source_model: {facts.source_model}")
    lines.append(f"target_model: {facts.target_model}")
    lines.append(f"worst_evaluator: {facts.worst_evaluator}")
    if facts.rates_basis:
        lines.append(f"rates_basis: {facts.rates_basis}")
    if facts.blocking_evaluators:
        lines.append(f"blocking_evaluators: {', '.join(facts.blocking_evaluators)}")
    if facts.unmeasured_budgets:
        lines.append(f"unmeasured_budgets: {', '.join(facts.unmeasured_budgets)}")
    if facts.unmeasured_evaluators:
        lines.append(f"unmeasured_evaluators: {', '.join(facts.unmeasured_evaluators)}")
    if facts.coverage_basis:
        lines.append(f"coverage_basis: {facts.coverage_basis}")
    if facts.failed_budgets:
        lines.append(f"failed_budgets: {', '.join(facts.failed_budgets)}")
    if facts.verdict_reason:
        lines.append(f"verdict_reason: {facts.verdict_reason}")
    # Budgets and categories reach the model under their display names: it is
    # told to copy tokens verbatim, so whatever appears here is what a reader
    # may end up seeing in the narrative. Each budget carries its status, so
    # the cause of a FAIL is read off the line rather than guessed from the
    # limit — a narrative once blamed a budget that held.
    for label, limit, status in facts.budget_statuses:
        lines.append(f'budget "{label}": limit {limit}, {status}')
    for category, count in facts.failure_categories:
        lines.append(f'failure_category "{category_label(category)}": {count}')
    body = "\n".join(lines)
    return f"FACTS (copy these verbatim; derive nothing):\n{body}"


def _samples_block(samples: list[RegressionSample]) -> str:
    """Render the worst regressions — the only place behavior is visible."""
    if not samples:
        return (
            "SAMPLED REGRESSIONS: none — no example regressed on any evaluator. "
            "Do not invent behavioral findings; return an empty findings list."
        )
    blocks = [
        "\n".join(
            [
                f"--- {sample.example_id} (worst delta {sample.delta}) ---",
                f"input: {sample.input_text}",
                f"source output: {sample.source_output}",
                f"target output: {sample.target_output}",
            ]
        )
        for sample in samples
    ]
    return "SAMPLED REGRESSIONS (worst first):\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


def _build_insight(text: str, *, facts: Facts, model: str) -> Insight:
    """Parse, clamp and validate one generation, or raise :class:`GenerationError`.

    Clamping happens *before* validation deliberately: truncating a summary can
    cut a figure in half, and ``0.02`` sliced out of ``0.025`` is exactly the
    invented number this module exists to catch.
    """
    payload = _parse_payload(text)
    prose = _build_prose(payload)
    findings = _build_findings(payload.get("findings"))

    banned = banned_identifiers(facts)
    offending: list[str] = []
    for field in (*prose.values(), *(t for f in findings for t in (f.title, f.detail))):
        offending.extend(validate_numbers(field, facts.allowed_numbers))
        offending.extend(t for t in validate_identifiers(field, banned) if t not in offending)
    # Only a FAIL assigns a cause, and only its verdict summary and
    # recommendation do the assigning; the advisory and economics summaries
    # describe signals, and findings describe behaviour.
    misblamed: list[str] = []
    if facts.verdict == "fail":
        held = [label for label, _, status in facts.budget_statuses if status == BUDGET_WITHIN]
        others = [label for label, _, status in facts.budget_statuses if status != BUDGET_WITHIN]
        for field in (prose["verdict_summary"], prose["recommendation"]):
            misblamed.extend(
                t for t in validate_blame(field, held, others=others) if t not in misblamed
            )
    if offending or misblamed:
        problems: list[str] = []
        if offending:
            problems.append(f"tokens that may not be shown: {', '.join(offending)}")
        if misblamed:
            problems.append(f"budgets blamed although they held: {', '.join(misblamed)}")
        raise GenerationError(
            "response contained " + "; ".join(problems),
            tokens=offending,
            misblamed=misblamed,
        )

    return Insight(
        model=clamp_text(model, MAX_MODEL_CHARS),
        generated_at=datetime.now(UTC),
        verdict_summary=prose["verdict_summary"],
        advisory_summary=prose["advisory_summary"],
        economics_summary=prose["economics_summary"],
        recommendation=prose["recommendation"],
        findings=findings,
    )


def _parse_payload(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a response, fenced or not."""
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(stripped)
        if match is None:
            raise GenerationError(f"no JSON object in response: {stripped[:200]!r}") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise GenerationError(f"malformed JSON in response: {match.group(0)[:200]!r}") from exc
    if not isinstance(data, dict):
        raise GenerationError(f"response was not a JSON object: {type(data).__name__}")
    return cast(dict[str, Any], data)


def _build_prose(payload: dict[str, Any]) -> dict[str, str]:
    """Clamp the four required summaries; a missing or empty one is a failure.

    The server's ``Insights`` model is ``min_length=1`` on every prose field, so
    an empty summary is a 400 at finalize — after the bundle has already been
    uploaded. Retrying is cheaper than discovering it there.
    """
    prose: dict[str, str] = {}
    for field in PROSE_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str):
            raise GenerationError(f"{field} was missing or not a string")
        clamped = clamp_text(value, MAX_SUMMARY_CHARS)
        if not clamped:
            raise GenerationError(f"{field} was empty")
        prose[field] = clamped
    return prose


def _build_findings(raw: Any) -> list[InsightFinding]:
    """Keep the shippable findings, drop the rest, retry only if none survive.

    A single unusable finding — an icon class the server does not define, an
    empty title — is not worth paying for a second generation when the others
    stand on their own.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise GenerationError(f"findings was not a list: {type(raw).__name__}")

    findings: list[InsightFinding] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "")).strip().lower()
        title = clamp_text(str(item.get("title", "")), MAX_FINDING_TITLE_CHARS)
        detail = clamp_text(str(item.get("detail", "")), MAX_FINDING_DETAIL_CHARS)
        if kind not in INSIGHT_KINDS or not title or not detail:
            log.debug("dropping unusable finding: kind=%r title=%r", kind, title[:60])
            continue
        findings.append(InsightFinding(kind=cast(InsightKind, kind), title=title, detail=detail))

    if raw and not findings:
        raise GenerationError("every finding in the response was unusable")
    return findings[:MAX_FINDINGS]


def _is_allowed(token: str, allowed: frozenset[str]) -> bool:
    if token in allowed:
        return True
    return token.lstrip(_SIGN_CHARS).replace(",", "") in allowed


__all__ = [
    "MAX_ATTEMPTS",
    "PROSE_FIELDS",
    "GenerationError",
    "build_prompt",
    "generate_insight",
    "validate_blame",
    "validate_numbers",
]
