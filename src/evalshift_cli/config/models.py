"""Pydantic models for the ``evalshift.yaml`` configuration file.

The schema is the contract between EvalShift and its users: any field added
here becomes part of the public API and must be documented in
``docs/configuration.md``. Models use ``extra='forbid'`` so unknown keys in a
user's YAML fail loudly rather than being silently ignored — typos in config
are a common source of frustrating bug reports, and we'd rather surface them
immediately.

The schema closely follows §5.1 of the EvalShift MVP build plan, with one
deliberate deviation: the top-level ``evaluators`` field is a typed
:class:`EvaluatorsConfig` rather than a loose ``dict``. The typed wrapper
gives us real validation and keeps ``mypy --strict`` happy.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalshift_cli.captures.store_uri import STORE_URI_FORMS, parse_store_uri
from evalshift_cli.suite.tags import RESERVED_SLICE_NAME

# Default judge model used across the config when none is specified.
# Centralised here so a single edit propagates everywhere. Gemini's
# flash-lite-preview tier is intentionally cheap; users who want a
# stronger judge should override via `evaluators.llm_judge[*].judge_model`.
DEFAULT_JUDGE_MODEL: str = "gemini-3.1-flash-lite-preview"

_REMOVED_THRESHOLDS_MESSAGE = (
    "`thresholds` was removed: it was free-form and gated nothing. "
    "Delete it from evalshift.yaml; migration_policy is the single "
    "source of truth for gating."
)
"""Error text for a config that still carries the removed ``thresholds`` block."""

_REMOVED_SLICES_MESSAGE = (
    "`slices` was removed: it never had any effect. Slices come from example "
    "`tags` automatically (one per distinct tag, plus `all`). Delete it from "
    "evalshift.yaml; per-slice budgets go under migration_policy.slices, keyed by tag."
)
"""Error text for a config that still carries the removed top-level ``slices`` block."""


class _StrictModel(BaseModel):
    """Base for every config model: forbid extra keys, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PromptDefinition(_StrictModel):
    """A single prompt that EvalShift will evaluate.

    A prompt is sourced either inline from ``evalshift.yaml`` (``detection:
    manual``) or extracted from a Python source file by AST-walking for a
    module-level string assignment (``detection: python_string``).

    Attributes:
        id: Stable, human-readable identifier used in reports and CLI output.
        detection: How to load the prompt body — ``manual`` (use ``content``)
            or ``python_string`` (use ``path`` + ``variable``).
        content: Inline prompt body (required when ``detection='manual'``).
        path: Path to a ``.py`` file containing the prompt string (required
            when ``detection='python_string'``).
        variable: Name of the module-level variable holding the prompt string
            (required when ``detection='python_string'``).
        variables: Names of template variables (``{var}`` placeholders) the
            prompt expects to be filled in by suite examples.
        max_tokens: Optional per-prompt override of ``defaults.max_tokens``.
            Raise it for prompts whose models emit long JSON / tool arguments
            that would otherwise be truncated.
    """

    id: str = Field(min_length=1, description="Stable identifier for the prompt.")
    detection: Literal["manual", "python_string"] = Field(
        description="How the prompt body is sourced.",
    )
    content: str | None = Field(default=None, description="Inline prompt body (manual only).")
    path: str | None = Field(
        default=None,
        description="Path to a .py file (python_string only).",
    )
    variable: str | None = Field(
        default=None,
        description="Module-level variable name (python_string only).",
    )
    variables: list[str] = Field(
        default_factory=list,
        description="Template variables the prompt expects.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Override defaults.max_tokens for this prompt's model calls.",
    )

    @model_validator(mode="after")
    def _check_detection_fields(self) -> Self:
        """Enforce field combinations based on the chosen ``detection`` mode."""
        if self.detection == "manual":
            if not self.content:
                raise ValueError("'content' is required when detection='manual'")
            if self.path is not None or self.variable is not None:
                raise ValueError(
                    "'path' and 'variable' must not be set when detection='manual'",
                )
        else:  # python_string
            if not self.path or not self.variable:
                raise ValueError(
                    "'path' and 'variable' are required when detection='python_string'",
                )
            if self.content is not None:
                raise ValueError(
                    "'content' must not be set when detection='python_string'",
                )
        return self


class StructuralEvaluatorConfig(_StrictModel):
    """Configuration for a single structural evaluator.

    Three flavours are supported in the MVP:

    * ``json_schema`` — validate the output against a JSON Schema.
    * ``regex`` — check the output matches a regex.
    * ``length`` — score based on whether the output length is within bounds.

    Attributes:
        type: Which structural check to run.
        schema_path: Path to a JSON Schema file (``json_schema`` only).
        pattern: Regex pattern to match (``regex`` only).
        min_chars: Minimum acceptable length in characters (``length`` only).
        max_chars: Maximum acceptable length in characters (``length`` only).
        applies_to: Glob list of prompt IDs this evaluator applies to.
            Defaults to ``["*"]`` (every prompt).
        blocking: Whether regressions from this evaluator can fail the
            migration verdict. Advisory evaluators (``blocking: false``) still
            score and appear in reports but never gate the decision.
    """

    type: Literal["json_schema", "regex", "length"]
    schema_path: str | None = None
    pattern: str | None = None
    min_chars: int | None = Field(default=None, ge=0)
    max_chars: int | None = Field(default=None, ge=0)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True

    @model_validator(mode="after")
    def _check_type_fields(self) -> Self:
        """Ensure the fields required by the chosen ``type`` are present."""
        if self.type == "json_schema":
            if not self.schema_path:
                raise ValueError("'schema_path' is required when type='json_schema'")
        elif self.type == "regex":
            if not self.pattern:
                raise ValueError("'pattern' is required when type='regex'")
        else:  # length
            if self.min_chars is None and self.max_chars is None:
                raise ValueError(
                    "at least one of 'min_chars' or 'max_chars' is required when type='length'",
                )
            if (
                self.min_chars is not None
                and self.max_chars is not None
                and self.min_chars > self.max_chars
            ):
                raise ValueError("'min_chars' must be <= 'max_chars'")
        return self


class SemanticEvaluatorConfig(_StrictModel):
    """Configuration for the cosine-similarity semantic evaluator.

    Attributes:
        embedding_model: LiteLLM-compatible embedding model identifier.
        min_similarity: Cosine similarity below which the target output is
            flagged as a semantic regression. Defaults to 0.9 so minor
            rewording/formatting drift does not trip the flag; set to 1.0 to
            flag any deviation from byte-identical.
        applies_to: Glob list of prompt IDs this evaluator applies to.
        blocking: Whether regressions from this evaluator can fail the
            migration verdict. Advisory evaluators (``blocking: false``) still
            score and appear in reports but never gate the decision. The
            library default is ``True``; ``evalshift init`` writes ``false``
            for this evaluator, because its yardstick is the *source* output
            (drift, not correctness) — a correct answer in different words
            reads as a regression, and at the suite sizes a fresh capture
            starts with that noise would dominate the verdict. The library
            default is deliberately not flipped to match: a hand-written
            config that omits the key relies on the gate it has today, and a
            minor release must not silently turn a failing migration into a
            passing one. Revisit under a ``version: 2`` schema.
    """

    embedding_model: str = "text-embedding-3-small"
    min_similarity: float = Field(default=0.9, ge=0.0, le=1.0)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True


class LLMJudgeConfig(_StrictModel):
    """Configuration for a single pairwise LLM-as-judge evaluator.

    Attributes:
        criterion_name: Short, stable identifier surfaced in reports.
        criterion_prompt: Free-form criterion the judge will apply (e.g.
            "Which output preserves more factual detail?").
        judge_model: Model used as the judge. Defaults to
            :data:`DEFAULT_JUDGE_MODEL` — a deliberately cheap tier, so a
            first run costs little; override it for a stronger judge.
        applies_to: Glob list of prompt IDs this evaluator applies to.
        blocking: Whether regressions from this evaluator can fail the
            migration verdict. Advisory evaluators (``blocking: false``) still
            score and appear in reports but never gate the decision. The
            library default is ``True``; ``evalshift init`` writes ``false``
            here for the same reason as :class:`SemanticEvaluatorConfig` —
            judge noise on a small fresh suite would dominate the verdict —
            and the library default stays ``True`` so a hand-written config
            that omits the key keeps the gate it has today.
    """

    criterion_name: str = Field(min_length=1)
    criterion_prompt: str = Field(min_length=1)
    judge_model: str = DEFAULT_JUDGE_MODEL
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True


# ---------------------------------------------------------------------------
# v0.2 — tool-call evaluator configs
# ---------------------------------------------------------------------------


class ToolSelectionEvaluatorConfig(_StrictModel):
    """Configuration for the tool-selection evaluator.

    The evaluator answers two independent questions on every pair and writes
    one record for each. They are configured separately because a single
    switch made them mutually exclusive, and the ground-truth answer silently
    won: a suite promoted from captures marks every row ``expected_no_tools``,
    which graded both models *absolutely* and scored two models that called
    entirely different tools as a zero delta — behavioural equivalence.

    Attributes:
        name: Stable identifier surfaced in reports.
        conformance: Did each side match the recorded ground truth? Scored
            per side, so both can fail at once and the delta stays 0.
            * ``expected`` (default) — match ``example.expected_tools`` in
              order.
            * ``expected_set`` — the same comparison, order-insensitive:
              multiset recall of ``example.expected_tools`` names. Use when
              the expected calls are a parallel fan-out whose order carries
              no meaning.
            * ``off`` — do not measure conformance. Reach for this when the
              suite's ground truth is weak enough to be noise.
            An example carrying ``expected_no_tools`` is scored against
            *that* expectation instead, under either setting; an example with
            no ground truth at all is not measured and writes no row.
        divergence: Did the target do what the source did? Source is its own
            baseline at 1.0, so a target that behaves differently produces a
            negative delta — a regression.
            * ``set`` (default) — Jaccard similarity on the tool-name sets.
              The default rather than ``exact`` because reordered identical
              calls are the same behaviour and must not read as drift.
            * ``exact`` — target's tool-name sequence must equal source's.
            * ``first`` — only the first call is compared.
            * ``off`` — do not measure divergence.
        applies_to: Glob list of prompt ids this evaluator applies to.
        severity_floor: When set, the analysis layer can floor severity
            at this level for any non-``none`` result on this evaluator
            (e.g. force ``high`` whenever the security agent loses a
            tool call).
        blocking: Whether regressions from this evaluator can fail the
            migration verdict. Advisory evaluators (``blocking: false``) still
            score and appear in reports but never gate the decision.
    """

    name: str = Field(min_length=1)
    conformance: Literal["expected", "expected_set", "off"] = "expected"
    divergence: Literal["exact", "set", "first", "off"] = "set"
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True
    severity_floor: Literal["low", "medium", "high", "critical"] | None = None

    @model_validator(mode="after")
    def _at_least_one_axis(self) -> Self:
        """Both axes ``off`` configures an evaluator that measures nothing."""
        if self.conformance == "off" and self.divergence == "off":
            raise ValueError(
                f"tool_selection {self.name!r}: conformance and divergence are both 'off', "
                "so this evaluator would measure nothing. Turn one of them on or remove it.",
            )
        return self


class ToolArgumentsEvaluatorConfig(_StrictModel):
    """Configuration for the tool-arguments evaluator.

    Attributes:
        name: Stable identifier surfaced in reports.
        applies_to: Glob list of prompt ids this evaluator applies to.
        against: What the arguments are compared to. ``source`` (default,
            pre-0.11 behaviour) scores target-vs-source drift and pins
            ``source_score`` at 1.0 by construction — it measures change, not
            correctness. ``expected`` scores **both** sides against
            ``example.expected_tools[].arguments``, so a source model that
            passed a value that does not exist is scored as wrong. Each
            expectation's ``match_strategy`` decides which keys are compared:
            ``exact`` takes the union of expected and actual keys, ``subset``
            and ``contains_per_field`` compare the expected keys only.
        strategies: Per-field strategy overrides. Field name → strategy.
            Unlisted fields fall back to ``default_strategy``. The ``semantic``
            strategy needs a configured ``evaluators.semantic`` to borrow an
            embedding model from; without one it degrades to ``exact``.
        default_strategy: Strategy for fields ``strategies`` does not name.
            ``auto`` (default) dispatches on the pair of values: strings are
            compared after normalization (case, surrounding and repeated
            whitespace) and, when still unequal, graded by similarity —
            embeddings when a ``semantic`` evaluator lent us a model,
            ``difflib`` otherwise; numbers go through ``numeric``; dicts and
            lists through ``subset``; anything else through ``exact``. Set it
            to ``exact`` for byte-equality scoring, where a capitalization
            difference is a wrong value.
        optional_fields_scored: How a field present on one side and absent on
            the other is scored. ``lenient`` (default) scores it 0.5 —
            omitting an optional parameter is a real difference but not a
            wrong value. ``strict`` scores it 0.0, the pre-0.11 behaviour.
        numeric_tolerance: Relative-error tolerance for the ``numeric``
            strategy (0.05 = 5%).
        use_llm_judge_fallback: Reserved for v0.3; ignored in v0.2.
        blocking: Whether regressions from this evaluator can fail the
            migration verdict. Advisory evaluators (``blocking: false``) still
            score and appear in reports but never gate the decision.
    """

    name: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True
    against: Literal["source", "expected"] = "source"
    strategies: dict[str, Literal["exact", "subset", "numeric", "semantic", "auto"]] = Field(
        default_factory=dict,
    )
    default_strategy: Literal["exact", "subset", "numeric", "semantic", "auto"] = "auto"
    numeric_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)
    optional_fields_scored: Literal["strict", "lenient"] = "lenient"
    use_llm_judge_fallback: bool = False


class ToolTraceStructureEvaluatorConfig(_StrictModel):
    """Configuration for the trace-structure evaluator.

    Attributes:
        name: Stable identifier surfaced in reports.
        applies_to: Glob list of prompt ids this evaluator applies to.
        check_call_count: Score the number of calls.
        check_parallelism: Score whether parallel-call usage matches.
        check_refusals: Score whether refusal alignment matches; a
            refusal regression always forces ``severity_floor="high"``.
        call_count_tolerance: ``+/- N`` calls considered equivalent.
        blocking: Whether regressions from this evaluator can fail the
            migration verdict. Advisory evaluators (``blocking: false``) still
            score and appear in reports but never gate the decision.
    """

    name: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True
    check_call_count: bool = True
    check_parallelism: bool = True
    check_refusals: bool = True
    call_count_tolerance: int = Field(default=1, ge=0)


class AgentTraceEvaluatorConfig(_StrictModel):
    """Configuration for imported agent-trace comparison."""

    name: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True
    check_tool_order: bool = True
    check_arguments: bool = True
    check_missing_verification: bool = True
    verification_tools: list[str] = Field(default_factory=list)
    dangerous_tools: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace invariants — hand-written rules every in-scope trace must satisfy
# ---------------------------------------------------------------------------


class ForbiddenToolsRule(_StrictModel):
    """The model must never call any tool in ``tools``."""

    type: Literal["forbidden"]
    id: str = Field(min_length=1)
    tools: list[str] = Field(min_length=1)


class RequiredToolsRule(_StrictModel):
    """Every tool in ``tools`` must be called at least once by the model.

    The only rule that is not vacuous on a trace that never mentions its
    tools, so scope it with ``applies_to``: on a single-shot replay it sees
    the first response only.
    """

    type: Literal["required"]
    id: str = Field(min_length=1)
    tools: list[str] = Field(min_length=1)


class ToolOrderRule(_StrictModel):
    """Every call to ``after`` must be preceded by a call to ``before``.

    "Preceded" counts the calls already in the response's context (recorded
    history, and on a teacher-forced replay the recorded earlier rounds) and
    calls emitted earlier in the same response.
    """

    type: Literal["order"]
    id: str = Field(min_length=1)
    before: str = Field(min_length=1)
    after: str = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_tools(self) -> Self:
        """A tool cannot be required to precede itself."""
        if self.before == self.after:
            raise ValueError(f"order rule {self.id!r}: before and after name the same tool")
        return self


class CallCountRule(_StrictModel):
    """Between ``min_calls`` and ``max_calls`` calls to ``tool`` in the conversation.

    Both bounds count the calls already in a response's context plus the
    model's own, so a recorded history that already charged leaves the model
    one charge fewer. ``max_calls`` is checked at every call; ``min_calls`` at
    the end of the last response. Exact cardinality is both bounds equal.
    Mirrors ``min_chars``/``max_chars`` on the length evaluator: at least one
    bound is required.
    """

    type: Literal["call_count"]
    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    min_calls: int | None = Field(default=None, ge=0)
    max_calls: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        """Require at least one bound, and a non-empty range when both are set."""
        if self.min_calls is None and self.max_calls is None:
            raise ValueError(
                f"call_count rule {self.id!r}: at least one of 'min_calls' or 'max_calls' "
                "is required",
            )
        if (
            self.min_calls is not None
            and self.max_calls is not None
            and self.max_calls < self.min_calls
        ):
            raise ValueError(
                f"call_count rule {self.id!r}: max_calls ({self.max_calls}) is below "
                f"min_calls ({self.min_calls})",
            )
        return self


class ArgumentsRule(_StrictModel):
    """Every call to ``tool`` must have arguments valid under ``json_schema``.

    JSON Schema Draft 2020-12, checked at config load so a typo in the schema
    fails ``evalshift.yaml`` rather than every call it was meant to guard.
    """

    type: Literal["arguments"]
    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    json_schema: dict[str, Any]

    @model_validator(mode="after")
    def _schema_is_valid(self) -> Self:
        """Reject a schema that is not itself valid Draft 2020-12."""
        try:
            Draft202012Validator.check_schema(self.json_schema)
        except SchemaError as exc:
            raise ValueError(
                f"arguments rule {self.id!r}: json_schema is not valid JSON Schema: {exc.message}",
            ) from exc
        return self


InvariantRule = Annotated[
    ForbiddenToolsRule | RequiredToolsRule | ToolOrderRule | CallCountRule | ArgumentsRule,
    Field(discriminator="type"),
]
"""One hand-written trace rule; ``type`` selects which."""


class TraceInvariantsEvaluatorConfig(_StrictModel):
    """Configuration for the trace-invariants evaluator.

    Deterministic rules every in-scope trace must satisfy, checked on both
    sides of every pair independently of each other: a rule the source also
    broke is still broken. ``migration_policy.max_invariant_violations``
    turns a target violation into a failed verdict.

    Attributes:
        name: Stable identifier surfaced in reports.
        applies_to: Glob list of prompt ids this evaluator applies to. Unlike
            the other families (where the field is not yet enforced), a prompt
            outside these globs is never checked and never counted.
        blocking: Whether violations can fail the migration verdict.
        traces: ``replayed`` (default) checks the traces ``evalshift run``
            replayed; ``imported`` checks traces brought in with
            ``evalshift traces import``.
        owner: Free-text owner of these rules (a team or handle), echoed onto
            every record and the report so a violation names who to ask.
        rules: The rules, at least one, ids unique within this evaluator.
    """

    name: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=lambda: ["*"])
    blocking: bool = True
    traces: Literal["replayed", "imported"] = "replayed"
    owner: str | None = None
    rules: list[InvariantRule] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> Self:
        """Rule ids name violations in reports, so they must not collide."""
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"trace_invariants {self.name!r}: duplicate rule id {rule.id!r}")
            seen.add(rule.id)
        return self


class EvaluatorsConfig(_StrictModel):
    """Container for all evaluator configurations attached to a run."""

    structural: list[StructuralEvaluatorConfig] = Field(default_factory=list)
    semantic: SemanticEvaluatorConfig | None = None
    llm_judge: list[LLMJudgeConfig] = Field(default_factory=list)
    # v0.2 — tool-call evaluators (all optional; v0.1 configs unaffected).
    tool_selection: list[ToolSelectionEvaluatorConfig] = Field(default_factory=list)
    tool_arguments: list[ToolArgumentsEvaluatorConfig] = Field(default_factory=list)
    tool_trace_structure: list[ToolTraceStructureEvaluatorConfig] = Field(
        default_factory=list,
    )
    # CLI Phase 2 — imported bring-your-own-agent trace evaluators.
    agent_trace: list[AgentTraceEvaluatorConfig] = Field(default_factory=list)
    # Hand-written trace rules, checked on both sides regardless of the source.
    trace_invariants: list[TraceInvariantsEvaluatorConfig] = Field(default_factory=list)

    @property
    def tool_evaluator_names(self) -> frozenset[str]:
        """Names of the evaluators that score tool-call correctness.

        Single definition on purpose: both the local report and the hosted
        bundle derive each example's ``tool_match`` flag from this set, and a
        second copy of the family list would let the two drift.
        """
        return frozenset(
            evaluator.name
            for family in (
                self.tool_selection,
                self.tool_arguments,
                self.tool_trace_structure,
            )
            for evaluator in family
        )


class SliceMigrationPolicy(_StrictModel):
    """Per-slice migration budget overrides.

    All ratio fields use decimal fractions: ``0.03`` means 3%.
    ``None`` means inherit the top-level migration policy value.

    ``max_overall_regression_rate``, ``min_equivalence_rate``,
    ``max_tool_argument_drift`` and ``max_tool_divergence`` are true rates
    bounded 0-1. The two *increase*
    budgets (``max_cost_increase`` / ``max_latency_increase``) may exceed 1.0 -
    a small-to-large model migration can be 200%+ slower or costlier - so they
    are capped at ``10.0`` (1000%) only to reject obvious typos.

    ``tool_argument_drift_floor`` is not a budget but the materiality threshold
    the drift *rate* is counted at — see :class:`MigrationPolicy`.
    """

    max_overall_regression_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_critical_regressions: int | None = Field(default=None, ge=0)
    min_equivalence_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tool_argument_drift: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tool_divergence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_invariant_violations: int | None = Field(default=None, ge=0)
    tool_argument_drift_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    max_cost_increase: float | None = Field(default=None, ge=0.0, le=10.0)
    max_latency_increase: float | None = Field(default=None, ge=0.0, le=10.0)


class MigrationPolicy(_StrictModel):
    """Regression-budget policy for deciding whether a migration is safe.

    Ratio fields use decimal fractions, not percent strings. For example,
    ``max_cost_increase=0.30`` means the target may cost up to 30% more
    than the source before the policy fails.

    The regression/equivalence rates (``max_overall_regression_rate``,
    ``min_equivalence_rate``, ``max_tool_argument_drift``,
    ``max_tool_divergence``) are true rates
    bounded 0-1. The two *increase* budgets (``max_cost_increase`` /
    ``max_latency_increase``) may exceed 1.0 - migrating a small model to a
    much larger one can be 200%+ slower or costlier (``2.0``) - so they are
    capped at ``10.0`` (1000%) only to reject obvious typos.

    Attributes:
        tool_argument_drift_floor: Target argument score at or above which a
            call is *not* counted toward ``max_tool_argument_drift``. Argument
            scoring is continuous - a reworded search query or an omitted
            optional filter scores below 1.0 without being wrong. Without a
            floor the drift rate counts every non-identical call, so a call that
            merely reworded a query would eat the drift budget.
        fail_on_dropped_params: Fail the run when either arm could not honour
            a generation parameter the source capture recorded (``state.json``
            → ``dropped_params``). Off by default: a dropped constraint is a
            caveat on the numbers, which the report states in its own banner,
            not by itself proof the migration is unsafe. Turn it on when the
            constraint *is* the contract — a suite replaying captures that
            pinned ``response_format`` measures nothing useful against a
            target that will not produce structured output, so passing on
            those scores would be worse than failing. Not available per slice:
            a model either accepts a parameter or does not, which is not a
            property any subset of examples can vary.
        max_invariant_violations: Most examples on which the target may break
            a blocking ``trace_invariants`` rule before the policy fails. A
            count, not a rate; ``0`` (default) fails on the first violation.
    """

    max_overall_regression_rate: float = Field(default=0.30, ge=0.0, le=1.0)
    max_critical_regressions: int = Field(default=1, ge=0)
    # ``min_equivalence_rate`` is a floor on the *non-regression* rate: a record
    # that is equivalent OR improved counts as passing. Improvements never fail a
    # migration (the downside is bounded by ``max_overall_regression_rate``), so
    # a target that beats the source can still meet the default 0.75 floor.
    min_equivalence_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    max_tool_argument_drift: float = Field(default=0.20, ge=0.0, le=1.0)
    # Share of ``tool_selection.divergence`` rows on which the target called
    # different tools than the source. A sibling of the drift budget above and
    # deliberately not given a materiality floor of its own: an argument score
    # slides continuously (a reworded query is not a wrong call), whereas a
    # divergence score below 1.0 means the target called a tool the source did
    # not, or skipped one it did — a behavioural difference at any magnitude.
    max_tool_divergence: float = Field(default=0.20, ge=0.0, le=1.0)
    # Examples on which the target broke a blocking trace rule. A count, not a
    # rate, and zero by default: a rule is an assertion the team wrote down,
    # so a single break is a finding rather than noise to budget for.
    max_invariant_violations: int = Field(default=0, ge=0)
    # Argument scoring is continuous: a reworded search query or an omitted
    # optional filter lands below 1.0 without being wrong. Counting every
    # negative delta would weigh a 0.98 the same as a 0.0 and burn the drift
    # budget above on calls that were never wrong. 0.9 mirrors the
    # semantic evaluator's ``min_similarity`` — the same "close enough" line
    # on the same kind of score. Deliberately above 0.5, the score an omitted
    # optional argument earns: omitting a filter changes which rows the tool
    # returns, so it is drift worth counting, not a near miss.
    tool_argument_drift_floor: float = Field(default=0.9, ge=0.0, le=1.0)
    max_cost_increase: float = Field(default=0.30, ge=0.0, le=10.0)
    max_latency_increase: float = Field(default=0.30, ge=0.0, le=10.0)
    fail_on_dropped_params: bool = False
    slices: dict[str, SliceMigrationPolicy] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_reserved_slice_override(self) -> Self:
        """The override key is a slice name, so it carries the same reservation."""
        if RESERVED_SLICE_NAME in self.slices:
            raise ValueError(
                f"migration_policy.slices key {RESERVED_SLICE_NAME!r} is reserved: the "
                "run-level budgets are the top-level fields, not a slice override",
            )
        return self


class SuiteEvaluatorsOverride(_StrictModel):
    """Per-suite evaluator block — what a single suite is scored with.

    A project's suites are rarely homogeneous: one calls tools, six answer in
    prose. Scoring them all with one top-level ``evaluators:`` block either
    leaves the tool-calling suite unmeasured or hands the tool-free ones
    evaluators with an empty denominator, which reads as an inconclusive gate
    rather than as "not applicable here".

    Field-for-field the same families as :class:`EvaluatorsConfig`, but every
    one is optional so the block can say *nothing* about a family it does not
    care about. Resolution is **family-level replacement**, applied by
    :meth:`EvalShiftConfig.evaluators_for`:

    * A family the block does not mention is inherited from the top level.
    * A family the block mentions replaces the top-level one wholesale —
      including when the value is ``[]`` or ``null``, which is how a suite
      *removes* a family it inherited. There is no deep merge and no
      per-evaluator-name merge: a suite's list is the suite's list.

    "Mentioned" means present in the YAML, which is why resolution reads
    ``model_fields_set`` rather than comparing against ``None`` — ``null`` and
    "absent" are different instructions and pydantic parses both to ``None``.
    """

    structural: list[StructuralEvaluatorConfig] | None = None
    semantic: SemanticEvaluatorConfig | None = None
    llm_judge: list[LLMJudgeConfig] | None = None
    tool_selection: list[ToolSelectionEvaluatorConfig] | None = None
    tool_arguments: list[ToolArgumentsEvaluatorConfig] | None = None
    tool_trace_structure: list[ToolTraceStructureEvaluatorConfig] | None = None
    agent_trace: list[AgentTraceEvaluatorConfig] | None = None
    trace_invariants: list[TraceInvariantsEvaluatorConfig] | None = None


class SuiteSource(_StrictModel):
    """A named suite the run can resolve via ``run --suite-name <name>``.

    Bridges the capture lifecycle: ``capture promote`` writes a golden suite to
    disk, and a ``suites`` entry records where it lives so the suite path need
    not be retyped on every ``run``.

    Attributes:
        source: ``captured`` (a suite built by ``capture promote``) or
            ``jsonl`` (a hand-authored golden JSONL). The value is advisory
            today — both resolve to ``path`` — but it documents provenance and
            lets future tooling treat captured suites specially.
        path: Path to the suite JSONL, relative to the config file's directory.
        evaluators: Evaluators this suite is scored with, replacing the
            top-level block family by family. ``None`` (the default) scores
            the suite with the top-level ``evaluators:`` exactly as before.
        managed: Whether ``capture sync`` owns this entry. ``True`` (the
            default) lets sync regenerate the entry — including its
            ``evaluators`` block — from what the suite's captures contain.
            Set it to ``False`` to freeze hand edits; sync then prints what it
            would have written instead of writing it.
    """

    source: Literal["captured", "jsonl"] = "captured"
    path: str = Field(min_length=1)
    evaluators: SuiteEvaluatorsOverride | None = None
    managed: bool = True


class Defaults(_StrictModel):
    """Top-level defaults applied across the run.

    Attributes:
        source_model: Default ``--from`` model. Overridable on the CLI.
        target_model: Default ``--to`` model. Overridable on the CLI.
        judge_model: Default LLM-as-judge model.
        insights_model: Model that writes the run narrative rendered in
            ``report.html`` and uploaded with the bundle. Falls back to
            ``judge_model`` when unset — writing analytical prose is a harder
            task than a pairwise A/B verdict, so it is worth tuning separately
            from the per-example judging cost. One call per run; skip it with
            ``--no-insights``.
        concurrency: Maximum in-flight LLM calls.
        cache: Whether to read/write the local SQLite cache.
        max_cost_usd: Soft per-run cost ceiling reserved for future
            enforcement; the orchestrator does not yet abort on it. The
            pre-flight confirmation prompt triggers above $10 regardless.
        max_tokens: Completion length cap sent to every model call. Raise it if
            outputs are being truncated (``finish_reason == "length"``);
            truncated calls are excluded from regression stats. A
            ``prompts[].max_tokens`` entry overrides this per prompt.
        samples_per_example: How many times each ``(prompt, example)`` is
            dispatched to *each* model. Default ``1``: one call per side.
            Above ``1`` every sample is a separate live call (the cache keys
            on the sample index), each sample pair is scored on its own, and
            the per-example row in ``scores.jsonl`` becomes the mean over
            samples with the within-pair spread recorded under
            ``metadata.samples`` — so the statistical ``n`` stays the number
            of examples and repeated sampling never inflates power. Only
            worth turning on when a model samples non-deterministically;
            cost and call count multiply by it.
    """

    source_model: str | None = None
    target_model: str | None = None
    judge_model: str = DEFAULT_JUDGE_MODEL
    insights_model: str | None = None
    concurrency: int = Field(default=10, gt=0, le=64)
    cache: bool = True
    max_cost_usd: float = Field(default=50.0, ge=0.0)
    max_tokens: int = Field(default=4096, gt=0)
    samples_per_example: int = Field(default=1, ge=1, le=20)


class Retention(_StrictModel):
    """Retention policy for `.evalshift/runs/` — keeps run history from growing without bound.

    Every ``run`` / ``all`` invocation creates a fresh ``r_<date>_<suite>_<hex>`` directory, so
    without a cap they accumulate indefinitely. Pruning is grouped per suite (parsed from the run
    id), ordered by directory mtime, and never touches an ``in_progress`` run or the run that just
    finished. ``EVALSHIFT_MAX_RUNS`` overrides ``max_runs_per_suite`` at load time.

    Attributes:
        max_runs_per_suite: Keep at most this many completed run dirs per suite; oldest evicted.
            ``0`` (or a negative value) disables count-based pruning entirely.
        run_ttl_days: Also evict run dirs older than this many days. ``None`` disables age pruning.
    """

    max_runs_per_suite: int = Field(default=20, ge=0)
    run_ttl_days: int | None = Field(default=None, ge=1)


class CapturesConfig(_StrictModel):
    """Where the capture SDK's recordings live, beyond the local ``.evalshift/`` directory.

    By default captures are read from ``<base>/captures/`` on this machine and nothing here
    is needed. Production hosts whose disk does not outlive them (Fargate, Lambda, pods) can
    ship captures to an object store instead (``EVALSHIFT_SINK`` on the SDK side); naming the
    same store here makes ``capture sync`` and ``capture list`` mirror new captures into the
    local directory first, with no flags.

    Attributes:
        store: ``s3://<bucket>/<prefix>``, ``gs://<bucket>/<prefix>`` or
            ``az://<account>/<container>/<prefix>``. Credentials come from the provider's
            default chain, never from this file; a URI carrying ``@`` or ``?`` is rejected.
            ``None`` (the default) means local disk only.
    """

    store: str | None = None

    @field_validator("store")
    @classmethod
    def _validate_store(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parse_store_uri(value)
        except ValueError as exc:
            # The parser's ``@`` / ``?`` messages deliberately omit both the URI (a secret
            # would sit there) and the accepted forms; a config error must still name them.
            message = str(exc)
            if "accepted forms" not in message:
                message = f"{message}; accepted forms: {STORE_URI_FORMS}"
            raise ValueError(message) from None
        return value.strip()


class EvalShiftConfig(_StrictModel):
    """Top-level configuration loaded from ``evalshift.yaml``."""

    version: Literal[1] = 1
    project: str | None = Field(default=None, pattern=r"^[a-z0-9-]+/[a-z0-9-]+$")
    prompts: list[PromptDefinition] = Field(min_length=1)
    defaults: Defaults = Field(default_factory=Defaults)
    evaluators: EvaluatorsConfig = Field(default_factory=EvaluatorsConfig)
    migration_policy: MigrationPolicy | None = None
    # Named suites (e.g. promoted captures) resolvable via `run --suite-name`.
    # Empty by default so every pre-existing config stays valid.
    suites: dict[str, SuiteSource] = Field(default_factory=dict)
    retention: Retention = Field(default_factory=Retention)
    captures: CapturesConfig = Field(default_factory=CapturesConfig)

    def evaluators_for(self, suite_name: str | None) -> EvaluatorsConfig:
        """Resolve the evaluator set a given named suite is scored with.

        The single resolution point for per-suite evaluators — evaluate,
        report and the hosted bundle all route through it, for the same
        reason :attr:`EvaluatorsConfig.tool_evaluator_names` is defined once:
        a second copy of the merge rule would let the scored set and the
        reported set drift.

        Merging is **family-level replacement**, never a deep merge. A family
        the suite's block mentions replaces the top-level one wholesale — a
        mentioned ``[]`` or ``null`` removes the family for that suite. A
        family it does not mention is inherited unchanged.

        Args:
            suite_name: The ``suites:`` key the run was launched with.
                ``None`` (a raw ``--suite <path>`` run) or a name with no
                entry resolves to the top-level block, so nothing that
                predates per-suite evaluators changes behaviour.

        Returns:
            The effective :class:`EvaluatorsConfig`. When the suite declares
            no overrides this is the top-level instance itself, not a copy.
        """
        suite = self.suites.get(suite_name) if suite_name is not None else None
        override = suite.evaluators if suite is not None else None
        if override is None or not override.model_fields_set:
            return self.evaluators

        update: dict[str, Any] = {}
        for family in override.model_fields_set:
            value = getattr(override, family)
            if value is None:
                # An explicit ``null`` means "remove", which is spelled by
                # whatever the top-level model calls empty for that family.
                value = EvaluatorsConfig.model_fields[family].get_default(
                    call_default_factory=True,
                )
            update[family] = value
        return self.evaluators.model_copy(update=update)

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, data: Any) -> Any:
        """Name the fields that were removed instead of calling them typos.

        ``extra="forbid"`` already rejects ``thresholds`` and ``slices``, but
        it says "Extra inputs are not permitted" — which reads as a
        misspelling and sends the reader hunting for the correct name of a
        field that is gone. Runs before validation because a forbidden extra
        never reaches an ``after`` validator.
        """
        if isinstance(data, dict):
            # Both at once, so deleting one does not uncover the other next run.
            messages = [
                message
                for key, message in (
                    ("thresholds", _REMOVED_THRESHOLDS_MESSAGE),
                    ("slices", _REMOVED_SLICES_MESSAGE),
                )
                if key in data
            ]
            if messages:
                raise ValueError(" ".join(messages))
        return data

    @model_validator(mode="after")
    def _check_unique_prompt_ids(self) -> Self:
        """Reject configs that declare the same prompt id twice."""
        ids = [p.id for p in self.prompts]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate prompt ids: {duplicates}")
        return self


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "AgentTraceEvaluatorConfig",
    "ArgumentsRule",
    "CallCountRule",
    "Defaults",
    "EvalShiftConfig",
    "EvaluatorsConfig",
    "ForbiddenToolsRule",
    "InvariantRule",
    "LLMJudgeConfig",
    "PromptDefinition",
    "RequiredToolsRule",
    "SemanticEvaluatorConfig",
    "StructuralEvaluatorConfig",
    "SuiteEvaluatorsOverride",
    "SuiteSource",
    "ToolArgumentsEvaluatorConfig",
    "ToolOrderRule",
    "ToolSelectionEvaluatorConfig",
    "ToolTraceStructureEvaluatorConfig",
    "TraceInvariantsEvaluatorConfig",
]
