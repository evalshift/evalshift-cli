"""Unit tests for the pydantic models in :mod:`evalshift_cli.config.models`.

These tests are the primary contract test for ``evalshift.yaml``: any
change to validation behaviour should land here first.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalshift_cli.config.models import (
    DEFAULT_JUDGE_MODEL,
    AgentTraceEvaluatorConfig,
    ArgumentsRule,
    CallCountRule,
    Defaults,
    EvalShiftConfig,
    EvaluatorsConfig,
    LLMJudgeConfig,
    MigrationPolicy,
    PromptDefinition,
    SemanticEvaluatorConfig,
    SliceMigrationPolicy,
    StructuralEvaluatorConfig,
    ToolArgumentsEvaluatorConfig,
    ToolOrderRule,
    ToolSelectionEvaluatorConfig,
    TraceInvariantsEvaluatorConfig,
)

# ---------------------------------------------------------------------------
# PromptDefinition
# ---------------------------------------------------------------------------


class TestPromptDefinition:
    def test_manual_with_content_is_valid(self) -> None:
        p = PromptDefinition(id="cs", detection="manual", content="hello {name}")
        assert p.detection == "manual"
        assert p.content == "hello {name}"
        assert p.path is None
        assert p.variable is None
        assert p.variables == []
        assert p.max_tokens is None

    def test_max_tokens_override_is_accepted(self) -> None:
        p = PromptDefinition(id="cs", detection="manual", content="hi", max_tokens=8192)
        assert p.max_tokens == 8192

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PromptDefinition(id="cs", detection="manual", content="hi", max_tokens=0)

    def test_unknown_field_still_rejected(self) -> None:
        # extra="forbid" must keep rejecting typos even with the new field.
        with pytest.raises(ValidationError):
            PromptDefinition(id="cs", detection="manual", content="hi", max_tokenz=8192)

    def test_manual_without_content_fails(self) -> None:
        with pytest.raises(ValidationError, match="'content' is required"):
            PromptDefinition(id="cs", detection="manual")

    def test_manual_with_path_fails(self) -> None:
        with pytest.raises(ValidationError, match="must not be set when detection='manual'"):
            PromptDefinition(
                id="cs",
                detection="manual",
                content="hi",
                path="src/prompts.py",
            )

    def test_python_string_with_path_and_variable_is_valid(self) -> None:
        p = PromptDefinition(
            id="cs",
            detection="python_string",
            path="src/prompts.py",
            variable="MY_PROMPT",
            variables=["a", "b"],
        )
        assert p.path == "src/prompts.py"
        assert p.variable == "MY_PROMPT"
        assert p.variables == ["a", "b"]

    def test_python_string_missing_path_fails(self) -> None:
        with pytest.raises(ValidationError, match="'path' and 'variable' are required"):
            PromptDefinition(
                id="cs",
                detection="python_string",
                variable="MY_PROMPT",
            )

    def test_python_string_missing_variable_fails(self) -> None:
        with pytest.raises(ValidationError, match="'path' and 'variable' are required"):
            PromptDefinition(
                id="cs",
                detection="python_string",
                path="src/prompts.py",
            )

    def test_python_string_with_content_fails(self) -> None:
        with pytest.raises(ValidationError, match="'content' must not be set"):
            PromptDefinition(
                id="cs",
                detection="python_string",
                path="src/prompts.py",
                variable="X",
                content="leaked",
            )

    def test_invalid_detection_value_fails(self) -> None:
        with pytest.raises(ValidationError):
            PromptDefinition(id="cs", detection="auto", content="hi")  # type: ignore[arg-type]

    def test_empty_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            PromptDefinition(id="", detection="manual", content="hi")

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PromptDefinition.model_validate(
                {
                    "id": "cs",
                    "detection": "manual",
                    "content": "hi",
                    "typo": "oops",
                },
            )


# ---------------------------------------------------------------------------
# StructuralEvaluatorConfig
# ---------------------------------------------------------------------------


class TestStructuralEvaluatorConfig:
    def test_json_schema_requires_schema_path(self) -> None:
        with pytest.raises(ValidationError, match="'schema_path' is required"):
            StructuralEvaluatorConfig(type="json_schema")

    def test_json_schema_with_schema_path_is_valid(self) -> None:
        c = StructuralEvaluatorConfig(type="json_schema", schema_path="schema.json")
        assert c.applies_to == ["*"]

    def test_regex_requires_pattern(self) -> None:
        with pytest.raises(ValidationError, match="'pattern' is required"):
            StructuralEvaluatorConfig(type="regex")

    def test_length_requires_at_least_one_bound(self) -> None:
        with pytest.raises(ValidationError, match="at least one of 'min_chars' or 'max_chars'"):
            StructuralEvaluatorConfig(type="length")

    def test_length_with_min_only_is_valid(self) -> None:
        c = StructuralEvaluatorConfig(type="length", min_chars=10)
        assert c.min_chars == 10
        assert c.max_chars is None

    def test_length_min_greater_than_max_fails(self) -> None:
        with pytest.raises(ValidationError, match="'min_chars' must be <= 'max_chars'"):
            StructuralEvaluatorConfig(type="length", min_chars=100, max_chars=10)

    def test_negative_length_bounds_fail(self) -> None:
        with pytest.raises(ValidationError):
            StructuralEvaluatorConfig(type="length", min_chars=-1)

    def test_applies_to_override(self) -> None:
        c = StructuralEvaluatorConfig(type="regex", pattern=r"^\d+$", applies_to=["cs-*"])
        assert c.applies_to == ["cs-*"]


# ---------------------------------------------------------------------------
# SemanticEvaluatorConfig & LLMJudgeConfig
# ---------------------------------------------------------------------------


class TestSemanticEvaluatorConfig:
    def test_defaults(self) -> None:
        c = SemanticEvaluatorConfig()
        assert c.embedding_model == "text-embedding-3-small"
        assert c.applies_to == ["*"]


class TestEvaluatorBlockingFlag:
    def test_blocking_defaults_to_true_everywhere(self) -> None:
        assert SemanticEvaluatorConfig().blocking is True
        assert LLMJudgeConfig(criterion_name="eq", criterion_prompt="x").blocking is True
        assert StructuralEvaluatorConfig(type="length", min_chars=1).blocking is True
        assert AgentTraceEvaluatorConfig(name="latency").blocking is True

    def test_blocking_false_parses(self) -> None:
        c = SemanticEvaluatorConfig(blocking=False)
        assert c.blocking is False
        j = LLMJudgeConfig(criterion_name="eq", criterion_prompt="x", blocking=False)
        assert j.blocking is False

    def test_unknown_field_still_rejected_alongside_blocking(self) -> None:
        with pytest.raises(ValidationError):
            SemanticEvaluatorConfig(blocking=False, blockign=True)


class TestLLMJudgeConfig:
    def test_minimum_valid(self) -> None:
        c = LLMJudgeConfig(
            criterion_name="factuality",
            criterion_prompt="Which output preserves more factual detail?",
        )
        assert c.judge_model == DEFAULT_JUDGE_MODEL

    def test_empty_criterion_name_fails(self) -> None:
        with pytest.raises(ValidationError):
            LLMJudgeConfig(criterion_name="", criterion_prompt="x")

    def test_empty_criterion_prompt_fails(self) -> None:
        with pytest.raises(ValidationError):
            LLMJudgeConfig(criterion_name="x", criterion_prompt="")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_factory_defaults(self) -> None:
        d = Defaults()
        assert d.source_model is None
        assert d.target_model is None
        assert d.judge_model == DEFAULT_JUDGE_MODEL
        assert d.concurrency == 10
        assert d.cache is True
        assert d.max_cost_usd == 50.0
        assert d.max_tokens == 4096

    @pytest.mark.parametrize("bad_concurrency", [0, -1, 65, 1000])
    def test_concurrency_bounds(self, bad_concurrency: int) -> None:
        with pytest.raises(ValidationError):
            Defaults(concurrency=bad_concurrency)

    def test_negative_max_cost_fails(self) -> None:
        with pytest.raises(ValidationError):
            Defaults(max_cost_usd=-1.0)

    @pytest.mark.parametrize("bad_max_tokens", [0, -1])
    def test_max_tokens_must_be_positive(self, bad_max_tokens: int) -> None:
        with pytest.raises(ValidationError):
            Defaults(max_tokens=bad_max_tokens)


# ---------------------------------------------------------------------------
# MigrationPolicy
# ---------------------------------------------------------------------------


class TestMigrationPolicy:
    def test_defaults_are_ci_friendly(self) -> None:
        policy = MigrationPolicy()
        assert policy.max_overall_regression_rate == pytest.approx(0.30)
        assert policy.max_critical_regressions == 1
        assert policy.min_equivalence_rate == pytest.approx(0.75)
        assert policy.max_tool_argument_drift == pytest.approx(0.20)
        assert policy.tool_argument_drift_floor == pytest.approx(0.9)
        assert policy.max_cost_increase == pytest.approx(0.30)
        assert policy.max_latency_increase == pytest.approx(0.30)
        # Off by default: a dropped constraint is a caveat on the numbers,
        # not by itself a reason to block a migration.
        assert policy.fail_on_dropped_params is False
        assert policy.slices == {}

    @pytest.mark.parametrize(
        "field",
        [
            "max_overall_regression_rate",
            "min_equivalence_rate",
            "max_tool_argument_drift",
            "tool_argument_drift_floor",
        ],
    )
    def test_rate_fields_reject_values_above_one(self, field: str) -> None:
        with pytest.raises(ValidationError):
            MigrationPolicy.model_validate({field: 1.5})

    @pytest.mark.parametrize("field", ["max_cost_increase", "max_latency_increase"])
    def test_increase_fields_accept_values_above_one(self, field: str) -> None:
        # A small→large migration can be 200%+ slower/costlier.
        policy = MigrationPolicy.model_validate({field: 2.0})
        assert getattr(policy, field) == pytest.approx(2.0)

    @pytest.mark.parametrize("field", ["max_cost_increase", "max_latency_increase"])
    def test_increase_fields_reject_out_of_bounds(self, field: str) -> None:
        with pytest.raises(ValidationError):
            MigrationPolicy.model_validate({field: 10.5})  # above the 10.0 cap
        with pytest.raises(ValidationError):
            MigrationPolicy.model_validate({field: -0.1})  # below zero

    def test_slice_overrides_are_validated(self) -> None:
        policy = MigrationPolicy(
            slices={
                "billing": SliceMigrationPolicy(
                    max_overall_regression_rate=0.0,
                    max_critical_regressions=0,
                    min_equivalence_rate=1.0,
                ),
            },
        )

        assert policy.slices["billing"].min_equivalence_rate == pytest.approx(1.0)

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MigrationPolicy.model_validate({"typo": 0.1})


# ---------------------------------------------------------------------------
# SliceMigrationPolicy
# ---------------------------------------------------------------------------


class TestSliceMigrationPolicy:
    @pytest.mark.parametrize("field", ["max_cost_increase", "max_latency_increase"])
    def test_increase_fields_accept_values_above_one(self, field: str) -> None:
        # Per-slice overrides share the top-level 10.0 (1000%) cap.
        policy = SliceMigrationPolicy.model_validate({field: 2.0})
        assert getattr(policy, field) == pytest.approx(2.0)

    @pytest.mark.parametrize("field", ["max_cost_increase", "max_latency_increase"])
    def test_increase_fields_reject_out_of_bounds(self, field: str) -> None:
        with pytest.raises(ValidationError):
            SliceMigrationPolicy.model_validate({field: 10.5})  # above the 10.0 cap
        with pytest.raises(ValidationError):
            SliceMigrationPolicy.model_validate({field: -0.1})  # below zero


# ---------------------------------------------------------------------------
# AgentTraceEvaluatorConfig
# ---------------------------------------------------------------------------


class TestToolSelectionEvaluatorConfig:
    def test_defaults_grade_ground_truth_and_compare_to_source(self) -> None:
        cfg = ToolSelectionEvaluatorConfig(name="routing")

        assert cfg.conformance == "expected"
        # Jaccard rather than `exact`: reordered identical calls are the same
        # behaviour and must not read as drift.
        assert cfg.divergence == "set"

    def test_the_two_axes_are_set_independently(self) -> None:
        cfg = ToolSelectionEvaluatorConfig(
            name="routing",
            conformance="expected_set",
            divergence="first",
        )

        assert (cfg.conformance, cfg.divergence) == ("expected_set", "first")

    @pytest.mark.parametrize(
        ("conformance", "divergence"),
        [("off", "set"), ("expected", "off")],
    )
    def test_either_axis_can_be_turned_off(self, conformance: str, divergence: str) -> None:
        cfg = ToolSelectionEvaluatorConfig(
            name="routing",
            conformance=conformance,  # type: ignore[arg-type]
            divergence=divergence,  # type: ignore[arg-type]
        )

        assert (cfg.conformance, cfg.divergence) == (conformance, divergence)

    def test_rejects_both_axes_off(self) -> None:
        """An evaluator with nothing switched on would measure nothing."""
        with pytest.raises(ValidationError, match="measure nothing"):
            ToolSelectionEvaluatorConfig(name="routing", conformance="off", divergence="off")

    def test_rejects_a_divergence_strategy_on_the_conformance_axis(self) -> None:
        with pytest.raises(ValidationError):
            ToolSelectionEvaluatorConfig(name="routing", conformance="set")  # type: ignore[arg-type]

    def test_rejects_unknown_strategies(self) -> None:
        with pytest.raises(ValidationError):
            ToolSelectionEvaluatorConfig(name="routing", conformance="expected_multiset")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            ToolSelectionEvaluatorConfig(name="routing", divergence="jaccard")  # type: ignore[arg-type]

    def test_the_deleted_mode_key_fails_loudly(self) -> None:
        """``mode`` is gone, not deprecated — a stale config must not run.

        There is no alias and no fallback: a project still carrying
        ``mode: expected`` would otherwise silently get the defaults, which
        is the class of silence this whole change exists to remove.
        """
        with pytest.raises(ValidationError, match="mode"):
            ToolSelectionEvaluatorConfig.model_validate({"name": "routing", "mode": "expected"})


class TestToolArgumentsEvaluatorConfig:
    def test_against_defaults_to_source(self) -> None:
        assert ToolArgumentsEvaluatorConfig(name="routing_args").against == "source"

    def test_accepts_against_expected(self) -> None:
        cfg = ToolArgumentsEvaluatorConfig(name="routing_args", against="expected")

        assert cfg.against == "expected"

    def test_rejects_unknown_against(self) -> None:
        with pytest.raises(ValidationError):
            ToolArgumentsEvaluatorConfig(name="routing_args", against="ground_truth")  # type: ignore[arg-type]

    def test_default_strategy_defaults_to_auto(self) -> None:
        """``auto`` ships as the default: exact-only scoring reads a
        capitalization difference as a wrong argument value."""
        assert ToolArgumentsEvaluatorConfig(name="routing_args").default_strategy == "auto"

    def test_default_strategy_accepts_exact_as_the_opt_out(self) -> None:
        cfg = ToolArgumentsEvaluatorConfig(name="routing_args", default_strategy="exact")

        assert cfg.default_strategy == "exact"

    def test_rejects_unknown_default_strategy(self) -> None:
        with pytest.raises(ValidationError):
            ToolArgumentsEvaluatorConfig(name="routing_args", default_strategy="fuzzy")  # type: ignore[arg-type]

    def test_per_field_strategies_accept_auto(self) -> None:
        cfg = ToolArgumentsEvaluatorConfig(
            name="routing_args",
            strategies={"description": "auto"},
        )

        assert cfg.strategies["description"] == "auto"

    def test_rejects_unknown_per_field_strategy(self) -> None:
        with pytest.raises(ValidationError):
            ToolArgumentsEvaluatorConfig(
                name="routing_args",
                strategies={"description": "vibes"},  # type: ignore[dict-item]
            )


class TestAgentTraceEvaluatorConfig:
    def test_agent_trace_defaults(self) -> None:
        cfg = AgentTraceEvaluatorConfig(name="trace_safety")

        assert cfg.name == "trace_safety"
        assert cfg.applies_to == ["*"]
        assert cfg.check_tool_order is True
        assert cfg.check_arguments is True
        assert cfg.check_missing_verification is True
        assert cfg.verification_tools == []
        assert cfg.dangerous_tools == []

    def test_agent_trace_config_forbids_extra_keys(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AgentTraceEvaluatorConfig.model_validate({"name": "trace", "typo": True})


# ---------------------------------------------------------------------------
# EvalShiftConfig — top level
# ---------------------------------------------------------------------------


class TestEvalShiftConfig:
    def test_minimal_valid(self) -> None:
        cfg = EvalShiftConfig(
            prompts=[
                PromptDefinition(id="cs", detection="manual", content="hi {n}"),
            ],
        )
        assert cfg.version == 1
        assert isinstance(cfg.defaults, Defaults)
        assert isinstance(cfg.evaluators, EvaluatorsConfig)
        assert cfg.project is None
        assert cfg.migration_policy is None
        assert cfg.evaluators.agent_trace == []

    def test_migration_policy_is_optional_public_config(self) -> None:
        cfg = EvalShiftConfig(
            prompts=[
                PromptDefinition(id="cs", detection="manual", content="hi {n}"),
            ],
            migration_policy=MigrationPolicy(max_overall_regression_rate=0.05),
        )

        assert cfg.migration_policy is not None
        assert cfg.migration_policy.max_overall_regression_rate == pytest.approx(0.05)

    def test_hosted_project_is_valid(self) -> None:
        cfg = EvalShiftConfig(
            prompts=[
                PromptDefinition(id="cs", detection="manual", content="hi {n}"),
            ],
            project="acme/model-migration",
        )

        assert cfg.project == "acme/model-migration"

    def test_removed_thresholds_field_is_rejected_by_name(self) -> None:
        """A config that still sets ``thresholds`` says what happened to it.

        ``extra="forbid"`` would reject the key on its own, but "Extra inputs
        are not permitted" reads as a typo and sends the user looking for the
        right spelling of a field that no longer exists.
        """
        with pytest.raises(ValidationError) as info:
            EvalShiftConfig(
                prompts=[
                    PromptDefinition(id="cs", detection="manual", content="hi {n}"),
                ],
                thresholds={"pass_rate_min": 0.91},
            )

        message = str(info.value)
        assert "`thresholds` was removed" in message
        assert "gated nothing" in message
        assert "migration_policy is the single source of truth" in message
        assert "Extra inputs are not permitted" not in message

    def test_removed_slices_field_is_rejected_by_name(self) -> None:
        """A config that still sets top-level ``slices`` says what happened to it.

        The block was validated and recorded but never read: slices come from
        example tags. Calling it an "extra input" would send the user looking
        for its new spelling instead of telling them it is gone, and where the
        one thing it looked like it did (per-slice budgets) actually lives.
        """
        with pytest.raises(ValidationError) as info:
            EvalShiftConfig.model_validate(
                {
                    "prompts": [{"id": "cs", "detection": "manual", "content": "hi {n}"}],
                    "slices": [{"name": "long", "filter": "long"}],
                },
            )

        message = str(info.value)
        assert "`slices` was removed" in message
        assert "never had any effect" in message
        assert "example `tags`" in message
        assert "migration_policy.slices" in message
        assert "keyed by tag" in message
        assert "Extra inputs are not permitted" not in message

    def test_an_empty_top_level_slices_list_is_rejected_too(self) -> None:
        """``slices: []`` did nothing either; the key itself is what is gone."""
        with pytest.raises(ValidationError, match="`slices` was removed"):
            EvalShiftConfig.model_validate(
                {
                    "prompts": [{"id": "cs", "detection": "manual", "content": "hi"}],
                    "slices": [],
                },
            )

    def test_both_removed_fields_are_named_at_once(self) -> None:
        """Fixing one removed key should not uncover the other on the next run."""
        with pytest.raises(ValidationError) as info:
            EvalShiftConfig.model_validate(
                {
                    "prompts": [{"id": "cs", "detection": "manual", "content": "hi"}],
                    "thresholds": {"pass_rate_min": 0.9},
                    "slices": [],
                },
            )

        message = str(info.value)
        assert "`thresholds` was removed" in message
        assert "`slices` was removed" in message

    def test_invalid_hosted_project_slug_fails(self) -> None:
        with pytest.raises(ValidationError):
            EvalShiftConfig(
                prompts=[
                    PromptDefinition(id="cs", detection="manual", content="hi {n}"),
                ],
                project="Acme/model migration",
            )

    def test_round_trip_through_dump(self) -> None:
        original = EvalShiftConfig(
            prompts=[
                PromptDefinition(
                    id="cs",
                    detection="python_string",
                    path="src/prompts.py",
                    variable="MY_PROMPT",
                    variables=["conversation"],
                ),
            ],
            evaluators=EvaluatorsConfig(
                structural=[
                    StructuralEvaluatorConfig(type="json_schema", schema_path="s.json"),
                ],
                semantic=SemanticEvaluatorConfig(),
                llm_judge=[
                    LLMJudgeConfig(criterion_name="fact", criterion_prompt="?"),
                ],
                agent_trace=[
                    AgentTraceEvaluatorConfig(
                        name="trace_safety",
                        verification_tools=["check_policy"],
                        dangerous_tools=["issue_refund"],
                    ),
                ],
            ),
        )
        recreated = EvalShiftConfig.model_validate(original.model_dump())
        assert recreated == original

    def test_empty_prompts_list_fails(self) -> None:
        with pytest.raises(ValidationError):
            EvalShiftConfig(prompts=[])

    def test_duplicate_prompt_ids_fail(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate prompt ids: \['cs'\]"):
            EvalShiftConfig(
                prompts=[
                    PromptDefinition(id="cs", detection="manual", content="a"),
                    PromptDefinition(id="cs", detection="manual", content="b"),
                ],
            )

    def test_wrong_version_fails(self) -> None:
        with pytest.raises(ValidationError):
            EvalShiftConfig.model_validate(
                {
                    "version": 2,
                    "prompts": [
                        {"id": "cs", "detection": "manual", "content": "hi"},
                    ],
                },
            )

    def test_extra_top_level_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvalShiftConfig.model_validate(
                {
                    "prompts": [
                        {"id": "cs", "detection": "manual", "content": "hi"},
                    ],
                    "rogue_section": {"foo": "bar"},
                },
            )

    def test_nested_validation_error_propagates(self) -> None:
        # An invalid evaluator should fail the top-level validation, not
        # silently slip through.
        with pytest.raises(ValidationError, match="'schema_path' is required"):
            EvalShiftConfig.model_validate(
                {
                    "prompts": [
                        {"id": "cs", "detection": "manual", "content": "hi"},
                    ],
                    "evaluators": {
                        "structural": [{"type": "json_schema"}],
                    },
                },
            )


class TestFailOnDroppedParams:
    """The opt-in gate for constraints LiteLLM could not deliver."""

    def test_accepts_an_explicit_true(self) -> None:
        assert MigrationPolicy.model_validate({"fail_on_dropped_params": True})

    def test_is_not_a_slice_level_knob(self) -> None:
        """A dropped parameter is a property of the model, not of a slice."""
        with pytest.raises(ValidationError):
            SliceMigrationPolicy.model_validate({"fail_on_dropped_params": True})


class TestSamplesPerExample:
    def test_defaults_to_one(self) -> None:
        assert Defaults().samples_per_example == 1

    def test_accepts_repeated_sampling(self) -> None:
        assert Defaults(samples_per_example=3).samples_per_example == 3

    @pytest.mark.parametrize("bad", [0, -1, 21])
    def test_bounds(self, bad: int) -> None:
        with pytest.raises(ValidationError):
            Defaults(samples_per_example=bad)


class TestTraceInvariantsConfig:
    def _cfg(self, **overrides: object) -> TraceInvariantsEvaluatorConfig:
        payload: dict[str, object] = {
            "name": "payments_contract",
            "rules": [
                {"id": "auth-first", "type": "order", "before": "authenticate", "after": "charge"},
                {"id": "one-charge", "type": "call_count", "tool": "charge", "max_calls": 1},
            ],
            **overrides,
        }
        return TraceInvariantsEvaluatorConfig.model_validate(payload)

    def test_rules_parse_into_their_own_types(self) -> None:
        cfg = self._cfg()
        assert isinstance(cfg.rules[0], ToolOrderRule)
        assert isinstance(cfg.rules[1], CallCountRule)
        assert cfg.traces == "replayed"
        assert cfg.applies_to == ["*"]
        assert cfg.blocking is True
        assert cfg.owner is None

    def test_an_unknown_rule_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(rules=[{"id": "x", "type": "sometimes", "tools": ["a"]}])

    def test_rule_ids_must_be_unique(self) -> None:
        with pytest.raises(ValidationError, match="duplicate rule id"):
            self._cfg(
                rules=[
                    {"id": "dup", "type": "forbidden", "tools": ["a"]},
                    {"id": "dup", "type": "forbidden", "tools": ["b"]},
                ],
            )

    def test_an_empty_rule_list_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(rules=[])

    def test_order_rejects_a_tool_ordered_before_itself(self) -> None:
        with pytest.raises(ValidationError, match="before and after"):
            self._cfg(rules=[{"id": "x", "type": "order", "before": "a", "after": "a"}])

    def test_an_invalid_json_schema_is_rejected_at_load(self) -> None:
        with pytest.raises(ValidationError, match="json_schema"):
            self._cfg(
                rules=[
                    {
                        "id": "amount",
                        "type": "arguments",
                        "tool": "charge",
                        "json_schema": {"type": "not-a-type"},
                    },
                ],
            )

    def test_a_valid_json_schema_is_kept_verbatim(self) -> None:
        schema = {"type": "object", "properties": {"amount": {"type": "number", "maximum": 5000}}}
        cfg = self._cfg(
            rules=[{"id": "amount", "type": "arguments", "tool": "charge", "json_schema": schema}],
        )
        rule = cfg.rules[0]
        assert isinstance(rule, ArgumentsRule)
        assert rule.json_schema == schema

    def test_call_count_rejects_a_negative_bound(self) -> None:
        with pytest.raises(ValidationError):
            self._cfg(rules=[{"id": "x", "type": "call_count", "tool": "a", "max_calls": -1}])

    def test_call_count_needs_at_least_one_bound(self) -> None:
        with pytest.raises(ValidationError, match="min_calls"):
            self._cfg(rules=[{"id": "x", "type": "call_count", "tool": "a"}])

    def test_call_count_rejects_max_below_min(self) -> None:
        with pytest.raises(ValidationError, match="below"):
            self._cfg(
                rules=[
                    {"id": "x", "type": "call_count", "tool": "a", "min_calls": 2, "max_calls": 1},
                ],
            )

    def test_call_count_exact_is_both_bounds_equal(self) -> None:
        cfg = self._cfg(
            rules=[{"id": "x", "type": "call_count", "tool": "a", "min_calls": 1, "max_calls": 1}],
        )
        rule = cfg.rules[0]
        assert isinstance(rule, CallCountRule)
        assert (rule.min_calls, rule.max_calls) == (1, 1)

    def test_the_family_defaults_to_empty(self) -> None:
        assert EvaluatorsConfig().trace_invariants == []


class TestMaxInvariantViolationsPolicy:
    def test_defaults_to_zero(self) -> None:
        assert MigrationPolicy().max_invariant_violations == 0

    def test_slice_override_defaults_to_inherit(self) -> None:
        assert SliceMigrationPolicy().max_invariant_violations is None

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            MigrationPolicy(max_invariant_violations=-1)
