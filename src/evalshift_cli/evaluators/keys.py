"""Which judge / embedding models in an evaluator set have no API key.

``semantic`` and ``llm_judge`` are the evaluators that call a model of their
own, so they are the ones a missing key silently breaks: every call raises,
the axis records only errors, and — advisory, as ``init`` scaffolds them —
nothing gates on it. This module finds those gaps *before* any call and owns
every message about them, so the preflight line, the evaluate stage's skip
and the decision's recommendation cannot drift apart.

An advisory gap is skipped; a blocking one is refused. ``semantic``'s
effective blocking flag also accounts for ``tool_arguments``, which borrows
its embedder: skipping it would silently move a blocking ``semantic``
strategy onto exact matching.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evalshift_cli.config.models import EvaluatorsConfig, ToolArgumentsEvaluatorConfig
from evalshift_cli.models.registry import missing_api_keys

SEMANTIC_EVALUATOR_NAME = "semantic.cosine"
SEMANTIC_KIND = "semantic"
LLM_JUDGE_KIND = "llm_judge"

EXACT_FALLBACK_NOTE = (
    "tool_arguments fields using the semantic strategy fall back to exact matching until then."
)
AUTO_FALLBACK_NOTE = (
    "tool_arguments compares free-text arguments with difflib instead of embeddings until then."
)


@dataclass(frozen=True, slots=True)
class EvaluatorKeyGap:
    """One ``semantic`` / ``llm_judge`` entry whose model has no API key.

    Attributes:
        evaluator_name: The name the evaluator stamps on its records
            (``semantic.cosine``, ``llm_judge.<criterion_name>``).
        kind: ``"semantic"`` or ``"llm_judge"``.
        label: How messages name it: ``semantic`` or the judge's evaluator name.
        model: The model id as written in config.
        env_vars: The env vars that would satisfy it, primary first.
        blocking: Effective flag — see the module docstring for ``semantic``.
        note: A side effect of skipping it, or ``""``.
    """

    evaluator_name: str
    kind: str
    label: str
    model: str
    env_vars: tuple[str, ...]
    blocking: bool
    note: str = ""


def evaluator_key_gaps(
    evaluators: EvaluatorsConfig,
    env: Mapping[str, str],
) -> list[EvaluatorKeyGap]:
    """Find every ``semantic`` / ``llm_judge`` entry whose model has no key.

    Args:
        evaluators: An already-resolved set (``cfg.evaluators_for(suite)``),
            so a suite's own block is what gets checked.
        env: Environment mapping, typically ``os.environ``.

    Returns:
        One gap per keyless entry: ``semantic`` first, then judges in config
        order. Empty when every model can be called.
    """
    gaps: list[EvaluatorKeyGap] = []
    semantic = evaluators.semantic
    if semantic is not None:
        keys = missing_api_keys(semantic.embedding_model, env)
        if keys:
            blocking, note = _semantic_effect(semantic.blocking, evaluators.tool_arguments)
            gaps.append(
                EvaluatorKeyGap(
                    evaluator_name=SEMANTIC_EVALUATOR_NAME,
                    kind=SEMANTIC_KIND,
                    label=SEMANTIC_KIND,
                    model=semantic.embedding_model,
                    env_vars=keys,
                    blocking=blocking,
                    note=note,
                )
            )
    for judge in evaluators.llm_judge:
        keys = missing_api_keys(judge.judge_model, env)
        if keys:
            name = f"{LLM_JUDGE_KIND}.{judge.criterion_name}"
            gaps.append(
                EvaluatorKeyGap(
                    evaluator_name=name,
                    kind=LLM_JUDGE_KIND,
                    label=name,
                    model=judge.judge_model,
                    env_vars=keys,
                    blocking=judge.blocking,
                )
            )
    return gaps


def without_gaps(
    evaluators: EvaluatorsConfig,
    gaps: Sequence[EvaluatorKeyGap],
) -> EvaluatorsConfig:
    """Return ``evaluators`` minus the entries named by ``gaps``.

    Args:
        evaluators: The resolved set the gaps were computed from.
        gaps: The gaps to drop (normally the advisory ones).

    Returns:
        ``evaluators`` itself when there is nothing to drop, else a copy with
        ``semantic`` cleared and/or the keyless judges removed.
    """
    if not gaps:
        return evaluators
    names = {gap.evaluator_name for gap in gaps}
    update: dict[str, object] = {}
    if SEMANTIC_EVALUATOR_NAME in names:
        update["semantic"] = None
    judges = [
        j for j in evaluators.llm_judge if f"{LLM_JUDGE_KIND}.{j.criterion_name}" not in names
    ]
    if len(judges) != len(evaluators.llm_judge):
        update["llm_judge"] = judges
    return evaluators.model_copy(update=update)


def skip_warning(*, label: str, model: str, env_vars: Sequence[str], note: str = "") -> str:
    """Terminal line (Rich markup) for an advisory evaluator that will be skipped.

    Args:
        label: How the evaluator is named (``semantic`` / ``llm_judge.<name>``).
        model: The keyless model id as written in config.
        env_vars: Env vars that would satisfy it, primary first.
        note: Optional side effect of skipping it, appended when non-empty.

    Returns:
        The warning, with ``note`` on its own indented line when given.
    """
    text = (
        f"[yellow]⚠[/yellow] {label} skipped: no API key for {model} — "
        f"export {' or '.join(env_vars)} to enable it."
    )
    return f"{text}\n  {note}" if note else text


def missing_key_error(*, label: str, model: str, env_vars: Sequence[str]) -> str:
    """Terminal line (Rich markup) for a blocking evaluator with no key.

    Args:
        label: How the evaluator is named (``semantic`` / ``llm_judge.<name>``).
        model: The keyless model id as written in config.
        env_vars: Env vars that would satisfy it, primary first.

    Returns:
        The error line naming the model, the evaluator and the env vars.
    """
    return (
        f"[red]✗[/red] missing API key for [bold]{model}[/bold] ({label}); "
        f"export {' or '.join(env_vars)}."
    )


def skip_recommendation(*, label: str, model: str, env_vars: Sequence[str], note: str = "") -> str:
    """Plain-text ``recommendations`` line for an evaluator the run skipped.

    Args:
        label: How the evaluator is named (``semantic`` / ``llm_judge.<name>``).
        model: The keyless model id as written in config.
        env_vars: Env vars that would satisfy it, primary first.
        note: Optional side effect of skipping it, appended when non-empty.

    Returns:
        One sentence-pair, with ``note`` appended after a space when given.
    """
    text = (
        f"{label} was skipped: no API key for {model}. Export {' or '.join(env_vars)} to enable it."
    )
    return f"{text} {note}" if note else text


def _uses(ta: ToolArgumentsEvaluatorConfig, strategy: str) -> bool:
    return ta.default_strategy == strategy or strategy in ta.strategies.values()


def _semantic_effect(
    semantic_blocking: bool,
    tool_arguments: Sequence[ToolArgumentsEvaluatorConfig],
) -> tuple[bool, str]:
    """Effective blocking flag and side-effect note for a keyless ``semantic``."""
    if any(ta.blocking and _uses(ta, "semantic") for ta in tool_arguments):
        return True, EXACT_FALLBACK_NOTE
    if any(_uses(ta, "semantic") for ta in tool_arguments):
        return semantic_blocking, EXACT_FALLBACK_NOTE
    if any(_uses(ta, "auto") for ta in tool_arguments):
        return semantic_blocking, AUTO_FALLBACK_NOTE
    return semantic_blocking, ""


__all__ = [
    "AUTO_FALLBACK_NOTE",
    "EXACT_FALLBACK_NOTE",
    "LLM_JUDGE_KIND",
    "SEMANTIC_EVALUATOR_NAME",
    "SEMANTIC_KIND",
    "EvaluatorKeyGap",
    "evaluator_key_gaps",
    "missing_key_error",
    "skip_recommendation",
    "skip_warning",
    "without_gaps",
]
