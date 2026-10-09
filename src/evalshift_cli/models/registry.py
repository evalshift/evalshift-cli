"""Registry of LLM models EvalShift knows about, plus a permissive resolver.

The registry is **advisory**, not gating. It provides:

* Friendly aliases so users can write ``claude-4.5-sonnet`` instead of
  ``anthropic/claude-sonnet-4-5``.
* Sensible default temperature / max_tokens for known models.
* Provider info for ``evalshift doctor`` and report rendering.

But the *authority* on whether a model is callable is **LiteLLM**, not us.
A user pulling a fresh preview id out of Google AI Studio (e.g.
``gemini-2.5-flash-lite-preview``) shouldn't have to wait for an
EvalShift release to use it. So we expose two functions:

* :func:`get_model` — strict registry lookup (raises
  :class:`UnknownModelError`). Used in tests and places that genuinely
  want to enforce the curated list.
* :func:`resolve_model` — never raises. Tries the registry, then falls
  back to inferring the provider from the id's prefix (``gemini-…`` →
  google, ``claude-…`` → anthropic, ``gpt-…`` / ``o1-…`` / ``o3-…`` →
  openai, ``deepseek-…`` → deepseek). Used by everything in the call
  path so LiteLLM gets the final say.

When a synthesised model is returned, :attr:`ModelMetadata.provider`
will be one of the standard providers (best-effort prefix inference)
or ``"other"`` if we can't tell. The id is rewritten to include a
provider prefix when we can confidently infer one, so LiteLLM's
routing works without further plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

Provider = Literal["anthropic", "openai", "google", "deepseek", "other"]

# Env vars LiteLLM reads to authenticate each provider, in preference
# order (primary first; the second entry is an accepted alias).
PROVIDER_ENV_VARS: Final[dict[Provider, tuple[str, ...]]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


class UnknownModelError(KeyError):
    """Raised by :func:`get_model` when no canonical id or alias matches."""

    def __init__(self, requested: str, *, suggestions: list[str] | None = None) -> None:
        self.requested = requested
        self.suggestions: list[str] = list(suggestions or [])
        message = f"unknown model: {requested!r}"
        if self.suggestions:
            message += f" (try one of: {', '.join(self.suggestions)})"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Static metadata for a model EvalShift supports.

    Attributes:
        id: The canonical LiteLLM model identifier (provider-prefixed).
            This is what EvalShift sends to ``litellm.acompletion``.
        provider: ``"anthropic"`` | ``"openai"`` | ``"google"`` | ``"deepseek"``.
        display_name: Human-friendly name shown in reports.
        aliases: Other strings that resolve to this model. Aliases must
            be globally unique across the registry.
        default_temperature: Used when the caller doesn't override.
            Defaults to 0 because we want deterministic outputs in the
            evaluation hot path.
        default_max_tokens: Sane default for completion length, used as the
            last-resort fallback when config supplies no value (e.g.
            ``test-call`` / direct client use). Users override per-call and via
            ``defaults.max_tokens`` / ``prompts[].max_tokens``.
    """

    id: str
    provider: Provider
    display_name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    default_temperature: float = 0.0
    default_max_tokens: int = 4096


# Registry. New models go here. Each entry's `id` is the canonical
# LiteLLM identifier EvalShift dispatches to.
_MODELS: Final[tuple[ModelMetadata, ...]] = (
    # ---- Anthropic -------------------------------------------------------
    # Sonnet 4.5 is the latest Anthropic Sonnet at the time of writing.
    # The PDF's forward-looking "claude-5-sonnet" alias resolves here for
    # now and will get its own row once Anthropic ships Claude 5.
    ModelMetadata(
        id="anthropic/claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        aliases=("claude-4.5-sonnet", "claude-sonnet-4-5", "claude-5-sonnet"),
    ),
    ModelMetadata(
        id="anthropic/claude-opus-4-5",
        provider="anthropic",
        display_name="Claude Opus 4.5",
        aliases=("claude-opus-4-5", "claude-5-opus"),
    ),
    # ---- OpenAI ----------------------------------------------------------
    ModelMetadata(
        id="openai/gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        aliases=("gpt-4o", "gpt-5"),  # gpt-5 mapped forward
    ),
    ModelMetadata(
        id="openai/gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o mini",
        aliases=("gpt-4o-mini", "gpt-5-mini"),
    ),
    # ---- Google ----------------------------------------------------------
    ModelMetadata(
        id="gemini/gemini-2.5-pro",
        provider="google",
        display_name="Gemini 2.5 Pro",
        aliases=("gemini-2.5-pro",),
    ),
    ModelMetadata(
        id="gemini/gemini-2.5-flash",
        provider="google",
        display_name="Gemini 2.5 Flash",
        aliases=("gemini-2.5-flash",),
    ),
    # ---- DeepSeek --------------------------------------------------------
    # The two ids DeepSeek's API serves as of 2026-09. Both run in thinking
    # mode by default, which ignores temperature — see models/deepseek.py.
    ModelMetadata(
        id="deepseek/deepseek-flash",
        provider="deepseek",
        display_name="DeepSeek V4.1 Flash",
        aliases=("deepseek-flash",),
    ),
    ModelMetadata(
        id="deepseek/deepseek-v4-pro",
        provider="deepseek",
        display_name="DeepSeek V4 Pro",
        aliases=("deepseek-v4-pro",),
    ),
)


def _build_lookup() -> dict[str, ModelMetadata]:
    """Build the canonical-id-or-alias → ``ModelMetadata`` lookup.

    Asserts every alias is globally unique and never collides with a
    canonical id from another row. Run at import time so registry typos
    fail loudly during ``pytest --collect-only`` rather than at runtime.
    """
    lookup: dict[str, ModelMetadata] = {}
    for meta in _MODELS:
        # Canonical ids must be unique.
        if meta.id in lookup and lookup[meta.id] is not meta:
            raise RuntimeError(f"model registry corrupt: id {meta.id!r} maps to two different rows")
        lookup[meta.id] = meta
        for alias in meta.aliases:
            if alias in lookup and lookup[alias] is not meta:
                raise RuntimeError(
                    f"model registry corrupt: alias {alias!r} resolves ambiguously "
                    f"to both {lookup[alias].id!r} and {meta.id!r}"
                )
            lookup[alias] = meta
    return lookup


_LOOKUP: Final[dict[str, ModelMetadata]] = _build_lookup()


def get_model(id_or_alias: str) -> ModelMetadata:
    """Resolve a canonical id or alias to its :class:`ModelMetadata`.

    Args:
        id_or_alias: Either a canonical LiteLLM id
            (``"gemini/gemini-2.5-flash"``) or a friendly alias
            (``"gemini-2.5-flash"``).

    Returns:
        The matching :class:`ModelMetadata`.

    Raises:
        UnknownModelError: If no entry resolves. The error includes a
            short list of supported ids to nudge users toward a fix.
    """
    if id_or_alias in _LOOKUP:
        return _LOOKUP[id_or_alias]
    raise UnknownModelError(
        requested=id_or_alias,
        suggestions=sorted({m.id for m in _MODELS}),
    )


def list_supported() -> list[ModelMetadata]:
    """Return every supported model in registration order.

    Result is a fresh list so callers can't accidentally mutate the
    registry by appending to it.
    """
    seen: set[str] = set()
    out: list[ModelMetadata] = []
    for meta in _MODELS:
        if meta.id in seen:
            continue
        seen.add(meta.id)
        out.append(meta)
    return out


def resolve_model(id_or_alias: str) -> ModelMetadata:
    """Resolve any user-supplied model id to :class:`ModelMetadata`.

    Unlike :func:`get_model`, this never raises. The registry is checked
    first; on miss, we synthesise metadata by inferring the provider
    from the id's prefix (or shape) and rewriting the canonical id to
    include the LiteLLM provider prefix when we can.

    Use this in the call path (orchestrator, model client, cost
    estimator) so users can pass model ids straight from a vendor
    playground or AI Studio without waiting for an EvalShift release
    to add them to the registry. LiteLLM is the source of truth at
    call time and will produce a clean error if the id genuinely
    doesn't resolve.

    Args:
        id_or_alias: Any user-supplied model identifier. May or may
            not be in the registry.

    Returns:
        Either the registered :class:`ModelMetadata` (when the id is
        known) or a synthesised one with sensible defaults. The
        returned object's ``id`` is always the form to send to LiteLLM.
    """
    if id_or_alias in _LOOKUP:
        return _LOOKUP[id_or_alias]
    canonical, provider = _infer_provider_and_canonical(id_or_alias)
    return ModelMetadata(
        id=canonical,
        provider=provider,
        display_name=f"{id_or_alias} (passthrough)",
    )


def missing_api_keys(model: str, env: Mapping[str, str]) -> tuple[str, ...]:
    """Return the env vars that would authenticate ``model`` when none is set.

    The single answer to "can this model be called?" — the arm preflight in
    ``run``/``compare``, the insights stage and the evaluator key checks all
    ask it, so an alias accepted in one place is accepted everywhere.

    Args:
        model: Any model id or alias :func:`resolve_model` accepts.
        env: Environment mapping, typically ``os.environ``. An empty-string
            value counts as unset: an unset CI secret arrives as ``""``.

    Returns:
        The provider's env-var aliases, primary first, when none of them holds
        a non-empty value. ``()`` when one does, or when the provider is
        ``"other"`` and the key it would need is unknown.
    """
    keys = PROVIDER_ENV_VARS.get(resolve_model(model).provider, ())
    if not keys or any(env.get(key) for key in keys):
        return ()
    return keys


def _infer_provider_and_canonical(id_or_alias: str) -> tuple[str, Provider]:
    """Best-effort guess of provider + LiteLLM-style canonical id.

    Decision tree (in order):

    * If the id already has a ``<provider>/`` prefix, trust it.
    * If it starts with ``gemini-`` → google, prefix ``gemini/``.
    * If it starts with ``claude-`` → anthropic, prefix ``anthropic/``.
    * If it starts with ``gpt-``, ``o1-``, or ``o3-`` → openai, prefix
      ``openai/``.
    * If it starts with ``deepseek-`` → deepseek, prefix ``deepseek/``.
    * If it starts with ``text-embedding-`` → openai, prefix ``openai/``
      (OpenAI's embedding ids; the config default is the bare form).
    * Otherwise → provider ``"other"``, id passed through unchanged.
    """
    if "/" in id_or_alias:
        prefix = id_or_alias.split("/", 1)[0]
        # LiteLLM uses ``gemini/...`` (the SDK name), but our Provider
        # taxonomy uses the company name ``google``. Map at the boundary.
        prefix_to_provider: dict[str, Provider] = {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "google",
            "gemini": "google",
            "deepseek": "deepseek",
        }
        return id_or_alias, prefix_to_provider.get(prefix, "other")
    if id_or_alias.startswith("gemini-"):
        return f"gemini/{id_or_alias}", "google"
    if id_or_alias.startswith("claude-"):
        return f"anthropic/{id_or_alias}", "anthropic"
    if id_or_alias.startswith(("gpt-", "o1-", "o3-")):
        return f"openai/{id_or_alias}", "openai"
    if id_or_alias.startswith("deepseek-"):
        return f"deepseek/{id_or_alias}", "deepseek"
    if id_or_alias.startswith("text-embedding-"):
        return f"openai/{id_or_alias}", "openai"
    return id_or_alias, "other"


__all__ = [
    "PROVIDER_ENV_VARS",
    "ModelMetadata",
    "Provider",
    "UnknownModelError",
    "get_model",
    "list_supported",
    "missing_api_keys",
    "resolve_model",
]
