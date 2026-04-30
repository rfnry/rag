"""LanguageModelProvider.context_size as a safety cap on FULL_CONTEXT mode.

When the generation provider declares a ``context_size``, RagEngine init must
refuse configurations where ``RoutingConfig.full_context_threshold`` plus a
reserve for system prompt + history + question + max output tokens would
exceed the advertised window. Skipped when ``context_size`` is None or
``generation.lm_client`` is None.
"""

from __future__ import annotations

import pytest

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.providers import LanguageModelClient, LanguageModelProvider
from rfnry_rag.server import (
    _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS,
    GenerationConfig,
    RagEngine,
    RagEngineConfig,
    RetrievalConfig,
    RoutingConfig,
)


def _make_engine_for_validation(
    *,
    context_size: int | None,
    full_context_threshold: int,
    max_tokens: int = 4096,
    has_lm_client: bool = True,
) -> RagEngine:
    """Build a RagEngine with a stub retrieval method so ``_validate_config``
    can run end-to-end. The retrieval stub satisfies the
    ``methods must not be empty`` check; the validation under test is the
    FULL_CONTEXT-window cross-check.
    """
    provider = LanguageModelProvider(
        provider="anthropic",
        model="claude-test",
        api_key="k",
        context_size=context_size,
    )
    lm_client: LanguageModelClient | None = (
        LanguageModelClient(provider=provider, max_tokens=max_tokens) if has_lm_client else None
    )
    generation = GenerationConfig(lm_client=lm_client, grounding_enabled=False)

    class _StubRetrievalMethod:
        weight = 1.0
        name = "stub"

    retrieval = RetrievalConfig(methods=[_StubRetrievalMethod()])  # type: ignore[list-item]
    routing = RoutingConfig(full_context_threshold=full_context_threshold)
    cfg = RagEngineConfig(retrieval=retrieval, generation=generation, routing=routing)
    return RagEngine(cfg)


def test_provider_context_size_must_be_positive_when_set() -> None:
    """``context_size=0`` and negatives are rejected at construction."""
    with pytest.raises(ConfigurationError, match="context_size"):
        LanguageModelProvider(provider="x", model="y", context_size=0)
    with pytest.raises(ConfigurationError, match="context_size"):
        LanguageModelProvider(provider="x", model="y", context_size=-1)


def test_provider_context_size_none_and_positive_accepted() -> None:
    """``None`` (default) and positive ints are accepted."""
    LanguageModelProvider(provider="x", model="y")
    LanguageModelProvider(provider="x", model="y", context_size=200_000)
    LanguageModelProvider(provider="x", model="y", context_size=1)


def test_validate_skipped_when_context_size_unset() -> None:
    """No ``context_size`` declared → validation is a no-op even with a wildly
    high threshold."""
    engine = _make_engine_for_validation(
        context_size=None,
        full_context_threshold=1_900_000,
    )
    engine._validate_full_context_fits_provider_window()  # must not raise


def test_validate_skipped_when_lm_client_unset() -> None:
    """No generation client (e.g., ingest-only engine) → validation is a no-op."""
    engine = _make_engine_for_validation(
        context_size=None,
        full_context_threshold=1_500_000,
        has_lm_client=False,
    )
    engine._validate_full_context_fits_provider_window()  # must not raise


def test_validate_passes_when_threshold_plus_reserve_fits() -> None:
    """``threshold + 16k + max_tokens ≤ context_size`` → accepted."""
    max_tokens = 4_096
    context_size = 200_000
    threshold = context_size - _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS - max_tokens
    engine = _make_engine_for_validation(
        context_size=context_size,
        full_context_threshold=threshold,
        max_tokens=max_tokens,
    )
    engine._validate_full_context_fits_provider_window()  # must not raise


def test_validate_rejects_when_threshold_overflows_window() -> None:
    """Default threshold 150k against a small 64k provider window → rejected."""
    with pytest.raises(ConfigurationError, match="exceeds LanguageModelProvider.context_size"):
        _make_engine_for_validation(
            context_size=64_000,
            full_context_threshold=150_000,
            max_tokens=4_096,
        )._validate_full_context_fits_provider_window()


def test_validate_error_names_provider() -> None:
    """The error mentions the provider name so multi-provider configs are
    debuggable."""
    engine = _make_engine_for_validation(
        context_size=64_000,
        full_context_threshold=150_000,
    )
    with pytest.raises(ConfigurationError, match="anthropic:claude-test"):
        engine._validate_full_context_fits_provider_window()


def test_validate_runs_at_initialize_time() -> None:
    """``RagEngine.initialize()`` calls ``_validate_config`` which must reject
    a context_size violation before any store is opened.
    """
    import asyncio

    engine = _make_engine_for_validation(
        context_size=32_000,
        full_context_threshold=150_000,
    )
    with pytest.raises(ConfigurationError, match="exceeds LanguageModelProvider.context_size"):
        asyncio.run(engine.initialize())


def test_boundary_exact_fit_accepted() -> None:
    """``threshold + reserve == context_size`` is at the limit but accepted
    (the cap is ``≤``, not ``<``)."""
    max_tokens = 4_096
    context_size = 200_000
    threshold = context_size - _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS - max_tokens
    engine = _make_engine_for_validation(
        context_size=context_size,
        full_context_threshold=threshold,
        max_tokens=max_tokens,
    )
    engine._validate_full_context_fits_provider_window()


def test_boundary_one_over_rejected() -> None:
    """One token past the limit raises."""
    max_tokens = 4_096
    context_size = 200_000
    threshold = context_size - _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS - max_tokens + 1
    with pytest.raises(ConfigurationError):
        _make_engine_for_validation(
            context_size=context_size,
            full_context_threshold=threshold,
            max_tokens=max_tokens,
        )._validate_full_context_fits_provider_window()
