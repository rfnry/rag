from __future__ import annotations

import pytest

from rfnry_rag.config import GenerationConfig, RagEngineConfig, RetrievalConfig, RoutingConfig
from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.providers import AnthropicModelProvider, GenerativeModelClient
from rfnry_rag.server import (
    _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS,
    RagEngine,
)


def _make_engine_for_validation(
    *,
    context_size: int | None,
    full_context_threshold: int,
    max_tokens: int = 4096,
    has_lm_client: bool = True,
) -> RagEngine:
    provider = AnthropicModelProvider(api_key="k", model="claude-test", context_size=context_size)
    lm_client: GenerativeModelClient | None = (
        GenerativeModelClient(provider=provider, max_tokens=max_tokens) if has_lm_client else None
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
    with pytest.raises(ConfigurationError, match="context_size"):
        AnthropicModelProvider(api_key="k", model="m", context_size=0)
    with pytest.raises(ConfigurationError, match="context_size"):
        AnthropicModelProvider(api_key="k", model="m", context_size=-1)


def test_provider_context_size_none_and_positive_accepted() -> None:
    AnthropicModelProvider(api_key="k", model="m")
    AnthropicModelProvider(api_key="k", model="m", context_size=200_000)
    AnthropicModelProvider(api_key="k", model="m", context_size=1)


def test_validate_skipped_when_context_size_unset() -> None:
    engine = _make_engine_for_validation(
        context_size=None,
        full_context_threshold=1_900_000,
    )
    engine._validate_full_context_fits_provider_window()


def test_validate_skipped_when_lm_client_unset() -> None:
    engine = _make_engine_for_validation(
        context_size=None,
        full_context_threshold=1_500_000,
        has_lm_client=False,
    )
    engine._validate_full_context_fits_provider_window()


def test_validate_passes_when_threshold_plus_reserve_fits() -> None:
    max_tokens = 4_096
    context_size = 200_000
    threshold = context_size - _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS - max_tokens
    engine = _make_engine_for_validation(
        context_size=context_size,
        full_context_threshold=threshold,
        max_tokens=max_tokens,
    )
    engine._validate_full_context_fits_provider_window()


def test_validate_rejects_when_threshold_overflows_window() -> None:
    with pytest.raises(ConfigurationError, match="context_size"):
        _make_engine_for_validation(
            context_size=64_000,
            full_context_threshold=150_000,
            max_tokens=4_096,
        )._validate_full_context_fits_provider_window()


def test_validate_error_names_provider() -> None:
    engine = _make_engine_for_validation(
        context_size=64_000,
        full_context_threshold=150_000,
    )
    with pytest.raises(ConfigurationError, match="anthropic:claude-test"):
        engine._validate_full_context_fits_provider_window()


def test_validate_runs_at_initialize_time() -> None:
    import asyncio

    engine = _make_engine_for_validation(
        context_size=32_000,
        full_context_threshold=150_000,
    )
    with pytest.raises(ConfigurationError, match="context_size"):
        asyncio.run(engine.initialize())


def test_boundary_exact_fit_accepted() -> None:
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
    max_tokens = 4_096
    context_size = 200_000
    threshold = context_size - _FULL_CONTEXT_NON_OUTPUT_RESERVE_TOKENS - max_tokens + 1
    with pytest.raises(ConfigurationError):
        _make_engine_for_validation(
            context_size=context_size,
            full_context_threshold=threshold,
            max_tokens=max_tokens,
        )._validate_full_context_fits_provider_window()
