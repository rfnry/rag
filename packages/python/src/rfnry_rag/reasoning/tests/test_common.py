from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from rfnry_rag.reasoning.common.concurrency import run_concurrent
from rfnry_rag.reasoning.common.errors import ConfigurationError, ReasoningError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient, LanguageModelProvider, build_registry


async def test_concurrent_preserves_order():
    async def double(x: int) -> int:
        return x * 2

    results = await run_concurrent([1, 2, 3, 4, 5], double, concurrency=3)
    assert results == [2, 4, 6, 8, 10]


async def test_concurrent_bounds_parallelism():
    active = 0
    peak = 0

    async def tracked(x: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return x

    await run_concurrent(range(10), tracked, concurrency=3)
    assert peak <= 3


async def test_concurrent_empty_input():
    async def identity(x: int) -> int:
        return x

    results = await run_concurrent([], identity, concurrency=5)
    assert results == []


async def test_concurrent_single_item():
    async def identity(x: int) -> int:
        return x

    results = await run_concurrent([42], identity, concurrency=1)
    assert results == [42]


async def test_concurrent_propagates_errors():
    async def fail(x: int) -> int:
        raise ValueError(f"boom-{x}")

    with pytest.raises(ValueError, match="boom"):
        await run_concurrent([1], fail, concurrency=1)


def test_lm_client_valid_primary_only():
    config = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini"),
    )
    assert config.strategy == "primary_only"
    assert config.max_retries == 3


def test_lm_client_valid_fallback():
    config = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o"),
        fallback=LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514"),
        strategy="fallback",
    )
    assert config.fallback is not None


def test_lm_client_invalid_strategy():
    with pytest.raises(ConfigurationError, match="Invalid strategy"):
        LanguageModelClient(
            provider=LanguageModelProvider(provider="openai", model="gpt-4o"),
            strategy="invalid",
        )


def test_lm_client_max_retries_negative():
    with pytest.raises(ConfigurationError, match="max_retries"):
        LanguageModelClient(
            provider=LanguageModelProvider(provider="openai", model="gpt-4o"),
            max_retries=-1,
        )


def test_lm_client_max_retries_exceeds_limit():
    with pytest.raises(ConfigurationError, match="max_retries"):
        LanguageModelClient(
            provider=LanguageModelProvider(provider="openai", model="gpt-4o"),
            max_retries=6,
        )


def test_lm_client_fallback_strategy_without_fallback():
    with pytest.raises(ConfigurationError, match="requires a fallback"):
        LanguageModelClient(
            provider=LanguageModelProvider(provider="openai", model="gpt-4o"),
            strategy="fallback",
        )


def test_build_registry_primary_only():
    config = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini", api_key="key"),
    )
    registry = build_registry(config)
    assert registry is not None


def test_build_registry_with_fallback():
    config = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o", api_key="key1"),
        fallback=LanguageModelProvider(provider="anthropic", model="claude-sonnet-4-20250514", api_key="key2"),
        strategy="fallback",
    )
    registry = build_registry(config)
    assert registry is not None


def test_build_registry_zero_retries():
    config = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini"),
        max_retries=0,
    )
    registry = build_registry(config)
    assert registry is not None


@patch.dict("os.environ", {}, clear=False)
def test_build_registry_sets_boundary_key():
    config = LanguageModelClient(
        provider=LanguageModelProvider(provider="openai", model="gpt-4o-mini"),
        boundary_api_key="test-boundary-key",
    )
    build_registry(config)
    import os

    assert os.environ.get("BOUNDARY_API_KEY") == "test-boundary-key"


def test_ace_error_is_base():
    from rfnry_rag.reasoning.common.errors import (
        AnalysisError,
        ClassificationError,
        ClusteringError,
        ComplianceError,
        EvaluationError,
    )

    assert issubclass(AnalysisError, ReasoningError)
    assert issubclass(ClassificationError, ReasoningError)
    assert issubclass(ClusteringError, ReasoningError)
    assert issubclass(ComplianceError, ReasoningError)
    assert issubclass(EvaluationError, ReasoningError)
    from rfnry_rag.common.errors import SdkBaseError

    assert issubclass(ConfigurationError, SdkBaseError)
