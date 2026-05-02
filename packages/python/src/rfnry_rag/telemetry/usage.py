from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import baml_py

from rfnry_rag.observability.context import current_obs
from rfnry_rag.telemetry.context import add_llm_usage


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_anthropic_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "tokens_input": _coerce_int(getattr(usage, "input_tokens", 0)),
        "tokens_output": _coerce_int(getattr(usage, "output_tokens", 0)),
        "tokens_cache_creation": _coerce_int(getattr(usage, "cache_creation_input_tokens", 0)),
        "tokens_cache_read": _coerce_int(getattr(usage, "cache_read_input_tokens", 0)),
    }


def extract_openai_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = _coerce_int(getattr(details, "cached_tokens", 0))
    return {
        "tokens_input": _coerce_int(getattr(usage, "prompt_tokens", 0)),
        "tokens_output": _coerce_int(getattr(usage, "completion_tokens", 0)),
        "tokens_cache_creation": 0,
        "tokens_cache_read": cached,
    }


def extract_gemini_usage(response: Any) -> dict[str, int]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    return {
        "tokens_input": _coerce_int(getattr(meta, "prompt_token_count", 0)),
        "tokens_output": _coerce_int(getattr(meta, "candidates_token_count", 0)),
        "tokens_cache_creation": 0,
        "tokens_cache_read": _coerce_int(getattr(meta, "cached_content_token_count", 0)),
    }


def extract_baml_usage(collector: baml_py.Collector) -> tuple[dict[str, int], str | None, str | None]:
    last = collector.last
    if last is None:
        return {}, None, None
    usage = last.usage
    provider = None
    model = None
    selected = last.selected_call
    if selected is not None:
        provider = getattr(selected, "provider", None)
        model = getattr(selected, "client_name", None)
    return (
        {
            "tokens_input": _coerce_int(getattr(usage, "input_tokens", 0)),
            "tokens_output": _coerce_int(getattr(usage, "output_tokens", 0)),
            "tokens_cache_creation": 0,
            "tokens_cache_read": _coerce_int(getattr(usage, "cached_input_tokens", 0)),
        },
        provider,
        model,
    )


async def instrument_call[T](
    *,
    provider: str,
    model: str,
    operation: str,
    extract_usage: Callable[[Any], dict[str, int]],
    call: Callable[[], Awaitable[T]],
) -> T:
    obs = current_obs()
    start = time.perf_counter()
    try:
        response = await call()
    except BaseException as exc:
        elapsed = int((time.perf_counter() - start) * 1000)
        if obs is not None:
            await obs.emit(
                "provider.error",
                f"{provider}/{model} {operation} failed",
                level="error",
                context={
                    "provider": provider,
                    "model": model,
                    "operation": operation,
                    "duration_ms": elapsed,
                },
                error=exc,
            )
        raise
    elapsed = int((time.perf_counter() - start) * 1000)
    usage = extract_usage(response) or {}
    add_llm_usage(provider, model, usage)
    if obs is not None:
        await obs.emit(
            "provider.call",
            f"{provider}/{model} {operation} ok",
            context={
                "provider": provider,
                "model": model,
                "operation": operation,
                "duration_ms": elapsed,
                "tokens_input": usage.get("tokens_input", 0),
                "tokens_output": usage.get("tokens_output", 0),
                "tokens_cache_creation": usage.get("tokens_cache_creation", 0),
                "tokens_cache_read": usage.get("tokens_cache_read", 0),
            },
        )
    return response


async def instrument_baml_call[T](
    *,
    operation: str,
    call: Callable[[baml_py.Collector], Awaitable[T]],
    fallback_provider: str = "baml",
    fallback_model: str = "baml",
) -> T:
    collector = baml_py.Collector()
    obs = current_obs()
    start = time.perf_counter()
    try:
        response = await call(collector)
    except BaseException as exc:
        elapsed = int((time.perf_counter() - start) * 1000)
        if obs is not None:
            await obs.emit(
                "provider.error",
                f"baml {operation} failed",
                level="error",
                context={
                    "provider": fallback_provider,
                    "model": fallback_model,
                    "operation": operation,
                    "duration_ms": elapsed,
                },
                error=exc,
            )
        raise
    elapsed = int((time.perf_counter() - start) * 1000)
    usage, provider, model = extract_baml_usage(collector)
    provider = provider or fallback_provider
    model = model or fallback_model
    add_llm_usage(provider, model, usage)
    if obs is not None:
        await obs.emit(
            "provider.call",
            f"baml {operation} ok",
            context={
                "provider": provider,
                "model": model,
                "operation": operation,
                "duration_ms": elapsed,
                "tokens_input": usage.get("tokens_input", 0),
                "tokens_output": usage.get("tokens_output", 0),
                "tokens_cache_creation": usage.get("tokens_cache_creation", 0),
                "tokens_cache_read": usage.get("tokens_cache_read", 0),
            },
        )
    return response
