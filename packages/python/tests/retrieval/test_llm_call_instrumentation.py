"""LLM call instrumentation: every wrapped site emits provider.* events
and accumulates token usage onto the active row.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rfnry_rag.observability import Observability, RecordingSink
from rfnry_rag.observability.context import _reset_obs, _set_obs
from rfnry_rag.telemetry import QueryTelemetryRow
from rfnry_rag.telemetry.context import _reset_row, _set_row, current_query_row
from rfnry_rag.telemetry.usage import (
    extract_anthropic_usage,
    extract_baml_usage,
    extract_gemini_usage,
    extract_openai_usage,
    instrument_baml_call,
    instrument_call,
)


def _row() -> QueryTelemetryRow:
    return QueryTelemetryRow(
        query_id="q-1",
        mode="indexed",
        routing_decision="indexed",
        outcome="success",
    )


@pytest.mark.asyncio
async def test_extract_anthropic_usage_pulls_all_token_fields() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=7,
        )
    )
    out = extract_anthropic_usage(response)
    assert out == {
        "tokens_input": 10,
        "tokens_output": 20,
        "tokens_cache_creation": 3,
        "tokens_cache_read": 7,
    }


@pytest.mark.asyncio
async def test_extract_openai_usage_pulls_prompt_completion_cached() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=22,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
        )
    )
    out = extract_openai_usage(response)
    assert out == {
        "tokens_input": 11,
        "tokens_output": 22,
        "tokens_cache_creation": 0,
        "tokens_cache_read": 5,
    }


@pytest.mark.asyncio
async def test_extract_gemini_usage_handles_metadata() -> None:
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=24,
            cached_content_token_count=6,
        )
    )
    out = extract_gemini_usage(response)
    assert out == {
        "tokens_input": 12,
        "tokens_output": 24,
        "tokens_cache_creation": 0,
        "tokens_cache_read": 6,
    }


@pytest.mark.asyncio
async def test_extract_handles_missing_usage() -> None:
    assert extract_anthropic_usage(SimpleNamespace()) == {}
    assert extract_openai_usage(SimpleNamespace()) == {}
    assert extract_gemini_usage(SimpleNamespace()) == {}


@pytest.mark.asyncio
async def test_instrument_call_accumulates_and_emits_event() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _row()
    row_token = _set_row(row)

    fake_response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=5, output_tokens=8, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    )

    async def _call() -> SimpleNamespace:
        return fake_response

    try:
        result = await instrument_call(
            provider="anthropic",
            model="claude",
            operation="text_generation",
            extract_usage=extract_anthropic_usage,
            call=_call,
        )
        assert result is fake_response
        active = current_query_row()
        assert active is row
        assert row.llm_calls == 1
        assert row.tokens_input == 5
        assert row.tokens_output == 8
        events = [r.kind for r in obs_sink.records]
        assert "provider.call" in events
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_instrument_call_emits_provider_error_on_exception() -> None:
    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _row()
    row_token = _set_row(row)

    async def _boom() -> None:
        raise RuntimeError("network down")

    try:
        with pytest.raises(RuntimeError):
            await instrument_call(
                provider="openai",
                model="gpt-4o",
                operation="text_generation",
                extract_usage=lambda _r: {},
                call=_boom,
            )
        events = [r.kind for r in obs_sink.records]
        assert "provider.error" in events
        err = next(r for r in obs_sink.records if r.kind == "provider.error")
        assert err.error_type == "RuntimeError"
        assert row.llm_calls == 0  # error path does not accumulate
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_multiple_calls_accumulate_onto_active_row() -> None:
    obs_token = _set_obs(Observability(sink=RecordingSink()))
    row = _row()
    row_token = _set_row(row)

    async def _resp(input_tokens: int, output_tokens: int):
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        )

    try:
        await instrument_call(
            provider="anthropic",
            model="claude",
            operation="text_generation",
            extract_usage=extract_anthropic_usage,
            call=lambda: _resp(10, 5),
        )
        await instrument_call(
            provider="anthropic",
            model="claude",
            operation="text_generation",
            extract_usage=extract_anthropic_usage,
            call=lambda: _resp(2, 3),
        )
        assert row.llm_calls == 2
        assert row.tokens_input == 12
        assert row.tokens_output == 8
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_extract_baml_usage_reads_collector() -> None:
    import baml_py

    collector = baml_py.Collector()
    usage, provider, model = extract_baml_usage(collector)
    # Collector with no calls: empty / None
    assert usage == {}
    assert provider is None
    assert model is None


@pytest.mark.asyncio
async def test_instrument_baml_call_runs_and_emits() -> None:
    import baml_py

    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))
    row = _row()
    row_token = _set_row(row)

    async def _baml(collector: baml_py.Collector) -> str:
        assert isinstance(collector, baml_py.Collector)
        return "ok"

    try:
        result = await instrument_baml_call(operation="judge", call=_baml)
        assert result == "ok"
        assert row.llm_calls == 1
        events = [r.kind for r in obs_sink.records]
        assert "provider.call" in events
    finally:
        _reset_row(row_token)
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_instrument_baml_call_emits_error_on_failure() -> None:
    import baml_py

    obs_sink = RecordingSink()
    obs_token = _set_obs(Observability(sink=obs_sink))

    async def _baml(_collector: baml_py.Collector) -> None:
        raise ValueError("baml went sideways")

    try:
        with pytest.raises(ValueError):
            await instrument_baml_call(operation="judge", call=_baml)
        events = [r.kind for r in obs_sink.records]
        assert "provider.error" in events
    finally:
        _reset_obs(obs_token)


@pytest.mark.asyncio
async def test_instrument_call_no_op_on_active_row_absence() -> None:
    """Library users who invoke text_generation outside an engine entry point
    must not crash on missing context — the row update is a no-op."""

    async def _resp() -> SimpleNamespace:
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=2))

    result = await instrument_call(
        provider="anthropic",
        model="claude",
        operation="text_generation",
        extract_usage=extract_anthropic_usage,
        call=_resp,
    )
    assert result is not None
