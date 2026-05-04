from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from rfnry_knowledge.exceptions import MemoryExtractionError
from rfnry_knowledge.memory.models import ExtractedMemory, Interaction, MemoryRow
from rfnry_knowledge.providers import ProviderClient, build_registry
from rfnry_knowledge.telemetry.usage import instrument_baml_call

b: Any = None


def _get_baml_client() -> Any:
    global b
    if b is None:
        from rfnry_knowledge.baml.baml_client.async_client import b as _b

        b = _b
    return b


@runtime_checkable
class BaseExtractor(Protocol):
    async def extract(
        self,
        interaction: Interaction,
        existing_memories: tuple[MemoryRow, ...] = (),
    ) -> tuple[ExtractedMemory, ...]: ...


def _format_interaction(interaction: Interaction) -> str:
    return "\n".join(f"[{t.role}] {t.content}" for t in interaction.turns)


def _format_existing(existing: tuple[MemoryRow, ...]) -> str:
    if not existing:
        return "(none)"
    return json.dumps(
        [{"id": m.memory_row_id, "text": m.text} for m in existing],
        ensure_ascii=False,
    )


class DefaultMemoryExtractor:
    def __init__(self, provider_client: ProviderClient) -> None:
        self._provider_client = provider_client
        self._registry = build_registry(provider_client)

    async def extract(
        self,
        interaction: Interaction,
        existing_memories: tuple[MemoryRow, ...] = (),
    ) -> tuple[ExtractedMemory, ...]:
        client = _get_baml_client()
        registry = self._registry
        valid_ids = {m.memory_row_id for m in existing_memories}
        occurred_at = interaction.occurred_at.isoformat() if interaction.occurred_at else "(unspecified)"
        try:
            response = await instrument_baml_call(
                operation="extract_memories",
                call=lambda collector: client.ExtractMemories(
                    _format_interaction(interaction),
                    occurred_at,
                    _format_existing(existing_memories),
                    baml_options={"client_registry": registry, "collector": collector},
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise MemoryExtractionError(str(exc)) from exc

        out: list[ExtractedMemory] = []
        for item in response.memories or []:
            text = (item.text or "").strip()
            if not text:
                continue
            links = tuple(rid for rid in (item.linked_memory_row_ids or []) if rid in valid_ids)
            out.append(
                ExtractedMemory(
                    text=text,
                    attributed_to=item.attributed_to,
                    linked_memory_row_ids=links,
                )
            )
        return tuple(out)
