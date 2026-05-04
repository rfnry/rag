from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rfnry_knowledge.memory.extraction import BaseExtractor, DefaultMemoryExtractor
from rfnry_knowledge.memory.models import (
    ExtractedMemory,
    Interaction,
    InteractionTurn,
    MemoryRow,
)


class _StubExtractor:
    async def extract(
        self, interaction: Interaction, existing_memories: tuple[MemoryRow, ...] = (),
    ) -> tuple[ExtractedMemory, ...]:
        return (ExtractedMemory(text="x", attributed_to="user"),)


def test_protocol_is_satisfied_by_duck_type() -> None:
    assert isinstance(_StubExtractor(), BaseExtractor)


def _baml_response(items: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        memories=[
            SimpleNamespace(
                text=i["text"],
                attributed_to=i.get("attributed_to"),
                linked_memory_row_ids=list(i.get("linked_memory_row_ids", [])),
            )
            for i in items
        ]
    )


async def test_default_extractor_calls_baml_and_maps_response() -> None:
    provider = SimpleNamespace(name="anthropic", model="claude-x")

    interaction = Interaction(
        turns=(InteractionTurn("user", "I moved to Lisbon."),),
        occurred_at=datetime(2026, 5, 4, tzinfo=UTC),
    )

    fake = AsyncMock(return_value=_baml_response([{"text": "user lives in Lisbon", "attributed_to": "user"}]))

    async def _fake_instrument(*, operation, call):
        return await call(None)

    with patch("rfnry_knowledge.memory.extraction.build_registry", return_value=object()):
        extractor = DefaultMemoryExtractor(provider_client=provider)  # type: ignore[arg-type]
        with patch("rfnry_knowledge.memory.extraction._get_baml_client") as gc:
            gc.return_value = SimpleNamespace(ExtractMemories=fake)
            with patch("rfnry_knowledge.memory.extraction.instrument_baml_call", side_effect=_fake_instrument):
                out = await extractor.extract(interaction)

    assert len(out) == 1
    assert out[0].text == "user lives in Lisbon"
    assert out[0].attributed_to == "user"
    assert out[0].linked_memory_row_ids == ()


async def test_default_extractor_drops_invented_links() -> None:
    """linked_memory_row_ids that aren't in existing_memories must be dropped."""
    provider = SimpleNamespace(name="anthropic", model="claude-x")

    now = datetime.now(UTC)
    existing = (
        MemoryRow(
            memory_row_id="r-real", memory_id="u", text="t", text_hash="h",
            attributed_to=None, linked_memory_row_ids=(), created_at=now,
            updated_at=now, interaction_metadata={},
        ),
    )
    interaction = Interaction(turns=(InteractionTurn("user", "."),))
    fake = AsyncMock(return_value=_baml_response([
        {"text": "fact", "linked_memory_row_ids": ["r-real", "r-fake"]},
    ]))

    async def _fake_instrument(*, operation, call):
        return await call(None)

    with patch("rfnry_knowledge.memory.extraction.build_registry", return_value=object()):
        extractor = DefaultMemoryExtractor(provider_client=provider)  # type: ignore[arg-type]
        with patch("rfnry_knowledge.memory.extraction._get_baml_client") as gc:
            gc.return_value = SimpleNamespace(ExtractMemories=fake)
            with patch("rfnry_knowledge.memory.extraction.instrument_baml_call", side_effect=_fake_instrument):
                out = await extractor.extract(interaction, existing_memories=existing)

    assert out[0].linked_memory_row_ids == ("r-real",)


async def test_default_extractor_drops_empty_text_items() -> None:
    provider = SimpleNamespace(name="anthropic", model="claude-x")
    fake = AsyncMock(return_value=_baml_response([
        {"text": "real fact", "attributed_to": "user"},
        {"text": "   ", "attributed_to": "user"},
        {"text": "", "attributed_to": None},
    ]))

    async def _fake_instrument(*, operation, call):
        return await call(None)

    with patch("rfnry_knowledge.memory.extraction.build_registry", return_value=object()), \
         patch("rfnry_knowledge.memory.extraction._get_baml_client") as gc, \
         patch("rfnry_knowledge.memory.extraction.instrument_baml_call", new=AsyncMock(side_effect=_fake_instrument)):
        gc.return_value = SimpleNamespace(ExtractMemories=fake)
        extractor = DefaultMemoryExtractor(provider_client=provider)  # type: ignore[arg-type]
        out = await extractor.extract(Interaction(turns=(InteractionTurn("user", "."),)))

    assert len(out) == 1
    assert out[0].text == "real fact"
