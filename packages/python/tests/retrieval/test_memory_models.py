from datetime import UTC, datetime

import pytest

from rfnry_knowledge.memory.models import (
    ExtractedMemory,
    Interaction,
    InteractionTurn,
    MemoryRow,
    MemorySearchResult,
)


def test_interaction_turn_is_frozen() -> None:
    t = InteractionTurn(role="user", content="hi")
    with pytest.raises((AttributeError, Exception)):
        t.role = "assistant"  # type: ignore[misc]


def test_interaction_defaults() -> None:
    i = Interaction(turns=(InteractionTurn("user", "hi"),))
    assert i.occurred_at is None
    assert i.metadata == {}


def test_extracted_memory_defaults_and_links() -> None:
    bare = ExtractedMemory(text="x", attributed_to=None)
    assert bare.linked_memory_row_ids == ()

    linked = ExtractedMemory(text="x", attributed_to="user", linked_memory_row_ids=("r0",))
    assert linked.attributed_to == "user"
    assert linked.linked_memory_row_ids == ("r0",)


def test_memory_row_holds_all_fields() -> None:
    now = datetime.now(UTC)
    row = MemoryRow(
        memory_row_id="r1",
        memory_id="u1",
        text="hello",
        text_hash="h",
        attributed_to="user",
        linked_memory_row_ids=("r0",),
        created_at=now,
        updated_at=now,
        interaction_metadata={"k": "v"},
    )
    assert row.memory_id == "u1"
    assert row.linked_memory_row_ids == ("r0",)
    with pytest.raises((AttributeError, Exception)):
        row.text = "mutated"  # type: ignore[misc]


def test_memory_search_result_carries_pillar_scores() -> None:
    now = datetime.now(UTC)
    row = MemoryRow(
        memory_row_id="r", memory_id="u", text="t", text_hash="h",
        attributed_to=None, linked_memory_row_ids=(), created_at=now,
        updated_at=now, interaction_metadata={},
    )
    result = MemorySearchResult(
        row=row, score=0.7,
        pillar_scores={"semantic": 0.7, "keyword": 0.3, "entity": 0.1},
    )
    assert set(result.pillar_scores) == {"semantic", "keyword", "entity"}
    assert result.pillar_scores["semantic"] == 0.7
    assert result.score == 0.7
