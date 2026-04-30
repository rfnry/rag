"""Parent-child retrieval: sum child scores per parent; parent-child on by default."""

import pytest

from rfnry_rag.retrieval.common.models import VectorResult


def _make_child(parent_id: str, score: float, point_id: str, idx: int) -> VectorResult:
    return VectorResult(
        point_id=point_id,
        score=score,
        payload={
            "chunk_type": "child",
            "parent_id": parent_id,
            "chunk_index": idx,
            "content": f"child-{idx}",
        },
    )


def test_expand_parents_sums_child_scores_when_shared_parent() -> None:
    """Three children all sharing parent P1 collapse to one parent result,
    scored by the sum of child scores."""
    from rfnry_rag.retrieval.methods.vector import VectorRetrieval

    children = [
        _make_child("P1", 0.8, "c1", 1),
        _make_child("P1", 0.7, "c2", 2),
        _make_child("P1", 0.6, "c3", 3),
    ]
    fake_parent_lookup = {
        "P1": {
            "chunk_type": "parent",
            "parent_id": "P1",
            "content": "PARENT-1",
            "chunk_index": 10,
        }
    }

    expanded = VectorRetrieval._merge_children_into_parents(children, fake_parent_lookup)

    assert len(expanded) == 1
    assert expanded[0].score == pytest.approx(2.1)
    assert expanded[0].payload.get("child_hit_count") == 3
    assert expanded[0].payload["content"] == "PARENT-1"


def test_expand_parents_with_distinct_parents_preserves_individual_scores() -> None:
    """Two children with DIFFERENT parents produce two distinct expanded results."""
    from rfnry_rag.retrieval.methods.vector import VectorRetrieval

    children = [
        _make_child("P1", 0.9, "c1", 1),
        _make_child("P2", 0.8, "c2", 2),
    ]
    fake_parent_lookup = {
        "P1": {"chunk_type": "parent", "parent_id": "P1", "content": "PARENT-1"},
        "P2": {"chunk_type": "parent", "parent_id": "P2", "content": "PARENT-2"},
    }

    expanded = VectorRetrieval._merge_children_into_parents(children, fake_parent_lookup)

    assert len(expanded) == 2
    by_score = sorted(expanded, key=lambda r: -r.score)
    assert by_score[0].score == pytest.approx(0.9)
    assert by_score[0].payload.get("child_hit_count") == 1
    assert by_score[1].score == pytest.approx(0.8)


def test_expand_parents_without_matching_parent_in_lookup_is_skipped() -> None:
    """A child whose parent_id has no entry in parent_lookup is dropped silently."""
    from rfnry_rag.retrieval.methods.vector import VectorRetrieval

    children = [
        _make_child("P1", 0.9, "c1", 1),  # P1 in lookup
        _make_child("P_MISSING", 0.8, "c2", 2),  # not in lookup
    ]
    fake_parent_lookup = {"P1": {"chunk_type": "parent", "parent_id": "P1", "content": "PARENT-1"}}

    expanded = VectorRetrieval._merge_children_into_parents(children, fake_parent_lookup)

    assert len(expanded) == 1
    assert expanded[0].payload.get("content") == "PARENT-1"


def test_default_parent_chunk_size_is_enabled() -> None:
    """IngestionConfig now defaults to parent-child retrieval enabled:
    parent_chunk_size = 3 * chunk_size when not overridden."""
    from rfnry_rag.server import IngestionConfig

    cfg = IngestionConfig(chunk_size=375)
    assert cfg.parent_chunk_size > 0, "parent-child should be enabled by default"
    assert cfg.parent_chunk_size == 3 * cfg.chunk_size


def test_parent_chunk_size_zero_still_disables_parent_child() -> None:
    """Explicit parent_chunk_size=0 still disables parent-child mode."""
    from rfnry_rag.server import IngestionConfig

    cfg = IngestionConfig(chunk_size=375, parent_chunk_size=0)
    assert cfg.parent_chunk_size == 0


def test_explicit_parent_chunk_size_respected() -> None:
    """An explicit positive parent_chunk_size overrides the 3x auto-resolution."""
    from rfnry_rag.server import IngestionConfig

    cfg = IngestionConfig(chunk_size=375, parent_chunk_size=2000)
    assert cfg.parent_chunk_size == 2000
