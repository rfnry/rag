from rfnry_knowledge.models import RetrievedChunk
from rfnry_knowledge.retrieval.search.fusion import reciprocal_rank_fusion


def _chunk(chunk_id: str, score: float = 0.0, source_weight: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content="text", score=score, source_weight=source_weight)


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self):
        results = [_chunk("a", score=0.9), _chunk("b", score=0.8)]
        fused = reciprocal_rank_fusion([results])
        assert [c.chunk_id for c in fused] == ["a", "b"]

    def test_deduplicates_across_lists(self):
        list_a = [_chunk("a"), _chunk("b")]
        list_b = [_chunk("b"), _chunk("c")]
        fused = reciprocal_rank_fusion([list_a, list_b])
        ids = [c.chunk_id for c in fused]
        assert len(ids) == 3
        assert set(ids) == {"a", "b", "c"}

    def test_shared_result_ranks_higher(self):
        list_a = [_chunk("shared"), _chunk("only_a")]
        list_b = [_chunk("shared"), _chunk("only_b")]
        fused = reciprocal_rank_fusion([list_a, list_b])
        assert fused[0].chunk_id == "shared"

    def test_rrf_scores_are_computed(self):
        results = [_chunk("a"), _chunk("b")]
        fused = reciprocal_rank_fusion([results], k=60)
        assert fused[0].score > 0
        assert fused[0].score > fused[1].score

    def test_source_type_weights_applied(self):
        high = _chunk("high", source_weight=2.0)
        low = _chunk("low", source_weight=0.5)
        list_a = [low, high]
        fused = reciprocal_rank_fusion([list_a], source_type_weights={"manual": 1.0})
        assert fused[0].chunk_id == "high"

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_method_weights_scale_rrf_scores(self):
        list_a = [_chunk("a")]
        list_b = [_chunk("b")]
        fused = reciprocal_rank_fusion([list_a, list_b], method_weights=[2.0, 0.5])
        scores = {c.chunk_id: c.score for c in fused}
        assert scores["a"] > scores["b"]

    def test_method_weights_none_defaults_to_one(self):
        list_a = [_chunk("a")]
        list_b = [_chunk("b")]
        fused_weighted = reciprocal_rank_fusion([list_a, list_b], method_weights=None)
        fused_default = reciprocal_rank_fusion([list_a, list_b])
        scores_w = {c.chunk_id: c.score for c in fused_weighted}
        scores_d = {c.chunk_id: c.score for c in fused_default}
        assert scores_w == scores_d

    def test_three_lists_fusion(self):
        list_a = [_chunk("a"), _chunk("b")]
        list_b = [_chunk("b"), _chunk("c")]
        list_c = [_chunk("c"), _chunk("a")]
        fused = reciprocal_rank_fusion([list_a, list_b, list_c])
        assert len(fused) == 3
        top_ids = {fused[0].chunk_id, fused[1].chunk_id}
        assert "a" in top_ids or "b" in top_ids or "c" in top_ids
