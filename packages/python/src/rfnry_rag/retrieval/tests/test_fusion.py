from rfnry_rag.retrieval.common.models import RetrievedChunk
from rfnry_rag.retrieval.modules.retrieval.search.fusion import reciprocal_rank_fusion


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


class TestMergeRetrievalResults:
    """Regression: _merge_retrieval_results must use RRF, not raw-score sort.

    Before the fix, unstructured RRF scores (~0.01-0.05) always lost to
    structured cosine scores (0-1), so structured results won regardless
    of rank. After the fix, both lists go through RRF — top-ranked
    unstructured result is at or near the top of merged output."""

    def test_rank1_unstructured_competes_with_rank1_structured(self):
        from rfnry_rag.retrieval.server import RagEngine

        # Unstructured top result has RRF score 0.04 (realistic value);
        # structured top has cosine 0.3. Raw-score sort would put structured first.
        unstructured = [
            _chunk("unstr_top", score=0.04),
            _chunk("unstr_mid", score=0.02),
        ]
        structured = [
            _chunk("struct_top", score=0.30),
            _chunk("struct_mid", score=0.15),
        ]

        merged = RagEngine._merge_retrieval_results(unstructured, structured)
        top_two = {merged[0].chunk_id, merged[1].chunk_id}
        # Both lists' rank-1 must be in the top two — not structured dominating on scale
        assert "unstr_top" in top_two
        assert "struct_top" in top_two

    def test_dedup_by_chunk_id(self):
        from rfnry_rag.retrieval.server import RagEngine

        shared = _chunk("shared", score=0.05)
        unstructured = [shared, _chunk("only_u", score=0.02)]
        structured = [_chunk("shared", score=0.3), _chunk("only_s", score=0.1)]

        merged = RagEngine._merge_retrieval_results(unstructured, structured)
        ids = [c.chunk_id for c in merged]
        assert ids.count("shared") == 1

    def test_empty_structured_returns_unstructured_as_is(self):
        from rfnry_rag.retrieval.server import RagEngine

        unstructured = [_chunk("a", score=0.04)]
        merged = RagEngine._merge_retrieval_results(unstructured, [])
        assert merged == unstructured

    def test_empty_unstructured_returns_structured_as_is(self):
        from rfnry_rag.retrieval.server import RagEngine

        structured = [_chunk("a", score=0.3)]
        merged = RagEngine._merge_retrieval_results([], structured)
        assert merged == structured
