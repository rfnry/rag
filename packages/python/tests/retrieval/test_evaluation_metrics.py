from rfnry_knowledge.models import RetrievedChunk
from rfnry_knowledge.observability.metrics import ExactMatch, F1Score
from rfnry_knowledge.observability.normalize import normalize_answer
from rfnry_knowledge.observability.retrieval_metrics import RetrievalPrecision, RetrievalRecall


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the quick brown fox") == "quick brown fox"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The SAE-30 Oil!") == "sae30 oil"


class TestExactMatch:
    def test_exact_match(self):
        em = ExactMatch()
        assert em.score("SAE 30", ["SAE 30"]) == 1.0

    def test_normalized_match(self):
        em = ExactMatch()
        assert em.score("the SAE 30", ["SAE 30"]) == 1.0

    def test_no_match(self):
        em = ExactMatch()
        assert em.score("SAE 40", ["SAE 30"]) == 0.0

    def test_multiple_references(self):
        em = ExactMatch()
        assert em.score("SAE 30", ["SAE 20", "SAE 30", "SAE 40"]) == 1.0

    def test_batch(self):
        em = ExactMatch()
        result = em.score_batch(["SAE 30", "SAE 40"], [["SAE 30"], ["SAE 30"]])
        assert result.mean == 0.5
        assert result.scores == [1.0, 0.0]


class TestF1Score:
    def test_perfect_match(self):
        f1 = F1Score()
        assert f1.score("SAE 30 oil", ["SAE 30 oil"]) == 1.0

    def test_partial_overlap(self):
        f1 = F1Score()
        score = f1.score("SAE 30 motor oil", ["SAE 30 oil"])
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        f1 = F1Score()
        assert f1.score("completely different", ["SAE 30 oil"]) == 0.0

    def test_multiple_references_takes_max(self):
        f1 = F1Score()
        score = f1.score("SAE 30 oil", ["wrong answer", "SAE 30 oil"])
        assert score == 1.0

    def test_batch(self):
        f1 = F1Score()
        result = f1.score_batch(["SAE 30 oil", "wrong"], [["SAE 30 oil"], ["SAE 30 oil"]])
        assert result.scores[0] == 1.0
        assert result.scores[1] == 0.0


def _make_chunk(content: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_id="s1", content=content, score=0.9)


class TestRetrievalRecall:
    def test_answer_found(self):
        recall = RetrievalRecall()
        chunks = [_make_chunk("The oil type is SAE 30 for this model.")]
        assert recall.score(chunks, ["SAE 30"]) == 1.0

    def test_answer_not_found(self):
        recall = RetrievalRecall()
        chunks = [_make_chunk("This document discusses air filters.")]
        assert recall.score(chunks, ["SAE 30"]) == 0.0

    def test_top_k_limit(self):
        recall = RetrievalRecall()
        chunks = [
            _make_chunk("Irrelevant content.", "c1"),
            _make_chunk("Irrelevant content.", "c2"),
            _make_chunk("The answer is SAE 30.", "c3"),
        ]
        assert recall.score(chunks, ["SAE 30"], top_k=2) == 0.0
        assert recall.score(chunks, ["SAE 30"], top_k=3) == 1.0

    def test_batch(self):
        recall = RetrievalRecall()
        result = recall.score_batch(
            [[_make_chunk("SAE 30 oil")], [_make_chunk("air filter")]],
            [["SAE 30"], ["SAE 30"]],
        )
        assert result.mean == 0.5


class TestRetrievalPrecision:
    def test_all_relevant(self):
        precision = RetrievalPrecision()
        chunks = [_make_chunk("SAE 30 oil", "c1"), _make_chunk("Use SAE 30", "c2")]
        assert precision.score(chunks, ["SAE 30"]) == 1.0

    def test_partial_relevant(self):
        precision = RetrievalPrecision()
        chunks = [_make_chunk("SAE 30 oil", "c1"), _make_chunk("air filter", "c2")]
        assert precision.score(chunks, ["SAE 30"]) == 0.5

    def test_none_relevant(self):
        precision = RetrievalPrecision()
        chunks = [_make_chunk("air filter", "c1")]
        assert precision.score(chunks, ["SAE 30"]) == 0.0
