from typing import Protocol

from rfnry_rag.models import RetrievedChunk
from rfnry_rag.observability.models import MetricResult
from rfnry_rag.observability.normalize import normalize_answer


class BaseRetrievalMetric(Protocol):
    """Protocol for retrieval-quality metrics."""

    name: str

    def score(self, chunks: list[RetrievedChunk], references: list[str], top_k: int = 5) -> float: ...

    def score_batch(
        self, chunks_list: list[list[RetrievedChunk]], references: list[list[str]], top_k: int = 5
    ) -> MetricResult: ...


class RetrievalRecall:
    """Whether any top-k retrieved chunk contains the answer (binary per query)."""

    name: str = "retrieval_recall"

    def score(self, chunks: list[RetrievedChunk], references: list[str], top_k: int = 5) -> float:
        top_chunks = chunks[:top_k]
        for chunk in top_chunks:
            normalized_content = normalize_answer(chunk.content)
            for ref in references:
                if normalize_answer(ref) in normalized_content:
                    return 1.0
        return 0.0

    def score_batch(
        self, chunks_list: list[list[RetrievedChunk]], references: list[list[str]], top_k: int = 5
    ) -> MetricResult:
        scores = [self.score(chunks, refs, top_k) for chunks, refs in zip(chunks_list, references, strict=True)]
        return MetricResult(mean=sum(scores) / len(scores) if scores else 0.0, scores=scores)


class RetrievalPrecision:
    """Fraction of top-k retrieved chunks that contain the answer."""

    name: str = "retrieval_precision"

    def score(self, chunks: list[RetrievedChunk], references: list[str], top_k: int = 5) -> float:
        top_chunks = chunks[:top_k]
        if not top_chunks:
            return 0.0

        hits = 0
        for chunk in top_chunks:
            normalized_content = normalize_answer(chunk.content)
            for ref in references:
                if normalize_answer(ref) in normalized_content:
                    hits += 1
                    break
        return hits / len(top_chunks)

    def score_batch(
        self, chunks_list: list[list[RetrievedChunk]], references: list[list[str]], top_k: int = 5
    ) -> MetricResult:
        scores = [self.score(chunks, refs, top_k) for chunks, refs in zip(chunks_list, references, strict=True)]
        return MetricResult(mean=sum(scores) / len(scores) if scores else 0.0, scores=scores)
