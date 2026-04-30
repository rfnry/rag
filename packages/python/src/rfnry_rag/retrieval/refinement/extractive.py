import re
from dataclasses import replace

import numpy as np

from rfnry_rag.common.logging import get_logger
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.common.models import RetrievedChunk

logger = get_logger("retrieval/refinement/extractive")

_SENTENCE_SPLIT = re.compile(r"(?<![A-Za-z]\.)(?<=[.!?])\s+")


class ExtractiveRefinement:
    """Sentence-level extractive refinement via embedding similarity.

    Splits each chunk into sentences, embeds them alongside the query,
    and keeps only the most relevant sentences. Chunks contributing zero
    relevant sentences are dropped.
    """

    def __init__(self, embeddings: BaseEmbeddings, max_sentences: int = 10) -> None:
        self._embeddings = embeddings
        self._max_sentences = max_sentences

    async def refine(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        chunk_sentences: list[list[str]] = []
        all_sentences: list[str] = []
        for chunk in chunks:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(chunk.content) if len(s.strip()) > 10]
            if not sentences:
                sentences = [chunk.content]
            chunk_sentences.append(sentences)
            all_sentences.extend(sentences)

        if not all_sentences:
            return chunks

        texts_to_embed = [query] + all_sentences
        embeddings = await self._embeddings.embed(texts_to_embed)
        query_emb = np.array(embeddings[0])
        sentence_embs = np.array(embeddings[1:])

        norms = np.linalg.norm(sentence_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        sentence_embs_normalized = sentence_embs / norms

        query_norm = np.linalg.norm(query_emb)
        query_emb_normalized = query_emb / query_norm if query_norm > 0 else query_emb

        similarities = sentence_embs_normalized @ query_emb_normalized

        refined: list[RetrievedChunk] = []
        offset = 0
        for chunk, sentences in zip(chunks, chunk_sentences, strict=True):
            n = len(sentences)
            chunk_sims = similarities[offset : offset + n]
            offset += n

            k = min(self._max_sentences, n)
            top_indices = np.argsort(chunk_sims)[-k:]
            top_indices_sorted = sorted(top_indices)

            selected = [sentences[i] for i in top_indices_sorted]
            if selected:
                refined.append(replace(chunk, content=" ".join(selected)))

        logger.info(
            "extractive refinement: %d -> %d chunks, %d total sentences processed",
            len(chunks),
            len(refined),
            len(all_sentences),
        )

        return refined
