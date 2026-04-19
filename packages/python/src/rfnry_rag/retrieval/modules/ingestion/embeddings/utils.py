from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings

EMBED_BATCH_SIZE = 100


async def embed_batched(
    embeddings: BaseEmbeddings, texts: list[str], batch_size: int = EMBED_BATCH_SIZE
) -> list[list[float]]:
    if not texts:
        return []
    if len(texts) <= batch_size:
        return await embeddings.embed(texts)
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = await embeddings.embed(batch)
        all_vectors.extend(vectors)
    return all_vectors
