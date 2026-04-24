import asyncio

import voyageai

from rfnry_rag.common.language_model import LanguageModelProvider

_VOYAGE_MAX_BATCH = 128
_EMBED_CONCURRENCY = 3  # bounded to stay well below provider rate limits


class _VoyageEmbeddings:
    def __init__(self, provider: LanguageModelProvider) -> None:
        self._client = voyageai.AsyncClient(api_key=provider.api_key)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def _embed_one(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self._model)
        return [[float(v) for v in vec] for vec in result.embeddings]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= _VOYAGE_MAX_BATCH:
            return await self._embed_one(texts)

        chunks = [texts[i : i + _VOYAGE_MAX_BATCH] for i in range(0, len(texts), _VOYAGE_MAX_BATCH)]
        sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

        async def embed_chunk(chunk: list[str]) -> list[list[float]]:
            async with sem:
                return await self._embed_one(chunk)

        results = await asyncio.gather(*(embed_chunk(c) for c in chunks))
        out: list[list[float]] = []
        for r in results:
            out.extend(r)
        return out

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
