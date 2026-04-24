import asyncio

from openai import AsyncOpenAI

from rfnry_rag.common.language_model import LanguageModelProvider

_OPENAI_MAX_BATCH = 2048
_EMBED_CONCURRENCY = 3  # bounded to stay well below provider rate limits


class _OpenAIEmbeddings:
    def __init__(self, provider: LanguageModelProvider, max_retries: int = 3) -> None:
        self._client = AsyncOpenAI(api_key=provider.api_key, max_retries=max_retries)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def _embed_one(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= _OPENAI_MAX_BATCH:
            return await self._embed_one(texts)

        chunks = [texts[i : i + _OPENAI_MAX_BATCH] for i in range(0, len(texts), _OPENAI_MAX_BATCH)]
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
