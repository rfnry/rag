from openai import AsyncOpenAI

from rfnry_rag.providers.provider import LanguageModelProvider

# Maximum texts per single API call. Callers should use embed_batched() from
# rfnry_rag.ingestion.embeddings.batching when sending more than this many texts — that
# helper owns sub-batch chunking and concurrency.
_OPENAI_MAX_BATCH = 2048


class _OpenAIEmbeddings:
    def __init__(self, provider: LanguageModelProvider, max_retries: int = 3) -> None:
        self._client = AsyncOpenAI(api_key=provider.api_key, max_retries=max_retries)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in a single API call.

        Callers must ensure ``len(texts) <= _OPENAI_MAX_BATCH``; use
        ``embed_batched()`` from ``rfnry_rag.ingestion.embeddings.batching`` to chunk and
        gather larger inputs automatically."""
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
