from openai import AsyncOpenAI

from rfnry_rag.common.language_model import LanguageModelProvider

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
        if len(texts) <= _OPENAI_MAX_BATCH:
            response = await self._client.embeddings.create(input=texts, model=self._model)
            return [item.embedding for item in response.data]
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _OPENAI_MAX_BATCH):
            batch = texts[i : i + _OPENAI_MAX_BATCH]
            response = await self._client.embeddings.create(input=batch, model=self._model)
            all_vectors.extend(item.embedding for item in response.data)
        return all_vectors

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
