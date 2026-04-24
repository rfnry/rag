import voyageai

from rfnry_rag.common.language_model import LanguageModelProvider

_VOYAGE_MAX_BATCH = 128


class _VoyageEmbeddings:
    def __init__(self, provider: LanguageModelProvider) -> None:
        self._client = voyageai.AsyncClient(api_key=provider.api_key)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= _VOYAGE_MAX_BATCH:
            result = await self._client.embed(texts, model=self._model)
            return [[float(v) for v in vec] for vec in result.embeddings]
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _VOYAGE_MAX_BATCH):
            batch = texts[i : i + _VOYAGE_MAX_BATCH]
            result = await self._client.embed(batch, model=self._model)
            all_vectors.extend([float(v) for v in vec] for vec in result.embeddings)
        return all_vectors

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
