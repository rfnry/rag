import voyageai

from rfnry_rag.common.language_model import LanguageModelProvider


class _VoyageEmbeddings:
    def __init__(self, provider: LanguageModelProvider) -> None:
        self._client = voyageai.AsyncClient(api_key=provider.api_key)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self._model)
        return [[float(v) for v in vec] for vec in result.embeddings]

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
