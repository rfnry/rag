import cohere

from rfnry_rag.common.language_model import LanguageModelProvider


class _CohereEmbeddings:
    def __init__(self, provider: LanguageModelProvider) -> None:
        self._client = cohere.AsyncClientV2(api_key=provider.api_key)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embed(
            texts=texts,
            model=self._model,
            input_type="search_document",
            embedding_types=["float"],
        )
        if response.embeddings.float_ is None:
            raise ValueError("Cohere embed response returned None for float embeddings")
        return [list(emb) for emb in response.embeddings.float_]

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
