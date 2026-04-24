import cohere

from rfnry_rag.common.language_model import LanguageModelProvider

_COHERE_MAX_BATCH = 96


class _CohereEmbeddings:
    def __init__(self, provider: LanguageModelProvider) -> None:
        self._client = cohere.AsyncClientV2(api_key=provider.api_key)
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= _COHERE_MAX_BATCH:
            response = await self._client.embed(
                texts=texts,
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            if response.embeddings.float_ is None:
                raise ValueError("Cohere embed response returned None for float embeddings")
            return [list(emb) for emb in response.embeddings.float_]
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _COHERE_MAX_BATCH):
            batch = texts[i : i + _COHERE_MAX_BATCH]
            response = await self._client.embed(
                texts=batch,
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            if response.embeddings.float_ is None:
                raise ValueError("Cohere embed response returned None for float embeddings")
            all_vectors.extend(list(emb) for emb in response.embeddings.float_)
        return all_vectors

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
