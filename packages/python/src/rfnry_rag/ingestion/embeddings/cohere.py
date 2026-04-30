import cohere

from rfnry_rag.providers.provider import LanguageModelProvider

# Maximum texts per single API call. Callers should use embed_batched() from
# rfnry_rag.common.embeddings when sending more than this many texts — that
# helper owns sub-batch chunking and concurrency.
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
        """Embed *texts* in a single API call.

        Callers must ensure ``len(texts) <= _COHERE_MAX_BATCH``; use
        ``embed_batched()`` from ``rfnry_rag.common.embeddings`` to chunk and
        gather larger inputs automatically."""
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
