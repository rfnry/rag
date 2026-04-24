import voyageai

from rfnry_rag.common.language_model import LanguageModelProvider

# Maximum texts per single API call. Callers should use embed_batched() from
# rfnry_rag.common.embeddings when sending more than this many texts — that
# helper owns sub-batch chunking and concurrency.
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
        """Embed *texts* in a single API call.

        Callers must ensure ``len(texts) <= _VOYAGE_MAX_BATCH``; use
        ``embed_batched()`` from ``rfnry_rag.common.embeddings`` to chunk and
        gather larger inputs automatically."""
        result = await self._client.embed(texts, model=self._model)
        return [[float(v) for v in vec] for vec in result.embeddings]

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
