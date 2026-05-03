from openai import AsyncOpenAI

from rfnry_knowledge.providers.provider import OpenAIModelProvider

_OPENAI_MAX_BATCH = 2048


class _OpenAIEmbeddings:
    def __init__(self, provider: OpenAIModelProvider, max_retries: int = 3) -> None:
        self._client = AsyncOpenAI(
            api_key=provider.api_key.get_secret_value(),
            base_url=provider.base_url,
            organization=provider.organization,
            project=provider.project,
            max_retries=max_retries,
        )
        self._model = provider.model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
