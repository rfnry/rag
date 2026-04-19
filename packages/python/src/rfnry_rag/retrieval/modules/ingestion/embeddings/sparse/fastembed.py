import asyncio

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import SparseVector

logger = get_logger(__name__)


def _to_int_list(arr) -> list[int]:
    """Convert numpy array or list to plain Python int list."""
    return arr.tolist() if hasattr(arr, "tolist") else [int(x) for x in arr]


def _to_float_list(arr) -> list[float]:
    """Convert numpy array or list to plain Python float list."""
    return arr.tolist() if hasattr(arr, "tolist") else [float(x) for x in arr]


class FastEmbedSparseEmbeddings:
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name)
        self._model_name = model_name

    @property
    def model(self) -> str:
        return self._model_name

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, lambda: list(self._model.embed(texts)))
        return [SparseVector(indices=_to_int_list(r.indices), values=_to_float_list(r.values)) for r in raw]

    async def embed_sparse_query(self, query: str) -> SparseVector:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, lambda: list(self._model.query_embed(query)))
        r = raw[0]
        return SparseVector(indices=_to_int_list(r.indices), values=_to_float_list(r.values))
