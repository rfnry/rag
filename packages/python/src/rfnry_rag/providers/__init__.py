from typing import TYPE_CHECKING

from rfnry_rag.providers.client import LanguageModelClient as LanguageModelClient
from rfnry_rag.providers.protocols import BaseEmbeddings as BaseEmbeddings
from rfnry_rag.providers.provider import LanguageModel as LanguageModel
from rfnry_rag.providers.registry import build_registry as build_registry

if TYPE_CHECKING:
    from rfnry_rag.providers.facades import Embeddings as Embeddings
    from rfnry_rag.providers.facades import Reranking as Reranking
    from rfnry_rag.providers.facades import Vision as Vision

_FACADE_NAMES = {"Embeddings", "Reranking", "Vision"}


def __getattr__(name: str) -> object:
    if name in _FACADE_NAMES:
        from rfnry_rag.providers import facades

        return getattr(facades, name)
    raise AttributeError(f"module 'rfnry_rag.providers' has no attribute {name!r}")
