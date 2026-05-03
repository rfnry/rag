from typing import TYPE_CHECKING

from rfnry_knowledge.providers.client import LLMClient as LLMClient
from rfnry_knowledge.providers.protocols import BaseEmbeddings as BaseEmbeddings
from rfnry_knowledge.providers.provider import AnthropicModelProvider as AnthropicModelProvider
from rfnry_knowledge.providers.provider import CohereModelProvider as CohereModelProvider
from rfnry_knowledge.providers.provider import GoogleModelProvider as GoogleModelProvider
from rfnry_knowledge.providers.provider import ModelProvider as ModelProvider
from rfnry_knowledge.providers.provider import OpenAIModelProvider as OpenAIModelProvider
from rfnry_knowledge.providers.provider import VoyageModelProvider as VoyageModelProvider
from rfnry_knowledge.providers.registry import build_registry as build_registry

if TYPE_CHECKING:
    from rfnry_knowledge.providers.facades import Embeddings as Embeddings
    from rfnry_knowledge.providers.facades import Reranking as Reranking
    from rfnry_knowledge.providers.facades import Vision as Vision

_FACADE_NAMES = {"Embeddings", "Reranking", "Vision"}


def __getattr__(name: str) -> object:
    if name in _FACADE_NAMES:
        from rfnry_knowledge.providers import facades

        return getattr(facades, name)
    raise AttributeError(f"module 'rfnry_knowledge.providers' has no attribute {name!r}")
