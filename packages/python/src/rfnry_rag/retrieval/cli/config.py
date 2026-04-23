from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from rfnry_rag.common.cli import get_api_key as _get_api_key
from rfnry_rag.retrieval.cli.constants import CONFIG_FILE, ENV_FILE, ConfigError, load_dotenv
from rfnry_rag.retrieval.common.language_model import LanguageModelClient, LanguageModelProvider
from rfnry_rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.embeddings.facade import Embeddings
from rfnry_rag.retrieval.modules.ingestion.embeddings.sparse.fastembed import FastEmbedSparseEmbeddings
from rfnry_rag.retrieval.modules.ingestion.vision.facade import Vision
from rfnry_rag.retrieval.modules.retrieval.search.reranking.facade import Reranking
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.hyde import HyDeRewriting
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.multi_query import MultiQueryRewriting
from rfnry_rag.retrieval.modules.retrieval.search.rewriting.step_back import StepBackRewriting
from rfnry_rag.retrieval.server import (
    GenerationConfig,
    IngestionConfig,
    PersistenceConfig,
    RagServerConfig,
    RetrievalConfig,
    TreeIndexingConfig,
    TreeSearchConfig,
)
from rfnry_rag.retrieval.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore
from rfnry_rag.retrieval.stores.vector.qdrant import QdrantVectorStore


def _build_vector_store(cfg: dict[str, Any]) -> QdrantVectorStore:
    provider = cfg.get("vector_store", "qdrant")
    if provider != "qdrant":
        raise ConfigError(f"Unknown vector store: {provider!r}. Supported: qdrant")
    return QdrantVectorStore(
        url=cfg.get("url", "http://localhost:6333"),
        collection=cfg.get("collection", "knowledge"),
    )


_EMBEDDINGS_KEYS = {
    "openai": "OPENAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
}

_EMBEDDINGS_DEFAULTS = {
    "openai": "text-embedding-3-small",
    "voyage": "voyage-3",
    "cohere": "embed-english-v3.0",
}


def _build_embeddings(cfg: dict[str, Any]) -> BaseEmbeddings:
    provider = cfg.get("embeddings", "openai")
    env_var = _EMBEDDINGS_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown embeddings provider: {provider!r}. Supported: {', '.join(_EMBEDDINGS_KEYS)}")
    api_key = _get_api_key(env_var, provider)
    model = cfg.get("model", _EMBEDDINGS_DEFAULTS[provider])

    return Embeddings(LanguageModelProvider(provider=provider, model=model, api_key=api_key))


_VISION_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_VISION_DEFAULTS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def _build_vision(cfg: dict[str, Any]):
    provider = cfg.get("vision")
    if not provider:
        return None
    env_var = _VISION_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown vision provider: {provider!r}. Supported: {', '.join(_VISION_KEYS)}")
    api_key = _get_api_key(env_var, provider)
    model = cfg.get("vision_model", _VISION_DEFAULTS[provider])

    return Vision(LanguageModelProvider(provider=provider, model=model, api_key=api_key))


_RERANKER_KEYS = {
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
}

_RERANKER_DEFAULTS = {
    "voyage": "rerank-2.5-lite",
    "cohere": "rerank-english-v3.0",
}


def _build_reranker(cfg: dict[str, Any]):
    provider = cfg.get("reranker")
    if not provider:
        return None
    env_var = _RERANKER_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown reranker: {provider!r}. Supported: {', '.join(_RERANKER_KEYS)}")
    api_key = _get_api_key(env_var, provider)
    model = cfg.get("reranker_model", _RERANKER_DEFAULTS[provider])

    return Reranking(LanguageModelProvider(provider=provider, model=model, api_key=api_key))


def _build_query_rewriter(cfg: dict[str, Any]):
    rewriter = cfg.get("rewriter")
    if not rewriter:
        return None

    provider = cfg.get("rewriter_provider")
    if not provider:
        raise ConfigError("[retrieval] rewriter requires 'rewriter_provider'")
    model = cfg.get("rewriter_model")
    if not model:
        raise ConfigError("[retrieval] rewriter requires 'rewriter_model'")

    env_var = _GENERATION_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown rewriter provider: {provider!r}. Supported: {', '.join(_GENERATION_KEYS)}")
    api_key = _get_api_key(env_var, provider)

    lm_client = LanguageModelClient(
        provider=LanguageModelProvider(provider=provider, model=model, api_key=api_key),
    )

    rewriters = {
        "hyde": HyDeRewriting,
        "multi_query": MultiQueryRewriting,
        "step_back": StepBackRewriting,
    }
    rewriter_cls = rewriters.get(rewriter)
    if rewriter_cls is None:
        raise ConfigError(f"Unknown rewriter: {rewriter!r}. Supported: {', '.join(rewriters)}")
    return rewriter_cls(lm_client=lm_client)


_GENERATION_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_GENERATION_DEFAULTS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def _build_generation_config(cfg: dict[str, Any]) -> GenerationConfig:
    provider = cfg.get("provider")
    if not provider:
        raise ConfigError("[generation] requires 'provider' (anthropic or openai)")
    env_var = _GENERATION_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown generation provider: {provider!r}. Supported: {', '.join(_GENERATION_KEYS)}")

    api_key = _get_api_key(env_var, provider)
    model = cfg.get("model", _GENERATION_DEFAULTS[provider])

    lm_client = LanguageModelClient(
        provider=LanguageModelProvider(
            provider=provider,
            model=model,
            api_key=api_key,
        ),
    )

    relevance_gate_lm = None
    if cfg.get("relevance_gate_enabled"):
        rg_provider = cfg.get("relevance_gate_provider", provider)
        rg_model = cfg.get("relevance_gate_model", model)
        rg_env_var = _GENERATION_KEYS.get(rg_provider)
        if rg_env_var is None:
            raise ConfigError(f"Unknown relevance_gate_provider: {rg_provider!r}")
        rg_api_key = _get_api_key(rg_env_var, rg_provider)
        relevance_gate_lm = LanguageModelClient(
            provider=LanguageModelProvider(provider=rg_provider, model=rg_model, api_key=rg_api_key),
        )

    return GenerationConfig(
        lm_client=lm_client,
        system_prompt=cfg.get("system_prompt", GenerationConfig.system_prompt),
        grounding_enabled=cfg.get("grounding_enabled", False),
        grounding_threshold=cfg.get("grounding_threshold", 0.5),
        relevance_gate_enabled=cfg.get("relevance_gate_enabled", False),
        relevance_gate_model=relevance_gate_lm,
        guiding_enabled=cfg.get("guiding_enabled", False),
    )


def _build_metadata_store(cfg: dict[str, Any]) -> SQLAlchemyMetadataStore:
    url = cfg.get("url")
    if not url:
        raise ConfigError("[persistence.metadata] requires 'url'")
    return SQLAlchemyMetadataStore(url=url)


def _build_tree_lm(cfg: dict[str, Any], section_name: str) -> LanguageModelClient | None:
    """Build a LanguageModelClient from a tree config section's provider/model."""
    provider = cfg.get("provider")
    if not provider:
        return None
    env_var = _GENERATION_KEYS.get(provider)
    if env_var is None:
        raise ConfigError(f"Unknown [{section_name}] provider: {provider!r}. Supported: {', '.join(_GENERATION_KEYS)}")
    api_key = _get_api_key(env_var, provider)
    model = cfg.get("model")
    if not model:
        raise ConfigError(f"[{section_name}] requires 'model' when provider is set")
    return LanguageModelClient(
        provider=LanguageModelProvider(provider=provider, model=model, api_key=api_key),
    )


def load_config(config_path: str | None = None) -> RagServerConfig:
    """Load TOML config + .env, build RagServerConfig."""
    return _load_config(config_path)


_ALLOWED_TOP_KEYS = {
    "persistence",
    "ingestion",
    "retrieval",
    "generation",
    "tree_indexing",
    "tree_search",
}


def _validate_toml_keys(toml: dict) -> None:
    """Reject unknown top-level keys in config.toml to surface typos early.

    Prior behavior silently ignored unknown keys, so a typo like
    `grounding_treshold = 0.7` in [generation] would quietly fall back to
    the default."""
    unknown = set(toml.keys()) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ConfigError(
            f"Unknown top-level key(s) in config.toml: {sorted(unknown)}. "
            f"Allowed keys: {sorted(_ALLOWED_TOP_KEYS)}"
        )


def _load_config(config_path: str | Path | None) -> RagServerConfig:
    path = Path(config_path) if config_path else CONFIG_FILE
    if not path.exists():
        raise ConfigError(f"Config not found: {path}\nRun 'rfnry-rag retrieval init' to create it.")

    env_path = path.parent / ".env" if config_path else ENV_FILE
    load_dotenv(env_path)

    with open(path, "rb") as f:
        toml = tomllib.load(f)

    _validate_toml_keys(toml)

    persistence_cfg = toml.get("persistence", {})
    if not persistence_cfg:
        raise ConfigError("[persistence] section required in config.toml")

    vector_store = _build_vector_store(persistence_cfg)
    metadata_store = None
    if "metadata" in persistence_cfg:
        metadata_store = _build_metadata_store(persistence_cfg["metadata"])

    ingestion_cfg = toml.get("ingestion", {})
    if not ingestion_cfg:
        raise ConfigError("[ingestion] section required in config.toml")

    embeddings = _build_embeddings(ingestion_cfg)
    sparse_embeddings = FastEmbedSparseEmbeddings() if ingestion_cfg.get("sparse_embeddings") else None

    # Accept either the new TOML key or the deprecated legacy one. New key wins
    # if both are present.
    chunk_context_headers = ingestion_cfg.get(
        "chunk_context_headers",
        ingestion_cfg.get("contextual_chunking", True),
    )
    ingestion = IngestionConfig(
        embeddings=embeddings,
        vision=_build_vision(ingestion_cfg),
        chunk_size=ingestion_cfg.get("chunk_size", 500),
        chunk_overlap=ingestion_cfg.get("chunk_overlap", 50),
        parent_chunk_size=ingestion_cfg.get("parent_chunk_size", 0),
        parent_chunk_overlap=ingestion_cfg.get("parent_chunk_overlap", 200),
        chunk_context_headers=chunk_context_headers,
        sparse_embeddings=sparse_embeddings,
    )

    retrieval_cfg = toml.get("retrieval", {})
    retrieval = RetrievalConfig(
        top_k=retrieval_cfg.get("top_k", 5),
        bm25_enabled=retrieval_cfg.get("bm25_enabled", False),
        reranker=_build_reranker(retrieval_cfg),
        query_rewriter=_build_query_rewriter(retrieval_cfg),
    )

    generation_cfg = toml.get("generation")
    generation = _build_generation_config(generation_cfg) if generation_cfg else GenerationConfig()

    tree_indexing_cfg = toml.get("tree_indexing", {})
    tree_indexing = (
        TreeIndexingConfig(
            enabled=tree_indexing_cfg.get("enabled", False),
            model=_build_tree_lm(tree_indexing_cfg, "tree_indexing"),
            toc_scan_pages=tree_indexing_cfg.get("toc_scan_pages", 20),
            max_pages_per_node=tree_indexing_cfg.get("max_pages_per_node", 10),
            max_tokens_per_node=tree_indexing_cfg.get("max_tokens_per_node", 20_000),
            generate_summaries=tree_indexing_cfg.get("generate_summaries", True),
            generate_description=tree_indexing_cfg.get("generate_description", True),
        )
        if tree_indexing_cfg
        else TreeIndexingConfig()
    )

    tree_search_cfg = toml.get("tree_search", {})
    tree_search = (
        TreeSearchConfig(
            enabled=tree_search_cfg.get("enabled", False),
            model=_build_tree_lm(tree_search_cfg, "tree_search"),
            max_steps=tree_search_cfg.get("max_steps", 5),
            max_context_tokens=tree_search_cfg.get("max_context_tokens", 50_000),
        )
        if tree_search_cfg
        else TreeSearchConfig()
    )

    return RagServerConfig(
        persistence=PersistenceConfig(
            vector_store=vector_store,
            metadata_store=metadata_store,
        ),
        ingestion=ingestion,
        retrieval=retrieval,
        generation=generation,
        tree_indexing=tree_indexing,
        tree_search=tree_search,
    )
