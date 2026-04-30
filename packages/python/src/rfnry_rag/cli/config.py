from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from rfnry_rag.cli.constants import CONFIG_FILE, ENV_FILE, ConfigError, load_dotenv
from rfnry_rag.cli.utils import get_api_key as _get_api_key
from rfnry_rag.config import (
    GenerationConfig,
    IngestionConfig,
    RagEngineConfig,
    RetrievalConfig,
)
from rfnry_rag.ingestion.embeddings.base import BaseEmbeddings
from rfnry_rag.ingestion.embeddings.sparse.fastembed import FastEmbedSparseEmbeddings
from rfnry_rag.ingestion.methods import VectorIngestion
from rfnry_rag.providers import Embeddings, LanguageModel, LanguageModelClient, Reranking
from rfnry_rag.retrieval.methods.vector import VectorRetrieval
from rfnry_rag.server import _derive_embedding_model_name
from rfnry_rag.stores.metadata.sqlalchemy import SQLAlchemyMetadataStore
from rfnry_rag.stores.vector.qdrant import QdrantVectorStore


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

    return Embeddings(LanguageModel(provider=provider, model=model, api_key=api_key))


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

    return Reranking(LanguageModel(provider=provider, model=model, api_key=api_key))


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
        lm=LanguageModel(
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
            lm=LanguageModel(provider=rg_provider, model=rg_model, api_key=rg_api_key),
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


def load_config(config_path: str | None = None) -> RagEngineConfig:
    """Load TOML config + .env, build RagEngineConfig."""
    return _load_config(config_path)


_ALLOWED_TOP_KEYS = {
    "persistence",
    "ingestion",
    "retrieval",
    "generation",
}


def _validate_toml_keys(toml: dict) -> None:
    """Reject unknown top-level keys in config.toml to surface typos early.

    Prior behavior silently ignored unknown keys, so a typo like
    `grounding_treshold = 0.7` in [generation] would quietly fall back to
    the default."""
    unknown = set(toml.keys()) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ConfigError(
            f"Unknown top-level key(s) in config.toml: {sorted(unknown)}. Allowed keys: {sorted(_ALLOWED_TOP_KEYS)}"
        )


def _load_config(config_path: str | Path | None) -> RagEngineConfig:
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

    chunk_context_headers = ingestion_cfg.get("chunk_context_headers", True)
    embedding_model_name = _derive_embedding_model_name(embeddings)
    ingestion = IngestionConfig(
        methods=[
            VectorIngestion(
                store=vector_store,
                embeddings=embeddings,
                embedding_model_name=embedding_model_name,
                sparse_embeddings=sparse_embeddings,
            )
        ],
        chunk_size=ingestion_cfg.get("chunk_size", 500),
        chunk_overlap=ingestion_cfg.get("chunk_overlap", 50),
        parent_chunk_size=ingestion_cfg.get("parent_chunk_size", 0),
        parent_chunk_overlap=ingestion_cfg.get("parent_chunk_overlap", 200),
        chunk_context_headers=chunk_context_headers,
    )

    retrieval_cfg = toml.get("retrieval", {})
    bm25_enabled = retrieval_cfg.get("bm25_enabled", False)
    retrieval = RetrievalConfig(
        methods=[
            VectorRetrieval(
                store=vector_store,
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
                bm25_enabled=bm25_enabled,
            )
        ],
        top_k=retrieval_cfg.get("top_k", 5),
        reranker=_build_reranker(retrieval_cfg),
    )

    generation_cfg = toml.get("generation")
    generation = _build_generation_config(generation_cfg) if generation_cfg else GenerationConfig()

    return RagEngineConfig(
        metadata_store=metadata_store,
        ingestion=ingestion,
        retrieval=retrieval,
        generation=generation,
    )
