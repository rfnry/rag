"""Back-compat shim.

The canonical config shape places stores inside the methods that use them
(``VectorIngestion(store=v, embeddings=e)``) and exposes ``metadata_store``
at the engine level. ``PersistenceConfig`` remains accepted as a legacy
``RagEngineConfig.persistence`` field — when supplied, the engine
auto-assembles ingestion + retrieval methods from it at initialize() time.

Prefer the explicit shape:

    RagEngineConfig(
        metadata_store=m,
        ingestion=IngestionConfig(methods=[VectorIngestion(store=v, embeddings=e), ...]),
        retrieval=RetrievalConfig(methods=[VectorRetrieval(store=v, embeddings=e), ...]),
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from rfnry_rag.stores.document.base import BaseDocumentStore
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.metadata.base import BaseMetadataStore
from rfnry_rag.stores.vector.base import BaseVectorStore


@dataclass
class PersistenceConfig:
    vector_store: BaseVectorStore | None = None
    metadata_store: BaseMetadataStore | None = None
    document_store: BaseDocumentStore | None = None
    graph_store: BaseGraphStore | None = None
