from __future__ import annotations

from dataclasses import dataclass

from rfnry_rag.stores.document.base import BaseDocumentStore
from rfnry_rag.stores.graph.base import BaseGraphStore
from rfnry_rag.stores.metadata.base import BaseMetadataStore
from rfnry_rag.stores.vector.base import BaseVectorStore


@dataclass
class PersistenceConfig:
    """Storage backends for the RAG engine.

    Two distinct routing concepts coexist in the engine:

    - **collection** (e.g. ``vector_store.collections=["knowledge", "logs"]``):
      the backend routing key — a Qdrant collection name, filesystem subdir, or
      Postgres schema. Chosen per-ingest/retrieve call via the ``collection=``
      argument and maps 1:1 to a pipeline instance.
    - **knowledge_id** (e.g. ``knowledge_id="tenant-42"``): a per-document
      partition filter applied at query time. Multiple knowledge_ids share the
      same collection; retrieval filters to the requested one.

    Use ``collection`` to physically separate data (different Qdrant clusters
    or schemas); use ``knowledge_id`` to logically partition within one
    collection.
    """

    vector_store: BaseVectorStore | None = None
    metadata_store: BaseMetadataStore | None = None
    document_store: BaseDocumentStore | None = None
    graph_store: BaseGraphStore | None = None
