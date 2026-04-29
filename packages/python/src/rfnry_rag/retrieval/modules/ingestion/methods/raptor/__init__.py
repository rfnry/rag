"""RAPTOR-style summarization retrieval (R2).

R2.1 ships the compile-time scaffold — config + BAML + schema migration +
registry. The runtime tree builder lands in R2.2; the ``RaptorRetrieval``
method + engine wiring land in R2.3. ``RaptorTreeBuilder`` stays internal
until R2.2 fills the stub; the rest of the surface is public.
"""

from rfnry_rag.retrieval.modules.ingestion.methods.raptor.builder import (
    RaptorTreeBuilder as RaptorTreeBuilder,
)
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.config import (
    RaptorConfig as RaptorConfig,
)
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.registry import (
    RaptorTreeRegistry as RaptorTreeRegistry,
)
from rfnry_rag.retrieval.modules.ingestion.methods.raptor.report import (
    RaptorBuildReport as RaptorBuildReport,
)

__all__ = [
    "RaptorBuildReport",
    "RaptorConfig",
    "RaptorTreeBuilder",
    "RaptorTreeRegistry",
]
