"""RAPTOR-style summarization retrieval.

Public surface for the RAPTOR ingestion subpackage: the dataclass config
(``RaptorConfig``), the tree builder (``RaptorTreeBuilder``), the
registry (``RaptorTreeRegistry``), and the build report
(``RaptorBuildReport``). Sibling retrieval method
(``RaptorRetrieval``) lives under ``modules/retrieval/methods/``.
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
