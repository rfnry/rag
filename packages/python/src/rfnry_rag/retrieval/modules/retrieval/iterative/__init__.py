"""Multi-hop iterative retrieval (R6).

R6.1 ships the compile-time scaffold (config + BAML function + service stub).
The hop loop, engine integration, and trace surface land in R6.2; post-loop
DIRECT escalation lands in R6.3.
"""

from rfnry_rag.retrieval.modules.retrieval.iterative.config import (
    IterativeRetrievalConfig as IterativeRetrievalConfig,
)
from rfnry_rag.retrieval.modules.retrieval.iterative.service import (
    IterativeRetrievalService as IterativeRetrievalService,
)
from rfnry_rag.retrieval.modules.retrieval.iterative.trace import (
    IterativeHopTrace as IterativeHopTrace,
)

__all__ = [
    "IterativeHopTrace",
    "IterativeRetrievalConfig",
    "IterativeRetrievalService",
]
