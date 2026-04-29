"""Multi-hop iterative retrieval (R6).

R6.1 shipped the compile-time scaffold (config + BAML function + service
stub). R6.2 lands the runtime hop loop, decomposer wiring, trace
surface, and engine integration. R6.3 will add post-loop DIRECT
escalation on top of this subpackage's outputs.
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
from rfnry_rag.retrieval.modules.retrieval.iterative.trace import (
    IterativeOutcome as IterativeOutcome,
)

__all__ = [
    "IterativeHopTrace",
    "IterativeOutcome",
    "IterativeRetrievalConfig",
    "IterativeRetrievalService",
]
