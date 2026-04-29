"""Multi-hop iterative retrieval.

Public surface for the iterative-retrieval subpackage: the dataclass
config (`IterativeRetrievalConfig`), the runtime service
(`IterativeRetrievalService`), and the trace types (`IterativeHopTrace`,
`IterativeOutcome`). The engine wraps the service and layers post-loop
DIRECT escalation on top.
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
