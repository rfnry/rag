from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rfnry_rag.exceptions import ConfigurationError
from rfnry_rag.retrieval.search.reranking.base import BaseReranking

if TYPE_CHECKING:
    from rfnry_rag.retrieval.base import BaseRetrievalMethod


@dataclass
class RetrievalConfig:
    """Cross-method retrieval config.

    Per-method knobs (BM25 settings, parent-expansion, weights, top_k overrides)
    live on the individual ``VectorRetrieval`` / ``DocumentRetrieval`` /
    ``GraphRetrieval`` instances passed in ``methods``.
    """

    methods: list[BaseRetrievalMethod] = field(default_factory=list)
    top_k: int = 5
    reranker: BaseReranking | None = None
    source_type_weights: dict[str, float] | None = None
    history_window: int = 3
    cross_reference_enrichment: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("top_k must be positive")
        if self.top_k > 200:
            raise ConfigurationError(
                f"top_k must be <= 200, got {self.top_k} — requesting thousands of results OOMs the reranker"
            )
        if not (1 <= self.history_window <= 20):
            raise ConfigurationError(f"history_window must be 1-20, got {self.history_window}")
        if self.source_type_weights is not None:
            for key, weight in self.source_type_weights.items():
                if not 0 < weight <= 10.0:
                    raise ConfigurationError(f"source_type_weights[{key!r}]={weight} — weight must be in (0, 10]")
