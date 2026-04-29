"""RaptorBuildReport — outcome of one ``RagEngine.build_raptor_index`` call.

R2.1 ships the dataclass shape only. Population happens in R2.2 when
``RaptorTreeBuilder.build`` lands. The shape mirrors R6.2's
``IterativeOutcome``: a dataclass returned alongside the persisted artefacts
so consumers can inspect what happened without parsing logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RaptorBuildReport:
    """Outcome of one ``RagEngine.build_raptor_index`` call.

    ``level_counts`` indexes from the leaf upward: ``level_counts[0]`` is the
    chunk-leaf count, ``level_counts[-1]`` is typically 1 (the tree root).
    ``total_summaries`` is ``sum(level_counts[1:])`` — the number of
    SummarizeCluster outputs persisted.
    """

    knowledge_id: str
    tree_id: str
    level_counts: list[int]
    total_summaries: int
    total_decompose_calls: int
    total_cost_usd: float | None = None
    duration_seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
