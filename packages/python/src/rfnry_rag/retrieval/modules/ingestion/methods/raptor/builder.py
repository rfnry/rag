"""RaptorTreeBuilder — cluster→summarize→embed→persist→recurse loop (R2.2 stub).

R2.1 ships an empty stub so the public name is importable and R2.2 can fill
it in without churning import sites. The actual cluster/summarize/embed loop,
atomic blue/green swap, and stale-tree GC land in R2.2.
"""

from __future__ import annotations

from .report import RaptorBuildReport


class RaptorTreeBuilder:
    """Builds the RAPTOR summary tree for one ``knowledge_id``.

    R2.1 ships the import-site placeholder. R2.2 lands the runtime
    implementation: cluster chunks under a knowledge_id, summarise each
    cluster via ``SummarizeCluster``, embed each summary, persist to the
    vector store with ``vector_role="raptor_summary"``, and recurse up to
    ``max_levels``. The atomic swap between an in-progress and the active
    tree id is handled via the ``RaptorTreeRegistry``.
    """

    async def build(self, _knowledge_id: str) -> RaptorBuildReport:
        raise NotImplementedError("RaptorTreeBuilder lands in R2.2")
