"""TreeSearchService — BAML tool-use loop for tree-based retrieval."""

from __future__ import annotations

from typing import Any

from rfnry_rag.retrieval.common.errors import TreeSearchError
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import RetrievedChunk, TreeIndex, TreeSearchResult
from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent
from rfnry_rag.retrieval.modules.retrieval.tree.tools import (
    fetch_pages,
    get_subtree,
    serialize_tree_for_prompt,
)

logger = get_logger(__name__)


class TreeSearchService:
    """Runs a BAML tool-use loop to navigate a document tree and retrieve relevant pages.

    Accepts a TreeSearchConfig and an optional BAML ClientRegistry. The search()
    method iteratively calls the LLM which can fetch pages, drill down into
    subtrees, or resolve final pages.
    """

    def __init__(self, config: Any, registry: Any = None) -> None:
        self._config = config
        self._registry = registry

    async def search(
        self,
        query: str,
        tree_index: TreeIndex,
        pages: list[PageContent],
    ) -> list[TreeSearchResult]:
        """Run the tree search loop.

        Serializes the tree, then loops up to max_steps calling the LLM.
        Each step returns one of:
          - ToolFetchPages: fetch page content and append to accumulated context
          - ToolDrillDown: zoom into a subtree
          - ToolResolvedPages: final answer with resolved pages

        Returns a list of TreeSearchResult (typically one).
        """
        from rfnry_rag.retrieval.baml.baml_client.async_client import b
        from rfnry_rag.retrieval.baml.baml_client.types import (
            ToolDrillDown,
            ToolFetchPages,
            ToolResolvedPages,
        )

        tree_str = serialize_tree_for_prompt(tree_index.structure)
        accumulated_context = ""
        fetched_pages: str = ""
        max_steps = self._config.max_steps
        max_context_tokens = self._config.max_context_tokens

        for step in range(max_steps):
            logger.info("tree search step %d/%d", step + 1, max_steps)

            try:
                result = await b.TreeRetrievalStep(
                    query=query,
                    tree_structure=tree_str,
                    accumulated_context=accumulated_context,
                    baml_options={"client_registry": self._registry},
                )
            except Exception as exc:
                raise TreeSearchError(f"BAML TreeRetrievalStep failed at step {step + 1}: {exc}") from exc

            if isinstance(result, ToolResolvedPages):
                logger.info("tree search resolved at step %d: pages=%s", step + 1, result.pages)
                content = fetch_pages(result.pages, pages)
                return [
                    TreeSearchResult(
                        node_id="root",
                        title=tree_index.doc_name,
                        pages=result.pages,
                        content=content,
                        reasoning=result.reasoning,
                    )
                ]

            if isinstance(result, ToolFetchPages):
                logger.info("tree search fetching pages: %s", result.pages)
                page_content = fetch_pages(result.pages, pages)
                accumulated_context += f"\n\n{page_content}"
                fetched_pages = result.pages

                # Check token budget (rough estimate: 1 token ~ 4 chars)
                approx_tokens = len(accumulated_context) // 4
                if approx_tokens >= max_context_tokens:
                    logger.warning(
                        "tree search context budget exceeded (%d >= %d tokens), resolving early",
                        approx_tokens,
                        max_context_tokens,
                    )
                    return [
                        TreeSearchResult(
                            node_id="root",
                            title=tree_index.doc_name,
                            pages=fetched_pages,
                            content=accumulated_context.strip(),
                            reasoning=f"Context budget exceeded after fetching pages {fetched_pages}",
                        )
                    ]
                continue

            if isinstance(result, ToolDrillDown):
                logger.info("tree search drilling down into node: %s", result.node_id)
                subtree = get_subtree(tree_index.structure, result.node_id)
                if subtree is not None and subtree.children:
                    tree_str = serialize_tree_for_prompt(subtree.children)
                elif subtree is not None:
                    # Leaf node — serialize the node itself
                    tree_str = serialize_tree_for_prompt([subtree])
                else:
                    logger.warning("drill-down node %s not found, keeping current tree", result.node_id)
                continue

            # Unexpected type — should not happen with well-formed BAML output
            logger.warning("unexpected BAML result type: %s", type(result).__name__)

        # Max steps exceeded — return whatever we have
        logger.warning("tree search max steps (%d) exceeded", max_steps)
        if accumulated_context.strip():
            return [
                TreeSearchResult(
                    node_id="root",
                    title=tree_index.doc_name,
                    pages=fetched_pages,
                    content=accumulated_context.strip(),
                    reasoning=f"Max steps ({max_steps}) exceeded, returning fetched content",
                )
            ]
        return []

    @staticmethod
    def to_retrieved_chunks(
        results: list[TreeSearchResult],
        tree_index: TreeIndex,
    ) -> list[RetrievedChunk]:
        """Convert TreeSearchResult list to RetrievedChunk list for RRF fusion.

        Each result becomes a RetrievedChunk with score 1.0 (tree results are
        pre-filtered by the LLM for relevance). Tree metadata is stored in
        source_metadata.
        """
        chunks: list[RetrievedChunk] = []
        for i, result in enumerate(results):
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"tree-{result.node_id}-{i}",
                    source_id=tree_index.source_id,
                    content=result.content,
                    score=1.0,
                    source_metadata={
                        "name": tree_index.doc_name,
                        "tree_pages": result.pages,
                        "tree_reasoning": result.reasoning,
                    },
                )
            )
        return chunks
