"""TreeIndexingService — orchestrates the full tree indexing pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from rfnry_rag.common.concurrency import run_concurrent
from rfnry_rag.retrieval.common.errors import TreeIndexingError
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import TreeIndex, TreeNode, TreePage
from rfnry_rag.retrieval.modules.ingestion.tree.structure import (
    build_tree,
    calculate_page_ranges,
    split_large_nodes,
)
from rfnry_rag.retrieval.modules.ingestion.tree.toc import (
    PageContent,
    TocPath,
    detect_toc,
    find_section_starts,
    parse_toc,
    verify_section_positions,
)
from rfnry_rag.retrieval.stores.metadata.base import BaseMetadataStore

logger = get_logger(__name__)

PAGES_PER_GROUP = 20
SUMMARY_CONCURRENCY = 10


class TreeIndexingService:
    """Orchestrates the full tree indexing pipeline.

    Accepts a TreeIndexingConfig, a metadata store, and an optional BAML
    ClientRegistry. Provides build_tree_index() to construct a TreeIndex
    from document pages, and save_tree_index() to persist it.
    """

    def __init__(
        self,
        config: Any,
        metadata_store: BaseMetadataStore,
        registry: Any = None,
    ) -> None:
        self._config = config
        self._metadata_store = metadata_store
        self._registry = registry

    async def build_tree_index(
        self,
        source_id: str,
        doc_name: str,
        pages: list[PageContent],
    ) -> TreeIndex:
        """Build a complete tree index for a document.

        Steps:
        1. Detect TOC
        2. Parse TOC or extract structure via LLM
        3. Verify section positions
        4. Build tree and calculate page ranges
        5. Split large nodes
        6. Optionally generate summaries
        7. Optionally generate doc description
        """
        # Step 1: detect TOC
        toc_info = await detect_toc(
            pages=pages,
            toc_scan_pages=self._config.toc_scan_pages,
            registry=self._registry,
        )

        # Step 2: get sections based on TOC path
        if toc_info.path == TocPath.NO_TOC:
            logger.debug("no TOC found, extracting structure via LLM")
            sections = await self._extract_structure_no_toc(pages)
        else:
            entries = await parse_toc(
                toc_pages=toc_info.toc_pages,
                registry=self._registry,
            )
            if toc_info.path == TocPath.WITHOUT_PAGE_NUMBERS:
                entries = await find_section_starts(
                    entries=entries,
                    pages=pages,
                    registry=self._registry,
                )
            sections = entries

        if not sections:
            raise TreeIndexingError("no sections found in document")

        # Step 3: verify section positions and filter out unverified
        sections = await verify_section_positions(
            sections=sections,
            pages=pages,
            registry=self._registry,
        )
        sections = [s for s in sections if s.get("verified", True)]

        if not sections:
            raise TreeIndexingError("no verified sections found in document")

        # Step 4: build tree and calculate page ranges
        roots = build_tree(sections)
        calculate_page_ranges(roots, total_pages=len(pages))

        # Step 5: split large nodes
        roots = split_large_nodes(
            nodes=roots,
            pages=pages,
            max_pages=self._config.max_pages_per_node,
            max_tokens=self._config.max_tokens_per_node,
        )

        # Step 6: optionally generate summaries
        if self._config.generate_summaries:
            await self._generate_summaries(roots, pages)

        # Step 7: optionally generate doc description
        doc_description: str | None = None
        if self._config.generate_description:
            doc_description = await self._generate_doc_description(roots)

        tree_pages = [TreePage(index=p.index, text=p.text, token_count=p.token_count) for p in pages]

        tree_index = TreeIndex(
            source_id=source_id,
            doc_name=doc_name,
            doc_description=doc_description,
            structure=roots,
            page_count=len(pages),
            created_at=datetime.now(UTC),
            pages=tree_pages,
        )

        logger.debug(
            "built tree index for %s: %d root nodes, %d pages",
            source_id,
            len(roots),
            len(pages),
        )

        return tree_index

    async def save_tree_index(self, tree_index: TreeIndex) -> None:
        """Serialize and persist a tree index to the metadata store."""
        tree_json = json.dumps(tree_index.to_dict())
        await self._metadata_store.save_tree_index(
            source_id=tree_index.source_id,
            tree_index_json=tree_json,
        )
        logger.debug("saved tree index for %s", tree_index.source_id)

    async def _extract_structure_no_toc(
        self,
        pages: list[PageContent],
    ) -> list[dict[str, Any]]:
        """Extract document structure via LLM when no TOC is available.

        Processes pages in groups (PAGES_PER_GROUP pages per group).
        Calls ExtractDocumentStructure for the first group, then
        ContinueDocumentStructure for subsequent groups with existing
        structure as context.
        """
        from rfnry_rag.retrieval.baml.baml_client.async_client import b

        all_sections: list[dict[str, Any]] = []

        for i in range(0, len(pages), PAGES_PER_GROUP):
            group = pages[i : i + PAGES_PER_GROUP]
            pages_text = "\n".join(f"--- Page {page.index} ---\n{page.text}" for page in group)

            if i == 0:
                # First group: extract from scratch
                result = await b.ExtractDocumentStructure(
                    pages_text=pages_text,
                    baml_options={"client_registry": self._registry},
                )
            else:
                # Subsequent groups: continue with existing structure
                existing_structure = json.dumps(all_sections, indent=2)
                result = await b.ContinueDocumentStructure(
                    existing_structure=existing_structure,
                    pages_text=pages_text,
                    baml_options={"client_registry": self._registry},
                )

            for section in result.sections:
                all_sections.append(
                    {
                        "structure": section.structure,
                        "title": section.title,
                        "page": section.start_page,
                    }
                )

        return all_sections

    async def _generate_summaries(
        self,
        nodes: list[TreeNode],
        pages: list[PageContent],
    ) -> None:
        """Generate summaries for all nodes recursively using run_concurrent."""

        def _collect_nodes(node_list: list[TreeNode]) -> list[TreeNode]:
            """Flatten all nodes in the tree for concurrent processing."""
            result: list[TreeNode] = []
            for node in node_list:
                result.append(node)
                if node.children:
                    result.extend(_collect_nodes(node.children))
            return result

        all_nodes = _collect_nodes(nodes)
        page_by_index = {page.index: page for page in pages}

        async def _summarize(node: TreeNode) -> None:
            from rfnry_rag.retrieval.baml.baml_client.async_client import b

            # Build section text from the node's page range
            section_pages = [
                page_by_index[idx] for idx in range(node.start_index, node.end_index + 1) if idx in page_by_index
            ]
            section_text = "\n".join(p.text for p in section_pages)

            if not section_text.strip():
                return

            node.summary = await b.GenerateNodeSummary(
                title=node.title,
                section_text=section_text,
                baml_options={"client_registry": self._registry},
            )

        await run_concurrent(all_nodes, _summarize, concurrency=SUMMARY_CONCURRENCY)

    async def _generate_doc_description(self, roots: list[TreeNode]) -> str:
        """Generate a document description from the tree structure."""
        from rfnry_rag.retrieval.baml.baml_client.async_client import b

        # Build a text representation of the tree structure
        def _format_tree(nodes: list[TreeNode], indent: int = 0) -> str:
            lines: list[str] = []
            for node in nodes:
                prefix = "  " * indent
                lines.append(f"{prefix}- {node.title} (pages {node.start_index}-{node.end_index})")
                if node.children:
                    lines.append(_format_tree(node.children, indent + 1))
            return "\n".join(lines)

        tree_structure = _format_tree(roots)

        description = await b.GenerateDocDescription(
            tree_structure=tree_structure,
            baml_options={"client_registry": self._registry},
        )

        return description
