"""End-to-end integration test for the tree retrieval pipeline.

Exercises the full flow: build a tree index from document pages, search it
using tree retrieval, and convert results to RetrievedChunk format.

All BAML calls are mocked at the module level.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rfnry_rag.retrieval.common.models import RetrievedChunk, TreeIndex, TreeNode, TreeSearchResult
from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent, TocInfo, TocPath
from rfnry_rag.retrieval.server import TreeIndexingConfig, TreeSearchConfig

# ---------------------------------------------------------------------------
# Helpers — fake document content
# ---------------------------------------------------------------------------


def _make_annual_report_pages() -> list[PageContent]:
    """Create a fake 8-page annual report with realistic content."""
    page_texts = [
        # Page 1: cover / TOC
        (
            "ACME Corp Annual Report 2025\n"
            "Table of Contents\n"
            "1. Executive Summary .............. 2\n"
            "2. Financial Highlights ........... 3\n"
            "  2.1 Revenue Breakdown ........... 4\n"
            "  2.2 Operating Expenses .......... 5\n"
            "3. Strategic Outlook .............. 6\n"
            "  3.1 Market Expansion ............ 7\n"
            "4. Appendix ....................... 8\n"
        ),
        # Page 2: Executive Summary
        (
            "1. Executive Summary\n\n"
            "ACME Corp achieved record revenue of $1.2B in 2025, representing "
            "a 15% year-over-year increase. Our strategic investments in cloud "
            "infrastructure and AI-driven analytics have positioned us as a "
            "market leader in the enterprise software segment."
        ),
        # Page 3: Financial Highlights
        (
            "2. Financial Highlights\n\n"
            "Total revenue: $1.2 billion\n"
            "Net income: $180 million\n"
            "EBITDA margin: 28%\n"
            "Free cash flow: $220 million\n"
            "Year-over-year revenue growth: 15%"
        ),
        # Page 4: Revenue Breakdown
        (
            "2.1 Revenue Breakdown\n\n"
            "Cloud services: $720M (60%)\n"
            "Enterprise licenses: $300M (25%)\n"
            "Professional services: $120M (10%)\n"
            "Other: $60M (5%)\n\n"
            "Cloud services grew 28% year-over-year, driven by new enterprise "
            "customer acquisitions and expansion of existing contracts."
        ),
        # Page 5: Operating Expenses
        (
            "2.2 Operating Expenses\n\n"
            "Research & Development: $350M (29% of revenue)\n"
            "Sales & Marketing: $240M (20% of revenue)\n"
            "General & Administrative: $95M (8% of revenue)\n"
            "Cost of Revenue: $335M (28% of revenue)\n\n"
            "We increased R&D spending by 22% to accelerate product innovation."
        ),
        # Page 6: Strategic Outlook
        (
            "3. Strategic Outlook\n\n"
            "Looking ahead to 2026, ACME Corp is focused on three strategic "
            "pillars: expanding our cloud platform capabilities, deepening our "
            "AI integration across products, and entering new geographic markets "
            "in APAC and Latin America."
        ),
        # Page 7: Market Expansion
        (
            "3.1 Market Expansion\n\n"
            "We plan to open offices in Tokyo, Singapore, and Sao Paulo by Q3 2026. "
            "Our total addressable market is estimated at $45B globally, with "
            "significant untapped potential in the Asia-Pacific region where "
            "enterprise cloud adoption is accelerating."
        ),
        # Page 8: Appendix
        (
            "4. Appendix\n\n"
            "Audited financial statements\n"
            "Board of directors listing\n"
            "Corporate governance policies\n"
            "Glossary of financial terms\n"
            "Contact information: investors@acme-corp.example"
        ),
    ]

    return [PageContent(index=i + 1, text=text, token_count=len(text) // 4) for i, text in enumerate(page_texts)]


# The TOC entries that parse_toc should return (matching page 1 content)
_EXPECTED_TOC_ENTRIES = [
    {"structure": "1", "title": "Executive Summary", "page": 2},
    {"structure": "2", "title": "Financial Highlights", "page": 3},
    {"structure": "2.1", "title": "Revenue Breakdown", "page": 4},
    {"structure": "2.2", "title": "Operating Expenses", "page": 5},
    {"structure": "3", "title": "Strategic Outlook", "page": 6},
    {"structure": "3.1", "title": "Market Expansion", "page": 7},
    {"structure": "4", "title": "Appendix", "page": 8},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTreeE2E:
    """End-to-end test: index -> search -> convert to RetrievedChunk."""

    @pytest.fixture
    def pages(self) -> list[PageContent]:
        return _make_annual_report_pages()

    @pytest.fixture
    def indexing_config(self) -> TreeIndexingConfig:
        return TreeIndexingConfig(
            enabled=True,
            toc_scan_pages=5,
            max_pages_per_node=10,
            max_tokens_per_node=20_000,
            generate_summaries=True,
            generate_description=True,
        )

    @pytest.fixture
    def search_config(self) -> TreeSearchConfig:
        return TreeSearchConfig(
            enabled=True,
            max_steps=5,
            max_context_tokens=50_000,
        )

    @pytest.fixture
    def metadata_store(self) -> MagicMock:
        store = MagicMock()
        store.save_tree_index = AsyncMock()
        return store

    # -- Indexing mocks --------------------------------------------------------

    def _mock_detect_toc(self, pages: list[PageContent]) -> AsyncMock:
        """Return a mock detect_toc that finds a TOC with page numbers."""
        mock = AsyncMock()
        # Return that page 1 is a TOC page with page numbers
        mock.return_value = TocInfo(
            path=TocPath.WITH_PAGE_NUMBERS,
            toc_pages=[pages[0]],
        )
        return mock

    def _mock_parse_toc(self) -> AsyncMock:
        """Return a mock parse_toc that returns expected TOC entries."""
        mock = AsyncMock()
        mock.return_value = _EXPECTED_TOC_ENTRIES
        return mock

    def _mock_verify_section_positions(self) -> AsyncMock:
        """Return a mock verify_section_positions that marks all sections as verified."""

        async def _verify(sections, pages, registry):
            return [{**s, "verified": True} for s in sections]

        mock = AsyncMock(side_effect=_verify)
        return mock

    def _mock_generate_node_summary(self) -> AsyncMock:
        """Return a mock for b.GenerateNodeSummary."""

        async def _summarize(title, section_text, baml_options=None):
            return f"Summary of {title}"

        return AsyncMock(side_effect=_summarize)

    def _mock_generate_doc_description(self) -> AsyncMock:
        """Return a mock for b.GenerateDocDescription."""
        return AsyncMock(return_value="ACME Corp Annual Report covering financials and strategy.")

    # -- Search mocks ----------------------------------------------------------

    def _mock_tree_retrieval_step_resolved(self) -> AsyncMock:
        """Return a mock for b.TreeRetrievalStep that immediately resolves pages."""
        from rfnry_rag.retrieval.baml.baml_client.types import ToolResolvedPages

        result = ToolResolvedPages(
            pages="3,4",
            reasoning="Pages 3-4 contain financial highlights and revenue breakdown.",
        )
        return AsyncMock(return_value=result)

    # -- The actual e2e test ---------------------------------------------------

    async def test_full_pipeline(
        self,
        pages: list[PageContent],
        indexing_config: TreeIndexingConfig,
        search_config: TreeSearchConfig,
        metadata_store: MagicMock,
    ) -> None:
        """Build a tree index, search it, and convert results to RetrievedChunks."""
        from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
        from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService

        # -- Step 1: Build the tree index --
        mock_detect = self._mock_detect_toc(pages)
        mock_parse = self._mock_parse_toc()
        mock_verify = self._mock_verify_section_positions()
        mock_summary = self._mock_generate_node_summary()
        mock_description = self._mock_generate_doc_description()

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                mock_detect,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.parse_toc",
                mock_parse,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                mock_verify,
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateNodeSummary",
                mock_summary,
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateDocDescription",
                mock_description,
            ),
        ):
            indexing_service = TreeIndexingService(
                config=indexing_config,
                metadata_store=metadata_store,
                registry=None,
            )
            tree_index = await indexing_service.build_tree_index(
                source_id="src-annual-report",
                doc_name="ACME Corp Annual Report 2025",
                pages=pages,
            )

        # -- Verify tree index structure --
        assert isinstance(tree_index, TreeIndex)
        assert tree_index.source_id == "src-annual-report"
        assert tree_index.doc_name == "ACME Corp Annual Report 2025"
        assert tree_index.doc_description == "ACME Corp Annual Report covering financials and strategy."
        assert tree_index.page_count == 8

        # Verify root-level structure: sections 1, 2, 3, 4
        root_titles = [n.title for n in tree_index.structure]
        assert "Executive Summary" in root_titles
        assert "Financial Highlights" in root_titles
        assert "Strategic Outlook" in root_titles
        assert "Appendix" in root_titles

        # Verify children: "Financial Highlights" has sub-sections 2.1 and 2.2
        fin_node = next(n for n in tree_index.structure if n.title == "Financial Highlights")
        assert len(fin_node.children) == 2
        child_titles = [c.title for c in fin_node.children]
        assert "Revenue Breakdown" in child_titles
        assert "Operating Expenses" in child_titles

        # Verify page ranges were calculated
        exec_node = next(n for n in tree_index.structure if n.title == "Executive Summary")
        assert exec_node.start_index == 2
        # Executive Summary ends just before Financial Highlights (page 3) => end_index = 2
        assert exec_node.end_index == 2

        assert fin_node.start_index == 3
        # Financial Highlights ends just before Strategic Outlook (page 6) => end_index = 5
        assert fin_node.end_index == 5

        # Verify summaries were generated for nodes
        all_nodes = _collect_all_nodes(tree_index.structure)
        for node in all_nodes:
            assert node.summary is not None, f"Node '{node.title}' has no summary"
            assert node.summary.startswith("Summary of ")

        # Verify detect_toc was called with correct args
        mock_detect.assert_called_once()
        call_kwargs = mock_detect.call_args
        assert call_kwargs.kwargs["toc_scan_pages"] == 5

        # Verify parse_toc was called
        mock_parse.assert_called_once()

        # Verify verify_section_positions was called
        mock_verify.assert_called_once()

        # -- Step 2: Search the tree index --
        mock_retrieval_step = self._mock_tree_retrieval_step_resolved()

        with patch(
            "rfnry_rag.retrieval.baml.baml_client.async_client.b.TreeRetrievalStep",
            mock_retrieval_step,
        ):
            search_service = TreeSearchService(
                config=search_config,
                registry=None,
            )
            search_results = await search_service.search(
                query="What was ACME Corp's revenue in 2025?",
                tree_index=tree_index,
                pages=pages,
            )

        # Verify search results
        assert len(search_results) == 1
        result = search_results[0]
        assert isinstance(result, TreeSearchResult)
        assert result.pages == "3,4"
        assert "Financial Highlights" in result.content or "$1.2 billion" in result.content
        assert "revenue" in result.reasoning.lower() or "financial" in result.reasoning.lower()

        # Verify the content includes the actual page text
        assert "Page 3" in result.content
        assert "Page 4" in result.content
        assert "$1.2 billion" in result.content
        assert "Cloud services: $720M" in result.content

        # -- Step 3: Convert to RetrievedChunks --
        chunks = TreeSearchService.to_retrieved_chunks(search_results, tree_index)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.source_id == "src-annual-report"
        assert chunk.score == 1.0
        assert chunk.content == result.content
        assert chunk.chunk_id.startswith("tree-")
        assert chunk.source_metadata["name"] == "ACME Corp Annual Report 2025"
        assert chunk.source_metadata["tree_pages"] == "3,4"
        assert (
            "revenue" in chunk.source_metadata["tree_reasoning"].lower()
            or "financial" in chunk.source_metadata["tree_reasoning"].lower()
        )

    async def test_search_with_drill_down_then_resolve(
        self,
        pages: list[PageContent],
        indexing_config: TreeIndexingConfig,
        search_config: TreeSearchConfig,
        metadata_store: MagicMock,
    ) -> None:
        """Test multi-step search: drill-down into a subtree, then resolve."""
        from rfnry_rag.retrieval.baml.baml_client.types import ToolDrillDown, ToolResolvedPages
        from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
        from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService

        # Build the tree index first (reuse mocks)
        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                self._mock_detect_toc(pages),
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.parse_toc",
                self._mock_parse_toc(),
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                self._mock_verify_section_positions(),
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateNodeSummary",
                self._mock_generate_node_summary(),
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateDocDescription",
                self._mock_generate_doc_description(),
            ),
        ):
            indexing_service = TreeIndexingService(
                config=indexing_config,
                metadata_store=metadata_store,
                registry=None,
            )
            tree_index = await indexing_service.build_tree_index(
                source_id="src-drill-test",
                doc_name="ACME Corp Annual Report 2025",
                pages=pages,
            )

        # Find the node_id for "Financial Highlights" to drill into
        fin_node = next(n for n in tree_index.structure if n.title == "Financial Highlights")
        fin_node_id = fin_node.node_id

        # Set up a 2-step search: first drill-down, then resolve
        call_count = 0
        drill_result = ToolDrillDown(
            node_id=fin_node_id,
            reasoning="Financial section likely contains revenue details.",
        )
        resolve_result = ToolResolvedPages(
            pages="4",
            reasoning="Page 4 contains the revenue breakdown.",
        )

        async def _step_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return drill_result
            return resolve_result

        mock_step = AsyncMock(side_effect=_step_side_effect)

        with patch(
            "rfnry_rag.retrieval.baml.baml_client.async_client.b.TreeRetrievalStep",
            mock_step,
        ):
            search_service = TreeSearchService(config=search_config, registry=None)
            results = await search_service.search(
                query="What is the cloud revenue?",
                tree_index=tree_index,
                pages=pages,
            )

        assert len(results) == 1
        assert results[0].pages == "4"
        assert "Cloud services: $720M" in results[0].content
        # Two calls: one drill-down, one resolve
        assert mock_step.call_count == 2

        # Convert and verify
        chunks = TreeSearchService.to_retrieved_chunks(results, tree_index)
        assert len(chunks) == 1
        assert chunks[0].source_id == "src-drill-test"

    async def test_search_with_fetch_then_resolve(
        self,
        pages: list[PageContent],
        indexing_config: TreeIndexingConfig,
        search_config: TreeSearchConfig,
        metadata_store: MagicMock,
    ) -> None:
        """Test multi-step search: fetch pages for context, then resolve."""
        from rfnry_rag.retrieval.baml.baml_client.types import ToolFetchPages, ToolResolvedPages
        from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
        from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService

        # Build the tree index
        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                self._mock_detect_toc(pages),
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.parse_toc",
                self._mock_parse_toc(),
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                self._mock_verify_section_positions(),
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateNodeSummary",
                self._mock_generate_node_summary(),
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateDocDescription",
                self._mock_generate_doc_description(),
            ),
        ):
            indexing_service = TreeIndexingService(
                config=indexing_config,
                metadata_store=metadata_store,
                registry=None,
            )
            tree_index = await indexing_service.build_tree_index(
                source_id="src-fetch-test",
                doc_name="ACME Corp Annual Report 2025",
                pages=pages,
            )

        # Set up: fetch page 6 first (strategic outlook), then resolve pages 6-7
        call_count = 0
        fetch_result = ToolFetchPages(
            pages="6",
            reasoning="Checking strategic outlook for market expansion details.",
        )
        resolve_result = ToolResolvedPages(
            pages="6,7",
            reasoning="Pages 6-7 cover strategic outlook and market expansion.",
        )

        async def _step_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fetch_result
            return resolve_result

        mock_step = AsyncMock(side_effect=_step_side_effect)

        with patch(
            "rfnry_rag.retrieval.baml.baml_client.async_client.b.TreeRetrievalStep",
            mock_step,
        ):
            search_service = TreeSearchService(config=search_config, registry=None)
            results = await search_service.search(
                query="What are ACME's expansion plans?",
                tree_index=tree_index,
                pages=pages,
            )

        assert len(results) == 1
        assert results[0].pages == "6,7"
        assert "Tokyo" in results[0].content or "APAC" in results[0].content

        # Verify the search made two BAML calls (fetch then resolve)
        assert mock_step.call_count == 2

    async def test_tree_index_structure_integrity(
        self,
        pages: list[PageContent],
        indexing_config: TreeIndexingConfig,
        metadata_store: MagicMock,
    ) -> None:
        """Verify detailed structural integrity of the built tree index."""
        from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                self._mock_detect_toc(pages),
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.parse_toc",
                self._mock_parse_toc(),
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                self._mock_verify_section_positions(),
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateNodeSummary",
                self._mock_generate_node_summary(),
            ),
            patch(
                "rfnry_rag.retrieval.baml.baml_client.async_client.b.GenerateDocDescription",
                self._mock_generate_doc_description(),
            ),
        ):
            service = TreeIndexingService(
                config=indexing_config,
                metadata_store=metadata_store,
                registry=None,
            )
            tree_index = await service.build_tree_index(
                source_id="src-structure-test",
                doc_name="Structure Test",
                pages=pages,
            )

        # 4 root nodes (1, 2, 3, 4)
        assert len(tree_index.structure) == 4

        # Section 2 has 2 children (2.1, 2.2)
        sec2 = tree_index.structure[1]
        assert sec2.title == "Financial Highlights"
        assert len(sec2.children) == 2

        # Section 3 has 1 child (3.1)
        sec3 = tree_index.structure[2]
        assert sec3.title == "Strategic Outlook"
        assert len(sec3.children) == 1
        assert sec3.children[0].title == "Market Expansion"

        # Sections 1 and 4 are leaf nodes
        assert len(tree_index.structure[0].children) == 0
        assert len(tree_index.structure[3].children) == 0

        # All nodes have unique IDs
        all_nodes = _collect_all_nodes(tree_index.structure)
        node_ids = [n.node_id for n in all_nodes]
        assert len(node_ids) == len(set(node_ids)), "Node IDs must be unique"

        # Page ranges: no gaps, no overlaps among siblings at each level
        for i, node in enumerate(tree_index.structure):
            assert node.start_index >= 1, f"Node {node.title} has invalid start_index"
            assert node.end_index >= node.start_index, f"Node {node.title} has end < start"
            if i + 1 < len(tree_index.structure):
                next_node = tree_index.structure[i + 1]
                assert node.end_index < next_node.start_index, (
                    f"Overlap: {node.title} ends at {node.end_index}, "
                    f"{next_node.title} starts at {next_node.start_index}"
                )

    async def test_to_retrieved_chunks_multiple_results(self) -> None:
        """Verify to_retrieved_chunks handles multiple TreeSearchResult entries."""
        from datetime import UTC, datetime

        from rfnry_rag.retrieval.modules.retrieval.tree.service import TreeSearchService

        tree_index = TreeIndex(
            source_id="src-multi",
            doc_name="Multi Result Doc",
            doc_description=None,
            structure=[],
            page_count=10,
            created_at=datetime.now(UTC),
        )

        results = [
            TreeSearchResult(
                node_id="n1",
                title="Section A",
                pages="1-3",
                content="Content from pages 1-3",
                reasoning="Relevant section A",
            ),
            TreeSearchResult(
                node_id="n2",
                title="Section B",
                pages="7,8",
                content="Content from pages 7-8",
                reasoning="Relevant section B",
            ),
        ]

        chunks = TreeSearchService.to_retrieved_chunks(results, tree_index)

        assert len(chunks) == 2

        assert chunks[0].chunk_id == "tree-n1-0"
        assert chunks[0].source_id == "src-multi"
        assert chunks[0].content == "Content from pages 1-3"
        assert chunks[0].score == 1.0
        assert chunks[0].source_metadata["tree_pages"] == "1-3"

        assert chunks[1].chunk_id == "tree-n2-1"
        assert chunks[1].source_id == "src-multi"
        assert chunks[1].content == "Content from pages 7-8"
        assert chunks[1].score == 1.0
        assert chunks[1].source_metadata["tree_pages"] == "7,8"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _collect_all_nodes(nodes: list[TreeNode]) -> list[TreeNode]:
    """Flatten a tree of TreeNodes into a flat list."""
    result: list[TreeNode] = []
    for node in nodes:
        result.append(node)
        if node.children:
            result.extend(_collect_all_nodes(node.children))
    return result
