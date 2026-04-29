"""Tests for TreeIndexingService."""

import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rfnry_rag.retrieval.common.errors import TreeIndexingError
from rfnry_rag.retrieval.common.models import TreeIndex
from rfnry_rag.retrieval.modules.ingestion.tree.service import TreeIndexingService
from rfnry_rag.retrieval.modules.ingestion.tree.toc import PageContent, TocInfo, TocPath


def _make_pages(n: int) -> list[PageContent]:
    """Create a list of mock PageContent objects."""
    return [PageContent(index=i + 1, text=f"Page {i + 1} content", token_count=100) for i in range(n)]


@dataclass
class _FakeConfig:
    enabled: bool = True
    model: object = None
    toc_scan_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20_000
    generate_summaries: bool = True
    generate_description: bool = True


class TestIndexWithToc:
    """Test build_tree_index when a TOC is detected."""

    async def test_index_with_toc(self):
        pages = _make_pages(30)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        toc_pages = pages[:2]
        toc_info = TocInfo(path=TocPath.WITH_PAGE_NUMBERS, toc_pages=toc_pages)

        parsed_entries = [
            {"structure": "1", "title": "Introduction", "page": 1},
            {"structure": "2", "title": "Methods", "page": 10},
            {"structure": "3", "title": "Results", "page": 20},
        ]

        verified_entries = [{**e, "verified": True} for e in parsed_entries]

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                new_callable=AsyncMock,
                return_value=toc_info,
            ) as mock_detect,
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.parse_toc",
                new_callable=AsyncMock,
                return_value=parsed_entries,
            ) as mock_parse,
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                new_callable=AsyncMock,
                return_value=verified_entries,
            ) as mock_verify,
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_summaries",
                new_callable=AsyncMock,
            ) as mock_summaries,
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_doc_description",
                new_callable=AsyncMock,
                return_value="A document about research methods.",
            ) as mock_description,
        ):
            service = TreeIndexingService(config, metadata_store, registry=None)
            result = await service.build_tree_index("src-001", "test.pdf", pages)

        assert isinstance(result, TreeIndex)
        assert result.source_id == "src-001"
        assert result.doc_name == "test.pdf"
        assert result.doc_description == "A document about research methods."
        assert result.page_count == 30
        assert len(result.structure) == 3
        assert result.structure[0].title == "Introduction"
        assert result.structure[1].title == "Methods"
        assert result.structure[2].title == "Results"

        mock_detect.assert_called_once()
        mock_parse.assert_called_once()
        mock_verify.assert_called_once()
        mock_summaries.assert_called_once()
        mock_description.assert_called_once()


class TestIndexWithTocWithoutPageNumbers:
    """Test build_tree_index when TOC is found but has no page numbers."""

    async def test_index_with_toc_without_page_numbers(self):
        pages = _make_pages(20)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        toc_pages = pages[:1]
        toc_info = TocInfo(path=TocPath.WITHOUT_PAGE_NUMBERS, toc_pages=toc_pages)

        parsed_entries = [
            {"structure": "1", "title": "Chapter 1", "page": None},
            {"structure": "2", "title": "Chapter 2", "page": None},
        ]

        entries_with_pages = [
            {"structure": "1", "title": "Chapter 1", "page": 1},
            {"structure": "2", "title": "Chapter 2", "page": 10},
        ]

        verified_entries = [{**e, "verified": True} for e in entries_with_pages]

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                new_callable=AsyncMock,
                return_value=toc_info,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.parse_toc",
                new_callable=AsyncMock,
                return_value=parsed_entries,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.find_section_starts",
                new_callable=AsyncMock,
                return_value=entries_with_pages,
            ) as mock_find,
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                new_callable=AsyncMock,
                return_value=verified_entries,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_summaries",
                new_callable=AsyncMock,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_doc_description",
                new_callable=AsyncMock,
                return_value="A test document.",
            ),
        ):
            service = TreeIndexingService(config, metadata_store, registry=None)
            result = await service.build_tree_index("src-002", "test2.pdf", pages)

        assert isinstance(result, TreeIndex)
        assert len(result.structure) == 2
        mock_find.assert_called_once()


class TestIndexNoToc:
    """Test build_tree_index when no TOC is detected (LLM extraction path)."""

    async def test_index_no_toc(self):
        pages = _make_pages(10)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        toc_info = TocInfo(path=TocPath.NO_TOC, toc_pages=[])

        extracted_sections = [
            {"structure": "1", "title": "Overview", "page": 1},
            {"structure": "2", "title": "Details", "page": 5},
        ]

        verified_sections = [{**s, "verified": True} for s in extracted_sections]

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                new_callable=AsyncMock,
                return_value=toc_info,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._extract_structure_no_toc",
                new_callable=AsyncMock,
                return_value=extracted_sections,
            ) as mock_extract,
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                new_callable=AsyncMock,
                return_value=verified_sections,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_summaries",
                new_callable=AsyncMock,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_doc_description",
                new_callable=AsyncMock,
                return_value="An overview document.",
            ),
        ):
            service = TreeIndexingService(config, metadata_store, registry=None)
            result = await service.build_tree_index("src-003", "test3.pdf", pages)

        assert isinstance(result, TreeIndex)
        assert result.source_id == "src-003"
        assert len(result.structure) == 2
        assert result.structure[0].title == "Overview"
        assert result.structure[1].title == "Details"
        mock_extract.assert_called_once()


class TestIndexSavesToMetadataStore:
    """Test that save_tree_index correctly calls the metadata store."""

    async def test_index_saves_to_metadata_store(self):
        config = _FakeConfig()
        metadata_store = AsyncMock()

        pages = _make_pages(10)
        toc_info = TocInfo(path=TocPath.NO_TOC, toc_pages=[])

        extracted_sections = [
            {"structure": "1", "title": "Section A", "page": 1},
        ]

        verified_sections = [{**s, "verified": True} for s in extracted_sections]

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                new_callable=AsyncMock,
                return_value=toc_info,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._extract_structure_no_toc",
                new_callable=AsyncMock,
                return_value=extracted_sections,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.verify_section_positions",
                new_callable=AsyncMock,
                return_value=verified_sections,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_summaries",
                new_callable=AsyncMock,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._generate_doc_description",
                new_callable=AsyncMock,
                return_value="Single section doc.",
            ),
        ):
            service = TreeIndexingService(config, metadata_store, registry=None)
            tree_index = await service.build_tree_index("src-004", "doc.pdf", pages)

        # Now save it
        await service.save_tree_index(tree_index)

        metadata_store.save_tree_index.assert_called_once()
        call_args = metadata_store.save_tree_index.call_args
        assert call_args.kwargs["source_id"] == "src-004"

        # Verify the JSON is valid and round-trips
        saved_json = call_args.kwargs["tree_index_json"]
        parsed = json.loads(saved_json)
        assert parsed["source_id"] == "src-004"
        assert parsed["doc_name"] == "doc.pdf"
        assert parsed["doc_description"] == "Single section doc."
        assert len(parsed["structure"]) == 1
        assert parsed["structure"][0]["title"] == "Section A"


class TestIndexRaisesOnEmptySections:
    """Test that TreeIndexingError is raised when no sections are found."""

    async def test_index_raises_on_empty_sections_no_toc(self):
        pages = _make_pages(5)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        toc_info = TocInfo(path=TocPath.NO_TOC, toc_pages=[])

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                new_callable=AsyncMock,
                return_value=toc_info,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.TreeIndexingService._extract_structure_no_toc",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            service = TreeIndexingService(config, metadata_store, registry=None)
            with pytest.raises(TreeIndexingError, match="no sections found"):
                await service.build_tree_index("src-005", "empty.pdf", pages)

    async def test_index_raises_on_empty_sections_with_toc(self):
        pages = _make_pages(5)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        toc_pages = pages[:1]
        toc_info = TocInfo(path=TocPath.WITH_PAGE_NUMBERS, toc_pages=toc_pages)

        with (
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.detect_toc",
                new_callable=AsyncMock,
                return_value=toc_info,
            ),
            patch(
                "rfnry_rag.retrieval.modules.ingestion.tree.service.parse_toc",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            service = TreeIndexingService(config, metadata_store, registry=None)
            with pytest.raises(TreeIndexingError, match="no sections found"):
                await service.build_tree_index("src-006", "empty2.pdf", pages)


BAML_CLIENT = "rfnry_rag.retrieval.baml.baml_client.async_client.b"


class TestExtractStructureNoToc:
    """Test the _extract_structure_no_toc method directly."""

    async def test_extract_single_group(self):
        """Pages fitting in a single group use ExtractDocumentStructure."""
        pages = _make_pages(10)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        mock_section = SimpleNamespace(structure="1", title="Intro", start_page=1)
        mock_result = SimpleNamespace(sections=[mock_section])

        mock_b = AsyncMock()
        mock_b.ExtractDocumentStructure = AsyncMock(return_value=mock_result)

        with patch(BAML_CLIENT, mock_b):
            service = TreeIndexingService(config, metadata_store, registry=None)
            sections = await service._extract_structure_no_toc(pages)

        assert len(sections) == 1
        assert sections[0]["structure"] == "1"
        assert sections[0]["title"] == "Intro"
        assert sections[0]["page"] == 1
        mock_b.ExtractDocumentStructure.assert_called_once()

    async def test_extract_multiple_groups(self):
        """Pages spanning multiple groups use Continue for subsequent groups."""
        pages = _make_pages(30)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        section1 = SimpleNamespace(structure="1", title="Part 1", start_page=1)
        section2 = SimpleNamespace(structure="2", title="Part 2", start_page=21)
        result1 = SimpleNamespace(sections=[section1])
        result2 = SimpleNamespace(sections=[section2])

        mock_b = AsyncMock()
        mock_b.ExtractDocumentStructure = AsyncMock(return_value=result1)
        mock_b.ContinueDocumentStructure = AsyncMock(return_value=result2)

        with patch(BAML_CLIENT, mock_b):
            service = TreeIndexingService(config, metadata_store, registry=None)
            sections = await service._extract_structure_no_toc(pages)

        assert len(sections) == 2
        assert sections[0]["title"] == "Part 1"
        assert sections[1]["title"] == "Part 2"
        mock_b.ExtractDocumentStructure.assert_called_once()
        mock_b.ContinueDocumentStructure.assert_called_once()


class TestGenerateSummaries:
    """Test summary generation for tree nodes."""

    async def test_generate_summaries_sets_summary_on_nodes(self):
        pages = _make_pages(10)
        config = _FakeConfig()
        metadata_store = AsyncMock()

        mock_b = AsyncMock()
        mock_b.GenerateNodeSummary = AsyncMock(return_value="A summary.")

        with patch(BAML_CLIENT, mock_b):
            service = TreeIndexingService(config, metadata_store, registry=None)

            from rfnry_rag.retrieval.common.models import TreeNode

            nodes = [
                TreeNode(node_id="0001", title="Chapter 1", start_index=1, end_index=5),
                TreeNode(node_id="0002", title="Chapter 2", start_index=6, end_index=10),
            ]

            await service._generate_summaries(nodes, pages)

        assert nodes[0].summary == "A summary."
        assert nodes[1].summary == "A summary."
        assert mock_b.GenerateNodeSummary.call_count == 2


class TestGenerateDocDescription:
    """Test doc description generation."""

    async def test_generate_doc_description_returns_string(self):
        config = _FakeConfig()
        metadata_store = AsyncMock()

        mock_b = AsyncMock()
        mock_b.GenerateDocDescription = AsyncMock(return_value="This document covers research methods.")

        with patch(BAML_CLIENT, mock_b):
            service = TreeIndexingService(config, metadata_store, registry=None)

            from rfnry_rag.retrieval.common.models import TreeNode

            roots = [
                TreeNode(node_id="0001", title="Intro", start_index=1, end_index=10),
            ]

            result = await service._generate_doc_description(roots)

        assert result == "This document covers research methods."
        mock_b.GenerateDocDescription.assert_called_once()
