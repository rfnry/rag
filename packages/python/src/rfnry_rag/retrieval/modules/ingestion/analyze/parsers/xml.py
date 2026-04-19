from pathlib import Path

from lxml import etree

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.analyze.models import DiscoveredEntity, PageAnalysis

logger = get_logger("analyze/ingestion/analyze/xml")

L5X_ROOT_TAG = "RSLogix5000Content"


def is_l5x(file_path: Path) -> bool:
    """Check if an XML file is L5X format by root element."""
    try:
        for _event, elem in etree.iterparse(str(file_path), events=("start",)):
            return elem.tag == L5X_ROOT_TAG
    except etree.XMLSyntaxError:
        return False
    return False


def parse_xml(file_path: Path) -> list[PageAnalysis]:
    """Parse a generic XML file into PageAnalysis objects grouped by top-level elements."""
    tree = etree.parse(str(file_path))
    root = tree.getroot()
    analyses = []

    for i, child in enumerate(root):
        tag = child.tag
        text_parts: list[str] = []
        entities: list[DiscoveredEntity] = []

        _render_element(child, text_parts, entities, depth=0)

        description = "\n".join(text_parts) if text_parts else f"Element: {tag}"
        analyses.append(
            PageAnalysis(
                page_number=i + 1,
                description=description,
                entities=entities,
                page_type="xml_element",
                metadata={"element_tag": tag},
            )
        )

    logger.info("parsed %d top-level elements, %d groups", len(root), len(analyses))
    return analyses


def _render_element(
    elem: etree._Element,
    text_parts: list[str],
    entities: list[DiscoveredEntity],
    depth: int,
) -> None:
    indent = "  " * depth
    attrs = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())
    header = f"{indent}{elem.tag}" + (f" ({attrs})" if attrs else "")
    text_parts.append(header)

    if elem.text and elem.text.strip():
        text_parts.append(f"{indent}  {elem.text.strip()}")

    name = elem.get("Name") or elem.get("name") or elem.get("id") or elem.get("Id")
    if name:
        entities.append(
            DiscoveredEntity(
                name=name,
                category=elem.tag.lower(),
                context=f"depth {depth}, attributes: {attrs}" if attrs else f"depth {depth}",
            )
        )

    for child in elem:
        _render_element(child, text_parts, entities, depth + 1)
