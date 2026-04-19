import pymupdf

from rfnry_rag.retrieval.common.errors import ParseError
from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage

logger = get_logger("chunk/ingestion/parse")


class PDFParser:
    def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        result: list[ParsedPage] = []
        try:
            with pymupdf.open(file_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    if pages is not None and page_num not in pages:
                        continue

                    text = page.get_text().strip()

                    if not text:
                        try:
                            tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                            text = page.get_text(textpage=tp).strip()
                            if text:
                                logger.info("page %d — ocr fallback extracted %d chars", page_num, len(text))
                        except RuntimeError:
                            logger.warning("page %d — ocr unavailable (tesseract not installed?)", page_num)
                            continue

                    if text:
                        result.append(
                            ParsedPage(
                                page_number=page_num,
                                content=text,
                                metadata={"char_count": len(text)},
                            )
                        )
        except Exception as exc:
            raise ParseError(f"Failed to parse PDF '{file_path}': {exc}") from exc
        return result
