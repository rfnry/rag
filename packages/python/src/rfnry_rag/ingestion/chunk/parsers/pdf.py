from pathlib import Path

import pymupdf

from rfnry_rag.common.logging import get_logger
from rfnry_rag.exceptions import ParseError
from rfnry_rag.ingestion.models import ParsedPage

logger = get_logger("chunk/ingestion/parse")

_MAX_PDF_BYTES = 500 * 1024 * 1024  # 500 MiB — defensive cap against OOM on huge PDFs
_MAX_PDF_PAGES = 5_000  # defensive cap against page-count DoS


class PDFParser:
    def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        size = Path(file_path).stat().st_size
        if size > _MAX_PDF_BYTES:
            raise ValueError(f"PDF size {size} bytes exceeds cap {_MAX_PDF_BYTES}")

        result: list[ParsedPage] = []
        try:
            with pymupdf.open(file_path) as doc:
                if doc.page_count > _MAX_PDF_PAGES:
                    raise ValueError(f"PDF has {doc.page_count} pages, exceeds cap {_MAX_PDF_PAGES}")
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
        except ValueError:
            raise
        except Exception as exc:
            raise ParseError(f"Failed to parse PDF '{file_path}': {exc}") from exc
        return result
