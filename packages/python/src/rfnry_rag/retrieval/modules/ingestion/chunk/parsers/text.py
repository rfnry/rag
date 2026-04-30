from pathlib import Path

from rfnry_rag.common.logging import get_logger
from rfnry_rag.retrieval.common.errors import ParseError
from rfnry_rag.retrieval.modules.ingestion.models import ParsedPage

logger = get_logger("chunk/ingestion/parse")


class TextParser:
    def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]:
        try:
            content = Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(f"File is not valid UTF-8: {file_path}") from exc
        if not content.strip():
            return []
        if pages is not None and 1 not in pages:
            return []
        return [ParsedPage(page_number=1, content=content, metadata={"char_count": len(content)})]
