from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ParsedPage:
    page_number: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkedContent:
    content: str
    page_number: int | None = None
    section: str | None = None
    chunk_index: int = 0
    context: str = ""
    contextualized: str = ""
    parent_id: str | None = None
    chunk_type: Literal["child", "parent"] = "child"
    was_hard_split: bool = False
    synthetic_queries: list[str] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Backward-compatible default: contextualized text + any synthetic queries.

        Existing callers that read this property keep getting expansion-aware text
        for free. Callers that need explicit gating use ``text_for_embedding`` /
        ``text_for_bm25`` directly with ``include_synthetic=...``.
        """
        return self.text_for_embedding(include_synthetic=True)

    def text_for_embedding(self, *, include_synthetic: bool = True) -> str:
        """Text to send to the embedding model. Folds in synthetic queries when allowed."""
        return self._compose(include_synthetic=include_synthetic)

    def text_for_bm25(self, *, include_synthetic: bool = True) -> str:
        """Text to send to the BM25 indexer. Folds in synthetic queries when allowed."""
        return self._compose(include_synthetic=include_synthetic)

    def _compose(self, *, include_synthetic: bool) -> str:
        base = self.contextualized or self.content
        if include_synthetic and self.synthetic_queries:
            joined = "\n".join(self.synthetic_queries)
            # Trailing block lets the embedding/BM25 model see both the passage
            # and the user-vocabulary questions (docT5query-style expansion).
            return f"{base}\n\nQuestions this passage answers:\n{joined}"
        return base
