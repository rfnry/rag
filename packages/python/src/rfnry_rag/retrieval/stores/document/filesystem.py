import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from rfnry_rag.retrieval.common.logging import get_logger
from rfnry_rag.retrieval.common.models import ContentMatch
from rfnry_rag.retrieval.stores.document.excerpt import extract_window

logger = get_logger("stores/document/filesystem")

_DEFAULT_KB = "_default"
_UNTYPED = "_untyped"

# Path-component whitelist: letters, digits, _, -, dot. No slashes, no leading/trailing dot,
# 1-128 chars. Applied to knowledge_id and source_type before they become directory names
# to prevent traversal outside the store root.
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_\-.]{0,127}$")


def _safe_path_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.match(value) or value in {".", ".."}:
        raise ValueError(f"invalid {field} path component: {value!r}")
    return value


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class _BM25Entry:
    index: BM25Okapi | None
    entries: list[dict[str, Any]] = field(default_factory=list)
    last_used: float = 0.0


class FilesystemDocumentStore:
    """Document store backed by markdown files on the filesystem with BM25 + exact search."""

    @staticmethod
    def _cache_key(knowledge_id: str | None) -> str:
        return knowledge_id if knowledge_id is not None else _DEFAULT_KB

    def __init__(self, base_path: str, max_cached_indexes: int = 16) -> None:
        self._base_path = Path(base_path)
        self._max_cached_indexes = max_cached_indexes
        self._cache: dict[str, _BM25Entry] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await asyncio.to_thread(self._base_path.mkdir, parents=True, exist_ok=True)

    async def store_content(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        title: str,
        content: str,
    ) -> None:
        kb_dir = _safe_path_component(knowledge_id, field="knowledge_id") if knowledge_id is not None else _DEFAULT_KB
        st_dir = _safe_path_component(source_type, field="source_type") if source_type is not None else _UNTYPED
        _safe_path_component(source_id, field="source_id")

        old_file = await self._find_file(source_id)
        if old_file is not None:
            await asyncio.to_thread(old_file.unlink, missing_ok=True)
            await self._clean_empty_parents(old_file)

        directory = self._base_path / kb_dir / st_dir
        resolved = directory.resolve()
        if not resolved.is_relative_to(self._base_path.resolve()):
            raise ValueError(f"resolved path escapes base_path: {resolved}")
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)

        file_path = directory / f"{source_id}.md"
        frontmatter = (
            f"---\nsource_id: {source_id}\ntitle: {title}\n"
            f"knowledge_id: {knowledge_id}\nsource_type: {source_type}\n---\n"
        )
        file_content = frontmatter + content
        await asyncio.to_thread(file_path.write_text, file_content, "utf-8")

        cache_key = self._cache_key(knowledge_id)
        async with self._lock:
            self._cache.pop(cache_key, None)

        logger.info("stored content source_id=%s in %s", source_id, file_path)

    async def search_content(
        self,
        query: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        top_k: int = 5,
    ) -> list[ContentMatch]:
        if knowledge_id is not None:
            _safe_path_component(knowledge_id, field="knowledge_id")
        if source_type is not None:
            _safe_path_component(source_type, field="source_type")
        cache_key = self._cache_key(knowledge_id)

        if cache_key not in self._cache:
            await self._build_index(knowledge_id)

        entry_obj = self._cache.get(cache_key)
        if entry_obj is None or not entry_obj.entries:
            return []

        entry_obj.last_used = time.monotonic()
        entries = (
            [e for e in entry_obj.entries if e.get("source_type") == source_type]
            if source_type is not None
            else entry_obj.entries
        )

        if not entries:
            return []

        exact_matches: dict[str, ContentMatch] = {}
        for entry in entries:
            if query.lower() in entry["content"].lower():
                excerpt = extract_window(entry["content"], query)
                exact_matches[entry["source_id"]] = ContentMatch(
                    source_id=entry["source_id"],
                    title=entry["title"],
                    excerpt=excerpt,
                    score=1.0,
                    match_type="exact",
                    source_type=entry["source_type"],
                )

        bm25_matches: dict[str, ContentMatch] = {}
        if entry_obj.index is not None:
            tokenized_query = _tokenize(query)
            scores = entry_obj.index.get_scores(tokenized_query)

            for score, cached in sorted(zip(scores, entry_obj.entries, strict=True), key=lambda x: x[0], reverse=True):
                if score <= 0:
                    break
                if source_type is not None and cached.get("source_type") != source_type:
                    continue
                sid = cached["source_id"]
                if sid not in bm25_matches:
                    excerpt = extract_window(cached["content"], query)
                    bm25_matches[sid] = ContentMatch(
                        source_id=sid,
                        title=cached["title"],
                        excerpt=excerpt,
                        score=float(score),
                        match_type="fulltext",
                        source_type=cached.get("source_type"),
                    )

        merged: dict[str, ContentMatch] = {}
        for sid, match in exact_matches.items():
            merged[sid] = match
        for sid, match in bm25_matches.items():
            if sid not in merged:
                merged[sid] = match

        results = sorted(merged.values(), key=lambda m: m.score, reverse=True)
        return results[:top_k]

    async def delete_content(self, source_id: str) -> None:
        file_path = await self._find_file(source_id)
        if file_path is None:
            return

        parsed = await asyncio.to_thread(self._parse_file, file_path)
        cache_key = self._cache_key(parsed["knowledge_id"])

        await asyncio.to_thread(file_path.unlink, missing_ok=True)
        await self._clean_empty_parents(file_path)

        async with self._lock:
            self._cache.pop(cache_key, None)

        logger.info("deleted content source_id=%s", source_id)

    async def shutdown(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def _find_file(self, source_id: str) -> Path | None:
        filename = f"{source_id}.md"

        def _search() -> Path | None:
            for p in self._base_path.rglob(filename):
                return p
            return None

        return await asyncio.to_thread(_search)

    def _parse_file(self, file_path: Path) -> dict[str, Any]:
        text = file_path.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        meta: dict[str, Any] = {
            "source_id": "",
            "title": "",
            "knowledge_id": None,
            "source_type": None,
            "content": "",
        }
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            content = parts[2].lstrip("\n")
            for line in frontmatter.splitlines():
                if ": " in line:
                    key, value = line.split(": ", 1)
                    key = key.strip()
                    value = value.strip()
                    if value == "None":
                        value = None  # type: ignore[assignment]
                    if key in meta:
                        meta[key] = value
            meta["content"] = content
        else:
            meta["content"] = text
        return meta

    async def _load_entries(
        self, knowledge_id: str | None = None, source_type: str | None = None
    ) -> list[dict[str, Any]]:
        kb_dir = _safe_path_component(knowledge_id, field="knowledge_id") if knowledge_id is not None else _DEFAULT_KB
        st_dir = _safe_path_component(source_type, field="source_type") if source_type is not None else None

        search_dir = self._base_path / kb_dir / st_dir if st_dir is not None else self._base_path / kb_dir

        def _load() -> list[dict[str, Any]]:
            if not search_dir.exists():
                return []
            entries = []
            for md_file in search_dir.rglob("*.md"):
                parsed = self._parse_file(md_file)
                entries.append(parsed)
            return entries

        return await asyncio.to_thread(_load)

    async def _build_index(self, knowledge_id: str | None) -> None:
        cache_key = self._cache_key(knowledge_id)

        async with self._lock:
            if cache_key in self._cache:
                return

            entries = await self._load_entries(knowledge_id)

            self._evict_lru()

            if not entries:
                self._cache[cache_key] = _BM25Entry(index=None, last_used=time.monotonic())
                return

            loop = asyncio.get_running_loop()
            tokenized = await loop.run_in_executor(None, lambda: [_tokenize(e["content"]) for e in entries])
            index = await loop.run_in_executor(None, lambda: BM25Okapi(tokenized))
            self._cache[cache_key] = _BM25Entry(index=index, entries=entries, last_used=time.monotonic())
            logger.info("built bm25 index for knowledge_id=%s: %d entries", knowledge_id, len(entries))

    def _evict_lru(self) -> None:
        if len(self._cache) < self._max_cached_indexes:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].last_used)
        del self._cache[oldest_key]
        logger.info("evicted bm25 index for key=%s (lru)", oldest_key)

    async def _clean_empty_parents(self, file_path: Path) -> None:
        """Remove empty parent directories up to (but not including) base_path."""

        def _clean() -> None:
            parent = file_path.parent
            while parent != self._base_path and parent.exists():
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break

        await asyncio.to_thread(_clean)
