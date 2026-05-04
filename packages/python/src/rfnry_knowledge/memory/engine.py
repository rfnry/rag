from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.config.memory import MemoryEngineConfig
from rfnry_knowledge.ingestion.embeddings.batching import embed_batched
from rfnry_knowledge.memory.models import (
    ExtractedMemory,
    Interaction,
    MemoryRow,
    MemorySearchResult,
)
from rfnry_knowledge.models import VectorPoint
from rfnry_knowledge.observability import Observability
from rfnry_knowledge.observability.context import _reset_obs, _set_obs
from rfnry_knowledge.providers import build_registry
from rfnry_knowledge.retrieval.methods.entity import EntityRetrieval
from rfnry_knowledge.retrieval.methods.keyword import KeywordRetrieval
from rfnry_knowledge.retrieval.methods.semantic import SemanticRetrieval
from rfnry_knowledge.retrieval.search.service import RetrievalService
from rfnry_knowledge.stores.graph.models import GraphEntity
from rfnry_knowledge.telemetry import (
    MemoryAddTelemetryRow,
    MemorySearchTelemetryRow,
    Telemetry,
)
from rfnry_knowledge.telemetry.usage import instrument_baml_call

logger = get_logger("memory.engine")


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


class MemoryEngine:
    def __init__(self, config: MemoryEngineConfig) -> None:
        self._cfg = config
        self._obs: Observability = config.observability
        self._tel: Telemetry = config.telemetry
        self._initialized = False
        self._stores_opened = False

    async def initialize(self) -> None:
        cfg = self._cfg
        self._stores_opened = True
        if cfg.metadata_store is not None:
            await cfg.metadata_store.initialize()
        if cfg.document_store is not None:
            await cfg.document_store.initialize()
        if cfg.graph_store is not None:
            await cfg.graph_store.initialize()
        vector_size = await cfg.ingestion.embeddings.embedding_dimension()
        await cfg.vector_store.initialize(vector_size)
        self._initialized = True

    async def shutdown(self) -> None:
        if not self._stores_opened:
            return
        self._stores_opened = False
        cfg = self._cfg
        for store in (cfg.vector_store, cfg.graph_store, cfg.document_store, cfg.metadata_store):
            if store is None:
                continue
            try:
                await store.shutdown()
            except Exception:
                logger.exception("error shutting down %s", type(store).__name__)
        self._initialized = False

    async def __aenter__(self) -> MemoryEngine:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MemoryEngine not initialized — use async with or call initialize()")

    async def add(self, interaction: Interaction, memory_id: str) -> tuple[MemoryRow, ...]:
        self._check_initialized()
        if not interaction.turns:
            raise ValueError("interaction.turns must not be empty")
        if not memory_id or not memory_id.strip():
            raise ValueError("memory_id must not be blank")

        cfg = self._cfg
        ing = cfg.ingestion
        tel_row = MemoryAddTelemetryRow(memory_id=memory_id, outcome="success")
        obs_token = _set_obs(self._obs)
        start = time.perf_counter()
        await self._obs.emit(
            "memory.add.start", "memory add started",
            context={"memory_id": memory_id, "turn_count": len(interaction.turns)},
        )

        try:
            interaction = self._with_default_occurred_at(interaction)
            existing = await self._fetch_dedup_context(interaction, memory_id)

            t0 = time.perf_counter()
            extracted = await ing.extractor.extract(interaction, existing_memories=existing)
            tel_row.extraction_duration_ms = int((time.perf_counter() - t0) * 1000)

            if not extracted:
                tel_row.outcome = "empty"
                await self._obs.emit("memory.add.empty", "extractor produced no memories",
                                     context={"memory_id": memory_id})
                return ()

            extracted = await self._drop_hash_dupes(extracted, memory_id, tel_row)
            if not extracted:
                tel_row.outcome = "empty"
                await self._obs.emit("memory.add.empty", "extractor produced no memories",
                                     context={"memory_id": memory_id})
                return ()

            valid_existing_ids = {m.memory_row_id for m in existing}
            mem_rows = self._build_rows(extracted, memory_id, interaction.metadata, valid_existing_ids, tel_row)
            await self._dispatch_pillars(mem_rows, tel_row)
            tel_row.row_count = len(mem_rows)
            await self._obs.emit(
                "memory.add.success", "memory add succeeded",
                context={"memory_id": memory_id, "row_count": tel_row.row_count,
                         "dropped_dedup_count": tel_row.dropped_dedup_count},
            )
            return tuple(mem_rows)
        except BaseException as exc:
            tel_row.outcome = "error"
            tel_row.error_type = type(exc).__name__
            tel_row.error_message = str(exc)
            await self._obs.emit("memory.add.error", "memory add failed", level="error",
                                 context={"memory_id": memory_id}, error=exc)
            raise
        finally:
            tel_row.total_duration_ms = int((time.perf_counter() - start) * 1000)
            try:
                await self._tel.write(tel_row)
            except Exception:
                logger.exception("telemetry write failed for memory add memory_id=%s", memory_id)
            _reset_obs(obs_token)

    async def search(
        self,
        query: str,
        memory_id: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[MemorySearchResult, ...]:
        self._check_initialized()
        if not query or not query.strip():
            raise ValueError("query must not be blank")
        if not memory_id or not memory_id.strip():
            raise ValueError("memory_id must not be blank")
        if filters:
            raise NotImplementedError("custom filters not yet supported in MemoryEngine.search")

        cfg = self._cfg
        ret_cfg = cfg.retrieval
        tel_row = MemorySearchTelemetryRow(memory_id=memory_id, outcome="success")
        obs_token = _set_obs(self._obs)
        start = time.perf_counter()
        await self._obs.emit("memory.search.start", "memory search started",
                             context={"memory_id": memory_id})
        try:
            methods: list[Any] = []
            if ret_cfg.semantic_weight > 0:
                methods.append(SemanticRetrieval(
                    store=cfg.vector_store,
                    embeddings=cfg.ingestion.embeddings,
                    sparse_embeddings=cfg.ingestion.sparse_embeddings,
                    weight=ret_cfg.semantic_weight,
                ))
            if ret_cfg.keyword_weight > 0:
                kw_kwargs: dict[str, Any] = {
                    "backend": cfg.ingestion.keyword_backend,
                    "weight": ret_cfg.keyword_weight,
                }
                if cfg.ingestion.keyword_backend == "bm25":
                    kw_kwargs["vector_store"] = cfg.vector_store
                    kw_kwargs["bm25_max_chunks"] = cfg.ingestion.bm25_max_chunks
                else:
                    kw_kwargs["document_store"] = cfg.document_store
                methods.append(KeywordRetrieval(**kw_kwargs))
            if ret_cfg.entity_weight > 0 and cfg.graph_store is not None:
                # EntityRetrieval hardcodes max_hops at search time today;
                # MemoryRetrievalConfig.entity_hops is plumbed but inert until
                # EntityRetrieval gains a max_hops constructor arg.
                methods.append(EntityRetrieval(
                    store=cfg.graph_store,
                    weight=ret_cfg.entity_weight,
                ))
            service = RetrievalService(
                retrieval_methods=methods,
                reranking=ret_cfg.rerank,
                top_k=top_k,
            )
            # knowledge_id=memory_id reuses the existing service's filter contract
            # by aliasing: memory rows are stored with knowledge_id == memory_id.
            chunks, trace = await service.retrieve(
                query=query, knowledge_id=memory_id, top_k=top_k, trace=True,
            )
            per_method = trace.per_method_results if trace else {}
            results: list[MemorySearchResult] = []
            for chunk in chunks:
                scores = {name: 0.0 for name in per_method}
                for name, items in per_method.items():
                    for item in items:
                        if item.chunk_id == chunk.chunk_id:
                            scores[name] = item.score
                            break
                row = self._payload_to_row({
                    "memory_row_id": chunk.chunk_id,
                    "memory_id": memory_id,
                    "text": chunk.content,
                    "content": chunk.content,
                    **(chunk.source_metadata or {}),
                })
                results.append(MemorySearchResult(row=row, score=chunk.score, pillar_scores=scores))
            tel_row.result_count = len(results)
            tel_row.top_score = results[0].score if results else None
            tel_row.methods_used = list(per_method.keys()) if per_method else []
            await self._obs.emit("memory.search.success", "memory search ok",
                                 context={"memory_id": memory_id, "result_count": len(results)})
            return tuple(results)
        except BaseException as exc:
            tel_row.outcome = "error"
            tel_row.error_type = type(exc).__name__
            tel_row.error_message = str(exc)
            await self._obs.emit("memory.search.error", "memory search failed", level="error",
                                 context={"memory_id": memory_id}, error=exc)
            raise
        finally:
            tel_row.duration_ms = int((time.perf_counter() - start) * 1000)
            try:
                await self._tel.write(tel_row)
            except Exception:
                logger.exception("telemetry write failed for memory search memory_id=%s", memory_id)
            _reset_obs(obs_token)

    @staticmethod
    def _with_default_occurred_at(interaction: Interaction) -> Interaction:
        if interaction.occurred_at is not None:
            return interaction
        return replace(interaction, occurred_at=datetime.now(UTC))

    async def _fetch_dedup_context(
        self, interaction: Interaction, memory_id: str,
    ) -> tuple[MemoryRow, ...]:
        ing = self._cfg.ingestion
        if ing.dedup_context_top_k <= 0:
            return ()
        recent = interaction.turns[-ing.dedup_context_recent_turns:]
        probe = "\n".join(t.content for t in recent)
        vectors = await embed_batched(ing.embeddings, [probe])
        if not vectors:
            return ()
        results = await self._cfg.vector_store.search(
            vector=vectors[0], top_k=ing.dedup_context_top_k,
            filters={"memory_id": memory_id},
        )
        return tuple(self._payload_to_row(r.payload) for r in results)

    async def _drop_hash_dupes(
        self, extracted: tuple[ExtractedMemory, ...], memory_id: str, tel_row: MemoryAddTelemetryRow,
    ) -> tuple[ExtractedMemory, ...]:
        existing_hashes = await self._existing_hashes(memory_id)
        kept: list[ExtractedMemory] = []
        for m in extracted:
            h = _hash(m.text)
            if h in existing_hashes:
                tel_row.dropped_dedup_count += 1
                await self._obs.emit(
                    "memory.add.dedup_hit", "hash dedup match",
                    context={"memory_id": memory_id, "text_hash": h},
                )
                continue
            kept.append(m)
        return tuple(kept)

    async def _existing_hashes(self, memory_id: str) -> set[str]:
        # Full-namespace scan per add(). Acceptable for v1: typical memory_id
        # stays small per consumer. If this regresses, replace with a
        # vector-store payload-indexed text_hash lookup.
        store = self._cfg.vector_store
        offset: str | None = None
        out: set[str] = set()
        while True:
            results, next_offset = await store.scroll(
                filters={"memory_id": memory_id}, limit=500, offset=offset,
            )
            for r in results:
                h = r.payload.get("text_hash")
                if h:
                    out.add(h)
            if not next_offset or not results:
                break
            offset = next_offset
        return out

    def _build_rows(
        self,
        extracted: tuple[ExtractedMemory, ...],
        memory_id: str,
        metadata: Mapping[str, Any],
        valid_existing_ids: set[str],
        tel_row: MemoryAddTelemetryRow,
    ) -> list[MemoryRow]:
        now = datetime.now(UTC)
        out: list[MemoryRow] = []
        for m in extracted:
            valid_links: list[str] = []
            for link in m.linked_memory_row_ids:
                if link in valid_existing_ids:
                    valid_links.append(link)
                else:
                    tel_row.dropped_invalid_link_count += 1
            out.append(
                MemoryRow(
                    memory_row_id=str(uuid.uuid4()),
                    memory_id=memory_id,
                    text=m.text,
                    text_hash=_hash(m.text),
                    attributed_to=m.attributed_to,
                    linked_memory_row_ids=tuple(valid_links),
                    created_at=now,
                    updated_at=now,
                    interaction_metadata=dict(metadata),
                )
            )
        return out

    async def _dispatch_pillars(self, rows: list[MemoryRow], tel_row: MemoryAddTelemetryRow) -> None:
        ing = self._cfg.ingestion

        async def _semantic() -> None:
            t0 = time.perf_counter()
            try:
                texts = [r.text for r in rows]
                vectors = await embed_batched(ing.embeddings, texts)
                points = [
                    VectorPoint(
                        point_id=row.memory_row_id,
                        vector=vec,
                        payload=self._row_to_payload(row),
                    )
                    for row, vec in zip(rows, vectors, strict=True)
                ]
                await self._cfg.vector_store.upsert(points)
            finally:
                tel_row.semantic_duration_ms = int((time.perf_counter() - t0) * 1000)

        async def _entity() -> None:
            if ing.entity_extraction is None or self._cfg.graph_store is None:
                return
            t0 = time.perf_counter()
            try:
                from rfnry_knowledge.baml.baml_client.async_client import b as baml_b
                registry = build_registry(self._cfg.provider)
                for r in rows:
                    def _make_entity_call(text: str):  # noqa: ANN202
                        async def _call(collector: Any) -> Any:
                            return await baml_b.ExtractEntitiesFromText(
                                text, baml_options={"client_registry": registry, "collector": collector},
                            )
                        return _call
                    try:
                        result = await instrument_baml_call(
                            operation="memory_extract_entities",
                            call=_make_entity_call(r.text),
                        )
                    except Exception as exc:
                        if ing.entity_required:
                            raise
                        logger.warning("memory entity extraction failed: %s", exc)
                        continue
                    if not result.entities:
                        continue
                    graph_entities = [
                        GraphEntity(
                            name=e.name,
                            entity_type=e.category or "entity",
                            category=e.category or "",
                            value=e.value,
                            properties={"memory_row_ids": [r.memory_row_id]},
                        )
                        for e in result.entities
                    ]
                    await self._cfg.graph_store.add_entities(
                        source_id=r.memory_row_id,
                        knowledge_id=r.memory_id,
                        entities=graph_entities,
                    )
            finally:
                tel_row.entity_duration_ms = int((time.perf_counter() - t0) * 1000)

        async def _keyword() -> None:
            if ing.keyword_backend != "postgres_fts" or self._cfg.document_store is None:
                return
            t0 = time.perf_counter()
            try:
                for r in rows:
                    await self._cfg.document_store.store_content(
                        source_id=r.memory_row_id,
                        knowledge_id=r.memory_id,
                        source_type=None,
                        title="",
                        content=r.text,
                    )
            finally:
                tel_row.keyword_duration_ms = int((time.perf_counter() - t0) * 1000)

        coros = [(_semantic, ing.semantic_required, "semantic"),
                 (_keyword, ing.keyword_required, "keyword"),
                 (_entity, ing.entity_required, "entity")]
        results = await asyncio.gather(*[c() for c, _, _ in coros], return_exceptions=True)
        for (_, required, name), res in zip(coros, results, strict=True):
            if isinstance(res, BaseException):
                if required:
                    raise res
                logger.warning("memory %s pillar failed (optional): %s", name, res)
                if tel_row.outcome == "success":
                    tel_row.outcome = "partial"
                    tel_row.error_type = type(res).__name__
                    tel_row.error_message = f"{name}: {res}"

    def _row_to_payload(self, r: MemoryRow) -> dict[str, Any]:
        # knowledge_id aliased to memory_id so existing RetrievalService scope filters work
        payload: dict[str, Any] = {
            "memory_row_id": r.memory_row_id,
            "memory_id": r.memory_id,
            "knowledge_id": r.memory_id,
            "text": r.text,
            "content": r.text,
            "text_hash": r.text_hash,
            "attributed_to": r.attributed_to,
            "linked_memory_row_ids": list(r.linked_memory_row_ids),
            "created_at": r.created_at.isoformat(),
        }
        for k, v in r.interaction_metadata.items():
            payload.setdefault(k, v)
        return payload

    @staticmethod
    def _payload_to_row(payload: Mapping[str, Any]) -> MemoryRow:
        created = payload.get("created_at")
        ts = datetime.fromisoformat(created) if isinstance(created, str) else datetime.now(UTC)
        return MemoryRow(
            memory_row_id=str(payload.get("memory_row_id", "")),
            memory_id=str(payload.get("memory_id", "")),
            text=str(payload.get("text") or payload.get("content") or ""),
            text_hash=str(payload.get("text_hash", "")),
            attributed_to=payload.get("attributed_to"),
            linked_memory_row_ids=tuple(payload.get("linked_memory_row_ids") or ()),
            created_at=ts,
            updated_at=ts,
            interaction_metadata={
                k: v for k, v in payload.items()
                if k not in {
                    "memory_row_id", "memory_id", "knowledge_id", "text", "content",
                    "text_hash", "attributed_to", "linked_memory_row_ids", "created_at",
                }
            },
        )
