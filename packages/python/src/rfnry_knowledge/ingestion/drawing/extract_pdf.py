"""Extract DrawingPageAnalysis from PDF pages via BAML AnalyzeDrawingPage."""

from __future__ import annotations

import asyncio
from typing import Any

from baml_py import ClientRegistry, Image
from baml_py import errors as baml_errors

from rfnry_knowledge.baml.baml_client.async_client import b
from rfnry_knowledge.common.logging import get_logger
from rfnry_knowledge.config.drawing import DrawingIngestionConfig
from rfnry_knowledge.exceptions import IngestionError
from rfnry_knowledge.ingestion.drawing.models import (
    DetectedComponent,
    DetectedConnection,
    DrawingPageAnalysis,
    OffPageConnector,
    Port,
)
from rfnry_knowledge.telemetry.usage import instrument_baml_call

logger = get_logger("drawing/ingestion/extract_pdf")


def build_symbol_library_prompt(config: DrawingIngestionConfig) -> str:
    """Serialise the resolved symbol library as a newline-delimited per-domain string.

    Format:
        DOMAIN: sym_a, sym_b, sym_c
        DOMAIN2: sym_d, sym_e
    """
    lib = config.symbol_library or {}
    lines = [f"{domain}: {', '.join(symbols)}" for domain, symbols in sorted(lib.items())]
    return "\n".join(lines)


def build_off_page_patterns_prompt(config: DrawingIngestionConfig) -> str:
    """Serialise off-page-connector regex patterns as a newline-delimited list."""
    return "\n".join(config.off_page_connector_patterns or [])


def _resolve_domain_hint(config: DrawingIngestionConfig, source_metadata: dict[str, Any]) -> str:
    """Per-call override via metadata['domain_hint'], else config default_domain."""
    return source_metadata.get("domain_hint") or config.default_domain


async def analyze_drawing_page(
    page_image_b64: str,
    page_number: int,
    config: DrawingIngestionConfig,
    registry: ClientRegistry,
    domain_hint: str,
    symbol_library_prompt: str,
    off_page_patterns_prompt: str,
) -> DrawingPageAnalysis:
    """Call BAML AnalyzeDrawingPage and convert the result into our Python dataclass."""
    _ = config  # reserved for future per-call knobs (e.g. temperature overrides)
    baml_image = Image.from_base64("image/png", page_image_b64)
    try:
        result = await instrument_baml_call(
            operation="analyze_drawing_page",
            call=lambda collector: b.AnalyzeDrawingPage(
                baml_image,
                domain_hint,
                symbol_library_prompt,
                off_page_patterns_prompt,
                baml_options={"client_registry": registry, "collector": collector},
            ),
        )
    except baml_errors.BamlValidationError as exc:
        raise IngestionError(
            f"AnalyzeDrawingPage failed on page {page_number}: LLM returned an unparseable response. Detail: {exc}"
        ) from exc
    except Exception as exc:
        raise IngestionError(f"AnalyzeDrawingPage failed on page {page_number}: {exc}") from exc
    return _baml_to_dataclass(result, page_number)


def _baml_to_dataclass(result: Any, fallback_page_number: int) -> DrawingPageAnalysis:
    """Convert the BAML DrawingPageAnalysis result to our Python dataclass."""
    components = [
        DetectedComponent(
            component_id=c.component_id,
            symbol_class=c.symbol_class,
            label=getattr(c, "label", None),
            bbox=list(c.bbox),
            ports=[
                Port(
                    port_id=p.port_id,
                    position=list(p.position) if getattr(p, "position", None) else None,
                )
                for p in getattr(c, "ports", []) or []
            ],
            properties=dict(c.properties) if getattr(c, "properties", None) else None,
        )
        for c in getattr(result, "components", []) or []
    ]
    connections = [
        DetectedConnection(
            from_component=conn.from_component,
            to_component=conn.to_component,
            from_port=getattr(conn, "from_port", None),
            to_port=getattr(conn, "to_port", None),
            net_label=getattr(conn, "net_label", None),
            wire_style=getattr(conn, "wire_style", None),
        )
        for conn in getattr(result, "connections", []) or []
    ]
    off_page = [
        OffPageConnector(
            tag=opc.tag,
            bound_component=opc.bound_component,
            target_hint=getattr(opc, "target_hint", None),
        )
        for opc in getattr(result, "off_page_connectors", []) or []
    ]
    return DrawingPageAnalysis(
        page_number=getattr(result, "page_number", fallback_page_number) or fallback_page_number,
        sheet_number=getattr(result, "sheet_number", None),
        zone_grid=getattr(result, "zone_grid", None),
        domain=getattr(result, "domain", "mixed") or "mixed",
        components=components,
        connections=connections,
        off_page_connectors=off_page,
        title_block=dict(result.title_block) if getattr(result, "title_block", None) else None,
        notes=list(getattr(result, "notes", []) or []),
        page_type=getattr(result, "page_type", "drawing") or "drawing",
    )


async def extract_pdf_analyses(
    page_rows: list[dict],
    config: DrawingIngestionConfig,
    registry: ClientRegistry,
    source_metadata: dict[str, Any],
) -> list[DrawingPageAnalysis]:
    """Run AnalyzeDrawingPage per page under a semaphore capped at analyze_concurrency."""
    sem = asyncio.Semaphore(config.analyze_concurrency)
    symbol_library_prompt = build_symbol_library_prompt(config)
    off_page_patterns_prompt = build_off_page_patterns_prompt(config)
    domain_hint = _resolve_domain_hint(config, source_metadata)

    async def _worker(row: dict) -> DrawingPageAnalysis:
        async with sem:
            return await analyze_drawing_page(
                row["data"]["page_image_b64"],
                row["page_number"],
                config,
                registry,
                domain_hint,
                symbol_library_prompt,
                off_page_patterns_prompt,
            )

    tasks = [asyncio.create_task(_worker(r)) for r in page_rows]
    return await asyncio.gather(*tasks)
