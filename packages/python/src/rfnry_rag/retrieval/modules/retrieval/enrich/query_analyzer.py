"""Analyze natural language queries to extract structured search terms via BAML."""

from typing import Any

from baml_py import ClientRegistry
from baml_py import errors as baml_errors

from rfnry_rag.common.logging import get_logger
from rfnry_rag.retrieval.baml.baml_client.async_client import b

logger = get_logger("enrich/retrieval/query")


async def analyze_query(query: str, registry: ClientRegistry) -> dict[str, Any]:
    """Extract entity references, keywords, and intent from a query."""
    try:
        result = await b.AnalyzeQuery(
            query,
            baml_options={"client_registry": registry},
        )
        return {
            "entity_references": list(result.entity_references) if result.entity_references else [],
            "keywords": list(result.keywords) if result.keywords else [],
            "intent": result.intent or "",
        }
    except baml_errors.BamlValidationError as exc:
        logger.exception(
            "AnalyzeQuery failed: LLM returned an unparseable response — using empty defaults. Detail: %s",
            exc,
        )
        return {"entity_references": [], "keywords": [], "intent": ""}
    except Exception as exc:
        logger.exception(
            "AnalyzeQuery failed: %s — using empty defaults",
            exc,
        )
        return {"entity_references": [], "keywords": [], "intent": ""}
