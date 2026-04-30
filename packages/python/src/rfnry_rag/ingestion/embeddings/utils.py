"""Compatibility re-exports — the real helper lives in rfnry_rag.common.embeddings."""

from rfnry_rag.common.embeddings import EMBED_BATCH_SIZE, embed_batched

__all__ = ["EMBED_BATCH_SIZE", "embed_batched"]
