from rfnry_rag.exceptions.base import RagError


class GenerationError(RagError):
    """Error during LLM generation."""
