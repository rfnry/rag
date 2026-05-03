from rfnry_knowledge.exceptions.base import KnowledgeEngineError


class GenerationError(KnowledgeEngineError):
    """Error during LLM generation."""
