from rfnry_rag.common.errors import BaseException
from rfnry_rag.common.errors import ConfigurationError as ConfigurationError


class ReasoningError(BaseException):
    """Base exception for reasoning SDK errors."""


class ClassificationError(ReasoningError):
    """Error during text classification."""


class ClusteringError(ReasoningError):
    """Error during text clustering."""


class EvaluationError(ReasoningError):
    """Error during evaluation."""


class ComplianceError(ReasoningError):
    """Error during compliance checking."""


class AnalysisError(ReasoningError):
    """Error during text analysis."""
