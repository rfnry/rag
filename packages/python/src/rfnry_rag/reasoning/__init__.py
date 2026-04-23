"""Reasoning services — analysis, classification, clustering, compliance, evaluation, pipelines."""

from rfnry_rag.reasoning.common.startup import check_baml as _check_baml

_check_baml()

from rfnry_rag.reasoning.common.errors import AnalysisError as AnalysisError
from rfnry_rag.reasoning.common.errors import ClassificationError as ClassificationError
from rfnry_rag.reasoning.common.errors import ClusteringError as ClusteringError
from rfnry_rag.reasoning.common.errors import ComplianceError as ComplianceError
from rfnry_rag.reasoning.common.errors import ConfigurationError as ConfigurationError
from rfnry_rag.reasoning.common.errors import EvaluationError as EvaluationError
from rfnry_rag.reasoning.common.errors import ReasoningError as ReasoningError
from rfnry_rag.reasoning.common.language_model import LanguageModelClient as LanguageModelClient
from rfnry_rag.reasoning.common.language_model import LanguageModelProvider as LanguageModelProvider
from rfnry_rag.reasoning.modules.analysis.models import AnalysisConfig as AnalysisConfig
from rfnry_rag.reasoning.modules.analysis.models import AnalysisResult as AnalysisResult
from rfnry_rag.reasoning.modules.analysis.models import ContextTrackingConfig as ContextTrackingConfig
from rfnry_rag.reasoning.modules.analysis.models import DimensionDefinition as DimensionDefinition
from rfnry_rag.reasoning.modules.analysis.models import DimensionResult as DimensionResult
from rfnry_rag.reasoning.modules.analysis.models import Entity as Entity
from rfnry_rag.reasoning.modules.analysis.models import EntityTypeDefinition as EntityTypeDefinition
from rfnry_rag.reasoning.modules.analysis.models import IntentShift as IntentShift
from rfnry_rag.reasoning.modules.analysis.models import Message as Message
from rfnry_rag.reasoning.modules.analysis.models import RetrievalHint as RetrievalHint
from rfnry_rag.reasoning.modules.analysis.service import AnalysisService as AnalysisService
from rfnry_rag.reasoning.modules.classification.models import CategoryDefinition as CategoryDefinition
from rfnry_rag.reasoning.modules.classification.models import Classification as Classification
from rfnry_rag.reasoning.modules.classification.models import ClassificationConfig as ClassificationConfig
from rfnry_rag.reasoning.modules.classification.models import ClassificationSetDefinition as ClassificationSetDefinition
from rfnry_rag.reasoning.modules.classification.models import ClassificationSetResult as ClassificationSetResult
from rfnry_rag.reasoning.modules.classification.service import ClassificationService as ClassificationService
from rfnry_rag.reasoning.modules.clustering.comparison import ClusterChange as ClusterChange
from rfnry_rag.reasoning.modules.clustering.comparison import ClusterComparison as ClusterComparison
from rfnry_rag.reasoning.modules.clustering.comparison import compare_clusters as compare_clusters
from rfnry_rag.reasoning.modules.clustering.models import Cluster as Cluster
from rfnry_rag.reasoning.modules.clustering.models import ClusteringConfig as ClusteringConfig
from rfnry_rag.reasoning.modules.clustering.models import ClusteringResult as ClusteringResult
from rfnry_rag.reasoning.modules.clustering.models import TextWithMetadata as TextWithMetadata
from rfnry_rag.reasoning.modules.clustering.service import ClusteringService as ClusteringService
from rfnry_rag.reasoning.modules.compliance.models import ComplianceConfig as ComplianceConfig
from rfnry_rag.reasoning.modules.compliance.models import ComplianceDimensionDefinition as ComplianceDimensionDefinition
from rfnry_rag.reasoning.modules.compliance.models import ComplianceResult as ComplianceResult
from rfnry_rag.reasoning.modules.compliance.models import Violation as Violation
from rfnry_rag.reasoning.modules.compliance.service import ComplianceService as ComplianceService
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationConfig as EvaluationConfig
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationDimensionDefinition as EvaluationDimensionDefinition
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationPair as EvaluationPair
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationReport as EvaluationReport
from rfnry_rag.reasoning.modules.evaluation.models import EvaluationResult as EvaluationResult
from rfnry_rag.reasoning.modules.evaluation.service import EvaluationService as EvaluationService
from rfnry_rag.reasoning.modules.pipeline.models import AnalyzeStep as AnalyzeStep
from rfnry_rag.reasoning.modules.pipeline.models import ClassifyStep as ClassifyStep
from rfnry_rag.reasoning.modules.pipeline.models import ComplianceStep as ComplianceStep
from rfnry_rag.reasoning.modules.pipeline.models import EvaluateStep as EvaluateStep
from rfnry_rag.reasoning.modules.pipeline.models import PipelineResult as PipelineResult
from rfnry_rag.reasoning.modules.pipeline.models import PipelineServices as PipelineServices
from rfnry_rag.reasoning.modules.pipeline.service import Pipeline as Pipeline
from rfnry_rag.reasoning.protocols import BaseEmbeddings as BaseEmbeddings
from rfnry_rag.reasoning.protocols import BaseSemanticIndex as BaseSemanticIndex

__all__ = [
    "BaseEmbeddings",
    "BaseSemanticIndex",
    "AnalysisError",
    "ClassificationError",
    "ClusteringError",
    "ComplianceError",
    "ConfigurationError",
    "EvaluationError",
    "ReasoningError",
    "LanguageModelClient",
    "LanguageModelProvider",
    "AnalysisConfig",
    "AnalysisResult",
    "AnalysisService",
    "DimensionDefinition",
    "DimensionResult",
    "Entity",
    "EntityTypeDefinition",
    "IntentShift",
    "Message",
    "RetrievalHint",
    "ContextTrackingConfig",
    "CategoryDefinition",
    "Classification",
    "ClassificationConfig",
    "ClassificationService",
    "ClassificationSetDefinition",
    "ClassificationSetResult",
    "Cluster",
    "ClusterChange",
    "ClusterComparison",
    "ClusteringConfig",
    "ClusteringResult",
    "ClusteringService",
    "TextWithMetadata",
    "compare_clusters",
    "EvaluationConfig",
    "EvaluationDimensionDefinition",
    "EvaluationPair",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationService",
    "ComplianceConfig",
    "ComplianceDimensionDefinition",
    "ComplianceResult",
    "ComplianceService",
    "Violation",
    "AnalyzeStep",
    "ClassifyStep",
    "ComplianceStep",
    "EvaluateStep",
    "Pipeline",
    "PipelineResult",
    "PipelineServices",
]
