"""
Utilities package for Nexuss Transformer Framework
"""

from .continual_learning import (
    EWCConfig,
    ReplayConfig,
    GEMConfig,
    ContinualLearningConfig,
    EWCRegularizer,
    ReplayBuffer,
    GEMOptimizer,
    LwFLoss,
    create_continual_learning_wrapper,
)

from .versioning import (
    ModelStage,
    ModelVersion,
    ModelMetadata,
    ModelRegistry,
    create_model_metadata,
)

from .metrics import (
    EvaluationResults,
    compute_perplexity,
    compute_accuracy,
    evaluate_model,
    benchmark_throughput,
    compare_models,
)

__all__ = [
    # Continual Learning
    "EWCConfig",
    "ReplayConfig",
    "GEMConfig",
    "ContinualLearningConfig",
    "EWCRegularizer",
    "ReplayBuffer",
    "GEMOptimizer",
    "LwFLoss",
    "create_continual_learning_wrapper",
    
    # Versioning
    "ModelStage",
    "ModelVersion",
    "ModelMetadata",
    "ModelRegistry",
    "create_model_metadata",
    
    # Metrics
    "EvaluationResults",
    "compute_perplexity",
    "compute_accuracy",
    "evaluate_model",
    "benchmark_throughput",
    "compare_models",
]
