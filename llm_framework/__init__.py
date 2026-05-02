"""
LLM Framework - Blank Slate Transformer Training System

A comprehensive framework for training decoder-only transformer models from scratch,
with full support for fine-tuning, RLHF, and continual learning.
"""

from llm_framework.models import (
    TransformerConfig,
    DecoderOnlyTransformer,
)
from llm_framework.training import (
    Trainer,
    TrainingConfig,
    CheckpointManager,
)
from llm_framework.finetuning import (
    LoRAConfig,
    PEFTTrainer,
    FullFinetuneTrainer,
    LayerFreezer,
)
from llm_framework.reward import (
    RewardModel,
    PPOTrainer,
    DPOTrainer,
)
from llm_framework.utils import (
    ContinualLearner,
    ModelVersioner,
    EvaluationMetrics,
)

__version__ = "0.1.0"
__all__ = [
    # Models
    "TransformerConfig",
    "DecoderOnlyTransformer",
    # Training
    "Trainer",
    "TrainingConfig",
    "CheckpointManager",
    # Fine-tuning
    "LoRAConfig",
    "PEFTTrainer",
    "FullFinetuneTrainer",
    "LayerFreezer",
    # Reward & RLHF
    "RewardModel",
    "PPOTrainer",
    "DPOTrainer",
    # Utils
    "ContinualLearner",
    "ModelVersioner",
    "EvaluationMetrics",
]
