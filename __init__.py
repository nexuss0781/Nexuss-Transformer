"""
Nexuss Transformer Framework (NTF) - Blank Slate Transformer Training System

A comprehensive framework for training decoder-only transformer models from scratch,
with full support for fine-tuning, RLHF, and continual learning.
Optimized for Ethiopian languages with NTFTokenizer (EthioBBPE-based) integration.
"""

from models import (
    NTFConfig,
    NexussTransformer,
)
from training import (
    NTFTrainer,
    TrainingArguments,
    CheckpointManager,
)
from finetuning import (
    setup_lora,
    full_finetune,
    freeze_layers,
)
from reward import (
    RewardModel,
    PPOTrainer,
    DPOTrainer,
)
from utils import (
    ContinualLearner,
    ModelVersioner,
    EvaluationMetrics,
)
from tokenizer import (
    NTFTokenizer,
    TokenizerOutput,
    get_ntf_tokenizer,
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "NTFConfig",
    "NexussTransformer",
    # Training
    "NTFTrainer",
    "TrainingArguments",
    "CheckpointManager",
    # Fine-tuning
    "setup_lora",
    "full_finetune",
    "freeze_layers",
    # Reward & RLHF
    "RewardModel",
    "PPOTrainer",
    "DPOTrainer",
    # Utils
    "ContinualLearner",
    "ModelVersioner",
    "EvaluationMetrics",
    # Tokenizer
    "NTFTokenizer",
    "TokenizerOutput",
    "get_ntf_tokenizer",
]
