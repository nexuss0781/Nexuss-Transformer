#!/usr/bin/env python3
"""
Nexuss Transformer Framework - Complete Component Registry

This module registers and provides unified access to all 22+ framework components
with automatic CLI integration, configuration management, and documentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Type
from enum import Enum


class ComponentCategory(Enum):
    """Categories for framework components."""
    MODEL = "model"
    TRAINING = "training"
    FINETUNING = "finetuning"
    REWARD = "reward"
    UTILS = "utils"
    DATA = "data"
    CHECKPOINT = "checkpoint"
    EVALUATION = "evaluation"
    CONVERSION = "conversion"


@dataclass
class ComponentMetadata:
    """Metadata for a framework component."""
    name: str
    category: ComponentCategory
    description: str
    module_path: str
    class_name: str
    config_class: Optional[str] = None
    default_config: Optional[Dict[str, Any]] = None
    required_args: List[str] = field(default_factory=list)
    optional_args: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    cli_flags: Dict[str, str] = field(default_factory=dict)


# Complete registry of all NTF components (22+)
NTF_COMPONENT_REGISTRY: Dict[str, ComponentMetadata] = {
    # ========== MODEL COMPONENTS (3) ==========
    "nexuss_transformer": ComponentMetadata(
        name="nexuss_transformer",
        category=ComponentCategory.MODEL,
        description="Main decoder-only transformer model with RoPE, SwiGLU, and RMSNorm",
        module_path="models.transformer",
        class_name="NexussTransformer",
        config_class="NTFConfig",
        required_args=["vocab_size", "d_model", "n_heads", "n_layers"],
        optional_args=["d_ff", "max_seq_len", "activation", "use_rope", "dropout"],
        examples=[
            "ntf train --config pretrain_small",
            "ntf train --config pretrain_small --override model.hidden_size=1024 model.num_hidden_layers=16",
        ],
        cli_flags={
            "--model-type": "Model architecture type",
            "--vocab-size": "Vocabulary size",
            "--hidden-size": "Hidden dimension size",
            "--num-layers": "Number of transformer layers",
            "--num-heads": "Number of attention heads",
        }
    ),
    
    "ntf_config": ComponentMetadata(
        name="ntf_config",
        category=ComponentCategory.MODEL,
        description="Configuration class for NexussTransformer architecture",
        module_path="models.config",
        class_name="NTFConfig",
        optional_args=["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "activation"],
        examples=[
            "Use via --override flags in CLI",
        ]
    ),
    
    "ethio_tokenizer": ComponentMetadata(
        name="ethio_tokenizer",
        category=ComponentCategory.MODEL,
        description="EthioBBPE tokenizer optimized for Ethiopian languages",
        module_path="models.tokenizer",
        class_name="EthioBBPE",
        required_args=["vocab_file"],
        optional_args=["merges_file", "normalization"],
        examples=[
            "Automatically configured via tokenizer_name in data config",
        ]
    ),
    
    # ========== TRAINING COMPONENTS (4) ==========
    "trainer": ComponentMetadata(
        name="trainer",
        category=ComponentCategory.TRAINING,
        description="Main training loop with Accelerate, mixed precision, gradient accumulation",
        module_path="training.trainer",
        class_name="Trainer",
        config_class="TrainingConfig",
        required_args=["model", "config"],
        optional_args=["train_dataset", "eval_dataset", "data_collator"],
        examples=[
            "ntf train --config pretrain_small",
            "ntf train --config pretrain_small --resume-from-checkpoint ./outputs/checkpoint-5000",
        ],
        cli_flags={
            "--resume-from-checkpoint": "Resume from checkpoint path",
            "--max-steps": "Override maximum training steps",
            "--learning-rate": "Override learning rate",
            "--batch-size": "Override batch size",
        }
    ),
    
    "training_config": ComponentMetadata(
        name="training_config",
        category=ComponentCategory.TRAINING,
        description="Comprehensive training hyperparameters configuration",
        module_path="training.config",
        class_name="TrainingConfig",
        optional_args=[
            "output_dir", "num_train_epochs", "max_steps", "learning_rate",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "warmup_ratio", "scheduler", "mixed_precision", "weight_decay",
        ],
        examples=[
            "--override training.learning_rate=1e-4 training.max_steps=50000",
        ]
    ),
    
    "data_collator": ComponentMetadata(
        name="data_collator",
        category=ComponentCategory.TRAINING,
        description="Dynamic padding and batch collation for language modeling",
        module_path="training.data",
        class_name="DataCollatorForLanguageModeling",
        required_args=["pad_token_id"],
        optional_args=["max_length", "return_tensors"],
        examples=[
            "Automatically configured from data config",
        ]
    ),
    
    "checkpoint_manager": ComponentMetadata(
        name="checkpoint_manager",
        category=ComponentCategory.CHECKPOINT,
        description="Checkpoint save/load with versioning and best model tracking",
        module_path="training.checkpoint",
        class_name="CheckpointManager",
        required_args=["output_dir"],
        optional_args=["save_total_limit"],
        examples=[
            "--override checkpoint.save_steps=1000 checkpoint.save_total_limit=5",
        ],
        cli_flags={
            "--save-steps": "Save checkpoint every N steps",
            "--save-total-limit": "Maximum checkpoints to keep",
        }
    ),
    
    # ========== FINETUNING COMPONENTS (4) ==========
    "peft_trainer": ComponentMetadata(
        name="peft_trainer",
        category=ComponentCategory.FINETUNING,
        description="Parameter-efficient fine-tuning with LoRA, AdaLoRA, and adapters",
        module_path="finetuning.peft_finetune",
        class_name="PEFTTrainer",
        config_class="LoRAConfig",
        required_args=["model", "config"],
        optional_args=["tokenizer"],
        examples=[
            "ntf finetune --config finetune_lora.yaml",
            "ntf finetune --config finetune_lora.yaml --override lora.r=32 lora.alpha=64",
        ],
        cli_flags={
            "--lora-r": "LoRA rank",
            "--lora-alpha": "LoRA alpha scaling factor",
            "--lora-dropout": "LoRA dropout",
            "--target-modules": "Modules to apply LoRA",
        }
    ),
    
    "lora_config": ComponentMetadata(
        name="lora_config",
        category=ComponentCategory.FINETUNING,
        description="LoRA adapter configuration with advanced options",
        module_path="finetuning.peft_finetune",
        class_name="LoRAConfig",
        optional_args=[
            "r", "alpha", "dropout", "target_modules", "bias",
            "modules_to_save", "layers_to_transform",
        ],
        examples=[
            "--override lora.r=16 lora.alpha=32 lora.target_modules=['q_proj','v_proj']",
        ]
    ),
    
    "full_finetuner": ComponentMetadata(
        name="full_finetuner",
        category=ComponentCategory.FINETUNING,
        description="Full parameter fine-tuning with discriminative LR and gradual unfreezing",
        module_path="finetuning.full_finetune",
        class_name="FullFinetuneTrainer",
        optional_args=["discriminative_lr", "layerwise_lr_decay"],
        examples=[
            "ntf finetune --config finetune_full.yaml --no-lora",
        ],
        cli_flags={
            "--no-lora": "Disable LoRA for full fine-tuning",
            "--discriminative-lr": "Different LRs per layer",
            "--layerwise-lr-decay": "LR decay for deeper layers",
        }
    ),
    
    "layer_freezer": ComponentMetadata(
        name="layer_freezer",
        category=ComponentCategory.FINETUNING,
        description="Flexible layer freezing strategies (top-k, bottom-k, alternating, custom)",
        module_path="finetuning.freeze",
        class_name="LayerFreezer",
        required_args=["model"],
        optional_args=[],
        examples=[
            "--freeze-top-k 4",
            "--freeze-bottom-k 2",
            "--freeze-alternating",
            "--freeze-modules q_proj,k_proj",
        ],
        cli_flags={
            "--freeze-top-k": "Freeze top K transformer layers",
            "--freeze-bottom-k": "Freeze bottom K transformer layers",
            "--freeze-alternating": "Freeze alternating layers",
            "--freeze-modules": "Freeze specific modules by name",
            "--unfreeze-modules": "Unfreeze specific modules",
        }
    ),
    
    # ========== REWARD/RLHF COMPONENTS (4) ==========
    "dpo_trainer": ComponentMetadata(
        name="dpo_trainer",
        category=ComponentCategory.REWARD,
        description="Direct Preference Optimization for RLHF alignment",
        module_path="reward.dpo_trainer",
        class_name="DPOTrainer",
        config_class="DPOTrainerConfig",
        required_args=["model", "ref_model", "train_dataset"],
        optional_args=["eval_dataset", "tokenizer"],
        examples=[
            "ntf align --config dpo_alignment.yaml --method dpo",
            "ntf align --config dpo_alignment.yaml --method dpo --override dpo.beta=0.2",
        ],
        cli_flags={
            "--method": "Alignment method (dpo/ppo)",
            "--beta": "DPO temperature parameter",
            "--loss-type": "DPO loss type (sigmoid/hinge/ipo)",
        }
    ),
    
    "ppo_trainer": ComponentMetadata(
        name="ppo_trainer",
        category=ComponentCategory.REWARD,
        description="Proximal Policy Optimization for RLHF with reward model",
        module_path="reward.ppo_trainer",
        class_name="PPOTrainer",
        config_class="PPOTrainerConfig",
        required_args=["policy_model", "ref_model", "reward_model"],
        optional_args=["dataset", "tokenizer"],
        examples=[
            "ntf align --config ppo_alignment.yaml --method ppo",
        ],
        cli_flags={
            "--ppo-epochs": "PPO optimization epochs per batch",
            "--clip-range": "PPO clipping range",
            "--gamma": "Reward discount factor",
            "--lam": "GAE lambda parameter",
        }
    ),
    
    "reward_model": ComponentMetadata(
        name="reward_model",
        category=ComponentCategory.REWARD,
        description="Reward model training for RLHF with pairwise ranking",
        module_path="reward.reward_model",
        class_name="RewardTrainer",
        config_class="RewardConfig",
        required_args=["train_dataset"],
        optional_args=["eval_dataset", "config"],
        examples=[
            "ntf reward-train --config reward_config.yaml",
        ],
        cli_flags={
            "--loss-type": "Reward loss (pairwise/pointwise)",
            "--margin": "Margin for ranking loss",
        }
    ),
    
    "dpo_config": ComponentMetadata(
        name="dpo_config",
        category=ComponentCategory.REWARD,
        description="DPO-specific hyperparameters configuration",
        module_path="reward.dpo_trainer",
        class_name="DPOTrainerConfig",
        optional_args=[
            "beta", "loss_type", "label_smoothing", "truncation_mode",
            "max_length", "max_prompt_length",
        ],
        examples=[
            "--override dpo.beta=0.1 dpo.loss_type=sigmoid",
        ]
    ),
    
    # ========== CONTINUAL LEARNING COMPONENTS (3) ==========
    "ewc_regularizer": ComponentMetadata(
        name="ewc_regularizer",
        category=ComponentCategory.UTILS,
        description="Elastic Weight Consolidation to prevent catastrophic forgetting",
        module_path="utils.continual_learning",
        class_name="EWCRegularizer",
        config_class="EWCConfig",
        required_args=["model", "config"],
        optional_args=[],
        examples=[
            "ntf train --config continual_ewc.yaml",
            "ntf train --config continual_ewc.yaml --override continual_learning.ewc_lambda=500",
        ],
        cli_flags={
            "--ewc-lambda": "EWC regularization strength",
            "--fisher-samples": "Samples for Fisher estimation",
        }
    ),
    
    "replay_buffer": ComponentMetadata(
        name="replay_buffer",
        category=ComponentCategory.UTILS,
        description="Experience replay buffer for continual learning",
        module_path="utils.continual_learning",
        class_name="ReplayBuffer",
        config_class="ReplayConfig",
        required_args=["config"],
        optional_args=[],
        examples=[
            "--override continual_learning.replay.replay_size=2000",
        ],
        cli_flags={
            "--replay-size": "Size of replay buffer",
            "--replay-ratio": "Ratio of replay data in batches",
            "--selection-strategy": "Buffer selection strategy",
        }
    ),
    
    "gem_optimizer": ComponentMetadata(
        name="gem_optimizer",
        category=ComponentCategory.UTILS,
        description="Gradient Episodic Memory optimizer for multi-task learning",
        module_path="utils.continual_learning",
        class_name="GEMOptimizer",
        config_class="GEMConfig",
        required_args=["model", "config"],
        optional_args=[],
        examples=[
            "--override continual_learning.gem.memory_size=200",
        ]
    ),
    
    # ========== EVALUATION COMPONENTS (2) ==========
    "metrics_evaluator": ComponentMetadata(
        name="metrics_evaluator",
        category=ComponentCategory.EVALUATION,
        description="Comprehensive evaluation metrics (perplexity, accuracy, BLEU, ROUGE)",
        module_path="utils.metrics",
        class_name="EvaluationResults",
        optional_args=["compute_generation_metrics", "tokenizer"],
        examples=[
            "ntf evaluate --model-path ./outputs/final --eval-data test.jsonl",
            "ntf evaluate --model-path ./outputs/final --metrics perplexity accuracy bleu",
        ],
        cli_flags={
            "--metrics": "Metrics to compute (perplexity/accuracy/bleu/rouge)",
            "--eval-data": "Path to evaluation dataset",
        }
    ),
    
    "throughput_benchmark": ComponentMetadata(
        name="throughput_benchmark",
        category=ComponentCategory.EVALUATION,
        description="Model throughput benchmarking (prefill and decode)",
        module_path="utils.metrics",
        class_name="benchmark_throughput",
        optional_args=["sequence_length", "batch_size", "num_iterations"],
        examples=[
            "ntf benchmark --model-path ./outputs/final --batch-size 1 --seq-len 512",
        ],
        cli_flags={
            "--batch-size": "Batch size for benchmarking",
            "--seq-len": "Sequence length for benchmarking",
            "--num-iters": "Number of iterations",
        }
    ),
    
    # ========== VERSIONING COMPONENTS (2) ==========
    "model_registry": ComponentMetadata(
        name="model_registry",
        category=ComponentCategory.UTILS,
        description="Model versioning, registry, and release management",
        module_path="utils.versioning",
        class_name="ModelRegistry",
        required_args=["registry_path"],
        optional_args=[],
        examples=[
            "ntf register-model --name my-model --version 1.0.0 --path ./outputs/final",
            "ntf list-models",
            "ntf promote-model --name my-model --version 1.0.0 --stage production",
        ],
        cli_flags={
            "--name": "Model name",
            "--version": "Semantic version (major.minor.patch)",
            "--stage": "Model stage (experimental/development/staging/production)",
        }
    ),
    
    "model_version": ComponentMetadata(
        name="model_version",
        category=ComponentCategory.UTILS,
        description="Semantic versioning for model releases",
        module_path="utils.versioning",
        class_name="ModelVersion",
        optional_args=["major", "minor", "patch"],
        examples=[
            "--version 2.1.0",
        ]
    ),
    
    # ========== DATA COMPONENTS (1) ==========
    "training_dataset": ComponentMetadata(
        name="training_dataset",
        category=ComponentCategory.DATA,
        description="Dataset creation and tokenization utilities",
        module_path="training.data",
        class_name="create_training_dataset",
        required_args=["texts", "tokenizer"],
        optional_args=["max_length", "stride"],
        examples=[
            "Automatically handled via data config paths",
        ],
        cli_flags={
            "--train-path": "Path to training data",
            "--val-path": "Path to validation data",
            "--max-length": "Maximum sequence length",
            "--tokenizer-name": "Tokenizer to use",
        }
    ),
}


def get_component(name: str) -> Optional[ComponentMetadata]:
    """Get component metadata by name."""
    return NTF_COMPONENT_REGISTRY.get(name)


def list_components(category: Optional[ComponentCategory] = None) -> List[str]:
    """List all component names, optionally filtered by category."""
    if category is None:
        return list(NTF_COMPONENT_REGISTRY.keys())
    return [
        name for name, meta in NTF_COMPONENT_REGISTRY.items()
        if meta.category == category
    ]


def get_components_by_category(category: ComponentCategory) -> Dict[str, ComponentMetadata]:
    """Get all components in a category."""
    return {
        name: meta for name, meta in NTF_COMPONENT_REGISTRY.items()
        if meta.category == category
    }


def generate_cli_help() -> str:
    """Generate comprehensive CLI help text."""
    lines = [
        "=" * 70,
        "NEXUSS TRANSFORMER FRAMEWORK - COMPLETE COMPONENT REFERENCE",
        "=" * 70,
        "",
        f"Total Components: {len(NTF_COMPONENT_REGISTRY)}",
        "",
    ]
    
    for category in ComponentCategory:
        components = get_components_by_category(category)
        if components:
            lines.append(f"\n{'='*20} {category.value.upper()} {'='*20}")
            for name, meta in sorted(components.items()):
                lines.append(f"\n  [{name}]")
                lines.append(f"    Description: {meta.description}")
                lines.append(f"    Module: {meta.module_path}.{meta.class_name}")
                if meta.config_class:
                    lines.append(f"    Config: {meta.config_class}")
                if meta.cli_flags:
                    lines.append(f"    CLI Flags:")
                    for flag, desc in meta.cli_flags.items():
                        lines.append(f"      {flag}: {desc}")
                if meta.examples:
                    lines.append(f"    Examples:")
                    for ex in meta.examples[:2]:
                        lines.append(f"      $ {ex}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# Export for CLI usage
__all__ = [
    "NTF_COMPONENT_REGISTRY",
    "ComponentCategory",
    "ComponentMetadata",
    "get_component",
    "list_components",
    "get_components_by_category",
    "generate_cli_help",
]
