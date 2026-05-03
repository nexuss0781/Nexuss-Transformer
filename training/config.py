"""
Training configuration for LLM pre-training and fine-tuning.

Provides comprehensive hyperparameter configuration for training loops,
optimization, checkpointing, and distributed training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class Precision(str, Enum):
    """Mixed precision options."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAMW_8BIT = "adamw_8bit"
    ADAMW_FUSED = "adamw_fused"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


@dataclass
class TrainingConfig:
    """
    Configuration for LLM training.
    
    Args:
        # Core training parameters
        output_dir: Directory to save checkpoints and logs
        num_train_epochs: Number of training epochs
        max_steps: Maximum number of training steps (overrides epochs if set)
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Steps to accumulate gradients before update
        
        # Optimization
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        optimizer: Optimizer type
        scheduler: Learning rate scheduler type
        warmup_steps: Number of warmup steps
        warmup_ratio: Warmup ratio (overrides warmup_steps if set)
        
        # Precision and performance
        mixed_precision: Mixed precision mode
        gradient_checkpointing: Enable gradient checkpointing
        use_flash_attention: Use flash attention if available
        
        # Checkpointing
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        logging_steps: Log metrics every N steps
        eval_steps: Evaluate every N steps (None uses save_steps)
        
        # Distributed training
        ddp_find_unused_parameters: Find unused parameters in DDP
        dataloader_num_workers: Number of data loading workers
        dataloader_pin_memory: Pin memory for data loading
        
        # Regularization
        max_grad_norm: Maximum gradient norm for clipping
        dropout: Dropout probability (overrides model config if set)
        
        # Logging and monitoring
        report_to: Reporting backends ('wandb', 'tensorboard', 'none')
        project_name: Project name for wandb/tensorboard
        run_name: Run name identifier
        
        # Advanced
        seed: Random seed for reproducibility
        bf16_full_eval: Use BF16 for evaluation
        prediction_loss_only: Only compute loss during evaluation
        remove_unused_columns: Remove unused columns from dataset
        label_names: Names of label columns in dataset
    """
    
    # Core training parameters
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.LINEAR
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Precision and performance
    mixed_precision: Precision = Precision.FP32
    gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    
    # Distributed training
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: Optional[int] = None
    dataloader_pin_memory: bool = True
    
    # Regularization
    max_grad_norm: float = 1.0
    dropout: Optional[float] = None
    
    # Logging and monitoring
    report_to: str = "none"
    project_name: str = "llm_training"
    run_name: Optional[str] = None
    
    # Advanced
    seed: int = 42
    bf16_full_eval: bool = False
    prediction_loss_only: bool = False
    remove_unused_columns: bool = True
    label_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate batch sizes
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("per_device_eval_batch_size must be positive")
        
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Validate warmup
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        
        # Validate steps
        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive")
        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be positive")
        
        # Set eval_steps if not provided
        if self.eval_steps is None:
            self.eval_steps = self.save_steps
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size across devices and accumulation."""
        import torch
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        return (
            self.per_device_train_batch_size
            * world_size
            * self.gradient_accumulation_steps
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result
    
    @classmethod
    def small_model(cls) -> "TrainingConfig":
        """Default config for small models (~60M params)."""
        return cls(
            learning_rate=1e-3,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=2,
            warmup_ratio=0.05,
            num_train_epochs=10,
        )
    
    @classmethod
    def medium_model(cls) -> "TrainingConfig":
        """Default config for medium models (~350M params)."""
        return cls(
            learning_rate=5e-4,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            num_train_epochs=5,
        )
    
    @classmethod
    def large_model(cls) -> "TrainingConfig":
        """Default config for large models (1B+ params)."""
        return cls(
            learning_rate=1e-4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_ratio=0.02,
            num_train_epochs=3,
            gradient_checkpointing=True,
            mixed_precision=Precision.BF16,
        )
