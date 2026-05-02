"""
PEFT/LoRA fine-tuning implementation.

Uses Hugging Face PEFT library for parameter-efficient fine-tuning
with LoRA, QLoRA, and other adapter methods.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from enum import Enum

import torch
import torch.nn as nn

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)


class PeftMethod(str, Enum):
    """PEFT method types."""
    LORA = "lora"
    ADALORA = "adalora"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"
    PROMPT_TUNING = "prompt_tuning"


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.
    
    Args:
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA layers
        target_modules: Module names to apply LoRA to
        bias: Whether to train bias parameters
        task_type: Task type for PEFT
        inference_mode: Whether to use inference mode
        modules_to_save: Modules to save in addition to adapters
        init_lora_weights: Initialization method for LoRA weights
        layers_to_transform: Layers to transform
        layer_pattern: Pattern to match layer names
        rank_pattern: Rank pattern for AdaLoRA
        alpha_pattern: Alpha pattern for scaling
    """
    
    # Core LoRA parameters
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    
    # Target modules (None = auto-detect)
    target_modules: Optional[Union[List[str], str]] = None
    
    # Training options
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: TaskType = TaskType.CAUSAL_LM
    inference_mode: bool = False
    
    # Additional modules
    modules_to_save: Optional[List[str]] = None
    
    # Initialization
    init_lora_weights: bool = True
    
    # Advanced
    layers_to_transform: Optional[Union[List[int], int]] = None
    layer_pattern: Optional[str] = None
    rank_pattern: Optional[Dict] = None
    alpha_pattern: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.r <= 0:
            raise ValueError("LoRA rank 'r' must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
    
    @property
    def scaling(self) -> float:
        """Calculate LoRA scaling factor."""
        return self.alpha / self.r
    
    def to_peft_config(self) -> LoraConfig:
        """Convert to Hugging Face PEFT LoraConfig."""
        # Build config dict with only supported parameters
        config_kwargs = {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
            "modules_to_save": self.modules_to_save,
            "init_lora_weights": self.init_lora_weights,
            "layers_to_transform": self.layers_to_transform,
        }
        
        # Only add rank_pattern and alpha_pattern if they are not None (must be dict type)
        if self.rank_pattern is not None and isinstance(self.rank_pattern, dict):
            config_kwargs["rank_pattern"] = self.rank_pattern
        if self.alpha_pattern is not None and isinstance(self.alpha_pattern, dict):
            config_kwargs["alpha_pattern"] = self.alpha_pattern
        
        # Use layers_pattern instead of layer_pattern (newer PEFT API)
        if self.layer_pattern is not None:
            config_kwargs["layers_pattern"] = [self.layer_pattern] if isinstance(self.layer_pattern, str) else self.layer_pattern
        
        return LoraConfig(**config_kwargs)
    
    @classmethod
    def default(cls) -> "LoRAConfig":
        """Default LoRA config for decoder-only transformers."""
        return cls(
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
    
    @classmethod
    def full_attention(cls) -> "LoRAConfig":
        """LoRA config targeting all attention modules."""
        return cls(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    
    @classmethod
    def full_model(cls) -> "LoRAConfig":
        """LoRA config targeting all linear layers."""
        return cls(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )


class PEFTTrainer:
    """
    Trainer for parameter-efficient fine-tuning.
    
    Wraps a base model with PEFT adapters and provides training
    capabilities with frozen base weights.
    
    Args:
        model: Base model to fine-tune
        config: LoRA configuration
        tokenizer: Optional tokenizer for preprocessing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig,
        tokenizer=None,
    ):
        self.base_model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Prepare model for k-bit training if needed
        if hasattr(model, "is_loaded_in_4bit") or hasattr(model, "is_loaded_in_8bit"):
            self.model = prepare_model_for_kbit_training(model)
        else:
            self.model = model
        
        # Apply PEFT
        peft_config = config.to_peft_config()
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        """Print number of trainable vs total parameters."""
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"All params: {all_params:,}")
        print(f"Frozen params: {all_params - trainable_params:,}")
    
    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """Get dictionary of trainable parameters."""
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def freeze_base_model(self):
        """Ensure base model is frozen."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze base model for full fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def merge_and_unload(self) -> nn.Module:
        """
        Merge LoRA weights into base model and unload adapters.
        
        Returns:
            Merged model without adapter structure
        """
        if isinstance(self.model, PeftModel):
            return self.model.merge_and_unload()
        return self.model
    
    def save_adapter(self, output_dir: str):
        """Save only the adapter weights."""
        self.model.save_pretrained(output_dir)
    
    def load_adapter(self, adapter_path: str):
        """Load adapter weights."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
    
    def get_model(self) -> nn.Module:
        """Get the wrapped model."""
        return self.model
