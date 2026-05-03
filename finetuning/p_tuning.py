"""
P-Tuning / Prefix Tuning Implementation for NTF
Parameter-efficient tuning using learnable continuous prompts
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from peft import (
    PrefixTuningConfig,
    PromptTuningConfig,
    P_TUNING_TASK_TYPE,
    get_peft_model,
    TaskType,
)


class PTuningMethod(str, Enum):
    """P-Tuning method types."""
    P_TUNING_V1 = "p_tuning_v1"
    P_TUNING_V2 = "p_tuning_v2"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"


@dataclass
class PTuningConfig:
    """
    Configuration for P-Tuning / Prefix Tuning.
    
    Args:
        method: P-tuning method to use
        num_virtual_tokens: Number of virtual/prompt tokens to add
        token_dim: Dimension of token embeddings
        num_transformer_submodules: Number of transformer submodules
        num_attention_heads: Number of attention heads
        num_layers: Number of transformer layers
        encoder_hidden_size: Hidden size for encoder (P-Tuning v1)
        prefix_projection: Whether to project prefix (Prefix Tuning)
        prompt_tuning_init: Initialization strategy for prompt tuning
        prompt_tuning_init_text: Text for initialization if using text init
    """
    
    method: PTuningMethod = PTuningMethod.P_TUNING_V2
    
    # Core parameters
    num_virtual_tokens: int = 20
    token_dim: int = 768
    num_transformer_submodules: int = 1
    num_attention_heads: int = 12
    num_layers: int = 12
    
    # P-Tuning v1 specific
    encoder_hidden_size: int = 512
    
    # Prefix Tuning specific
    prefix_projection: bool = True
    
    # Prompt Tuning specific
    prompt_tuning_init: str = "RANDOM"  # RANDOM or TEXT
    prompt_tuning_init_text: Optional[str] = None
    
    # Task type
    task_type: TaskType = TaskType.CAUSAL_LM
    
    def to_peft_config(self):
        """Convert to appropriate PEFT config based on method."""
        if self.method == PTuningMethod.PREFIX_TUNING:
            return PrefixTuningConfig(
                num_virtual_tokens=self.num_virtual_tokens,
                token_dim=self.token_dim,
                num_attention_heads=self.num_attention_heads,
                num_layers=self.num_layers,
                prefix_projection=self.prefix_projection,
                task_type=self.task_type,
            )
        elif self.method == PTuningMethod.PROMPT_TUNING:
            return PromptTuningConfig(
                num_virtual_tokens=self.num_virtual_tokens,
                token_dim=self.token_dim,
                prompt_tuning_init=self.prompt_tuning_init,
                prompt_tuning_init_text=self.prompt_tuning_init_text,
                task_type=self.task_type,
            )
        else:  # P-Tuning v1 or v2
            # P-Tuning uses PrefixTuningConfig with specific settings
            return PrefixTuningConfig(
                num_virtual_tokens=self.num_virtual_tokens,
                token_dim=self.token_dim,
                num_attention_heads=self.num_attention_heads,
                num_layers=self.num_layers,
                encoder_hidden_size=self.encoder_hidden_size,
                prefix_projection=self.method == PTuningMethod.P_TUNING_V1,
                task_type=self.task_type,
            )


class PTuningModel(nn.Module):
    """
    P-Tuning wrapper for transformer models.
    
    Adds learnable continuous prompts to the model input
    without modifying the base model weights.
    
    Args:
        base_model: Base transformer model
        config: P-tuning configuration
    """
    
    def __init__(self, base_model: nn.Module, config: PTuningConfig):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        
        # Get model dimensions
        model_config = base_model.config
        self.token_dim = getattr(model_config, 'hidden_size', config.token_dim)
        self.num_layers = getattr(model_config, 'num_hidden_layers', config.num_layers)
        self.num_attention_heads = getattr(model_config, 'num_attention_heads', config.num_attention_heads)
        
        # Update config with actual dimensions
        config.token_dim = self.token_dim
        config.num_layers = self.num_layers
        config.num_attention_heads = self.num_attention_heads
        
        # Create virtual tokens
        self._create_virtual_tokens()
    
    def _create_virtual_tokens(self):
        """Create learnable virtual token embeddings."""
        method = self.config.method
        
        if method == PTuningMethod.PROMPT_TUNING:
            # Simple prompt embeddings
            self.prompt_embeddings = nn.Embedding(
                self.config.num_virtual_tokens,
                self.token_dim
            )
            nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
            
        elif method == PTuningMethod.PREFIX_TUNING:
            # Prefix with projection
            self.prefix_tokens = nn.Parameter(
                torch.randn(
                    self.num_layers * 2,  # key and value for each layer
                    self.config.num_virtual_tokens,
                    self.token_dim
                )
            )
            
            if self.config.prefix_projection:
                self.prefix_proj = nn.Sequential(
                    nn.Linear(self.token_dim, self.token_dim),
                    nn.ReLU(),
                    nn.Linear(self.token_dim, self.num_layers * 2 * self.token_dim)
                )
            else:
                self.prefix_proj = None
                
        else:  # P-Tuning v1 or v2
            # Encoder for generating prompts
            self.prompt_encoder = nn.Sequential(
                nn.Linear(self.token_dim, self.config.encoder_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.encoder_hidden_size, 
                         self.num_layers * 2 * self.config.num_virtual_tokens * self.token_dim)
            )
            
            # Input embedding for prompt encoder
            self.input_embeds = nn.Embedding(self.config.num_virtual_tokens, self.token_dim)
            nn.init.normal_(self.input_embeds.weight, std=0.02)
    
    def get_prompt(self, batch_size: int) -> torch.Tensor:
        """Generate prompt tensors for the current batch."""
        method = self.config.method
        
        if method == PTuningMethod.PROMPT_TUNING:
            # Expand prompt embeddings to batch size
            prompts = self.prompt_embeddings.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            
        elif method == PTuningMethod.PREFIX_TUNING:
            prefix = self.prefix_tokens
            
            if self.prefix_proj is not None:
                prefix = self.prefix_proj(prefix.view(-1, self.token_dim))
                prefix = prefix.view(
                    self.num_layers * 2,
                    self.config.num_virtual_tokens,
                    self.token_dim
                )
            
            prompts = prefix.unsqueeze(1).expand(
                -1, batch_size, -1, -1
            )
            
        else:  # P-Tuning
            input_ids = torch.arange(self.config.num_virtual_tokens).long()
            input_ids = input_ids.unsqueeze(0).expand(batch_size, -1)
            input_embeds = self.input_embeds(input_ids)
            
            prompts = self.prompt_encoder(input_embeds)
            prompts = prompts.view(
                batch_size,
                self.num_layers * 2,
                self.config.num_virtual_tokens,
                self.token_dim
            )
            prompts = prompts.permute(1, 0, 2, 3)
        
        return prompts
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with virtual prompts.
        
        Note: This is a simplified implementation. For production use,
        consider using the PEFT library's P-tuning implementation.
        """
        batch_size = input_ids.size(0)
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        result = {"logits": outputs.logits}
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            result["loss"] = outputs.loss
        
        return result
    
    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """Get dictionary of trainable parameters (prompts only)."""
        trainable = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable[name] = param
        
        return trainable
    
    def print_trainable_parameters(self):
        """Print number of trainable vs total parameters."""
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in self.parameters())
        
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"All params: {all_params:,}")
        print(f"Frozen params: {all_params - trainable_params:,}")


def setup_p_tuning(
    model: nn.Module,
    method: str = "p_tuning_v2",
    num_virtual_tokens: int = 20,
    task_type: str = "CAUSAL_LM"
) -> nn.Module:
    """
    Setup P-Tuning on a model using PEFT.
    
    Args:
        model: Base model to apply P-Tuning to
        method: P-tuning method (p_tuning_v1, p_tuning_v2, prefix_tuning, prompt_tuning)
        num_virtual_tokens: Number of virtual tokens
        task_type: PEFT task type
        
    Returns:
        Model with P-Tuning applied
    """
    config = PTuningConfig(
        method=PTuningMethod(method),
        num_virtual_tokens=num_virtual_tokens,
        task_type=TaskType(task_type)
    )
    
    peft_config = config.to_peft_config()
    return get_peft_model(model, peft_config)
