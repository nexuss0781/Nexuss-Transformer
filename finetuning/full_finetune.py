"""
Full fine-tuning implementation.

Provides full parameter fine-tuning with optional layer freezing
for partial fine-tuning scenarios.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

from training.trainer import Trainer, TrainingConfig
from training.data import DataCollatorForLanguageModeling


class FullFinetuneTrainer(Trainer):
    """
    Full fine-tuning trainer that overrides all model weights.
    
    Extends the base Trainer with fine-tuning specific optimizations:
    - Discriminative learning rates by layer
    - Gradual unfreezing
    - Layer-wise learning rate decay
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        discriminative_lr: Optional[List[float]] = None,
        layerwise_lr_decay: float = 1.0,
    ):
        """
        Initialize full fine-tuning trainer.
        
        Args:
            model: Model to fine-tune
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collation function
            discriminative_lr: Different learning rates per layer group
            layerwise_lr_decay: Decay factor for deeper layers
        """
        self.discriminative_lr = discriminative_lr
        self.layerwise_lr_decay = layerwise_lr_decay
        
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with optional discriminative learning rates."""
        if self.discriminative_lr is not None:
            return self._create_discriminative_optimizer()
        elif self.layerwise_lr_decay != 1.0:
            return self._create_layerwise_decay_optimizer()
        else:
            return super()._create_optimizer()
    
    def _create_discriminative_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for different parts."""
        param_groups = []
        
        # Get model components
        embed_params = list(self.model.embed_tokens.parameters())
        transformer_params = list(self.model.layers.parameters())
        norm_params = list(self.model.norm.parameters())
        
        if self.model.lm_head is not None:
            head_params = list(self.model.lm_head.parameters())
        else:
            head_params = embed_params  # Tied weights
        
        n_layers = len(self.model.layers)
        lr_per_layer = self._distribute_lrs(n_layers)
        
        # Embedding layer (lowest LR)
        param_groups.append({
            "params": embed_params,
            "lr": self.discriminative_lr[0] if len(self.discriminative_lr) > 0 else self.config.learning_rate * 0.1,
            "weight_decay": self.config.weight_decay,
            "name": "embeddings",
        })
        
        # Transformer layers (increasing LR from bottom to top)
        for i, layer in enumerate(self.model.layers):
            layer_lr = lr_per_layer[i] if i < len(lr_per_layer) else self.config.learning_rate
            param_groups.append({
                "params": layer.parameters(),
                "lr": layer_lr,
                "weight_decay": self.config.weight_decay,
                "name": f"layer_{i}",
            })
        
        # Final norm and head (highest LR)
        param_groups.append({
            "params": norm_params,
            "lr": self.discriminative_lr[-1] if len(self.discriminative_lr) > 1 else self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "name": "final_norm",
        })
        
        if self.model.lm_head is not None:
            param_groups.append({
                "params": head_params,
                "lr": self.discriminative_lr[-1] if len(self.discriminative_lr) > 1 else self.config.learning_rate,
                "weight_decay": 0.0,  # No weight decay on output head
                "name": "lm_head",
            })
        
        from torch.optim import AdamW
        return AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    
    def _create_layerwise_decay_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with layer-wise learning rate decay."""
        no_decay = ["bias", "layer_norm.weight", "norm.weight"]
        
        # Get number of layers
        if hasattr(self.model, 'layers'):
            n_layers = len(self.model.layers)
        else:
            n_layers = 1
        
        optimizer_grouped_parameters = []
        
        for i in range(n_layers + 1):  # +1 for embeddings/head
            # Calculate layer-specific learning rate
            layer_lr = self.config.learning_rate * (self.layerwise_lr_decay ** (n_layers - i))
            
            # Get parameters for this layer
            if i == 0:
                params = list(self.model.embed_tokens.parameters())
                name_prefix = "embed"
            elif i == n_layers:
                params = list(self.model.norm.parameters())
                if self.model.lm_head is not None:
                    params.extend(self.model.lm_head.parameters())
                name_prefix = "output"
            else:
                params = list(self.model.layers[i-1].parameters())
                name_prefix = f"layer{i-1}"
            
            # Split into decay and no-decay groups
            for use_decay in [True, False]:
                grouped_params = [
                    p for n, p in zip([name_prefix] * len(params), params)
                    if (any(nd in n for nd in no_decay) != (not use_decay))
                    and p.requires_grad
                ]
                
                if grouped_params:
                    optimizer_grouped_parameters.append({
                        "params": grouped_params,
                        "weight_decay": self.config.weight_decay if use_decay else 0.0,
                        "lr": layer_lr,
                    })
        
        from torch.optim import AdamW
        return AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95), eps=1e-8)
    
    def _distribute_lrs(self, n_layers: int) -> List[float]:
        """Distribute learning rates across layers."""
        if len(self.discriminative_lr) == 2:
            # Linear interpolation between min and max LR
            min_lr, max_lr = self.discriminative_lr
            return [
                min_lr + (max_lr - min_lr) * (i / (n_layers - 1))
                for i in range(n_layers)
            ]
        else:
            # Use provided LRs or default
            return [self.config.learning_rate] * n_layers
    
    def gradual_unfreeze(
        self,
        start_layers: int = 1,
        total_epochs: int = 3,
        current_epoch: int = 0,
    ):
        """
        Gradually unfreeze layers during training.
        
        Args:
            start_layers: Number of layers to start with (top layers)
            total_epochs: Total number of epochs
            current_epoch: Current epoch number
        """
        n_layers = len(self.model.layers)
        
        # Calculate how many layers to unfreeze
        layers_to_unfreeze = start_layers + (
            (n_layers - start_layers) * current_epoch // total_epochs
        )
        
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze top layers
        for i in range(n_layers - layers_to_unfreeze, n_layers):
            for param in self.model.layers[i].parameters():
                param.requires_grad = True
        
        # Always keep norm and head trainable
        for param in self.model.norm.parameters():
            param.requires_grad = True
        if self.model.lm_head is not None:
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
        
        print(f"Epoch {current_epoch}: Unfrozen {layers_to_unfreeze}/{n_layers} layers")
