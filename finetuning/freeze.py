"""
Layer freezing utilities for fine-tuning.

Provides flexible layer freezing strategies for partial fine-tuning,
including top-k, bottom-k, alternating, and custom patterns.
"""

from typing import List, Optional, Union, Callable
import torch.nn as nn


class LayerFreezer:
    """
    Utility class for freezing/unfreezing model layers.
    
    Supports various freezing strategies:
    - Top-k layers (last k layers)
    - Bottom-k layers (first k layers)
    - Alternating layers
    - Custom layer selection
    - Named module freezing
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize layer freezer.
        
        Args:
            model: Model to apply freezing to
        """
        self.model = model
        self.original_state = {}
        self._save_original_state()
    
    def _save_original_state(self):
        """Save original requires_grad state of all parameters."""
        for name, param in self.model.named_parameters():
            self.original_state[name] = param.requires_grad
    
    def freeze_all(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        print("Frozen all parameters")
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Unfrozen all parameters")
    
    def freeze_top_k(self, k: int, include_norm: bool = True, include_head: bool = True):
        """
        Freeze the top-k transformer layers.
        
        Args:
            k: Number of top layers to freeze
            include_norm: Whether to freeze final normalization
            include_head: Whether to freeze output head
        """
        if not hasattr(self.model, 'layers'):
            raise ValueError("Model must have 'layers' attribute")
        
        n_layers = len(self.model.layers)
        
        # Freeze top-k layers
        for i in range(n_layers - k, n_layers):
            for param in self.model.layers[i].parameters():
                param.requires_grad = False
        
        # Optionally freeze norm and head
        if include_norm:
            for param in self.model.norm.parameters():
                param.requires_grad = False
        
        if include_head and hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
            for param in self.model.lm_head.parameters():
                param.requires_grad = False
        
        print(f"Frozen top {k}/{n_layers} layers")
    
    def freeze_bottom_k(self, k: int):
        """
        Freeze the bottom-k transformer layers.
        
        Args:
            k: Number of bottom layers to freeze
        """
        if not hasattr(self.model, 'layers'):
            raise ValueError("Model must have 'layers' attribute")
        
        n_layers = len(self.model.layers)
        
        # Freeze bottom-k layers
        for i in range(min(k, n_layers)):
            for param in self.model.layers[i].parameters():
                param.requires_grad = False
        
        print(f"Frozen bottom {k}/{n_layers} layers")
    
    def freeze_alternating(self, pattern: str = "freeze"):
        """
        Freeze layers in an alternating pattern.
        
        Args:
            pattern: Either "freeze" (freeze even, keep odd) or "unfreeze" (keep even, freeze odd)
        """
        if not hasattr(self.model, 'layers'):
            raise ValueError("Model must have 'layers' attribute")
        
        n_layers = len(self.model.layers)
        
        for i in range(n_layers):
            should_freeze = (i % 2 == 0) if pattern == "freeze" else (i % 2 == 1)
            for param in self.model.layers[i].parameters():
                param.requires_grad = not should_freeze
        
        print(f"Applied alternating {pattern} pattern")
    
    def freeze_by_name(
        self,
        module_names: Union[str, List[str]],
        exclude: Optional[List[str]] = None,
    ):
        """
        Freeze modules by name pattern.
        
        Args:
            module_names: Name or list of names/patterns to freeze
            exclude: Names/patterns to exclude from freezing
        """
        if isinstance(module_names, str):
            module_names = [module_names]
        
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # Check if any pattern matches
            should_freeze = any(pattern in name for pattern in module_names)
            
            # Check exclusions
            if exclude and any(ex_pattern in name for ex_pattern in exclude):
                should_freeze = False
            
            if should_freeze:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"Frozen {frozen_count} parameters matching patterns: {module_names}")
    
    def unfreeze_by_name(
        self,
        module_names: Union[str, List[str]],
    ):
        """
        Unfreeze modules by name pattern.
        
        Args:
            module_names: Name or list of names/patterns to unfreeze
        """
        if isinstance(module_names, str):
            module_names = [module_names]
        
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in module_names):
                param.requires_grad = True
                unfrozen_count += 1
        
        print(f"Unfrozen {unfrozen_count} parameters matching patterns: {module_names}")
    
    def freeze_except(
        self,
        keep_trainable: Union[str, List[str]],
    ):
        """
        Freeze everything except specified modules.
        
        Args:
            keep_trainable: Module names/patterns to keep trainable
        """
        if isinstance(keep_trainable, str):
            keep_trainable = [keep_trainable]
        
        # First freeze all
        self.freeze_all()
        
        # Then unfreeze specified
        self.unfreeze_by_name(keep_trainable)
    
    def get_trainable_layers(self) -> List[str]:
        """Get list of trainable layer/module names."""
        trainable = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Extract layer name
                parts = name.split('.')
                if len(parts) > 2 and parts[0] == 'layers':
                    layer_name = f"layers.{parts[1]}"
                    if layer_name not in trainable:
                        trainable.append(layer_name)
                else:
                    root = parts[0]
                    if root not in trainable:
                        trainable.append(root)
        
        return trainable
    
    def get_frozen_layers(self) -> List[str]:
        """Get list of frozen layer/module names."""
        frozen = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                parts = name.split('.')
                if len(parts) > 2 and parts[0] == 'layers':
                    layer_name = f"layers.{parts[1]}"
                    if layer_name not in frozen:
                        frozen.append(layer_name)
                else:
                    root = parts[0]
                    if root not in frozen:
                        frozen.append(root)
        
        return frozen
    
    def print_status(self):
        """Print detailed freezing status."""
        trainable_params = sum(1 for p in self.model.parameters() if p.requires_grad)
        frozen_params = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params
        
        print(f"\n{'='*50}")
        print(f"Parameter Status:")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"  Total: {total_params:,}")
        print(f"\nTrainable modules: {self.get_trainable_layers()}")
        print(f"Frozen modules: {self.get_frozen_layers()}")
        print(f"{'='*50}\n")
    
    def restore_original_state(self):
        """Restore original requires_grad state."""
        for name, param in self.model.named_parameters():
            if name in self.original_state:
                param.requires_grad = self.original_state[name]
        print("Restored original parameter states")
