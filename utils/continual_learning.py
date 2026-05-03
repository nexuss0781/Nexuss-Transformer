"""
Continual Learning Utilities for Nexuss Transformer Framework
Mechanisms to avoid catastrophic forgetting during continuous training
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import OrderedDict
import copy


@dataclass
class EWCConfig:
    """Configuration for Elastic Weight Consolidation"""
    
    ewc_lambda: float = 1000.0  # Strength of EWC regularization
    fisher_samples: int = 200  # Number of samples to estimate Fisher information
    damping: float = 0.1  # Damping factor for Fisher matrix
    mc_samples: int = 1  # Monte Carlo samples for Fisher estimation
    

@dataclass
class ReplayConfig:
    """Configuration for Experience Replay"""
    
    replay_size: int = 1000  # Size of replay buffer
    replay_ratio: float = 0.5  # Ratio of replay data in each batch
    selection_strategy: str = "uniform"  # uniform, recent, diverse
    reservoir_sampling: bool = True  # Use reservoir sampling for streaming data


@dataclass
class GEMConfig:
    """Configuration for Gradient Episodic Memory"""
    
    memory_size: int = 100  # Number of examples per task
    num_tasks: int = 5  # Expected number of tasks
    use_quadprog: bool = True  # Use quadratic programming for constraint solving
    

@dataclass
class ContinualLearningConfig:
    """Unified configuration for continual learning strategies"""
    
    strategy: str = "none"  # none, ewc, replay, gem, lwf
    ewc: Optional[EWCConfig] = field(default_factory=EWCConfig)
    replay: Optional[ReplayConfig] = field(default_factory=ReplayConfig)
    gem: Optional[GEMConfig] = field(default_factory=GEMConfig)
    
    # LwF (Learning without Forgetting) settings
    lwf_alpha: float = 1.0  # Distillation loss weight
    lwf_temperature: float = 2.0  # Temperature for knowledge distillation
    
    # Regularization
    weight_decay: float = 0.01
    grad_clip: float = 1.0


class EWCRegularizer:
    """Elastic Weight Consolidation implementation"""
    
    def __init__(self, model: nn.Module, config: EWCConfig):
        self.model = model
        self.config = config
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
    def compute_fisher(self, dataloader: DataLoader, device: torch.device):
        """Compute Fisher Information Matrix diagonal approximation"""
        
        self.model.train()
        fisher_dict = {name: torch.zeros_like(param) 
                      for name, param in self.model.named_parameters() 
                      if param.requires_grad}
        
        samples_processed = 0
        
        for batch in dataloader:
            if samples_processed >= self.config.fisher_samples:
                break
            
            self.model.zero_grad()
            
            # Forward pass
            inputs = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            outputs = self.model(inputs)
            
            # Compute log-likelihood gradient
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            loss = log_probs.mean()
            
            # Compute gradients
            grads = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], 
                                       retain_graph=False)
            
            # Accumulate squared gradients (Fisher diagonal)
            for (name, _), grad in zip(self.model.named_parameters(), grads):
                if name in fisher_dict:
                    fisher_dict[name] += grad.pow(2)
            
            samples_processed += inputs.size(0)
        
        # Average and store
        n_samples = max(samples_processed, 1)
        self.fisher = {name: tensor / n_samples + self.config.damping 
                      for name, tensor in fisher_dict.items()}
        
        # Store optimal parameters
        self.optimal_params = {name: param.clone().detach() 
                              for name, param in self.model.named_parameters() 
                              if param.requires_grad}
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        
        if not self.fisher or not self.optimal_params:
            return torch.tensor(0.0)
        
        ewc_loss = torch.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher:
                delta = param - self.optimal_params[name]
                ewc_loss += (self.fisher[name] * delta.pow(2)).sum()
        
        return self.config.ewc_lambda * ewc_loss


class ReplayBuffer:
    """Experience Replay Buffer for continual learning"""
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.buffer: List[Dict[str, Any]] = []
        self.task_data: Dict[int, List[Dict[str, Any]]] = {}
        
    def add(self, samples: List[Dict[str, Any]], task_id: Optional[int] = None):
        """Add samples to replay buffer"""
        
        if self.config.reservoir_sampling and len(self.buffer) + len(samples) > self.config.replay_size:
            # Reservoir sampling for streaming data
            for sample in samples:
                if len(self.buffer) < self.config.replay_size:
                    self.buffer.append(sample)
                else:
                    # Randomly replace with decreasing probability
                    j = torch.randint(0, len(self.buffer) + 1, (1,)).item()
                    if j < self.config.replay_size:
                        self.buffer[j] = sample
        else:
            self.buffer.extend(samples)
            
            # Trim if exceeds size
            if len(self.buffer) > self.config.replay_size:
                if self.config.selection_strategy == "recent":
                    self.buffer = self.buffer[-self.config.replay_size:]
                elif self.config.selection_strategy == "diverse":
                    # Simple diversity: keep every nth item
                    step = len(self.buffer) // self.config.replay_size
                    self.buffer = self.buffer[::step][:self.config.replay_size]
                else:  # uniform
                    indices = torch.randperm(len(self.buffer))[:self.config.replay_size]
                    self.buffer = [self.buffer[i] for i in indices]
        
        # Store by task if task_id provided
        if task_id is not None:
            if task_id not in self.task_data:
                self.task_data[task_id] = []
            self.task_data[task_id].extend(samples)
    
    def get_batch(self, current_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Mix replay data with current batch"""
        
        if not self.buffer:
            return current_batch
        
        replay_size = int(current_batch["input_ids"].size(0) * self.config.replay_ratio)
        replay_size = min(replay_size, len(self.buffer))
        
        if replay_size == 0:
            return current_batch
        
        # Sample from buffer
        indices = torch.randperm(len(self.buffer))[:replay_size]
        replay_samples = [self.buffer[i] for i in indices]
        
        # Combine with current batch (simplified - in practice need proper merging)
        # This is a placeholder - actual implementation depends on your data format
        return current_batch  # TODO: Implement proper batch merging
    
    def get_task_buffer(self, task_id: int) -> List[Dict[str, Any]]:
        """Get replay buffer for specific task"""
        return self.task_data.get(task_id, [])


class GEMOptimizer:
    """Gradient Episodic Memory optimizer"""
    
    def __init__(self, model: nn.Module, config: GEMConfig):
        self.model = model
        self.config = config
        self.memory: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(config.num_tasks)}
        self.gradient_memory: Dict[int, torch.Tensor] = {}
        
    def store_in_memory(self, samples: List[Dict[str, Any]], task_id: int):
        """Store samples in task-specific memory"""
        
        available_space = self.config.memory_size - len(self.memory[task_id])
        
        if available_space >= len(samples):
            self.memory[task_id].extend(samples)
        else:
            # Random subsample
            indices = torch.randperm(len(samples))[:available_space]
            self.memory[task_id].extend([samples[i] for i in indices])
    
    def compute_gradient_constraints(self, task_id: int, device: torch.device) -> List[torch.Tensor]:
        """Compute stored gradients for previous tasks"""
        
        constraints = []
        
        for prev_task_id in range(task_id):
            if prev_task_id not in self.gradient_memory:
                continue
            
            constraints.append(self.gradient_memory[prev_task_id])
        
        return constraints
    
    def project_gradient(self, gradient: torch.Tensor, constraints: List[torch.Tensor]) -> torch.Tensor:
        """Project gradient to satisfy memory constraints using quadratic programming"""
        
        if not constraints:
            return gradient
        
        projected = gradient.clone()
        
        for constraint in constraints:
            # Check if gradient violates constraint
            dot_product = torch.dot(projected.flatten(), constraint.flatten())
            
            if dot_product < 0:
                # Project gradient
                norm_sq = constraint.pow(2).sum()
                if norm_sq > 1e-8:
                    projection_coef = dot_product / norm_sq
                    projected -= projection_coef * constraint
        
        return projected
    
    def update_gradient_memory(self, task_id: int, dataloader: DataLoader, device: torch.device):
        """Update stored gradients for current task"""
        
        self.model.eval()
        
        # Compute average gradient over memory samples
        total_gradient = None
        count = 0
        
        for batch in dataloader:
            self.model.zero_grad()
            
            inputs = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            outputs = self.model(inputs)
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.mean()
            
            grads = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad])
            
            # Flatten and concatenate all gradients
            flat_grad = torch.cat([g.flatten() for g in grads])
            
            if total_gradient is None:
                total_gradient = flat_grad
            else:
                total_gradient += flat_grad
            
            count += 1
        
        if count > 0 and total_gradient is not None:
            self.gradient_memory[task_id] = total_gradient / count


class LwFLoss(nn.Module):
    """Learning without Forgetting loss using knowledge distillation"""
    
    def __init__(self, config: ContinualLearningConfig):
        super().__init__()
        self.config = config
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute LwF distillation loss"""
        
        T = self.config.lwf_temperature
        
        # Apply temperature scaling
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        
        # Knowledge distillation loss
        kd_loss = self.kl_div(student_log_probs, teacher_probs) * (T ** 2)
        
        return self.config.lwf_alpha * kd_loss


class SIRegularizer:
    """Synaptic Intelligence implementation for continual learning."""
    
    def __init__(self, model: nn.Module, c: float = 0.1):
        self.model = model
        self.c = c  # Importance weight
        self.importance: Dict[str, torch.Tensor] = {}
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.trajectory: Dict[str, torch.Tensor] = {}
        
    def initialize_trajectory(self):
        """Initialize trajectory tracking for parameters."""
        self.prev_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.trajectory = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.importance = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def update_trajectory(self):
        """Update parameter change trajectory after each training step."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.prev_params:
                    delta = param - self.prev_params[name]
                    self.trajectory[name] += delta.pow(2)
                    self.prev_params[name] = param.clone().detach()
    
    def compute_importance(self, loss_change: float):
        """
        Compute parameter importance based on loss change.
        
        Args:
            loss_change: Change in loss from previous iteration
        """
        if loss_change > 0:  # Only update if loss decreased
            for name in self.importance:
                if name in self.trajectory:
                    denom = self.trajectory[name] + 1e-8
                    self.importance[name] += loss_change / denom
    
    def compute_si_loss(self) -> torch.Tensor:
        """Compute Synaptic Intelligence regularization loss."""
        si_loss = torch.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance:
                delta = param - self.prev_params.get(name, param)
                si_loss += (self.importance[name] * delta.pow(2)).sum()
        
        return self.c * si_loss


class LwFRegularizer(nn.Module):
    """Learning without Forgetting using knowledge distillation."""
    
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Distillation loss weight
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.old_outputs: Dict[str, torch.Tensor] = {}
        
    def store_old_outputs(self, task_name: str, outputs: torch.Tensor):
        """Store outputs from old model for distillation."""
        self.old_outputs[task_name] = outputs.detach()
    
    def clear_old_outputs(self):
        """Clear stored old outputs."""
        self.old_outputs.clear()
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        task_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Compute LwF distillation loss.
        
        Args:
            student_logits: Current model logits
            teacher_logits: Old model logits (stored or provided)
            task_name: Optional task name for stored outputs
            
        Returns:
            Knowledge distillation loss
        """
        T = self.temperature
        
        # Apply temperature scaling
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        
        # Knowledge distillation loss
        kd_loss = self.kl_div(student_log_probs, teacher_probs) * (T ** 2)
        
        return self.alpha * kd_loss


def create_continual_learning_wrapper(trainer, config: ContinualLearningConfig):
    """
    Wrap existing trainer with continual learning capabilities.
    Returns modified trainer with CL methods integrated.
    """
    
    if config.strategy == "ewc":
        trainer.ewc_regularizer = EWCRegularizer(trainer.model, config.ewc)
        
        # Hook into training loop to add EWC loss
        original_compute_loss = trainer.compute_loss
        
        def compute_loss_with_ewc(model, inputs, return_outputs=False):
            loss = original_compute_loss(model, inputs, return_outputs)
            ewc_loss = trainer.ewc_regularizer.compute_ewc_loss()
            
            if return_outputs:
                return loss + ewc_loss, outputs
            return loss + ewc_loss
        
        trainer.compute_loss = compute_loss_with_ewc
        
    elif config.strategy == "replay":
        trainer.replay_buffer = ReplayBuffer(config.replay)
        
        # Modify data loading to include replay
        # Implementation depends on trainer's data loading mechanism
        
    elif config.strategy == "gem":
        trainer.gem_optimizer = GEMOptimizer(trainer.model, config.gem)
        
        # Hook into optimization step to project gradients
        # Implementation depends on trainer's optimization loop
        
    elif config.strategy == "lwf":
        trainer.lwf_loss = LwFLoss(config)
        # Store teacher model outputs for distillation
        # Implementation depends on training setup

    elif config.strategy == "si":
        trainer.si_regularizer = SIRegularizer(trainer.model, c=config.weight_decay)
        
        # Initialize trajectory tracking
        trainer.si_regularizer.initialize_trajectory()
        
        # Hook into training loop
        original_compute_loss = trainer.compute_loss
        
        def compute_loss_with_si(model, inputs, return_outputs=False):
            loss = original_compute_loss(model, inputs, return_outputs)
            si_loss = trainer.si_regularizer.compute_si_loss()
            
            if return_outputs:
                return loss + si_loss, outputs
            return loss + si_loss
        
        trainer.compute_loss = compute_loss_with_si
        
        # Hook into optimizer step to update trajectory
        original_step = trainer.optimizer.step if hasattr(trainer, 'optimizer') else None
        if original_step:
            def step_with_trajectory():
                original_step()
                trainer.si_regularizer.update_trajectory()
            trainer.optimizer.step = step_with_trajectory
    
    return trainer


class ContinualLearningWrapper:
    """
    High-level wrapper for applying continual learning methods.
    
    Provides a unified API for EWC, SI, and LwF regularization.
    
    Args:
        model: Model to wrap
        method: Continual learning method (ewc, si, lwf)
    """
    
    def __init__(self, model: nn.Module, method: str = "ewc"):
        self.model = model
        self.method = method
        self.ewc = None
        self.si = None
        self.lwf = None
        
        if method == "ewc":
            self.ewc = EWCRegularizer(model, EWCConfig())
        elif method == "si":
            self.si = SIRegularizer(model)
            self.si.initialize_trajectory()
        elif method == "lwf":
            self.lwf = LwFRegularizer()
    
    def apply_ewc_regularization(self, lambda_ewc: float = 0.5):
        """Apply Elastic Weight Consolidation regularization."""
        if self.ewc is None:
            self.ewc = EWCRegularizer(self.model, EWCConfig(ewc_lambda=lambda_ewc))
        else:
            self.ewc.config.ewc_lambda = lambda_ewc
        return self
    
    def apply_si_regularization(self, c: float = 0.1):
        """Apply Synaptic Intelligence regularization."""
        if self.si is None:
            self.si = SIRegularizer(self.model, c=c)
            self.si.initialize_trajectory()
        else:
            self.si.c = c
        return self
    
    def apply_lwf_regularization(self, alpha: float = 0.5):
        """Apply Learning without Forgetting regularization."""
        if self.lwf is None:
            self.lwf = LwFRegularizer(alpha=alpha)
        else:
            self.lwf.alpha = alpha
        return self
    
    def compute_fisher(self, dataloader: DataLoader, device: torch.device):
        """Compute Fisher information matrix for EWC."""
        if self.ewc:
            self.ewc.compute_fisher(dataloader, device)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get current regularization loss."""
        if self.ewc:
            return self.ewc.compute_ewc_loss()
        elif self.si:
            return self.si.compute_si_loss()
        return torch.tensor(0.0)
    
    def progressive_unfreeze(
        self,
        start_layers: int = 4,
        unfreeze_every_n_epochs: int = 2,
        max_layers: Optional[int] = None
    ):
        """
        Progressive unfreezing strategy for continual learning.
        
        Args:
            start_layers: Number of layers to keep unfrozen initially
            unfreeze_every_n_epochs: Epochs between unfreezing
            max_layers: Maximum layers to unfreeze (None = all)
        """
        self.start_layers = start_layers
        self.unfreeze_every_n_epochs = unfreeze_every_n_epochs
        self.max_layers = max_layers
        self.current_epoch = 0
        
        # Initially freeze all but top layers
        self._unfreeze_layers(start_layers)
    
    def _unfreeze_layers(self, num_layers: int):
        """Unfreeze top N layers of the model."""
        layers = list(self.model.modules())
        # Unfreeze from the end (top layers)
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def step_epoch(self):
        """Call at end of each epoch for progressive unfreezing."""
        if hasattr(self, 'unfreeze_every_n_epochs'):
            self.current_epoch += 1
            if self.current_epoch % self.unfreeze_every_n_epochs == 0:
                current_unfrozen = self.start_layers + (self.current_epoch // self.unfreeze_every_n_epochs) * 2
                if self.max_layers is None or current_unfrozen <= self.max_layers:
                    self._unfreeze_layers(current_unfrozen)
