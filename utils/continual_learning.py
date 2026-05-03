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
    
    return trainer
