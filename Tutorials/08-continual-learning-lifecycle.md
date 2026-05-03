# Tutorial 08: Continual Learning & Model Lifecycle Management

## Overview

This tutorial covers continual learning strategies for updating models with new data while preventing catastrophic forgetting, along with complete model lifecycle management including versioning, deployment, monitoring, and retirement.

## Table of Contents

1. [Continual Learning Fundamentals](#continual-learning-fundamentals)
2. [Catastrophic Forgetting](#catastrophic-forgetting)
3. [Replay-Based Methods](#replay-based-methods)
4. [Regularization-Based Methods](#regularization-based-methods)
5. [Architecture-Based Methods](#architecture-based-methods)
6. [Incremental Training Strategies](#incremental-training-strategies)
7. [Model Versioning](#model-versioning)
8. [Deployment Strategies](#deployment-strategies)
9. [Production Monitoring](#production-monitoring)
10. [Model Retirement and Archival](#model-retirement-and-archival)

---

## Continual Learning Fundamentals

### What is Continual Learning?

Continual learning (also called lifelong learning or incremental learning) enables models to:
- Learn continuously from new data over time
- Adapt to changing distributions and tasks
- Accumulate knowledge without forgetting previous capabilities
- Operate in non-stationary environments

### Key Challenges

1. **Catastrophic Forgetting**: Learning new information causes loss of old knowledge
2. **Stability-Plasticity Dilemma**: Balance between retaining old knowledge and learning new patterns
3. **Task Boundary Detection**: Knowing when a new task or distribution shift occurs
4. **Computational Efficiency**: Avoiding full retraining on all historical data
5. **Memory Constraints**: Storing representative examples without keeping everything

### Continual Learning Scenarios

```python
from enum import Enum

class ContinualLearningScenario(Enum):
    """Different continual learning scenarios."""
    
    TASK_INCREMENTAL = "task_incremental"
    # New tasks arrive sequentially with clear boundaries
    # Example: First train on sentiment analysis, then on QA
    
    DOMAIN_INCREMENTAL = "domain_incremental"  
    # Same task but domains change over time
    # Example: News articles from 2020, then 2021, then 2022
    
    CLASS_INCREMENTAL = "class_incremental"
    # New classes appear over time
    # Example: First classify cats/dogs, then add birds, then fish
    
    INSTANCE_INCREMENTAL = "instance_incremental"
    # Same task and classes, just more data arrives
    # Example: Continuous stream of customer support tickets
```

### Evaluation Metrics for Continual Learning

```python
import numpy as np

class ContinualLearningMetrics:
    def __init__(self):
        self.task_accuracies = {}  # {task_id: {timestamp: accuracy}}
        
    def record_accuracy(self, task_id, timestamp, accuracy):
        """Record accuracy for a task at a specific time."""
        if task_id not in self.task_accuracies:
            self.task_accuracies[task_id] = {}
        self.task_accuracies[task_id][timestamp] = accuracy
    
    def calculate_forward_transfer(self):
        """
        Forward Transfer: How much does learning task A help with task B?
        
        Positive values indicate beneficial transfer.
        """
        transfers = []
        
        task_ids = sorted(self.task_accuracies.keys())
        
        for i, task_b in enumerate(task_ids[1:], 1):
            # Get initial accuracy on task B before training
            # Compare to accuracy after training on previous tasks
            
            # Simplified: compare first evaluation to best later evaluation
            times_b = sorted(self.task_accuracies[task_b].keys())
            if len(times_b) > 1:
                initial_acc = self.task_accuracies[task_b][times_b[0]]
                best_acc = max(self.task_accuracies[task_b][t] for t in times_b)
                transfer = best_acc - initial_acc
                transfers.append(transfer)
        
        return np.mean(transfers) if transfers else 0.0
    
    def calculate_backward_transfer(self):
        """
        Backward Transfer: How does learning new tasks affect old tasks?
        
        Negative values indicate forgetting.
        """
        transfers = []
        
        task_ids = sorted(self.task_accuracies.keys())
        
        for i, task_a in enumerate(task_ids[:-1]):
            times_a = sorted(self.task_accuracies[task_a].keys())
            
            if len(times_a) > 1:
                # Accuracy immediately after training on task A
                initial_acc = self.task_accuracies[task_a][times_a[0]]
                
                # Accuracy after training on all subsequent tasks
                final_acc = self.task_accuracies[task_a][times_a[-1]]
                
                transfer = final_acc - initial_acc
                transfers.append(transfer)
        
        return np.mean(transfers) if transfers else 0.0
    
    def calculate_forgetting_measure(self):
        """
        Forgetting Measure: Maximum decrease in accuracy on any old task.
        """
        forgetting_scores = []
        
        for task_id, time_accuracies in self.task_accuracies.items():
            times = sorted(time_accuracies.keys())
            
            if len(times) > 1:
                max_acc = max(time_accuracies[t] for t in times)
                final_acc = time_accuracies[times[-1]]
                
                forgetting = max_acc - final_acc
                forgetting_scores.append(forgetting)
        
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    def calculate_average_accuracy(self, final_only=False):
        """
        Average Accuracy across all tasks.
        """
        all_final_accuracies = []
        
        for task_id, time_accuracies in self.task_accuracies.items():
            if final_only:
                # Only use final accuracy
                final_time = max(time_accuracies.keys())
                all_final_accuracies.append(time_accuracies[final_time])
            else:
                # Average across all evaluations
                all_final_accuracies.extend(time_accuracies.values())
        
        return np.mean(all_final_accuracies) if all_final_accuracies else 0.0
    
    def generate_report(self):
        """Generate comprehensive continual learning report."""
        report = {
            'average_accuracy': self.calculate_average_accuracy(final_only=True),
            'forward_transfer': self.calculate_forward_transfer(),
            'backward_transfer': self.calculate_backward_transfer(),
            'forgetting_measure': self.calculate_forgetting_measure()
        }
        
        print("Continual Learning Performance Report")
        print("=" * 50)
        print(f"Average Accuracy:     {report['average_accuracy']:.4f}")
        print(f"Forward Transfer:     {report['forward_transfer']:+.4f}")
        print(f"Backward Transfer:    {report['backward_transfer']:+.4f}")
        print(f"Forgetting Measure:   {report['forgetting_measure']:.4f}")
        print("=" * 50)
        
        if report['forgetting_measure'] < 0.05:
            print("✅ Minimal forgetting detected")
        elif report['forgetting_measure'] < 0.15:
            print("⚠️  Moderate forgetting - consider mitigation")
        else:
            print("❌ Severe forgetting - immediate action needed")
        
        return report
```

---

## Catastrophic Forgetting

### Understanding the Problem

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def demonstrate_catastrophic_forgetting(model, task1_data, task2_data):
    """
    Demonstrate catastrophic forgetting phenomenon.
    
    Train on Task 1, then Task 2, observe performance drop on Task 1.
    """
    metrics = {
        'task1_before': [],
        'task1_after_task2': [],
        'task2_after': []
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Train on Task 1
    print("Phase 1: Training on Task 1...")
    for epoch in range(10):
        model.train()
        for texts, labels in task1_data:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate on Task 1
    task1_acc = evaluate(model, task1_data)
    metrics['task1_before'].append(task1_acc)
    print(f"Task 1 Accuracy after Task 1 training: {task1_acc:.4f}")
    
    # Phase 2: Train on Task 2 (without seeing Task 1 data)
    print("\nPhase 2: Training on Task 2...")
    for epoch in range(10):
        model.train()
        for texts, labels in task2_data:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate on both tasks
    task1_acc_after = evaluate(model, task1_data)
    task2_acc = evaluate(model, task2_data)
    
    metrics['task1_after_task2'].append(task1_acc_after)
    metrics['task2_after'].append(task2_acc)
    
    print(f"\nTask 1 Accuracy after Task 2 training: {task1_acc_after:.4f}")
    print(f"Task 2 Accuracy: {task2_acc:.4f}")
    
    forgetting = task1_acc - task1_acc_after
    print(f"\n📉 FORGETTING: {forgetting:.4f} ({forgetting/task1_acc*100:.1f}% drop)")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(['Task 1\n(Before)', 'Task 1\n(After)', 'Task 2'], 
           [task1_acc, task1_acc_after, task2_acc],
           color=['green', 'red', 'blue'], alpha=0.7)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Demonstration of Catastrophic Forgetting')
    ax.set_ylim(0, 1)
    
    for i, v in enumerate([task1_acc, task1_acc_after, task2_acc]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_demo.png')
    plt.show()
    
    return metrics, forgetting
```

### Why Does Forgetting Happen?

1. **Weight Interference**: New task optimizes weights in directions that conflict with old task
2. **Representation Shift**: Hidden representations change to accommodate new patterns
3. **Decision Boundary Movement**: Classification boundaries shift away from old class regions
4. **Capacity Limits**: Model has finite capacity; new knowledge displaces old

---

## Replay-Based Methods

### Experience Replay

```python
import random
from collections import deque
import pickle

class ExperienceReplayBuffer:
    """
    Store and sample past experiences for replay during training.
    """
    
    def __init__(self, max_size=10000, strategy='uniform'):
        """
        Args:
            max_size: Maximum number of samples to store
            strategy: 'uniform', 'reservoir', 'class_balanced'
        """
        self.max_size = max_size
        self.strategy = strategy
        self.buffer = deque(maxlen=max_size)
        self.class_buffers = {}  # For class-balanced sampling
        
    def add(self, sample, label=None):
        """Add a sample to the replay buffer."""
        if self.strategy == 'class_balanced' and label is not None:
            if label not in self.class_buffers:
                self.class_buffers[label] = deque(maxlen=self.max_size // 10)
            self.class_buffers[label].append(sample)
        else:
            self.buffer.append((sample, label))
    
    def sample(self, batch_size):
        """Sample a batch from the replay buffer."""
        if self.strategy == 'class_balanced':
            # Sample equally from each class
            all_samples = []
            for label, class_buf in self.class_buffers.items():
                if len(class_buf) > 0:
                    n_samples = min(batch_size // len(self.class_buffers), len(class_buf))
                    all_samples.extend(random.sample(list(class_buf), n_samples))
            
            # Pad if necessary
            while len(all_samples) < batch_size and self.buffer:
                all_samples.append(random.choice(list(self.buffer)))
            
            return random.sample(all_samples, min(batch_size, len(all_samples)))
        
        elif self.strategy == 'reservoir':
            # Reservoir sampling already handled by add method
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        
        else:  # uniform
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path):
        """Save replay buffer to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'class_buffers': {k: list(v) for k, v in self.class_buffers.items()},
                'strategy': self.strategy
            }, f)
    
    def load(self, path):
        """Load replay buffer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data['buffer'], maxlen=self.max_size)
            self.class_buffers = {
                k: deque(v, maxlen=self.max_size // 10) 
                for k, v in data['class_buffers'].items()
            }
            self.strategy = data['strategy']


class ReplayBasedTrainer:
    """
    Trainer with experience replay for continual learning.
    """
    
    def __init__(self, model, replay_buffer, config):
        self.model = model
        self.replay_buffer = replay_buffer
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, current_batch, replay_batch_size=32):
        """
        Train on current data mixed with replayed examples.
        """
        self.model.train()
        
        # Get replay samples
        if len(self.replay_buffer) > 0:
            replay_samples = self.replay_buffer.sample(replay_batch_size)
            replay_texts = [s[0] for s in replay_samples]
            replay_labels = [s[1] for s in replay_samples]
            
            # Combine current and replay data
            combined_texts = current_batch['texts'] + replay_texts
            combined_labels = current_batch['labels'] + replay_labels
        else:
            combined_texts = current_batch['texts']
            combined_labels = current_batch['labels']
        
        # Train on combined batch
        self.optimizer.zero_grad()
        outputs = self.model(combined_texts)
        loss = self.criterion(outputs, combined_labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_on_task(self, task_data_loader, task_id, epochs=5):
        """Train on a new task while replaying old experiences."""
        print(f"\nTraining on Task {task_id} with replay...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in task_data_loader:
                loss = self.train_step(batch, replay_batch_size=32)
                total_loss += loss
                
                # Add current batch to replay buffer
                for text, label in zip(batch['texts'], batch['labels']):
                    self.replay_buffer.add((text, label), label=label)
            
            avg_loss = total_loss / len(task_data_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save replay buffer checkpoint
        self.replay_buffer.save(f'replay_buffer_task{task_id}.pkl')
```

### Generative Replay

```python
class GenerativeReplay:
    """
    Use a generative model to recreate past data instead of storing it.
    
    Benefits:
    - Privacy: Don't store actual past data
    - Compression: Generate many variations from compact model
    - Scalability: No memory limit on "stored" experiences
    """
    
    def __init__(self, generator_model, generator_config):
        self.generator = generator_model
        self.config = generator_config
    
    def train_generator(self, data_loader, epochs=10):
        """Train generator on current task data."""
        self.generator.train()
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Train generator to reconstruct input data
                reconstructed = self.generator(batch['texts'])
                loss = self.reconstruction_loss(batch['texts'], reconstructed)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Generator Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")
    
    def generate_replay_samples(self, n_samples=100):
        """Generate synthetic samples resembling past data."""
        self.generator.eval()
        
        generated_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from latent space
                latent = torch.randn(1, self.config.latent_dim)
                
                # Generate text/features
                generated = self.generator.decode(latent)
                generated_samples.append(generated)
        
        return generated_samples
    
    def reconstruction_loss(self, original, reconstructed):
        """Calculate reconstruction loss for generator training."""
        # Implementation depends on generator architecture
        # Could be MSE, cross-entropy, or perceptual loss
        return nn.MSELoss()(original, reconstructed)
```

### Dark Experience Replay

```python
class DarkExperienceReplay:
    """
    Store model outputs (logits) along with inputs for replay.
    
    Instead of storing (x, y), store (x, model_output_at_time_t).
    This preserves the model's learned behavior, not just labels.
    """
    
    def __init__(self, model, max_size=5000):
        self.model = model
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
    
    def collect_experience(self, texts, labels):
        """Collect experiences with model predictions."""
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model.get_logits(texts)
            probabilities = torch.softmax(logits, dim=-1)
        
        # Store input, true label, and model's predicted distribution
        for text, label, prob_dist in zip(texts, labels, probabilities):
            experience = {
                'text': text,
                'true_label': label,
                'predicted_distribution': prob_dist.cpu().numpy(),
                'timestamp': len(self.memory)
            }
            self.memory.append(experience)
    
    def replay_loss(self, current_logits, stored_experiences):
        """
        Calculate distillation loss to match old predictions.
        """
        total_loss = 0
        
        for exp in stored_experiences:
            # Get current prediction for stored input
            current_pred = current_logits[exp['text']]
            
            # Stored prediction (from old model)
            old_pred = torch.tensor(exp['predicted_distribution'])
            
            # KL divergence to match old predictions
            kl_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log_softmax(current_pred, dim=-1),
                old_pred
            )
            
            total_loss += kl_loss
        
        return total_loss / len(stored_experiences)
```

---

## Regularization-Based Methods

### Elastic Weight Consolidation (EWC)

```python
import torch.nn.functional as F

class EWC:
    """
    Elastic Weight Consolidation: Penalize changes to important weights.
    
    Key idea: Some weights are more important for previous tasks.
    Constrain important weights to stay close to their old values.
    """
    
    def __init__(self, model, fisher_diagonal=None):
        self.model = model
        self.fisher_diagonal = fisher_diagonal  # Importance weights
        self.optimal_weights = None  # Weights after previous task
    
    def estimate_fisher_information(self, data_loader, device='cpu'):
        """
        Estimate Fisher Information Matrix diagonal.
        
        Fisher Information measures how sensitive the loss is to each parameter.
        High Fisher = parameter is important, should not change much.
        """
        self.model.eval()
        
        # Initialize Fisher as zeros
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Accumulate squared gradients
        for batch in data_loader:
            self.model.zero_grad()
            
            # Get predictions
            outputs = self.model(batch['texts'])
            loss = F.cross_entropy(outputs, batch['labels'])
            
            # Compute gradients
            loss.backward()
            
            # Square and accumulate gradients (Fisher diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)
        
        # Average over samples
        n_samples = len(data_loader.dataset)
        for name in fisher:
            fisher[name] /= n_samples
        
        self.fisher_diagonal = fisher
        
        # Store optimal weights (current weights after training)
        self.optimal_weights = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        return fisher
    
    def ewc_loss(self, lambda_ewc=1000):
        """
        Calculate EWC regularization loss.
        
        L_ewc = Σ_i F_i * (θ_i - θ*_i)^2
        
        where:
        - F_i is Fisher Information for parameter i
        - θ_i is current parameter value
        - θ*_i is optimal parameter value from previous task
        """
        if self.fisher_diagonal is None or self.optimal_weights is None:
            return torch.tensor(0.0)
        
        ewc_loss = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_diagonal:
                # Squared distance from optimal weights, weighted by Fisher
                diff = param - self.optimal_weights[name]
                ewc_loss += (self.fisher_diagonal[name] * diff.pow(2)).sum()
        
        return lambda_ewc * ewc_loss
    
    def save_checkpoint(self, path):
        """Save EWC state (Fisher and optimal weights)."""
        torch.save({
            'fisher_diagonal': self.fisher_diagonal,
            'optimal_weights': self.optimal_weights
        }, path)
    
    def load_checkpoint(self, path):
        """Load EWC state."""
        checkpoint = torch.load(path)
        self.fisher_diagonal = checkpoint['fisher_diagonal']
        self.optimal_weights = checkpoint['optimal_weights']


class EWC_Trainer:
    """Trainer with EWC regularization for continual learning."""
    
    def __init__(self, model, config, lambda_ewc=1000):
        self.model = model
        self.config = config
        self.lambda_ewc = lambda_ewc
        self.ewc = EWC(model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def train_on_task(self, task_data_loader, task_id, epochs=5):
        """Train on new task with EWC regularization."""
        print(f"\nTraining Task {task_id} with EWC...")
        
        for epoch in range(epochs):
            total_loss = 0
            total_ewc_loss = 0
            
            for batch in task_data_loader:
                self.model.train()
                self.optimizer.zero_grad()
                
                # Standard task loss
                outputs = self.model(batch['texts'])
                task_loss = F.cross_entropy(outputs, batch['labels'])
                
                # EWC regularization loss
                ewc_loss = self.ewc.ewc_loss(self.lambda_ewc)
                
                # Total loss
                total = task_loss + ewc_loss
                
                total.backward()
                self.optimizer.step()
                
                total_loss += task_loss.item()
                total_ewc_loss += ewc_loss.item()
            
            avg_loss = total_loss / len(task_data_loader)
            avg_ewc = total_ewc_loss / len(task_data_loader)
            
            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Task Loss: {avg_loss:.4f}")
            print(f"  EWC Loss: {avg_ewc:.4f}")
        
        # After training, update Fisher and optimal weights
        print("Updating Fisher Information Matrix...")
        self.ewc.estimate_fisher_information(task_data_loader)
        self.ewc.save_checkpoint(f'ewc_checkpoint_task{task_id}.pth')
```

### Synaptic Intelligence (SI)

```python
class SynapticIntelligence:
    """
    Synaptic Intelligence: Track parameter importance during training.
    
    Unlike EWC which computes Fisher after training,
    SI accumulates importance online during training.
    """
    
    def __init__(self, model):
        self.model = model
        self.importance = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.previous_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.delta_loss = {name: torch.zeros_like(param) for name in self.importance}
    
    def update_importance(self, loss_change):
        """
        Update parameter importance based on contribution to loss decrease.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Change in parameter
                delta_param = param.detach() - self.previous_params[name]
                
                # Contribution to loss decrease (approximated)
                contribution = -param.grad * delta_param * loss_change
                
                # Accumulate importance
                self.importance[name] += contribution.abs()
    
    def si_loss(self, c_si=100):
        """
        Calculate SI regularization loss.
        
        L_si = Σ_j Ω_j * (θ_j - θ*_j)^2
        
        where Ω_j is accumulated importance for parameter j.
        """
        si_loss = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance:
                diff = param - self.previous_params[name]
                si_loss += (self.importance[name] * diff.pow(2)).sum()
        
        return c_si * si_loss
    
    def update_previous_params(self):
        """Store current parameters as previous after training step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.previous_params[name] = param.clone().detach()
```

### Learning without Forgetting (LwF)

```python
class LearningWithoutForgetting:
    """
    Learning without Forgetting: Use knowledge distillation from old model.
    
    Keep a copy of the old model and distill its knowledge
    while training on new data.
    """
    
    def __init__(self, model):
        self.model = model
        self.old_model = None  # Copy of model from previous task
    
    def create_old_model_copy(self):
        """Create a frozen copy of current model before training on new task."""
        import copy
        self.old_model = copy.deepcopy(self.model)
        
        # Freeze old model
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        self.old_model.eval()
    
    def distillation_loss(self, texts, temperature=2.0):
        """
        Calculate distillation loss to match old model's outputs.
        
        Uses softened probability distributions (with temperature)
        to capture dark knowledge.
        """
        if self.old_model is None:
            return torch.tensor(0.0)
        
        self.model.train()
        self.old_model.eval()
        
        with torch.no_grad():
            # Old model's soft predictions
            old_logits = self.old_model.get_logits(texts)
            old_soft_probs = F.softmax(old_logits / temperature, dim=-1)
        
        # Current model's predictions
        current_logits = self.model.get_logits(texts)
        current_soft_probs = F.log_softmax(current_logits / temperature, dim=-1)
        
        # KL divergence between distributions
        dist_loss = F.kl_div(
            current_soft_probs,
            old_soft_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return dist_loss
    
    def combined_loss(self, task_loss, dist_loss, alpha=0.5):
        """
        Combine task loss and distillation loss.
        
        L_total = α * L_task + (1-α) * L_distill
        """
        return alpha * task_loss + (1 - alpha) * dist_loss
```

---

## Architecture-Based Methods

### Progressive Neural Networks

```python
class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Networks: Add new columns for new tasks.
    
    Each task gets its own neural network column.
    Columns can read from previous columns but not modify them.
    """
    
    def __init__(self, base_column_class, config):
        super().__init__()
        self.base_column_class = base_column_class
        self.config = config
        self.columns = nn.ModuleList()
        self.task_to_column = {}
    
    def add_task_column(self, task_id):
        """Add a new column for a new task."""
        # Create new column
        new_column = self.base_column_class(self.config)
        
        # If there are previous columns, add lateral connections
        if len(self.columns) > 0:
            new_column.add_lateral_connections(self.columns)
        
        self.columns.append(new_column)
        self.task_to_column[task_id] = len(self.columns) - 1
        
        # Freeze all previous columns
        for col_idx in range(len(self.columns) - 1):
            for param in self.columns[col_idx].parameters():
                param.requires_grad = False
        
        print(f"Added column {len(self.columns) - 1} for task {task_id}")
    
    def forward(self, x, task_id):
        """Forward pass through appropriate column."""
        column_idx = self.task_to_column[task_id]
        return self.columns[column_idx](x)
    
    def get_total_parameters(self):
        """Count total parameters across all columns."""
        return sum(p.numel() for col in self.columns for p in col.parameters())
```

### Adapter Modules

```python
class AdapterModule(nn.Module):
    """
    Adapter modules: Small trainable modules inserted into frozen backbone.
    
    Keep pretrained model frozen, only train lightweight adapters.
    Different adapters for different tasks.
    """
    
    def __init__(self, hidden_dim, adapter_dim=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # Residual connection
        residual = x
        
        # Adapter bottleneck
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        
        # Add residual and normalize
        x = self.layer_norm(x + residual)
        
        return x


class AdapterConfig:
    """Configuration for adapter-based fine-tuning."""
    
    def __init__(self, 
                 adapter_dim=64,
                 adapter_locations='all_layers',
                 freeze_backbone=True,
                 task_adapters=True):
        self.adapter_dim = adapter_dim
        self.adapter_locations = adapter_locations  # 'all_layers', 'last_n', etc.
        self.freeze_backbone = freeze_backbone
        self.task_adapters = task_adapters  # Separate adapter per task


def insert_adapters_into_transformer(transformer_model, config):
    """Insert adapter modules into a transformer model."""
    
    if config.freeze_backbone:
        # Freeze all backbone parameters
        for param in transformer_model.parameters():
            param.requires_grad = False
    
    adapters = {}
    
    # Insert adapters into each layer
    for layer_idx, layer in enumerate(transformer_model.layers):
        if config.adapter_locations == 'all_layers' or \
           (config.adapter_locations.startswith('last_') and 
            layer_idx >= len(transformer_model.layers) - int(config.adapter_locations[5:])):
            
            # Create adapter
            adapter = AdapterModule(
                hidden_dim=layer.hidden_dim,
                adapter_dim=config.adapter_dim
            )
            
            # Insert after attention sublayer
            layer.insert_adapter(adapter)
            
            adapters[f'layer_{layer_idx}'] = adapter
    
    return adapters
```

### Dynamic Architecture Expansion

```python
class DynamicArchitectureExpansion:
    """
    Dynamically expand model capacity when needed.
    
    Monitor performance; if degradation detected, add capacity.
    """
    
    def __init__(self, model, expansion_threshold=0.05):
        self.model = model
        self.expansion_threshold = expansion_threshold
        self.baseline_performance = None
        self.expansion_history = []
    
    def monitor_and_expand(self, validation_data, current_task_id):
        """
        Check if model needs expansion based on performance drop.
        """
        current_performance = self.evaluate(self.model, validation_data)
        
        if self.baseline_performance is not None:
            performance_drop = self.baseline_performance - current_performance
            
            if performance_drop > self.expansion_threshold:
                print(f"Performance drop detected: {performance_drop:.4f}")
                print("Expanding model architecture...")
                
                self.expand_architecture(current_task_id)
                
                # Re-evaluate after expansion
                new_performance = self.evaluate(self.model, validation_data)
                print(f"Performance after expansion: {new_performance:.4f}")
        
        self.baseline_performance = current_performance
    
    def expand_architecture(self, task_id):
        """Add new capacity to the model."""
        # Strategy 1: Add new neurons to hidden layers
        # Strategy 2: Add new layers
        # Strategy 3: Add task-specific heads
        
        # Example: Add task-specific output head
        new_head = nn.Linear(self.model.hidden_dim, self.model.num_new_classes)
        setattr(self.model, f'task_{task_id}_head', new_head)
        
        self.expansion_history.append({
            'task_id': task_id,
            'expansion_type': 'new_head',
            'timestamp': len(self.expansion_history)
        })
```

---

## Incremental Training Strategies

### Scheduled Fine-Tuning

```python
class ScheduledFineTuner:
    """
    Schedule fine-tuning with decreasing learning rates and selective unfreezing.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.training_history = []
    
    def progressive_unfreezing(self, n_stages=3):
        """
        Progressively unfreeze layers from top to bottom.
        
        Stage 1: Only train top layers
        Stage 2: Train top + middle layers
        Stage 3: Train all layers
        """
        total_layers = len(self.model.layers)
        layers_per_stage = total_layers // n_stages
        
        schedule = []
        
        for stage in range(n_stages):
            # Number of layers to unfreeze
            n_unfrozen = (stage + 1) * layers_per_stage
            
            # Freeze/unfreeze accordingly
            for i, layer in enumerate(self.model.layers):
                if i < total_layers - n_unfrozen:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            # Learning rate decreases with deeper unfreezing
            lr = self.config.base_lr * (0.5 ** (n_stages - stage - 1))
            
            schedule.append({
                'stage': stage,
                'n_unfrozen': n_unfrozen,
                'learning_rate': lr
            })
        
        return schedule
    
    def train_with_schedule(self, data_loader, schedule):
        """Train using progressive unfreezing schedule."""
        for stage_config in schedule:
            print(f"\nStage {stage_config['stage'] + 1}: "
                  f"Unfreezing {stage_config['n_unfrozen']} layers")
            
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = stage_config['learning_rate']
            
            # Train for this stage
            self.train_epoch(data_loader)
            
            # Record history
            self.training_history.append(stage_config)
```

### Curriculum Learning for Continual Learning

```python
class CurriculumContinualLearner:
    """
    Apply curriculum learning principles to continual learning.
    
    Order tasks/examples from easy to hard to facilitate transfer.
    """
    
    def __init__(self, model):
        self.model = model
        self.task_difficulty = {}
    
    def estimate_task_difficulty(self, task_data):
        """
        Estimate difficulty of a task based on initial performance.
        """
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in task_data:
                outputs = self.model(batch['texts'])
                predictions = outputs.argmax(dim=-1)
                
                correct += (predictions == batch['labels']).sum().item()
                total += len(batch['labels'])
        
        initial_accuracy = correct / total
        
        # Difficulty inversely related to accuracy
        difficulty = 1.0 - initial_accuracy
        
        return difficulty
    
    def order_tasks_by_curriculum(self, tasks):
        """
        Order tasks from easiest to hardest.
        """
        # Estimate difficulty for each task
        for task_id, task_data in tasks.items():
            self.task_difficulty[task_id] = self.estimate_task_difficulty(task_data)
        
        # Sort by difficulty (easy to hard)
        ordered_tasks = sorted(
            tasks.items(),
            key=lambda x: self.task_difficulty[x[0]]
        )
        
        print("Task Curriculum (Easy → Hard):")
        for task_id, _ in ordered_tasks:
            print(f"  {task_id}: difficulty = {self.task_difficulty[task_id]:.3f}")
        
        return ordered_tasks
    
    def train_with_curriculum(self, ordered_tasks):
        """Train on tasks in curriculum order."""
        for task_id, task_data in ordered_tasks:
            print(f"\n{'='*50}")
            print(f"Training on task: {task_id}")
            print(f"Difficulty: {self.task_difficulty[task_id]:.3f}")
            print(f"{'='*50}")
            
            # Train on this task
            self.train_on_task(task_data, task_id)
```

---

## Model Versioning

### Semantic Versioning for Models

```python
from datetime import datetime
import json
import hashlib

class ModelVersion:
    """
    Semantic versioning for ML models.
    
    Format: MAJOR.MINOR.PATCH
    
    - MAJOR: Breaking changes (architecture change, incompatible API)
    - MINOR: New features, improved performance (backward compatible)
    - PATCH: Bug fixes, minor improvements (fully backward compatible)
    """
    
    def __init__(self, major=0, minor=0, patch=0, metadata=None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
    
    def bump_major(self):
        """Increment major version (breaking change)."""
        self.major += 1
        self.minor = 0
        self.patch = 0
    
    def bump_minor(self):
        """Increment minor version (new feature)."""
        self.minor += 1
        self.patch = 0
    
    def bump_patch(self):
        """Increment patch version (bug fix)."""
        self.patch += 1
    
    def __str__(self):
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.metadata:
            metadata_str = '+'.join(f"{k}={v}" for k, v in self.metadata.items())
            version_str += f"+{metadata_str}"
        
        return version_str
    
    def to_dict(self):
        return {
            'version': str(self),
            'major': self.major,
            'minor': self.minor,
            'patch': self.patch,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_string(cls, version_str):
        """Parse version string to ModelVersion object."""
        # Simple parsing (can be extended for metadata)
        parts = version_str.split('+')[0].split('.')
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )


class ModelRegistry:
    """
    Centralized registry for model versions and artifacts.
    """
    
    def __init__(self, registry_path='./model_registry'):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models = {}  # {model_name: {version: metadata}}
        self.load_registry()
    
    def register_model(self, model_name, version, model_path, metrics, metadata=None):
        """Register a new model version."""
        if model_name not in self.models:
            self.models[model_name] = {}
        
        # Calculate model hash for integrity
        model_hash = self.calculate_file_hash(model_path)
        
        # Create metadata
        model_metadata = {
            'version': str(version),
            'model_path': str(model_path),
            'model_hash': model_hash,
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'status': 'active'  # active, deprecated, archived
        }
        
        self.models[model_name][str(version)] = model_metadata
        
        # Save registry
        self.save_registry()
        
        print(f"Registered {model_name} v{version}")
        print(f"  Path: {model_path}")
        print(f"  Hash: {model_hash[:16]}...")
        print(f"  Metrics: {metrics}")
        
        return model_metadata
    
    def get_latest_version(self, model_name):
        """Get the latest active version of a model."""
        if model_name not in self.models:
            return None
        
        active_versions = [
            v for v, m in self.models[model_name].items()
            if m['status'] == 'active'
        ]
        
        if not active_versions:
            return None
        
        # Sort by version number
        latest = max(active_versions, key=lambda v: ModelVersion.from_string(v))
        return latest
    
    def deprecate_version(self, model_name, version_str):
        """Mark a model version as deprecated."""
        if model_name in self.models and version_str in self.models[model_name]:
            self.models[model_name][version_str]['status'] = 'deprecated'
            self.models[model_name][version_str]['deprecated_at'] = datetime.now().isoformat()
            self.save_registry()
            print(f"Deprecated {model_name} v{version_str}")
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of model file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_path / 'registry.json'
        with open(registry_file, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_path / 'registry.json'
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                self.models = json.load(f)
    
    def list_models(self):
        """List all registered models and versions."""
        print("Model Registry")
        print("=" * 70)
        
        for model_name, versions in self.models.items():
            print(f"\n{model_name}:")
            for version_str, metadata in sorted(versions.items()):
                status = metadata['status']
                metrics = metadata.get('metrics', {})
                accuracy = metrics.get('accuracy', 'N/A')
                
                print(f"  v{version_str:15} [{status:10}] Acc: {accuracy}")
        
        print("=" * 70)
```

---

## Deployment Strategies

### Canary Deployments

```python
class CanaryDeployment:
    """
    Gradually roll out new model version to subset of traffic.
    """
    
    def __init__(self, old_model, new_model, initial_percentage=5):
        self.old_model = old_model
        self.new_model = new_model
        self.canary_percentage = initial_percentage
        self.deployment_log = []
    
    def route_request(self, request):
        """Route request to old or new model based on canary percentage."""
        if random.random() < self.canary_percentage / 100:
            model = self.new_model
            variant = 'canary'
        else:
            model = self.old_model
            variant = 'stable'
        
        prediction = model.predict(request)
        
        # Log for monitoring
        self.deployment_log.append({
            'timestamp': datetime.now().isoformat(),
            'variant': variant,
            'request_id': request.get('id'),
            'prediction': prediction
        })
        
        return prediction
    
    def increase_canary(self, increment=10):
        """Increase canary traffic percentage."""
        self.canary_percentage = min(100, self.canary_percentage + increment)
        print(f"Canary traffic increased to {self.canary_percentage}%")
    
    def rollback(self):
        """Rollback to 100% old model."""
        self.canary_percentage = 0
        print("Rolled back to stable model")
    
    def analyze_canary_performance(self, ground_truth):
        """Compare performance of canary vs stable."""
        canary_correct = 0
        canary_total = 0
        stable_correct = 0
        stable_total = 0
        
        for log_entry in self.deployment_log:
            # Match with ground truth (simplified)
            is_correct = check_prediction(log_entry, ground_truth)
            
            if log_entry['variant'] == 'canary':
                canary_correct += is_correct
                canary_total += 1
            else:
                stable_correct += is_correct
                stable_total += 1
        
        canary_acc = canary_correct / canary_total if canary_total > 0 else 0
        stable_acc = stable_correct / stable_total if stable_total > 0 else 0
        
        print(f"Canary Accuracy: {canary_acc:.4f} (n={canary_total})")
        print(f"Stable Accuracy: {stable_acc:.4f} (n={stable_total})")
        
        improvement = canary_acc - stable_acc
        
        if improvement > 0.02:  # 2% improvement threshold
            print("✅ Canary performing better - consider increasing traffic")
            return 'promote'
        elif improvement < -0.02:
            print("❌ Canary performing worse - consider rollback")
            return 'rollback'
        else:
            print("⚠️  Similar performance - continue monitoring")
            return 'monitor'
```

### Blue-Green Deployment

```python
class BlueGreenDeployment:
    """
    Maintain two identical production environments.
    
    - Blue: Currently serving all traffic
    - Green: Idle environment with new model
    
    Switch traffic instantly when ready.
    """
    
    def __init__(self):
        self.active_environment = 'blue'
        self.environments = {
            'blue': {'model': None, 'status': 'inactive'},
            'green': {'model': None, 'status': 'inactive'}
        }
    
    def deploy_to_inactive(self, new_model):
        """Deploy new model to inactive environment."""
        inactive_env = 'green' if self.active_environment == 'blue' else 'blue'
        
        self.environments[inactive_env]['model'] = new_model
        self.environments[inactive_env]['status'] = 'ready'
        
        print(f"Deployed new model to {inactive_env} environment")
    
    def switch_traffic(self):
        """Switch all traffic to the other environment."""
        old_active = self.active_environment
        self.active_environment = 'green' if self.active_environment == 'blue' else 'blue'
        
        self.environments[old_active]['status'] = 'inactive'
        self.environments[self.active_environment]['status'] = 'active'
        
        print(f"Traffic switched from {old_active} to {self.active_environment}")
    
    def predict(self, request):
        """Route request to active environment."""
        model = self.environments[self.active_environment]['model']
        return model.predict(request)
    
    def rollback(self):
        """Quick rollback by switching environments."""
        self.switch_traffic()
        print("Rolled back to previous environment")
```

### Shadow Mode Deployment

```python
class ShadowModeDeployment:
    """
    Run new model in shadow mode alongside production.
    
    New model receives all requests but doesn't serve predictions.
    Used for validation without risk.
    """
    
    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.shadow_predictions = []
    
    def predict(self, request):
        """Serve from production, record shadow predictions."""
        # Production prediction (served to user)
        production_pred = self.production_model.predict(request)
        
        # Shadow prediction (recorded only)
        shadow_pred = self.shadow_model.predict(request)
        
        # Store for analysis
        self.shadow_predictions.append({
            'request': request,
            'production': production_pred,
            'shadow': shadow_pred,
            'timestamp': datetime.now().isoformat()
        })
        
        return production_pred
    
    def analyze_discrepancies(self, ground_truth=None):
        """Analyze differences between production and shadow."""
        discrepancies = 0
        total = len(self.shadow_predictions)
        
        for entry in self.shadow_predictions:
            if entry['production'] != entry['shadow']:
                discrepancies += 1
        
        discrepancy_rate = discrepancies / total if total > 0 else 0
        
        print(f"Shadow Mode Analysis:")
        print(f"  Total requests: {total}")
        print(f"  Discrepancies: {discrepancies} ({discrepancy_rate:.2%})")
        
        if ground_truth:
            # Evaluate which model performed better
            prod_correct = sum(
                1 for e, gt in zip(self.shadow_predictions, ground_truth)
                if e['production'] == gt
            )
            shadow_correct = sum(
                1 for e, gt in zip(self.shadow_predictions, ground_truth)
                if e['shadow'] == gt
            )
            
            print(f"  Production accuracy: {prod_correct/total:.4f}")
            print(f"  Shadow accuracy: {shadow_correct/total:.4f}")
        
        return discrepancy_rate
```

---

## Production Monitoring

### Real-Time Performance Monitoring

```python
import time
from collections import defaultdict, deque

class ProductionMonitor:
    """
    Monitor model performance in production.
    """
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        
        # Metrics windows
        self.latency_window = deque(maxlen=window_size)
        self.throughput_window = deque(maxlen=window_size)
        self.prediction_distribution = defaultdict(int)
        self.confidence_window = deque(maxlen=window_size)
        
        # Alerts
        self.alerts = []
        self.alert_thresholds = {
            'latency_p99': 1000,  # ms
            'throughput_min': 10,  # requests/sec
            'confidence_low': 0.5
        }
    
    def record_prediction(self, prediction, confidence, latency_ms):
        """Record a prediction event."""
        timestamp = time.time()
        
        # Record metrics
        self.latency_window.append((timestamp, latency_ms))
        self.prediction_distribution[prediction] += 1
        self.confidence_window.append((timestamp, confidence))
        self.throughput_window.append(timestamp)
        
        # Check for alerts
        self.check_alerts()
    
    def check_alerts(self):
        """Check if any metrics exceed thresholds."""
        current_time = time.time()
        
        # P99 Latency
        latencies = [l for _, l in self.latency_window]
        if latencies:
            p99_latency = np.percentile(latencies, 99)
            if p99_latency > self.alert_thresholds['latency_p99']:
                self.create_alert('HIGH_LATENCY', f"P99 latency: {p99_latency:.0f}ms")
        
        # Throughput
        recent_throughput = sum(
            1 for t in self.throughput_window
            if current_time - t < 1.0  # Last second
        )
        if recent_throughput < self.alert_thresholds['throughput_min']:
            self.create_alert('LOW_THROUGHPUT', f"Throughput: {recent_throughput} req/s")
        
        # Low confidence
        confidences = [c for _, c in self.confidence_window]
        if confidences:
            low_conf_ratio = np.mean([c < self.alert_thresholds['confidence_low'] for c in confidences])
            if low_conf_ratio > 0.2:  # More than 20% low confidence
                self.create_alert('LOW_CONFIDENCE', f"Low confidence ratio: {low_conf_ratio:.2%}")
    
    def create_alert(self, alert_type, message):
        """Create an alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts.append(alert)
        print(f"🚨 ALERT [{alert_type}]: {message}")
    
    def get_dashboard_metrics(self):
        """Get current metrics for dashboard."""
        current_time = time.time()
        
        # Latency stats
        latencies = [l for _, l in self.latency_window]
        latency_stats = {
            'mean': np.mean(latencies) if latencies else 0,
            'p50': np.percentile(latencies, 50) if latencies else 0,
            'p95': np.percentile(latencies, 95) if latencies else 0,
            'p99': np.percentile(latencies, 99) if latencies else 0
        }
        
        # Throughput
        throughput = sum(1 for t in self.throughput_window if current_time - t < 1.0)
        
        # Confidence stats
        confidences = [c for _, c in self.confidence_window]
        confidence_stats = {
            'mean': np.mean(confidences) if confidences else 0,
            'std': np.std(confidences) if confidences else 0
        }
        
        # Prediction distribution
        total_preds = sum(self.prediction_distribution.values())
        pred_distribution = {
            k: v / total_preds if total_preds > 0 else 0
            for k, v in self.prediction_distribution.items()
        }
        
        return {
            'latency': latency_stats,
            'throughput': throughput,
            'confidence': confidence_stats,
            'prediction_distribution': pred_distribution,
            'active_alerts': len([a for a in self.alerts if a['timestamp'] > str(current_time - 3600)])
        }
```

### Drift Detection in Production

```python
class ProductionDriftDetector:
    """
    Detect data drift and concept drift in production.
    """
    
    def __init__(self, reference_data, model, detection_method='ks_test'):
        self.reference_data = reference_data
        self.model = model
        self.detection_method = detection_method
        
        # Reference statistics
        self.reference_stats = self.compute_reference_statistics()
        
        # Production data window
        self.production_window = deque(maxlen=1000)
        
        # Drift alerts
        self.drift_alerts = []
    
    def compute_reference_statistics(self):
        """Compute statistics from reference (training) data."""
        stats = {}
        
        # Feature statistics
        features = self.extract_features(self.reference_data)
        stats['feature_means'] = np.mean(features, axis=0)
        stats['feature_stds'] = np.std(features, axis=0)
        
        # Prediction distribution
        reference_preds = self.model.predict(self.reference_data)
        stats['prediction_distribution'] = np.bincount(reference_preds) / len(reference_preds)
        
        # Confidence distribution
        reference_confs = self.model.get_confidences(self.reference_data)
        stats['confidence_mean'] = np.mean(reference_confs)
        stats['confidence_std'] = np.std(reference_confs)
        
        return stats
    
    def extract_features(self, data):
        """Extract features from data for drift detection."""
        # Implementation depends on data type
        # Could be raw features, embeddings, etc.
        pass
    
    def detect_feature_drift(self, new_data):
        """Detect drift in input features."""
        new_features = self.extract_features(new_data)
        
        if self.detection_method == 'ks_test':
            # Kolmogorov-Smirnov test for each feature
            drift_scores = []
            p_values = []
            
            for i in range(new_features.shape[1]):
                stat, p_val = stats.ks_2samp(
                    self.reference_data[:, i],
                    new_features[:, i]
                )
                drift_scores.append(stat)
                p_values.append(p_val)
            
            # Significant drift if many features have p < 0.05
            significant_drift = np.mean([p < 0.05 for p in p_values])
            
            return {
                'drift_detected': significant_drift > 0.3,  # 30% features drifted
                'drift_scores': drift_scores,
                'p_values': p_values,
                'fraction_drifted': significant_drift
            }
        
        elif self.detection_method == 'population_stability_index':
            # PSI for categorical features
            pass
    
    def detect_prediction_drift(self, new_predictions):
        """Detect drift in prediction distribution."""
        new_distribution = np.bincount(new_predictions) / len(new_predictions)
        
        # Ensure same length
        max_len = max(len(self.reference_stats['prediction_distribution']), 
                     len(new_distribution))
        
        ref_dist = np.zeros(max_len)
        new_dist = np.zeros(max_len)
        
        ref_dist[:len(self.reference_stats['prediction_distribution'])] = \
            self.reference_stats['prediction_distribution']
        new_dist[:len(new_distribution)] = new_distribution
        
        # KL divergence
        epsilon = 1e-10
        kl_div = np.sum(ref_dist * np.log((ref_dist + epsilon) / (new_dist + epsilon)))
        
        # Jensen-Shannon divergence (symmetric)
        js_div = 0.5 * kl_div + 0.5 * np.sum(new_dist * np.log((new_dist + epsilon) / (ref_dist + epsilon)))
        
        drift_detected = js_div > 0.1  # Threshold
        
        return {
            'drift_detected': drift_detected,
            'js_divergence': js_div,
            'reference_distribution': ref_dist,
            'new_distribution': new_dist
        }
    
    def monitor(self, new_data, new_predictions):
        """Run all drift detection methods."""
        results = {}
        
        # Feature drift
        results['feature_drift'] = self.detect_feature_drift(new_data)
        
        # Prediction drift
        results['prediction_drift'] = self.detect_prediction_drift(new_predictions)
        
        # Overall drift decision
        overall_drift = (
            results['feature_drift']['drift_detected'] or
            results['prediction_drift']['drift_detected']
        )
        
        if overall_drift:
            self.create_drift_alert(results)
        
        return {
            'drift_detected': overall_drift,
            'details': results
        }
    
    def create_drift_alert(self, drift_results):
        """Create drift alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'feature_drift': drift_results['feature_drift'],
            'prediction_drift': drift_results['prediction_drift']
        }
        self.drift_alerts.append(alert)
        
        print("🚨 DRIFT DETECTED!")
        print(f"  Feature drift: {drift_results['feature_drift']['fraction_drifted']:.2%}")
        print(f"  Prediction drift (JS): {drift_results['prediction_drift']['js_divergence']:.4f}")
```

---

## Model Retirement and Archival

### Model Deprecation Process

```python
class ModelDeprecationManager:
    """
    Manage the deprecation and retirement lifecycle of models.
    """
    
    def __init__(self, model_registry):
        self.registry = model_registry
        self.deprecation_schedule = {}
    
    def initiate_deprecation(self, model_name, version, reason, timeline_days=90):
        """
        Begin the deprecation process for a model version.
        """
        deprecation_date = datetime.now()
        retirement_date = deprecation_date + timedelta(days=timeline_days)
        
        self.deprecation_schedule[f"{model_name}:{version}"] = {
            'model_name': model_name,
            'version': version,
            'reason': reason,
            'deprecation_date': deprecation_date.isoformat(),
            'retirement_date': retirement_date.isoformat(),
            'status': 'deprecated',
            'replacement': None,
            'migration_guide': None
        }
        
        # Update registry
        self.registry.deprecate_version(model_name, version)
        
        print(f"Initiated deprecation for {model_name} v{version}")
        print(f"  Reason: {reason}")
        print(f"  Deprecation date: {deprecation_date.strftime('%Y-%m-%d')}")
        print(f"  Retirement date: {retirement_date.strftime('%Y-%m-%d')}")
        
        return self.deprecation_schedule[f"{model_name}:{version}"]
    
    def set_replacement(self, model_name, old_version, new_model_name, new_version):
        """Specify replacement model for deprecated version."""
        key = f"{model_name}:{old_version}"
        
        if key in self.deprecation_schedule:
            self.deprecation_schedule[key]['replacement'] = {
                'model_name': new_model_name,
                'version': new_version
            }
            
            # Generate migration guide
            self.generate_migration_guide(model_name, old_version, new_model_name, new_version)
    
    def generate_migration_guide(self, old_model, old_version, new_model, new_version):
        """Generate migration guide for users."""
        guide = f"""
# Migration Guide: {old_model} v{old_version} → {new_model} v{new_version}

## Timeline
- Deprecation Date: {self.deprecation_schedule[f'{old_model}:{old_version}']['deprecation_date']}
- Retirement Date: {self.deprecation_schedule[f'{old_model}:{old_version}']['retirement_date']}

## Breaking Changes
[List breaking changes here]

## API Differences
[Document API changes]

## Performance Improvements
[Document improvements]

## Migration Steps
1. Update model reference in configuration
2. Test with new model in staging environment
3. Validate outputs match expectations
4. Deploy to production with canary rollout
5. Monitor for issues

## Support
Contact: ml-platform@company.com
        """
        
        self.deprecation_schedule[f"{old_model}:{old_version}"]["migration_guide"] = guide
        
        # Save guide
        guide_path = f"migration_guides/{old_model}_{old_version}_to_{new_model}_{new_version}.md"
        with open(guide_path, 'w') as f:
            f.write(guide)
    
    def retire_model(self, model_name, version):
        """
        Fully retire a model version (after deprecation period).
        """
        key = f"{model_name}:{version}"
        
        if key not in self.deprecation_schedule:
            print(f"No deprecation record found for {model_name}:{version}")
            return False
        
        deprecation_info = self.deprecation_schedule[key]
        retirement_date = datetime.fromisoformat(deprecation_info['retirement_date'])
        
        if datetime.now() < retirement_date:
            print(f"Cannot retire before {retirement_date.strftime('%Y-%m-%d')}")
            return False
        
        # Archive model
        self.archive_model(model_name, version)
        
        # Update status
        deprecation_info['status'] = 'retired'
        deprecation_info['retired_at'] = datetime.now().isoformat()
        
        print(f"Retired {model_name} v{version}")
        
        return True
    
    def archive_model(self, model_name, version):
        """Move model to cold storage."""
        # Get model path from registry
        model_metadata = self.registry.models[model_name][version]
        model_path = Path(model_metadata['model_path'])
        
        # Create archive directory
        archive_dir = Path('./model_archives') / model_name / version
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model files
        import shutil
        shutil.move(str(model_path), str(archive_dir / model_path.name))
        
        # Save metadata
        metadata_path = archive_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Compress archive
        shutil.make_archive(str(archive_dir), 'gztar', archive_dir)
        
        print(f"Archived {model_name} v{version} to {archive_dir}")
    
    def get_deprecation_status(self, model_name, version=None):
        """Get deprecation status for a model."""
        if version:
            key = f"{model_name}:{version}"
            return self.deprecation_schedule.get(key, None)
        else:
            # Return all versions
            return {
                k: v for k, v in self.deprecation_schedule.items()
                if v['model_name'] == model_name
            }
```

### Model Lineage Tracking

```python
class ModelLineageTracker:
    """
    Track complete lineage of models from training to retirement.
    """
    
    def __init__(self):
        self.lineage_graph = {}  # {model_id: lineage_info}
    
    def record_training_run(self, model_id, training_config, data_version, 
                           code_version, hyperparameters):
        """Record details of a training run."""
        self.lineage_graph[model_id] = {
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
            'training': {
                'config': training_config,
                'data_version': data_version,
                'code_version': code_version,
                'hyperparameters': hyperparameters,
                'environment': self.capture_environment()
            },
            'parent_models': [],  # For fine-tuned models
            'child_models': [],   # Models derived from this one
            'evaluation_results': {},
            'deployment_history': [],
            'retirement_info': None
        }
    
    def record_fine_tuning(self, child_model_id, parent_model_id, 
                          fine_tuning_data, fine_tuning_config):
        """Record fine-tuning relationship."""
        if parent_model_id in self.lineage_graph:
            # Add child to parent
            self.lineage_graph[parent_model_id]['child_models'].append(child_model_id)
            
            # Create child record
            self.lineage_graph[child_model_id] = {
                'model_id': child_model_id,
                'created_at': datetime.now().isoformat(),
                'parent_models': [parent_model_id],
                'fine_tuning': {
                    'data': fine_tuning_data,
                    'config': fine_tuning_config
                },
                'child_models': [],
                'evaluation_results': {},
                'deployment_history': [],
                'retirement_info': None
            }
    
    def record_evaluation(self, model_id, dataset_name, metrics):
        """Record evaluation results."""
        if model_id in self.lineage_graph:
            self.lineage_graph[model_id]['evaluation_results'][dataset_name] = {
                'metrics': metrics,
                'recorded_at': datetime.now().isoformat()
            }
    
    def record_deployment(self, model_id, environment, deployment_config):
        """Record deployment event."""
        if model_id in self.lineage_graph:
            self.lineage_graph[model_id]['deployment_history'].append({
                'environment': environment,
                'config': deployment_config,
                'deployed_at': datetime.now().isoformat()
            })
    
    def capture_environment(self):
        """Capture training environment details."""
        import sys
        import platform
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'packages': self.get_installed_packages()
        }
    
    def get_installed_packages(self):
        """Get list of installed packages and versions."""
        import pkg_resources
        return {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
        }
    
    def get_model_lineage(self, model_id):
        """Get complete lineage for a model."""
        if model_id not in self.lineage_graph:
            return None
        
        lineage = self.lineage_graph[model_id].copy()
        
        # Recursively get parent lineages
        if lineage['parent_models']:
            lineage['parent_lineages'] = [
                self.get_model_lineage(parent_id)
                for parent_id in lineage['parent_models']
            ]
        
        return lineage
    
    def visualize_lineage(self, model_id):
        """Visualize model lineage as a graph."""
        try:
            import graphviz
        except ImportError:
            print("Install graphviz: pip install graphviz")
            return
        
        dot = graphviz.Digraph(comment='Model Lineage')
        
        def add_node(mid):
            if mid not in self.lineage_graph:
                return
            
            info = self.lineage_graph[mid]
            label = f"{mid}\\n{info['created_at'][:10]}"
            
            if info['retirement_info']:
                dot.node(mid, label, style='filled', fillcolor='lightgray')
            else:
                dot.node(mid, label)
            
            # Add edges to parents
            for parent_id in info['parent_models']:
                dot.edge(parent_id, mid)
                add_node(parent_id)
        
        add_node(model_id)
        
        # Render graph
        dot.render('model_lineage.gv', view=True)
```

---

## Best Practices Checklist

### Continual Learning Best Practices

- [ ] **Choose Right Strategy**: Select replay, regularization, or architecture based on constraints
- [ ] **Monitor Forgetting**: Track backward transfer and forgetting metrics
- [ ] **Balance Stability-Plasticity**: Tune hyperparameters for optimal balance
- [ ] **Validate Frequently**: Evaluate on all tasks after each new task
- [ ] **Document Task Boundaries**: Clearly define when new tasks begin
- [ ] **Plan Capacity**: Ensure model has enough capacity for expected tasks
- [ ] **Test Transfer**: Measure forward transfer between tasks

### Model Lifecycle Best Practices

- [ ] **Version Everything**: Models, data, code, configurations
- [ ] **Automate Deployment**: Use CI/CD pipelines for model deployment
- [ ] **Monitor Continuously**: Track performance, latency, drift in production
- [ ] **Plan Deprecation**: Have clear retirement criteria and processes
- [ ] **Maintain Lineage**: Track complete history from training to retirement
- [ ] **Document Decisions**: Record why models were created, changed, retired
- [ ] **Security**: Control access to model artifacts and endpoints

---

## Next Steps

In the next tutorial, we'll cover:
- **Complete Production Pipeline**: End-to-end example from training to serving
- **Scaling Strategies**: Distributed training and inference at scale
- **Cost Optimization**: Reducing training and inference costs
- **Team Collaboration**: MLOps workflows for teams
- **Case Studies**: Real-world examples and lessons learned
