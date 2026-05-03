# Tutorial 014: Continual Learning - Avoiding Catastrophic Forgetting

## 📌 Overview

**Level**: Advanced  
**Duration**: 60 minutes  
**Prerequisites**: Tutorials 001-013 completed

Learn strategies to continuously update models without forgetting previously learned knowledge.

---

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Understand catastrophic forgetting problem
- Implement Elastic Weight Consolidation (EWC)
- Use experience replay buffers
- Apply Gradient Episodic Memory (GEM)
- Deploy continual learning in production

---

## 1. The Catastrophic Forgetting Problem

### What is Catastrophic Forgetting?

When a neural network learns a new task, it often **completely forgets** how to perform previous tasks.

```
Task 1 Training:          Task 2 Training:
┌──────────────┐          ┌──────────────┐
│ Learn English│          │ Learn French │
│ Accuracy: 95%│    →     │ Accuracy: 92%│
└──────────────┘          └──────────────┘
                                 ↓
                        Test on English: 35% ❌
                        (Forgot everything!)
```

### Why It Happens

- Neural networks overwrite weights during training
- New task optimization conflicts with old task optima
- No mechanism to preserve important weights

---

## 2. Continual Learning Strategies

### Overview of Methods

| Method | Approach | Memory | Complexity | Best For |
|--------|----------|--------|------------|----------|
| **EWC** | Regularization | Low | Medium | Sequential tasks |
| **Replay** | Data buffering | High | Low | Streaming data |
| **GEM** | Gradient constraints | Medium | High | Multiple tasks |
| **LwF** | Knowledge distillation | Low | Medium | Task-free |

### Choose Your Strategy

```python
def choose_cl_strategy(scenario):
    """Recommend continual learning strategy."""
    
    if scenario == "sequential_tasks":
        return "ewc"  # Clear task boundaries
    elif scenario == "streaming_data":
        return "replay"  # Continuous data flow
    elif scenario == "multiple_domains":
        return "gem"  # Many related tasks
    else:
        return "lwf"  # General purpose
```

---

## 3. Elastic Weight Consolidation (EWC)

### How EWC Works

EWC identifies **important weights** for previous tasks and penalizes changes to them.

```
Loss = Loss_new + λ × Σ F_i × (θ_i - θ*_i)²
       │         │    │   │
       │         │    │   └─ Optimal params from old task
       │         │    └─ Fisher information (importance)
       │         └─ EWC strength
       └─ Standard training loss
```

### Implementation

```python
from utils.continual_learning import EWCConfig, EWCRegularizer

# Configure EWC
ewc_config = EWCConfig(
    ewc_lambda=1000.0,      # Strength of regularization
    fisher_samples=200,      # Samples to estimate importance
    damping=0.1,             # Numerical stability
)

# After training on Task 1
model = train_task_1(model, task1_data)

# Compute Fisher Information Matrix
regularizer = EWCRegularizer(model, ewc_config)
regularizer.compute_fisher(task1_dataloader, device)

# Train on Task 2 with EWC
for batch in task2_dataloader:
    # Standard loss
    loss = compute_loss(model, batch)
    
    # Add EWC regularization
    ewc_loss = regularizer.compute_ewc_loss()
    
    # Total loss
    total_loss = loss + ewc_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

### Tuning EWC Lambda

```python
# Experiment with different lambda values
lambda_study = {
    "λ=100": {"task1_retention": "95%", "task2_learning": "slow"},
    "λ=500": {"task1_retention": "92%", "task2_learning": "good"},
    "λ=1000": {"task1_retention": "88%", "task2_learning": "fast"},
    "λ=5000": {"task1_retention": "70%", "task2_learning": "very fast"},
}

# Recommendation: Start with λ=1000, adjust based on retention needs
```

---

## 4. Experience Replay

### How Replay Works

Store examples from previous tasks and **replay them** during new task training.

```
Task 1 Data:              Replay Buffer:
┌──────────────┐          ┌──────────────┐
│ [All samples]│   →      │ [1000 random]│
└──────────────┘          └──────────────┘
                                 ↓
                        Mix with Task 2 data during training
```

### Implementation

```python
from utils.continual_learning import ReplayBuffer, ReplayConfig

# Configure replay
replay_config = ReplayConfig(
    replay_size=1000,           # Buffer capacity
    replay_ratio=0.5,           # 50% of each batch from replay
    selection_strategy="uniform", # Random sampling
    reservoir_sampling=True,    # For streaming data
)

# Initialize buffer
buffer = ReplayBuffer(replay_config)

# After Task 1, store samples
buffer.add(task1_samples, task_id=1)

# During Task 2 training
for batch in task2_dataloader:
    # Get replay samples
    mixed_batch = buffer.get_batch(batch)
    
    # Train on mixed data
    loss = compute_loss(model, mixed_batch)
    loss.backward()
    optimizer.step()

# Add Task 2 samples to buffer
buffer.add(task2_samples, task_id=2)
```

### Selection Strategies

```python
# Uniform: Random sampling
uniform_buffer = ReplayBuffer(ReplayConfig(
    selection_strategy="uniform"
))

# Recent: Prefer recent samples
recent_buffer = ReplayBuffer(ReplayConfig(
    selection_strategy="recent"
))

# Diverse: Maximize variety
diverse_buffer = ReplayBuffer(ReplayConfig(
    selection_strategy="diverse"
))
```

---

## 5. Gradient Episodic Memory (GEM)

### How GEM Works

GEM stores gradients from previous tasks and **projects** new gradients to avoid interference.

```
New Gradient: ∇L_new
     ↓
Check against stored gradients: ∇L_old
     ↓
Project if conflicting: ∇L_projected
     ↓
Update weights with projected gradient
```

### Implementation

```python
from utils.continual_learning import GEMOptimizer, GEMConfig

# Configure GEM
gem_config = GEMConfig(
    memory_size=100,          # Examples per task
    num_tasks=5,              # Expected number of tasks
    use_quadprog=True,        # Use quadratic programming
)

# Initialize optimizer
gem_optimizer = GEMOptimizer(model, gem_config)

# After each task, store gradients
gem_optimizer.update_gradient_memory(
    task_id=1,
    dataloader=task1_dataloader,
    device=device,
)

# During training on new task
for batch in task2_dataloader:
    # Compute standard gradient
    loss = compute_loss(model, batch)
    loss.backward()
    
    # Get constraints from previous tasks
    constraints = gem_optimizer.compute_gradient_constraints(
        task_id=2,
        device=device,
    )
    
    # Project gradient
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = gem_optimizer.project_gradient(
                param.grad,
                constraints,
            )
    
    optimizer.step()
```

---

## 6. Learning without Forgetting (LwF)

### How LwF Works

Use **knowledge distillation** to preserve old model's outputs while learning new tasks.

```
Old Model (Frozen)     New Model (Training)
     │                       │
     │  Output: p_old        │  Output: p_new
     │       │               │
     └───────┴───────────────┘
             ↓
     Distillation Loss: KL(p_old || p_new)
```

### Implementation

```python
from utils.continual_learning import LwFLoss, ContinualLearningConfig

# Configure LwF
cl_config = ContinualLearningConfig(
    strategy="lwf",
    lwf_alpha=1.0,            # Distillation weight
    lwf_temperature=2.0,      # Soften probabilities
)

# Store old model
old_model = copy.deepcopy(model)
old_model.eval()

# Create LwF loss
lwf_loss = LwFLoss(cl_config)

# Train on new task
for batch in new_task_dataloader:
    # Get old model outputs (no gradients)
    with torch.no_grad():
        old_outputs = old_model(batch["input_ids"])
    
    # New model forward
    new_outputs = model(batch["input_ids"])
    
    # Standard task loss
    task_loss = compute_task_loss(new_outputs, batch["labels"])
    
    # Distillation loss
    distill_loss = lwf_loss(
        new_outputs.logits,
        old_outputs.logits,
    )
    
    # Combined loss
    total_loss = task_loss + cl_config.lwf_alpha * distill_loss
    
    total_loss.backward()
    optimizer.step()
```

---

## 7. Unified Continual Learning Wrapper

### Integrate with Trainer

```python
from utils.continual_learning import (
    ContinualLearningConfig,
    create_continual_learning_wrapper,
)

# Choose strategy
cl_config = ContinualLearningConfig(
    strategy="ewc",  # or "replay", "gem", "lwf"
    ewc=EWCConfig(ewc_lambda=1000),
    replay=ReplayConfig(replay_size=1000),
    gem=GEMConfig(memory_size=100),
)

# Wrap existing trainer
trainer = Trainer(model, training_config, dataset)
cl_trainer = create_continual_learning_wrapper(trainer, cl_config)

# Train normally - CL methods applied automatically
cl_trainer.train()
```

---

## 8. Multi-Task Continual Learning

### Scenario: Learning Multiple Domains

```python
# Sequence of tasks
tasks = [
    {"name": "news", "data": news_dataset},
    {"name": "social", "data": social_dataset},
    {"name": "scientific", "data": science_dataset},
    {"name": "conversational", "data": chat_dataset},
]

# Initialize
model = initialize_model()
buffer = ReplayBuffer(ReplayConfig(replay_size=2000))
ewc_regularizer = None

# Sequential training
for i, task in enumerate(tasks):
    print(f"\nTraining on {task['name']}...")
    
    # Train on current task
    for epoch in range(3):
        for batch in task["data"]:
            # Mix with replay
            if len(buffer.buffer) > 0:
                batch = buffer.get_batch(batch)
            
            # Compute loss
            loss = compute_loss(model, batch)
            
            # Add EWC if not first task
            if ewc_regularizer and i > 0:
                loss += ewc_regularizer.compute_ewc_loss()
            
            loss.backward()
            optimizer.step()
    
    # Update replay buffer
    buffer.add(task["data"].samples, task_id=i)
    
    # Compute Fisher for next task
    if i < len(tasks) - 1:
        ewc_regularizer = EWCRegularizer(model, EWCConfig())
        ewc_regularizer.compute_fisher(task["data"], device)
    
    # Evaluate on all previous tasks
    for j in range(i + 1):
        acc = evaluate(model, tasks[j]["data"])
        print(f"  Task {tasks[j]['name']} accuracy: {acc:.2%}")
```

---

## 9. Evaluating Continual Learning

### Metrics to Track

```python
class CLMetrics:
    def __init__(self):
        self.task_accuracies = {}
        self.forgetting_scores = {}
    
    def compute_forgetting(self, task_id, initial_acc, current_acc):
        """Measure how much was forgotten."""
        return initial_acc - current_acc
    
    def compute_backward_transfer(self, task_id, acc_alone, acc_with_cl):
        """Measure if later tasks help earlier ones."""
        return acc_with_cl - acc_alone
    
    def compute_forward_transfer(self, task_id, acc_from_scratch, acc_with_cl):
        """Measure if earlier tasks help later ones."""
        return acc_with_cl - acc_from_scratch

# Usage
metrics = CLMetrics()

# After training all tasks
for task_id in range(num_tasks):
    # Measure forgetting
    forgetting = metrics.compute_forgetting(
        task_id,
        initial_accuracy[task_id],
        final_accuracy[task_id],
    )
    print(f"Task {task_id} forgetting: {forgetting:.2%}")
```

### Benchmark Results

```
Method      | Task 1 | Task 2 | Task 3 | Avg Retention
------------|--------|--------|--------|---------------
Fine-tuning | 95%    | 92%    | 90%    | 45% (bad!)
EWC         | 92%    | 90%    | 88%    | 85%
Replay      | 93%    | 91%    | 89%    | 88%
GEM         | 91%    | 90%    | 89%    | 87%
Joint Train | 94%    | 93%    | 92%    | 93% (oracle)
```

---

## 10. Production Deployment

### Continual Learning Pipeline

```python
class ProductionCL:
    def __init__(self, model, strategy="replay"):
        self.model = model
        self.strategy = strategy
        self.task_count = 0
        
        if strategy == "replay":
            self.buffer = ReplayBuffer(ReplayConfig(
                replay_size=5000,
                replay_ratio=0.3,
            ))
    
    def adapt_to_new_domain(self, new_data, epochs=1):
        """Adapt model to new domain while preserving old knowledge."""
        
        self.model.train()
        
        for epoch in range(epochs):
            for batch in new_data:
                # Mix with historical data
                if hasattr(self, 'buffer'):
                    batch = self.buffer.get_batch(batch)
                
                # Standard training step
                loss = compute_loss(self.model, batch)
                loss.backward()
                optimizer.step()
        
        # Store samples from new domain
        if hasattr(self, 'buffer'):
            self.buffer.add(new_data.samples, task_id=self.task_count)
        
        self.task_count += 1
        
        # Evaluate on validation sets
        return self.evaluate_all_domains()
    
    def evaluate_all_domains(self):
        """Check performance across all learned domains."""
        results = {}
        for domain in self.validation_sets:
            acc = evaluate(self.model, self.validation_sets[domain])
            results[domain] = acc
        
        # Alert if any domain drops below threshold
        for domain, acc in results.items():
            if acc < 0.80:  # 80% threshold
                print(f"⚠️  ALERT: {domain} performance dropped to {acc:.2%}")
        
        return results
```

---

## 11. Best Practices

### When to Use Continual Learning

✅ **Use CL when**:
- Data arrives sequentially over time
- Cannot store all historical data
- Need to adapt to changing distributions
- Multiple related tasks over time

❌ **Don't use CL when**:
- All data available upfront (use joint training)
- Tasks are completely unrelated
- Can afford to retrain from scratch
- Performance requirements are extremely high

### Hyperparameter Guidelines

```python
# EWC
ewc_lambda = 1000      # Start here, tune ±
fisher_samples = 200   # More = better estimate, slower

# Replay
replay_size = 1000     # Larger = better retention, more memory
replay_ratio = 0.5     # Balance old/new learning

# GEM
memory_size = 100      # Per task
num_tasks = 5          # Plan ahead

# LwF
lwf_alpha = 1.0        # Distillation strength
temperature = 2.0      # Softer = more knowledge transfer
```

---

## 12. Real-World Example

### Customer Support Bot Evolution

```python
# Scenario: Chatbot learning from customer interactions

cl_system = ProductionCL(model, strategy="replay")

# Month 1: General inquiries
general_data = load_customer_data("2024-01")
cl_system.adapt_to_new_domain(general_data)

# Month 2: Product-specific questions
product_data = load_customer_data("2024-02")
cl_system.adapt_to_new_domain(product_data)

# Month 3: Technical support
tech_data = load_customer_data("2024-03")
cl_system.adapt_to_new_domain(tech_data)

# Month 4: Billing inquiries
billing_data = load_customer_data("2024-04")
cl_system.adapt_to_new_domain(billing_data)

# Check all domains still work
performance = cl_system.evaluate_all_domains()
print("Domain Performance:")
for domain, acc in performance.items():
    print(f"  {domain}: {acc:.2%}")
```

---

## 📚 Summary

### Key Takeaways

✅ **Catastrophic forgetting** is a major challenge in sequential learning  
✅ **EWC** protects important weights with regularization  
✅ **Replay** maintains performance by mixing old and new data  
✅ **GEM** prevents gradient interference between tasks  
✅ **LwF** uses distillation to preserve knowledge  

### Quick Reference

```python
# Standard CL workflow
config = ContinualLearningConfig(strategy="ewc")
trainer = create_continual_learning_wrapper(trainer, config)
trainer.train(new_task_data)
evaluate_all_previous_tasks()
```

---

**Congratulations!** You've mastered continual learning!

➡️ **Next**: [Tutorial 015: Scaling to Production](./015_scaling_production.md)
