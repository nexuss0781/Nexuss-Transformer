# Tutorial 004: First Training Run - Your First Model Training

## 📌 Overview

**Level**: Intermediate  
**Duration**: 60-90 minutes  
**Prerequisites**: Tutorials 001-003 completed

In this tutorial, you'll run your first complete model training from start to finish.

---

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Set up a complete training pipeline
- Configure training hyperparameters
- Monitor training progress
- Save and load checkpoints
- Evaluate your trained model

---

## 1. Complete Training Pipeline Setup

### Step 1: Import Required Modules

```python
import torch
from torch.utils.data import DataLoader
from models.config import NTFConfig
from models.transformer import NexussTransformer
from training.config import TrainingConfig
from training.trainer import Trainer
from training.data import TextDataset, DataCollatorForLanguageModeling
```

### Step 2: Create Small Dataset for Testing

```python
# Create sample training data
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand text.",
    "Transformers have revolutionized deep learning for sequences.",
    "Attention mechanisms allow models to focus on relevant parts.",
] * 100  # Repeat for more samples

# Save to file
with open("sample_data.txt", "w") as f:
    for text in sample_texts:
        f.write(text + "\n")

print(f"Created dataset with {len(sample_texts)} samples")
```

### Step 3: Initialize Model and Configuration

```python
# Use small config for quick training
config = NTFConfig(
    vocab_size=1000,  # Character-level vocab
    d_model=128,
    n_heads=4,
    n_layers=4,
    max_seq_len=64,
    d_ff=512,
)

model = NexussTransformer(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 2. Training Configuration

### Understanding TrainingConfig

```python
training_config = TrainingConfig(
    # Output settings
    output_dir="./outputs/first_run",
    num_train_epochs=5,
    
    # Batch size settings
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    
    # Optimization
    learning_rate=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    scheduler="linear",
    
    # Checkpointing
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    
    # Performance
    mixed_precision="fp32",  # Change to "bf16" if supported
    gradient_checkpointing=False,
    
    # Reproducibility
    seed=42,
)

print("Training configuration created")
print(f"Effective batch size: {training_config.effective_batch_size}")
```

### Key Hyperparameters Explained

| Parameter | Recommended Value | Impact |
|-----------|------------------|--------|
| **learning_rate** | 1e-3 (small), 1e-4 (large) | Too high = unstable, too low = slow |
| **batch_size** | 8-32 (small GPU), 64+ (large) | Larger = more stable, more memory |
| **warmup_ratio** | 0.05-0.15 | Prevents early instability |
| **weight_decay** | 0.01-0.1 | Regularization, prevents overfitting |
| **num_train_epochs** | 3-10 (small data), 1-3 (large) | More = risk of overfitting |

---

## 3. Prepare Data Loaders

### Create Dataset and Collator

```python
# Load dataset
train_dataset = TextDataset(
    file_path="sample_data.txt",
    max_length=64,
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    pad_token_id=0,
    max_length=64,
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator,
)

print(f"Dataset size: {len(train_dataset)}")
print(f"Batches per epoch: {len(train_loader)}")
```

---

## 4. Initialize Trainer

### Create Trainer Instance

```python
trainer = Trainer(
    model=model,
    config=training_config,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("Trainer initialized successfully")
```

---

## 5. Run Training

### Start Training Loop

```python
print("Starting training...")
print("=" * 50)

# Train the model
training_metrics = trainer.train()

print("=" * 50)
print("Training completed!")
print(f"Final training loss: {training_metrics['train_loss']:.4f}")
print(f"Total steps: {training_metrics['global_step']}")
print(f"Training time: {training_metrics['training_time_seconds']:.1f} seconds")
```

### Expected Output

```
Starting training...
==================================================
Training:  23%|██▎       | 115/500 [00:45<02:34,  2.49it/s, loss=2.15, lr=8.5e-4]
Training:  46%|████▋     | 230/500 [01:32<01:48,  2.49it/s, loss=1.87, lr=6.2e-4]
Training:  69%|██████▉   | 345/500 [02:18<01:02,  2.49it/s, loss=1.54, lr=3.8e-4]
Training:  92%|█████████▏| 460/500 [03:04<00:16,  2.49it/s, loss=1.32, lr=1.5e-4]
Training: 100%|██████████| 500/500 [03:20<00:00,  2.49it/s, loss=1.28, lr=8.7e-5]
==================================================
Training completed!
Final training loss: 1.4523
Total steps: 500
Training time: 200.5 seconds
```

---

## 6. Monitor Training Progress

### Understanding Training Logs

```python
# Access training history
logs = trainer.logs_history

# Print sample logs
for log in logs[::50]:  # Every 50th log
    print(f"Step {log['train/global_step']}: Loss={log['train/loss']:.4f}, LR={log['train/learning_rate']:.2e}")
```

### Visualizing Training (Optional)

```python
import matplotlib.pyplot as plt

# Extract loss values
steps = [log['train/global_step'] for log in logs]
losses = [log['train/loss'] for log in logs]
lrs = [log['train/learning_rate'] for log in logs]

# Plot training loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()
```

---

## 7. Checkpoint Management

### Loading Saved Checkpoints

```python
from training.checkpoint import CheckpointManager

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(
    output_dir="./outputs/first_run",
    save_total_limit=3,
)

# List available checkpoints
checkpoints = checkpoint_manager.list_checkpoints()
print("Available checkpoints:")
for ckpt in checkpoints:
    print(f"  - {ckpt}")

# Get latest checkpoint
latest = checkpoint_manager.get_latest_checkpoint()
print(f"\nLatest checkpoint: {latest}")
```

### Resuming Training from Checkpoint

```python
# Resume training
if latest:
    print(f"Resuming from {latest}")
    
    # Load checkpoint
    checkpoint_data = checkpoint_manager.load_checkpoint(latest)
    
    # Load model state
    model.load_state_dict(checkpoint_data['model_state'])
    
    # Resume training
    training_metrics = trainer.train(resume_from_checkpoint=latest)
```

---

## 8. Evaluate Trained Model

### Compute Perplexity

```python
from utils.metrics import compute_perplexity, compute_accuracy

# Create evaluation loader (same as training, no shuffle)
eval_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
)

# Compute perplexity
model.eval()
perplexity = compute_perplexity(model, eval_loader, device)
accuracy = compute_accuracy(model, eval_loader, device)

print(f"Evaluation Results:")
print(f"  Perplexity: {perplexity:.2f}")
print(f"  Token Accuracy: {accuracy:.2%}")
```

### Understanding Metrics

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| **Perplexity** | <10 (excellent), <50 (good) | Lower is better; measures prediction uncertainty |
| **Accuracy** | >50% (decent), >70% (good) | Higher is better; % of correct next-token predictions |

---

## 9. Generate Text with Your Model

### Simple Text Generation

```python
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0):
    """Generate text given a prompt."""
    model.eval()
    
    # Tokenize prompt
    input_ids = torch.tensor(tokenizer._tokenize(prompt)).unsqueeze(0).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if max length reached
            if generated.size(1) >= max_length:
                break
    
    # Decode
    tokens = generated.squeeze().tolist()
    text = ''.join([tokenizer.idx_to_char.get(t, '?') for t in tokens])
    
    return text

# Generate sample text
prompt = "The quick"
generated = generate_text(model, train_dataset.tokenizer, prompt, max_length=100)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
```

---

## 10. Troubleshooting Common Issues

### Issue 1: Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Solutions**:
```python
# Try lower learning rate
training_config.learning_rate = 1e-4

# Increase warmup
training_config.warmup_ratio = 0.15

# Check for NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
```

### Issue 2: Out of Memory

**Solutions**:
```python
# Reduce batch size
training_config.per_device_train_batch_size = 4

# Enable gradient checkpointing
training_config.gradient_checkpointing = True

# Use gradient accumulation
training_config.gradient_accumulation_steps = 4
training_config.per_device_train_batch_size = 2
```

### Issue 3: Slow Training

**Solutions**:
```python
# Use mixed precision
training_config.mixed_precision = "bf16"  # or "fp16"

# Increase batch size if memory allows
training_config.per_device_train_batch_size = 32

# Reduce logging frequency
training_config.logging_steps = 50
```

---

## 11. Best Practices Summary

### Before Training
- ✅ Calculate parameter count and memory requirements
- ✅ Start with small model for debugging
- ✅ Prepare validation set
- ✅ Set random seed for reproducibility

### During Training
- ✅ Monitor loss curves regularly
- ✅ Check for NaN/Inf values
- ✅ Save checkpoints frequently
- ✅ Track learning rate schedule

### After Training
- ✅ Evaluate on held-out test set
- ✅ Test generation quality
- ✅ Compare against baselines
- ✅ Document hyperparameters

---

## 12. Hands-On Exercise

### Exercise: Train Different Model Sizes

```python
# TODO: Train three models and compare results
configs = [
    NTFConfig(d_model=64, n_layers=2, n_heads=2),   # Tiny
    NTFConfig(d_model=128, n_layers=4, n_heads=4),  # Small
    NTFConfig(d_model=256, n_layers=6, n_heads=8),  # Medium
]

results = []
for i, config in enumerate(configs):
    print(f"\nTraining model {i+1}/3...")
    # TODO: Implement training loop
    # Record final loss and training time
```

### Exercise: Hyperparameter Sweep

```python
# TODO: Experiment with different learning rates
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]

for lr in learning_rates:
    training_config.learning_rate = lr
    # TODO: Train and record final loss
```

---

## 📚 Summary

### What You Accomplished

✅ Configured training hyperparameters  
✅ Ran complete training loop  
✅ Monitored training progress  
✅ Saved and loaded checkpoints  
✅ Evaluated model performance  
✅ Generated text with your model  

### Quick Reference

```python
# Complete training workflow
config = NTFConfig.small()
model = NexussTransformer(config)
train_config = TrainingConfig()
trainer = Trainer(model, train_config, dataset)
metrics = trainer.train()
```

---

**Congratulations!** You've successfully trained your first LLM!

➡️ **Next**: [Tutorial 005: Training Optimization](./005_training_optimization.md)
