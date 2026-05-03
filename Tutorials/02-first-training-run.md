# Tutorial 02: Your First Training Run

## Introduction

Congratulations! You now understand blank slate models. In this tutorial, you'll run your **first actual training** and see your model learn from data.

By the end of this tutorial, you will:
- Set up a complete training pipeline
- Configure optimization hyperparameters
- Monitor training progress in real-time
- Save and load checkpoints
- Debug common training issues

---

## The Training Pipeline Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Dataset   │ →   │   Trainer   │ →   │ Checkpoints │
│             │     │             │     │             │
│ • Load data │     │ • Forward   │     │ • Save      │
│ • Tokenize  │     │ • Loss      │     │ • Resume    │
│ • Batching  │     │ • Backward  │     │ • Load      │
└─────────────┘     │ • Optimize  │     └─────────────┘
                    └─────────────┘
                          ↓
                    ┌─────────────┐
                    │   Metrics   │
                    │             │
                    │ • Loss      │
                    │ • Perplexity│
                    │ • LR        │
                    └─────────────┘
```

---

## Step 1: Prepare Your Training Data

### Data Format

Our framework expects JSONL format (one JSON per line):

```jsonl
{"text": "This is the first training example. It should be representative of your domain."}
{"text": "The second example continues the pattern. More data means better learning."}
{"text": "Each line is treated as a separate document for training purposes."}
```

### Create Sample Data

Let's create a small dataset for testing:

```python
# save as: data/sample_train.jsonl
sample_data = [
    {"text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter."},
    {"text": "Machine learning is a subset of artificial intelligence that enables systems to learn."},
    {"text": "Natural language processing helps computers understand human language."},
    {"text": "Deep learning uses neural networks with many layers to process information."},
    {"text": "Transformers have revolutionized NLP with self-attention mechanisms."},
] * 100  # Repeat to get enough data

with open("data/sample_train.jsonl", "w") as f:
    for item in sample_data:
        import json
        f.write(json.dumps(item) + "\n")

print(f"Created dataset with {len(sample_data)} examples")
```

---

## Step 2: Configure Training

### Understanding TrainingConfig

```python
from training.config import TrainingConfig, Precision, OptimizerType, SchedulerType

config = TrainingConfig(
    # Output
    output_dir="./outputs/first-training",
    
    # Training duration
    num_train_epochs=3,           # Number of passes through dataset
    max_steps=-1,                 # -1 means use epochs (or set specific steps)
    
    # Batch sizes
    per_device_train_batch_size=4,   # Samples per GPU per step
    per_device_eval_batch_size=8,    # Samples per GPU for eval
    gradient_accumulation_steps=2,   # Accumulate gradients before update
    
    # Effective batch size = 4 * 2 = 8 samples per update
    
    # Learning rate
    learning_rate=5e-4,
    weight_decay=0.01,
    
    # Optimizer
    optimizer=OptimizerType.ADAMW,
    
    # Scheduler
    scheduler=SchedulerType.LINEAR,
    warmup_ratio=0.05,          # 5% of steps for warmup
    
    # Precision
    mixed_precision=Precision.FP32,  # Use FP16 or BF16 for faster training
    
    # Checkpointing
    save_steps=100,
    save_total_limit=3,         # Keep only 3 most recent checkpoints
    
    # Logging
    logging_steps=10,
    eval_steps=100,
    
    # Advanced
    gradient_checkpointing=False,  # Enable for large models
    max_grad_norm=1.0,            # Gradient clipping
    seed=42,                      # For reproducibility
)
```

### Pre-built Configurations

For convenience, use pre-built configs:

```python
# Small model (~60M params)
config = TrainingConfig.small_model()
# Sets: lr=1e-3, batch=32, accum=2, warmup=5%, epochs=10

# Medium model (~350M params)
config = TrainingConfig.medium_model()
# Sets: lr=5e-4, batch=16, accum=4, warmup=3%, epochs=5

# Large model (1B+ params)
config = TrainingConfig.large_model()
# Sets: lr=1e-4, batch=4, accum=8, warmup=2%, epochs=3, checkpointing=True, BF16
```

---

## Step 3: Create the Model

```python
from models.transformer import NexussTransformer
from models.config import NTFConfig

# Define architecture
model_config = NTFConfig(
    vocab_size=16000,      # Match your tokenizer
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072,
    max_seq_len=2048,
    use_rope=True,
    activation="swiglu",
)

# Create blank slate model
model = NexussTransformer(model_config)

# Print parameter count
param_info = model.count_parameters()
print(f"Model created: {param_info['total_millions']} parameters")
```

---

## Step 4: Prepare Dataset

### Simple Dataset Class

```python
import torch
from torch.utils.data import Dataset
import json

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['text'])
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone(),  # Same as input for LM
        }

# Usage
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")
# Or use a custom tokenizer

train_dataset = TextDataset(
    file_path="data/sample_train.jsonl",
    tokenizer=tokenizer,
    max_length=256,
)
```

### Data Collator

```python
def data_collator(batch):
    """Collate batch items into tensors."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
```

---

## Step 5: Initialize Trainer

```python
from training.trainer import Trainer

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=None,  # Add validation set if available
    data_collator=data_collator,
)

print("Trainer initialized!")
print(f"Training samples: {len(train_dataset)}")
print(f"Effective batch size: {config.effective_batch_size}")
```

---

## Step 6: Start Training!

```python
# Begin training
print("Starting training...")
metrics = trainer.train()

print("\n=== Training Complete ===")
print(f"Final loss: {metrics['train_loss']:.4f}")
print(f"Steps trained: {metrics['global_step']}")
print(f"Epochs: {metrics['epochs_trained']}")
print(f"Training time: {metrics['training_time_seconds']:.1f}s")
print(f"Samples/sec: {metrics['samples_per_second']:.2f}")
```

### Expected Output

```
Starting training...
Training:   0%|          | 0/1000 [00:00<?, ?it/s]
Step 10: loss=2.5432, lr=5.00e-05
Step 20: loss=2.3421, lr=1.00e-04
Step 30: loss=2.1234, lr=1.50e-04
...
Saved checkpoint to outputs/first-training/checkpoints/checkpoint-000100_20240101_120000

=== Training Complete ===
Final loss: 1.2345
Steps trained: 1000
Epochs: 3
Training time: 3600.5s
Samples/sec: 125.34
```

---

## Understanding the Training Loop

### What Happens Each Step?

```python
# Simplified view of what Trainer.train() does:

for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass
        outputs = model(**batch)
        
        # 2. Compute loss
        loss = outputs.loss  # Cross-entropy for language modeling
        
        # 3. Backward pass (compute gradients)
        loss.backward()
        
        # 4. Update weights (after accumulation steps)
        if step % gradient_accumulation_steps == 0:
            optimizer.step()      # Update weights
            scheduler.step()      # Update learning rate
            optimizer.zero_grad() # Clear gradients
        
        # 5. Log metrics
        if step % logging_steps == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
        
        # 6. Save checkpoint
        if step % save_steps == 0:
            save_checkpoint()
```

### Learning Rate Schedule

```
LR
↑
│         ╱───────╲
│       ╱           ╲
│     ╱               ╲
│   ╱                   ╲
│ ╱                       ╲
└──────────────────────────→ Steps
  Warmup    Decay    End
  
Warmup: Gradually increase LR (prevents early instability)
Decay: Gradually decrease LR (fine-tune near optimum)
```

---

## Monitoring Training

### Key Metrics to Watch

#### 1. Training Loss

```
Good Training:          Problematic Training:
Loss ↓                  Loss ↓
2.5 ─┐                  2.5 ─┐
     │ ╲                     │ ╲
2.0 ─┤  ╲                   2.0 ─┤  ╲____
     │   ╲                        │       ╲___
1.5 ─┤    ╲_____              1.5 ─┤           \
     │          ╲__                │            \__
1.0 ─┤             ╲__           1.0 ─┤               \__
     └─────────────────            └─────────────────
        Steps                         Steps
     (Converging)                (Oscillating/Not converging)
```

**What to expect**:
- Initial loss: 2.5-4.0 (random guessing)
- After warmup: Steady decrease
- Final loss: 1.0-2.0 (depends on task complexity)

#### 2. Learning Rate

```python
# Plot LR over time
import matplotlib.pyplot as plt

plt.plot(trainer.logs_history)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.show()
```

#### 3. Gradient Norm

High gradient norms → unstable training
Low gradient norms → vanishing gradients

```python
# Add to training loop
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm: {grad_norm:.4f}")
```

---

## Checkpoint Management

### Automatic Checkpointing

Checkpoints are saved automatically every `save_steps`:

```
outputs/first-training/
├── checkpoints/
│   ├── checkpoint-000100_20240101_120000/
│   │   ├── model.pt           # Model weights
│   │   ├── training_state.pt  # Optimizer, scheduler
│   │   └── metadata.json      # Step, epoch, config
│   ├── checkpoint-000200_20240101_130000/
│   └── checkpoint-000300_20240101_140000/
└── best_model/
    ├── model.pt
    ├── training_state.pt
    └── metadata.json
```

### Manual Checkpoint Saving

```python
# Save current state
trainer._save_checkpoint()

# Save best model (based on eval loss)
eval_metrics = trainer.evaluate()
trainer._save_best_checkpoint(eval_metrics)
```

### Resuming from Checkpoint

```python
# Find latest checkpoint
latest = trainer.checkpoint_manager.get_latest_checkpoint()
print(f"Resuming from: {latest}")

# Resume training
metrics = trainer.train(resume_from_checkpoint=latest)
```

### Loading Checkpoint Manually

```python
checkpoint_data = trainer.checkpoint_manager.load_checkpoint(checkpoint_path)

# Apply to model
model.load_state_dict(checkpoint_data['model_state'])
optimizer.load_state_dict(checkpoint_data['optimizer_state'])
scheduler.load_state_dict(checkpoint_data['scheduler_state'])

print(f"Resumed from step {checkpoint_data['global_step']}")
```

---

## Debugging Common Issues

### Issue 1: Loss is NaN

**Symptoms**: Loss becomes `nan` or `inf`

**Causes & Solutions**:

```python
# 1. Learning rate too high
config.learning_rate = config.learning_rate / 10  # Reduce by 10x

# 2. Gradient explosion
config.max_grad_norm = 0.5  # More aggressive clipping

# 3. Mixed precision issues
config.mixed_precision = Precision.FP32  # Disable mixed precision

# 4. Bad data (check for empty strings)
for item in dataset:
    assert len(item['text']) > 0, "Empty text found!"
```

### Issue 2: Loss Not Decreasing

**Symptoms**: Loss stays flat or oscillates

**Causes & Solutions**:

```python
# 1. Learning rate too low
config.learning_rate = config.learning_rate * 10  # Increase by 10x

# 2. Not enough warmup
config.warmup_ratio = 0.1  # Increase warmup

# 3. Batch size too small
config.per_device_train_batch_size *= 2
config.gradient_accumulation_steps *= 2

# 4. Model too small for task
# Consider increasing d_model, n_layers
```

### Issue 3: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:

```python
# 1. Reduce batch size
config.per_device_train_batch_size = 2

# 2. Enable gradient checkpointing
config.gradient_checkpointing = True

# 3. Use mixed precision
config.mixed_precision = Precision.BF16

# 4. Reduce sequence length
max_length = 256  # Instead of 512 or 1024

# 5. Use gradient accumulation
config.gradient_accumulation_steps = 8  # Compensate for smaller batch
```

### Issue 4: Slow Training

**Symptoms**: Training is slower than expected

**Solutions**:

```python
# 1. Increase batch size (if memory allows)
config.per_device_train_batch_size *= 2

# 2. Use mixed precision
config.mixed_precision = Precision.BF16

# 3. Reduce logging frequency
config.logging_steps = 50  # Instead of 10

# 4. Use more workers for data loading
config.dataloader_num_workers = 4

# 5. Pin memory
config.dataloader_pin_memory = True
```

---

## Complete Training Script

Here's a complete, runnable training script:

```python
#!/usr/bin/env python3
"""
First Training Run - Complete Example
Train a blank slate transformer on sample data.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from models.transformer import NexussTransformer
from models.config import NTFConfig
from training.trainer import Trainer
from training.config import TrainingConfig, Precision


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""
    
    def __init__(self, file_path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['text'])
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': input_ids.clone(),
        }


def data_collator(batch):
    """Collate batch items."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


def main():
    print("=" * 60)
    print("FIRST TRAINING RUN")
    print("=" * 60)
    
    # 1. Configuration
    print("\n1. Setting up configuration...")
    config = TrainingConfig(
        output_dir="./outputs/first-training",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        mixed_precision=Precision.FP32,
        seed=42,
    )
    
    # 2. Model
    print("\n2. Creating model...")
    model_config = NTFConfig(
        vocab_size=16000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=2048,
        use_rope=True,
        activation="swiglu",
    )
    model = NexussTransformer(model_config)
    param_info = model.count_parameters()
    print(f"Model: {param_info['total_millions']} parameters")
    
    # 3. Dataset
    print("\n3. Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using GPT2 tokenizer as example
    train_dataset = TextDataset(
        file_path="data/sample_train.jsonl",
        tokenizer=tokenizer,
        max_length=256,
    )
    
    # 4. Trainer
    print("\n4. Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 5. Train
    print("\n5. Starting training...")
    print("-" * 60)
    metrics = trainer.train()
    print("-" * 60)
    
    # 6. Results
    print("\n6. Training Complete!")
    print(f"Final loss: {metrics['train_loss']:.4f}")
    print(f"Steps: {metrics['global_step']}")
    print(f"Time: {metrics['training_time_seconds']:.1f}s")
    print(f"Speed: {metrics['samples_per_second']:.2f} samples/sec")
    
    # 7. Test generation
    print("\n7. Testing generation...")
    model.eval()
    input_text = "Machine learning is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Output: {generated_text}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Model saved to:", config.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## Summary

You've completed your first training run! You now know how to:

✅ Prepare training data in JSONL format  
✅ Configure training hyperparameters  
✅ Create and initialize a model  
✅ Set up the Trainer  
✅ Monitor training progress  
✅ Save and load checkpoints  
✅ Debug common training issues  

---

## Next Steps

Ready to go deeper? Continue to:

**[Tutorial 03: Data Preparation & Tokenization](03-data-preparation.md)**

You'll learn:
- Building custom tokenizers
- Data cleaning and preprocessing
- Efficient data loading pipelines
- Handling multi-language data
- Dataset versioning

---

*Nexuss AI Company - Training Division*
*From blank slate to trained model, one step at a time.*
