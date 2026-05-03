# Tutorial 05: Parameter-Efficient Fine-Tuning (PEFT/LoRA)

## Introduction

Welcome to one of the most important techniques in modern LLM training! **Parameter-Efficient Fine-Tuning (PEFT)** allows you to adapt massive models while updating only a tiny fraction of parameters.

By the end of this tutorial, you will:
- Understand why PEFT matters
- Master LoRA (Low-Rank Adaptation)
- Implement QLoRA for memory efficiency
- Know when to use different PEFT methods
- Apply PEFT to your models

---

## Why PEFT?

### The Problem with Full Fine-Tuning

```
Full Fine-Tuning a 7B Model:
├─ Parameters: 7,000,000,000
├─ Memory needed: ~28 GB (FP16) + gradients + optimizer states
├─ Total VRAM: ~80+ GB
├─ Checkpoint size: 14 GB per task
└─ Catastrophic forgetting: High risk
```

### The PEFT Solution

```
LoRA Fine-Tuning a 7B Model:
├─ Trainable parameters: ~4,000,000 (0.06%)
├─ Memory needed: ~8 GB (frozen base + small adapters)
├─ Total VRAM: ~16-24 GB
├─ Checkpoint size: 8 MB per task
└─ Catastrophic forgetting: Minimal
```

### Comparison Table

| Aspect | Full Fine-Tuning | LoRA/PEFT |
|--------|-----------------|-----------|
| Trainable params | 100% | 0.1-10% |
| GPU Memory | Very High | Low-Medium |
| Training Speed | Slower | Faster |
| Checkpoint Size | Large (GBs) | Tiny (MBs) |
| Multiple Tasks | One model each | Swap adapters |
| Forgetting | High risk | Low risk |

---

## Understanding LoRA

### Core Idea

Instead of updating all weights `W`, LoRA learns a **low-rank update**:

```
Original: W' = W + ΔW

LoRA: ΔW = B × A

Where:
- W is frozen (base model)
- A: d_model → r (down-projection)
- B: r → d_model (up-projection)
- r << d_model (typically 8-64)
```

### Visual Representation

```
                    Frozen Base Weight (W)
                    ┌───────────────────┐
Input ─────────────→│                   │─────────────→ Output
                    │   768 × 768       │
                    │   (FROZEN)        │
                    └───────────────────┘
                           │
                           │ Add
                           ↓
                    ┌───────────────────┐
                    │     LoRA (A×B)    │
Input ───────→[A]→(r)→[B]───────────────┘
              768→8   8→768
              (Trainable low-rank update)
```

### Mathematical Insight

For a weight matrix `W ∈ ℝ^(d×d)`:
- Full update: `d × d` parameters
- LoRA update: `d × r + r × d = 2dr` parameters

**Example** (d=768, r=8):
- Full: 768 × 768 = 589,824 parameters
- LoRA: 2 × 768 × 8 = 12,288 parameters
- **Reduction: 97.9%**

---

## LoRA Configuration

### Basic LoRAConfig

```python
from finetuning.peft_finetune import LoRAConfig, PEFTTrainer

config = LoRAConfig(
    # Core parameters
    r=8,                    # Rank (smaller = fewer params)
    alpha=16,               # Scaling factor
    dropout=0.05,           # LoRA dropout
    
    # Target modules (which layers to adapt)
    target_modules=["q_proj", "v_proj"],
    
    # Training options
    bias="none",            # Don't train bias
    task_type="CAUSAL_LM",  # Causal language modeling
    
    # Advanced
    modules_to_save=None,   # Additional modules to fully train
    init_lora_weights=True, # Initialize LoRA weights
)

print(f"Scaling factor: {config.scaling}")  # alpha / r = 2.0
```

### Pre-built Configurations

```python
# Default: Good starting point
config = LoRAConfig.default()
# r=16, alpha=32, targets=["q_proj", "v_proj"]

# Full attention: More expressive
config = LoRAConfig.full_attention()
# Targets all attention projections

# Full model: Maximum flexibility
config = LoRAConfig.full_model()
# Targets all linear layers (more params)
```

### Choosing Rank (r)

| Rank | Params | Use Case | Quality |
|------|--------|----------|---------|
| 4    | Minimal | Simple tasks, very limited VRAM | Baseline |
| 8    | Low     | Standard fine-tuning | Good |
| 16   | Medium  | Complex tasks, good VRAM | Better |
| 32   | High    | Very complex, ample VRAM | Best |
| 64+  | Very High | Research, specialized domains | Marginal gains |

**Rule of thumb**: Start with r=8 or r=16, increase only if needed.

---

## Applying LoRA to Your Model

### Step-by-Step Implementation

```python
import torch
from models.transformer import NexussTransformer
from models.config import NTFConfig
from finetuning.peft_finetune import PEFTTrainer, LoRAConfig

# 1. Load pre-trained base model
base_model = NexussTransformer.from_pretrained("./outputs/pretrained")
base_model.eval()  # Ensure frozen

# 2. Configure LoRA
lora_config = LoRAConfig(
    r=16,
    alpha=32,
    dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 3. Wrap with PEFT trainer (applies LoRA)
peft_trainer = PEFTTrainer(
    model=base_model,
    config=lora_config,
    tokenizer=tokenizer,
)

# Get the wrapped model
model = peft_trainer.get_model()

# 4. Verify trainable parameters
peft_trainer.print_trainable_parameters()
# Output example:
# Trainable params: 2,457,600 (0.42%)
# All params: 589,824,000
# Frozen params: 587,366,400
```

### What Gets Trained?

```python
# Inspect which parameters are trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.shape}")

# Typical output:
# base_model.model.layers.0.self_attn.q_proj.lora_A.weight: [8, 768]
# base_model.model.layers.0.self_attn.q_proj.lora_B.weight: [768, 8]
# base_model.model.layers.0.self_attn.v_proj.lora_A.weight: [8, 768]
# base_model.model.layers.0.self_attn.v_proj.lora_B.weight: [768, 8]
# ... (repeated for each layer and target module)
```

---

## Training with LoRA

### Complete Training Script

```python
#!/usr/bin/env python3
"""
LoRA Fine-Tuning Example
Adapt a pre-trained model with minimal parameters.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from models.transformer import NexussTransformer
from training.trainer import Trainer
from training.config import TrainingConfig
from finetuning.peft_finetune import PEFTTrainer, LoRAConfig


class InstructionDataset(Dataset):
    """Dataset for instruction tuning."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        import json
        self.data = []
        self.tokenizer = tokenizer
        
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Format: instruction + input -> output
                text = f"""### Instruction:
{item['instruction']}

### Input:
{item.get('input', '')}

### Response:
{item['output']}"""
                self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': input_ids.clone(),
        }


def main():
    print("=" * 60)
    print("LORA FINE-TUNING")
    print("=" * 60)
    
    # 1. Load base model
    print("\n1. Loading pre-trained base model...")
    base_model = NexussTransformer.from_pretrained("./outputs/pretrained")
    
    # 2. Apply LoRA
    print("\n2. Applying LoRA adapters...")
    lora_config = LoRAConfig(
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    
    peft_wrapper = PEFTTrainer(
        model=base_model,
        config=lora_config,
    )
    
    model = peft_wrapper.get_model()
    
    # 3. Training config (lower LR for fine-tuning)
    print("\n3. Setting up training configuration...")
    train_config = TrainingConfig(
        output_dir="./outputs/lora-finetuned",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # Lower than pre-training
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=100,
        weight_decay=0.01,
    )
    
    # 4. Dataset
    print("\n4. Loading instruction dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataset = InstructionDataset(
        data_path="data/instructions.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )
    
    # 5. Trainer
    print("\n5. Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataset=train_dataset,
    )
    
    # 6. Train
    print("\n6. Starting LoRA fine-tuning...")
    metrics = trainer.train()
    
    # 7. Save adapter only
    print("\n7. Saving LoRA adapter...")
    peft_wrapper.save_adapter("./outputs/lora-adapter")
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"Base model: ./outputs/pretrained")
    print(f"LoRA adapter: ./outputs/lora-adapter")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## QLoRA: Quantized LoRA

### What is QLoRA?

QLoRA combines LoRA with **quantization** for even lower memory usage:

```
Standard LoRA:
- Base model: FP16 (2 bytes/param)
- Adapters: FP16

QLoRA:
- Base model: 4-bit (0.5 bytes/param) ← Quantized!
- Adapters: FP16
```

### Memory Comparison

| Method | 7B Model VRAM | 13B Model VRAM |
|--------|---------------|----------------|
| Full FT | 80+ GB | 160+ GB (impossible) |
| LoRA    | 16-24 GB | 32-48 GB |
| QLoRA   | 8-12 GB | 16-24 GB |

### Using QLoRA

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # Normalized float 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Double quantization
)

# 2. Load model in 4-bit
model = NexussTransformer.from_pretrained(
    "./outputs/pretrained",
    quantization_config=bnb_config,
)

# 3. Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# 4. Apply LoRA (same as before)
lora_config = LoRAConfig(r=16, alpha=32, ...)
peft_wrapper = PEFTTrainer(model=model, config=lora_config)

# Now train with much less memory!
```

---

## Advanced LoRA Techniques

### 1. Target Module Selection

Which modules should you adapt?

```python
# Minimal (fastest, least expressive)
target_modules = ["q_proj", "v_proj"]

# Standard (good balance)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Full attention (more expressive)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "attn_out"
]

# Full model (most expressive, more params)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",  # FFN
    "lm_head",  # Output layer
]
```

### 2. LoRA + Full Layers

Train some modules fully while using LoRA for others:

```python
lora_config = LoRAConfig(
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["lm_head", "norm"],  # Fully train these
)
```

### 3. LoRA Dropout

Prevent overfitting with dropout:

```python
lora_config = LoRAConfig(
    r=16,
    alpha=32,
    dropout=0.05,  # Standard
    # dropout=0.1,  # More regularization
    # dropout=0.0,  # No regularization (risk of overfitting)
)
```

### 4. Alpha Scaling

The effective update is scaled by `alpha / r`:

```python
# Equivalent updates:
LoRA(r=8, alpha=16)   # Scale = 2.0
LoRA(r=16, alpha=32)  # Scale = 2.0
LoRA(r=32, alpha=64)  # Scale = 2.0

# Common practice: alpha = 2 * r
```

---

## Merging and Unmerging LoRA

### Merge LoRA into Base Model

After training, merge adapters for inference:

```python
# Method 1: Merge and unload
merged_model = peft_wrapper.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./outputs/merged-model")

# Now you have a single model (no adapter needed)
```

### Keep Adapter Separate

For multi-task scenarios, keep adapters separate:

```python
# Save only the adapter (tiny!)
peft_wrapper.save_adapter("./outputs/task1-adapter")

# Later, load different adapters for different tasks
peft_wrapper.load_adapter("./outputs/task1-adapter")
# Generate for task 1

peft_wrapper.load_adapter("./outputs/task2-adapter")
# Generate for task 2

# Same base model, different behaviors!
```

---

## Multi-Task Learning with LoRA

```python
# Train separate adapters for different tasks
tasks = ["summarization", "translation", "qa", "code"]

for task in tasks:
    print(f"Training {task} adapter...")
    
    # Load base model
    base_model = NexussTransformer.from_pretrained("./outputs/pretrained")
    
    # Apply LoRA
    peft = PEFTTrainer(base_model, lora_config)
    
    # Train on task data
    dataset = load_task_data(task)
    trainer = Trainer(peft.get_model(), train_dataset=dataset)
    trainer.train()
    
    # Save adapter
    peft.save_adapter(f"./outputs/{task}-adapter")

# Inference: swap adapters as needed
base_model = NexussTransformer.from_pretrained("./outputs/pretrained")
peft = PEFTTrainer(base_model, lora_config)

for task in tasks:
    peft.load_adapter(f"./outputs/{task}-adapter")
    result = generate_for_task(task)
```

---

## Debugging LoRA

### Issue 1: No Learning

**Symptoms**: Loss doesn't decrease

**Solutions**:
```python
# Check if LoRA is applied
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")

# If nothing prints, LoRA wasn't applied correctly

# Increase rank
lora_config.r = 32  # From 16

# Increase learning rate
train_config.learning_rate = 2e-4  # From 1e-4

# Target more modules
lora_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Issue 2: Overfitting

**Symptoms**: Train loss ↓ but eval loss ↑

**Solutions**:
```python
# Increase dropout
lora_config.dropout = 0.1  # From 0.05

# Reduce rank
lora_config.r = 8  # From 16

# Add weight decay
train_config.weight_decay = 0.1  # From 0.01

# Early stopping
# Monitor eval loss and stop when it increases
```

### Issue 3: Poor Generation Quality

**Symptoms**: Coherent training but bad outputs

**Solutions**:
```python
# Use full model LoRA
lora_config = LoRAConfig.full_model()

# Increase alpha
lora_config.alpha = 64  # From 32

# Train longer
train_config.num_train_epochs = 10  # From 5

# Fine-tune learning rate
for lr in [5e-5, 1e-4, 2e-4]:
    test_training(lr)
```

---

## When to Use LoRA vs Full Fine-Tuning

### Use LoRA When:
- ✅ Limited GPU memory
- ✅ Multiple tasks/domains
- ✅ Quick experimentation
- ✅ Catastrophic forgetting is a concern
- ✅ Deploying on edge devices
- ✅ Resource-constrained environment

### Use Full Fine-Tuning When:
- ✅ Maximum performance is critical
- ✅ Ample compute resources
- ✅ Single task focus
- ✅ Domain is very different from pre-training
- ✅ Research requiring full model access

---

## Summary

You've mastered PEFT/LoRA! You now know:

✅ **Why PEFT**: Efficiency, multi-task, less forgetting  
✅ **How LoRA works**: Low-rank decomposition  
✅ **Configuration**: Rank, alpha, target modules  
✅ **QLoRA**: Quantization for extreme efficiency  
✅ **Advanced techniques**: Merging, multi-task  
✅ **Debugging**: Common issues and solutions  

---

## Next Steps

Continue your journey:

**[Tutorial 06: Layer Freezing Strategies](06-layer-freezing.md)**

You'll learn:
- Selective freezing techniques
- Top-k, bottom-k, alternating patterns
- When to freeze vs fine-tune
- Combining freezing with LoRA

---

*Nexuss AI Company - Training Division*
*Efficient adaptation, maximum impact.*
