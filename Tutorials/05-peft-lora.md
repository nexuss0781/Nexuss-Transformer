# Tutorial 05: Parameter-Efficient Fine-Tuning (PEFT/LoRA)

## Introduction

Welcome to one of the most important techniques in modern LLM training! **Parameter-Efficient Fine-Tuning (PEFT)** allows you to adapt massive models while updating only a tiny fraction of parameters.

By the end of this tutorial, you will:
- Understand why PEFT matters
- Master LoRA (Low-Rank Adaptation), AdaLoRA, and LoHa
- Use NTF's unified `PEFTTrainer` interface
- Know when to use different PEFT methods
- Apply PEFT to your models with configuration-driven setup

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

## Using NTF's PEFTTrainer

### Configuration-Driven PEFT Setup

NTF provides a unified interface for all PEFT methods through `PEFTTrainer` and configuration classes:

```python
from ntf.config import NTFConfig, PEFTConfig, ModelConfig, TrainingConfig
from ntf.models import ModelRegistry
from ntf.finetuning import PEFTTrainer

# Configuration-driven PEFT
config = NTFConfig(
    model=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
    peft=PEFTConfig(
        method="lora",  # or "adalora", "loha"
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    ),
    training=TrainingConfig(
        output_dir="./outputs/lora-finetuned",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
    )
)

# Load model with registry
registry = ModelRegistry(config.model)
model, tokenizer = registry.load_model_and_tokenizer()

# Apply PEFT adapters
adapter_config = registry.apply_peft_adapters(config.peft)

# Use PEFTTrainer with built-in adapter handling
trainer = PEFTTrainer(
    model=model,
    adapter_config=adapter_config,
    training_config=config.training,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save only adapter weights (small footprint)
registry.save_adapter(adapter_config, output_dir="./lora_adapter", version="1.0.0")

# Later: Load adapter for inference
registry.load_adapter(model, adapter_path="./lora_adapter")
```

### PEFT Methods Comparison

| Method | NTF Support | Best For |
|--------|-------------|----------|
| LoRA | ✅ Full | General purpose fine-tuning |
| AdaLoRA | ✅ Full | Dynamic rank allocation |
| LoHa | ✅ Full | Complex tasks requiring capacity |
| Prefix Tuning | ⚠️ Partial | Task-specific prompts |
| P-Tuning | ❌ Not implemented | - |

### PEFTConfig Options

```python
from ntf.config import PEFTConfig

# LoRA Configuration
lora_config = PEFTConfig(
    method="lora",
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,        # Dropout
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# AdaLoRA Configuration (dynamic rank allocation)
adalora_config = PEFTConfig(
    method="adalora",
    init_r=12,               # Initial rank
    target_r=8,              # Target rank after pruning
    tinit=200,               # Steps before pruning starts
    tfinal=2000,             # Steps when pruning ends
    deltaT=10,               # Pruning interval
    target_modules=["q_proj", "v_proj"],
)

# LoHa Configuration (Hadamard product)
loha_config = PEFTConfig(
    method="loha",
    r=8,
    alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)
```

---

## Training with LoRA

### Complete Training Script with PEFTTrainer

```python
#!/usr/bin/env python3
"""
LoRA Fine-Tuning Example using NTF's PEFTTrainer
Adapt a pre-trained model with minimal parameters.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ntf.config import NTFConfig, PEFTConfig, ModelConfig, TrainingConfig
from ntf.models import ModelRegistry
from ntf.finetuning import PEFTTrainer


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
    print("LORA FINE-TUNING WITH PEFTTRAINER")
    print("=" * 60)
    
    # 1. Configuration-driven setup
    print("\n1. Creating NTF configuration...")
    config = NTFConfig(
        model=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
        peft=PEFTConfig(
            method="lora",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        training=TrainingConfig(
            output_dir="./outputs/lora-finetuned",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            logging_steps=10,
            save_steps=100,
            weight_decay=0.01,
        )
    )
    
    # 2. Load model with registry
    print("\n2. Loading pre-trained base model...")
    registry = ModelRegistry(config.model)
    model, tokenizer = registry.load_model_and_tokenizer()
    
    # 3. Apply PEFT adapters
    print("\n3. Applying LoRA adapters...")
    adapter_config = registry.apply_peft_adapters(config.peft)
    
    # 4. Dataset
    print("\n4. Loading instruction dataset...")
    train_dataset = InstructionDataset(
        data_path="data/instructions.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )
    
    # 5. Use PEFTTrainer with built-in adapter handling
    print("\n5. Initializing PEFTTrainer...")
    trainer = PEFTTrainer(
        model=model,
        adapter_config=adapter_config,
        training_config=config.training,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    # 6. Train
    print("\n6. Starting LoRA fine-tuning...")
    trainer.train()
    
    # 7. Save only adapter weights (small footprint)
    print("\n7. Saving LoRA adapter...")
    registry.save_adapter(adapter_config, output_dir="./outputs/lora-adapter", version="1.0.0")
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"Base model: {config.model.name}")
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

### Using QLoRA with PEFTConfig

```python
from ntf.config import NTFConfig, PEFTConfig, ModelConfig, TrainingConfig
from ntf.models import ModelRegistry
from ntf.finetuning import PEFTTrainer
from transformers import BitsAndBytesConfig

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # Normalized float 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Double quantization
)

# 2. Create NTF config with quantization support
config = NTFConfig(
    model=ModelConfig(
        name="meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,  # Pass quantization config
    ),
    peft=PEFTConfig(
        method="lora",
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    ),
    training=TrainingConfig(...)
)

# 3. Load model with registry (automatically handles quantization)
registry = ModelRegistry(config.model)
model, tokenizer = registry.load_model_and_tokenizer()

# 4. Apply PEFT adapters
adapter_config = registry.apply_peft_adapters(config.peft)

# 5. Use PEFTTrainer - automatically prepared for k-bit training
trainer = PEFTTrainer(
    model=model,
    adapter_config=adapter_config,
    training_config=config.training,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()
```

---

## Advanced LoRA Techniques

### 1. Target Module Selection

Which modules should you adapt?

```python
from ntf.config import PEFTConfig

# Minimal (fastest, least expressive)
lora_config = PEFTConfig(
    method="lora",
    target_modules=["q_proj", "v_proj"],
)

# Standard (good balance)
lora_config = PEFTConfig(
    method="lora",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Full attention (more expressive)
lora_config = PEFTConfig(
    method="lora",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "attn_out"
    ],
)

# Full model (most expressive, more params)
lora_config = PEFTConfig(
    method="lora",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # FFN
        "lm_head",  # Output layer
    ],
)
```

### 2. LoRA + Full Layers

Train some modules fully while using LoRA for others:

```python
from ntf.config import PEFTConfig

lora_config = PEFTConfig(
    method="lora",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["lm_head", "norm"],  # Fully train these
)
```

### 3. LoRA Dropout

Prevent overfitting with dropout:

```python
from ntf.config import PEFTConfig

lora_config = PEFTConfig(
    method="lora",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,  # Standard
    # lora_dropout=0.1,  # More regularization
    # lora_dropout=0.0,  # No regularization (risk of overfitting)
)
```

### 4. Alpha Scaling

The effective update is scaled by `alpha / r`:

```python
from ntf.config import PEFTConfig

# Equivalent updates:
PEFTConfig(method="lora", r=8, lora_alpha=16)   # Scale = 2.0
PEFTConfig(method="lora", r=16, lora_alpha=32)  # Scale = 2.0
PEFTConfig(method="lora", r=32, lora_alpha=64)  # Scale = 2.0

# Common practice: lora_alpha = 2 * r
```

---

## Merging and Unmerging LoRA

### Merge LoRA into Base Model

After training, merge adapters for inference:

```python
from ntf.models import ModelRegistry

# Method 1: Use registry to merge and save
registry = ModelRegistry(config.model)
merged_model = registry.merge_adapter(model, adapter_config)

# Save merged model
merged_model.save_pretrained("./outputs/merged-model")

# Now you have a single model (no adapter needed)
```

### Keep Adapter Separate

For multi-task scenarios, keep adapters separate:

```python
from ntf.models import ModelRegistry

registry = ModelRegistry(config.model)

# Save only the adapter (tiny!)
registry.save_adapter(adapter_config, output_dir="./outputs/task1-adapter", version="1.0.0")

# Later, load different adapters for different tasks
registry.load_adapter(model, adapter_path="./outputs/task1-adapter")
# Generate for task 1

registry.load_adapter(model, adapter_path="./outputs/task2-adapter")
# Generate for task 2

# Same base model, different behaviors!
```

---

## Multi-Task Learning with LoRA

```python
from ntf.config import NTFConfig, PEFTConfig, ModelConfig, TrainingConfig
from ntf.models import ModelRegistry
from ntf.finetuning import PEFTTrainer

# Train separate adapters for different tasks
tasks = ["summarization", "translation", "qa", "code"]

base_model_name = "meta-llama/Llama-2-7b-hf"

for task in tasks:
    print(f"Training {task} adapter...")
    
    # Create config for this task
    config = NTFConfig(
        model=ModelConfig(name=base_model_name),
        peft=PEFTConfig(method="lora", r=16, lora_alpha=32),
        training=TrainingConfig(output_dir=f"./outputs/{task}-adapter", num_train_epochs=5)
    )
    
    # Load base model
    registry = ModelRegistry(config.model)
    model, tokenizer = registry.load_model_and_tokenizer()
    
    # Apply LoRA
    adapter_config = registry.apply_peft_adapters(config.peft)
    
    # Train on task data
    dataset = load_task_data(task)
    trainer = PEFTTrainer(
        model=model,
        adapter_config=adapter_config,
        training_config=config.training,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    trainer.train()
    
    # Save adapter
    registry.save_adapter(adapter_config, output_dir=f"./outputs/{task}-adapter", version="1.0.0")

# Inference: swap adapters as needed
registry = ModelRegistry(ModelConfig(name=base_model_name))
model, tokenizer = registry.load_model_and_tokenizer()

for task in tasks:
    registry.load_adapter(model, adapter_path=f"./outputs/{task}-adapter")
    result = generate_for_task(task)
```

---

## Debugging LoRA

### Issue 1: No Learning

**Symptoms**: Loss doesn't decrease

**Solutions**:
```python
from ntf.config import PEFTConfig

# Check if LoRA is applied
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")

# If nothing prints, LoRA wasn't applied correctly

# Increase rank
config.peft.r = 32  # From 16

# Increase learning rate
config.training.learning_rate = 2e-4  # From 1e-4

# Target more modules
config.peft.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Issue 2: Overfitting

**Symptoms**: Train loss ↓ but eval loss ↑

**Solutions**:
```python
from ntf.config import PEFTConfig

# Increase dropout
config.peft.lora_dropout = 0.1  # From 0.05

# Reduce rank
config.peft.r = 8  # From 16

# Add weight decay
config.training.weight_decay = 0.1  # From 0.01

# Early stopping
# Monitor eval loss and stop when it increases
```

### Issue 3: Poor Generation Quality

**Symptoms**: Coherent training but bad outputs

**Solutions**:
```python
from ntf.config import PEFTConfig

# Use full model LoRA
config.peft.target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Increase alpha
config.peft.lora_alpha = 64  # From 32

# Train longer
config.training.num_train_epochs = 10  # From 5

# Fine-tune learning rate
for lr in [5e-5, 1e-4, 2e-4]:
    config.training.learning_rate = lr
    test_training(config)
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
