# Tutorial 008: PEFT and LoRA - Parameter-Efficient Fine-Tuning

## 📌 Overview

**Level**: Intermediate  
**Duration**: 60 minutes  
**Prerequisites**: Tutorials 001-007 completed

Learn how to fine-tune large models efficiently using Low-Rank Adaptation (LoRA) and other PEFT methods.

---

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Understand why parameter-efficient fine-tuning matters
- Learn how LoRA works mathematically
- Apply LoRA to your models
- Compare different PEFT methods
- Optimize LoRA hyperparameters

---

## 1. Why Parameter-Efficient Fine-Tuning?

### The Problem with Full Fine-Tuning

| Method | Parameters | Memory | Storage | Use Case |
|--------|------------|--------|---------|----------|
| **Full Fine-Tuning** | 100% trainable | Very High | One copy per task | Small models |
| **PEFT/LoRA** | <1% trainable | Low | Adapter only | Large models |

### Benefits of LoRA

```
Traditional Fine-Tuning:          LoRA Fine-Tuning:
┌─────────────────────┐           ┌─────────────────────┐
│  Full Model         │           │  Frozen Base Model  │
│  (All weights ✓)    │     →     │  (No gradients)     │
│  7B parameters      │           │  + Tiny adapters    │
│  28 GB memory       │           │  ~50 MB memory      │
└─────────────────────┘           │  + LoRA weights     │
                                  │  (<1% trainable)    │
                                  └─────────────────────┘
```

---

## 2. How LoRA Works

### Mathematical Foundation

Instead of updating full weight matrix W:

**Traditional**: W' = W + ΔW (update all parameters)

**LoRA**: W' = W + BA where:
- W ∈ ℝ^(d×k) is frozen
- B ∈ ℝ^(d×r) is trainable (down-projection)
- A ∈ ℝ^(r×k) is trainable (up-projection)
- r << d,k (typically r=8, 16, 32)

### Visual Representation

```
Input x                    Output y
   │                         ▲
   │                         │
   ▼                    ┌────┴────┐
┌──────┐               │  Add    │
│  W   │ (frozen)      └────┬────┘
│(base)│                    │
└──┬───┘              ┌─────┴──────┐
   │                 │   B × A    │
   │                 │  (LoRA)    │
   │                 └─────┬──────┘
   │                       ▲
   └───────────────────────┘
            x
```

### Key Insight

With r=8 and d_model=4096:
- Original: 4096 × 4096 = 16,777,216 parameters
- LoRA: (4096 × 8) + (8 × 4096) = 65,536 parameters
- **Reduction: 99.6% fewer parameters!**

---

## 3. Setting Up LoRA

### Import Required Modules

```python
from peft import LoraConfig, get_peft_model, TaskType
from finetuning.peft_finetune import PEFTTrainer, LoRAConfig
from models.config import NTFConfig
from models.transformer import NexussTransformer
import torch
```

### Basic LoRA Configuration

```python
# Create base model
config = NTFConfig.medium()
model = NexussTransformer(config)

# Configure LoRA
lora_config = LoRAConfig(
    r=16,                    # Rank of update matrices
    alpha=32,                # Scaling factor (alpha/r)
    dropout=0.05,            # LoRA dropout
    target_modules=["q_proj", "v_proj"],  # Which modules to adapt
    bias="none",             # Don't train bias
    task_type=TaskType.CAUSAL_LM,
)

print(f"LoRA scaling factor: {lora_config.scaling}")
```

### Apply LoRA to Model

```python
# Wrap model with PEFT
peft_trainer = PEFTTrainer(
    model=model,
    config=lora_config,
)

# Get the wrapped model
lora_model = peft_trainer.get_model()

# Print parameter statistics
peft_trainer.print_trainable_parameters()
```

### Expected Output

```
Trainable params: 2,097,152 (0.59%)
All params: 350,000,000
Frozen params: 347,902,848
```

---

## 4. Target Module Selection

### Which Modules Should You Adapt?

```python
# Minimal (fastest, least memory)
minimal_target = ["q_proj", "v_proj"]

# Standard (good balance)
standard_target = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Comprehensive (best performance, more params)
full_attention = [
    "q_proj", "k_proj", "v_proj", "o_proj",
]

# All linear layers (maximum adaptation)
all_linear = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

### Impact on Parameter Count

```python
def calculate_lora_params(config, target_modules):
    """Estimate LoRA parameters for given configuration."""
    r = config.r
    d_model = config.d_model
    d_ff = config.d_ff
    
    params = 0
    
    for module in target_modules:
        if "proj" in module:
            if module in ["gate_proj", "up_proj", "down_proj"]:
                # FFN layers: d_model × d_ff or d_ff × d_model
                params += 2 * d_model * d_ff * r
            else:
                # Attention layers: d_model × d_model
                params += 2 * d_model * d_model * r
    
    return params

# Example calculation
targets = ["q_proj", "v_proj"]
params = calculate_lora_params(lora_config, targets)
print(f"LoRA parameters: {params:,} ({params/1e6:.2f}M)")
```

---

## 5. Training with LoRA

### Setup Training Configuration

```python
from training.config import TrainingConfig

# LoRA typically uses higher learning rates
training_config = TrainingConfig(
    output_dir="./outputs/lora_finetune",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    
    # Higher LR for LoRA
    learning_rate=2e-4,
    warmup_ratio=0.1,
    scheduler="cosine",
    
    # Save adapters
    save_steps=100,
    logging_steps=10,
)
```

### Initialize Trainer

```python
from training.trainer import Trainer

trainer = Trainer(
    model=lora_model,
    config=training_config,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
```

### Train

```python
print("Starting LoRA fine-tuning...")
metrics = trainer.train()

print(f"\nTraining completed!")
print(f"Final loss: {metrics['train_loss']:.4f}")
```

---

## 6. Saving and Loading Adapters

### Save Adapter Weights

```python
# Save only LoRA adapters (~50MB vs ~7GB for full model)
adapter_path = "./lora_adapter"
peft_trainer.save_adapter(adapter_path)

print(f"Adapter saved to {adapter_path}")
```

### Load Adapter Later

```python
# Start with fresh base model
base_model = NexussTransformer(NTFConfig.medium())

# Load adapter
peft_trainer.load_adapter(adapter_path)

# Ready to use!
```

### Merge Adapter into Base Model

```python
# Merge LoRA weights permanently
merged_model = peft_trainer.merge_and_unload()

# Now it's a standard model without adapter structure
# Save as regular model
merged_model.save_pretrained("./merged_model")
```

---

## 7. Advanced LoRA Configurations

### QLoRA: Quantized LoRA

For even more memory efficiency:

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model in 4-bit
model_4bit = NexussTransformer.from_pretrained(
    "base_model",
    quantization_config=quant_config,
)

# Apply LoRA on top
lora_model = get_peft_model(model_4bit, lora_config)
```

### AdaLoRA: Adaptive Rank Allocation

```python
from peft import AdaLoraConfig

adalora_config = AdaLoraConfig(
    init_r=12,          # Initial rank
    target_r=8,         # Target rank
    beta1=0.85,         # EMA momentum
    beta2=0.999,
    tinit=200,          # Warmup steps
    tfinal=1000,        # Final pruning step
    deltaT=10,          # Pruning interval
)
```

---

## 8. Hyperparameter Optimization

### Rank (r) Selection

```python
# Experiment with different ranks
rank_study = {
    "r=4": {"params": "0.3%", "performance": "baseline"},
    "r=8": {"params": "0.6%", "performance": "good"},
    "r=16": {"params": "1.2%", "performance": "better"},
    "r=32": {"params": "2.4%", "performance": "diminishing returns"},
    "r=64": {"params": "4.8%", "performance": "overkill"},
}

# Recommendation: Start with r=8 or r=16
```

### Alpha Scaling

```python
# Common practice: alpha = 2 × r
configurations = [
    LoRAConfig(r=8, alpha=16),    # Standard
    LoRAConfig(r=16, alpha=32),   # Recommended
    LoRAConfig(r=32, alpha=64),   # High capacity
]

# Scaling factor = alpha / r
# Higher scaling = stronger LoRA influence
```

### Dropout Tuning

```python
# Typical range: 0.05 - 0.2
dropout_configs = [
    LoRAConfig(r=16, alpha=32, dropout=0.05),  # Low dropout
    LoRAConfig(r=16, alpha=32, dropout=0.1),   # Medium (recommended)
    LoRAConfig(r=16, alpha=32, dropout=0.2),   # High regularization
]
```

---

## 9. Comparing PEFT Methods

### Available Methods

```python
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    P_TUNING_CONFIG,
)
```

### Method Comparison

| Method | Params | Performance | Best For |
|--------|--------|-------------|----------|
| **LoRA** | 0.5-2% | Excellent | General purpose |
| **Prefix Tuning** | 0.1-0.5% | Good | Generation tasks |
| **Prompt Tuning** | <0.1% | Fair | Simple tasks |
| **P-Tuning** | 0.1-0.3% | Good | Understanding tasks |

### Choose Your Method

```python
def choose_peft_method(task_type, model_size, resources):
    """Recommend PEFT method based on constraints."""
    
    if resources == "very_limited":
        return "prompt_tuning"
    elif model_size > 10e9:  # >10B params
        return "lora"
    elif task_type == "generation":
        return "prefix_tuning"
    else:
        return "lora"  # Default recommendation
```

---

## 10. Multi-Task Learning with LoRA

### Train Multiple Adapters

```python
# Base model stays frozen
base_model = NexussTransformer(NTFConfig.medium())

# Task 1: Summarization
lora_summarization = LoRAConfig(task_type="SEQ_2_SEQ_LM")
model_summ = get_peft_model(base_model, lora_summarization, adapter_name="summarization")

# Task 2: Translation
lora_translation = LoRAConfig(task_type="SEQ_2_SEQ_LM")
model_summ.add_adapter("translation", lora_translation)

# Task 3: QA
lora_qa = LoRAConfig(task_type="CAUSAL_LM")
model_summ.add_adapter("qa", lora_qa)

# Switch between tasks
model_summ.set_adapter("summarization")
# ... do summarization ...

model_summ.set_adapter("translation")
# ... do translation ...
```

### Adapter Fusion

```python
# Combine multiple adapters
model_summ.add_adapter(
    "combined",
    adapter_list=["summarization", "translation"],
    weights=[0.5, 0.5],
)
```

---

## 11. Troubleshooting LoRA

### Problem: Poor Performance

**Solutions**:
```python
# Increase rank
config.r = 32

# Target more modules
config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Increase alpha
config.alpha = 64

# Lower dropout
config.dropout = 0.05
```

### Problem: Still Too Much Memory

**Solutions**:
```python
# Use QLoRA (4-bit quantization)
# Reduce batch size
# Use gradient accumulation
# Enable gradient checkpointing
```

### Problem: Slow Training

**Solutions**:
```python
# Use fused operators if available
# Increase batch size
# Use mixed precision (BF16)
# Profile to find bottlenecks
```

---

## 12. Real-World Example

### Fine-tuning for Medical Domain

```python
# Scenario: Adapt general LLM to medical text

# 1. Load pre-trained base
base_model = NexussTransformer.from_pretrained("general_lm_1b")

# 2. Configure LoRA for medical adaptation
medical_lora = LoRAConfig(
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"],
    dropout=0.1,
)

# 3. Apply and train
model = get_peft_model(base_model, medical_lora)
trainer = Trainer(model, medical_training_config, medical_dataset)
trainer.train()

# 4. Save tiny adapter (50MB vs 4GB full model)
model.save_pretrained("medical_adapter")
```

---

## 📚 Summary

### Key Takeaways

✅ **LoRA reduces trainable parameters by 99%+**  
✅ **Target attention modules for best results**  
✅ **Use r=16, alpha=32 as starting point**  
✅ **Save/load adapters independently**  
✅ **Combine with quantization for maximum efficiency**  

### Quick Reference

```python
# Standard LoRA setup
config = LoRAConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
trainer.train()
model.save_pretrained("adapter")
```

---

**Congratulations!** You've mastered parameter-efficient fine-tuning!

➡️ **Next**: [Tutorial 009: Reward Modeling](./009_reward_modeling.md)
