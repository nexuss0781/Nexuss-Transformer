# Phase 1: Foundation & Critical Fixes

## Overview
Phase 1 focuses on immediate critical fixes that establish the foundation for all other tutorials. These changes address architecture misalignment, file reference errors, and introduce NTF-native components.

**Timeline**: Week 1-2  
**Priority**: 🔴 CRITICAL/HIGH

---

## Spec 1.1: Fix Tutorial Navigation & References

### Task
Fix all file reference mismatches in Tutorial 00 Table of Contents and throughout documentation.

### Actions
1. Audit all internal links in `Tutorial_00_Introduction_to_Fine_Tuning.md`
2. Update references to match actual filenames:
   - `Tutorial_01_Setting_Up_Your_Environment.md` → `Tutorial_01_Environment_Setup.md`
   - `Tutorial_03_Full_Parameter_Fine_Tuning.md` → `Tutorial_03_Full_Fine_Tuning.md`
   - Continue for all 13 tutorials
3. Verify all cross-references between tutorials

### Acceptance Criteria
- [ ] All internal markdown links resolve correctly
- [ ] No 404 errors when clicking tutorial navigation
- [ ] Consistent naming convention across all references

---

## Spec 1.2: Remove Speculative Hardware Estimates

### Task
Remove or qualify all ungrounded hardware estimates and training time predictions.

### Files Affected
- Tutorial 00: Introduction to Fine-Tuning
- Tutorial 01: Environment Setup
- Tutorial 03: Full Fine-Tuning
- Tutorial 10: Distributed Training

### Actions
1. Search and remove statements like:
   - ❌ "Small Models (7B): 40-80GB VRAM"
   - ❌ "Medium Models (13B-70B): 80GB+ VRAM"
   - ❌ "Training takes 2-3 days on 8x A100"
   - ❌ "Batch size of 32 recommended"

2. Replace with qualified language:
   - ✅ "VRAM requirements vary based on sequence length, batch size, precision, and optimization techniques."
   - ✅ "Use NTF's `LayerFreezer` and gradient checkpointing to reduce memory footprint."
   - ✅ "Start with small batch sizes and scale up based on available memory."

### Acceptance Criteria
- [ ] Zero speculative numerical estimates remain
- [ ] All hardware guidance includes qualification about variability
- [ ] Users directed to NTF utilities for optimization

---

## Spec 1.3: Add NTF Architecture Overview to Tutorial 00

### Task
Introduce core NTF components at the beginning of the tutorial series.

### File: Tutorial_00_Introduction_to_Fine_Tuning.md

### Content to Add
```markdown
## NTF Architecture Overview

Before diving into fine-tuning, understand the core components you'll use:

### Training Components
- **FullFinetuneTrainer**: Production-ready training orchestrator with distributed support via Accelerate
- **PEFTTrainer**: Parameter-efficient fine-tuning with LoRA, AdaLoRA, and LoHa adapters
- **LayerFreezer**: Selectively freeze backbone layers to reduce memory and prevent knowledge degradation

### Model Management
- **ModelRegistry**: Central hub for loading, configuring, and versioning models with semantic versioning
- **Adapter Utilities**: Load and save PEFT adapters with metadata tracking

### Data Pipeline
- **TextDataset**: Unified data loading with built-in chat template support
- **Data Collators**: Preconfigured collators for common tasks

### Evaluation & RLHF
- **Metrics Utilities**: Comprehensive suite (perplexity, BLEU, ROUGE, BERTScore)
- **RewardModel**: Reward model implementation for RLHF
- **RLHFPipeline**: End-to-end RLHF workflow with PPO

### Configuration System
- **NTFConfig**: YAML-based configuration with nested classes for models, training, data, PEFT
- **Config Validation**: Catch configuration errors before training

These components work together to provide a streamlined fine-tuning experience from research to production.
```

### Acceptance Criteria
- [ ] Core components introduced before first code example
- [ ] Clear mapping between component names and their purposes
- [ ] Links to detailed tutorials for each component

---

## Spec 1.4: Add Foundational Fine-Tuning Types Section

### Task
Add educational content explaining fine-tuning approaches before Tutorial 03.

### Placement
Insert after Tutorial 00 architecture overview, before Tutorial 03 practical examples.

### Content to Add
```markdown
## Understanding Fine-Tuning Types

Fine-tuning adapts pre-trained models to specific tasks. NTF supports three main approaches:

### 1. Full Fine-Tuning
- **What**: Update all model parameters
- **When**: Sufficient VRAM, domain shift is large
- **NTF Component**: `FullFinetuneTrainer` + `LayerFreezer`
- **Trade-offs**: Best performance, highest resource usage

### 2. Parameter-Efficient Fine-Tuning (PEFT)
- **What**: Update small adapter parameters, freeze backbone
- **When**: Limited VRAM, multiple tasks, quick iteration
- **NTF Component**: `PEFTTrainer` (LoRA, AdaLoRA, LoHa)
- **Trade-offs**: Lower resource usage, slightly reduced performance

### 3. Continual Fine-Tuning
- **What**: Sequential fine-tuning on multiple domains
- **When**: Lifelong learning, multi-domain deployment
- **NTF Component**: `ContinualLearningWrapper` + regularization
- **Trade-offs**: Maintains knowledge across domains, requires careful tuning

Choose your approach based on resources and requirements. The following tutorials will demonstrate each method using NTF's native components.
```

### Acceptance Criteria
- [ ] Clear comparison table of approaches
- [ ] Each approach linked to NTF component
- [ ] Decision guidance for users

---

## Spec 1.5: Update Tutorial 02 to Use TextDataset

### Task
Replace custom dataset implementation with NTF's built-in `TextDataset`.

### File: Tutorial_02_Working_with_Datasets.md

### Current Issue
Tutorial creates custom `CustomDataset` class from scratch, conflicting with existing `TextDataset` in `training/data.py`.

### Replacement Code
```python
from ntf.training.data import TextDataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Use NTF's built-in dataset
dataset = TextDataset(
    data_path="formatted_data.json",
    tokenizer=tokenizer,
    max_length=512,
    use_chat_template=True,  # Built-in support
    column_mapping={
        "instruction": "instruction",
        "input": "context", 
        "output": "response"
    }
)

# Access preprocessed data
train_data = dataset.get_train_dataset()
eval_data = dataset.get_eval_dataset()
```

### Additional Content
Add section on custom data collators:
```python
from ntf.training.data import create_data_collator

collator = create_data_collator(
    tokenizer=tokenizer,
    padding=True,
    max_length=512
)
```

### Acceptance Criteria
- [ ] Custom dataset class removed
- [ ] TextDataset demonstrated with chat template
- [ ] Column mapping explained
- [ ] Data collator section added

---

## Spec 1.6: Rewrite Tutorial 03 with FullFinetuneTrainer

### Task
Complete rewrite of Tutorial 03 to use NTF's `FullFinetuneTrainer` instead of raw HuggingFace Trainer.

### File: Tutorial_03_Full_Fine_Tuning.md

### Current Issues
- Uses raw HuggingFace `Trainer`
- Manual training loop without accelerator support
- Missing gradient checkpointing, mixed precision hooks
- No ModelRegistry usage
- No LayerFreezer demonstration

### Complete Replacement Code
```python
from ntf.config import NTFConfig, ModelConfig, TrainingConfig
from ntf.models import ModelRegistry
from ntf.finetuning import FullFinetuneTrainer
from ntf.training.data import TextDataset

# 1. Configuration-driven setup
config = NTFConfig(
    model=ModelConfig(
        name="meta-llama/Llama-2-7b-hf",
        trust_remote_code=True,
        torch_dtype="bfloat16"
    ),
    training=TrainingConfig(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4
    )
)

# 2. Use ModelRegistry for model loading
registry = ModelRegistry(config.model)
model, tokenizer = registry.load_model_and_tokenizer()

# Optional: Freeze backbone layers to reduce memory
from ntf.finetuning import LayerFreezer
freezer = LayerFreezer(model)
freezer.freeze_backbone(num_layers_to_keep=-1)  # Keep all trainable, or specify number

# 3. Prepare dataset with NTF utilities
dataset = TextDataset(
    data_path="formatted_data.json",
    tokenizer=tokenizer,
    max_length=512,
    use_chat_template=True
)

# 4. Initialize NTF's FullFinetuneTrainer
trainer = FullFinetuneTrainer(
    model=model,
    config=config.training,
    train_dataset=dataset.get_train_dataset(),
    eval_dataset=dataset.get_eval_dataset(),
    tokenizer=tokenizer
)

# 5. Train with built-in accelerator support
trainer.train()

# 6. Save with versioning
registry.save_model(trainer.model, output_dir="./final_model", version="1.0.0")
```

### Acceptance Criteria
- [ ] FullFinetuneTrainer used as primary interface
- [ ] ModelRegistry demonstrated for loading/saving
- [ ] LayerFreezer included with explanation
- [ ] Configuration system showcased
- [ ] Gradient checkpointing and mixed precision enabled
- [ ] No raw HuggingFace Trainer code

---

## Spec 1.7: Add Verification Step to Tutorial 01

### Task
Add NTF component verification after installation.

### File: Tutorial_01_Environment_Setup.md

### Content to Add
```python
from ntf.models import ModelRegistry
from ntf.finetuning import FullFinetuneTrainer
from ntf.config import NTFConfig

# Verify installation
print(f"NTF Version: {ntf.__version__}")
print("Core components imported successfully!")
```

### Acceptance Criteria
- [ ] Verification code block added
- [ ] Users can confirm installation works
- [ ] Common import errors documented with solutions

---

## Deliverables Checklist

- [ ] Tutorial 00: Fixed navigation, added architecture overview, added fine-tuning types section
- [ ] Tutorial 01: Added verification step, removed hardware estimates
- [ ] Tutorial 02: Replaced custom dataset with TextDataset
- [ ] Tutorial 03: Complete rewrite with FullFinetuneTrainer, ModelRegistry, LayerFreezer
- [ ] All files: Removed speculative hardware estimates
- [ ] All internal links verified working

---

## Testing Plan

1. **Link Testing**: Click through all tutorial navigation links
2. **Code Testing**: Run all code examples in isolated environment
3. **Import Testing**: Verify all NTF components import correctly
4. **User Testing**: Have new user follow tutorials 00-03 end-to-end

---

## Dependencies

- Requires `TextDataset`, `FullFinetuneTrainer`, `ModelRegistry`, `LayerFreezer` to be fully implemented
- Requires configuration system (`NTFConfig`) to support all shown parameters
- Requires `training/data.py` to have `create_data_collator` function

---

*Phase 1 establishes the foundation. Subsequent phases build on these corrected fundamentals.*
