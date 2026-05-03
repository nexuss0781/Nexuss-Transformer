# Phase 2: PEFT, Evaluation & Versioning

## Overview
Phase 2 addresses medium-to-high priority tutorials focusing on parameter-efficient fine-tuning, evaluation metrics, and model versioning. These tutorials build on Phase 1 foundations.

**Timeline**: Month 1 (Weeks 3-4)  
**Priority**: 🟡 MEDIUM-HIGH

---

## Spec 2.1: Update Tutorial 05 to Use PEFTTrainer

### Task
Replace manual LoRA setup with NTF's unified `PEFTTrainer` interface.

### File: Tutorial_05_Parameter_Efficient_Fine_Tuning.md

### Current Issues
- Uses manual `LoraConfig` setup instead of NTF's `PEFTTrainer`
- Missing adapter loading/saving utilities
- No comparison of PEFT methods supported by NTF

### Replacement Code
```python
from ntf.config import NTFConfig, PEFTConfig
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
    training=TrainingConfig(...)
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

### Additional Content: PEFT Methods Comparison Table
```markdown
| Method | NTF Support | Best For |
|--------|-------------|----------|
| LoRA | ✅ Full | General purpose fine-tuning |
| AdaLoRA | ✅ Full | Dynamic rank allocation |
| LoHa | ✅ Full | Complex tasks requiring capacity |
| Prefix Tuning | ⚠️ Partial | Task-specific prompts |
| P-Tuning | ❌ Not implemented | - |
```

### Acceptance Criteria
- [ ] PEFTTrainer used as primary interface
- [ ] All three PEFT methods (LoRA, AdaLoRA, LoHa) documented
- [ ] Adapter saving/loading demonstrated
- [ ] Comparison table included
- [ ] No manual LoraConfig code

---

## Spec 2.2: Update Tutorial 07 to Use Metrics Utilities

### Task
Replace manual metric implementations with NTF's `utils/metrics.py` functions.

### File: Tutorial_07_Evaluation_and_Metrics.md

### Current Issues
- Implements perplexity, accuracy manually
- Missing comprehensive metric coverage
- No unified evaluation interface

### Replacement Code
```python
from ntf.utils.metrics import (
    compute_perplexity,
    compute_accuracy,
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    evaluate_generation
)

# Use NTF's unified evaluation
results = evaluate_generation(
    model=model,
    tokenizer=tokenizer,
    test_dataset=test_dataset,
    metrics=["perplexity", "bleu", "rouge", "bertscore"],
    device="cuda"
)

print(f"Perplexity: {results['perplexity']:.2f}")
print(f"BLEU-4: {results['bleu']:.4f}")
print(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
print(f"BERTScore F1: {results['bertscore']['f1']:.4f}")

# Compare multiple checkpoints
from ntf.utils.metrics import compare_checkpoints
comparison = compare_checkpoints(
    model_paths=["checkpoint1", "checkpoint2", "checkpoint3"],
    eval_dataset=val_dataset,
    metrics=["perplexity", "accuracy"]
)
```

### Additional Content: Metric Selection Guide
```markdown
| Task Type | Recommended Metrics |
|-----------|---------------------|
| Text Generation | Perplexity, BLEU, ROUGE, BERTScore |
| Classification | Accuracy, F1, Precision, Recall |
| Summarization | ROUGE, BERTScore |
| Translation | BLEU, chrF, COMET |
| Question Answering | Exact Match, F1 |
```

### Acceptance Criteria
- [ ] Manual metric implementations removed
- [ ] All NTF metrics utilities demonstrated
- [ ] Metric selection guide included
- [ ] Checkpoint comparison shown

---

## Spec 2.3: Rewrite Tutorial 09 with ModelRegistry Versioning

### Task
Replace manual versioning approach with NTF's `ModelRegistry` semantic versioning.

### File: Tutorial_09_Model_Versioning_and_Checkpointing.md

### Current Issues
- Shows manual directory management with timestamps
- Missing semantic versioning
- No metadata tracking
- Conflicts with NTF's existing ModelRegistry

### Replacement Code
```python
from ntf.models import ModelRegistry
from ntf.config import ModelConfig

# Initialize registry with versioning enabled
registry = ModelRegistry(
    model_config=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
    registry_path="./model_registry",
    enable_versioning=True
)

# After training, save with automatic versioning
registry.save_model(
    model=trained_model,
    tokenizer=tokenizer,
    version="1.0.0",  # Semantic versioning
    metadata={
        "training_config": config.to_dict(),
        "metrics": {"eval_loss": 0.234, "perplexity": 12.5},
        "dataset": "custom_instructions_v1",
        "peft_method": "lora",
        "notes": "Initial fine-tuning run"
    }
)

# List all versions
versions = registry.list_versions()
print(f"Available versions: {versions}")

# Load specific version
model_v1, tokenizer = registry.load_model_and_tokenizer(version="1.0.0")

# Compare versions
comparison = registry.compare_versions(["1.0.0", "1.1.0"], metrics=["eval_loss"])

# Rollback to previous version if needed
registry.rollback("1.0.0")
```

### Additional Content: Versioning Best Practices
```markdown
### Semantic Versioning Guidelines

- **MAJOR.MINOR.PATCH** format (e.g., 1.0.0, 2.1.3)
- **MAJOR**: Breaking changes, architecture modifications
- **MINOR**: New features, performance improvements
- **PATCH**: Bug fixes, minor adjustments

### Metadata Best Practices

- Always include training configuration
- Document dataset version and preprocessing
- Record evaluation metrics
- Add notes about known limitations
- Tag production-ready models
```

### Acceptance Criteria
- [ ] Manual timestamp-based versioning removed
- [ ] Semantic versioning demonstrated
- [ ] Metadata tracking shown
- [ ] Version comparison and rollback covered
- [ ] Best practices section added

---

## Spec 2.4: Enhance Tutorial 08 with NTF Config Integration

### Task
Integrate hyperparameter tuning with NTF's configuration system.

### File: Tutorial_08_Hyperparameter_Tuning.md

### Current State
- Good conceptual coverage of grid search, random search, Bayesian optimization
- Missing NTF config integration
- No early stopping demonstration

### Enhancement Code
```python
from ntf.config import NTFConfig, TrainingConfig
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Define search space aligned with NTF config
search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-4),
    "batch_size": tune.choice([4, 8, 16]),
    "warmup_ratio": tune.uniform(0.05, 0.2),
    "weight_decay": tune.loguniform(1e-4, 1e-2)
}

def train_ntf(config):
    # Build NTF config from trial config
    ntf_config = NTFConfig(
        model=ModelConfig(...),
        training=TrainingConfig(
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            warmup_ratio=config["warmup_ratio"],
            weight_decay=config["weight_decay"],
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            early_stopping_patience=3  # NTF early stopping
        )
    )
    
    # Run training
    trainer = FullFinetuneTrainer(config=ntf_config, ...)
    result = trainer.train()
    
    return {"eval_loss": result.metrics["eval_loss"]}

# Run hyperparameter search
scheduler = ASHAScheduler(metric="eval_loss", mode="min")
analysis = tune.run(
    train_ntf,
    config=search_space,
    num_samples=20,
    scheduler=scheduler,
    resources_per_trial={"gpu": 1}
)

# Get best config
best_config = analysis.get_best_config("eval_loss", "min")
print(f"Best config: {best_config}")
```

### Acceptance Criteria
- [ ] Ray integration demonstrated with NTF config
- [ ] Early stopping via NTF config shown
- [ ] Search space examples provided
- [ ] Best config extraction and usage shown

---

## Spec 2.5: Refocus Tutorial 04 on Sequential Domain Adaptation

### Task
Refocus Tutorial 04 from non-existent multi-task learning to sequential fine-tuning using ContinualLearningWrapper.

### File: Tutorial_04_Multi_Task_Fine_Tuning.md

### Current Issue
- Multi-task learning with task-specific heads not implemented in NTF
- Documents capabilities that don't exist

### Replacement Approach: Sequential Domain Adaptation
```python
from ntf.utils.continual_learning import ContinualLearningWrapper
from ntf.finetuning import FullFinetuneTrainer

# Sequential fine-tuning on multiple domains
wrapper = ContinualLearningWrapper(model)

# Domain 1: Code generation
trainer1 = FullFinetuneTrainer(...)
trainer1.train()
wrapper.save_state("domain1_checkpoint")

# Domain 2: Math reasoning (with regularization to prevent forgetting)
wrapper.apply_ewc_regularization(lambda_ewc=0.5)
trainer2 = FullFinetuneTrainer(...)
trainer2.train()

# Domain 3: Creative writing
wrapper.load_state("domain1_checkpoint")
wrapper.apply_ewc_regularization(lambda_ewc=0.3)
trainer3 = FullFinetuneTrainer(...)
trainer3.train()
```

### Alternative Option
If ContinualLearningWrapper is not fully implemented:
- Clearly mark tutorial as "Planned Feature"
- Provide conceptual overview
- Link to issue tracker for implementation status

### Acceptance Criteria
- [ ] Non-existent feature documentation removed
- [ ] Sequential fine-tuning with EWC demonstrated OR
- [ ] Tutorial marked as planned feature with clear disclaimer
- [ ] Users not misled about available functionality

---

## Spec 2.6: Clarify Tutorial 10 Distributed Training Capabilities

### Task
Clarify current distributed training capabilities vs. roadmap.

### File: Tutorial_10_Distributed_Training.md

### Current Issues
- DeepSpeed integration unclear
- No practical launch scripts
- Missing troubleshooting guide

### Clarification Content
```python
from ntf.config import NTFConfig, TrainingConfig
from ntf.finetuning import FullFinetuneTrainer

# NTF automatically handles distributed training via Accelerate
config = NTFConfig(
    model=ModelConfig(...),
    training=TrainingConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # Accelerate auto-detects distributed setup
        fp16=False,
        bf16=True,
        gradient_checkpointing=True
    )
)

# Trainer automatically uses all available GPUs
trainer = FullFinetuneTrainer(config=config, ...)
trainer.train()  # Distributed training handled internally
```

### Required Disclaimer
```markdown
> **Note**: NTF currently supports multi-GPU training on a single node via Accelerate. 
> Multi-node distributed training is planned for future releases. 
> For large-scale training, consider using external orchestration tools like Kubernetes 
> or cloud provider solutions.
```

### Launch Script Example (Single Node Multi-GPU)
```bash
# Using accelerate launch
accelerate launch --num_processes=4 train_script.py

# Or with explicit config
accelerate config  # Interactive setup
accelerate launch train_script.py
```

### Acceptance Criteria
- [ ] Current capabilities clearly stated
- [ ] Roadmap items identified as such
- [ ] Single-node multi-GPU example working
- [ ] Troubleshooting section for common distributed issues

---

## Deliverables Checklist

- [ ] Tutorial 04: Refocused on sequential domain adaptation
- [ ] Tutorial 05: PEFTTrainer with all PEFT methods documented
- [ ] Tutorial 07: Metrics utilities with selection guide
- [ ] Tutorial 08: Hyperparameter tuning integrated with NTF config
- [ ] Tutorial 09: ModelRegistry semantic versioning
- [ ] Tutorial 10: Distributed training capabilities clarified

---

## Testing Plan

1. **PEFT Testing**: Verify LoRA, AdaLoRA, LoHa all work with PEFTTrainer
2. **Metrics Testing**: Run all metric functions on sample outputs
3. **Versioning Testing**: Test save/load/compare/rollback workflow
4. **Distributed Testing**: Verify multi-GPU training on test hardware
5. **Integration Testing**: Ensure Phase 2 tutorials build on Phase 1 concepts

---

## Dependencies

- Requires `PEFTTrainer` to support LoRA, AdaLoRA, LoHa
- Requires `utils/metrics.py` with all listed functions
- Requires `ModelRegistry` with versioning and metadata support
- Requires `ContinualLearningWrapper` with EWC regularization OR clear roadmap
- Requires Accelerate integration in `FullFinetuneTrainer`

---

*Phase 2 builds efficient fine-tuning workflows on the foundation from Phase 1.*
