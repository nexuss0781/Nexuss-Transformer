# Phase 3: RLHF, Quantization & Production

## Overview
Phase 3 addresses advanced topics including RLHF implementation, quantization/integration, and production deployment. These are critical for users moving from experimentation to production systems.

**Timeline**: Month 1-2 (Weeks 5-8)  
**Priority**: 🔴 CRITICAL/HIGH

---

## Spec 3.1: Rewrite Tutorial 06 with NTF RLHF Pipeline

### Task
Replace generic RLHF implementation with NTF's `RewardModel` and RLHF pipeline utilities.

### File: Tutorial_06_RLHF_Fine_Tuning.md

### Current Issues
- Uses generic `AutoModelForSequenceClassification` for reward model
- Creates custom preference dataset instead of using NTF utilities
- Shows manual PPO implementation
- No integration with NTF training pipeline

### Complete Replacement Code
```python
from ntf.reward import RewardModel, PreferenceDataset
from ntf.models import ModelRegistry
from ntf.config import RewardConfig

# 1. Load base model
registry = ModelRegistry(model_config)
base_model, tokenizer = registry.load_model_and_tokenizer()

# 2. Initialize NTF's RewardModel
reward_config = RewardConfig(
    base_model_name="meta-llama/Llama-2-7b-hf",
    num_labels=1,
    pad_token_id=tokenizer.pad_token_id
)
reward_model = RewardModel(reward_config)
reward_model.load_base_model(base_model)

# 3. Load preference data with NTF utilities
pref_dataset = PreferenceDataset(
    data_path="preferences.jsonl",
    tokenizer=tokenizer,
    max_length=512
)

# 4. Train reward model
from ntf.reward.trainer import RewardTrainer
reward_trainer = RewardTrainer(
    model=reward_model,
    dataset=pref_dataset,
    config=reward_config
)
reward_trainer.train()

# 5. Use in RLHF pipeline
from ntf.reward.rlhf_pipeline import RLHFPipeline
pipeline = RLHFPipeline(
    policy_model=policy_model,
    reward_model=reward_model,
    reference_model=ref_model,
    tokenizer=tokenizer
)

pipeline.run_ppo(
    prompts=prompts,
    num_iterations=100,
    kl_coeff=0.2
)
```

### Additional Content: RLHF Workflow Diagram
```markdown
### RLHF Workflow

1. **Supervised Fine-Tuning (SFT)**: Train on instruction-following data
2. **Reward Modeling**: Train reward model on human preference data
3. **RL Optimization**: Use PPO to optimize policy against reward model
4. **Evaluation**: Assess alignment with human preferences

NTF provides native components for each stage, ensuring consistency and reproducibility.
```

### Acceptance Criteria
- [ ] RewardModel used instead of generic AutoModel
- [ ] PreferenceDataset demonstrated
- [ ] RLHFPipeline with PPO shown
- [ ] Integration with FullFinetuneTrainer for SFT stage
- [ ] No manual PPO implementation code

---

## Spec 3.2: Enhance Tutorial 11 with NTF Quantization Integration

### Task
Integrate quantization with NTF's ModelRegistry and configuration system.

### File: Tutorial_11_Quantization_and_Optimization.md

### Current State
- Appropriately uses external tools (bitsandbytes, GPTQ, AWQ)
- Missing NTF integration points
- Serving optimization not connected to NTF

### Enhancement Code
```python
from ntf.config import ModelConfig, QuantizationConfig
from ntf.models import ModelRegistry

# Quantization config integrated with NTF
quant_config = QuantizationConfig(
    method="bitsandbytes",  # or "gptq", "awq"
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True
)

model_config = ModelConfig(
    name="meta-llama/Llama-2-7b-hf",
    quantization=quant_config
)

# Registry handles quantized model loading
registry = ModelRegistry(model_config)
model, tokenizer = registry.load_model_and_tokenizer()
# Model automatically loaded in quantized format
```

### Serving Integration Section
```markdown
### Serving Optimized Models

After quantization, deploy your model using optimized serving backends:

#### Option 1: vLLM Integration
```python
from vllm import LLM, SamplingParams

llm = LLM(model="./quantized_model", quantization="awq")
outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=256))
```

#### Option 2: Text Generation Inference (TGI)
```bash
docker run --gpus all \
  -p 8080:80 \
  -v ./quantized_model:/models \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /models \
  --quantize awq
```

#### Option 3: NTF Serving Module (if available)
```python
from ntf.serving import serve_model

serve_model(
    model_path="./quantized_model",
    port=8080,
    backend="vllm"  # or "tgi", "torchserve"
)
```
```

### Required Clarification
```markdown
> **Note**: Quantization is handled by external libraries (bitsandbytes, AutoGPTQ, AutoAWQ). 
> NTF provides configuration integration and streamlined loading via ModelRegistry.
> Serving optimizations use external tools (vLLM, TGI) - NTF does not include a built-in server.
```

### Acceptance Criteria
- [ ] QuantizationConfig integrated with ModelConfig
- [ ] ModelRegistry handles quantized loading
- [ ] Serving options documented with clear tool boundaries
- [ ] External vs. NTF functionality clearly distinguished

---

## Spec 3.3: Resolve Tutorial 12 MLflow vs. ModelRegistry Conflict

### Task
Resolve conflicting registry systems by integrating MLflow with NTF ModelRegistry or replacing with NTF-native approach.

### File: Tutorial_12_Production_Deployment.md

### Current Issue
- Tutorial uses MLflow for model registry
- NTF has its own ModelRegistry class
- Creates confusion about which to use

### Option A: Integrate MLflow with NTF ModelRegistry (Recommended)
```python
from ntf.models import ModelRegistry
import mlflow

# Use NTF for local versioning, MLflow for enterprise registry
registry = ModelRegistry(...)

# Save to NTF registry first
registry.save_model(model, version="1.0.0", metadata={...})

# Then log to MLflow for enterprise tracking
with mlflow.start_run():
    model_uri = registry.get_model_path("1.0.0")
    mlflow.pytorch.log_model(model_uri, "model")
    
    # Log NTF metadata to MLflow
    metadata = registry.get_metadata("1.0.0")
    for key, value in metadata.items():
        if isinstance(value, (int, float, str)):
            mlflow.log_param(key, value)
    
    # Log metrics
    metrics = metadata.get("metrics", {})
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)
```

### Option B: Replace MLflow with NTF ModelRegistry
```python
from ntf.models import ModelRegistry

# NTF ModelRegistry as primary registry
registry = ModelRegistry(registry_path="./production_registry")

# Deploy directly from NTF registry
model, tokenizer = registry.load_model_and_tokenizer(version="1.0.0")

# Export for serving
registry.export_for_serving(
    version="1.0.0",
    format="onnx",  # or "torchscript"
    output_path="./serving_model"
)
```

### Recommended Approach Documentation
```markdown
### Choosing Your Registry Strategy

| Scenario | Recommended Approach |
|----------|---------------------|
| Single developer, local development | NTF ModelRegistry only |
| Small team, shared storage | NTF ModelRegistry with shared path |
| Enterprise, compliance requirements | NTF ModelRegistry + MLflow integration |
| Multi-cloud deployment | NTF ModelRegistry + cloud-native registry |

NTF ModelRegistry provides versioning and metadata tracking out of the box. 
MLflow integration adds enterprise features like experiment tracking, model lineage, and access control.
```

### Monitoring Integration Section
```python
from ntf.utils.metrics import evaluate_generation
import prometheus_client

# Continuous monitoring with NTF metrics
def monitor_model_performance(model_version, test_dataset):
    model, tokenizer = registry.load_model_and_tokenizer(version=model_version)
    
    results = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        metrics=["perplexity", "accuracy"]
    )
    
    # Push to monitoring system
    prometheus_client.Gauge('model_perplexity', 'Model perplexity').set(results['perplexity'])
    prometheus_client.Gauge('model_accuracy', 'Model accuracy').set(results['accuracy'])
```

### Acceptance Criteria
- [ ] Clear guidance on when to use MLflow vs. NTF ModelRegistry
- [ ] Integration pattern demonstrated if both used
- [ ] Monitoring section shows NTF metrics feeding into monitoring systems
- [ ] No conflicting registry approaches without explanation

---

## Spec 3.4: Add NTF-Specific Debugging to Tutorial 13

### Task
Enhance Tutorial 13 with NTF-specific debugging patterns and utilities.

### File: Tutorial_13_Debugging_and_Troubleshooting.md

### Current State
- Good universal content (OOM, NaN losses, slow training)
- Missing NTF-specific debugging
- No common NTF error patterns

### NTF-Specific Debugging Section
```python
# Enable verbose logging in NTF
from ntf.config import NTFConfig
from ntf.utils.logging import setup_logging

setup_logging(level="DEBUG")

config = NTFConfig(
    model=ModelConfig(...),
    training=TrainingConfig(
        logging_level="DEBUG",
        log_on_each_node=True
    )
)

# Validate config before training
from ntf.config import validate_config
errors = validate_config(config)
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")

# Debug dataset preprocessing
from ntf.training.data import TextDataset
dataset = TextDataset(...)

# Inspect processed samples
for i in range(5):
    sample = dataset[i]
    print(f"Sample {i}:")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Attention mask sum: {sample['attention_mask'].sum()}")
```

### Common NTF Error Patterns Table
```markdown
| Error | Cause | Solution |
|-------|-------|----------|
| `ModelRegistryError: Version not found` | Version doesn't exist in registry | Use `list_versions()` to check available versions |
| `PEFT adapter dimension mismatch` | Adapter trained on different model | Ensure same base model and adapter config |
| `TextDataset column mapping error` | Column names don't match | Verify `column_mapping` parameter |
| `FullFinetuneTrainer: Config validation failed` | Invalid training parameters | Run `validate_config()` before training |
| `LayerFreezer: No layers frozen` | Incorrect layer naming | Use `print_trainable_parameters()` to verify |
```

### Debugging Workflows
```markdown
### Debugging Workflow for Common Issues

#### Issue: Training Loss is NaN

1. Check learning rate (try reducing by 10x)
2. Verify gradient clipping is enabled
3. Check for overflow in mixed precision
4. Enable NTF debug logging

```python
config.training.gradient_clip_val = 1.0
config.training.fp16 = False  # Temporarily disable to isolate issue
setup_logging(level="DEBUG")
```

#### Issue: Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing
3. Use LayerFreezer to freeze backbone
4. Try quantization

```python
from ntf.finetuning import LayerFreezer
freezer = LayerFreezer(model)
freezer.freeze_backbone(num_layers_to_keep=4)  # Keep only last 4 layers trainable

config.training.gradient_checkpointing = True
config.training.per_device_train_batch_size = 1
```

#### Issue: Poor Evaluation Metrics

1. Verify dataset preprocessing
2. Check for train/test data leakage
3. Compare multiple checkpoints
4. Evaluate on different metrics

```python
from ntf.utils.metrics import compare_checkpoints
comparison = compare_checkpoints(
    model_paths=["checkpoint-100", "checkpoint-200", "checkpoint-300"],
    eval_dataset=val_dataset,
    metrics=["perplexity", "accuracy", "bleu"]
)
```
```

### Acceptance Criteria
- [ ] NTF logging utilities demonstrated
- [ ] Config validation shown
- [ ] Common NTF error patterns documented
- [ ] Debugging workflows for specific issues included
- [ ] Dataset inspection techniques shown

---

## Deliverables Checklist

- [ ] Tutorial 06: Complete rewrite with RewardModel, PreferenceDataset, RLHFPipeline
- [ ] Tutorial 11: QuantizationConfig integration, serving options clarified
- [ ] Tutorial 12: MLflow vs. ModelRegistry conflict resolved
- [ ] Tutorial 13: NTF-specific debugging patterns added
- [ ] All tutorials: External vs. NTF functionality clearly distinguished

---

## Testing Plan

1. **RLHF Testing**: End-to-end RLHF workflow with reward model training and PPO
2. **Quantization Testing**: Verify all quantization methods (bitsandbytes, GPTQ, AWQ)
3. **Deployment Testing**: Test both MLflow integration and NTF-only approaches
4. **Debugging Testing**: Reproduce common errors and verify solutions work
5. **Integration Testing**: Ensure Phase 3 tutorials build on Phases 1-2 concepts

---

## Dependencies

- Requires `RewardModel`, `PreferenceDataset`, `RLHFPipeline` to be fully implemented
- Requires `QuantizationConfig` class in config system
- Requires `ModelRegistry.export_for_serving()` method (or document alternative)
- Requires `utils/logging.py` with setup_logging function
- Requires `validate_config()` function in config module
- May require implementing missing features or clearly marking as roadmap items

---

## Risk Mitigation

### If RLHF Pipeline Not Fully Implemented
- Mark tutorial as "Advanced/Experimental"
- Provide conceptual overview with code structure
- Link to GitHub issue for implementation status
- Offer alternative: Use external libraries (TRL, DeepSpeed-Chat) with NTF integration examples

### If Serving Module Not Implemented
- Clearly state NTF does not include built-in server
- Focus on export formats (ONNX, TorchScript)
- Provide integration examples with vLLM, TGI, TorchServe
- Document recommended deployment architectures

---

*Phase 3 bridges the gap from experimentation to production-ready deployments.*
