# NTF CLI & Quick Start Guide

## Overview

The Nexuss Transformer Framework (NTF) now includes a professional Command Line Interface (CLI) and simplified Python API for streamlined model training, fine-tuning, and deployment.

## Installation

```bash
# Install in editable mode with CLI support
pip install -e .

# The 'ntf' command will be available in your PATH
ntf --help
```

## CLI Commands

### 1. Training (`ntf train`)

Pre-train or continue training a transformer model.

**Basic Usage:**
```bash
# Train with default config
ntf train --config pretrain_small

# Train with custom YAML config
ntf train --config my_config.yaml
```

**Advanced Usage:**
```bash
# Override config values via CLI
ntf train --config pretrain_small \
    --override model.hidden_size=1024 training.learning_rate=1e-4

# Resume from checkpoint
ntf train --config pretrain_small \
    --resume-from-checkpoint ./outputs/checkpoint-5000

# Enable verbose output
ntf train --config pretrain_small --verbose
```

**Config Override Syntax:**
- Use dot notation: `section.key=value`
- Multiple overrides: `--override key1=val1 key2=val2`
- Supports numbers, booleans, strings, JSON arrays/objects

Examples:
```bash
--override model.hidden_size=1024
--override training bf16=true
--override training.learning_rate=0.0001
--override lora.target_modules='["q_proj","v_proj"]'
```

### 2. Fine-tuning (`ntf finetune`)

Fine-tune a pretrained model using full fine-tuning or LoRA.

**LoRA Fine-tuning (Default):**
```bash
ntf finetune --config finetune_lora.yaml
```

**Full Fine-tuning:**
```bash
ntf finetune --config finetune_full.yaml --no-lora
```

**With Overrides:**
```bash
ntf finetune --config finetune_lora.yaml \
    --override lora.r=32 lora.alpha=64 \
    training.learning_rate=1e-4
```

### 3. Alignment (`ntf align`)

Perform RLHF alignment using DPO or PPO.

**DPO (Direct Preference Optimization):**
```bash
ntf align --config dpo_alignment.yaml --method dpo
```

**PPO (Proximal Policy Optimization):**
```bash
ntf align --config ppo_alignment.yaml --method ppo
```

### 4. Evaluation (`ntf evaluate`)

Evaluate a trained model on various metrics.

```bash
ntf evaluate --model ./outputs/checkpoint-5000 \
    --eval-data test.jsonl \
    --metrics perplexity accuracy
```

### 5. Conversion (`ntf convert`)

Convert models to different formats for deployment.

```bash
# Convert to ONNX
ntf convert --model ./outputs/final --format onnx

# Convert to SafeTensors
ntf convert --model ./outputs/final --format safetensors

# Custom output path
ntf convert --model ./outputs/final --format onnx --output ./deploy/model
```

## Configuration System

### Built-in Configs

NTF includes several pre-configured setups:

| Config | Description | Use Case |
|--------|-------------|----------|
| `pretrain_small` | ~60M params | Testing, edge devices |
| `finetune_lora` | LoRA fine-tuning | Downstream tasks |
| `dpo_alignment` | DPO alignment | RLHF alignment |
| `continual_ewc` | EWC continual learning | Lifelong learning |

### Config Structure

YAML configuration files follow this structure:

```yaml
model:
  vocab_size: 50257
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 12
  max_position_embeddings: 2048
  intermediate_size: 3072
  hidden_act: "swiglu"
  use_rope: true
  gradient_checkpointing: true

data:
  train_path: "data/train.jsonl"
  val_path: "data/val.jsonl"
  tokenizer_name: "ethiobbpe"
  max_length: 2048

training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 3e-4
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  max_steps: 100000
  bf16: true

checkpoint:
  save_steps: 1000
  save_total_limit: 3

validation:
  eval_steps: 500

logging:
  logging_steps: 50
  report_to: ["tensorboard", "wandb"]

output_dir: "outputs/pretrain_small"
seed: 42
```

### Loading Configs Programmatically

```python
from cli.config_loader import load_config, resolve_config_path, parse_cli_overrides, apply_cli_overrides

# Load config
config = load_config('configs/pretrain_small.yaml')

# Or resolve config name
config_path = resolve_config_path('pretrain_small')
config = load_config(config_path)

# Parse CLI overrides
overrides = parse_cli_overrides([
    'model.hidden_size=1024',
    'training.learning_rate=1e-4'
])

# Apply overrides
final_config = apply_cli_overrides(config, overrides)
```

## Python Quick Start API

For rapid prototyping, use the simplified Python API:

### Simple Training

```python
from quickstart import train, finetune_lora, QuickPipeline

# Method 1: Direct function call
trainer = train('small', overrides={
    'training.learning_rate': 1e-3,
    'training.batch_size': 8
})
trainer.train()

# Method 2: Pipeline approach
pipeline = QuickPipeline('small')
pipeline.train(data='train.jsonl', epochs=3)
pipeline.finetune_lora(data='finetune.jsonl', r=16)
pipeline.save('./outputs/my_model')
```

### LoRA Fine-tuning

```python
from quickstart import finetune_lora

# Quick LoRA setup
trainer = finetune_lora(
    model_path='./outputs/pretrained',
    r=16,
    alpha=32,
    target_modules=['q_proj', 'v_proj']
)

# Train with your dataset
# trainer.train(dataset=my_dataset)
```

### Evaluation

```python
from quickstart import evaluate

results = evaluate(
    model_path='./outputs/checkpoint-5000',
    data_path='test.jsonl',
    metrics=['perplexity', 'accuracy']
)

print(f"Perplexity: {results['perplexity']}")
```

## Advanced Usage

### Custom Configuration Files

Create your own config file:

```yaml
# my_custom_config.yaml
model:
  vocab_size: 32000
  hidden_size: 1024
  num_attention_heads: 16
  num_hidden_layers: 24
  max_position_embeddings: 4096
  intermediate_size: 4096
  hidden_act: "silu"
  use_rope: true
  rope_theta: 100000.0
  gradient_checkpointing: true

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  weight_decay: 0.1
  warmup_ratio: 0.05
  lr_scheduler_type: "cosine_with_restarts"
  max_steps: 50000
  bf16: true

lora:
  enable: true
  r: 32
  alpha: 64
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

output_dir: "outputs/my_custom_model"
```

Use it:
```bash
ntf train --config my_custom_config.yaml
```

### Distributed Training

NTF integrates with Hugging Face `accelerate` for distributed training:

```bash
# Multi-GPU training
accelerate launch -m cli.main train --config pretrain_small

# With specific accelerate config
accelerate launch --num_processes=4 --mixed_precision=bf16 \
    -m cli.main train --config pretrain_small
```

### Integration with Experiment Trackers

Enable WandB or TensorBoard in your config:

```yaml
logging:
  report_to: ["wandb"]
  project_name: "my-llm-project"
  run_name: "experiment-1"
```

Or via CLI:
```bash
ntf train --config pretrain_small \
    --override logging.report_to='["wandb"]' \
    logging.project_name="my-project"
```

## Examples

### Example 1: Pre-training a Small Model

```bash
ntf train --config pretrain_small \
    --override data.train_path="data/amharic_corpus.jsonl" \
    data.val_path="data/val.jsonl" \
    training.max_steps=50000 \
    output_dir="outputs/amharic-llm-small"
```

### Example 2: LoRA Fine-tuning for QA Task

```bash
ntf finetune --config finetune_lora.yaml \
    --override model.model_name_or_path="outputs/amharic-llm-small" \
    data.train_path="data/qa_train.jsonl" \
    lora.r=32 \
    lora.alpha=64 \
    output_dir="outputs/qa-lora"
```

### Example 3: DPO Alignment

```bash
ntf align --config dpo_alignment.yaml \
    --method dpo \
    --override model.model_name_or_path="outputs/qa-lora" \
    data.train_path="data/preferences.jsonl" \
    dpo.beta=0.1 \
    output_dir="outputs/aligned-model"
```

### Example 4: Full Pipeline

```python
from quickstart import QuickPipeline

# Create pipeline
pipeline = QuickPipeline(preset='small')

# Pre-train
pipeline.train(
    data='corpus.jsonl',
    epochs=5,
    batch_size=8,
    lr=3e-4,
    output_dir='outputs/base'
)

# Fine-tune with LoRA
pipeline.finetune_lora(
    data='instructions.jsonl',
    r=16,
    epochs=3
)

# Save
pipeline.save('outputs/final_model')
```

## Troubleshooting

### Common Issues

**Config not found:**
```bash
# Use full path or config name from configs/ directory
ntf train --config /full/path/to/config.yaml
# or
ntf train --config pretrain_small  # Uses built-in config
```

**Override syntax error:**
```bash
# Correct: no spaces around '='
--override key=value

# For lists, use quotes
--override lora.target_modules='["q_proj","v_proj"]'
```

**Out of memory:**
```bash
# Reduce batch size and increase gradient accumulation
--override training.per_device_train_batch_size=2 \
    training.gradient_accumulation_steps=8

# Enable gradient checkpointing
--override model.gradient_checkpointing=true

# Enable mixed precision
--override training.bf16=true
```

## Best Practices

1. **Start Small**: Begin with `pretrain_small` config for testing
2. **Use Checkpoints**: Save frequently with `checkpoint.save_steps`
3. **Monitor Resources**: Adjust batch sizes based on GPU memory
4. **Experiment Tracking**: Enable WandB/TensorBoard for reproducibility
5. **Version Control**: Keep configs in version control alongside code

## Migration from Direct API

If you were using the direct Python API, here's how to migrate:

**Before:**
```python
from models import NTFConfig, NexussTransformer
from training import Trainer, TrainingConfig

config = NTFConfig(d_model=768, n_heads=12, n_layers=12)
model = NexussTransformer(config)
train_config = TrainingConfig(learning_rate=1e-4)
trainer = Trainer(model, train_config)
trainer.train()
```

**After (with CLI):**
```bash
ntf train --config pretrain_small \
    --override model.hidden_size=768 training.learning_rate=0.0001
```

**After (with Quick Start):**
```python
from quickstart import train

trainer = train('small', overrides={
    'model.hidden_size': 768,
    'training.learning_rate': 1e-4
})
trainer.train()
```

## Contributing

To add new commands or extend the CLI:

1. Add command function in `cli/commands.py`
2. Add subparser in `cli/main.py`
3. Update this documentation

## Support

For issues or questions:
- GitHub Issues: https://github.com/nexuss0781/Nexuss-Transformer/issues
- Documentation: See README.md and TRAINING.md
