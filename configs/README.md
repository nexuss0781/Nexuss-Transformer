# Nexuss Transformer Framework - Production Configuration Examples

This directory contains ready-to-use YAML configuration files for various training scenarios.

## Quick Start

```bash
# Pre-training (blank slate)
python train.py --config configs/pretrain_small.yaml

# Full fine-tuning
python train.py --config configs/finetune_full.yaml

# LoRA fine-tuning
python train.py --config configs/finetune_lora.yaml

# DPO alignment
python train.py --config configs/dpo_alignment.yaml

# Continual learning with EWC
python train.py --config configs/continual_ewc.yaml
```

## Configuration Files

### Pre-training Configurations

| File | Description | Parameters | Context |
|------|-------------|------------|---------|
| `pretrain_nano.yaml` | Tiny model for testing | 10M | 512 |
| `pretrain_small.yaml` | Small research model | 100M | 1024 |
| `pretrain_medium.yaml` | Medium production model | 500M | 2048 |
| `pretrain_large.yaml` | Large scale model | 1B+ | 4096 |

### Fine-tuning Configurations

| File | Description | Method | Use Case |
|------|-------------|--------|----------|
| `finetune_full.yaml` | Full parameter tuning | Full | Domain adaptation |
| `finetune_lora.yaml` | Parameter-efficient | LoRA | Quick adaptation |
| `finetune_qalora.yaml` | Quantized LoRA | QLoRA | Low VRAM |

### RLHF Configurations

| File | Description | Method |
|------|-------------|--------|
| `reward_training.yaml` | Reward model training | Pairwise ranking |
| `ppo_alignment.yaml` | PPO-based alignment | On-policy RL |
| `dpo_alignment.yaml` | DPO-based alignment | Direct preference |

### Continual Learning

| File | Description | Strategy |
|------|-------------|----------|
| `continual_ewc.yaml` | Elastic Weight Consolidation | Regularization |
| `continual_replay.yaml` | Experience Replay | Rehearsal |
| `continual_gem.yaml` | Gradient Episodic Memory | Constraint-based |

## Custom Configuration

Create your own config by combining elements:

```yaml
# my_custom_config.yaml
model:
  vocab_size: 32000
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 2048
  use_rope: true
  use_swiglu: true

training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 3e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_steps: 100000
  
  # Mixed precision
  bf16: true
  fp16: false
  
  # Optimizations
  gradient_checkpointing: true
  flash_attention: true

checkpoint:
  save_steps: 1000
  keep_last_n: 3
  save_best_only: true

validation:
  eval_steps: 500
  metrics:
    - perplexity
    - accuracy
```

## Environment Variables

Override config values with environment variables:

```bash
export NTF_LEARNING_RATE=1e-4
export NTF_BATCH_SIZE=16
export NTF_OUTPUT_DIR=./my_experiments

python train.py --config configs/pretrain_small.yaml
```

## Distributed Training

All configs support distributed training via Accelerate:

```bash
# Multi-GPU
accelerate launch --num_processes=4 train.py --config configs/pretrain_medium.yaml

# Multi-node
accelerate launch --num_machines=4 --num_processes=32 train.py --config configs/pretrain_large.yaml
```

## Monitoring

Configs include Weights & Biases integration by default:

```yaml
logging:
  report_to: wandb
  project_name: nexuss-transformer
  run_name: experiment-001
```

Set `WANDB_API_KEY` environment variable to enable.
