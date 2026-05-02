# Nexuss-Transformer: End-to-End Training Guide

Welcome to the **Nexuss-Transformer** framework. This guide provides a comprehensive, step-by-step workflow to train a blank-slate Decoder-Only Transformer from scratch, perform fine-tuning (Full & PEFT/LoRA), implement RLHF, and manage model versions.

## 📋 Table of Contents
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Phase 1: Pre-Training (Blank Slate)](#phase-1-pre-training-blank-slate)
4. [Phase 2: Fine-Tuning](#phase-2-fine-tuning)
   - [Option A: Full Fine-Tuning](#option-a-full-fine-tuning)
   - [Option B: PEFT/LoRA (Efficient)](#option-b-peftlora-efficient)
5. [Phase 3: Alignment (RLHF/Reward)](#phase-3-alignment-rlhfreward)
6. [Continual Learning & Catastrophic Forgetting](#continual-learning--catastrophic-forgetting)
7. [Checkpointing & Resuming](#checkpointing--resuming)
8. [Model Versioning & Release](#model-versioning--release)
9. [Evaluation](#evaluation)

---

## 1. Installation

Ensure you have Python 3.9+ installed.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for speedup (requires CUDA)
# pip install flash-attn --no-build-isolation
```

---

## 2. Configuration

We use YAML configurations for reproducibility. Default configs are located in `configs/`.

- `configs/pretrain_small.yaml`: Small model for testing/debugging.
- `configs/pretrain_base.yaml`: Standard base model configuration.
- `configs/finetune_lora.yaml`: LoRA configuration for efficient tuning.
- `configs/rlhf_ppo.yaml`: PPO configuration for alignment.

**Key Parameters to Adjust:**
- `data.train_path`: Path to your dataset (JSONL/Parquet).
- `output_dir`: Where checkpoints and logs are saved.
- `model.vocab_size`: Must match your tokenizer.

---

## 3. Phase 1: Pre-Training (Blank Slate)

This phase trains the model from random initialization on a large corpus.

### Command
```bash
python scripts/train.py \
    --config configs/pretrain_base.yaml \
    --mode pretrain
```

### Key Features Enabled:
- **Distributed Training**: Automatically detects GPUs via `accelerate`.
- **Gradient Checkpointing**: Saves memory for larger contexts.
- **Mixed Precision**: Uses FP16/BF16 automatically.
- **Validation**: Runs periodic validation if `data.val_path` is provided.

### Monitoring
Check TensorBoard logs:
```bash
tensorboard --logdir runs/
```

---

## 4. Phase 2: Fine-Tuning

Once the base model is trained, adapt it to specific tasks.

### Option A: Full Fine-Tuning
Updates all model weights. Best for domain adaptation where the target domain differs significantly from pre-training.

```bash
python scripts/train.py \
    --config configs/finetune_full.yaml \
    --mode finetune \
    --model_path outputs/pretrain/checkpoint-final \
    --freeze_strategy none
```

### Option B: PEFT/LoRA (Efficient)
Freezes base weights and trains low-rank adapters. Ideal for multi-task learning and limited VRAM.

```bash
python scripts/train.py \
    --config configs/finetune_lora.yaml \
    --mode finetune \
    --model_path outputs/pretrain/checkpoint-final \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32
```

**Merging Adapters:**
To merge LoRA weights back into the base model for deployment:
```bash
python scripts/merge_lora.py \
    --base_model outputs/pretrain/checkpoint-final \
    --adapter_path outputs/finetune/lora-final \
    --output_path outputs/merged_model_v1
```

---

## 5. Phase 3: Alignment (RLHF/Reward)

Align the model with human preferences using Reward Modeling and PPO/DPO.

### Step 3.1: Train Reward Model
```bash
python scripts/train_reward.py \
    --config configs/reward_model.yaml \
    --data_path data/preference_pairs.jsonl \
    --base_model outputs/merged_model_v1
```

### Step 3.2: PPO Fine-Tuning
```bash
python scripts/train_ppo.py \
    --config configs/rlhf_ppo.yaml \
    --actor_model outputs/merged_model_v1 \
    --reward_model outputs/reward_model/final \
    --ref_model outputs/merged_model_v1
```

*Note: DPO (Direct Preference Optimization) is also supported via `scripts/train_dpo.py` as a more stable alternative to PPO.*

---

## 6. Continual Learning & Catastrophic Forgetting

To train on new data without forgetting previous knowledge, enable regularization strategies in your config:

```yaml
continual_learning:
  enabled: true
  method: "ewc"  # Options: ewc, replay, gem
  ewc_lambda: 5000
  replay_buffer_size: 1000
```

- **EWC (Elastic Weight Consolidation)**: Penalizes changes to important weights.
- **Replay**: Mixes a small buffer of old data with new data.
- **GEM (Gradient Episodic Memory)**: Projects gradients to avoid increasing loss on old tasks.

---

## 7. Checkpointing & Resuming

The framework automatically saves checkpoints every `save_steps`.

### Resume Training
If training stops unexpectedly, resume exactly where it left off:

```bash
python scripts/train.py \
    --config configs/pretrain_base.yaml \
    --resume_from_checkpoint outputs/pretrain/checkpoint-1000
```

### Load Specific Checkpoint for Fine-Tuning
```bash
python scripts/train.py \
    --mode finetune \
    --model_path outputs/pretrain/checkpoint-5000
```

---

## 8. Model Versioning & Release

Manage model releases using semantic versioning.

### Create a Release
Tag a specific checkpoint as a release version:

```bash
python scripts/versioning.py \
    --checkpoint_path outputs/finetune/checkpoint-final \
    --version "1.0.0" \
    --message "Initial instruction tuned release"
```

This creates a structured directory `releases/v1.0.0/` containing:
- The model weights
- `config.json`
- `tokenizer`
- `training_args.json` (for reproducibility)

### Push to Hugging Face Hub
```bash
python scripts/push_to_hub.py \
    --model_path releases/v1.0.0 \
    --repo_id "nexuss0781/Nexuss-Transformer-1B" \
    --token "YOUR_HF_TOKEN"
```

---

## 9. Evaluation

Validate model performance during or after training.

### Automatic Validation
Set `data.val_path` in your config to run automatic perplexity evaluation during training.

### Manual Evaluation
Run generation tests on a specific checkpoint:

```bash
python scripts/evaluate.py \
    --model_path releases/v1.0.0 \
    --tasks mmlu,truthfulqa,hellaswag \
    --batch_size 4
```

### Generate Text
```bash
python scripts/generate.py \
    --model_path releases/v1.0.0 \
    --prompt "Explain quantum entanglement simply:" \
    --max_new_tokens 256 \
    --temperature 0.7
```

---

## Troubleshooting

- **OOM (Out of Memory)**: Reduce `batch_size`, increase `gradient_accumulation_steps`, or enable `gradient_checkpointing`.
- **Loss Spikes**: Lower learning rate or increase warmup steps.
- **Slow Training**: Ensure Flash Attention is installed and `bf16` is enabled if using Ampere+ GPUs.

## License
MIT License. See LICENSE file for details.
