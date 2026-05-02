# Nexuss Transformer Framework (NTF) - Complete Component Documentation

## Overview

The Nexuss Transformer Framework is a **professional, production-ready system** for training, fine-tuning, and aligning transformer language models. It provides **23 core components** organized into 8 categories, with full CLI automation and configuration management.

---

## Component Summary (23 Components)

| Category | Components | Count |
|----------|------------|-------|
| **Model** | nexuss_transformer, ntf_config, ethio_tokenizer | 3 |
| **Training** | trainer, training_config, data_collator, checkpoint_manager | 4 |
| **Fine-tuning** | peft_trainer, lora_config, full_finetuner, layer_freezer | 4 |
| **Reward/RLHF** | dpo_trainer, ppo_trainer, reward_model, dpo_config | 4 |
| **Continual Learning** | ewc_regularizer, replay_buffer, gem_optimizer | 3 |
| **Evaluation** | metrics_evaluator, throughput_benchmark | 2 |
| **Versioning** | model_registry, model_version | 2 |
| **Data** | training_dataset | 1 |
| **TOTAL** | | **23** |

---

## Detailed Component Documentation

### 1. MODEL COMPONENTS

#### 1.1 NexussTransformer (`models.transformer.NexussTransformer`)
**Description:** Main decoder-only transformer model with RoPE, SwiGLU, and RMSNorm

**Configuration Class:** `NTFConfig`

**Required Parameters:**
- `vocab_size`: Vocabulary size (e.g., 16000 for EthioBBPE)
- `d_model`: Hidden dimension size
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers

**Optional Parameters:**
- `d_ff`: Feed-forward dimension (default: 4 * d_model)
- `max_seq_len`: Maximum sequence length
- `activation`: Activation function (swiglu, gelu, relu)
- `use_rope`: Enable Rotary Position Embeddings
- `dropout`: Dropout probability

**CLI Usage:**
```bash
ntf train --config pretrain_small
ntf train --config pretrain_small --override model.hidden_size=1024 model.num_hidden_layers=16
```

**Python API:**
```python
from models import NTFConfig, NexussTransformer

config = NTFConfig(
    vocab_size=16000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    use_rope=True,
    activation='swiglu'
)
model = NexussTransformer(config)
```

---

#### 1.2 NTFConfig (`models.config.NTFConfig`)
**Description:** Configuration class for NexussTransformer architecture

**Parameters:**
- All architecture hyperparameters
- Validation and default value handling

**CLI Usage:**
```bash
# Override via dot notation
--override model.vocab_size=32000 model.dropout=0.1
```

---

#### 1.3 EthioBBPE Tokenizer (`models.tokenizer.EthioBBPE`)
**Description:** BPE tokenizer optimized for Ethiopian languages (Amharic, Tigrinya, Oromo, etc.)

**Required Parameters:**
- `vocab_file`: Path to vocabulary file
- `merges_file`: Path to merge rules file

**Optional Parameters:**
- `normalization`: Text normalization strategy

**Configuration:**
```yaml
data:
  tokenizer_name: "ethiobbpe"
  max_length: 2048
```

---

### 2. TRAINING COMPONENTS

#### 2.1 Trainer (`training.trainer.Trainer`)
**Description:** Main training loop with Accelerate integration for distributed training

**Features:**
- Mixed precision (FP16/BF16/FP32)
- Gradient accumulation
- Distributed training (DDP, FSDP)
- Checkpointing and resume
- Evaluation during training
- Logging to TensorBoard/W&B

**Required Parameters:**
- `model`: Model to train
- `config`: TrainingConfig instance

**Optional Parameters:**
- `train_dataset`: Training dataset
- `eval_dataset`: Evaluation dataset
- `data_collator`: Batch collation function

**CLI Usage:**
```bash
ntf train --config pretrain_small
ntf train --config pretrain_small --resume-from-checkpoint ./outputs/checkpoint-5000
```

**CLI Flags:**
- `--resume-from-checkpoint`: Resume from checkpoint path
- `--max-steps`: Override maximum training steps
- `--learning-rate`: Override learning rate
- `--batch-size`: Override batch size

---

#### 2.2 TrainingConfig (`training.config.TrainingConfig`)
**Description:** Comprehensive training hyperparameters configuration

**Core Parameters:**
- `output_dir`: Directory for checkpoints and logs
- `num_train_epochs`: Number of training epochs
- `max_steps`: Maximum training steps (-1 for epochs)
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps

**Optimization:**
- `learning_rate`: Peak learning rate
- `weight_decay`: Weight decay coefficient
- `warmup_ratio`: Warmup ratio
- `scheduler`: LR scheduler type (linear, cosine, constant)

**Precision:**
- `mixed_precision`: FP16, BF16, or FP32
- `gradient_checkpointing`: Enable gradient checkpointing

**Checkpointing:**
- `save_steps`: Save every N steps
- `save_total_limit`: Maximum checkpoints to keep

**CLI Usage:**
```bash
--override training.learning_rate=1e-4 training.max_steps=50000 training.mixed_precision=bf16
```

---

#### 2.3 DataCollatorForLanguageModeling (`training.data.DataCollatorForLanguageModeling`)
**Description:** Dynamic padding and batch collation for language modeling

**Features:**
- Dynamic padding to max length in batch
- Label creation for next-token prediction
- Attention mask generation
- Padding token handling

**Parameters:**
- `pad_token_id`: Token ID for padding
- `max_length`: Maximum sequence length (None for dynamic)
- `return_tensors`: Type of tensors ('pt', 'np')

---

#### 2.4 CheckpointManager (`training.checkpoint.CheckpointManager`)
**Description:** Checkpoint save/load with versioning and best model tracking

**Features:**
- Automatic checkpoint naming with timestamps
- Limit total saved checkpoints
- Best model tracking
- Metadata preservation
- Resume capability

**Parameters:**
- `output_dir`: Base directory for checkpoints
- `save_total_limit`: Maximum number of checkpoints to keep

**CLI Flags:**
- `--save-steps`: Save checkpoint every N steps
- `--save-total-limit`: Maximum checkpoints to keep

---

### 3. FINE-TUNING COMPONENTS

#### 3.1 PEFTTrainer (`finetuning.peft_finetune.PEFTTrainer`)
**Description:** Parameter-efficient fine-tuning with LoRA, AdaLoRA, and adapters

**Features:**
- LoRA adapter training
- QLoRA support (4-bit/8-bit)
- Adapter merging and unloading
- Trainable parameter tracking

**Required Parameters:**
- `model`: Base model to fine-tune
- `config`: LoRAConfig instance

**Optional Parameters:**
- `tokenizer`: Tokenizer for preprocessing

**CLI Usage:**
```bash
ntf finetune --config finetune_lora.yaml
ntf finetune --config finetune_lora.yaml --override lora.r=32 lora.alpha=64
```

**CLI Flags:**
- `--lora-r`: LoRA rank
- `--lora-alpha`: LoRA alpha scaling factor
- `--lora-dropout`: LoRA dropout
- `--target-modules`: Modules to apply LoRA

---

#### 3.2 LoRAConfig (`finetuning.peft_finetune.LoRAConfig`)
**Description:** LoRA adapter configuration with advanced options

**Core Parameters:**
- `r`: LoRA rank (dimension of low-rank matrices)
- `alpha`: LoRA scaling factor
- `dropout`: Dropout probability for LoRA layers

**Target Modules:**
- `target_modules`: Module names to apply LoRA (e.g., ['q_proj', 'v_proj'])

**Advanced:**
- `bias`: Whether to train bias parameters
- `modules_to_save`: Modules to fully train in addition to adapters
- `layers_to_transform`: Specific layers to transform

**Presets:**
- `default()`: Standard LoRA config
- `full_attention()`: Target all attention modules
- `full_model()`: Target all linear layers

---

#### 3.3 FullFinetuneTrainer (`finetuning.full_finetune.FullFinetuneTrainer`)
**Description:** Full parameter fine-tuning with discriminative LR and gradual unfreezing

**Features:**
- Discriminative learning rates by layer
- Gradual unfreezing during training
- Layer-wise learning rate decay

**Optional Parameters:**
- `discriminative_lr`: Different learning rates per layer group
- `layerwise_lr_decay`: Decay factor for deeper layers

**CLI Usage:**
```bash
ntf finetune --config finetune_full.yaml --no-lora
```

**CLI Flags:**
- `--no-lora`: Disable LoRA for full fine-tuning
- `--discriminative-lr`: Different LRs per layer
- `--layerwise-lr-decay`: LR decay for deeper layers

---

#### 3.4 LayerFreezer (`finetuning.freeze.LayerFreezer`)
**Description:** Flexible layer freezing strategies for partial fine-tuning

**Freezing Strategies:**
- **Top-k:** Freeze the last K transformer layers
- **Bottom-k:** Freeze the first K transformer layers
- **Alternating:** Freeze even/odd layers
- **By name:** Freeze specific module patterns
- **Custom:** Keep only specified modules trainable

**Methods:**
- `freeze_all()`: Freeze all parameters
- `unfreeze_all()`: Unfreeze all parameters
- `freeze_top_k(k)`: Freeze top K layers
- `freeze_bottom_k(k)`: Freeze bottom K layers
- `freeze_alternating(pattern)`: Alternating pattern
- `freeze_by_name(patterns)`: Freeze by name patterns
- `freeze_except(keep_trainable)`: Freeze all except specified

**CLI Flags:**
- `--freeze-top-k`: Freeze top K transformer layers
- `--freeze-bottom-k`: Freeze bottom K transformer layers
- `--freeze-alternating`: Freeze alternating layers
- `--freeze-modules`: Freeze specific modules by name
- `--unfreeze-modules`: Unfreeze specific modules

---

### 4. REWARD/RLHF COMPONENTS

#### 4.1 DPOTrainer (`reward.dpo_trainer.DPOTrainer`)
**Description:** Direct Preference Optimization for RLHF alignment

**Features:**
- Simpler alternative to PPO
- Direct optimization from preferences
- Multiple loss types (sigmoid, hinge, IPO)
- LoRA integration

**Required Parameters:**
- `model`: Policy model to train
- `ref_model`: Reference model (frozen)
- `train_dataset`: Preference dataset

**DPO Hyperparameters:**
- `beta`: Temperature parameter (default: 0.1)
- `loss_type`: Loss type (sigmoid, hinge, ipo, kto_pair)
- `label_smoothing`: Label smoothing factor

**CLI Usage:**
```bash
ntf align --config dpo_alignment.yaml --method dpo
ntf align --config dpo_alignment.yaml --method dpo --override dpo.beta=0.2
```

**CLI Flags:**
- `--method`: Alignment method (dpo/ppo)
- `--beta`: DPO temperature parameter
- `--loss-type`: DPO loss type

---

#### 4.2 PPOTrainer (`reward.ppo_trainer.PPOTrainer`)
**Description:** Proximal Policy Optimization for RLHF with reward model

**Features:**
- Full PPO implementation via TRL
- Reward model integration
- Value network training
- Generation during training

**PPO Hyperparameters:**
- `ppo_epochs`: PPO optimization epochs per batch
- `cliprange`: PPO clipping range
- `gamma`: Reward discount factor
- `lam`: GAE lambda parameter
- `whiten_rewards`: Normalize rewards

**CLI Flags:**
- `--ppo-epochs`: PPO optimization epochs per batch
- `--clip-range`: PPO clipping range
- `--gamma`: Reward discount factor
- `--lam`: GAE lambda parameter

---

#### 4.3 RewardTrainer (`reward.reward_model.RewardTrainer`)
**Description:** Reward model training for RLHF with pairwise ranking

**Loss Types:**
- `pairwise`: Ranking loss (chosen > rejected)
- `pointwise`: Margin-based loss

**Features:**
- Pairwise comparison training
- Accuracy metric tracking
- LoRA support for efficiency

**CLI Flags:**
- `--loss-type`: Reward loss (pairwise/pointwise)
- `--margin`: Margin for ranking loss

---

#### 4.4 DPOTrainerConfig (`reward.dpo_trainer.DPOTrainerConfig`)
**Description:** DPO-specific hyperparameters configuration

**Parameters:**
- `beta`: Temperature parameter
- `loss_type`: Loss type
- `label_smoothing`: Label smoothing
- `truncation_mode`: Truncation strategy
- `max_length`: Maximum sequence length
- `max_prompt_length`: Maximum prompt length

---

### 5. CONTINUAL LEARNING COMPONENTS

#### 5.1 EWCRegularizer (`utils.continual_learning.EWCRegularizer`)
**Description:** Elastic Weight Consolidation to prevent catastrophic forgetting

**Mechanism:**
- Computes Fisher Information Matrix
- Penalizes changes to important weights
- Preserves knowledge from previous tasks

**Parameters:**
- `ewc_lambda`: Regularization strength (default: 1000.0)
- `fisher_samples`: Samples for Fisher estimation
- `damping`: Damping factor for numerical stability

**CLI Usage:**
```bash
ntf train --config continual_ewc.yaml
ntf train --config continual_ewc.yaml --override continual_learning.ewc_lambda=500
```

**CLI Flags:**
- `--ewc-lambda`: EWC regularization strength
- `--fisher-samples`: Samples for Fisher estimation

---

#### 5.2 ReplayBuffer (`utils.continual_learning.ReplayBuffer`)
**Description:** Experience replay buffer for continual learning

**Selection Strategies:**
- `uniform`: Random sampling
- `recent`: Keep most recent samples
- `diverse`: Maximize diversity

**Features:**
- Reservoir sampling for streaming data
- Task-specific buffers
- Batch mixing with current data

**Parameters:**
- `replay_size`: Size of replay buffer
- `replay_ratio`: Ratio of replay data in batches
- `selection_strategy`: Selection strategy

**CLI Flags:**
- `--replay-size`: Size of replay buffer
- `--replay-ratio`: Ratio of replay data in batches
- `--selection-strategy`: Buffer selection strategy

---

#### 5.3 GEMOptimizer (`utils.continual_learning.GEMOptimizer`)
**Description:** Gradient Episodic Memory optimizer for multi-task learning

**Mechanism:**
- Stores gradients from previous tasks
- Projects current gradient to avoid interference
- Quadratic programming for constraint solving

**Parameters:**
- `memory_size`: Examples per task
- `num_tasks`: Expected number of tasks
- `use_quadprog`: Use quadratic programming

---

### 6. EVALUATION COMPONENTS

#### 6.1 MetricsEvaluator (`utils.metrics.EvaluationResults`)
**Description:** Comprehensive evaluation metrics

**Metrics:**
- **Core:** Perplexity, Loss, Accuracy
- **Token-level:** Token accuracy, Exact match
- **Generation:** BLEU, ROUGE-L
- **Task-specific:** Custom metrics

**CLI Usage:**
```bash
ntf evaluate --model-path ./outputs/final --eval-data test.jsonl
ntf evaluate --model-path ./outputs/final --metrics perplexity accuracy bleu
```

**CLI Flags:**
- `--metrics`: Metrics to compute
- `--eval-data`: Path to evaluation dataset

---

#### 6.2 ThroughputBenchmark (`utils.metrics.benchmark_throughput`)
**Description:** Model throughput benchmarking

**Metrics:**
- **Prefill throughput:** Tokens/sec for prompt processing
- **Decode throughput:** Tokens/sec for generation

**Parameters:**
- `sequence_length`: Input sequence length
- `batch_size`: Batch size
- `num_iterations`: Benchmark iterations

**CLI Flags:**
- `--batch-size`: Batch size for benchmarking
- `--seq-len`: Sequence length for benchmarking
- `--num-iters`: Number of iterations

---

### 7. VERSIONING COMPONENTS

#### 7.1 ModelRegistry (`utils.versioning.ModelRegistry`)
**Description:** Model versioning, registry, and release management

**Features:**
- Semantic versioning (major.minor.patch)
- Model metadata tracking
- Release packaging
- Stage promotion (experimental → production)
- Checksum verification

**Operations:**
- `register_model()`: Register new model version
- `get_model()`: Retrieve model by name/version
- `create_release()`: Create official release package
- `promote_model()`: Promote to new stage
- `archive_model()`: Archive old versions

**CLI Usage:**
```bash
ntf register-model --name my-model --version 1.0.0 --path ./outputs/final
ntf list-models
ntf promote-model --name my-model --version 1.0.0 --stage production
```

**CLI Flags:**
- `--name`: Model name
- `--version`: Semantic version
- `--stage`: Model stage

---

#### 7.2 ModelVersion (`utils.versioning.ModelVersion`)
**Description:** Semantic versioning for model releases

**Format:** `MAJOR.MINOR.PATCH`
- **MAJOR:** Breaking changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

---

### 8. DATA COMPONENTS

#### 8.1 TrainingDataset (`training.data.create_training_dataset`)
**Description:** Dataset creation and tokenization utilities

**Features:**
- Text tokenization
- Sequence grouping with stride
- Support for EthioBBPE and HF tokenizers

**Parameters:**
- `texts`: List of text strings
- `tokenizer`: Tokenizer instance
- `max_length`: Maximum sequence length
- `stride`: Stride for chunking long texts

**CLI Flags:**
- `--train-path`: Path to training data
- `--val-path`: Path to validation data
- `--max-length`: Maximum sequence length
- `--tokenizer-name`: Tokenizer to use

---

## Quick Start Guide

### Installation
```bash
pip install -e .
```

### Pre-training
```bash
ntf train --config pretrain_small
```

### Fine-tuning with LoRA
```bash
ntf finetune --config finetune_lora.yaml
```

### RLHF Alignment
```bash
ntf align --config dpo_alignment.yaml --method dpo
```

### Evaluation
```bash
ntf evaluate --model-path ./outputs/final --metrics perplexity accuracy
```

---

## Configuration Files

Available configurations in `/workspace/configs/`:
- `pretrain_small.yaml`: Small model pre-training
- `finetune_lora.yaml`: LoRA fine-tuning
- `dpo_alignment.yaml`: DPO alignment
- `continual_ewc.yaml`: Continual learning with EWC

---

## License

Nexuss Transformer Framework - Professional LLM Training System
