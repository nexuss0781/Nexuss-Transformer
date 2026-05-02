# LLM Framework - Blank Slate Transformer Training System

A comprehensive, production-ready framework for training decoder-only transformer models from scratch, with full support for fine-tuning, RLHF, and continual learning.

## Architecture Overview

```
llm_framework/
├── models/                 # Model architectures
│   ├── __init__.py
│   ├── config.py          # Model configuration classes
│   ├── transformer.py     # Core decoder-only transformer
│   └── model.py           # Main model wrapper
├── training/              # Pre-training infrastructure
│   ├── __init__.py
│   ├── trainer.py         # Main training loop
│   ├── checkpoint.py      # Checkpoint management
│   └── data.py            # Data loading & preprocessing
├── finetuning/            # Fine-tuning modules
│   ├── __init__.py
│   ├── full_finetune.py   # Full parameter fine-tuning
│   ├── peft_finetune.py   # PEFT/LoRA fine-tuning
│   └── freeze.py          # Layer freezing utilities
├── reward/                # Reward modeling & RLHF
│   ├── __init__.py
│   ├── reward_model.py    # Reward model architecture
│   └── rlhf.py            # PPO/DPO implementations
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── continual_learning.py  # EWC, replay buffers
│   ├── versioning.py      # Model versioning
│   └── metrics.py         # Evaluation metrics
├── configs/               # Configuration files
│   ├── base_config.yaml
│   ├── finetune_config.yaml
│   └── rlhf_config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Features

### Core Architecture
- **Blank Slate Decoder-Only Transformer**: Pure decoder architecture optimized for autoregressive language modeling
- **Hugging Face Integration**: Built on `transformers`, `accelerate`, `peft`, and `trl` libraries
- **Scalable Training**: Distributed training with gradient checkpointing, mixed precision, and batch optimization

### Training Capabilities
- **Pre-training**: Full training from scratch with configurable hyperparameters
- **Fine-tuning**: 
  - Full fine-tuning (all parameters)
  - Parameter-efficient fine-tuning (LoRA, AdaLoRA, QLoRA)
  - Selective layer freezing/unfreezing
- **Checkpoint Management**: Save/restore with versioning, automatic recovery

### Advanced Features
- **RLHF Pipeline**: Complete reward modeling + PPO/DPO reinforcement learning
- **Continual Learning**: 
  - Elastic Weight Consolidation (EWC)
  - Experience replay buffers
  - Progressive networks
  - Gradient episodic memory
- **Version Control**: Semantic versioning for model releases, branch-and-merge workflows
- **Validation**: In-training validation, next-token prediction evaluation

## Quick Start

```python
from llm_framework.models import TransformerConfig, DecoderOnlyTransformer
from llm_framework.training import Trainer, TrainingConfig
from llm_framework.finetuning import LoRAConfig, PEFTTrainer
from llm_framework.reward import RewardModel, PPOTrainer

# 1. Create blank slate model
config = TransformerConfig(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    max_seq_len=2048,
    dropout=0.1
)
model = DecoderOnlyTransformer(config)

# 2. Pre-train from scratch
train_config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    warmup_steps=1000,
    checkpoint_every=1000,
    use_gradient_checkpointing=True,
    mixed_precision="bf16"
)
trainer = Trainer(model, train_config)
trainer.train(train_dataset)

# 3. Fine-tune with LoRA
lora_config = LoRAConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"])
peft_trainer = PEFTTrainer(model, lora_config)
peft_trainer.train(finetune_dataset)

# 4. RLHF Pipeline
reward_model = RewardModel.from_pretrained("path/to/reward")
ppo_trainer = PPOTrainer(model, reward_model)
ppo_trainer.train(rl_dataset)
```

## Installation

```bash
pip install -e .
```

## License

MIT License
