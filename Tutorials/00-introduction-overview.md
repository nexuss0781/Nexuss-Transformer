# Nexuss AI Training Framework - Complete Guide

## Welcome to End-to-End Model Training

This is your comprehensive guide to training, fine-tuning, and deploying transformer models using the Nexuss Transformer Framework (NTF). Whether you're a beginner taking your first steps in AI or an advanced engineer scaling to production, this tutorial series will guide you through every stage.

---

## Table of Contents

### Foundation Track (Beginner)
1. [Introduction & Overview](00-introduction-overview.md) ← You are here
2. [Understanding Blank Slate Models](01-blank-slate-models.md)
3. [Your First Training Run](02-first-training-run.md)
4. [Data Preparation & Tokenization](03-data-preparation.md)

### Intermediate Track
5. [Full Fine-Tuning Fundamentals](04-full-finetuning.md)
6. [Parameter-Efficient Fine-Tuning (PEFT/LoRA)](05-peft-lora.md)
7. [Layer Freezing Strategies](06-layer-freezing.md)
8. [Evaluation & Validation](07-evaluation-validation.md)

### Advanced Track
9. [Reinforcement Learning from Human Feedback (RLHF)](08-rlhf-intro.md)
10. [Reward Model Training](09-reward-model-training.md)
11. [PPO (Proximal Policy Optimization)](10-ppo-training.md)
12. [DPO (Direct Preference Optimization)](11-dpo-training.md)
13. [Version Control & Release Management](12-versioning-release.md)
14. [Continual Learning](13-continual-learning.md)
15. [Scaling to Production](14-scaling-production.md)

---

## What This Framework Provides

The Nexuss Transformer Framework is built for:

### 1. **Blank Slate Training**
Train models from scratch with custom architectures:
- Decoder-only transformers
- Rotary embeddings (RoPE)
- RMS normalization
- SwiGLU activations
- Sliding window attention

### 2. **Fine-Tuning Methods**
Multiple approaches to adapt pre-trained models:
- **Full Fine-Tuning**: Update all parameters
- **Freeze Tuning**: Selective layer freezing
- **LoRA/PEFT**: Parameter-efficient adapters (1-10% trainable params)

### 3. **Alignment & RLHF**
Make models helpful and safe:
- Reward modeling
- PPO (Proximal Policy Optimization)
- DPO (Direct Preference Optimization)

### 4. **Production Features**
- Distributed training (multi-GPU, multi-node)
- Mixed precision (FP16/BF16)
- Gradient checkpointing
- Checkpoint management
- Version control
- Continual learning

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NEXUSS TRAINING FRAMEWORK                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   MODELS     │  │   TRAINING   │  │  FINETUNING  │      │
│  │              │  │              │  │              │      │
│  │ • Transformer│  │ • Trainer    │  │ • Full FT    │      │
│  │ • Config     │  │ • Config     │  │ • PEFT/LoRA  │      │
│  │ • Embeddings │  │ • Checkpoint │  │ • Freeze     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    REWARD    │  │    DATA      │  │    UTILS     │      │
│  │              │  │              │  │              │      │
│  │ • Reward Net │  │ • Tokenizer  │  │ • Logging    │      │
│  │ • PPO        │  │ • Collators  │  │ • Metrics    │      │
│  │ • DPO        │  │ • Datasets   │  │ • Profiling  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start: Your First Model in 5 Steps

### Step 1: Install Dependencies
```bash
pip install torch accelerate transformers peft trl
```

### Step 2: Prepare Your Data
```jsonl
{"text": "Your training text goes here..."}
{"text": "More training data..."}
```

### Step 3: Configure Training
```python
from training.config import TrainingConfig

config = TrainingConfig.small_model()
config.output_dir = "./outputs/my-first-model"
config.num_train_epochs = 3
```

### Step 4: Train
```python
from training.trainer import Trainer
from models.transformer import NexussTransformer
from models.config import NTFConfig

# Create model from scratch
model_config = NTFConfig(vocab_size=16000, d_model=768, n_layers=12, n_heads=12)
model = NexussTransformer(model_config)

# Initialize trainer
trainer = Trainer(model=model, config=config, train_dataset=dataset)

# Train!
metrics = trainer.train()
```

### Step 5: Generate Text
```python
input_ids = tokenizer.encode("Hello, I am", return_tensors="pt")
output = model.generate(input_ids, max_length=100)
print(tokenizer.decode(output[0]))
```

---

## Understanding the Training Lifecycle

Every model goes through these stages:

```
1. BLANK SLATE          → Random weights, no knowledge
         ↓
2. PRE-TRAINING         → Learn language patterns from large corpus
         ↓
3. SUPERVISED FINE-TUNE → Learn specific tasks/instructions
         ↓
4. ALIGNMENT (RLHF)     → Align with human preferences
         ↓
5. EVALUATION           → Test on held-out data
         ↓
6. VERSION & RELEASE    → Tag, freeze, deploy
         ↓
7. CONTINUAL LEARNING   → Adapt to new data/tasks
```

Each stage has specific techniques covered in subsequent tutorials.

---

## Key Concepts Preview

### What is a "Blank Slate" Model?
A blank slate model has randomly initialized weights—it knows nothing. Pre-training teaches it language patterns, grammar, facts, and reasoning from raw text.

### Why Fine-Tune?
Pre-trained models are generalists. Fine-tuning makes them specialists for your specific use case (medical, legal, code, etc.).

### What is LoRA?
Low-Rank Adaptation (LoRA) adds tiny trainable "adapters" to a frozen model. Instead of updating billions of parameters, you update less than 1%, saving memory and time.

### What is RLHF?
Reinforcement Learning from Human Feedback aligns models with human values. It involves:
1. Training a reward model on human preferences
2. Using PPO or DPO to optimize the model for higher rewards

### What is Continual Learning?
Continual learning lets models learn new tasks without forgetting old ones—critical for real-world deployment where data evolves.

---

## Hardware Requirements

| Model Size | Parameters | Minimum RAM | Recommended GPU | Training Time* |
|------------|------------|-------------|-----------------|----------------|
| Small      | ~60M       | 8 GB        | 1x RTX 3090     | 6-12 hours     |
| Medium     | ~350M      | 16 GB       | 2x RTX 4090     | 1-2 days       |
| Large      | ~1B        | 32 GB       | 4x A100         | 3-5 days       |
| XL         | ~7B        | 80 GB+      | 8x A100         | 1-2 weeks      |

*Estimates vary based on dataset size and configuration

---

## How to Use This Tutorial Series

### For Beginners
Start from Tutorial 1 and follow sequentially. Don't skip—each builds on previous concepts.

### For Intermediate Users
Jump to Section 5 (Fine-Tuning) if you understand basic training. Review earlier sections as needed.

### For Advanced Users
Go straight to Sections 9-15 for RLHF, production scaling, and continual learning.

---

## Support & Resources

- **Code Examples**: Every tutorial includes working code
- **Config Templates**: See `/configs/` for YAML configurations
- **API Reference**: Docstrings in source code
- **Community**: Issues and discussions on GitHub

---

## Next Steps

Ready to begin? Continue to:

**[Tutorial 01: Understanding Blank Slate Models](01-blank-slate-models.md)**

You'll learn:
- What "blank slate" really means
- How random initialization works
- The transformer architecture from scratch
- Why we train from blank slate vs. using pre-trained models

---

*Nexuss AI Company - Training Division*
*Building the future of language models, one tutorial at a time.*
