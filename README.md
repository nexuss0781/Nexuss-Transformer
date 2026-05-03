# 🌌 Nexuss Transformer Framework (NTF)

> **From Blank Slate to Superintelligence.**  
> A production-grade, end-to-end framework for training, fine-tuning, and aligning Decoder-Only Large Language Models from scratch.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-f9d84b?logo=huggingface)](https://huggingface.co/Nexuss0781/Nexuss-Transformer)
[![EthioBBPE](https://img.shields.io/badge/🇪🇹-EthioBBPE-9370db?logo=python)](https://github.com/nexuss0781/Ethio_BBPE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/nexuss0781/Nexuss-Transformer)

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture) • [Tokenizer](#-native-ethiopian-tokenization) • [HuggingFace model](https://huggingface.co/Nexuss0781/Nexuss-Transformer)

**📚 Full documentation available at [Nexuss-Transformer.gt.tc](https://Nexuss-Transformer.gt.tc)**

</div>

---

## 🚀 Overview

**Nexuss Transformer Framework (NTF)** is a comprehensive architectural solution designed for researchers and engineers who need full control over the LLM lifecycle. Unlike wrapper libraries, NTF provides a **blank-slate implementation** of the transformer architecture while leveraging battle-tested backends like Hugging Face `accelerate`, `peft`, and `trl` for distributed training and optimization.

Whether you are pre-training a foundation model on terabytes of text, performing parameter-efficient fine-tuning (LoRA), or aligning models via RLHF (PPO/DPO), NTF offers a unified, modular, and scalable interface.

### 🔹 Native Ethiopian Tokenization

NTF seamlessly integrates with **[EthioBBPE](https://github.com/nexuss0781/Ethio_BBPE)**, a specialized Byte-Pair Encoding tokenizer optimized for Amharic and Ge'ez scripts. This integration ensures superior compression ratios and cultural nuance preservation for Ethiopian languages right out of the box.

```bash
pip install ethiobbpe  # Optional: For enhanced Ethiopian language support
```

## ✨ Key Features

### 🏗️ Core Architecture
- **Pure Decoder-Only Stack**: Custom implementation featuring RoPE embeddings, RMSNorm, SwiGLU activations, and Multi-Query Attention (MQA).
- **Blank Slate Training**: Build knowledge from scratch with no pretrained weights bias.
- **Gradient Checkpointing**: Optimized memory usage for training massive models on limited hardware.
- **Flash Attention Ready**: Seamless integration with flash-attn for 2-3x speedups.

### 🎛️ Advanced Fine-Tuning
- **Full Fine-Tuning**: Discriminative learning rates, layer-wise decay, and gradual unfreezing strategies.
- **PEFT & LoRA**: Low-Rank Adaptation support via `peft` for efficient downstream task adaptation.
- **Flexible Freezing**: Freeze top-k, bottom-k, or alternating layers to preserve base knowledge.

### 🧠 Alignment & RLHF
- **Reward Modeling**: Train reward models to score model outputs.
- **PPO & DPO**: Integrated Proximal Policy Optimization and Direct Preference Optimization using `trl`.
- **Human Feedback Loop**: End-to-end pipeline for RLHF alignment.

### 🔄 Continual Learning
- **Catastrophic Forgetting Prevention**: Built-in Elastic Weight Consolidation (EWC), Experience Replay, and Gradient Episodic Memory (GEM).
- **Lifelong Learning**: Seamlessly continue training on new domains without losing previous capabilities.

### 🛠️ Production Engineering
- **Distributed Training**: Native support for DeepSpeed ZeRO and FSDP via `accelerate`.
- **Smart Checkpointing**: Versioned saves, auto-resume, and "best model" tracking based on validation loss.
- **Model Versioning**: Semantic versioning system for releasing and rolling back model iterations.

---

## ⚡ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nexuss0781/Nexuss-Transformer.git
cd Nexuss-Transformer

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### 1. Pre-Training from Scratch

Initialize a blank slate model and train on your corpus:

```python
from ntf.models import TransformerConfig, TransformerModel
from ntf.training import Trainer, TrainingConfig

# Define Model Architecture (Small Preset)
config = TransformerConfig.from_preset("small") 
model = TransformerModel(config)

# Configure Training
train_config = TrainingConfig(
    output_dir="./outputs/my-first-llm",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    num_train_epochs=10,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=train_config,
    train_dataset=my_tokenized_dataset,
    eval_dataset=my_val_dataset
)

# Launch Training
trainer.train()
```

### 2. Parameter-Efficient Fine-Tuning (LoRA)

Adapt your pretrained model to a specific task (e.g., Medical QA) with minimal resources:

```python
from ntf.finetuning import PeftFinetuner

finetuner = PeftFinetuner(
    model_path="./outputs/my-first-llm/checkpoint-5000",
    lora_r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"]
)

finetuner.train(dataset=medical_qa_dataset)
finetuner.save_adapter("./adapters/medical-lora")
```

### 3. RLHF Alignment (DPO)

Align your model with human preferences:

```python
from ntf.reward import DPOTrainer

dpo = DPOTrainer(
    model_ref="./outputs/my-first-llm",
    beta=0.1,
    loss_type="sigmoid"
)

dpo.train(preference_dataset=human_prefs)
```

---

## 📚 Documentation

### 🌐 Interactive Tutorials
For a step-by-step learning experience, check out our **Interactive Documentation**:
- **🔗 URL**: Open `Tutorials/index.html` in your browser or visit [Nexuss-Transformer.gt.tc](https://Nexuss-Transformer.gt.tc)
- **📖 Content**: comprehensive modules from blank slate to production deployment
- **✨ Features**: Search, filtering by difficulty, dark mode, and syntax highlighting

### 📄 Technical Documentation
| Module | Description |
| :--- | :--- |
| **[TRAINING.md](./TRAINING.md)** | **Comprehensive End-to-End Guide**. Covers installation, configuration, pre-training, fine-tuning, RLHF, and continual learning strategies. |
| **[Tutorials/README.md](./Tutorials/README.md)** | **Tutorial Series Index**. Overview of all 14 incremental learning modules. |
| `models/` | Core Transformer architecture, configurations, and tokenization utilities. |
| `training/` | Distributed trainer, checkpoint manager, and data collators. |
| `finetuning/` | Full fine-tuning, LoRA/PEFT wrappers, and layer freezing logic. |
| `reward/` | Reward modeling and RLHF algorithms (PPO, DPO). |
| `utils/` | Continual learning helpers, metrics, and model versioning tools. |

---

## 🏛️ Architecture Diagram

```mermaid
graph TD
    A[Raw Text Data] --> B(Tokenization)
    B --> C{Training Mode}
    
    C -->|Blank Slate| D[Pre-Training Loop]
    C -->|Adaptation| E[Fine-Tuning Full/LoRA]
    C -->|Alignment| F[RLHF PPO/DPO]
    
    D --> G[Base Model Weights]
    E --> G
    F --> H[Aligned Model]
    
    G --> I[Checkpoint Manager]
    H --> I
    
    I --> J[Model Registry / Versioning]
    J --> K[Inference / Deployment]
    
    subgraph Optimization
    L[Gradient Checkpointing]
    M[Mixed Precision FP16/BF16]
    N[DeepSpeed ZeRO]
    end
    
    D -.-> L
    D -.-> M
    D -.-> N
```

---

## 🛡️ Continual Learning Strategies

NTF includes specialized mechanisms to prevent **Catastrophic Forgetting** when updating models with new data:

1.  **Elastic Weight Consolidation (EWC)**: Penalizes changes to weights important for previous tasks.
2.  **Experience Replay**: Interleaves new training batches with stored samples from previous distributions.
3.  **Gradual Unfreezing**: Slowly unfreezes layers from top to bottom to stabilize representations.
4.  **Learning Rate Decay**: Applies layer-wise decay to protect lower-level features.

---

## 📦 Model Zoo & Presets

NTF comes with predefined architectural presets ready for instantiation:

| Preset | Parameters | Layers | Heads | Dim | Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `nano` | ~10M | 4 | 4 | 128 | Testing / Debugging |
| `small` | ~60M | 6 | 8 | 256 | Edge Devices / Mobile |
| `medium` | ~350M | 12 | 12 | 512 | Research / Specialized Tasks |
| `large` | ~1.2B | 24 | 16 | 768 | General Purpose Foundation |
| `xl` | ~3.5B | 32 | 20 | 1024 | High-Performance Applications |

---

## 🤝 Contributing

We welcome contributions from the community! Whether it's optimizing kernels, adding new RLHF algorithms, or improving documentation, please check our [Contributing Guidelines](CONTRIBUTING.md).

1.  Fork the repo.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built upon the shoulders of giants:
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [Hugging Face TRL](https://huggingface.co/docs/trl)
- [PyTorch](https://pytorch.org/)

---

<div align="center">

**Nexuss Transformer Framework**  
*Empowering the next generation of Language Models.*

[GitHub](https://github.com/nexuss0781/Nexuss-Transformer) • [Issues](https://github.com/nexuss0781/Nexuss-Transformer/issues) • [Discussions](https://github.com/nexuss0781/Nexuss-Transformer/discussions)

</div>
