# Nexuss AI Training Framework - Complete End-to-End Tutorials

## Welcome to Nexuss AI Engineering Tutorials

This comprehensive tutorial series guides you through the complete lifecycle of training large language models (LLMs) from scratch to production deployment. Whether you're a beginner taking your first steps in AI or an advanced engineer scaling to production, these tutorials provide practical, hands-on knowledge.

---

## 📖 Interactive Documentation

For the best reading experience, open our **beautiful dark-mode interactive documentation**:

👉 **[Open Interactive Docs](./docs/index.html)**

Features:
- ✨ Animated background effects with gradient orbs
- 🌲 Tree-style navigation sidebar
- ⬅️➡️ Previous/Next navigation buttons
- 🎨 Optimized color scheme for focused reading
- 📱 Responsive design for all devices

---

## 📚 Table of Contents

| Tutorial | Title | Level | Focus Area |
|----------|-------|-------|------------|
| 🌱 | [Blank Slate Basics](./blank-slate-basics.md) | Beginner | Understanding blank-slate models from random initialization |
| 🚀 | [First Training Run](./first-training-run.md) | Intermediate | Your first complete model training walkthrough |
| ⚡ | [PEFT and LoRA](./peft-lora.md) | Intermediate | Parameter-efficient fine-tuning techniques |
| 📦 | [Versioning and Release](./versioning-release.md) | Advanced | Model versioning, semantic versioning, and release management |
| 🔄 | [Continual Learning](./continual-learning.md) | Advanced | Strategies for continual learning without catastrophic forgetting |
| 🏗️ | [Scaling to Production](./scaling-production.md) | Advanced | Distributed training and production deployment |

---

## 🎯 Learning Paths

### Path 1: Beginner Foundation
Perfect for those new to LLM training. You'll learn:
- What is a blank-slate model
- How transformers work
- Running your first training

**Start with:** [Blank Slate Basics](./blank-slate-basics.md) → [First Training Run](./first-training-run.md)

### Path 2: Practical Training
For practitioners ready to train models:
- Running efficient training jobs
- Optimizing training performance
- Fine-tuning techniques (full and PEFT/LoRA)

**Continue with:** [PEFT and LoRA](./peft-lora.md)

### Path 3: Production Excellence
Master production deployment:
- Model versioning and release management
- Continual learning strategies
- Scaling to production

**Advanced topics:** [Versioning and Release](./versioning-release.md) → [Continual Learning](./continual-learning.md) → [Scaling to Production](./scaling-production.md)

### Path 3: Advanced Alignment (Tutorials 009-011)
Master model alignment:
- Reward modeling
- RLHF with PPO
- DPO for simpler alignment

### Path 4: Production Excellence (Tutorials 012-015)
Production-ready deployment:
- Validation and testing
- Version management
- Continual learning
- Scaling strategies

---

## 🏗️ Framework Architecture

The Nexuss AI Training Framework consists of these core components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Nexuss AI Framework                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Models    │  │   Training   │  │  Fine-tuning │       │
│  │              │  │              │  │              │       │
│  │ • Transformer│  │ • Trainer    │  │ • Full FT    │       │
│  │ • Config     │  │ • Data       │  │ • PEFT/LoRA  │       │
│  │ • Arch       │  │ • Checkpoint │  │ • Freezing   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Reward    │  │    Utils     │  │   Metrics    │       │
│  │              │  │              │  │              │       │
│  │ • PPO        │  │ • Versioning │  │ • Perplexity │       │
│  │ • DPO        │  │ • Continual  │  │ • Accuracy   │       │
│  │ • RM         │  │ • Registry   │  │ • Benchmarks │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Quick Reference

### Key Concepts Glossary

| Term | Definition |
|------|------------|
| **Blank-Slate Model** | A model trained from random initialization without pre-training |
| **Pre-training** | Training on large corpus to learn general language patterns |
| **Fine-tuning** | Adapting a pre-trained model to specific tasks |
| **PEFT** | Parameter-Efficient Fine-Tuning (e.g., LoRA) |
| **LoRA** | Low-Rank Adaptation - efficient fine-tuning method |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **PPO** | Proximal Policy Optimization - RL algorithm |
| **DPO** | Direct Preference Optimization - alternative to PPO |
| **Continual Learning** | Learning new tasks without forgetting old ones |

### Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run small model training
python examples/train_small.py

# Fine-tune with LoRA
python examples/finetune_lora.py

# Evaluate model
python examples/evaluate.py
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (minimum), 32GB+ recommended

### Installation Steps

```bash
# Clone repository
git clone https://github.com/nexuss-ai/framework.git
cd framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import nexuss; print(nexuss.__version__)"
```

### Hardware Requirements by Model Size

| Model Size | Parameters | VRAM (Training) | VRAM (Inference) | Disk Space |
|------------|------------|-----------------|------------------|------------|
| Small | ~60M | 8-16 GB | 1-2 GB | 500 MB |
| Medium | ~350M | 16-32 GB | 2-4 GB | 2 GB |
| Large | ~1.5B | 32-80 GB | 4-8 GB | 10 GB |
| XL | ~7B | 80+ GB (multi-GPU) | 8-16 GB | 50 GB |

*Note: Actual requirements vary based on batch size, sequence length, and optimization techniques.*

---

## 📖 How to Use These Tutorials

1. **Start at your level**: Beginners should start from Blank Slate Basics. Experienced practitioners can jump to relevant sections.

2. **Follow sequentially**: Each tutorial builds on previous concepts.

3. **Run the examples**: Each tutorial includes executable code examples. Run them to reinforce learning.

4. **Experiment**: Modify hyperparameters and observe effects. This is crucial for deep understanding.

5. **Use the interactive docs**: For the best reading experience, use our [interactive documentation](./docs/index.html) with navigation buttons and tree view.

---

## 🎓 About Nexuss AI

Nexuss AI is committed to democratizing large language model development. Our framework provides:

- **Transparency**: All code is open and well-documented
- **Flexibility**: Support for various model sizes and training strategies
- **Scalability**: From single-GPU experiments to multi-node clusters
- **Best Practices**: Industry-standard techniques built-in

---

## 📞 Support & Community

- **Documentation**: Available at docs.nexuss.ai
- **GitHub Issues**: Report bugs and request features
- **Discord**: Join our community server for discussions
- **Email**: support@nexuss.ai for enterprise inquiries

---

## 📄 License

This framework and tutorials are released under the Apache 2.0 License. See LICENSE file for details.

---

**Ready to begin?** Start with [Blank Slate Basics](./blank-slate-basics.md) or open the [Interactive Documentation](./docs/index.html)

*Last Updated: 2024*  
*Nexuss AI Engineering Team*
