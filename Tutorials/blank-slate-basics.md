# Tutorial 001: Blank Slate Basics - Understanding LLMs from Scratch

## 📌 Overview

**Level**: Beginner  
**Duration**: 30-45 minutes  
**Prerequisites**: Basic Python knowledge

In this tutorial, you'll learn what a "blank-slate" model is and why starting from scratch matters in modern AI development.

---

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Understand what a blank-slate model is
- Know when to train from scratch vs. fine-tune
- Learn the key components of an LLM
- Set up your development environment
- Create your first model configuration

---

## 1. What is a Blank-Slate Model?

### Definition

A **blank-slate model** (also called "tabula rasa") is a neural network initialized with random weights and trained from absolute zero—without any pre-existing knowledge or pre-training on large datasets.

### Analogy

Think of it like teaching a child:
- **Blank-slate**: A newborn learning language from scratch
- **Pre-trained model**: An adult who already knows general language, learning a specific dialect
- **Fine-tuning**: A specialist expanding their expertise in a narrow field

### When to Use Blank-Slate Training

| Scenario | Approach | Reason |
|----------|----------|--------|
| New language/dialect | Blank-slate | No existing models available |
| Specialized domain (medical, legal) | Blank-slate + domain data | Avoid irrelevant knowledge |
| Research/education | Blank-slate | Full control and understanding |
| Resource constraints | Pre-trained | Faster, cheaper |
| General purpose | Pre-trained | Better performance with less data |

---

## 2. Anatomy of a Large Language Model

### Core Components

Every decoder-only transformer (like GPT-style models) consists of:

```
┌─────────────────────────────────────┐
│         Input Tokens                │
│      [The] [cat] [sat] [...]        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Token Embeddings              │
│   (Convert tokens to vectors)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Positional Encodings             │
│   (Add sequence order info)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     Transformer Block × N           │
│  ┌───────────────────────────────┐  │
│  │  Multi-Head Self-Attention    │  │
│  │  (Context understanding)      │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  Feed-Forward Network         │  │
│  │  (Feature transformation)     │  │
│  └───────────────────────────────┘  │
│  + LayerNorm + Residual Connections │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Final Layer Norm              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Output Projection             │
│   (Convert to vocabulary scores)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Next Token Prediction         │
│   [on] (with highest probability)   │
└─────────────────────────────────────┘
```

### Key Parameters Explained

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **vocab_size** | Number of unique tokens | Larger = more expressive, more memory |
| **d_model** | Embedding dimension | Larger = more capacity, slower |
| **n_heads** | Attention heads | More = better parallelization |
| **n_layers** | Transformer blocks | More = deeper understanding, slower |
| **max_seq_len** | Maximum context length | Longer = more context, quadratic memory |
| **d_ff** | Feed-forward dimension | Usually 4× d_model |

---

## 3. Setting Up Your Environment

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /workspace

# Install required packages
pip install torch numpy tqdm
```

### Step 2: Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 4. Creating Your First Model Configuration

### Understanding NTFConfig

The `NTFConfig` class holds all hyperparameters for your model. Let's explore it:

```python
from models.config import NTFConfig

# Create a small model configuration
config = NTFConfig.small()

print(f"Vocabulary size: {config.vocab_size}")
print(f"Embedding dimension: {config.d_model}")
print(f"Number of layers: {config.n_layers}")
print(f"Number of attention heads: {config.n_heads}")
print(f"Maximum sequence length: {config.max_seq_len}")
print(f"Feed-forward dimension: {config.d_ff}")
```

### Expected Output

```
Vocabulary size: 50257
Embedding dimension: 512
Number of layers: 6
Number of attention heads: 8
Maximum sequence length: 1024
Feed-forward dimension: 2048
```

### Calculating Model Size

Let's calculate approximate parameter count:

```python
def estimate_parameters(config):
    """Estimate total parameters in a transformer model."""
    
    # Embeddings
    embed_params = config.vocab_size * config.d_model
    
    # Each transformer layer
    per_layer_params = (
        # Self-attention (Q, K, V, O projections)
        4 * (config.d_model * config.d_model) +
        # Feed-forward (two linear layers)
        2 * (config.d_model * config.d_ff) +
        # Layer norms
        4 * config.d_model
    )
    
    # All layers
    transformer_params = per_layer_params * config.n_layers
    
    # Final norm and output projection
    final_params = config.d_model + (config.d_model * config.vocab_size)
    
    total = embed_params + transformer_params + final_params
    return total

# Calculate for small config
params = estimate_parameters(config)
print(f"Estimated parameters: {params:,} ({params/1e6:.2f}M)")
```

### Output

```
Estimated parameters: 63,158,784 (63.16M)
```

---

## 5. Custom Configurations

### Create Your Own Model Size

```python
# Tiny model for quick experiments (~8M params)
tiny_config = NTFConfig(
    vocab_size=1000,      # Small vocabulary for character-level
    d_model=128,
    n_heads=4,
    n_layers=4,
    max_seq_len=256,
    d_ff=512,
)

# Medium model (~350M params)
medium_config = NTFConfig.medium()

# Large model (~1.5B params)
large_config = NTFConfig.large()

print("Tiny model parameters:", estimate_parameters(tiny_config))
print("Medium model parameters:", estimate_parameters(medium_config))
print("Large model parameters:", estimate_parameters(large_config))
```

### Configuration Best Practices

| Model Size | Use Case | Training Time | Data Required |
|------------|----------|---------------|---------------|
| Tiny (<10M) | Testing, education | Minutes | Thousands of samples |
| Small (50-100M) | Prototyping, simple tasks | Hours | Millions of samples |
| Medium (300-500M) | Production, specialized | Days | Hundreds of millions |
| Large (1B+) | General purpose | Weeks+ | Billions of samples |

---

## 6. Initializing the Model

### Creating the Model Instance

```python
from models.transformer import NexussTransformer

# Initialize model with config
model = NexussTransformer(config)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model device: {next(model.parameters()).device}")
```

### Inspecting Model Architecture

```python
# Print model summary
print(model)

# Count actual parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

---

## 7. Forward Pass Example

### Understanding Model Input/Output

```python
import torch

# Create dummy input (batch_size=2, seq_len=10)
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

print(f"Input shape: {input_ids.shape}")
print(f"Input tokens: {input_ids}")

# Forward pass
with torch.no_grad():  # No gradient computation for inference
    outputs = model(input_ids=input_ids)

# Examine outputs
print(f"\nOutput type: {type(outputs)}")
print(f"Logits shape: {outputs.logits.shape}")

# Get next token predictions
next_token_logits = outputs.logits[:, -1, :]  # Last position
next_tokens = next_token_logits.argmax(dim=-1)

print(f"Predicted next tokens: {next_tokens}")
print(f"Prediction probabilities shape: {next_token_logits.shape}")
```

### Understanding the Output

```
Input shape: torch.Size([2, 10])
Logits shape: torch.Size([2, 10, 50257])
Predicted next tokens: tensor([1234, 5678])
```

**Key insight**: For each position in the sequence, the model outputs scores for every word in the vocabulary (50,257 in this case).

---

## 8. Hands-On Exercise

### Exercise 1: Create Different Model Sizes

```python
# TODO: Create three configurations
# 1. A model with ~5M parameters
# 2. A model with ~100M parameters  
# 3. A model optimized for long sequences (4096+)

# Hint: Adjust d_model, n_layers, n_heads, and d_ff
```

### Exercise 2: Experiment with Architecture

```python
# TODO: Modify these and observe effects on parameter count:
# - Change activation function to 'swiglu'
# - Enable sliding window attention
# - Disable tied embeddings

custom_config = NTFConfig(
    vocab_size=50257,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=1024,
    activation='gelu',  # Try 'swiglu'
    tie_word_embeddings=True,  # Try False
    # Add your modifications here
)

custom_model = NexussTransformer(custom_config)
print("Custom model parameters:", sum(p.numel() for p in custom_model.parameters()))
```

---

## 9. Common Pitfalls & Solutions

### Problem 1: Out of Memory (OOM)

**Symptom**: CUDA out of memory error

**Solutions**:
- Reduce batch size
- Decrease sequence length
- Use gradient checkpointing
- Choose smaller model

```python
# Enable gradient checkpointing for memory efficiency
config.gradient_checkpointing = True
```

### Problem 2: Slow Training

**Symptom**: Training is extremely slow

**Solutions**:
- Use mixed precision (FP16/BF16)
- Increase batch size if memory allows
- Use gradient accumulation
- Ensure GPU utilization

### Problem 3: Model Not Learning

**Symptom**: Loss doesn't decrease

**Check**:
- Learning rate (try 1e-4 to 1e-3 for small models)
- Data quality and preprocessing
- Gradient flow (check for NaN/Inf)
- Proper weight initialization

---

## 10. Summary & Key Takeaways

### What You Learned

✅ **Blank-slate models** start from random initialization  
✅ **When to use**: New domains, research, full control  
✅ **Core components**: Embeddings, attention, feed-forward, normalization  
✅ **Configuration**: Control model size via d_model, n_layers, n_heads  
✅ **Parameter estimation**: Calculate before training  

### Quick Reference

```python
# Standard workflow
from models.config import NTFConfig
from models.transformer import NexussTransformer

# 1. Choose configuration
config = NTFConfig.small()

# 2. Create model
model = NexussTransformer(config)

# 3. Move to device
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# 4. Ready for training!
```

---

## 📚 Additional Resources

### Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

### Next Steps
- **Tutorial 002**: Deep dive into transformer architecture
- **Tutorial 003**: Prepare your training dataset

---

## 🎓 Knowledge Check

1. **What is a blank-slate model?**
   - A model trained from random initialization without pre-training

2. **When would you choose blank-slate over fine-tuning?**
   - New languages, specialized domains, research purposes

3. **Which parameter most affects model capacity?**
   - d_model (embedding dimension) and n_layers (depth)

4. **What does the output logits shape [2, 10, 50257] mean?**
   - Batch of 2, sequence length 10, vocabulary size 50,257

---

**Congratulations!** You've completed Tutorial 001. You now understand the fundamentals of blank-slate LLMs and can configure your own models.

➡️ **Next**: [Tutorial 002: Model Architecture Deep Dive](./002_model_architecture.md)
