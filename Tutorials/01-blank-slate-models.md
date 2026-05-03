# Tutorial 01: Understanding Blank Slate Models

## Introduction

Welcome to your first deep dive into model training! In this tutorial, we'll demystify what "blank slate" means and build a complete understanding of transformer models from the ground up.

By the end of this tutorial, you will:
- Understand what "blank slate" initialization means
- Know how transformers work at a component level
- Be able to configure a model architecture
- Understand the difference between pre-training and fine-tuning

---

## What Does "Blank Slate" Mean?

### The Concept

A **blank slate model** (also called "tabula rasa") is a neural network with **randomly initialized weights**. It has:

- ✅ An architecture (structure)
- ✅ Parameters (weights)
- ❌ Knowledge (learned patterns)
- ❌ Skills (language understanding)

Think of it like a human brain at birth—the hardware is there, but no experiences have shaped it yet.

### Visual Comparison

```
BLANK SLATE MODEL          PRE-TRAINED MODEL
┌─────────────────┐        ┌─────────────────┐
│   Architecture  │        │   Architecture  │
│   ✓             │        │   ✓             │
│   Random Weights│        │   Learned Weights│
│   ✗ Knowledge   │        │   ✓ Language    │
│   ✗ Skills      │        │   ✓ Patterns    │
└─────────────────┘        └─────────────────┘
     Starting Point            Ready to Use
```

### Why Start from Blank Slate?

| Scenario | Use Blank Slate | Use Pre-Trained |
|----------|----------------|-----------------|
| Custom architecture | ✅ Yes | ❌ No |
| Unique vocabulary | ✅ Yes | ❌ Limited |
| Low-resource language | ✅ Yes | ⚠️ If available |
| Research/experimentation | ✅ Yes | ⚠️ Maybe |
| Standard NLP task | ❌ No | ✅ Yes |
| Limited compute | ❌ No | ✅ Yes |
| Time constraints | ❌ No | ✅ Yes |

---

## Transformer Architecture Deep Dive

Our framework uses a **decoder-only transformer**—the same architecture family as GPT, LLaMA, and many others.

### High-Level Architecture

```
Input Text
    ↓
[Token Embeddings] ← Converts tokens to vectors
    ↓
[Position Embeddings] ← Adds position information (RoPE)
    ↓
┌───────────────────────────────┐
│  Transformer Block (Layer 1)  │
│  ├─ Self-Attention            │
│  └─ Feed-Forward Network      │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│  Transformer Block (Layer 2)  │
│  ├─ Self-Attention            │
│  └─ Feed-Forward Network      │
└───────────────────────────────┘
    ↓
         ... (repeats N times)
    ↓
┌───────────────────────────────┐
│  Transformer Block (Layer N)  │
└───────────────────────────────┘
    ↓
[Final Layer Norm]
    ↓
[Output Projection] → Logits over vocabulary
    ↓
Softmax → Next token probabilities
```

### Key Components Explained

#### 1. Token Embeddings

**Purpose**: Convert discrete tokens (word IDs) into continuous vectors.

```python
# Example: vocab_size=16000, embedding_dim=768
embedding_layer = nn.Embedding(16000, 768)

# Input: token ID
token_id = 42  # e.g., the word "hello"
embedding = embedding_layer(torch.tensor([token_id]))
# Output: vector of shape [768]
```

**What happens**: Each of the 16,000 tokens gets a unique 768-dimensional vector. These vectors are learned during training.

#### 2. Rotary Position Embeddings (RoPE)

**Purpose**: Tell the model where each token appears in the sequence.

Unlike older methods that added position vectors, RoPE **rotates** the query and key vectors based on position:

```python
# Simplified RoPE concept
position_0: [1, 0, 0, 1]  # No rotation
position_1: [cos(θ), -sin(θ), sin(θ), cos(θ)]  # Rotated by θ
position_2: [cos(2θ), -sin(2θ), sin(2θ), cos(2θ)]  # Rotated by 2θ
```

**Why RoPE?**
- Better extrapolation to longer sequences
- More efficient than absolute position embeddings
- Used in modern models (LLaMA, PaLM, etc.)

#### 3. Multi-Head Self-Attention

**Purpose**: Let each token "attend" to other tokens in the sequence.

```
Query (Q)     Key (K)      Value (V)
    │            │            │
    └────┬───────┘            │
         │                    │
    Compute Attention         │
    Scores (Q·K/√d)          │
         │                    │
    Softmax + Mask            │
         │                    │
         └────────┬───────────┘
                  │
           Weighted Sum of V
                  │
              Output
```

**Causal Mask**: In decoder-only models, tokens can only attend to **previous** tokens (not future ones). This ensures autoregressive generation.

```python
# Causal mask for sequence length 4
# 0 = can attend, -inf = cannot attend
mask = [
    [0, -inf, -inf, -inf],  # Position 0 sees only itself
    [0,   0, -inf, -inf],  # Position 1 sees 0 and 1
    [0,   0,   0, -inf],  # Position 2 sees 0, 1, 2
    [0,   0,   0,   0],   # Position 3 sees all
]
```

#### 4. Feed-Forward Network (FFN)

**Purpose**: Process and transform the attention output.

Our framework supports multiple activation types:

```python
# SwiGLU (recommended)
output = down_proj(silu(gate_proj(x)) * up_proj(x))

# Standard GeLU
output = dropout(geLU(linear1(x)))
output = linear2(dropout(output))
```

**SwiGLU** combines a gating mechanism with the Swish activation—used in LLaMA and other modern architectures.

#### 5. RMS Normalization

**Purpose**: Stabilize training by normalizing activations.

```python
# RMSNorm vs LayerNorm
# LayerNorm: subtracts mean, divides by std
# RMSNorm: just divides by RMS (root mean square)

def rms_norm(x, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

**Why RMSNorm?**
- Simpler and faster than LayerNorm
- Works better for large models
- Used in LLaMA, PaLM, T5

#### 6. Residual Connections

**Purpose**: Allow gradients to flow directly through layers.

```python
# Pre-normalization architecture (modern approach)
def transformer_block(x):
    # Attention with residual
    h = x + attention(rms_norm(x))
    # FFN with residual
    out = h + ffn(rms_norm(h))
    return out
```

**Pre-norm vs Post-norm**:
- **Pre-norm** (ours): Normalize before each sub-layer → better for deep models
- **Post-norm**: Normalize after each sub-layer → original transformer design

---

## Model Configuration

Let's configure a small blank slate model:

### Configuration Parameters

```python
from models.config import NTFConfig

config = NTFConfig(
    # Vocabulary
    vocab_size=16000,        # Size of tokenizer vocabulary
    
    # Architecture
    d_model=768,            # Hidden dimension (embedding size)
    n_layers=12,            # Number of transformer blocks
    n_heads=12,             # Number of attention heads
    d_ff=3072,              # Feed-forward dimension (usually 4x d_model)
    
    # Sequence
    max_seq_len=2048,       # Maximum context length
    
    # RoPE
    use_rope=True,          # Use rotary embeddings
    rope_theta=10000.0,     # RoPE base frequency
    
    # Normalization
    layer_norm_eps=1e-6,    # Epsilon for numerical stability
    
    # Dropout
    dropout=0.1,            # Embedding dropout
    attention_dropout=0.0,  # Attention dropout
    hidden_dropout=0.0,     # FFN dropout
    
    # Activation
    activation="swiglu",    # FFN activation function
    
    # Optimization
    gradient_checkpointing=False,  # Save memory, slower
    use_cache=True,         # Use KV cache for inference
    tie_word_embeddings=True,  # Share input/output embeddings
)
```

### Model Size Calculator

```python
def calculate_params(vocab_size, d_model, n_layers, n_heads, d_ff):
    # Embeddings
    embed_params = vocab_size * d_model
    
    # Per layer parameters
    # Attention: Q, K, V, O projections
    attn_params = 4 * (d_model * d_model)
    
    # FFN: gate, up, down projections (for SwiGLU)
    ffn_params = 3 * (d_model * d_ff)
    
    # Norms and misc
    norm_params = 2 * d_model  # Two RMSNorm per layer
    
    # Total per layer
    per_layer = attn_params + ffn_params + norm_params
    
    # Total
    total = embed_params + (n_layers * per_layer)
    
    return {
        "embeddings": f"{embed_params/1e6:.2f}M",
        "per_layer": f"{per_layer/1e6:.2f}M",
        "total": f"{total/1e6:.2f}M",
    }

# Our small model
params = calculate_params(
    vocab_size=16000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072
)
print(params)
# Output: {'embeddings': '12.29M', 'per_layer': '11.53M', 'total': '~150M'}
```

---

## Creating Your Blank Slate Model

### Step-by-Step Implementation

```python
import torch
from models.transformer import NexussTransformer
from models.config import NTFConfig

# 1. Define configuration
config = NTFConfig(
    vocab_size=16000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    d_ff=3072,
    max_seq_len=2048,
    use_rope=True,
    activation="swiglu",
)

# 2. Create model (random initialization!)
model = NexussTransformer(config)

# 3. Check parameter count
param_count = model.count_parameters()
print(f"Total parameters: {param_count['total_millions']}")
# Output: Total parameters: ~150M

# 4. Verify random initialization
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")
    # All should be close to mean=0, std=initializer_range (~0.02)
```

### Understanding Initialization

Our model uses **normal initialization**:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**Why this matters**:
- Too large initialization → exploding gradients
- Too small initialization → vanishing gradients
- Proper initialization → stable training from step 1

---

## Before vs After Training

### Blank Slate (Before Training)

```python
input_text = "The capital of France is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate with blank slate
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))

# Output: Gibberish or repetitive nonsense
# "The capital of France is is is is..."
```

**Why?** The model hasn't learned:
- Grammar rules
- Factual knowledge
- Coherent sentence structure
- Token relationships

### After Pre-Training

```python
# Same input after pre-training
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))

# Output: Coherent continuation
# "The capital of France is Paris, known for its rich history..."
```

**What changed?** Through training, the model learned:
- Language patterns and grammar
- Factual associations (France → Paris)
- Coherent text generation
- Contextual understanding

---

## Pre-Training vs Fine-Tuning

### Pre-Training

**Goal**: Learn general language patterns

**Data**: Large, diverse corpus (web text, books, articles)

**Objective**: Next-token prediction

```python
# Loss: Cross-entropy between predicted and actual next token
loss = cross_entropy(predicted_logits, actual_next_token)
```

**Duration**: Days to weeks on large datasets

**Result**: General-purpose language model

### Fine-Tuning

**Goal**: Adapt to specific tasks/domains

**Data**: Task-specific dataset (instructions, Q&A, code, etc.)

**Objective**: Task-specific loss

```python
# Example: Instruction tuning
instruction = "Translate to French:"
text = "Hello, how are you?"
target = "Bonjour, comment allez-vous?"

# Train model to produce target given instruction + text
```

**Duration**: Hours to days on smaller datasets

**Result**: Specialized model for your use case

---

## Practical Exercise: Inspect Your Model

Let's examine a blank slate model in detail:

```python
import torch
from models.transformer import NexussTransformer
from models.config import NTFConfig

# Create a tiny model for inspection
config = NTFConfig(
    vocab_size=1000,   # Tiny vocab for demo
    d_model=128,       # Small hidden size
    n_layers=4,        # Few layers
    n_heads=4,         # Few heads
    d_ff=256,          # Small FFN
    max_seq_len=512,
)

model = NexussTransformer(config)

# 1. Count parameters
print("=== Parameter Count ===")
param_info = model.count_parameters()
for key, value in param_info.items():
    print(f"{key}: {value}")

# 2. Examine architecture
print("\n=== Model Architecture ===")
print(model)

# 3. Check weight statistics
print("\n=== Weight Statistics ===")
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {param.mean():.6f}")
        print(f"  Std: {param.std():.6f}")
        print(f"  Min: {param.min():.6f}")
        print(f"  Max: {param.max():.6f}")

# 4. Test forward pass
print("\n=== Forward Pass ===")
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

with torch.no_grad():
    outputs = model(input_ids=input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Vocab size: {outputs.logits.shape[-1]}")
    
    # Get next token predictions
    next_token_probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
    print(f"Top 5 predictions for last token:")
    top_values, top_indices = torch.topk(next_token_probs[0], k=5)
    for prob, token_id in zip(top_values, top_indices):
        print(f"  Token {token_id.item()}: {prob.item():.4f}")
```

---

## Common Questions

### Q: How do I choose model size?

**A**: Consider these factors:

| Factor | Small Model | Large Model |
|--------|-------------|-------------|
| Dataset size | < 1GB | > 10GB |
| Compute budget | Limited | Generous |
| Latency needs | Real-time | Batch OK |
| Task complexity | Simple | Complex |
| Memory constraints | Edge devices | Cloud GPUs |

**Rule of thumb**: Model should be small enough to train in reasonable time, large enough to capture patterns in your data.

### Q: Can I use pre-trained embeddings?

**A**: Yes! You can initialize embeddings with pre-trained vectors (Word2Vec, GloVe, FastText):

```python
# Load pre-trained embeddings
pretrained_embeddings = load_word2vec("vectors.bin")

# Set in model
model.embed_tokens.weight.data.copy_(pretrained_embeddings)

# Optionally freeze
model.embed_tokens.weight.requires_grad = False
```

### Q: What if my model produces gibberish after training?

**A**: Common causes:
1. **Learning rate too high** → Reduce by 10x
2. **Not enough training** → Train longer
3. **Bad data quality** → Clean your dataset
4. **Architecture mismatch** → Verify config matches data

### Q: Should I tie word embeddings?

**A**: Tie when:
- ✅ Vocabulary is large (saves parameters)
- ✅ Model is small (regularization effect)
- ✅ Standard language modeling

Don't tie when:
- ❌ Input/output distributions differ significantly
- ❌ Using special tokens differently

---

## Summary

You now understand:

✅ **Blank slate** = randomly initialized model with no knowledge  
✅ **Transformer architecture** = embeddings + attention + FFN + norms  
✅ **Key components** = RoPE, RMSNorm, SwiGLU, causal masking  
✅ **Configuration** = choosing the right hyperparameters  
✅ **Pre-training vs fine-tuning** = general learning vs specialization  

---

## Next Steps

Ready to train your first model? Continue to:

**[Tutorial 02: Your First Training Run](02-first-training-run.md)**

You'll learn:
- Setting up the training loop
- Configuring optimization
- Monitoring training progress
- Saving and loading checkpoints
- Debugging common issues

---

## Reference: Complete Model Creation Script

```python
#!/usr/bin/env python3
"""
Create a blank slate transformer model.
Save configuration for later training.
"""

import json
from models.transformer import NexussTransformer
from models.config import NTFConfig

def create_blank_slate_model(
    vocab_size: int = 16000,
    d_model: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    d_ff: int = 3072,
    max_seq_len: int = 2048,
    save_path: str = "./models/my_model",
):
    """Create and save a blank slate model."""
    
    # Configuration
    config = NTFConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        use_rope=True,
        rope_theta=10000.0,
        activation="swiglu",
        layer_norm_eps=1e-6,
        dropout=0.1,
        gradient_checkpointing=False,
        tie_word_embeddings=True,
    )
    
    # Create model
    model = NexussTransformer(config)
    
    # Print info
    param_count = model.count_parameters()
    print(f"Created model with {param_count['total_millions']} parameters")
    print(f"Configuration saved to {save_path}/config.json")
    
    # Save
    model.save_pretrained(save_path)
    
    # Also save config as JSON for reference
    with open(f"{save_path}/config_readable.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    return model, config

if __name__ == "__main__":
    model, config = create_blank_slate_model()
    print("Model ready for training!")
```

---

*Nexuss AI Company - Training Division*
*Building understanding, one layer at a time.*
