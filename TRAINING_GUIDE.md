# Blank Slate Training - Ethiopian Religious Texts

## Overview

This training setup creates a **blank slate** (from scratch) decoder-only transformer model trained on Ethiopian Orthodox religious texts. The model will learn these datasets as its **base knowledge**, which can then be frozen for downstream tasks.

## Datasets

### 1. Synaxarium Dataset
- **Source**: Ethiopian Orthodox daily readings
- **Size**: 366 entries (one for each day of the Ethiopian calendar)
- **Language**: Amharic
- **Content**: Complete synaxarium texts for all 13 months

### 2. Canon Biblical Dataset
- **Source**: Parallel Bible corpus
- **Size**: 31,920 verses
- **Languages**: Amharic (primary) + English (optional)
- **Books**: 67 books including Deuterocanonical texts
- **Structure**: Book, Chapter, Verse format

**Combined Dataset Statistics:**
- Total samples: 32,286
- Total characters: ~5.3 million
- Average length: 165 characters per sample

## Model Architecture

```
Small Decoder-Only Transformer (~25M parameters)
├── Vocabulary: 379 tokens (character-level)
├── Embedding dimension: 512
├── Transformer layers: 6
├── Attention heads: 8
├── Feed-forward dimension: 2048
├── Max sequence length: 512
├── RoPE positional encoding: Yes
└── Activation: SwiGLU
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Effective batch size | 16 |
| Learning rate | 1e-3 |
| LR scheduler | Linear decay |
| Warmup ratio | 5% |
| Weight decay | 0.01 |
| Gradient accumulation | 2 steps |
| Checkpoint frequency | Every 100 steps |
| Max checkpoints kept | 3 |
| Precision | FP32 |
| Seed | 42 |

**Total training steps**: ~20,180 steps

## Quick Start

### Option 1: Run with Bash Script
```bash
cd /workspace
./run_training.sh
```

### Option 2: Run Directly with Python
```bash
cd /workspace
python3 train_blank_slate.py
```

## Expected Output

```
============================================================
BLANK SLATE TRAINING - Ethiopian Religious Texts
============================================================

[1/6] Loading datasets...
Loaded 366 Synaxarium entries
Loaded 31920 Canon Biblical verses

[2/6] Creating combined dataset...

=== Dataset Statistics ===
Synaxarium entries: 366
Biblical verses: 31920
Total samples: 32286
Total characters: 5,336,571
Average length: 165.3 chars

[3/6] Creating tokenizer...
Vocabulary size: 379

[4/6] Creating blank slate model...
Model architecture: Small (25.37M parameters)
  - Vocabulary: 379
  - Embedding dim: 512
  - Layers: 6
  - Heads: 8
  - Max sequence: 512

[5/6] Preparing training dataset...

[6/6] Configuring training...
Training configuration:
  - Output directory: /workspace/outputs/blank_slate_ethiopian_religious
  - Epochs: 10
  - Effective batch size: 16
  - Learning rate: 0.001
  - Warmup ratio: 0.05
  - Save every 100 steps

============================================================
INITIALIZING TRAINER...
============================================================

============================================================
STARTING TRAINING...
============================================================
Training:   0%|          | 0/20180 [00:00<?, ?it/s]
Step 10: loss=2.5432, lr=5.00e-05
Step 20: loss=2.3421, lr=1.00e-04
...
Saved checkpoint to outputs/blank_slate_ethiopian_religious/checkpoints/...
```

## Output Structure

```
outputs/blank_slate_ethiopian_religious/
├── checkpoints/
│   ├── checkpoint-000100_YYYYMMDD_HHMMSS/
│   │   ├── model.pt          # Model weights
│   │   ├── optimizer.pt      # Optimizer state
│   │   ├── scheduler.pt      # LR scheduler state
│   │   └── training_state.json
│   ├── checkpoint-000200_.../
│   └── ...
└── logs/                     # Training logs
```

## After Training

### 1. Freeze Base Knowledge
The trained model now has Ethiopian religious texts as base knowledge. To freeze these weights:

```python
from models.transformer import NexussTransformer
from models.config import NTFConfig

# Load trained model
model = NexussTransformer.from_pretrained("outputs/blank_slate_ethiopian_religious/checkpoints/checkpoint-latest")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

print(f"Base knowledge frozen: {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 2. Fine-tune on Downstream Tasks
Add task-specific layers on top of the frozen base:

```python
# Add classification head or other task-specific layers
# Only train the new layers while keeping base knowledge frozen
```

### 3. Evaluate Model Performance
```python
# Generate text
from transformers import GenerationConfig

input_text = "ስንክሳር - ወር: መስከረም, ቀን: 1"
output = model.generate(input_text, max_length=256)
print(output)
```

## Hardware Requirements

### Minimum (CPU Training)
- RAM: 8GB+
- Storage: 10GB+
- Time: ~4-8 hours for full training

### Recommended (GPU Training)
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB+
- Storage: 20GB+
- Time: ~30-60 minutes for full training

## Customization

### Adjust Model Size
Edit `train_blank_slate.py`:

```python
# For even smaller model (~10M params)
model_config = NTFConfig(
    vocab_size=vocab_size,
    d_model=256,
    n_layers=4,
    n_heads=4,
    max_seq_len=256,
)

# For larger model (~60M params)
model_config = NTFConfig.small()  # Default
model_config.d_model = 768
model_config.n_layers = 12
```

### Adjust Training Duration
```python
training_config = TrainingConfig(
    num_train_epochs=5,      # Reduce epochs
    max_steps=5000,          # Or set specific step count
    save_steps=50,           # More frequent checkpoints
)
```

### Include English Text (Bilingual Training)
```python
biblical_texts = load_canon_biblical_dataset(
    str(biblical_path), 
    include_english=True  # Enable bilingual
)
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing=True`

### Slow Training
- Use GPU if available
- Set `dataloader_num_workers=4`
- Enable mixed precision: `mixed_precision=Precision.FP16`

### Poor Convergence
- Increase learning rate to 2e-3
- Increase warmup ratio to 0.1
- Train for more epochs

## License

This training setup is part of the Nexuss-Transformer Framework and follows the same license terms.

## Citation

If you use this model in your research:
```bibtex
@misc{nexuss-transformer,
  title={Nexuss Transformer Framework: Blank Slate LLM Training},
  author={Nexuss0781},
  year={2024},
  howpublished={\\url{https://huggingface.co/Nexuss0781/Nexuss-Transformer}}
}
```
