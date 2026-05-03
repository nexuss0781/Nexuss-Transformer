# NTF Blank Slate Training Guide

## 1. EthioBBPE Tokenization Status ✅

**YES, EthioBBPE is fully integrated and utilized!**

The training script automatically leverages the NTFTokenizer with EthioBBPE in three ways:

### Automatic Loading (Recommended)
```python
# The script automatically loads from HuggingFace Hub
tokenizer = NTFTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")
```

### Manual Loading (If you have custom files)
```bash
python train_blank_slate.py \
  --vocab_file path/to/vocab.json \
  --merges_file path/to/merges.txt
```

### Fallback Mechanism
If EthioBBPE is unavailable, the script falls back to character-level tokenization, but **this is NOT recommended** for Amharic/Ge'ez text.

### Verification
When you run the training, you should see:
```
Loading default EthioBBPE tokenizer from HuggingFace Hub...
NTFTokenizer loaded successfully!
Vocabulary size: 16000
```

**Critical Notes:**
- EthioBBPE vocabulary: **16,000 tokens** (optimized for Amharic/Ge'ez)
- Properly handles Ethiopian script morphology
- Significantly better than character-level or standard BPE
- **DO NOT** use `--use_char_tokenizer` flag unless absolutely necessary

---

## 2. Model Parameter Counts

| Model Size | Parameters | GPU Memory (FP16) | Recommended Use |
|------------|-----------|-------------------|-----------------|
| **SMALL**  | 27.07M    | ~100 MB           | Testing, prototyping |
| **MEDIUM** | 167.40M   | ~650 MB           | Production, good balance |
| **LARGE**  | 1,240.83M | ~4.8 GB           | Maximum performance |

### Architecture Details

**SMALL (27M):**
- Hidden size: 512
- Layers: 6
- Attention heads: 8
- FFN size: 2048

**MEDIUM (167M):**
- Hidden size: 1024
- Layers: 12
- Attention heads: 16
- FFN size: 4096

**LARGE (1.24B):**
- Hidden size: 2048
- Layers: 24
- Attention heads: 32
- FFN size: 8192

All models use:
- ✅ Tied embeddings (saves memory)
- ✅ RoPE (Rotary Positional Embeddings)
- ✅ SwiGLU activation
- ✅ RMSNorm normalization

---

## 3. Recommended Training Configurations

### Dataset Statistics
- **Synaxarium:** 366 entries
- **Bible:** 31,920 verses
- **Total:** 32,286 samples
- **Estimated tokens:** ~1.6M (with EthioBBPE)

Since this is a relatively small dataset, we recommend **more epochs** to ensure proper convergence.

---

### 🚀 QUICK START COMMANDS

#### For SMALL Model (Testing/Prototyping)
```bash
python train_blank_slate.py \
  --model_size small \
  --num_epochs 50 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --warmup_steps 1000 \
  --max_seq_length 512 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --output_dir checkpoints/blank_slate_small \
  --save_every_n_epochs 5 \
  --log_every_n_steps 50
```

#### For MEDIUM Model (RECOMMENDED for Production) ⭐
```bash
python train_blank_slate.py \
  --model_size medium \
  --num_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --warmup_steps 2000 \
  --max_seq_length 512 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --output_dir checkpoints/blank_slate_medium \
  --save_every_n_epochs 5 \
  --log_every_n_steps 100
```

#### For LARGE Model (Maximum Performance)
```bash
python train_blank_slate.py \
  --model_size large \
  --num_epochs 20 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 0.0002 \
  --warmup_steps 5000 \
  --max_seq_length 512 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --output_dir checkpoints/blank_slate_large \
  --save_every_n_epochs 3 \
  --log_every_n_steps 200
```

---

### Detailed Parameter Explanations

#### Essential Parameters

| Parameter | Small | Medium | Large | Description |
|-----------|-------|--------|-------|-------------|
| `--model_size` | small | medium | large | Model architecture size |
| `--num_epochs` | 50 | 30 | 20 | Training epochs (more for smaller data) |
| `--batch_size` | 32 | 16 | 8 | Samples per batch (adjust for GPU memory) |
| `--learning_rate` | 1e-3 | 5e-4 | 2e-4 | Initial learning rate |
| `--warmup_steps` | 1000 | 2000 | 5000 | LR warmup period |

#### Performance Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--mixed_precision` | bf16 | Use BF16 (or fp16 if BF16 unavailable) |
| `--gradient_checkpointing` | ✓ | Enable to save memory |
| `--gradient_accumulation_steps` | 1-4 | Accumulate gradients for larger effective batch |
| `--max_seq_length` | 512 | Maximum sequence length |

#### Output & Logging

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--output_dir` | checkpoints/... | Where to save checkpoints |
| `--save_every_n_epochs` | 3-5 | Save checkpoint frequency |
| `--log_every_n_steps` | 50-200 | Logging frequency |
| `--resume_from_checkpoint` | (optional) | Resume training from checkpoint |

---

### GPU Memory Requirements

| Model Size | Min GPU (FP32) | Recommended (FP16/BF16) | With Gradient Checkpointing |
|------------|---------------|------------------------|----------------------------|
| SMALL | 2 GB | 1 GB | < 1 GB |
| MEDIUM | 8 GB | 4 GB | 2 GB |
| LARGE | 24 GB | 12 GB | 6 GB |

**Note:** If you get OOM (Out of Memory) errors:
1. Reduce `--batch_size`
2. Increase `--gradient_accumulation_steps`
3. Enable `--gradient_checkpointing`
4. Use `--mixed_precision bf16` or `fp16`

---

### Advanced Options

#### Include English for Bilingual Training
```bash
python train_blank_slate.py \
  ... \
  --include_english
```

#### Custom Tokenizer Path
```bash
python train_blank_slate.py \
  ... \
  --tokenizer_path "path/to/custom/tokenizer"
```

#### Disable Advanced Features (Not Recommended)
```bash
python train_blank_slate.py \
  ... \
  --no_rope \
  --no_swiglu \
  --no_rmsnorm
```

#### Resume Training
```bash
python train_blank_slate.py \
  ... \
  --resume_from_checkpoint checkpoints/blank_slate_medium/checkpoint_epoch_10
```

---

## 4. Monitoring Training

### What to Watch For

1. **Loss Decrease:** Should steadily decrease over epochs
   - Initial loss: ~4.0-6.0
   - Target loss: < 2.0 (depends on model size)

2. **Learning Rate Schedule:**
   - Warmup phase: LR increases linearly
   - After warmup: LR decays with cosine schedule

3. **Checkpoint Saving:**
   - Checkpoints saved every N epochs
   - Latest checkpoint always updated

### Sample Training Output
```
Epoch 1/30 [====================] 100%
Step: 100 | Loss: 4.523 | LR: 0.00025 | Time: 45.2s
Step: 200 | Loss: 3.891 | LR: 0.00050 | Time: 42.1s
...
Epoch 1 completed! Average Loss: 3.756
Checkpoint saved: checkpoints/blank_slate_medium/checkpoint_epoch_1

Validation Loss: 3.234
```

---

## 5. Post-Training

### Using Your Trained Model

```python
from ntf import NTFTransformer, NTFTokenizer

# Load trained model
model = NTFTransformer.from_pretrained("checkpoints/blank_slate_medium/checkpoint_epoch_30")
tokenizer = NTFTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")

# Generate text
prompt = "<doc><type>ስንክሳር</type><ወር>መስከረም</ወር><ቀን>1</ቀን><content>"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=200, temperature=0.7)
text = tokenizer.decode(outputs[0])
print(text)
```

### Model Evaluation

After training, evaluate on held-out data:
```bash
python evaluate_model.py \
  --model_path checkpoints/blank_slate_medium/checkpoint_epoch_30 \
  --test_data data/test.parquet
```

---

## 6. Troubleshooting

### Issue: EthioBBPE Not Loading
**Solution:**
```bash
pip install huggingface_hub
# Or provide manual tokenizer files
python train_blank_slate.py --vocab_file vocab.json --merges_file merges.txt
```

### Issue: Out of Memory (OOM)
**Solution:**
1. Reduce batch_size: `--batch_size 4`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use mixed precision: `--mixed_precision bf16`
4. Reduce sequence length: `--max_seq_length 256`

### Issue: Loss Not Decreasing
**Solution:**
1. Lower learning rate: `--learning_rate 0.0001`
2. Increase warmup: `--warmup_steps 5000`
3. Check data quality
4. Verify tokenizer is working correctly

### Issue: Slow Training
**Solution:**
1. Enable mixed precision: `--mixed_precision bf16`
2. Reduce logging frequency: `--log_every_n_steps 200`
3. Use larger batch with gradient accumulation

---

## 7. Best Practices Summary

✅ **DO:**
- Use EthioBBPE tokenizer (default behavior)
- Enable mixed precision (bf16 or fp16)
- Use gradient checkpointing for larger models
- Start with MEDIUM model for production
- Monitor loss curves
- Save checkpoints frequently
- Use structured XML-like format for data

❌ **DON'T:**
- Use character tokenizer unless necessary
- Train LARGE model without sufficient GPU memory
- Set learning rate too high (>0.001)
- Skip warmup steps
- Train without gradient checkpointing on limited memory

---

## Quick Reference Card

```bash
# RECOMMENDED: Medium model, production-ready
python train_blank_slate.py \
  --model_size medium \
  --num_epochs 30 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --warmup_steps 2000 \
  --max_seq_length 512 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --output_dir checkpoints/blank_slate_medium

# FAST TESTING: Small model
python train_blank_slate.py \
  --model_size small \
  --num_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --output_dir checkpoints/blank_slate_small_test

# MAXIMUM PERFORMANCE: Large model (requires good GPU)
python train_blank_slate.py \
  --model_size large \
  --num_epochs 20 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 0.0002 \
  --warmup_steps 5000 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --output_dir checkpoints/blank_slate_large
```

---

For questions or issues, check the training logs and adjust parameters accordingly. The key is to start with the MEDIUM model configuration and tune based on your specific hardware and requirements.
