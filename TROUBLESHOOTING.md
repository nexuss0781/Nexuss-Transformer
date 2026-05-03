# NTF Troubleshooting Decision Tree

Interactive troubleshooting guide for common issues.

## Start Here: What's your issue?

### Training fails to start

#### Import errors
- **ModuleNotFoundError** → `pip install ntf-transformers`
- **Version conflicts** → Check requirements.txt, run `pip install -r requirements.txt`

#### CUDA out of memory
→ See [Memory Issues](#memory-issues)

#### Config validation errors
→ Run `validate_config()` before training

---

### Memory Issues

#### GPU OOM during training
1. **Reduce batch size** → Set `per_device_train_batch_size=1`
2. **Enable gradient checkpointing** → Set `gradient_checkpointing=True`
3. **Use LayerFreezer** → `freeze_backbone(num_layers_to_keep=4)`
4. **Try quantization** → Set `load_in_4bit=True`
5. **Use gradient accumulation** → Accumulate over more steps with smaller batches

#### CPU RAM exhaustion
1. **Reduce dataloader workers** → Set `dataloader_num_workers=0`
2. **Use streaming dataset** → Load data in chunks
3. **Reduce max_length** → Shorter sequences use less memory

---

### Training instability

#### Loss is NaN
1. **Reduce learning rate** → Try 1e-5 → 1e-6
2. **Enable gradient clipping** → Set `gradient_clip_val=1.0`
3. **Disable mixed precision temporarily** → Set `fp16=False`
4. **Check for overflow in loss computation**

#### Loss not decreasing
1. **Check learning rate** → Too low? Increase it
2. **Verify dataset preprocessing** → Check tokenization
3. **Increase model capacity** → Use larger model or more trainable params
4. **Check label quality** → Ensure labels are correct

#### Overfitting
1. **Add regularization** → Increase `weight_decay`
2. **Enable early stopping** → Monitor validation loss
3. **Get more training data** → Augment dataset
4. **Add dropout** → Increase dropout rate

---

### Poor evaluation results

#### Metrics worse than base model
1. **Check for data leakage** → Ensure train/eval split is clean
2. **Verify eval dataset quality** → Check label distribution
3. **Compare multiple checkpoints** → Earlier checkpoint might be better

#### Inconsistent metrics
1. **Increase eval dataset size** → More samples for stable estimates
2. **Use multiple random seeds** → Average across runs
3. **Check metric implementation** → Verify correctness

---

### Deployment issues

#### Model loading fails in production
1. **Verify version exists** → Call `list_versions()`
2. **Check file permissions** → Ensure read access
3. **Ensure same NTF version** → Match training and inference versions

#### Slow inference
1. **Enable quantization** → Use 4-bit or 8-bit inference
2. **Use optimized serving** → vLLM, TGI for high throughput
3. **Batch requests** → Process multiple inputs together
4. **Cache embeddings** → For repeated prompts

---

### PEFT/LoRA specific issues

#### LoRA not training
1. **Check target modules** → Verify they exist in model
2. **Verify adapters are applied** → Check `print_trainable_parameters()`
3. **Ensure base model is frozen** → Only adapter weights should be trainable

#### Adapter merge fails
1. **Check compatibility** → Base model must match original
2. **Verify adapter path** → Ensure files exist
3. **Reload base model** → Fresh load before merging

---

### Quick Reference Commands

```bash
# Verify installation
ntf-verify-installation

# Check GPU memory
nvidia-smi

# List available models
python -c "from ntf.models import ModelRegistry; r = ModelRegistry(); print(r.list_versions())"

# Validate config
python -c "from ntf.utils import validate_config; validate_config('config.yaml')"
```

---

### Getting Help

1. **Search existing issues** → GitHub Issues
2. **Check tutorial comments** → Solutions may be documented
3. **Provide reproducible example** → Include code, error, environment info
4. **Include version info** → NTF, Python, PyTorch, Transformers versions

---

## Copy-Paste Solutions

### Fix OOM Error
```python
from ntf.config import TrainingConfig

config = TrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
)
```

### Fix NaN Loss
```python
from ntf.config import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-6,
    gradient_clip_val=1.0,
    fp16=False,
)
```

### Setup Experience Replay
```python
from ntf.utils import ReplayConfig, ReplayBuffer

config = ReplayConfig(
    replay_size=1000,
    replay_ratio=0.5,
    selection_strategy="uniform",
)
buffer = ReplayBuffer(config)
```
