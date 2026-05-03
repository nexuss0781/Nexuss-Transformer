# Tutorial 13: Troubleshooting & Performance Debugging

## Overview
Even with perfect theory, training often fails in practice. You will encounter **NaN losses**, **Out-Of-Memory (OOM)** errors, **non-convergence**, and **mysteriously slow training**. This tutorial provides a systematic debugging framework to diagnose and fix these issues efficiently.

## Prerequisites
- Experience running training loops (Tutorial 02)
- Basic understanding of CUDA/GPU architecture
- Familiarity with PyTorch debugging tools

---

## 1. The Debugging Mindset

**Rule #1**: Reproduce the issue on the smallest possible scale.
- If it fails on 8 GPUs, try 1 GPU.
- If it fails on 50B tokens, try 1M tokens.
- If it fails with batch size 64, try batch size 2.

**Rule #2**: Isolate the variable.
- Change only one thing at a time (LR, batch size, model size).
- Use a deterministic seed (`torch.manual_seed(42)`) to ensure reproducibility.

---

## 2. Diagnosing NaN/Inf Losses

The dreaded `loss = nan` is the most common failure mode.

### Causes & Solutions

#### 2.1 Learning Rate Too High
The optimizer takes steps too large, shooting past the minimum into infinity.
- **Symptom**: Loss spikes suddenly then becomes NaN.
- **Fix**: Reduce LR by 10x. Use a learning rate warmup.
```python
# Check for gradient explosion
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if torch.isinf(grad_norm) or torch.isnan(grad_norm):
            print(f"Bad Grad: {name}, Norm: {grad_norm}")
```

#### 2.2 Mixed Precision Instability
FP16 has a limited range ($6 \times 10^{-5}$ to $65504$). Small gradients underflow to 0; large activations overflow to Inf.
- **Symptom**: NaN appears in specific layers (often attention).
- **Fix**: 
  - Use **Gradient Scaling** (automatic in `GradScaler`).
  - Switch to **BF16** (Bfloat16) if hardware supports (A100/H100). BF16 has the same range as FP32.
  - Disable AMP (Automatic Mixed Precision) for debugging.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16): # Try bfloat16 instead of float16
        outputs = model(batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 2.3 Division by Zero / Log(0)
Common in custom loss functions.
- **Symptom**: Immediate NaN at step 1.
- **Fix**: Add epsilon ($\epsilon = 1e-8$) to denominators and log inputs.
```python
# Bad
loss = -torch.log(probs)

# Good
loss = -torch.log(probs + 1e-8)
```

#### 2.4 Bad Data
Input contains NaNs or Infs.
- **Check**:
```python
assert not torch.isnan(inputs).any(), "Input contains NaNs"
assert not torch.isinf(inputs).any(), "Input contains Infs"
```

### Systematic NaN Hunt Script
Run this when NaN occurs:
```python
def debug_nan(model, inputs, loss):
    print(f"Loss: {loss.item()}")
    
    # 1. Check Inputs
    if torch.isnan(inputs).any(): print("NaN in Inputs")
    
    # 2. Check Parameters
    for name, p in model.named_parameters():
        if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
            print(f"NaN/Inf in Param: {name}")
    
    # 3. Check Gradients (after backward)
    for name, p in model.named_parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            print(f"NaN/Inf in Grad: {name}")
    
    # 4. Check Activations (Hook method)
    def hook(module, inp, out):
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"NaN/Inf in Output of: {module.__class__.__name__}")
    
    for module in model.modules():
        module.register_forward_hook(hook)
    
    # Re-run forward pass to trigger hooks
    _ = model(inputs)
```

---

## 3. Fixing Out-Of-Memory (OOM) Errors

`CUDA out of memory. Tried to allocate X GiB.`

### Step 3.1: Analyze Memory Usage
Use `nvidia-smi` to see total usage. Use PyTorch profiler to see *what* is using memory.

```python
import torch.cuda.memory as mem

print(f"Allocated: {mem.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {mem.memory_reserved()/1e9:.2f} GB")
```

### Step 3.2: Reduction Strategies

#### A. Reduce Batch Size
Most direct fix. Use **Gradient Accumulation** to maintain effective batch size.
```python
# Instead of batch_size=64 (OOM)
# Use batch_size=8, accumulation_steps=8
effective_batch = 8 * 8 = 64
```

#### B. Enable Gradient Checkpointing
Trade compute for memory. Recompute activations during backward pass instead of storing them.
- **Memory Save**: Up to 60%.
- **Cost**: 20-30% slower training.
```python
model.gradient_checkpointing_enable()
# Or manually
from torch.utils.checkpoint import checkpoint
# Wrap expensive layers: hidden = checkpoint(layer_module, hidden)
```

#### C. Use ZeRO / FSDP
Shard model states across GPUs (see Tutorial 10).
- **ZeRO-1**: Shard Optimizer.
- **ZeRO-3**: Shard Parameters (fits largest models).

#### D. Quantization
Load model in 8-bit or 4-bit (if inference or QLoRA).
```python
model = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True)
```

#### E. CPU Offload
Move optimizer states or parameters to CPU RAM.
- **DeepSpeed Config**:
```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": { "device": "cpu", "pin_memory": true },
  "offload_param": { "device": "cpu", "pin_memory": true }
}
```
*Note: Slower due to PCIe transfer.*

---

## 4. Convergence Issues

The model trains but doesn't learn (loss stays flat) or learns too slowly.

### Checklist

#### 4.1 Data Loader Issues
- **Problem**: Labels are misaligned or all zeros.
- **Debug**: Print a sample batch. Verify `input_ids` correspond to `labels`.
- **Problem**: Shuffling is off, model sees same pattern repeatedly.
- **Fix**: Ensure `shuffle=True` in DataLoader (unless sequential).

#### 4.2 Learning Rate Problems
- **Too Low**: Loss decreases imperceptibly.
  - *Fix*: Increase LR by 10x.
- **Too High**: Loss oscillates wildly.
  - *Fix*: Decrease LR, add warmup.
- **No Warmup**: Transformers need warmup to stabilize embeddings.
  - *Fix*: Use scheduler with warmup (e.g., `get_linear_schedule_with_warmup`).

#### 4.3 Vanishing/Exploding Gradients
- **Symptom**: Early layers have near-zero gradients; later layers have huge ones.
- **Fix**: 
  - **Gradient Clipping**:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```
  - **Proper Initialization**: Ensure you aren't re-initializing a pre-trained model randomly.

#### 4.4 Label Smoothing
Sometimes helps convergence by preventing over-confidence.
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

#### 4.5 Verify Forward Pass
Is the model just predicting the mean?
- **Test**: Run a batch through. Check output distribution.
- If softmax probabilities are uniform ($1/Vocab$), the model hasn't learned.
- If they are peaked at one token always, maybe bias is too high.

---

## 5. Performance Bottlenecks (Slow Training)

Training is working but slower than expected (< 40% MFU).

### 5.1 Profiling with PyTorch Profiler
Identify where time is spent.

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
             schedule=torch.profiler.schedule(wait=1, warmup=1, active=3), 
             on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')) as prof:
    
    for step, batch in enumerate(dataloader):
        # ... train step ...
        
        prof.step()
        if step > 5: break

# View results: tensorboard --logdir ./log
```

### Common Bottlenecks

#### A. Data Loading (CPU Bound)
GPU waits for CPU to feed data.
- **Symptom**: GPU utilization drops to 0% periodically.
- **Fix**:
  - Increase `num_workers` in DataLoader (try 4, 8, 16).
  - Use `pin_memory=True`.
  - Pre-process data offline (save tokenized IDs to disk).
  - Use asynchronous loading.

#### B. Communication Overhead (Distributed)
GPUs wait for each other.
- **Symptom**: Low scaling efficiency when adding GPUs.
- **Fix**:
  - Ensure NVLink is enabled (`nvidia-smi topo -m`).
  - Use `overlap_comm` in DeepSpeed/FSDP.
  - Increase micro-batch size to reduce frequency of sync.
  - Check network bandwidth (InfiniBand vs Ethernet).

#### C. Kernel Launch Overhead
Many small operations.
- **Fix**: Use **Fused Kernels**.
  - `apex.optimizers.FusedAdam` instead of `torch.optim.AdamW`.
  - `FlashAttention` (replaces standard attention, 2-3x faster).
  - `xformers` library for memory-efficient attention.

#### D. Python GIL / Overhead
- **Fix**: Use `torch.compile()` (PyTorch 2.0+).
```python
model = torch.compile(model) # JIT compilation
```
*Note: May have compatibility issues with some dynamic graphs.*

---

## 6. Distributed Debugging Specifics

### 6.1 Hanging Processes
One rank finishes, others wait forever.
- **Cause**: Mismatched tensor sizes causing broadcast failure.
- **Cause**: One rank skips a step (e.g., `if rank == 0` inside training loop without sync).
- **Debug**: Add print statements with `dist.barrier()` to find which rank stops.
```python
print(f"Rank {dist.get_rank()} starting step")
dist.barrier() # All ranks must reach here
```

### 6.2 NCCL Errors
`NCCL error: unhandled system error`.
- **Cause**: P2P communication failed.
- **Fix**:
  - `export NCCL_P2P_DISABLE=1`
  - `export NCCL_IB_DISABLE=1` (if using InfiniBand issues)
  - Check for ECC errors in `nvidia-smi`.

---

## 7. Practical Exercise: Debugging a Broken Run

**Scenario**: You start training and get `Loss: NaN` at step 50.

**Step 1: Reproduce with minimal config**
- Set `batch_size=2`, `max_steps=100`, `model=tiny`.

**Step 2: Add Debug Hooks**
Insert the `debug_nan` function from Section 2.

**Step 3: Check Gradient Norms**
Log gradient norms per layer.
```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Grad Norm: {total_norm}")
# If > 1e5, you have exploding gradients -> Clip!
```

**Step 4: Verify Data**
Print the input IDs that cause the crash.
```python
if torch.isnan(loss):
    print("Crash Input:", input_ids[batch_idx])
    break
```

**Step 5: Apply Fix**
- Found gradients exploding? → Add `clip_grad_norm_`.
- Found FP16 overflow? → Switch to BF16 or reduce LR.
- Found bad token ID (-100 where it shouldn't be)? → Fix tokenizer.

---

## 8. Summary Checklist

| Issue | Symptom | Likely Cause | Fix |
| :--- | :--- | :--- | :--- |
| **NaN Loss** | Loss becomes `nan` | High LR, FP16 overflow, Bad Data | Reduce LR, Use BF16, Check Data |
| **OOM** | `CUDA out of memory` | Batch too big, No checkpointing | Reduce Batch, Gradient Checkpointing, ZeRO |
| **No Convergence** | Flat loss curve | LR too low, Bad labels, No warmup | Increase LR, Check Labels, Add Warmup |
| **Oscillation** | Loss jumps up/down | LR too high, Small batch | Reduce LR, Increase Batch |
| **Slow Training** | Low GPU util | Data loading, Comm overhead | More workers, Fused kernels, Flash Attn |
| **Hang** | Process stalls | Rank mismatch, Missing barrier | Check control flow, Add barriers |

## Final Words
Debugging AI systems is iterative. Always start small, isolate variables, and use tools (profilers, hooks) rather than guessing. Keep detailed logs of experiments (Tutorial 12) so you can trace back what changed when things break.

---

## Series Conclusion

You have now completed the **End-to-End AI Training Tutorial Series**.

**Recap of Journey**:
1.  **Foundations**: Built models from scratch, understood transformers.
2.  **Training**: Ran first runs, mastered full fine-tuning and PEFT.
3.  **Advanced**: Explored RLHF, DPO, and multi-task learning.
4.  **Scale**: Mastered distributed training (TP/PP/ZeRO).
5.  **Deployment**: Optimized inference, quantization, serving.
6.  **Ops**: Built CI/CD pipelines, governance, monitoring.
7.  **Debugging**: Learned to fix NaNs, OOMs, and bottlenecks.

**Next Steps for You**:
- Experiment with open-source models (Llama, Mistral, Qwen).
- Contribute to frameworks (Hugging Face, DeepSpeed, vLLM).
- Build a portfolio project: Train a domain-specific assistant end-to-end.
- Stay updated: The field moves fast (new architectures, new quantization methods).

Happy Training! 🚀
