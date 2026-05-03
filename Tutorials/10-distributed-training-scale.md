# Tutorial 10: Distributed Training at Scale

## Overview
When models exceed the memory capacity of a single GPU (e.g., Llama-3-70B, Mixtral), or when training time needs to be reduced from months to days, we must distribute the workload across multiple GPUs and nodes. This tutorial covers NTF's distributed training capabilities via Accelerate, along with the three pillars of distributed training: **Data Parallelism**, **Tensor Parallelism**, and **Pipeline Parallelism**, along with **ZeRO** optimization.

> **Note**: NTF currently supports multi-GPU training on a single node via Accelerate. Multi-node distributed training is planned for future releases. For large-scale training, consider using external orchestration tools like Kubernetes or cloud provider solutions.

## Prerequisites
- Completion of Tutorial 03 (Full Fine-Tuning)
- Multi-GPU environment (2+ GPUs, ideally 8+ for TP/PP)
- NCCL backend installed (`torch.distributed` support)

---

## 1. NTF Distributed Training with Accelerate

NTF automatically handles distributed training via Accelerate. Simply configure your training parameters and let NTF manage the complexity.

### Basic Distributed Configuration

```python
from ntf.config import NTFConfig, TrainingConfig, ModelConfig
from ntf.finetuning import FullFinetuneTrainer

# NTF automatically handles distributed training via Accelerate
config = NTFConfig(
    model=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
    training=TrainingConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # Accelerate auto-detects distributed setup
        fp16=False,
        bf16=True,
        gradient_checkpointing=True
    )
)

# Trainer automatically uses all available GPUs
trainer = FullFinetuneTrainer(config=config, train_dataset=train_dataset)
trainer.train()  # Distributed training handled internally
```

### Launch Script Example (Single Node Multi-GPU)

```bash
# Using accelerate launch
accelerate launch --num_processes=4 train_script.py

# Or with explicit config
accelerate config  # Interactive setup
accelerate launch train_script.py
```

### Checking Your Distributed Setup

```python
import torch
from accelerate import Accelerator

accelerator = Accelerator()

print(f"Number of processes: {accelerator.num_processes}")
print(f"Local process index: {accelerator.local_process_index}")
print(f"Device: {accelerator.device}")
print(f"Distributed type: {accelerator.distributed_type}")
```

---

## 2. Concepts: The 3D Parallelism Strategy

To train massive models, we slice the computation in three dimensions:

1.  **Data Parallelism (DP)**: Replicate the model on all GPUs. Split the *batch* across GPUs. Each GPU computes gradients on its slice, then averages them.
    *   *Limit*: Model must fit on one GPU.
2.  **Tensor Parallelism (TP)**: Split individual *matrix multiplications* within a layer across GPUs.
    *   *Example*: A linear layer $Y = XA$ is split so GPU1 computes $X A_1$ and GPU2 computes $X A_2$. They sync to get $Y$.
    *   *Requirement*: High-speed interconnect (NVLink) within a node.
3.  **Pipeline Parallelism (PP)**: Split the model by *layers*. GPU1 holds layers 1-10, GPU2 holds 11-20. Micro-batches flow through the pipeline like an assembly line.
    *   *Requirement*: Handles inter-node communication better than TP.

**Hybrid Approach**: Modern training uses all three: `DP x TP x PP`.

---

## 2. Data Parallelism (DDP) Deep Dive

PyTorch's `DistributedDataParallel` (DDP) is the baseline. Unlike `DataParallel` (which is Python-threaded and slow), DDP spawns separate processes per GPU.

### Step 2.1: Initializing the Process Group
You must initialize the distributed context before building your model.

```python
# train_ddp.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_rank(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # 1. Load Model (Sharded or Replicated depending on strategy)
    # For standard DDP, the full model loads on every GPU
    model_name = "microsoft/phi-2" 
    model = AutoModelForCausalLM.from_pretrained(model_name).to(rank)
    
    # 2. Wrap with DDP
    # bucket_cap_mb tunes gradient bucketing for communication overlap
    ddp_model = DDP(model, device_ids=[rank], bucket_cap_mb=25)

    # 3. Use DistributedSampler to split data uniquely per rank
    # dataset = MyDataset(...)
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Dummy training loop
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    
    for step in range(100):
        # input_ids = next(dataloader) # shape: [batch/world_size, seq_len]
        # inputs = input_ids.to(rank)
        
        # Fake input for demo
        inputs = torch.randint(0, 1000, (4, 128)).to(rank)
        labels = inputs.clone()
        
        outputs = ddp_model(inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if rank == 0 and step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Starting DDP with {world_size} GPUs")
    mp.spawn(train_rank, args=(world_size,), nprocs=world_size, join=True)
```

### Key DDP Concepts:
- **Gradient Bucketing**: DDP groups gradients into buckets to overlap communication (all-reduce) with backprop computation.
- **Static Graph**: DDP expects the computation graph to be static. If you have dynamic control flow (e.g., variable depth), set `find_unused_parameters=True` (slower).

---

## 3. ZeRO (Zero Redundancy Optimizer)

Standard DDP replicates **Model**, **Gradients**, and **Optimizer States** on every GPU.
- For a 7B model (fp16):
    - Parameters: 14GB
    - Gradients: 14GB
    - Optimizer States (Adam): 28GB (fp32 master weights + momentum + variance)
    - **Total per GPU**: ~56GB. This limits us to small models even on 8x80GB GPUs.

**ZeRO** (implemented in DeepSpeed and FSDP) shards these states across GPUs.

### ZeRO Stages:
1.  **ZeRO-1**: Shard **Optimizer States**. Gradients and Params replicated.
    *   *Memory Save*: Significant (Adam states are huge).
2.  **ZeRO-2**: Shard **Optimizer States + Gradients**. Params replicated.
    *   *Memory Save*: Massive. Communication overhead increases slightly.
3.  **ZeRO-3**: Shard **Optimizer States + Gradients + Parameters**.
    *   *Memory Save*: Maximum. Allows training models 10x larger than VRAM.
    *   *Cost*: High communication. Requires gathering parameters before every forward pass.

### Using PyTorch FSDP (Fully Sharded Data Parallel)
FSDP is PyTorch's native implementation of ZeRO-3.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def get_fsdp_policy():
    return transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer}
    )

def train_fsdp(rank, world_size):
    setup(rank, world_size)
    
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Configure FSDP
    fsdp_model = FSDP(
        model,
        device_id=rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # ZeRO-3
        auto_wrap_policy=get_fsdp_policy(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        use_orig_params=True, # Important for optimizer compatibility
    )
    
    # Optimizer must be created AFTER wrapping
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    
    # Training loop similar to DDP, but FSDP handles parameter gathering internally
    # ...
    cleanup()
```

---

## 4. Tensor Parallelism (TP)

When a single layer's weights don't fit on one GPU, we split the matrix.
In Transformers, the heavy parts are:
1.  QKV Projection ($W_{qkv}$)
2.  Output Projection ($W_{out}$)
3.  MLP Up/Down Projections

### How TP Works (Column/Row Splitting)
Consider $Y = XA$.
- Split $A$ by columns: $A = [A_1, A_2]$.
- GPU1 computes $Y_1 = X A_1$.
- GPU2 computes $Y_2 = X A_2$.
- Result $Y = [Y_1, Y_2]$ is split naturally.
- For the next layer $Z = YB$, we split $B$ by rows to match.

**Implementation Note**: Implementing TP manually is error-prone. We use **DeepSpeed** or **Megatron-LM** patterns.

#### DeepSpeed Configuration for TP
Create `ds_config_tp.json`:
```json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 1,
  "fp16": { "enabled": true },
  "zero_optimization": {
    "stage": 0, 
    "offload_optimizer": { "device": "none" }
  },
  "tensor_parallel": {
    "tp_size": 4, 
    "enabled": true
  },
  "pipeline_parallel": {
    "pp_size": 1,
    "enabled": false
  }
}
```
*Note: TP usually disables ZeRO sharding of parameters because the parameters are already split.*

---

## 5. Pipeline Parallelism (PP)

PP splits layers across devices.
- GPU0: Layers 1-8
- GPU1: Layers 9-16
- GPU2: Layers 17-24

### The Bubble Problem
In naive PP, GPU1 waits for GPU0 to finish the forward pass before starting. This creates idle time ("bubble").
**Solution**: **Micro-batching**.
Split a global batch into 4 micro-batches.
1. GPU0 processes MicroBatch 1 (Forward).
2. GPU0 sends activation to GPU1, then starts MicroBatch 2 (Forward).
3. GPU1 processes MicroBatch 1 (Forward) while GPU0 does MB2.
This overlaps computation and reduces the bubble size.

### GPipe vs 1F1B
- **GPipe**: Simple forward-all, backward-all. Large bubble.
- **1F1B (One-Forward-One-Backward)**: Interleaves forward and backward passes to minimize bubble. Used in modern schedulers.

---

## 6. Putting It All Together: Hybrid Parallelism

For a 70B model on a 64-GPU cluster:
- **TP=4**: Split each layer across 4 GPUs (within a node).
- **PP=4**: Split 80 layers into 4 chunks of 20 layers.
- **DP=4**: Replicate this TP+PP setup 4 times to process different data batches.
- Total GPUs: $4 \times 4 \times 4 = 64$.

### Launching with DeepSpeed
DeepSpeed unifies these configurations.

```bash
deepspeed --num_gpus=8 train.py \
  --deepspeed ds_config_hybrid.json
```

**ds_config_hybrid.json**:
```json
{
  "fp16": { "enabled": true },
  "bf16": { "enabled": false },
  "zero_optimization": {
    "stage": 1, 
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "tensor_parallel": { "tp_size": 2, "enabled": true },
  "pipeline_parallel": { "pp_size": 2, "enabled": true, "schedule": "1F1B" },
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4
}
```

---

## 7. Practical Exercise: Scaling a 7B Model

**Scenario**: You have 4x A100 (40GB). You want to fine-tune Llama-2-7B.
- Full model (fp16): 14GB.
- Optimizer states (fp32): 28GB.
- Total needed: 42GB. **OOM on 40GB cards.**

**Solution A: ZeRO-2 (Recommended for 7B)**
Shard optimizer and gradients.
- Memory per GPU $\approx$ Params (14) + Shard(Opt+Grad) $\approx$ 14 + 10 = 24GB. Fits easily.

**Solution B: ZeRO-3 + Offload**
If you had 2 GPUs, ZeRO-3 shards params too.
- Memory per GPU $\approx$ Shard(Params) + Small Buffer.
- If still tight, offload optimizer to CPU (`offload_optimizer: {device: "cpu"}`).

**Code Snippet for ZeRO-2 Config**:
```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8 }
  }
}
```

---

## 8. Debugging Distributed Training

1.  **Hangs/Timeouts**: Usually caused by mismatched tensor shapes causing one rank to wait forever for a broadcast.
    *   *Fix*: Ensure `DistributedSampler` drops last batch consistently or pad data. Check `find_unused_parameters`.
2.  **NCCL Errors**: `unhandled system error`.
    *   *Cause*: Network issues, P2P access disabled.
    *   *Fix*: `export NCCL_P2P_DISABLE=1`, check `nvidia-smi topo -m`.
3.  **Slow Scaling**:
    *   *Check*: Are you using `overlap_comm`? Is the network bandwidth sufficient for ZeRO-3?
    *   *Metric*: Monitor "MFU" (Model Flops Utilization). >40% is good for H100s.

---

## 9. Summary Checklist

| Strategy | When to use | Memory Efficiency | Comm Overhead | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **DDP** | Model fits on 1 GPU | Low | Low | Low |
| **ZeRO-1** | Optimizer states too big | Medium | Low-Med | Low |
| **ZeRO-2** | Gradients too big | High | Med | Low |
| **ZeRO-3** | Model params too big | Very High | High | Med |
| **TP** | Single layer too big | High (Intra-node) | Very High | High |
| **PP** | Total layers too many | High (Inter-node) | Med (Bubble) | High |

## Next Steps
In Tutorial 11, we will explore **Inference Optimization**, covering quantization (INT8/FP4), KV Caching, and serving engines like vLLM and TGI to deploy these massive models efficiently.
