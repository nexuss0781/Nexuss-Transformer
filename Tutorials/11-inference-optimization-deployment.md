# Tutorial 11: Inference Optimization & Deployment

## Overview
Training a model is only half the battle. Deploying Large Language Models (LLMs) for inference presents unique challenges: high memory bandwidth requirements, latency constraints, and the need to serve multiple users concurrently. This tutorial covers **Quantization**, **KV Caching**, **Speculative Decoding**, and production serving engines like **vLLM** and **TGI**.

## Prerequisites
- Understanding of Transformer architecture (Tutorial 01)
- Basic PyTorch knowledge
- Familiarity with Hugging Face `transformers`

---

## 1. The Inference Bottleneck

Unlike training, which is compute-bound (matrix multiplications), inference is often **memory-bandwidth bound**.

### The Memory Wall
To generate one token:
1. Load all model weights from VRAM to Compute Units.
2. Perform calculation.
3. Store result.

For a 7B model (14GB in fp16):
- To generate 1 token at batch size 1, you must read 14GB of data.
- On an A100 (1.5 TB/s bandwidth), this takes ~9ms just for memory transfer.
- Compute time is negligible (~0.1ms).
- **Result**: You can only generate ~100 tokens/sec regardless of compute power.

**Solution**: Reduce memory footprint (Quantization) and reuse memory (KV Cache).

---

## 2. Quantization: Reducing Precision

Quantization reduces the number of bits used to represent weights and activations.

### Types of Quantization

#### 2.1 Post-Training Quantization (PTQ)
Quantize a pre-trained model without retraining.
- **INT8**: Weights scaled to 8-bit integers. Minimal accuracy loss.
- **FP8**: New format supported by H100s. Good balance of range and precision.
- **INT4**: Aggressive compression (e.g., GPTQ, AWQ). Requires careful calibration.

#### 2.2 Quantization-Aware Training (QAT)
Simulate quantization noise during training/fine-tuning.
- Model learns to compensate for precision loss.
- Best for INT4 or lower, but computationally expensive.

### Implementing INT8 with BitsAndBytes
The easiest way to quantize for inference using Hugging Face.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import infer_auto_device_map

model_id = "meta-llama/Llama-2-7b-hf"

# 4-bit configuration (QLoRA style inference)
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",  # Normalized Float 4
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True, # Nested quantization
}

# 8-bit configuration
# bnb_config = {"load_in_8bit": True}

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Automatically spread across GPUs
    quantization_config=bnb_config if "load_in_4bit" in bnb_config else None,
    torch_dtype=torch.float16 if not bnb_config.get("load_in_8bit") else None,
    low_cpu_mem_usage=True
)

# Inference
input_text = "Explain quantum entanglement."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### GPTQ & AWQ (GPU-Aware Quantization)
For INT4, generic quantization fails. **GPTQ** and **AWQ** use activation-aware weight selection.
- **AWQ (Activation-Aware Weight Quantization)**: Preserves salient weights (those causing large activations) in FP16, quantizes the rest.
- **Tool**: Use `auto-gptq` or `llama-cpp-python` for GGUF formats.

```bash
# Install auto-gptq
pip install auto-gptq optimum

# Quantize script example
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=False,
)

model = AutoGPTQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantize_config)
# ... calibrate on dataset ...
model.quantize(calibration_data)
model.save_quantized("llama-2-7b-gptq")
```

---

## 3. KV Cache: Speeding Up Generation

Transformers are autoregressive. To generate token $t$, we need all previous tokens $0 \dots t-1$.
Naive approach: Re-compute the entire sequence history for every new token. $O(N^2)$.

**Optimization**: Cache the Key (K) and Value (V) matrices from previous steps.
- At step $t$, only compute Q, K, V for the *new* token.
- Retrieve cached K, V for previous tokens.
- Perform Attention.

### Memory Cost of KV Cache
For a model with $L$ layers, $H$ heads, hidden size $D$, sequence length $S$, batch size $B$:
$$ \text{Memory} = 2 \times L \times H \times D_{head} \times S \times B \times \text{precision\_bytes} $$

For Llama-2-7B (32 layers, 4096 hidden, fp16):
- Approx 0.5 MB per token per batch item.
- For Batch=32, SeqLen=2048: ~32GB just for KV cache!
- **Implication**: KV cache often limits batch size more than model weights.

### Paged Attention (vLLM Innovation)
Traditional KV cache allocation is static (pre-allocate max seq len). Wasteful.
**vLLM** uses OS-style virtual memory paging:
- Split KV cache into fixed-size blocks.
- Dynamically allocate blocks as tokens are generated.
- Share blocks between sequences (useful for beam search or same prompt).
- **Result**: 100% memory utilization, higher throughput.

---

## 4. Speculative Decoding

Idea: Use a small "draft" model to guess the next $K$ tokens, then verify them with the large "target" model in parallel.

1. Draft model (fast) generates $x_1, x_2, x_3$.
2. Target model (slow) computes probabilities for all 3 in one forward pass.
3. Accept/Reject logic:
   - If $P_{target}(x_1) \approx P_{draft}(x_1)$, accept.
   - If rejected, resample.
4. **Speedup**: If draft is accurate, we generate $K$ tokens in the time of 1 target forward pass.

```python
# Native PyTorch Speculative Decoding (Accelerate)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to("cuda")
target_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to("cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Write a python function to sort a list.", return_tensors="pt").to("cuda")

# Assisted generation
outputs = target_model.generate(
    **inputs,
    assistant_model=draft_model,
    num_assistant_tokens=5,  # How many tokens to draft
    max_new_tokens=50
)
```

---

## 5. Production Serving Engines

Do not use `model.generate()` in production. It lacks concurrency control, batching, and streaming optimization. Use dedicated servers.

### 5.1 vLLM (High Throughput)
Best for high-throughput scenarios (batch processing, heavy traffic). Implements PagedAttention.

**Installation**:
```bash
pip install vllm
```

**Running a Server**:
```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9
```

**Client Usage**:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

**Key Features**:
- Continuous batching: As soon as one request finishes, start a new one in the same batch slot.
- PagedAttention: Efficient memory usage.
- Supports quantization (AWQ, GPTQ, SqueezeLLM).

### 5.2 Text Generation Inference (TGI)
Developed by Hugging Face. Optimized for text generation, supports Tensor Parallelism out of the box. Written in Rust/C++.

**Run with Docker**:
```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
  -e MODEL_ID=meta-llama/Llama-2-7b-hf \
  -e QUANTIZE=bitsandbytes \
  ghcr.io/huggingface/text-generation-inference:latest
```

**Key Features**:
- Flash Attention integration.
- Token streaming.
- Watermarking (for AI detection).
- Logits warping (temperature, top_p) handled efficiently.

### 5.3 Llama.cpp (CPU/Mac Inference)
Uses GGUF format (highly quantized). Runs on CPU, Apple Silicon, or GPU.
- Ideal for local deployment, edge devices, or MacBooks.
- Supports Q4_K_M, Q5_K_M quantizations.

```bash
# Convert HF model to GGUF
python convert.py meta-llama/Llama-2-7b-hf --outfile llama-2-7b.gguf --outtype q4_0

# Run inference
./main -m llama-2-7b.gguf -p "Hello world" -n 128
```

---

## 6. Deployment Strategies

### 6.1 Latency vs. Throughput
- **Latency**: Time to first token (TTFT) + time per token. Critical for chatbots.
  - *Optimization*: Smaller models, fewer layers, speculative decoding.
- **Throughput**: Total tokens generated per second across all users. Critical for batch analysis.
  - *Optimization*: Large batch sizes, vLLM, Tensor Parallelism.

### 6.2 Scaling Topology
1. **Replica Scaling**: Run multiple instances behind a load balancer (Kubernetes). Simplest.
2. **Tensor Parallelism**: Split one model across 4-8 GPUs for a single request. Needed for >30B models.
3. **Pipeline Parallelism**: Rare for inference due to bubble overhead, but useful for massive models on limited GPUs.

### 6.3 Monitoring Metrics
Track these in production:
- **TTFT (Time To First Token)**: User perceived latency.
- **TPOT (Time Per Output Token)**: Reading speed match (aim for <50ms/token).
- **Queue Depth**: How many requests waiting?
- **GPU Utilization**: Should be high for throughput, moderate for latency-sensitive apps.
- **KV Cache Hit Rate**: (If using prefix caching for system prompts).

---

## 7. Practical Exercise: Deploying a Quantized Model

**Goal**: Serve a 4-bit Llama-2-7B model using vLLM with high concurrency.

**Step 1: Prepare the Quantized Model**
Use `TheBloke` models from HuggingFace (pre-quantized) or quantize your own.
```bash
# Example: Using a pre-quantized AWQ model
MODEL="TheBloke/Llama-2-7B-Chat-AWQ"
```

**Step 2: Launch vLLM**
```bash
python -m vllm.entrypoints.api_server \
    --model $MODEL \
    --quantization awq \
    --port 8000 \
    --max-num-seqs 128 \
    --max-model-len 4096
```

**Step 3: Load Test**
Use `locust` or a simple script to simulate 50 concurrent users.
```python
# load_test.py
import concurrent.futures
import requests
import time

def send_request(i):
    start = time.time()
    resp = requests.post("http://localhost:8000/generate", json={
        "prompt": f"Question {i}: What is AI?",
        "max_tokens": 50
    })
    duration = time.time() - start
    return duration

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(send_request, i) for i in range(50)]
    results = [f.result() for f in futures]

print(f"Average Latency: {sum(results)/len(results):.2f}s")
print(f"Total Requests: {len(results)}")
```

---

## 8. Security & Safety in Deployment

1. **Input Validation**: Prevent prompt injection attacks. Filter malicious inputs before sending to model.
2. **Output Filtering**: Use classifiers to detect hate speech, PII, or toxicity before returning to user.
3. **Rate Limiting**: Protect against DoS and cost overruns.
4. **Model Watermarking**: Embed invisible signals to identify AI-generated text (supported in TGI).

---

## 9. Summary Checklist

| Technique | Benefit | Trade-off | Best For |
| :--- | :--- | :--- | :--- |
| **INT8/FP8** | 2x memory savings, 2x speed | Slight accuracy drop | General inference |
| **INT4 (AWQ/GPTQ)** | 4x memory savings | Calibration needed, HW specific | Consumer GPUs, Edge |
| **KV Cache** | Linear vs Quadratic time | High VRAM usage | Long context |
| **PagedAttention** | Max VRAM utilization | Implementation complexity | High concurrency |
| **Speculative Decoding** | 2-3x speedup | Needs draft model | Latency sensitive |
| **vLLM/TGI** | Production features | Extra infrastructure | Production APIs |

## Next Steps
In Tutorial 12, we will cover **MLOps, Automation & Governance**, focusing on CI/CD pipelines for models, model registries, compliance tracking, and automated evaluation gates.
