# Tutorial 015: Scaling to Production - From Experiment to Deployment

## 📌 Overview

**Level**: Advanced  
**Duration**: 60 minutes  
**Prerequisites**: Tutorials 001-014 completed

Learn how to scale your models from single-GPU experiments to production deployments serving millions of users.

---

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Understand distributed training strategies
- Implement data parallel training
- Deploy models for inference at scale
- Optimize inference performance
- Monitor production systems

---

## 1. Scaling Training: Distributed Strategies

### Types of Parallelism

| Strategy | Best For | Memory Efficiency | Communication | Complexity |
|----------|----------|-------------------|---------------|------------|
| **Data Parallel** | Small-medium models (<7B) | Medium | Low | Easy |
| **Tensor Parallel** | Large models (>7B) | High | Very High | Hard |
| **Pipeline Parallel** | Very large models | High | Medium | Medium |
| **Hybrid (3D)** | Massive models (>100B) | Highest | Highest | Very Hard |

### Data Parallel Implementation

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size):
    setup_distributed(rank, world_size)
    
    model = NexussTransformer(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for batch in dataloader:
        loss = compute_loss(ddp_model, batch)
        loss.backward()
        optimizer.step()
    
    dist.destroy_process_group()

# Launch: torchrun --nproc_per_node=4 train_ddp.py
```

---

## 2. Memory Optimization Techniques

### Gradient Checkpointing

```python
config = NTFConfig(gradient_checkpointing=True)
model = NexussTransformer(config)
```

**Memory Savings**: ~60% reduction | **Compute Overhead**: ~20% slower

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast(dtype=torch.bfloat16):
        outputs = model(batch["input_ids"])
        loss = compute_loss(outputs, batch["labels"])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Memory Savings**: 50% | **Speed Improvement**: 2-3× faster

---

## 3. Inference Optimization

### KV Cache for Faster Generation

```python
class OptimizedGenerator:
    def generate(self, prompt, max_length=100):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        generated = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        return tokenizer.decode(generated[0])
```

**Speed Improvement**: 5-10× faster for long sequences

### Batched Inference

```python
def generate_batch(self, prompts):
    grouped = self._group_by_length(prompts)
    all_outputs = []
    
    for batch_prompts in grouped:
        batch_inputs = self._pad_batch(batch_prompts)
        with torch.no_grad():
            outputs = model.generate(batch_inputs, max_new_tokens=100)
        all_outputs.extend(outputs)
    
    return all_outputs
```

---

## 4. Model Quantization for Deployment

### Post-Training Quantization

```python
import torch.quantization as quantization

model_int8 = quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

**Size Reduction**: 4× smaller | **Speed**: 2-3× faster on CPU | **Accuracy Loss**: <1%

### INT4 Quantization (GPTQ/AWQ)

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(bits=4, group_size=128)
model = AutoGPTQForCausalLM.from_pretrained("nexuss-base", quantize_config)
model.quantize(calibration_dataset)
model.save_quantized("nexuss-int4")
```

**Size Reduction**: 8× smaller | **Speed**: 3-4× faster | **Accuracy Loss**: 2-5%

---

## 5. Production Deployment Patterns

### FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = NexussTransformer.from_pretrained("nexuss-base").cuda().eval()
    tokenizer = load_tokenizer()

@app.post("/generate")
async def generate(request: GenerationRequest):
    inputs = tokenizer.encode(request.prompt, return_tensors="pt").cuda()
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=request.max_tokens)
    return {"text": tokenizer.decode(outputs[0])}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Load Balancing with Docker

```yaml
# docker-compose.yml
services:
  model-server-1:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  nginx:
    image: nginx:latest
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

---

## 6. Monitoring and Observability

### Key Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('inference_requests_total', 'Total requests', ['status'])
REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Latency', buckets=[0.01, 0.1, 0.5, 1.0, 2.0])
GPU_MEMORY = Gauge('gpu_memory_usage_bytes', 'GPU memory', ['gpu_id'])
TOKENS_GENERATED = Histogram('tokens_generated', 'Tokens per request', buckets=[10, 50, 100, 500])
```

### Alerting Rules

```yaml
# prometheus_alerts.yml
groups:
  - name: model_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
      
      - alert: HighErrorRate
        expr: rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
```

---

## 7. Cost Optimization

### Instance Selection Guide

| Instance Type | GPU | VRAM | Hourly Cost | Best For |
|--------------|-----|------|-------------|----------|
| g4dn.xlarge | T4 | 16GB | $0.52 | Small models, dev |
| g5.xlarge | A10G | 24GB | $1.01 | Medium models |
| p4d.24xlarge | A100 × 8 | 320GB | $32.77 | Large-scale training |
| inf1.xlarge | Inferentia | - | $0.23 | Inference optimized |

### Kubernetes Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 8. Security Considerations

### Input Validation

```python
def validate_input(prompt: str, max_length: int = 4096) -> bool:
    if len(prompt) > max_length:
        return False
    
    dangerous_patterns = ["<script>", "javascript:", "data:", "{{", "{%"]
    for pattern in dangerous_patterns:
        if pattern.lower() in prompt.lower():
            return False
    
    return True
```

### Rate Limiting

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate(request: Request, gen_request: GenerationRequest):
    # ... generation code ...
```

---

## 9. Complete Production Pipeline

```python
class ProductionPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.model = self._load_optimized_model()
        self.setup_monitoring()
        self.warm_up()
    
    def _load_optimized_model(self):
        if self.config.quantization == "int8":
            model = load_quantized_int8(self.config.model_path)
        elif self.config.quantization == "int4":
            model = load_quantized_int4(self.config.model_path)
        else:
            model = NexussTransformer.from_pretrained(self.config.model_path)
        
        model = model.cuda().eval()
        if self.config.use_compile:
            model = torch.compile(model)
        return model
    
    def generate(self, prompt: str, **kwargs) -> dict:
        if not validate_input(prompt):
            raise ValueError("Invalid input")
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=kwargs.get('max_tokens', 100),
            )
        latency = time.time() - start_time
        
        text = self.tokenizer.decode(outputs[0])
        self.log_metrics(latency, len(outputs[0]))
        
        return {
            "text": text,
            "latency_ms": latency * 1000,
            "tokens_generated": len(outputs[0]) - len(inputs[0]),
        }

# Usage
pipeline = ProductionPipeline("production_config.yaml")
result = pipeline.generate("Explain quantum computing")
print(result)
```

---

## 📚 Summary

### Key Takeaways

✅ **Distributed training** enables scaling to billions of parameters  
✅ **Mixed precision** provides 2-3× speedup with minimal accuracy loss  
✅ **KV caching** is essential for efficient text generation  
✅ **Quantization** reduces model size 4-8× for deployment  
✅ **Monitoring** is critical for production reliability  

### Quick Reference

```python
# Standard production setup
model = load_quantized_model("path")
model = model.cuda().eval()
model = torch.compile(model)  # PyTorch 2.0+

# Inference
with torch.no_grad():
    outputs = model.generate(inputs, use_cache=True)
```

---

**Congratulations!** You've completed the entire Nexuss AI Training Framework tutorial series!

You now have comprehensive knowledge covering:
- Blank-slate model training
- Fine-tuning techniques (full and PEFT)
- Reward modeling and alignment (RLHF, DPO)
- Validation and testing
- Version management
- Continual learning
- Production deployment

🎓 **You're ready to build production-grade LLM systems!**

---

*End of Tutorial Series*  
*Nexuss AI Engineering Team*
