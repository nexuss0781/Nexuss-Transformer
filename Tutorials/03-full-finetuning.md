# Tutorial 03: Full Fine-Tuning - Complete Model Adaptation

## Overview

Full fine-tuning updates **all parameters** of a pre-trained model for your specific task. Unlike PEFT methods that freeze most weights, full fine-tuning allows the entire model to adapt to your domain.

### When to Use Full Fine-Tuning

| Scenario | Recommendation |
|----------|---------------|
| Large dataset (>10k examples) | ✅ Full Fine-Tuning |
| Domain significantly different from base model | ✅ Full Fine-Tuning |
| Limited GPU memory (<24GB) | ❌ Use PEFT/LoRA |
| Need maximum performance | ✅ Full Fine-Tuning |
| Multiple tasks from same base | ❌ Use PEFT (keep base intact) |

### Resource Requirements

**Minimum Hardware:**
- 7B model: 80GB+ GPU memory (with gradient checkpointing + mixed precision)
- 13B model: Multi-GPU setup (2-4x A100/H100)
- 70B model: 8+ GPU cluster required

**Optimization Techniques Required:**
- Gradient Checkpointing: Reduces memory by 60-70%
- Mixed Precision (FP16/BF16): 2x memory reduction
- DeepSpeed ZeRO: Shards optimizer states across GPUs
- CPU Offloading: Moves optimizer states to CPU RAM

---

## Section 1: Understanding Full Fine-Tuning Mechanics

### What Happens During Full Fine-Tuning?

```python
# Conceptual flow of full fine-tuning
class FullFineTuning:
    def __init__(self, model, tokenizer):
        self.model = model  # All parameters trainable
        self.tokenizer = tokenizer
        
    def training_step(self, batch):
        # Forward pass through ALL layers
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        # Compute loss
        loss = outputs.loss
        
        # Backward pass: gradients flow through EVERY parameter
        loss.backward()
        
        # Update ALL weights
        optimizer.step()
        
        return loss
```

### Memory Breakdown for Full Fine-Tuning

For a 7B model in FP16:

```
Model weights:          7B × 2 bytes = 14 GB
Gradients:              7B × 2 bytes = 14 GB
Optimizer states:       7B × 8 bytes = 56 GB (Adam: 2× momentum + variance)
Activations:            ~10-20 GB (depends on sequence length)
─────────────────────────────────────────────────────
Total (naive):          ~94-104 GB
```

**With Optimizations:**
```
Gradient Checkpointing: Activations reduced to ~3-5 GB
DeepSpeed ZeRO-2:       Optimizer states sharded across GPUs
Mixed Precision:        Weights + gradients in FP16
─────────────────────────────────────────────────────
Optimized (single GPU): ~40-50 GB (still needs A100 80GB)
Optimized (4 GPUs):     ~15-20 GB per GPU (feasible!)
```

---

## Section 2: Configuration for Full Fine-Tuning

### Basic Configuration

Create `configs/full_finetune_7b.yaml`:

```yaml
# Full Fine-Tuning Configuration for 7B Model
model:
  name_or_path: "meta-llama/Llama-2-7b-hf"
  model_type: "causal_lm"
  trust_remote_code: false

data:
  train_file: "data/instruction_dataset.jsonl"
  validation_file: "data/instruction_dataset_val.jsonl"
  preprocessing_num_workers: 4
  max_seq_length: 512
  pack_sequences: true

training:
  # Full fine-tuning: all parameters trainable
  tune_all_parameters: true
  
  # Critical optimizations for memory
  gradient_checkpointing: true
  fp16: true
  bf16: false  # Set true if using A100/H100
  
  # Batch sizing
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 2×8×num_gpus
  
  # Optimization
  optim: "adamw_torch"
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  
  # Training duration
  num_train_epochs: 3
  max_steps: -1  # -1 means use epochs
  
  # Logging and checkpoints
  logging_steps: 10
  save_steps: 500
  save_total_limit: 2
  evaluation_strategy: "steps"
  eval_steps: 500
  
  # DeepSpeed configuration (multi-GPU)
  deepspeed: "configs/deepspeed_zero2.json"
  
  # Dataloader
  dataloader_num_workers: 4
  dataloader_pin_memory: true

output:
  output_dir: "outputs/full_finetune_7b"
  overwrite_output_dir: true
  run_name: "llama2-7b-full-ft-instruct"
```

### DeepSpeed Configuration for Multi-GPU

Create `configs/deepspeed_zero2.json`:

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

**DeepSpeed Stages Explained:**

| Stage | Optimizer States | Gradients | Parameters | Memory Savings |
|-------|-----------------|-----------|------------|----------------|
| ZeRO-1 | Sharded | Replicated | Replicated | 4x |
| ZeRO-2 | Sharded | Sharded | Replicated | 8x |
| ZeRO-3 | Sharded | Sharded | Sharded | 16x+ |

---

## Section 3: Data Preparation for Full Fine-Tuning

### Instruction Dataset Format

Full fine-tuning requires high-quality, diverse data:

```jsonl
# data/instruction_dataset.jsonl
{"instruction": "Explain quantum entanglement", "input": "", "output": "Quantum entanglement is a phenomenon where..."}
{"instruction": "Translate to French", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous?"}
{"instruction": "Write Python code", "input": "Function to calculate factorial", "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"}
{"instruction": "Summarize article", "input": "[Long article text here...]", "output": "The article discusses..."}
```

### Advanced Data Preprocessing

```python
# scripts/prepare_full_ft_data.py
import json
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import argparse

def format_instruction(example):
    """Format data into instruction template"""
    if example.get('input', ''):
        text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        text = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    
    return {"text": text}

def prepare_dataset(
    data_files,
    tokenizer_name,
    max_seq_length=512,
    pack_sequences=True
):
    """Prepare dataset for full fine-tuning"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    datasets = []
    for file in data_files:
        ds = load_dataset('json', data_files=file, split='train')
        datasets.append(ds)
    
    # Combine all datasets
    combined = concatenate_datasets(datasets)
    
    # Format instructions
    formatted = combined.map(
        format_instruction,
        remove_columns=combined.column_names
    )
    
    # Tokenize
    def tokenize(example):
        tokens = tokenizer(
            example['text'],
            truncation=True,
            max_length=max_seq_length,
            padding=False
        )
        # Add EOS token
        tokens['input_ids'].append(tokenizer.eos_token_id)
        tokens['attention_mask'].append(1)
        return tokens
    
    tokenized = formatted.map(
        tokenize,
        remove_columns=['text']
    )
    
    # Sequence packing (optional but recommended)
    if pack_sequences:
        tokenized = pack_sequences_dataset(
            tokenized, 
            max_seq_length,
            tokenizer.eos_token_id
        )
    
    # Split train/validation
    train_test = tokenized.train_test_split(test_size=0.05, seed=42)
    
    return train_test['train'], train_test['test'], tokenizer

def pack_sequences_dataset(dataset, max_length, eos_token_id):
    """Pack multiple short sequences into one"""
    # Implementation for sequence packing
    # Combines multiple samples to fill context window
    packed_data = []
    current_pack = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for sample in dataset:
        if len(current_pack['input_ids']) + len(sample['input_ids']) <= max_length:
            # Add to current pack
            for key in ['input_ids', 'attention_mask']:
                current_pack[key].extend(sample[key])
            # Labels: -100 for padding/instruction, actual tokens for response
            labels = [-100] * len(sample['input_ids'])
            current_pack['labels'].extend(labels)
        else:
            # Save current pack and start new one
            if len(current_pack['input_ids']) > 0:
                # Pad to max_length
                padding_length = max_length - len(current_pack['input_ids'])
                for key in ['input_ids', 'attention_mask', 'labels']:
                    current_pack[key].extend([0 if key != 'labels' else -100] * padding_length)
                packed_data.append(current_pack)
            
            current_pack = {
                'input_ids': sample['input_ids'][:],
                'attention_mask': sample['attention_mask'][:],
                'labels': [-100] * len(sample['input_ids'])
            }
    
    return type(dataset).from_list(packed_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", nargs='+', required=True)
    parser.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    
    train_ds, val_ds, tokenizer = prepare_dataset(
        args.data_files,
        args.tokenizer,
        args.max_seq_length
    )
    
    train_ds.save_to_disk(f"{args.output_dir}/train")
    val_ds.save_to_disk(f"{args.output_dir}/val")
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✓ Saved {len(train_ds)} training samples")
    print(f"✓ Saved {len(val_ds)} validation samples")
```

---

## Section 4: Running Full Fine-Tuning

### Single GPU Training (24GB+)

```bash
# For 24GB GPU (RTX 3090/4090) - use smaller model or heavy optimizations
CUDA_VISIBLE_DEVICES=0 python src/trainer.py \
    --config configs/full_finetune_7b.yaml \
    --training.gradient_checkpointing true \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 16 \
    --training.fp16 true \
    --deepspeed configs/deepspeed_zero2.json
```

### Multi-GPU Training (Recommended)

```bash
# 4x A100 GPUs
torchrun --nproc_per_node=4 src/trainer.py \
    --config configs/full_finetune_7b.yaml \
    --deepspeed configs/deepspeed_zero2.json

# Or using accelerate
accelerate launch --config_file configs/accelerate_multi_gpu.yaml \
    src/trainer.py \
    --config configs/full_finetune_7b.yaml
```

### Using Our Framework's Trainer

```python
# scripts/run_full_finetune.py
from src.config import TrainingConfig
from src.trainer import Trainer
from src.models import load_model
from src.data import load_training_data

def main():
    # Load configuration
    config = TrainingConfig.from_yaml("configs/full_finetune_7b.yaml")
    
    # Load model - ALL parameters will be trainable
    model, tokenizer = load_model(
        config.model.name_or_path,
        model_type=config.model.model_type,
        load_in_8bit=False,  # Don't quantize for full fine-tuning
        device_map="auto"
    )
    
    # Verify all parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    assert trainable_params == total_params, "Not all parameters are trainable!"
    
    # Load data
    train_dataset, val_dataset = load_training_data(
        config.data.train_file,
        config.data.validation_file,
        tokenizer,
        max_length=config.data.max_seq_length
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(config.output.output_dir)
    tokenizer.save_pretrained(config.output.output_dir)
    
    print(f"✓ Full fine-tuning complete! Model saved to {config.output.output_dir}")

if __name__ == "__main__":
    main()
```

---

## Section 5: Monitoring and Debugging

### Key Metrics to Watch

```python
# Add to your training loop for monitoring
class TrainingMonitor:
    def __init__(self):
        self.gpu_memory_history = []
        self.loss_history = []
        self.grad_norm_history = []
    
    def on_step_end(self, step, logs):
        import torch
        
        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            self.gpu_memory_history.append({
                'step': step,
                'allocated_gb': allocated,
                'reserved_gb': reserved
            })
        
        # Loss
        self.loss_history.append({
            'step': step,
            'loss': logs.get('loss', 0)
        })
        
        # Gradient norm
        grad_norm = logs.get('grad_norm', 0)
        self.grad_norm_history.append({
            'step': step,
            'grad_norm': grad_norm
        })
        
        # Print warnings
        if grad_norm > 10.0:
            print(f"⚠️  High gradient norm at step {step}: {grad_norm:.2f}")
        if logs.get('loss', 0) > 5.0:
            print(f"⚠️  High loss at step {step}: {logs['loss']:.2f}")
```

### Common Issues and Solutions

#### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```yaml
# Reduce batch size
per_device_train_batch_size: 1

# Increase gradient accumulation
gradient_accumulation_steps: 16  # Maintain effective batch size

# Enable gradient checkpointing
gradient_checkpointing: true

# Use DeepSpeed ZeRO-3 for more memory savings
deepspeed: "configs/deepspeed_zero3.json"

# Enable CPU offload (slower but saves GPU memory)
# In deepspeed config:
"offload_optimizer": {
  "device": "cpu",
  "pin_memory": true
}
```

#### Issue 2: Loss Not Decreasing

**Symptoms:**
```
Step 100: loss=4.523
Step 200: loss=4.501
Step 300: loss=4.489  # Barely decreasing
```

**Solutions:**
```yaml
# Learning rate might be too low
learning_rate: 5.0e-5  # Try 1e-5 to 5e-5 for full fine-tuning

# Warmup might be too long
warmup_ratio: 0.01  # Reduce from 0.1

# Check data quality
# Ensure labels are correctly formatted
# Verify tokenization is correct

# Try different scheduler
lr_scheduler_type: "linear"  # Instead of cosine
```

#### Issue 3: Catastrophic Forgetting

**Symptoms:**
- Model performs well on fine-tuning task
- But loses general capabilities

**Solutions:**
```yaml
# Use regularization
weight_decay: 0.1  # Increase from 0.01

# Mix in general domain data
# Add 10-20% of pretraining data to fine-tuning dataset

# Use lower learning rate
learning_rate: 1.0e-5  # More conservative

# Early stopping
evaluation_strategy: "steps"
eval_steps: 100
load_best_model_at_end: true
metric_for_best_model: "eval_loss"
greater_is_better: false
```

---

## Section 6: Advanced Techniques

### Discriminative Learning Rates

Different layers learn at different rates:

```python
# scripts/discriminative_lr.py
def get_param_groups(model, base_lr=2e-5):
    """Assign different learning rates to different layers"""
    
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "embed_tokens" in n],
            "lr": base_lr * 0.1,  # Embeddings: lower LR
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "layers.0" in n or "layers.1" in n],
            "lr": base_lr * 0.5,  # Early layers: moderate LR
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "layers.2" in n or "layers.3" in n],
            "lr": base_lr,  # Middle layers: base LR
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "lm_head" in n],
            "lr": base_lr * 2.0,  # Head: higher LR
        },
    ]
    
    return param_groups

# Use in training
optimizer = torch.optim.AdamW(
    get_param_groups(model),
    weight_decay=0.01
)
```

### Layer-wise Learning Rate Decay

```python
def get_llrd_param_groups(model, base_lr=2e-5, decay_rate=0.95):
    """
    Layer-wise Learning Rate Decay (LLRD)
    Lower layers get lower learning rates
    """
    num_layers = model.config.num_hidden_layers
    
    param_groups = []
    
    # Embeddings
    param_groups.append({
        "params": [p for n, p in model.named_parameters() 
                  if "embed_tokens" in n],
        "lr": base_lr * (decay_rate ** (num_layers + 1))
    })
    
    # Each transformer layer
    for layer_idx in range(num_layers):
        param_groups.append({
            "params": [p for n, p in model.named_parameters() 
                      if f"layers.{layer_idx}." in n],
            "lr": base_lr * (decay_rate ** (num_layers - layer_idx))
        })
    
    # LM Head
    param_groups.append({
        "params": [p for n, p in model.named_parameters() 
                  if "lm_head" in n],
        "lr": base_lr
    })
    
    return param_groups
```

### Curriculum Learning

Start with easy examples, progress to harder ones:

```python
# scripts/curriculum_learning.py
class CurriculumDataset:
    def __init__(self, dataset, difficulty_field='difficulty'):
        self.dataset = dataset
        self.difficulty_field = difficulty_field
        
        # Sort by difficulty
        self.sorted_indices = sorted(
            range(len(dataset)),
            key=lambda i: dataset[i][difficulty_field]
        )
    
    def get_epoch_dataset(self, epoch, total_epochs):
        """Get dataset subset for current epoch"""
        progress = epoch / total_epochs
        
        # Calculate how much of dataset to use
        coverage = min(1.0, 0.3 + 0.7 * progress)  # Start at 30%, grow to 100%
        num_samples = int(len(self.dataset) * coverage)
        
        # Get easiest samples first
        indices = self.sorted_indices[:num_samples]
        
        return Subset(self.dataset, indices)

# Usage in training
curriculum_ds = CurriculumDataset(full_dataset)

for epoch in range(num_epochs):
    epoch_dataset = curriculum_ds.get_epoch_dataset(epoch, num_epochs)
    trainer.train(epoch_dataset)
```

---

## Section 7: Evaluation and Validation

### Comprehensive Evaluation Suite

```python
# scripts/evaluate_full_ft.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

class FullFineTuneEvaluator:
    def __init__(self, model_path, tokenizer_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def evaluate_generations(self, test_examples, max_new_tokens=256):
        """Evaluate generation quality"""
        results = []
        
        for example in test_examples:
            prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated.split("### Response:")[-1].strip()
            
            results.append({
                'instruction': example['instruction'],
                'expected': example['output'],
                'generated': response
            })
        
        return results
    
    def evaluate_perplexity(self, test_dataset):
        """Calculate perplexity on test set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_dataset:
                inputs = {
                    'input_ids': batch['input_ids'].unsqueeze(0).to(self.model.device),
                    'attention_mask': batch['attention_mask'].unsqueeze(0).to(self.model.device),
                    'labels': batch['labels'].unsqueeze(0).to(self.model.device)
                }
                
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity.item()
        }
    
    def save_evaluation_report(self, results, output_path):
        """Save comprehensive evaluation report"""
        report = {
            'generation_samples': results[:20],  # First 20 examples
            'metrics': self.evaluate_perplexity(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Evaluation report saved to {output_path}")

# Usage
evaluator = FullFineTuneEvaluator(
    "outputs/full_finetune_7b",
    "outputs/full_finetune_7b"
)

test_data = load_dataset('json', data_files='data/test.jsonl', split='train')
results = evaluator.evaluate_generations(list(test_data)[:100])
evaluator.save_evaluation_report(results, "evaluation_report.json")
```

### Benchmark Comparisons

Compare against base model:

```python
# Compare base vs fine-tuned
base_model = "meta-llama/Llama-2-7b-hf"
ft_model = "outputs/full_finetune_7b"

prompts = [
    "Explain machine learning in simple terms",
    "Write a haiku about programming",
    "What is the capital of France?",
    "Debug this Python code: def add(a,b): return a+b"
]

print("=" * 80)
print("BASE MODEL vs FINE-TUNED MODEL COMPARISON")
print("=" * 80)

for prompt in prompts:
    print(f"\nPrompt: {prompt}\n")
    
    # Base model
    base_output = generate(base_model, prompt)
    print(f"Base: {base_output}\n")
    
    # Fine-tuned model
    ft_output = generate(ft_model, prompt)
    print(f"Fine-tuned: {ft_output}\n")
    
    print("-" * 80)
```

---

## Section 8: Best Practices and Production Tips

### Checklist Before Full Fine-Tuning

- [ ] **Data Quality**: Clean, deduplicated, properly formatted
- [ ] **Hardware**: Sufficient GPU memory or multi-GPU setup
- [ ] **Backups**: Save base model before starting
- [ ] **Logging**: Set up experiment tracking (WandB, TensorBoard)
- [ ] **Checkpoints**: Configure frequent saving
- [ ] **Validation**: Hold-out validation set ready
- [ ] **Evaluation**: Automated evaluation pipeline prepared

### Cost Optimization

```yaml
# Use spot instances for cost savings
# AWS: Use spot instances with checkpointing
# GCP: Use preemptible VMs

# Estimated costs (as of 2024):
# 4x A100 (80GB) on-demand: ~$15-20/hour
# 4x A100 spot: ~$5-8/hour
# Training time for 7B model (3 epochs, 10k examples): ~10-20 hours
# Total cost: $50-400 depending on hardware choice

# Optimization tips:
# 1. Use gradient accumulation instead of larger batches
# 2. Enable CPU offload if GPU memory is tight
# 3. Use mixed precision (FP16/BF16)
# 4. Profile first with small dataset
```

### Version Control for Models

```bash
# Track model versions with Git LFS or DVC
git lfs install
git lfs track "outputs/*/pytorch_model.bin"

# Or use DVC
dvc init
dvc add outputs/full_finetune_7b
git add outputs/full_finetune_7b.dvc .gitignore
git commit -m "Add full fine-tuned model v1"
```

---

## Summary

Full fine-tuning provides maximum adaptation capability but requires significant resources:

**Pros:**
- Maximum performance on target task
- Complete model adaptation
- No architectural constraints

**Cons:**
- High memory requirements (needs multi-GPU for large models)
- Risk of catastrophic forgetting
- One model per task (no parameter sharing)

**When to choose full fine-tuning:**
- You have sufficient hardware (multi-GPU)
- Your domain is very different from base model
- You need maximum possible performance
- You're fine-tuning for a single, critical task

**Next Steps:**
- If resources are limited → See Tutorial 05 (PEFT/LoRA)
- If you need alignment → See Tutorial 06 (RLHF/DPO)
- If you want to continue learning → See Tutorial 08 (Continual Learning)

---

## Exercises

### Beginner
1. Run full fine-tuning on a small model (1-3B parameters) with a tiny dataset (100 examples)
2. Monitor GPU memory usage throughout training
3. Compare base model vs fine-tuned model on 5 test examples

### Intermediate
1. Fine-tune a 7B model on 10k instruction examples using 4 GPUs
2. Implement discriminative learning rates
3. Evaluate perplexity before and after fine-tuning

### Advanced
1. Implement curriculum learning with difficulty-sorted data
2. Experiment with layer-wise learning rate decay
3. Fine-tune on multiple domains and measure catastrophic forgetting
4. Optimize training to fit 7B model on a single 24GB GPU using DeepSpeed ZeRO-3 + CPU offload

---

## Additional Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Hugging Face Full Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Gradient Checkpointing Paper](https://arxiv.org/abs/1604.06174)
