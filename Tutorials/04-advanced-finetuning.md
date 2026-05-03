# Tutorial 04: Sequential Domain Adaptation

## Overview

This tutorial covers sequential domain adaptation strategies for adapting models to multiple domains over time while preventing catastrophic forgetting. We'll explore NTF's `ContinualLearningWrapper` with EWC regularization, incremental training approaches, and best practices for maintaining model performance across domains.

### What You'll Learn

- Sequential fine-tuning across multiple domains
- Elastic Weight Consolidation (EWC) for preventing forgetting
- ContinualLearningWrapper integration with NTF
- Domain adaptation strategies and evaluation
- Knowledge preservation techniques
- Practical workflows for multi-domain scenarios

---

## Section 1: Sequential Domain Adaptation Fundamentals

### Understanding Sequential Domain Adaptation

Sequential domain adaptation involves fine-tuning a model on multiple domains one after another, while preserving knowledge from previous domains. This is crucial when:

- You need to adapt to new domains as they emerge
- Training data from all domains cannot be loaded simultaneously
- Privacy or regulatory constraints prevent data mixing
- Computational resources limit joint training

**Benefits:**
- Incremental adaptation without full retraining
- Privacy-preserving (data stays in original domain)
- Resource efficient (train on one domain at a time)
- Flexible deployment (add domains as needed)

**Challenges:**
- Catastrophic forgetting of previous domains
- Determining optimal adaptation order
- Balancing specialization vs. generalization
- Evaluating cross-domain performance

### Continual Learning Scenarios

```python
from enum import Enum

class DomainAdaptationScenario(Enum):
    """Different sequential adaptation scenarios."""
    
    TEMPORAL = "temporal"
    # Domains arrive in temporal order
    # Example: News from 2020 → 2021 → 2022
    
    GEOGRAPHIC = "geographic"  
    # Different geographic regions
    # Example: US English → UK English → Australian English
    
    TECHNICAL = "technical"
    # Increasing technical complexity
    # Example: General → Scientific → Medical → Legal
    
    CUSTOMER_SEGMENT = "customer_segment"
    # Different customer verticals
    # Example: Retail → Finance → Healthcare → Manufacturing
```

---

## Section 2: Using ContinualLearningWrapper with EWC

NTF provides the `ContinualLearningWrapper` to simplify sequential domain adaptation with built-in EWC regularization.

### Basic Setup

```python
from ntf.utils.continual_learning import ContinualLearningWrapper, EWCConfig
from ntf.finetuning import FullFinetuneTrainer
from ntf.config import NTFConfig, ModelConfig, TrainingConfig

# Initialize base model
config = NTFConfig(
    model=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
    training=TrainingConfig(
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=3
    )
)

model, tokenizer = load_model_and_tokenizer(config.model)

# Wrap with ContinualLearningWrapper
wrapper = ContinualLearningWrapper(model)

# Configure EWC regularization
ewc_config = EWCConfig(
    ewc_lambda=1000.0,      # Strength of EWC regularization
    fisher_samples=200,      # Samples for Fisher estimation
    damping=0.1              # Damping factor
)

wrapper.configure_ewc(ewc_config)
```

### Sequential Domain Training Workflow

```python
# Domain 1: Code Generation
print("=== Training on Domain 1: Code ===")
code_trainer = FullFinetuneTrainer(
    model=wrapper.model,
    train_dataset=code_dataset,
    config=config
)
code_trainer.train()

# Save state after Domain 1
wrapper.save_state("domain1_code_checkpoint")

# Compute Fisher Information for Domain 1
wrapper.compute_fisher_information(code_dataloader)

# Domain 2: Math Reasoning (with EWC regularization)
print("\n=== Training on Domain 2: Math ===")
wrapper.apply_ewc_regularization(lambda_ewc=500.0)

math_trainer = FullFinetuneTrainer(
    model=wrapper.model,
    train_dataset=math_dataset,
    config=config
)
math_trainer.train()

# Save state after Domain 2
wrapper.save_state("domain2_math_checkpoint")

# Domain 3: Creative Writing
print("\n=== Training on Domain 3: Creative Writing ===")
wrapper.apply_ewc_regularization(lambda_ewc=300.0)

writing_trainer = FullFinetuneTrainer(
    model=wrapper.model,
    train_dataset=writing_dataset,
    config=config
)
writing_trainer.train()

# Final save
wrapper.save_state("final_multidomain_checkpoint")
```

### Advanced: Loading Previous States

```python
# Option 1: Continue from latest state
wrapper.load_state("domain2_math_checkpoint")
wrapper.apply_ewc_regularization(lambda_ewc=400.0)

# Option 2: Branch from earlier state
wrapper.load_state("domain1_code_checkpoint")
# Now train on a different domain branch
```
  fp16: true
  
  # Logging per task
  log_task_metrics: true
  
output:
  output_dir: "outputs/multitask_llama2_7b"
  run_name: "llama2-7b-multitask-v1"
```

### Multi-Task Trainer Implementation

```python
# src/multitask_trainer.py
import torch
import random
from typing import Dict, List, Optional
from transformers import Trainer
import numpy as np

class MultiTaskTrainer(Trainer):
    def __init__(self, task_weights: Dict[str, float] = None, **kwargs):
        super().__init__(**kwargs)
        self.task_weights = task_weights or {}
        self.task_losses = {}
        
    def get_train_dataloader(self):
        """Create balanced dataloader for multi-task learning"""
        if hasattr(self, 'task_datasets'):
            return self._create_balanced_dataloader()
        else:
            return super().get_train_dataloader()
    
    def _create_balanced_dataloader(self):
        """Sample from tasks based on weights"""
        task_loaders = {}
        for task_name, dataset in self.task_datasets.items():
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True
            )
            task_loaders[task_name] = iter(loader)
        
        # Calculate sampling probabilities
        total_weight = sum(self.task_weights.values())
        probs = {k: v/total_weight for k, v in self.task_weights.items()}
        
        while True:
            # Sample task based on weights
            task = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            
            try:
                batch = next(task_loaders[task])
                batch['task_id'] = task
                yield batch
            except StopIteration:
                # Reset exhausted loader
                task_loaders[task] = iter(torch.utils.data.DataLoader(
                    self.task_datasets[task],
                    batch_size=self.args.per_device_train_batch_size,
                    shuffle=True
                ))
                batch = next(task_loaders[task])
                batch['task_id'] = task
                yield batch
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with task-specific weighting"""
        task_id = inputs.pop('task_id', None)
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply task-specific weight
        if task_id and task_id[0] in self.task_weights:
            weight = self.task_weights[task_id[0]]
            loss = loss * weight
        
        # Track per-task losses
        if task_id:
            task = task_id[0]
            if task not in self.task_losses:
                self.task_losses[task] = []
            self.task_losses[task].append(loss.item())
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging for multi-task"""
        # Add per-task loss statistics
        for task, losses in self.task_losses.items():
            if losses:
                logs[f'{task}_loss'] = np.mean(losses[-100:])
        
        super().log(logs)
```

### Gradient Normalization for Multi-Task

Balance gradients across tasks to prevent dominance:

```python
# src/gradient_balance.py
import torch
import torch.nn as nn

class GradNormLoss(nn.Module):
    """
    Gradient Normalization for Multi-Task Learning
    Automatically adjusts task weights to balance gradient magnitudes
    """
    def __init__(self, model, task_names, alpha=1.5):
        super().__init__()
        self.model = model
        self.task_names = task_names
        self.alpha = alpha
        
        # Initialize task weights
        self.task_weights = nn.ParameterDict({
            task: nn.Parameter(torch.ones(1)) 
            for task in task_names
        })
        
        # Store initial losses for normalization
        self.initial_losses = {}
        
    def forward(self, inputs, task_id):
        """Compute weighted losses"""
        outputs = self.model(**inputs)
        base_loss = outputs.loss
        
        # Get current task weight
        weight = self.task_weights[task_id]
        weighted_loss = weight * base_loss
        
        return weighted_loss
    
    def update_weights(self, task_losses, gradients):
        """
        Update task weights based on gradient norms
        Tasks with smaller gradients get higher weights
        """
        # Compute gradient norms for each task
        grad_norms = {}
        for task in self.task_names:
            if task in gradients:
                grad_norms[task] = torch.norm(gradients[task]).item()
        
        # Compute average gradient norm
        avg_norm = np.mean(list(grad_norms.values()))
        
        # Update weights inversely proportional to gradient norm
        for task in self.task_names:
            if task in grad_norms:
                # Tasks with small gradients get higher weights
                target_norm = avg_norm / (grad_norms[task] + 1e-8)
                new_weight = self.task_weights[task].item() * (target_norm ** self.alpha)
                self.task_weights[task].data.fill_(new_weight)
        
        # Normalize weights
        total = sum(w.value for w in self.task_weights.values())
        for task in self.task_names:
            self.task_weights[task].data.fill_(
                self.task_weights[task].item() / total * len(self.task_names)
            )
```

---

## Section 2: Domain Adaptation

### Two-Stage Domain Adaptation

Adapt pre-trained models to new domains efficiently:

```yaml
# configs/domain_adaptation.yaml
# Stage 1: Continue pre-training on domain corpus
stage1:
  type: "continued_pretraining"
  data:
    file: "data/domain_corpus.txt"  # Raw text, no labels needed
    max_seq_length: 512
  
  training:
    learning_rate: 1.0e-4
    num_train_epochs: 5
    per_device_train_batch_size: 32
    mlm_probability: 0.15  # For masked LM
  
  output: "outputs/domain_adapted_base"

# Stage 2: Fine-tune on downstream task
stage2:
  type: "instruction_finetuning"
  base_model: "outputs/domain_adapted_base"
  data:
    file: "data/domain_instructions.jsonl"
    max_seq_length: 512
  
  training:
    learning_rate: 2.0e-5
    num_train_epochs: 3
    per_device_train_batch_size: 4
  
  output: "outputs/domain_specialized_model"
```

### Domain Adaptation Script

```python
# scripts/domain_adaptation.py
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

def continued_pretraining(base_model, domain_corpus, output_dir):
    """
    Stage 1: Continue pre-training on domain-specific text
    """
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load raw domain text
    dataset = load_dataset('text', data_files=domain_corpus, split='train')
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Data collator for MLM-style training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # False for causal LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_strategy="epoch",
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

def evaluate_domain_adaptation(adapted_model, base_model, test_examples):
    """Compare adapted vs base model on domain knowledge"""
    
    prompts = [
        "Explain the key concepts in quantum physics",
        "What are the main challenges in healthcare AI?",
        "Describe the process of drug discovery"
    ]
    
    print("Domain Adaptation Comparison")
    print("=" * 80)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}\n")
        
        # Base model
        base_response = generate(base_model, prompt, max_tokens=200)
        print(f"Base Model: {base_response}\n")
        
        # Adapted model
        adapted_response = generate(adapted_model, prompt, max_tokens=200)
        print(f"Adapted Model: {adapted_response}\n")
        
        print("-" * 80)
```

### Progressive Domain Adaptation

Gradually adapt through intermediate domains:

```python
# scripts/progressive_adaptation.py
adaptation_stages = [
    {
        "name": "general_knowledge",
        "data": "data/wikipedia_subset.txt",
        "epochs": 2,
        "lr": 2e-5
    },
    {
        "name": "science_domain",
        "data": "data/science_papers.txt",
        "epochs": 3,
        "lr": 1e-5
    },
    {
        "name": "medical_domain",
        "data": "data/medical_texts.txt",
        "epochs": 3,
        "lr": 1e-5
    },
    {
        "name": "clinical_instructions",
        "data": "data/clinical_qa.jsonl",
        "epochs": 2,
        "lr": 5e-6
    }
]

current_model = "meta-llama/Llama-2-7b-hf"

for stage in adaptation_stages:
    print(f"Starting stage: {stage['name']}")
    
    output_dir = f"outputs/adaptation_{stage['name']}"
    
    # Train on this stage
    train_on_data(
        model_path=current_model,
        data_file=stage['data'],
        epochs=stage['epochs'],
        lr=stage['lr'],
        output_dir=output_dir
    )
    
    # Use output as input for next stage
    current_model = output_dir
    
print(f"Final adapted model: {current_model}")
```

---

## Section 3: Instruction Tuning at Scale

### Building Large Instruction Datasets

Combine multiple sources for comprehensive instruction tuning:

```python
# scripts/build_instruction_dataset.py
from datasets import load_dataset, concatenate_datasets
import json

def load_alpaca_format(dataset_name):
    """Load dataset in Alpaca format"""
    if dataset_name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
    elif dataset_name == "dolly":
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    elif dataset_name == "oasst1":
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        # Convert to instruction format
        ds = ds.filter(lambda x: x['rank'] == 0)  # Keep only top-ranked responses
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return ds

def normalize_to_instruction_format(example, source):
    """Normalize different formats to unified instruction format"""
    
    if source in ["alpaca", "dolly"]:
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
    elif source == "oasst1":
        instruction = example.get('text', '')
        input_text = ''
        output = example.get('assistant', '')
    
    # Create unified format
    if input_text:
        formatted = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        formatted = f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    return {"text": formatted, "source": source}

def build_comprehensive_dataset(sources, output_file, max_samples_per_source=50000):
    """Build large instruction dataset from multiple sources"""
    
    all_datasets = []
    
    for source in sources:
        print(f"Loading {source}...")
        
        ds = load_alpaca_format(source)
        
        # Normalize format
        ds = ds.map(
            lambda x: normalize_to_instruction_format(x, source),
            remove_columns=ds.column_names
        )
        
        # Limit samples per source for balance
        if len(ds) > max_samples_per_source:
            ds = ds.shuffle(seed=42).select(range(max_samples_per_source))
        
        all_datasets.append(ds)
        print(f"  ✓ Loaded {len(ds)} samples from {source}")
    
    # Combine all datasets
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    
    # Save
    combined.to_json(output_file, orient="records", lines=True)
    
    print(f"\n✓ Total dataset: {len(combined)} samples")
    print(f"✓ Saved to {output_file}")
    
    return combined

# Usage
sources = ["alpaca", "dolly", "oasst1", "selfinstruct"]
dataset = build_comprehensive_dataset(
    sources,
    "data/comprehensive_instructions.jsonl",
    max_samples_per_source=50000
)
```

### Self-Instruct for Data Augmentation

Generate synthetic instructions using the model itself:

```python
# scripts/self_instruct.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SelfInstructGenerator:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Seed tasks for bootstrapping
        self.seed_tasks = [
            "Write a poem about nature",
            "Explain photosynthesis",
            "Debug this Python code",
            "Translate to French",
            "Summarize the article"
        ]
    
    def generate_instructions(self, seed_tasks, num_instructions=1000):
        """Generate new instructions from seed tasks"""
        
        generated = []
        
        for i in range(num_instructions):
            # Create prompt for instruction generation
            prompt = self._create_generation_prompt(seed_tasks + generated[-10:])
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_instruction = self._extract_instruction(generated_text)
            
            if new_instruction and self._is_valid_instruction(new_instruction):
                generated.append(new_instruction)
                print(f"Generated {len(generated)} instructions...")
        
        return generated
    
    def _create_generation_prompt(self, existing_tasks):
        """Create prompt for generating new instructions"""
        prompt = "Here are some example instructions:\n\n"
        for i, task in enumerate(existing_tasks[-5:], 1):
            prompt += f"{i}. {task}\n"
        prompt += "\nGenerate a new, diverse instruction that is different from the above:\n"
        return prompt
    
    def _extract_instruction(self, text):
        """Extract instruction from generated text"""
        # Simple extraction - can be improved with parsing
        lines = text.strip().split('\n')
        for line in lines:
            if line.strip() and not line.startswith(('Here', 'Generate', str(range(10)))):
                return line.strip()
        return None
    
    def _is_valid_instruction(self, instruction):
        """Filter valid instructions"""
        if len(instruction) < 10 or len(instruction) > 200:
            return False
        if any(word in instruction.lower() for word in ['generate', 'instruction', 'example']):
            return False
        return True

# Usage
generator = SelfInstructGenerator("meta-llama/Llama-2-7b-hf")
instructions = generator.generate_instructions(
    generator.seed_tasks,
    num_instructions=1000
)

# Save generated instructions
with open("data/self_generated_instructions.txt", "w") as f:
    for inst in instructions:
        f.write(inst + "\n")
```

---

## Section 4: Contrastive Learning for Fine-Tuning

### SimPO (Simple Preference Optimization)

Direct preference optimization without reward modeling:

```yaml
# configs/simpo_training.yaml
model:
  name_or_path: "meta-llama/Llama-2-7b-hf"

data:
  preference_file: "data/preference_pairs.jsonl"
  # Format: {"prompt": "...", "chosen": "...", "rejected": "..."}

training:
  method: "simpo"
  
  # SimPO-specific parameters
  simpo_beta: 0.5  # Controls KL penalty
  simpo_gamma: 0.5  # Margin parameter
  
  # Standard params
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-7  # Lower LR for DPO/SimPO
  num_train_epochs: 3
  
  fp16: true
  gradient_checkpointing: true

output:
  output_dir: "outputs/llama2-7b-simpo"
```

### SimPO Implementation

```python
# src/simpo_trainer.py
import torch
import torch.nn.functional as F
from transformers import Trainer

class SimPOTrainer(Trainer):
    def __init__(self, beta=0.5, gamma=0.5, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        SimPO Loss: Direct preference optimization with implicit reward
        """
        # Prepare chosen and rejected inputs
        chosen_inputs = {
            'input_ids': inputs['chosen_input_ids'],
            'attention_mask': inputs['chosen_attention_mask'],
            'labels': inputs['chosen_labels']
        }
        
        rejected_inputs = {
            'input_ids': inputs['rejected_input_ids'],
            'attention_mask': inputs['rejected_attention_mask'],
            'labels': inputs['rejected_labels']
        }
        
        # Get log probabilities for chosen and rejected
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)
        
        # Calculate average log prob over sequence length
        chosen_log_probs = self._get_batch_logps(
            chosen_outputs.logits,
            chosen_inputs['labels']
        )
        
        rejected_log_probs = self._get_batch_logps(
            rejected_outputs.logits,
            rejected_inputs['labels']
        )
        
        # SimPO loss
        # L = -log sigmoid(beta * (r_chosen - r_rejected) - gamma)
        logits = self.beta * (chosen_log_probs - rejected_log_probs) - self.gamma
        loss = -F.logsigmoid(logits).mean()
        
        return (loss, {'chosen_logps': chosen_log_probs, 'rejected_logps': rejected_log_probs}) \
               if return_outputs else loss
    
    def _get_batch_logps(self, logits, labels, average=True):
        """Calculate log probabilities for batch"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        if average:
            # Average over sequence length (SimPO characteristic)
            return token_log_probs.sum(dim=-1) / (shift_labels != -100).sum(dim=-1)
        else:
            return token_log_probs.sum(dim=-1)
```

### DPO (Direct Preference Optimization)

Alternative to RLHF without explicit reward model:

```python
# src/dpo_trainer.py
class DPOTrainer(Trainer):
    def __init__(self, ref_model, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """DPO Loss"""
        
        # Get policy and reference log probs
        policy_chosen_logps, policy_rejected_logps = self._get_logps(
            model, 
            inputs
        )
        
        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self._get_logps(
                self.ref_model,
                inputs
            )
        
        # Calculate implicit rewards
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        
        # DPO loss
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()
        
        # Accuracy metric
        rewards_accuracy = (logits > 0).float().mean()
        
        return (loss, {'rewards_accuracy': rewards_accuracy}) \
               if return_outputs else loss
    
    def _get_logps(self, model, inputs):
        """Get log probabilities for chosen and rejected"""
        # Similar to SimPO but with different aggregation
        # Implementation details omitted for brevity
        pass
```

---

## Section 5: Active Learning for Data Selection

### Uncertainty Sampling

Select most informative samples for labeling:

```python
# scripts/active_learning.py
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActiveLearningSelector:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def select_uncertain_samples(self, unlabeled_pool, num_select=100):
        """Select samples with highest uncertainty"""
        
        uncertainties = []
        
        for i, sample in enumerate(unlabeled_pool):
            # Get model's prediction entropy
            entropy = self._calculate_entropy(sample)
            uncertainties.append((i, entropy))
        
        # Sort by uncertainty (highest first)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Select top uncertain samples
        selected_indices = [idx for idx, _ in uncertainties[:num_select]]
        
        return [unlabeled_pool[i] for i in selected_indices]
    
    def _calculate_entropy(self, text):
        """Calculate prediction entropy for a sample"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get probabilities for next token
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            return entropy.item()
    
    def diversity_sampling(self, unlabeled_pool, num_select=100):
        """Select diverse samples using embeddings"""
        
        # Get embeddings for all samples
        embeddings = []
        for sample in unlabeled_pool:
            emb = self._get_embedding(sample)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_select, random_state=42)
        kmeans.fit(embeddings)
        
        # Select sample closest to each cluster center
        selected = []
        for i in range(num_select):
            cluster_mask = kmeans.labels_ == i
            cluster_indices = np.where(cluster_mask)[0]
            
            # Find sample closest to centroid
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            
            selected.append(unlabeled_pool[closest_idx])
        
        return selected
    
    def _get_embedding(self, text):
        """Get sentence embedding"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state, mean pooling
            hidden = outputs.hidden_states[-1]
            embedding = hidden.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding

# Usage
selector = ActiveLearningSelector("meta-llama/Llama-2-7b-hf")

# From large unlabeled pool, select most informative 1000 samples
unlabeled_data = load_unlabeled_corpus("data/unlabeled.txt")
selected = selector.select_uncertain_samples(unlabeled_data, num_select=1000)

# Send selected samples for human labeling
save_for_labeling(selected, "data/to_label.jsonl")
```

---

## Section 6: Knowledge Distillation

### Distilling from Larger Models

Transfer knowledge from teacher to student:

```yaml
# configs/distillation.yaml
teacher:
  model: "meta-llama/Llama-2-70b-hf"
  temperature: 2.0

student:
  model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  # Will learn from teacher

data:
  file: "data/distillation_corpus.txt"
  max_seq_length: 512

training:
  # Distillation-specific
  distillation_alpha: 0.5  # Balance between hard and soft targets
  distillation_temperature: 2.0
  
  # Standard
  learning_rate: 1.0e-4
  num_train_epochs: 5
  per_device_train_batch_size: 32

output:
  output_dir: "outputs/distilled_1b"
```

### Distillation Trainer

```python
# src/distillation_trainer.py
import torch
import torch.nn.functional as F
from transformers import Trainer, AutoModelForCausalLM

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Distillation loss combining hard and soft targets"""
        
        # Student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Teacher outputs (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Hard loss (cross-entropy with true labels)
        hard_loss = student_outputs.loss
        
        # Soft loss (KL divergence from teacher)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return (loss, {'hard_loss': hard_loss, 'soft_loss': soft_loss}) \
               if return_outputs else loss

# Usage
teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

trainer = DistillationTrainer(
    model=student_model,
    teacher_model=teacher,
    alpha=0.7,
    temperature=2.0,
    train_dataset=train_dataset,
    args=training_args
)

trainer.train()
```

---

## Summary

Advanced fine-tuning techniques enable:

1. **Multi-task learning**: Single model for multiple tasks
2. **Domain adaptation**: Specialize models for specific domains
3. **Instruction tuning at scale**: Build comprehensive instruction datasets
4. **Contrastive learning**: Direct preference optimization (DPO/SimPO)
5. **Active learning**: Select most informative data
6. **Knowledge distillation**: Compress large models into smaller ones

### When to Use Each Technique

| Technique | Use Case | Resource Requirements |
|-----------|----------|----------------------|
| Multi-task | Multiple related tasks | Medium-High |
| Domain adaptation | Specialized domain | Medium |
| Instruction tuning | General capability | High (needs large data) |
| DPO/SimPO | Alignment without RLHF | Medium |
| Active learning | Limited labeling budget | Low-Medium |
| Distillation | Model compression | High (needs large teacher) |

---

## Exercises

### Beginner
1. Create a simple multi-task dataset with 2 tasks
2. Fine-tune on combined dataset
3. Evaluate on both tasks separately

### Intermediate
1. Implement domain adaptation with 2 stages
2. Build instruction dataset from 3 sources
3. Compare DPO vs supervised fine-tuning

### Advanced
1. Implement gradient normalization for multi-task
2. Build active learning pipeline with uncertainty sampling
3. Distill 7B model into 1B model and compare performance

---

## Next Steps

- Tutorial 05: PEFT and LoRA (already covered)
- Tutorial 06: Reinforcement Learning from Human Feedback (RLHF)
- Tutorial 07: Reward Modeling and Preference Learning
- Tutorial 08: Continual Learning and Catastrophic Forgetting Prevention
