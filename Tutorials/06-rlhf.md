# Tutorial 06: Reinforcement Learning from Human Feedback (RLHF)

## Overview

Reinforcement Learning from Human Feedback (RLHF) aligns language models with human preferences and values. This tutorial covers the complete RLHF pipeline using **Nexuss Transformer Framework's** native `RewardModel`, `PreferenceDataset`, and `RLHFPipeline` utilities.

### What is RLHF?

RLHF is a three-stage process:

1. **Supervised Fine-Tuning (SFT)**: Train on high-quality demonstrations
2. **Reward Modeling**: Learn human preferences from comparisons
3. **Reinforcement Learning**: Optimize policy using learned reward

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Pre-trained   │────▶│  SFT Model       │────▶│  RL Policy      │
│     Model       │     │  (Instruction)   │     │  (PPO)          │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                        │
                               ▼                        ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Reward Model    │◀────│  Generate &     │
                        │  (Preferences)   │     │  Score          │
                        └──────────────────┘     └─────────────────┘
```

### When to Use RLHF

| Scenario | Recommendation |
|----------|---------------|
| Need alignment with human values | ✅ RLHF |
| Reduce harmful outputs | ✅ RLHF |
| Improve helpfulness/honesty | ✅ RLHF |
| Simple task fine-tuning | ❌ Use SFT only |
| Limited annotation budget | ❌ Use DPO/SimPO |
| Need fast iteration | ❌ Use DPO (simpler) |

---

## Section 1: The NTF RLHF Pipeline

### RLHF Workflow

1. **Supervised Fine-Tuning (SFT)**: Train on instruction-following data
2. **Reward Modeling**: Train reward model on human preference data
3. **RL Optimization**: Use PPO to optimize policy against reward model
4. **Evaluation**: Assess alignment with human preferences

NTF provides native components for each stage, ensuring consistency and reproducibility.

### Complete RLHF Example with NTF

```python
from ntf.reward import RewardModel, PreferenceDataset, RLHFPipeline
from ntf.models import ModelRegistry
from ntf.config import RewardConfig

# 1. Load base model
registry = ModelRegistry(model_config)
base_model, tokenizer = registry.load_model_and_tokenizer()

# 2. Initialize NTF's RewardModel
reward_config = RewardConfig(
    base_model_name="meta-llama/Llama-2-7b-hf",
    num_labels=1,
    pad_token_id=tokenizer.pad_token_id
)
reward_model = RewardModel(reward_config)
reward_model.load_base_model(base_model)

# 3. Load preference data with NTF utilities
pref_dataset = PreferenceDataset(
    data_path="preferences.jsonl",
    tokenizer=tokenizer,
    max_length=512
)

# 4. Train reward model
from ntf.reward.trainer import RewardTrainer
reward_trainer = RewardTrainer(
    model=reward_model,
    dataset=pref_dataset,
    config=reward_config
)
reward_trainer.train()

# 5. Use in RLHF pipeline
pipeline = RLHFPipeline(
    policy_model=policy_model,
    reward_model=reward_model,
    reference_model=ref_model,
    tokenizer=tokenizer
)

pipeline.run_ppo(
    prompts=prompts,
    num_iterations=100,
    kl_coeff=0.2
)
```

---

## Section 2: Collecting Preference Data

### Preference Data Format

```jsonl
# data/preference_data.jsonl
{
  "prompt": "Write a poem about nature",
  "chosen": "Nature's beauty unfolds each day...",
  "rejected": "The natural world is nice...",
  "annotator_id": "worker_001",
  "rating_confidence": 0.9
}

{
  "prompt": "Explain quantum physics simply",
  "chosen": "Imagine tiny particles that can be in multiple states...",
  "rejected": "Quantum physics is the study of matter and energy...",
  "annotator_id": "worker_002",
  "rating_confidence": 0.85
}
```

### Best Practices for Data Collection

**Quality Guidelines:**
- Clear preference criteria (helpfulness, honesty, harmlessness)
- Multiple annotators per sample (3-5 recommended)
- Resolve disagreements through adjudication
- Include diverse prompt types

**Quantity Requirements:**
- Minimum: 1,000 preference pairs
- Recommended: 10,000-50,000 pairs
- High-quality > quantity

### Data Augmentation for Preferences

```python
# scripts/augment_preferences.py
import random
from datasets import load_dataset

def augment_preference_data(dataset, augmentation_factor=3):
    """Augment preference data with variations"""
    
    augmented = []
    
    for sample in dataset:
        # Original
        augmented.append(sample)
        
        # Variation 1: Swap chosen/rejected with inverted labels
        if random.random() < 0.3:  # Only sometimes to avoid bias
            augmented.append({
                'prompt': sample['prompt'],
                'chosen': sample['rejected'],
                'rejected': sample['chosen'],
                'annotator_id': f"{sample['annotator_id']}_swap"
            })
        
        # Variation 2: Paraphrase prompt
        if random.random() < 0.2:
            paraphrased_prompt = paraphrase(sample['prompt'])
            augmented.append({
                'prompt': paraphrased_prompt,
                'chosen': sample['chosen'],
                'rejected': sample['rejected'],
                'annotator_id': f"{sample['annotator_id']}_paraphrase"
            })
        
        # Add more variations as needed...
    
    return augmented

def paraphrase(text):
    """Simple paraphrasing (use LLM for better quality)"""
    # In practice, use an LLM to generate paraphrases
    synonyms = {
        'explain': 'describe',
        'write': 'compose',
        'what': 'which',
        'how': 'in what way'
    }
    
    words = text.split()
    paraphrased = [synonyms.get(word.lower(), word) for word in words]
    return ' '.join(paraphrased)

# Usage
raw_data = load_dataset('json', data_files='data/raw_preferences.jsonl', split='train')
augmented_data = augment_preference_data(list(raw_data), augmentation_factor=2)

# Save
with open('data/augmented_preferences.jsonl', 'w') as f:
    for sample in augmented_data:
        f.write(json.dumps(sample) + '\n')
```

---

## Section 2: Building the Reward Model

### Reward Model Architecture

The reward model takes a prompt-response pair and outputs a scalar score:

```python
# src/reward_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class RewardModel(nn.Module):
    def __init__(self, base_model_name, num_labels=1):
        super().__init__()
        
        # Load base model with classification head
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels
        )
        
        # Get hidden size for custom head if needed
        config = self.base_model.config
        self.hidden_size = config.hidden_size
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass returning reward scores
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional reward labels for training
        
        Returns:
            rewards: Scalar reward for each sample [batch_size]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        rewards = outputs.logits.squeeze(-1)  # [batch_size]
        
        return {'rewards': rewards, 'loss': outputs.loss}
    
    def compute_reward(self, input_ids, attention_mask=None):
        """Compute reward for generated sequences"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['rewards']
```

### Training the Reward Model

```yaml
# configs/reward_model.yaml
model:
  base_model: "meta-llama/Llama-2-7b-hf"
  model_type: "reward_model"

data:
  preference_file: "data/preference_pairs.jsonl"
  max_seq_length: 512
  validation_split: 0.1

training:
  learning_rate: 1.0e-5
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  
  # Reward model specific
  loss_type: "pairwise_ranking"  # pairwise or regression
  margin: 0.1  # For pairwise ranking
  
  # Regularization
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  fp16: true
  gradient_checkpointing: true

output:
  output_dir: "outputs/reward_model_v1"
  run_name: "llama2-7b-reward-v1"
```

### Pairwise Ranking Loss Implementation

```python
# src/reward_trainer.py
import torch
import torch.nn.functional as F
from transformers import Trainer

class RewardTrainer(Trainer):
    def __init__(self, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Pairwise ranking loss for reward model
        
        L = -log(sigmoid(reward_chosen - reward_rejected - margin))
        """
        
        # Prepare chosen and rejected inputs
        chosen_inputs = {
            'input_ids': inputs['chosen_input_ids'],
            'attention_mask': inputs['chosen_attention_mask']
        }
        
        rejected_inputs = {
            'input_ids': inputs['rejected_input_ids'],
            'attention_mask': inputs['rejected_attention_mask']
        }
        
        # Get rewards for both
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)
        
        chosen_rewards = chosen_outputs['rewards']  # [batch_size]
        rejected_rewards = rejected_outputs['rewards']  # [batch_size]
        
        # Pairwise ranking loss
        # We want chosen_rewards > rejected_rewards
        diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(diff - self.margin).mean()
        
        # Calculate accuracy metric
        accuracy = (diff > 0).float().mean()
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'chosen_reward_mean': chosen_rewards.mean(),
            'rejected_reward_mean': rejected_rewards.mean(),
            'reward_margin': diff.mean()
        }
        
        return (loss, metrics) if return_outputs else loss
    
    def log(self, logs):
        """Enhanced logging for reward training"""
        if 'accuracy' in logs:
            print(f"Step {self.state.global_step}: "
                  f"Accuracy: {logs['accuracy']:.3f}, "
                  f"Loss: {logs['loss']:.4f}")
        super().log(logs)

# Training script
def train_reward_model():
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    def preprocess(example):
        # Tokenize chosen
        chosen = tokenizer(
            example['prompt'] + example['chosen'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        
        # Tokenize rejected
        rejected = tokenizer(
            example['prompt'] + example['rejected'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        
        return {
            'chosen_input_ids': chosen['input_ids'],
            'chosen_attention_mask': chosen['attention_mask'],
            'rejected_input_ids': rejected['input_ids'],
            'rejected_attention_mask': rejected['attention_mask']
        }
    
    dataset = load_dataset('json', data_files='data/preference_pairs.jsonl', split='train')
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    
    # Split
    train_test = tokenized.train_test_split(test_size=0.1)
    
    # Initialize model
    model = RewardModel("meta-llama/Llama-2-7b-hf")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="outputs/reward_model_v1",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # Train
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_test['train'],
        eval_dataset=train_test['test']
    )
    
    trainer.train()
    trainer.save_model("outputs/reward_model_v1")
    tokenizer.save_pretrained("outputs/reward_model_v1")
```

### Evaluating Reward Model

```python
# scripts/evaluate_reward_model.py
import torch
from transformers import AutoTokenizer
from src.reward_model import RewardModel

def evaluate_reward_model(model_path, test_examples):
    """Evaluate reward model on held-out test set"""
    
    model = RewardModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    correct = 0
    total = 0
    
    for example in test_examples:
        # Tokenize chosen and rejected
        chosen_text = example['prompt'] + example['chosen']
        rejected_text = example['prompt'] + example['rejected']
        
        chosen_inputs = tokenizer(chosen_text, return_tensors='pt', truncation=True, max_length=512)
        rejected_inputs = tokenizer(rejected_text, return_tensors='pt', truncation=True, max_length=512)
        
        # Get rewards
        with torch.no_grad():
            chosen_reward = model.compute_reward(**chosen_inputs).item()
            rejected_reward = model.compute_reward(**rejected_inputs).item()
        
        # Check if model correctly prefers chosen
        if chosen_reward > rejected_reward:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"Reward Model Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    return accuracy

# Example usage
test_data = load_dataset('json', data_files='data/test_preferences.jsonl', split='train')
accuracy = evaluate_reward_model("outputs/reward_model_v1", list(test_data))
```

---

## Section 3: PPO Training Loop

### Understanding PPO for Language Models

Proximal Policy Optimization (PPO) optimizes the policy to maximize reward while staying close to the reference model:

**Objective:**
```
L_PPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

Where:
- `r_t = π_new(a|s) / π_old(a|s)` (probability ratio)
- `A_t` = Advantage estimate
- `ε` = clipping parameter (typically 0.2)

### PPO Configuration

```yaml
# configs/ppo_training.yaml
policy:
  model: "outputs/sft_model_v1"  # Start from SFT model
  ref_model: "outputs/sft_model_v1"  # Reference model (frozen)

reward:
  model: "outputs/reward_model_v1"
  kl_coeff: 0.1  # KL penalty coefficient

data:
  prompts_file: "data/ppo_prompts.jsonl"
  max_prompt_length: 256
  max_response_length: 256

training:
  # PPO-specific
  ppo_epochs: 4
  mini_batch_size: 4
  ppo_clip_range: 0.2
  vf_clip_range: 0.2
  gamma: 1.0
  lam: 0.95  # GAE lambda
  
  # Generation
  num_rollouts: 128
  temperature: 1.0
  top_k: 0
  top_p: 0.9
  
  # Optimization
  learning_rate: 1.0e-6
  gradient_accumulation_steps: 4
  
  # KL control
  init_kl_coeff: 0.1
  target_kl: 6.0  # Adaptive KL target
  
  fp16: true

output:
  output_dir: "outputs/rlhf_policy_v1"
  run_name: "llama2-7b-rlhf-ppo"
```

### PPO Trainer Implementation

```python
# src/ppo_trainer.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class PPOTrainer:
    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        config
    ):
        self.policy = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # KL coefficient
        self.kl_coeff = config.kl_coeff
    
    def generate_rollouts(self, prompts, num_samples_per_prompt=1):
        """Generate responses from current policy"""
        
        all_responses = []
        all_log_probs = []
        
        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_prompt_length
            ).to(self.policy.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.policy.generate(
                    **inputs,
                    max_new_tokens=self.config.max_response_length,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Extract generated tokens and log probs
            generated_ids = outputs.sequences[0, inputs['input_ids'].shape[1]:]
            log_probs = self._extract_log_probs(outputs.scores, generated_ids)
            
            all_responses.append(generated_ids)
            all_log_probs.append(log_probs)
        
        return all_responses, all_log_probs
    
    def compute_rewards(self, prompts, responses):
        """Compute rewards including KL penalty"""
        
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # Concatenate prompt + response
            full_text = prompt + self.tokenizer.decode(response, skip_special_tokens=True)
            inputs = self.tokenizer(full_text, return_tensors='pt').to(self.policy.device)
            
            # Get reward from reward model
            with torch.no_grad():
                reward = self.reward_model.compute_reward(**inputs).item()
            
            # Compute KL divergence from reference model
            kl_div = self._compute_kl_divergence(prompt, response)
            
            # Final reward with KL penalty
            final_reward = reward - self.kl_coeff * kl_div
            
            rewards.append(final_reward)
        
        return torch.tensor(rewards)
    
    def _compute_kl_divergence(self, prompt, response):
        """Compute KL divergence between policy and reference"""
        
        # Get log probs from both models
        full_text = prompt + self.tokenizer.decode(response, skip_special_tokens=True)
        inputs = self.tokenizer(full_text, return_tensors='pt').to(self.policy.device)
        
        with torch.no_grad():
            policy_logits = self.policy(**inputs).logits
            ref_logits = self.ref_model(**inputs).logits
        
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # KL divergence
        kl = (ref_log_probs - policy_log_probs).exp() * (ref_log_probs - policy_log_probs)
        kl = kl.sum(dim=-1).mean()
        
        return kl.item()
    
    def ppo_update(self, rollouts, advantages):
        """Perform PPO update on collected rollouts"""
        
        total_loss = 0
        
        for epoch in range(self.config.ppo_epochs):
            for minibatch in self._get_minibatches(rollouts, self.config.mini_batch_size):
                
                # Compute probability ratios
                old_log_probs = minibatch['old_log_probs']
                new_log_probs = self._get_new_log_probs(minibatch)
                
                ratio = (new_log_probs - old_log_probs).exp()
                
                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.config.ppo_clip_range, 
                                   1+self.config.ppo_clip_range) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = self._compute_value_loss(minibatch)
                
                # Entropy bonus (encourages exploration)
                entropy = self._compute_entropy(minibatch)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
        
        return total_loss / (self.config.ppo_epochs * len(rollouts))
    
    def train(self, prompts, num_iterations=100):
        """Main PPO training loop"""
        
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # 1. Generate rollouts
            responses, log_probs = self.generate_rollouts(prompts)
            
            # 2. Compute rewards
            rewards = self.compute_rewards(prompts, responses)
            
            # 3. Compute advantages (GAE)
            advantages = self._compute_gae(rewards, log_probs)
            
            # 4. PPO update
            rollouts = {
                'responses': responses,
                'log_probs': log_probs,
                'rewards': rewards
            }
            loss = self.ppo_update(rollouts, advantages)
            
            # 5. Logging
            print(f"Iteration {iteration + 1}:")
            print(f"  Mean Reward: {rewards.mean().item():.3f}")
            print(f"  KL Divergence: {self._compute_mean_kl():.4f}")
            print(f"  PPO Loss: {loss:.4f}")
            
            # 6. Adaptive KL coefficient
            self._update_kl_coeff()
            
            # 7. Save checkpoint
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_{iteration + 1}")
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✓ Checkpoint saved to {path}")
```

### Complete RLHF Pipeline

```python
# scripts/run_rlhf_pipeline.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.reward_model import RewardModel
from src.ppo_trainer import PPOTrainer
import yaml

def run_complete_rlhf():
    # Load configuration
    with open('configs/ppo_training.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("STARTING COMPLETE RLHF PIPELINE")
    print("=" * 80)
    
    # Step 1: Load SFT model (already trained)
    print("\n[1/4] Loading SFT model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        config['policy']['model'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['policy']['ref_model'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['policy']['model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Loaded policy model from {config['policy']['model']}")
    
    # Step 2: Load reward model
    print("\n[2/4] Loading reward model...")
    reward_model = RewardModel.from_pretrained(
        config['reward']['model'],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    reward_model.eval()
    print(f"✓ Loaded reward model from {config['reward']['model']}")
    
    # Step 3: Load prompts for RLHF
    print("\n[3/4] Loading prompts...")
    prompts = load_prompts(config['data']['prompts_file'])
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Step 4: Initialize PPO trainer
    print("\n[4/4] Initializing PPO trainer...")
    trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config['training']
    )
    
    # Step 5: Run PPO training
    print("\n" + "=" * 80)
    print("STARTING PPO TRAINING")
    print("=" * 80)
    
    trainer.train(
        prompts=prompts,
        num_iterations=config['training']['num_iterations']
    )
    
    # Step 6: Save final model
    print("\n" + "=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)
    
    trainer.save_checkpoint(config['output']['output_dir'])
    print(f"\n✓ RLHF complete! Model saved to {config['output']['output_dir']}")

if __name__ == "__main__":
    run_complete_rlhf()
```

---

## Section 4: Monitoring and Debugging RLHF

### Key Metrics to Track

```python
# src/rlhf_monitor.py
class RLHFMonitor:
    def __init__(self):
        self.metrics_history = {
            'rewards': [],
            'kl_divergence': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'generation_length': []
        }
    
    def log_iteration(self, iteration, metrics):
        """Log metrics for each iteration"""
        
        for key in self.metrics_history:
            if key in metrics:
                self.metrics_history[key].append({
                    'iteration': iteration,
                    'value': metrics[key]
                })
        
        # Print summary
        print(f"\nIteration {iteration}:")
        print(f"  Reward: {metrics.get('reward', 0):.3f}")
        print(f"  KL Div: {metrics.get('kl_divergence', 0):.4f}")
        print(f"  Policy Loss: {metrics.get('policy_loss', 0):.4f}")
        print(f"  Entropy: {metrics.get('entropy', 0):.4f}")
    
    def check_for_issues(self, metrics):
        """Detect common RLHF issues"""
        issues = []
        
        # KL divergence too high
        if metrics.get('kl_divergence', 0) > 10.0:
            issues.append("⚠️  High KL divergence - policy drifting too far")
        
        # Reward hacking (high reward but low quality)
        if metrics.get('reward', 0) > 5.0 and metrics.get('kl_divergence', 0) > 5.0:
            issues.append("⚠️  Possible reward hacking detected")
        
        # Collapsing entropy
        if metrics.get('entropy', 0) < 0.1:
            issues.append("⚠️  Low entropy - policy may be collapsing")
        
        # Negative rewards
        if metrics.get('reward', 0) < 0:
            issues.append("⚠️  Negative rewards - check reward model")
        
        return issues

# Usage in training loop
monitor = RLHFMonitor()

for iteration in range(num_iterations):
    # ... training code ...
    
    metrics = {
        'reward': rewards.mean().item(),
        'kl_divergence': kl_div,
        'policy_loss': loss,
        'entropy': entropy.item()
    }
    
    monitor.log_iteration(iteration, metrics)
    
    issues = monitor.check_for_issues(metrics)
    for issue in issues:
        print(issue)
```

### Common RLHF Issues and Solutions

#### Issue 1: KL Divergence Too High

**Symptoms:**
- KL > 10 nats
- Generated text becomes nonsensical
- Reward increases but quality decreases

**Solutions:**
```yaml
# Increase KL penalty
kl_coeff: 0.2  # Increase from 0.1

# Use adaptive KL
target_kl: 6.0
adaptive_kl: true

# Lower learning rate
learning_rate: 5.0e-7  # Reduce from 1e-6
```

#### Issue 2: Reward Hacking

**Symptoms:**
- Rewards increase rapidly
- Generated text exploits reward model quirks
- Human evaluators rate outputs poorly

**Solutions:**
```yaml
# Stronger KL penalty
kl_coeff: 0.5

# Better reward model
# Retrain with more diverse examples
# Add adversarial examples

# Limit generation length
max_response_length: 128  # Prevent verbose exploitation

# Ensemble reward models
# Use multiple reward models and average
```

#### Issue 3: Policy Collapse

**Symptoms:**
- Entropy drops to near zero
- Model generates same response repeatedly
- No diversity in outputs

**Solutions:**
```yaml
# Add entropy bonus
entropy_coefficient: 0.05  # Increase exploration

# Lower KL penalty temporarily
kl_coeff: 0.05

# Restart from earlier checkpoint
# Reduce learning rate
learning_rate: 5.0e-7
```

---

## Section 5: Evaluation After RLHF

### Comprehensive Evaluation Suite

```python
# scripts/evaluate_rlhf.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RLHFEvaluator:
    def __init__(self, model_path, baseline_path=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if baseline_path:
            self.baseline = AutoModelForCausalLM.from_pretrained(
                baseline_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.baseline = None
    
    def evaluate_helpfulness(self, test_prompts):
        """Evaluate helpfulness on standard benchmarks"""
        
        results = []
        
        for prompt in test_prompts:
            response = self.generate(prompt, max_new_tokens=256)
            
            # Rate helpfulness (in practice, use human eval or LLM judge)
            helpfulness_score = self._rate_helpfulness(prompt, response)
            
            results.append({
                'prompt': prompt,
                'response': response,
                'helpfulness': helpfulness_score
            })
        
        avg_helpfulness = sum(r['helpfulness'] for r in results) / len(results)
        return {'avg_helpfulness': avg_helpfulness, 'samples': results}
    
    def evaluate_safety(self, safety_prompts):
        """Evaluate safety on adversarial prompts"""
        
        safe_count = 0
        
        for prompt in safety_prompts:
            response = self.generate(prompt, max_new_tokens=256)
            
            # Check for harmful content
            is_safe = self._check_safety(response)
            
            if is_safe:
                safe_count += 1
        
        safety_rate = safe_count / len(safety_prompts)
        return {'safety_rate': safety_rate}
    
    def compare_with_baseline(self, comparison_prompts):
        """Compare RLHF model with baseline"""
        
        wins = 0
        losses = 0
        ties = 0
        
        for prompt in comparison_prompts:
            rlhf_response = self.generate(prompt)
            baseline_response = self.generate(prompt, model=self.baseline)
            
            # Compare (use human eval or LLM judge)
            result = self._compare_responses(rlhf_response, baseline_response)
            
            if result == 'rlhf_wins':
                wins += 1
            elif result == 'baseline_wins':
                losses += 1
            else:
                ties += 1
        
        return {
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_rate': wins / len(comparison_prompts)
        }
    
    def generate(self, prompt, max_new_tokens=256, model=None):
        """Generate response"""
        if model is None:
            model = self.model
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _rate_helpfulness(self, prompt, response):
        """Rate helpfulness (placeholder - use LLM judge in practice)"""
        # In production, use trained judge or human evaluation
        return 0.8  # Placeholder
    
    def _check_safety(self, response):
        """Check if response is safe"""
        # In production, use safety classifier
        harmful_keywords = ['hate', 'violence', 'illegal']
        return not any(word in response.lower() for word in harmful_keywords)
    
    def _compare_responses(self, response1, response2):
        """Compare two responses"""
        # In production, use pairwise comparison
        return 'rlhf_wins'  # Placeholder

# Usage
evaluator = RLHFEvaluator(
    model_path="outputs/rlhf_policy_v1",
    baseline_path="outputs/sft_model_v1"
)

# Evaluate
helpfulness = evaluator.evaluate_helpfulness(test_prompts)
safety = evaluator.evaluate_safety(safety_prompts)
comparison = evaluator.compare_with_baseline(eval_prompts)

print(f"Helpfulness: {helpfulness['avg_helpfulness']:.3f}")
print(f"Safety: {safety['safety_rate']:.3f}")
print(f"Win Rate vs Baseline: {comparison['win_rate']:.3f}")
```

---

## Summary

RLHF pipeline consists of:

1. **Collect preference data**: High-quality human comparisons
2. **Train reward model**: Learn to predict human preferences
3. **PPO training**: Optimize policy with reward + KL constraint
4. **Evaluate**: Comprehensive testing for helpfulness and safety

### RLHF vs Alternatives

| Method | Complexity | Data Needed | Performance | Best For |
|--------|-----------|-------------|-------------|----------|
| SFT only | Low | Demonstrations | Good | Basic tasks |
| RLHF (PPO) | High | Preferences | Best | Production alignment |
| DPO | Medium | Preferences | Very Good | Fast iteration |
| SimPO | Medium | Preferences | Very Good | Resource-constrained |

---

## Exercises

### Beginner
1. Train reward model on 1000 preference pairs
2. Evaluate reward model accuracy
3. Generate samples and manually inspect quality

### Intermediate
1. Implement complete PPO training loop
2. Tune KL coefficient for stable training
3. Compare RLHF vs SFT-only on evaluation set

### Advanced
1. Implement adaptive KL coefficient
2. Build ensemble of reward models
3. Deploy RLHF model with monitoring
4. Experiment with different PPO hyperparameters

---

## Next Steps

- Tutorial 07: Advanced Reward Modeling Techniques
- Tutorial 08: Continual Learning and Catastrophic Forgetting
- Tutorial 09: Production Deployment and Scaling
- Tutorial 10: Multi-Modal Training and Extensions
