"""
Reward Model Training for Nexuss Transformer Framework
Trains a reward model to score responses for RLHF
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model


@dataclass
class RewardConfig:
    """Configuration for Reward Model Training"""
    
    # Model settings
    model_name: str = "nexuss-transformer-base"
    num_labels: int = 1
    pooling_strategy: str = "last"  # last, mean, max
    
    # LoRA settings for efficient training
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["query_proj", "value_proj"])
    
    # Training settings
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 10000
    
    # Loss settings
    loss_type: str = "pairwise"  # pairwise, pointwise
    margin: float = 0.0  # For margin-based ranking loss
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./reward_model"
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False


class RewardDataset(Dataset):
    """Dataset for reward model training with pairwise comparisons"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Expected format: {prompt: str, chosen: str, rejected: str}
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Tokenize chosen and rejected responses
        chosen_input = self.tokenizer(
            prompt + chosen,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        rejected_input = self.tokenizer(
            prompt + rejected,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        return {
            "chosen_input_ids": torch.tensor(chosen_input["input_ids"]),
            "chosen_attention_mask": torch.tensor(chosen_input["attention_mask"]),
            "rejected_input_ids": torch.tensor(rejected_input["input_ids"]),
            "rejected_attention_mask": torch.tensor(rejected_input["attention_mask"]),
        }


def collate_fn(batch):
    """Custom collate function for reward dataset"""
    chosen_input_ids = [item["chosen_input_ids"] for item in batch]
    chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
    rejected_input_ids = [item["rejected_input_ids"] for item in batch]
    rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
    
    # Pad sequences
    max_len_chosen = max(len(x) for x in chosen_input_ids)
    max_len_rejected = max(len(x) for x in rejected_input_ids)
    max_len = max(max_len_chosen, max_len_rejected)
    
    def pad_sequence(seq, max_len):
        pad_len = max_len - len(seq)
        return torch.cat([seq, torch.zeros(pad_len, dtype=seq.dtype)])
    
    return {
        "chosen_input_ids": torch.stack([pad_sequence(x, max_len) for x in chosen_input_ids]),
        "chosen_attention_mask": torch.stack([pad_sequence(x, max_len) for x in chosen_attention_mask]),
        "rejected_input_ids": torch.stack([pad_sequence(x, max_len) for x in rejected_input_ids]),
        "rejected_attention_mask": torch.stack([pad_sequence(x, max_len) for x in rejected_attention_mask]),
    }


class RewardTrainer(Trainer):
    """Custom Trainer for Reward Model with pairwise loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_input_ids = inputs.pop("chosen_input_ids")
        chosen_attention_mask = inputs.pop("chosen_attention_mask")
        rejected_input_ids = inputs.pop("rejected_input_ids")
        rejected_attention_mask = inputs.pop("rejected_attention_mask")
        
        # Forward pass for chosen and rejected
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            return_dict=True,
        )
        
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            return_dict=True,
        )
        
        chosen_rewards = chosen_outputs.logits.squeeze(-1)
        rejected_rewards = rejected_outputs.logits.squeeze(-1)
        
        # Pairwise ranking loss
        if self.args.loss_type == "pairwise":
            loss = -nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        else:  # pointwise with margin
            loss = torch.relu(margin - (chosen_rewards - rejected_rewards)).mean()
        
        # Compute accuracy metric
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            self.log({"reward_accuracy": accuracy.item()})
        
        return (loss, chosen_outputs) if return_outputs else loss


def create_reward_model(config: RewardConfig) -> PreTrainedModel:
    """Create and configure reward model"""
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32,
    )
    
    # Apply LoRA if enabled
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def train_reward_model(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[RewardConfig] = None,
    tokenizer=None,
):
    """Main function to train reward model"""
    
    if config is None:
        config = RewardConfig()
    
    # Create model
    model = create_reward_model(config)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        remove_unused_columns=False,
        report_to="wandb",
    )
    
    # Add custom attributes for loss computation
    training_args.loss_type = config.loss_type
    
    # Initialize trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn if tokenizer else None,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(config.output_dir)
    
    return trainer, model
