"""
DPO (Direct Preference Optimization) Trainer for RLHF
Simpler alternative to PPO that directly optimizes from preferences
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel, TrainingArguments
from trl import DPOConfig as TRLDPOConfig, DPOTrainer as TRLDPOTrainer
from peft import LoraConfig, get_peft_model


@dataclass
class DPOTrainerConfig:
    """Configuration for DPO Training"""
    
    # Model settings
    model_name: str = "nexuss-transformer-base"
    ref_model_name: Optional[str] = None  # If None, uses same as model_name
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["query_proj", "value_proj", "key_proj"])
    
    # DPO hyperparameters
    beta: float = 0.1  # Temperature parameter for DPO
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, kto_pair
    label_smoothing: float = 0.0
    truncation_mode: str = "keep_end"  # keep_end, keep_start
    
    # Dataset settings
    max_length: int = 512
    max_prompt_length: int = 256
    max_target_length: int = 256
    
    # Training settings
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 10000
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    output_dir: str = "./dpo_model"
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # Generation settings for evaluation
    generation_max_length: int = 256
    generation_num_beams: int = 1


def create_dpo_config(config: DPOTrainerConfig) -> TRLDPOConfig:
    """Convert NTF config to TRL DPOConfig"""
    
    return TRLDPOConfig(
        beta=config.beta,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
        truncation_mode=config.truncation_mode,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        max_target_length=config.max_target_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to="wandb",
        remove_unused_columns=False,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.generation_num_beams,
    )


def prepare_model_for_dpo(
    model: PreTrainedModel,
    config: DPOTrainerConfig,
) -> PreTrainedModel:
    """Prepare model with LoRA for DPO training"""
    
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def train_dpo(
    model: PreTrainedModel,
    ref_model: Optional[PreTrainedModel],
    tokenizer,
    train_dataset,
    eval_dataset=None,
    config: Optional[DPOTrainerConfig] = None,
):
    """Main function to train with DPO"""
    
    if config is None:
        config = DPOTrainerConfig()
    
    # Prepare model
    model = prepare_model_for_dpo(model, config)
    
    # Load reference model if not provided
    if ref_model is None:
        if config.ref_model_name:
            from transformers import AutoModelForCausalLM
            ref_model = AutoModelForCausalLM.from_pretrained(
                config.ref_model_name,
                torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            )
        else:
            ref_model = None  # TRL will create a copy automatically
    
    # Create DPO config
    dpo_config = create_dpo_config(config)
    
    # Initialize DPO Trainer
    dpo_trainer = TRLDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=None,  # TRL provides default collator for DPO
    )
    
    # Train
    dpo_trainer.train()
    
    # Save final model
    dpo_trainer.save_model(config.output_dir)
    
    return dpo_trainer, model


# Example dataset format for DPO
# Each sample should be a dict with:
# {
#     "prompt": str,           # The input prompt
#     "chosen": str,           # The preferred response
#     "rejected": str          # The less preferred response
# }

def create_dpo_dataset(data: List[Dict[str, str]], tokenizer, config: DPOTrainerConfig):
    """
    Create dataset for DPO training.
    TRL's DPOTrainer handles the dataset internally, but this shows expected format.
    """
    # Validate data format
    for item in data:
        assert "prompt" in item, "Missing 'prompt' key"
        assert "chosen" in item, "Missing 'chosen' key"
        assert "rejected" in item, "Missing 'rejected' key"
    
    # TRL's DPOTrainer expects a datasets.Dataset object
    # You can convert your list of dicts using:
    # from datasets import Dataset
    # dataset = Dataset.from_list(data)
    
    return data  # Return as-is, TRL will handle conversion
