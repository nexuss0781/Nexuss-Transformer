"""
PPO (Proximal Policy Optimization) Trainer for RLHF
Uses TRL library for efficient reinforcement learning from human feedback
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import PreTrainedModel
from trl import PPOConfig as TRLPPOConfig, PPOTrainer as TRLPPOTrainer
from peft import LoraConfig


@dataclass
class PPOTrainerConfig:
    """Configuration for PPO Training"""
    
    # Model settings
    model_name: str = "nexuss-transformer-base"
    reward_model_name: str = "nexuss-reward-model"
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["query_proj", "value_proj", "key_proj"])
    
    # PPO hyperparameters
    batch_size: int = 128
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    whiten_rewards: bool = True
    
    # Generation settings
    max_length: int = 512
    min_length: int = 16
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.7
    
    # Value network settings
    use_score_scaling: bool = True
    use_score_norm: bool = True
    score_clip: float = 0.5
    
    # Training settings
    total_episodes: int = 10000
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_steps: int = 500
    output_dir: str = "./ppo_model"
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False


def create_ppo_config(config: PPOTrainerConfig) -> TRLPPOConfig:
    """Convert NTF config to TRL PPOConfig"""
    
    return TRLPPOConfig(
        model_name=config.model_name,
        reward_model_name=config.reward_model_name,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        ppo_epochs=config.ppo_epochs,
        learning_rate=config.learning_rate,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        gamma=config.gamma,
        lam=config.lam,
        whiten_rewards=config.whiten_rewards,
        max_length=config.max_length,
        min_length=config.min_length,
        top_k=config.top_k,
        top_p=config.top_p,
        temperature=config.temperature,
        use_score_scaling=config.use_score_scaling,
        use_score_norm=config.use_score_norm,
        score_clip=config.score_clip,
        total_episodes=config.total_episodes,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        output_dir=config.output_dir,
        bf16=config.bf16,
        fp16=config.fp16,
        log_with="wandb",
    )


def create_ppo_trainer(
    policy_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer,
    config: PPOTrainerConfig,
):
    """Create PPO Trainer instance"""
    
    # Apply LoRA if enabled
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Note: TRL PPOTrainer handles PEFT integration internally
        # We pass the base model and let TRL apply LoRA
    
    # Create TRL PPO config
    ppo_config = create_ppo_config(config)
    
    # Initialize PPO Trainer
    ppo_trainer = TRLPPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    return ppo_trainer


def train_ppo(
    policy_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer,
    dataset,
    config: Optional[PPOTrainerConfig] = None,
):
    """Main function to train with PPO"""
    
    if config is None:
        config = PPOTrainerConfig()
    
    # Create trainer
    ppo_trainer = create_ppo_trainer(policy_model, ref_model, tokenizer, config)
    
    # Training loop
    for epoch in range(config.total_episodes // config.batch_size):
        for batch in dataset:
            # Tokenize query
            query_tensors = [tokenizer.encode(batch["prompt"], return_tensors="pt").squeeze(0) for _ in range(config.batch_size)]
            
            # Generate response
            response_tensors = ppo_trainer.generate(query_tensors)
            response_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Get rewards from reward model
            # In practice, you'd call your reward model here
            rewards = []
            for prompt, response in zip(batch["prompt"] * config.batch_size, response_texts):
                # Placeholder - replace with actual reward model inference
                reward = compute_reward(prompt, response)  # You need to implement this
                rewards.append(torch.tensor(reward))
            
            # Run PPO optimization step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log metrics
            if epoch % config.logging_steps == 0:
                print(f"Epoch {epoch}, KL: {stats['objective/kl']}, Reward: {stats['reward/mean']}")
        
        # Save checkpoint
        if epoch % (config.save_steps // config.batch_size) == 0:
            ppo_trainer.save_pretrained(f"{config.output_dir}/checkpoint-{epoch}")
    
    return ppo_trainer


def compute_reward(prompt: str, response: str) -> float:
    """
    Placeholder reward computation.
    Replace with actual reward model inference in production.
    """
    # This should load your trained reward model and compute scores
    # For now, return a dummy value
    return 0.5
