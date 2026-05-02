"""
Nexuss Transformer Framework - Reward Modeling & RLHF
Implements PPO, DPO, and Reward Modeling using TRL
"""

from .reward_model import RewardTrainer, RewardConfig
from .ppo_trainer import PPOTrainerConfig, PPOTrainer
from .dpo_trainer import DPOTrainerConfig, DPOTrainer

__all__ = [
    "RewardTrainer",
    "RewardConfig",
    "PPOTrainerConfig",
    "PPOTrainer",
    "DPOTrainerConfig",
    "DPOTrainer",
]
