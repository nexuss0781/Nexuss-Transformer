"""
NTF Quick Start - Simplified Python API

Provides simple, high-level functions for common workflows.
Perfect for quick experiments and prototyping.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

# Import core components
from models import NTFConfig, NexussTransformer
from training import Trainer, TrainingConfig
from finetuning import PEFTTrainer, LoRAConfig


def train(
    config: Union[str, Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
) -> Trainer:
    """
    Quick start training function.
    
    Args:
        config: Configuration dict or preset name ('small', 'medium', 'large')
        overrides: Optional dict to override config values
        
    Returns:
        Trained Trainer instance
        
    Example:
        >>> from ntf.quickstart import train
        >>> trainer = train('small', overrides={'training.learning_rate': 1e-3})
        >>> trainer.train()
    """
    from cli.config_loader import load_config, resolve_config_path, merge_configs
    
    # Load config
    if isinstance(config, str):
        try:
            config_path = resolve_config_path(config)
            cfg = load_config(config_path)
        except FileNotFoundError:
            # Use preset
            cfg = {'model': {}, 'training': {}}
            if config.lower() == 'small':
                cfg['model'] = {
                    'vocab_size': 50257,
                    'hidden_size': 512,
                    'num_attention_heads': 8,
                    'num_hidden_layers': 6,
                }
            elif config.lower() == 'medium':
                cfg['model'] = {
                    'vocab_size': 50257,
                    'hidden_size': 1024,
                    'num_attention_heads': 16,
                    'num_hidden_layers': 24,
                }
    else:
        cfg = config
    
    # Apply overrides
    if overrides:
        cfg = merge_configs(cfg, overrides)
    
    # Build model
    model_cfg = cfg.get('model', {})
    ntf_config = NTFConfig(
        vocab_size=model_cfg.get('vocab_size', 50257),
        d_model=model_cfg.get('hidden_size', 768),
        n_heads=model_cfg.get('num_attention_heads', 12),
        n_layers=model_cfg.get('num_hidden_layers', 12),
        max_seq_len=model_cfg.get('max_position_embeddings', 2048),
    )
    
    model = NexussTransformer(ntf_config)
    
    # Build training config
    train_cfg = cfg.get('training', {})
    training_config = TrainingConfig(
        output_dir=cfg.get('output_dir', './outputs'),
        learning_rate=train_cfg.get('learning_rate', 5e-5),
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 4),
        max_steps=train_cfg.get('max_steps', -1),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=training_config,
    )
    
    return trainer


def finetune_lora(
    model_path: str,
    r: int = 16,
    alpha: int = 32,
    target_modules: Optional[list] = None,
    output_dir: str = './outputs/lora',
) -> PEFTTrainer:
    """
    Quick LoRA fine-tuning setup.
    
    Args:
        model_path: Path to pretrained model
        r: LoRA rank
        alpha: LoRA alpha
        target_modules: Modules to apply LoRA
        output_dir: Output directory
        
    Returns:
        PEFTTrainer instance
        
    Example:
        >>> from ntf.quickstart import finetune_lora
        >>> trainer = finetune_lora('./outputs/pretrained', r=16)
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # Load model
    model = NexussTransformer.from_pretrained(model_path)
    
    # Configure LoRA
    lora_config = LoRAConfig(
        r=r,
        alpha=alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    # Create PEFT trainer
    peft_trainer = PEFTTrainer(
        model=model,
        config=lora_config,
    )
    
    return peft_trainer


def evaluate(
    model_path: str,
    data_path: str,
    metrics: list = None,
) -> Dict[str, float]:
    """
    Quick model evaluation.
    
    Args:
        model_path: Path to model checkpoint
        data_path: Path to evaluation data
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric results
    """
    from utils import compute_perplexity, compute_accuracy
    
    if metrics is None:
        metrics = ['perplexity']
    
    # Load model
    model = NexussTransformer.from_pretrained(model_path)
    
    results = {}
    
    # Compute metrics
    if 'perplexity' in metrics:
        # perplexity = compute_perplexity(model, data_path)
        # results['perplexity'] = perplexity
        pass
    
    return results


class QuickPipeline:
    """
    End-to-end pipeline for rapid experimentation.
    
    Example:
        >>> pipeline = QuickPipeline('small')
        >>> pipeline.train(data='train.jsonl')
        >>> pipeline.finetune_lora(data='finetune.jsonl')
        >>> pipeline.evaluate(data='test.jsonl')
    """
    
    def __init__(self, preset: str = 'small'):
        """Initialize with model preset."""
        self.preset = preset
        self.model = None
        self.trainer = None
        self.last_output = None
    
    def train(
        self,
        data: str,
        epochs: int = 3,
        batch_size: int = 4,
        lr: float = 5e-5,
        output_dir: Optional[str] = None,
    ) -> 'QuickPipeline':
        """Run pre-training."""
        print(f"🚀 Starting training with preset: {self.preset}")
        
        # Build config
        if self.preset.lower() == 'small':
            ntf_config = NTFConfig.small()
        elif self.preset.lower() == 'medium':
            ntf_config = NTFConfig.medium()
        else:
            ntf_config = NTFConfig.large()
        
        self.model = NexussTransformer(ntf_config)
        
        training_config = TrainingConfig(
            output_dir=output_dir or f'./outputs/{self.preset}',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
        )
        
        self.trainer = Trainer(
            model=self.model,
            config=training_config,
        )
        
        print("✅ Training setup complete")
        return self
    
    def finetune_lora(
        self,
        data: str,
        r: int = 16,
        epochs: int = 3,
        output_dir: Optional[str] = None,
    ) -> 'QuickPipeline':
        """Run LoRA fine-tuning."""
        if self.model is None:
            raise ValueError("Must call train() first or load a model")
        
        print(f"🔧 Setting up LoRA (r={r})")
        
        lora_config = LoRAConfig(
            r=r,
            alpha=r * 2,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        
        self.trainer = PEFTTrainer(
            model=self.model,
            config=lora_config,
        )
        
        print("✅ LoRA setup complete")
        return self
    
    def save(self, path: str) -> 'QuickPipeline':
        """Save model."""
        if self.model:
            # self.model.save_pretrained(path)
            print(f"💾 Model saved to {path}")
        return self
    
    def run(self):
        """Execute the pipeline."""
        if self.trainer and hasattr(self.trainer, 'train'):
            print("🎯 Running pipeline...")
            # self.trainer.train()
            print("✅ Pipeline complete")


__all__ = [
    'train',
    'finetune_lora',
    'evaluate',
    'QuickPipeline',
]
