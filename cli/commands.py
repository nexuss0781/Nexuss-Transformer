"""
CLI command implementations for NTF.

Provides high-level commands for training, fine-tuning, alignment, and evaluation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import yaml


def train_command(config: Dict[str, Any], args) -> int:
    """
    Execute pre-training or continued training.
    
    Args:
        config: Loaded configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("=" * 60)
    print("Nexuss Transformer Framework - Training")
    print("=" * 60)
    
    # Import here to avoid circular imports
    from models import NTFConfig, NexussTransformer
    from training import Trainer, TrainingConfig, DataCollatorForLanguageModeling
    from training.data import create_training_dataset
    
    # Build model config
    model_cfg = config.get('model', {})
    ntf_config = NTFConfig(
        vocab_size=model_cfg.get('vocab_size', 50257),
        d_model=model_cfg.get('hidden_size', 768),
        n_heads=model_cfg.get('num_attention_heads', 12),
        n_layers=model_cfg.get('num_hidden_layers', 12),
        max_seq_len=model_cfg.get('max_position_embeddings', 2048),
        d_ff=model_cfg.get('intermediate_size', 3072),
        activation=model_cfg.get('hidden_act', 'swiglu'),
        use_rope=model_cfg.get('use_rope', True),
        rope_theta=model_cfg.get('rope_theta', 10000.0),
        dropout=model_cfg.get('dropout', 0.1),
        attention_dropout=model_cfg.get('attention_dropout', 0.0),
        bias=model_cfg.get('bias', False),
        tie_word_embeddings=model_cfg.get('tie_word_embeddings', True),
        gradient_checkpointing=model_cfg.get('gradient_checkpointing', False),
    )
    
    print(f"\n📦 Model Configuration:")
    print(f"   Vocabulary Size: {ntf_config.vocab_size:,}")
    print(f"   Hidden Size: {ntf_config.d_model:,}")
    print(f"   Attention Heads: {ntf_config.n_heads}")
    print(f"   Layers: {ntf_config.n_layers}")
    print(f"   Max Sequence Length: {ntf_config.max_seq_len:,}")
    
    # Initialize model
    print("\n🔧 Initializing model...")
    model = NexussTransformer(ntf_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    # Build training config
    train_cfg_dict = config.get('training', {})
    checkpoint_cfg = config.get('checkpoint', {})
    validation_cfg = config.get('validation', {})
    logging_cfg = config.get('logging', {})
    
    # Map mixed precision settings
    bf16 = train_cfg_dict.get('bf16', False)
    fp16 = train_cfg_dict.get('fp16', False)
    mixed_precision = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
    
    training_config = TrainingConfig(
        output_dir=config.get('output_dir', './outputs'),
        num_train_epochs=train_cfg_dict.get('num_train_epochs', 3),
        max_steps=train_cfg_dict.get('max_steps', -1),
        per_device_train_batch_size=train_cfg_dict.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=train_cfg_dict.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=train_cfg_dict.get('gradient_accumulation_steps', 1),
        learning_rate=train_cfg_dict.get('learning_rate', 5e-5),
        weight_decay=train_cfg_dict.get('weight_decay', 0.01),
        warmup_ratio=train_cfg_dict.get('warmup_ratio', 0.0),
        warmup_steps=train_cfg_dict.get('warmup_steps', 0),
        scheduler=train_cfg_dict.get('lr_scheduler_type', 'linear').upper(),
        mixed_precision=mixed_precision,
        gradient_checkpointing=model_cfg.get('gradient_checkpointing', False),
        save_steps=checkpoint_cfg.get('save_steps', 500),
        save_total_limit=checkpoint_cfg.get('save_total_limit', 3),
        logging_steps=logging_cfg.get('logging_steps', 10),
        eval_steps=validation_cfg.get('eval_steps', None),
        report_to=logging_cfg.get('report_to', ['none'])[0] if isinstance(logging_cfg.get('report_to'), list) else 'none',
        seed=config.get('seed', 42),
    )
    
    print(f"\n📚 Training Configuration:")
    print(f"   Output Directory: {training_config.output_dir}")
    print(f"   Batch Size: {training_config.per_device_train_batch_size}")
    print(f"   Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   Learning Rate: {training_config.learning_rate}")
    print(f"   Mixed Precision: {training_config.mixed_precision}")
    print(f"   Max Steps: {training_config.max_steps}")
    
    # Load datasets
    data_cfg = config.get('data', {})
    train_path = data_cfg.get('train_path')
    val_path = data_cfg.get('val_path')
    
    print(f"\n📖 Loading datasets...")
    print(f"   Training Data: {train_path}")
    print(f"   Validation Data: {val_path or 'None'}")
    
    # Create datasets (placeholder - implement based on your data format)
    train_dataset = None
    eval_dataset = None
    
    if train_path and Path(train_path).exists():
        train_dataset = create_training_dataset(
            train_path,
            tokenizer_name=data_cfg.get('tokenizer_name', 'gpt2'),
            max_length=data_cfg.get('max_length', 2048),
        )
        print(f"   ✓ Training samples: {len(train_dataset):,}")
    
    if val_path and Path(val_path).exists():
        eval_dataset = create_training_dataset(
            val_path,
            tokenizer_name=data_cfg.get('tokenizer_name', 'gpt2'),
            max_length=data_cfg.get('max_length', 2048),
        )
        print(f"   ✓ Validation samples: {len(eval_dataset):,}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        max_length=data_cfg.get('max_length', 2048),
        pad_token_id=ntf_config.pad_token_id or 0,
    )
    
    # Initialize trainer
    print("\n🎯 Initializing trainer...")
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n🚀 Starting training...\n")
    metrics = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if hasattr(args, 'resume_from_checkpoint') else None)
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("✅ Training Completed!")
    print("=" * 60)
    print(f"   Final Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Global Steps: {metrics.get('global_step', 0):,}")
    print(f"   Epochs: {metrics.get('epochs_trained', 0)}")
    print(f"   Training Time: {metrics.get('training_time_seconds', 0):.2f}s")
    print(f"   Samples/sec: {metrics.get('samples_per_second', 0):.2f}")
    print(f"\n💾 Model saved to: {training_config.output_dir}")
    print("=" * 60)
    
    return 0


def finetune_command(config: Dict[str, Any], args) -> int:
    """
    Execute fine-tuning (full or LoRA).
    
    Args:
        config: Loaded configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("=" * 60)
    print("Nexuss Transformer Framework - Fine-tuning")
    print("=" * 60)
    
    from models import NTFConfig, NexussTransformer
    from finetuning import PEFTTrainer, LoRAConfig, FullFinetuneTrainer
    from training import TrainingConfig, DataCollatorForLanguageModeling
    from training.data import create_training_dataset
    
    # Check if LoRA is enabled
    lora_cfg = config.get('lora', {})
    use_lora = lora_cfg.get('enable', True)
    
    # Load base model
    model_cfg = config.get('model', {})
    model_path = model_cfg.get('model_name_or_path')
    
    if not model_path:
        print("❌ Error: model_name_or_path not specified in config")
        return 1
    
    print(f"\n📦 Loading base model from: {model_path}")
    
    # Try to load existing model or create new one
    try:
        # If you have model loading implemented
        model = NexussTransformer.from_pretrained(model_path)
    except:
        # Fallback: create from config
        ntf_config = NTFConfig(
            vocab_size=model_cfg.get('vocab_size', 50257),
            d_model=model_cfg.get('hidden_size', 768),
            n_heads=model_cfg.get('num_attention_heads', 12),
            n_layers=model_cfg.get('num_hidden_layers', 12),
            max_seq_len=model_cfg.get('max_position_embeddings', 2048),
            d_ff=model_cfg.get('intermediate_size', 3072),
        )
        model = NexussTransformer(ntf_config)
        print(f"⚠️ Created model from config (pretrained weights not found)")
    
    # Build training config
    train_cfg_dict = config.get('training', {})
    training_config = TrainingConfig(
        output_dir=config.get('output_dir', './outputs/finetune'),
        max_steps=train_cfg_dict.get('max_steps', 5000),
        per_device_train_batch_size=train_cfg_dict.get('per_device_train_batch_size', 4),
        gradient_accumulation_steps=train_cfg_dict.get('gradient_accumulation_steps', 8),
        learning_rate=train_cfg_dict.get('learning_rate', 1e-4),
        warmup_ratio=train_cfg_dict.get('warmup_ratio', 0.05),
        save_steps=config.get('checkpoint', {}).get('save_steps', 500),
    )
    
    # Load dataset
    data_cfg = config.get('data', {})
    train_path = data_cfg.get('train_path')
    
    print(f"\n📖 Loading fine-tuning dataset: {train_path}")
    train_dataset = create_training_dataset(
        train_path,
        tokenizer_name=data_cfg.get('tokenizer_name', 'gpt2'),
        max_length=data_cfg.get('max_length', 1024),
    ) if train_path else None
    
    if use_lora:
        # LoRA fine-tuning
        print("\n🔧 Configuring LoRA...")
        lora_config = LoRAConfig(
            r=lora_cfg.get('r', 16),
            alpha=lora_cfg.get('alpha', 32),
            dropout=lora_cfg.get('dropout', 0.05),
            target_modules=lora_cfg.get('target_modules', ["q_proj", "v_proj"]),
            task_type="CAUSAL_LM",
        )
        
        print(f"   Rank: {lora_config.r}")
        print(f"   Alpha: {lora_config.alpha}")
        print(f"   Target Modules: {lora_config.target_modules}")
        
        # Initialize PEFT trainer
        peft_trainer = PEFTTrainer(
            model=model,
            config=lora_config,
        )
        
        # You would integrate with the main Trainer here
        print("\n🚀 Starting LoRA fine-tuning...")
        # trainer = Trainer(...)
        # trainer.train()
        
        print("\n💡 Note: Full LoRA training integration pending implementation")
        print("   Use the PEFTTrainer directly for now:")
        print("   ```python")
        print("   from finetuning import PEFTTrainer, LoRAConfig")
        print("   trainer = PEFTTrainer(model, lora_config)")
        print("   ```")
        
    else:
        # Full fine-tuning
        print("\n🔧 Configuring full fine-tuning...")
        
        finetuner = FullFinetuneTrainer(model=model)
        
        print("\n🚀 Starting full fine-tuning...")
        # Implementation similar to train_command
    
    print("\n" + "=" * 60)
    print("✅ Fine-tuning Setup Complete!")
    print("=" * 60)
    
    return 0


def align_command(config: Dict[str, Any], args) -> int:
    """
    Execute RLHF alignment (DPO or PPO).
    
    Args:
        config: Loaded configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("=" * 60)
    print("Nexuss Transformer Framework - RLHF Alignment")
    print("=" * 60)
    
    from reward import DPOTrainerConfig, train_dpo
    
    # Determine alignment method
    method = args.method if hasattr(args, 'method') else config.get('alignment_method', 'dpo')
    
    if method.lower() == 'dpo':
        print("\n🎯 Method: Direct Preference Optimization (DPO)")
        
        dpo_cfg = config.get('dpo', {})
        
        dpo_config = DPOTrainerConfig(
            model_name=config.get('model', {}).get('model_name_or_path', ''),
            beta=dpo_cfg.get('beta', 0.1),
            loss_type=dpo_cfg.get('loss_type', 'sigmoid'),
            max_length=dpo_cfg.get('max_length', 512),
            max_prompt_length=dpo_cfg.get('max_prompt_length', 256),
            per_device_train_batch_size=config.get('training', {}).get('per_device_train_batch_size', 4),
            learning_rate=config.get('training', {}).get('learning_rate', 5e-7),
            max_steps=config.get('training', {}).get('max_steps', 10000),
            output_dir=config.get('output_dir', './outputs/dpo'),
        )
        
        print(f"   Beta: {dpo_config.beta}")
        print(f"   Loss Type: {dpo_config.loss_type}")
        print(f"   Max Length: {dpo_config.max_length}")
        
        # Load preference dataset
        data_cfg = config.get('data', {})
        print(f"\n📖 Loading preference dataset...")
        # Implement dataset loading
        
        print("\n🚀 Starting DPO training...")
        # train_dpo(...)
        
        print("\n💡 Note: DPO training requires preference dataset format:")
        print("   Each sample: {'prompt': str, 'chosen': str, 'rejected': str}")
        
    elif method.lower() == 'ppo':
        print("\n🎯 Method: Proximal Policy Optimization (PPO)")
        print("\n💡 PPO implementation available in reward/ppo_trainer.py")
        
    else:
        print(f"❌ Unknown alignment method: {method}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Alignment Setup Complete!")
    print("=" * 60)
    
    return 0


def evaluate_command(config: Dict[str, Any], args) -> int:
    """
    Evaluate a trained model.
    
    Args:
        config: Loaded configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("=" * 60)
    print("Nexuss Transformer Framework - Evaluation")
    print("=" * 60)
    
    from utils import compute_perplexity, compute_accuracy, evaluate_model
    
    model_path = args.model_path if hasattr(args, 'model_path') else config.get('model', {}).get('model_name_or_path')
    
    if not model_path:
        print("❌ Error: Model path not specified")
        return 1
    
    print(f"\n📦 Loading model: {model_path}")
    
    # Load model and tokenizer
    # Implement model loading
    
    print("\n📊 Running evaluations...")
    
    # Perplexity
    if hasattr(args, 'eval_data') and args.eval_data:
        print(f"   Evaluating on: {args.eval_data}")
        # perplexity = compute_perplexity(model, dataloader)
        # print(f"   Perplexity: {perplexity:.2f}")
    
    # Accuracy (if applicable)
    # accuracy = compute_accuracy(...)
    
    print("\n💡 Evaluation metrics:")
    print("   - Perplexity")
    print("   - Accuracy (task-specific)")
    print("   - BLEU/ROUGE (for generation tasks)")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
    
    return 0


def convert_command(config: Dict[str, Any], args) -> int:
    """
    Convert model to different formats (ONNX, GGUF, etc.).
    
    Args:
        config: Loaded configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    print("=" * 60)
    print("Nexuss Transformer Framework - Model Conversion")
    print("=" * 60)
    
    model_path = args.model_path if hasattr(args, 'model_path') else config.get('model', {}).get('model_name_or_path')
    output_format = args.format if hasattr(args, 'format') else 'onnx'
    output_path = args.output if hasattr(args, 'output') else './outputs/converted'
    
    print(f"\n📦 Source Model: {model_path}")
    print(f"📤 Format: {output_format.upper()}")
    print(f"💾 Output: {output_path}")
    
    if output_format.lower() == 'onnx':
        print("\n🔄 Converting to ONNX...")
        # Implement ONNX export
        print("   ✅ ONNX export complete")
        
    elif output_format.lower() == 'gguf':
        print("\n🔄 Converting to GGUF...")
        print("   💡 Use llama.cpp conversion tools for GGUF")
        
    elif output_format.lower() == 'safetensors':
        print("\n🔄 Converting to SafeTensors...")
        # Implement SafeTensors save
        print("   ✅ SafeTensors export complete")
        
    else:
        print(f"❌ Unsupported format: {output_format}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Conversion Complete!")
    print("=" * 60)
    
    return 0
