#!/usr/bin/env python3
"""
Blank Slate Training Script for Synaxarium and Canon Biblical Datasets

This script trains a decoder-only transformer from scratch on Ethiopian Orthodox
religious texts (Synaxarium) and Amharic-English parallel Bible corpus.

The trained model will have these datasets as frozen base knowledge.

Features:
- NTFTokenizer (EthioBBPE-based) for optimal Amharic/Ge'ez tokenization
- Advanced model architectures (RoPE, SwiGLU, RMSNorm)
- Mixed precision training (BF16/FP16)
- Gradient checkpointing for memory efficiency
- Comprehensive logging and metrics tracking
- Checkpoint management with resume capability
- Support for distributed training via Accelerate
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
import random
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.config import NTFConfig, QuantizationConfig
from models.transformer import NexussTransformer
from training.config import TrainingConfig, Precision, OptimizerType, SchedulerType
from training.trainer import Trainer
from training.data import DataCollatorForLanguageModeling, create_training_dataset
from tokenizer.ntf_tokenizer import NTFTokenizer


# ============================================================================
# DATASET LOADING AND PREPROCESSING
# ============================================================================

def load_synaxarium_dataset(parquet_path: str) -> List[Dict[str, str]]:
    """
    Load Synaxarium dataset and format as structured text.
    
    Args:
        parquet_path: Path to synaxarium_dataset.parquet
    
    Returns:
        List of dictionaries with formatted text
    """
    df = pd.read_parquet(parquet_path)
    samples = []
    
    for _, row in df.iterrows():
        month = row['ወር']
        day = row['ቀን']
        content = row['መጽሃፍ']
        
        # Structured format for better learning
        text = f"""<doc>
<type>ስንክሳር</type>
<ወር>{month}</ወር>
<ቀን>{day}</ቀን>
<content>{content}</content>
</doc>"""
        samples.append({"text": text})
    
    print(f"✓ Loaded {len(samples)} Synaxarium entries")
    return samples


def load_canon_biblical_dataset(
    parquet_path: str,
    include_english: bool = False,
    amharic_only: bool = True
) -> List[Dict[str, str]]:
    """
    Load Canon Biblical dataset and format as structured text.
    
    Args:
        parquet_path: Path to train-00000-of-00001.parquet
        include_english: Whether to include English verses
        amharic_only: If True, only use Amharic text
    
    Returns:
        List of dictionaries with formatted text
    """
    df = pd.read_parquet(parquet_path)
    samples = []
    
    for _, row in df.iterrows():
        book = row['መጽሐፍ']
        chapter = row['ምዕራፍ']
        verse_num = row['ቁጥር']
        amharic = row['ጥቅስ']
        english = row.get('verse', '')
        
        if amharic_only or not include_english or not english:
            # Amharic only format
            text = f"""<doc>
<type>መጽሐፍ ቅዱስ</type>
<መጽሐፍ>{book}</መጽሐፍ>
<ምዕራፍ>{chapter}</ምዕራፍ>
<ቁጥር>{verse_num}</ቁጥር>
<content>{amharic}</content>
</doc>"""
        else:
            # Bilingual format
            text = f"""<doc>
<type>መጽሐፍ ቅዱስ / Holy Bible</type>
<መጽሐፍ>{book}</መጽሐፍ>
<ምዕራፍ>{chapter}</ምዕራፍ>
<ቁጥር>{verse_num}</ቁጥር>
<አማርኛ>{amharic}</አማርኛ>
<English>{english}</English>
</doc>"""
        
        samples.append({"text": text})
    
    print(f"✓ Loaded {len(samples)} Biblical verses")
    return samples


def create_combined_dataset(
    synaxarium_samples: List[Dict[str, str]],
    biblical_samples: List[Dict[str, str]],
    shuffle: bool = True,
    seed: int = 42
) -> Dataset:
    """
    Combine datasets into a single HuggingFace Dataset.
    
    Args:
        synaxarium_samples: Samples from Synaxarium
        biblical_samples: Samples from Bible
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed for reproducibility
    
    Returns:
        HuggingFace Dataset
    """
    # Create individual datasets
    synaxarium_ds = Dataset.from_list(synaxarium_samples)
    biblical_ds = Dataset.from_list(biblical_samples)
    
    # Combine datasets
    combined = concatenate_datasets([synaxarium_ds, biblical_ds])
    
    # Shuffle if requested
    if shuffle:
        combined = combined.shuffle(seed=seed)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Synaxarium entries: {len(synaxarium_samples):,}")
    print(f"Biblical verses:    {len(biblical_samples):,}")
    print(f"Total samples:      {len(combined):,}")
    
    # Calculate character statistics
    total_chars = sum(len(s['text']) for s in synaxarium_samples + biblical_samples)
    avg_chars = total_chars / len(combined) if len(combined) > 0 else 0
    print(f"Total characters:   {total_chars:,}")
    print(f"Average length:     {avg_chars:.1f} chars")
    print("=" * 60)
    
    return combined


# ============================================================================
# TOKENIZER SETUP
# ============================================================================

def get_ntf_tokenizer(
    pretrained_path: Optional[str] = None,
    vocab_file: Optional[str] = None,
    merges_file: Optional[str] = None,
    special_tokens: Optional[List[str]] = None
) -> NTFTokenizer:
    """
    Initialize or load NTFTokenizer (EthioBBPE-based).
    
    Args:
        pretrained_path: Path to pretrained tokenizer or HuggingFace model ID
        vocab_file: Path to vocabulary file
        merges_file: Path to merge rules file
        special_tokens: Additional special tokens
    
    Returns:
        NTFTokenizer instance
    """
    try:
        if pretrained_path:
            # Load from pretrained path or HuggingFace Hub
            print(f"Loading NTFTokenizer from: {pretrained_path}")
            tokenizer = NTFTokenizer.from_pretrained(pretrained_path)
        elif vocab_file and merges_file:
            # Load from vocab and merges files
            print(f"Loading NTFTokenizer from files:")
            print(f"  Vocab:  {vocab_file}")
            print(f"  Merges: {merges_file}")
            tokenizer = NTFTokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                special_tokens=special_tokens
            )
        else:
            # Try default EthioBBPE from HuggingFace
            print("Loading default EthioBBPE tokenizer from HuggingFace Hub...")
            tokenizer = NTFTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")
        
        print(f"✓ NTFTokenizer loaded successfully")
        print(f"  Vocabulary size: {tokenizer.get_vocab_size():,}")
        print(f"  Special tokens: {tokenizer.special_tokens}")
        
        return tokenizer
        
    except Exception as e:
        print(f"⚠ Could not load NTFTokenizer: {e}")
        print("Falling back to creating a new tokenizer...")
        return None


def create_simple_tokenizer_from_dataset(
    texts: List[str],
    max_vocab_size: int = 16000
) -> NTFTokenizer:
    """
    Create a simple character/subword tokenizer from dataset.
    Used as fallback when EthioBBPE is not available.
    
    Args:
        texts: List of training texts
        max_vocab_size: Maximum vocabulary size
    
    Returns:
        Simple tokenizer instance
    """
    print("Creating simple character-level tokenizer from dataset...")
    
    # Collect all unique characters
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    
    print(f"Found {len(all_chars):,} unique characters")
    
    # For now, return None and use the built-in simple tokenizer in train script
    # A full implementation would train a BPE tokenizer here
    return None


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_blank_slate_model(
    vocab_size: int,
    model_size: str = "small",
    max_seq_len: int = 512,
    use_rope: bool = True,
    use_swiglu: bool = True,
    use_rmsnorm: bool = True,
    dropout: float = 0.1,
    tie_embeddings: bool = True,
) -> Tuple[NexussTransformer, NTFConfig]:
    """
    Create a blank slate transformer model from scratch.
    
    Args:
        vocab_size: Size of vocabulary
        model_size: Model size preset ('small', 'medium', 'large')
        max_seq_len: Maximum sequence length
        use_rope: Use Rotary Positional Embeddings
        use_swiglu: Use SwiGLU activation
        use_rmsnorm: Use RMSNorm instead of LayerNorm
        dropout: Dropout probability
        tie_embeddings: Tie input/output embeddings
    
    Returns:
        Tuple of (model, config)
    """
    print(f"\nCreating blank slate model (size: {model_size})...")
    
    # Get base configuration
    if model_size == "small":
        config = NTFConfig.small()
    elif model_size == "medium":
        config = NTFConfig.medium()
    elif model_size == "large":
        config = NTFConfig.large()
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Customize configuration
    config.vocab_size = vocab_size
    config.max_seq_len = max_seq_len
    config.use_rope = use_rope
    config.rope_theta = 10000.0  # Standard RoPE theta
    config.activation = "swiglu" if use_swiglu else "gelu"
    config.dropout = dropout
    config.attention_dropout = dropout * 0.5
    config.hidden_dropout = dropout * 0.5
    config.tie_word_embeddings = tie_embeddings
    config.use_rmsnorm = use_rmsnorm
    config.bias = False  # Better for transformer models
    
    # Set special token IDs (will be updated after tokenizer creation)
    config.bos_token_id = 2
    config.eos_token_id = 3
    config.pad_token_id = 0
    
    # Create model
    model = NexussTransformer(config)
    
    # Print model info
    param_info = model.count_parameters()
    total_params = param_info['total'] / 1e6
    
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(f"Model size:       {model_size.upper()} ({total_params:.2f}M parameters)")
    print(f"Vocabulary:       {vocab_size:,}")
    print(f"Embedding dim:    {config.d_model}")
    print(f"Layers:           {config.n_layers}")
    print(f"Heads:            {config.n_heads}")
    print(f"Head dim:         {config.head_dim}")
    print(f"FFN dim:          {config.d_ff}")
    print(f"Max sequence:     {config.max_seq_len}")
    print(f"RoPE:             {'✓' if use_rope else '✗'}")
    print(f"SwiGLU:           {'✓' if use_swiglu else '✗'}")
    print(f"RMSNorm:          {'✓' if use_rmsnorm else '✗'}")
    print(f"Tie embeddings:   {'✓' if tie_embeddings else '✗'}")
    print(f"Dropout:          {dropout}")
    print("=" * 60)
    
    return model, config


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

def create_training_config(
    output_dir: str,
    run_name: str,
    num_epochs: int = 10,
    batch_size: int = 8,
    gradient_accumulation: int = 2,
    learning_rate: float = 1e-3,
    warmup_ratio: float = 0.05,
    warmup_steps: Optional[int] = None,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    save_steps: int = 100,
    logging_steps: int = 10,
    mixed_precision: str = "fp32",
    gradient_checkpointing: bool = False,
    seed: int = 42,
) -> TrainingConfig:
    """
    Create training configuration.
    
    Args:
        output_dir: Directory to save checkpoints
        run_name: Name of the training run
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation: Gradient accumulation steps
        learning_rate: Peak learning rate
        warmup_ratio: Warmup ratio
        warmup_steps: Number of warmup steps (overrides warmup_ratio if specified)
        weight_decay: Weight decay
        max_grad_norm: Maximum gradient norm
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        mixed_precision: Precision mode ('fp32', 'fp16', 'bf16')
        gradient_checkpointing: Enable gradient checkpointing
        seed: Random seed
    
    Returns:
        TrainingConfig instance
    """
    # Map precision string to enum
    precision_map = {
        "fp32": Precision.FP32,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
    }
    precision = precision_map.get(mixed_precision.lower(), Precision.FP32)
    
    config = TrainingConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=OptimizerType.ADAMW,
        scheduler=SchedulerType.LINEAR,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        mixed_precision=precision,
        gradient_checkpointing=gradient_checkpointing,
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        eval_steps=save_steps,
        max_grad_norm=max_grad_norm,
        seed=seed,
        report_to="none",  # Can be changed to 'wandb' or 'tensorboard'
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        dataloader_pin_memory=True,
    )
    
    effective_batch = config.effective_batch_size
    
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Output directory:    {output_dir}")
    print(f"Run name:           {run_name}")
    print(f"Epochs:             {num_epochs}")
    print(f"Batch size:         {batch_size} (per device)")
    print(f"Gradient accum:     {gradient_accumulation}")
    print(f"Effective batch:    {effective_batch}")
    print(f"Learning rate:      {learning_rate:.2e}")
    print(f"Warmup ratio:       {warmup_ratio}")
    print(f"Weight decay:       {weight_decay}")
    print(f"Mixed precision:    {mixed_precision.upper()}")
    print(f"Gradient ckpt:      {'✓' if gradient_checkpointing else '✗'}")
    print(f"Save every:         {save_steps} steps")
    print(f"Log every:          {logging_steps} steps")
    print("=" * 60)
    
    return config


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    """Main training function."""
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "BLANK SLATE TRAINING - NTF FRAMEWORK" + " " * 11 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\nTraining Ethiopian Orthodox Religious Texts")
    print("(Synaxarium + Canon Biblical Corpus)\n")
    
    # =====================
    # Paths and Configuration
    # =====================
    
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "blank_slate_ethiopian"
    
    synaxarium_path = data_dir / "synaxarium_dataset.parquet"
    biblical_path = data_dir / "train-00000-of-00001.parquet"
    
    # Verify data files exist
    assert synaxarium_path.exists(), f"Synaxarium dataset not found: {synaxarium_path}"
    assert biblical_path.exists(), f"Biblical dataset not found: {biblical_path}"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =====================
    # Load Datasets
    # =====================
    
    print("\n[1/7] Loading datasets...")
    synaxarium_samples = load_synaxarium_dataset(str(synaxarium_path))
    biblical_samples = load_canon_biblical_dataset(
        str(biblical_path),
        include_english=args.include_english,
        amharic_only=not args.include_english
    )
    
    # =====================
    # Create Combined Dataset
    # =====================
    
    print("\n[2/7] Creating combined dataset...")
    dataset = create_combined_dataset(
        synaxarium_samples,
        biblical_samples,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    # Sample some texts for tokenizer creation if needed
    sample_texts = [s['text'] for s in synaxarium_samples[:1000] + biblical_samples[:1000]]
    
    # =====================
    # Initialize Tokenizer
    # =====================
    
    print("\n[3/7] Initializing tokenizer...")
    
    tokenizer = None
    
    # Try to load NTFTokenizer (EthioBBPE)
    if args.tokenizer_path:
        tokenizer = get_ntf_tokenizer(pretrained_path=args.tokenizer_path)
    elif args.vocab_file and args.merges_file:
        tokenizer = get_ntf_tokenizer(
            vocab_file=args.vocab_file,
            merges_file=args.merges_file
        )
    else:
        # Try default EthioBBPE from HuggingFace
        try:
            tokenizer = get_ntf_tokenizer()
        except Exception as e:
            print(f"Could not load EthioBBPE: {e}")
    
    if tokenizer is None:
        print("\n⚠ Using fallback: Simple character-level tokenization via DataCollator")
        print("For optimal results, install ethiobbpe or provide tokenizer files.")
        
        # Create a minimal vocabulary based on dataset characters
        all_chars = set()
        for text in sample_texts:
            all_chars.update(text)
        
        vocab_size = len(all_chars) + 100  # Add buffer for special tokens
        vocab_size = min(vocab_size, args.max_vocab_size)
        print(f"Estimated vocabulary size: {vocab_size}")
    else:
        vocab_size = tokenizer.get_vocab_size()
    
    # Handle max_seq_length alias
    if args.max_seq_length is not None:
        args.max_seq_len = args.max_seq_length
    
    # Handle warmup_steps vs warmup_ratio
    if args.warmup_steps is not None:
        # warmup_steps takes precedence, will be converted to ratio later
        pass
    
    # =====================
    # Create Model
    # =====================
    
    print("\n[4/7] Creating blank slate model...")
    
    # Check GPU and Flash Attention availability before model creation
    has_gpu = torch.cuda.is_available()
    flash_attn_available = False
    
    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_capability = torch.cuda.get_device_capability(0)
        print(f"\n🚀 GPU Detected: {gpu_name}")
        print(f"   Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
        
        # Try to import Flash Attention
        try:
            from flash_attn import flash_attn_func
            flash_attn_available = True
            print(f"   ✓ Flash Attention 2: Available")
        except ImportError:
            print(f"   ⚠ Flash Attention 2: Not installed")
            print(f"   💡 Install with: pip install flash-attn --no-build-isolation")
        
        if gpu_capability[0] >= 8 and flash_attn_available:
            print(f"   → Will auto-enable Flash Attention in model layers")
        elif gpu_capability[0] < 8:
            print(f"   → GPU architecture too old for Flash Attention (requires Ampere+)")
    else:
        print("\n⚠️  No GPU detected - training will run on CPU (slow)")
        print("   Enable GPU in your environment for faster training")
    
    model, model_config = create_blank_slate_model(
        vocab_size=vocab_size,
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        use_rope=not args.no_rope,
        use_swiglu=not args.no_swiglu,
        use_rmsnorm=not args.no_rmsnorm,
        dropout=args.dropout,
        tie_embeddings=not args.no_tie_embeddings,
    )
    
    # Update config with tokenizer special tokens if available
    if tokenizer:
        model_config.bos_token_id = tokenizer.vocab.get(tokenizer.bos_token, 2)
        model_config.eos_token_id = tokenizer.vocab.get(tokenizer.eos_token, 3)
        model_config.pad_token_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
    
    # =====================
    # Prepare Training Data
    # =====================
    
    print("\n[5/7] Preparing training data...")
    
    # Tokenize the dataset if we have a tokenizer
    if tokenizer:
        print("Tokenizing dataset with NTFTokenizer...")
        
        def tokenize_function(examples):
            # Use NTFTokenizer's encode_batch method
            encoded_outputs = tokenizer.encode_batch(examples["text"], add_special_tokens=True)
            # Extract ids from TokenizerOutput objects
            input_ids = [out.ids for out in encoded_outputs]
            return {"input_ids": input_ids}
        
        # Tokenize dataset in batches with tqdm progress bar
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],  # Remove raw text after tokenization
            desc="Tokenizing dataset"
            # Note: 'disable' argument removed as it's not supported in datasets.map()
        )
        print(f"✓ Dataset tokenized: {len(tokenized_dataset)} samples")
    else:
        # Fallback: simple character-level tokenization on-the-fly via custom dataset
        print("Using on-the-fly character tokenization (fallback mode)...")
        tokenized_dataset = dataset  # Keep raw text, will tokenize in collator
        
        # Create a simple char-to-id mapping for fallback
        all_chars = set()
        for sample in synaxarium_samples[:500] + biblical_samples[:500]:
            all_chars.update(sample['text'])
        
        char_to_id = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        for i, char in enumerate(sorted(all_chars)):
            if char not in char_to_id:
                char_to_id[char] = len(char_to_id)
        
        id_to_char = {v: k for k, v in char_to_id.items()}
        
        def char_tokenize_collator(examples):
            """Custom collator that does character-level tokenization."""
            texts = [ex['text'] for ex in examples]
            input_ids_list = []
            for text in texts:
                ids = [char_to_id.get(c, 1) for c in text[:args.max_seq_len]]
                if not ids:
                    ids = [0]
                input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            
            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
            labels = input_ids.clone()
            labels[input_ids == 0] = -100  # Ignore padding in loss
            
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": (input_ids != 0).long(),
            }
        
        data_collator = char_tokenize_collator
        pad_token_id = 0
    
    # Create standard data collator if not using fallback
    if tokenizer:
        pad_token_id = tokenizer.vocab.get(tokenizer.pad_token, 0)
        
        data_collator = DataCollatorForLanguageModeling(
            pad_token_id=pad_token_id,
            max_length=args.max_seq_len,
        )
    
    print(f"✓ Data preparation complete")
    print(f"  Pad token ID: {pad_token_id}")
    print(f"  Max length: {args.max_seq_len}")
    print(f"  Dataset size: {len(tokenized_dataset):,} samples")
    
    # =====================
    # Configure Training
    # =====================
    
    print("\n[6/7] Configuring training...")
    
    training_config = create_training_config(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )
    
    # =====================
    # Initialize Trainer
    # =====================
    
    print("\n[7/7] Initializing trainer...")
    
    # Use tokenized_dataset if we tokenized, otherwise use original dataset with custom collator
    train_data = tokenized_dataset if tokenizer else dataset
    
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataset=train_data,
        eval_dataset=None,  # Can add validation split if needed
        data_collator=data_collator,
    )
    
    # =====================
    # Start Training
    # =====================
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 20 + "STARTING TRAINING" + " " * 21 + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nModel will be trained from random initialization (blank slate).")
    print(f"Base knowledge: Ethiopian Orthodox religious texts")
    print(f"Output directory: {output_dir}")
    print(f"\nPress Ctrl+C to interrupt training.\n")
    
    try:
        # Train the model
        metrics = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Calculate detailed model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Calculate memory footprint (in MB, assuming float32)
        param_memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Calculate theoretical FLOPs for one forward pass (approximate)
        # For transformer: ~2 * params * seq_len for attention + FFN
        avg_seq_len = args.max_seq_len
        flops_per_token = 2 * total_params  # Simplified estimate
        total_flops = flops_per_token * len(tokenized_dataset) * args.num_epochs
        
        # Print comprehensive final results
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 18 + "TRAINING COMPLETED!" + " " * 21 + "║")
        print("╚" + "═" * 58 + "╝")
        
        print(f"\n📊 FINAL TRAINING METRICS:")
        print("─" * 60)
        print(f"  • Training loss:     {metrics['train_loss']:.4f}")
        print(f"  • Global steps:      {metrics['global_step']:,}")
        print(f"  • Epochs trained:    {metrics['epochs_trained']}")
        print(f"  • Training time:     {metrics['training_time_seconds']:.1f}s ({metrics['training_time_seconds']/60:.1f} min)")
        print(f"  • Samples/sec:       {metrics['samples_per_second']:.2f}")
        
        print(f"\n🧠 MODEL ARCHITECTURE STATISTICS:")
        print("─" * 60)
        print(f"  • Total parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  • Trainable parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        if frozen_params > 0:
            print(f"  • Frozen parameters:     {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        print(f"  • Parameter memory:      {param_memory_mb:.2f} MB (FP32 equivalent)")
        print(f"  • Vocabulary size:       {vocab_size:,}")
        print(f"  • Max sequence length:   {args.max_seq_len}")
        print(f"  • Model depth:           {model_config.num_layers} layers")
        print(f"  • Hidden dimension:      {model_config.dim_model}")
        print(f"  • Attention heads:       {model_config.num_heads}")
        print(f"  • FFN dimension:         {model_config.dim_ffn}")
        
        # Architecture features
        features = []
        if model_config.use_rope: features.append("RoPE")
        if model_config.use_swiglu: features.append("SwiGLU")
        if model_config.use_rmsnorm: features.append("RMSNorm")
        if model_config.tie_embeddings: features.append("Tied Embeddings")
        if args.gradient_checkpointing: features.append("Gradient Checkpointing")
        print(f"  • Architecture features: {', '.join(features)}")
        
        print(f"\n💻 COMPUTATIONAL STATISTICS:")
        print("─" * 60)
        print(f"  • Dataset size:          {len(tokenized_dataset):,} samples")
        print(f"  • Total tokens processed:{len(tokenized_dataset) * avg_seq_len:,}")
        print(f"  • Estimated FLOPs:       {total_flops/1e12:.2f} TFLOPs")
        print(f"  • Batch size:            {args.batch_size}")
        print(f"  • Gradient accumulation: {args.gradient_accumulation}")
        print(f"  • Effective batch size:  {args.batch_size * args.gradient_accumulation}")
        
        print(f"\n🎯 TOKENIZER INFORMATION:")
        print("─" * 60)
        if tokenizer:
            print(f"  • Tokenizer type:        NTFTokenizer (EthioBBPE)")
            print(f"  • Vocab file:            {args.tokenizer_path or 'Default EthioBBPE'}")
        else:
            print(f"  • Tokenizer type:        Character-level (fallback)")
        print(f"  • Special tokens:        <pad>, <unk>, <s>, </s>, <mask>")
        
        print(f"\n💾 OUTPUT FILES:")
        print("─" * 60)
        print(f"  • Model directory:       {output_dir}")
        print(f"  • Final checkpoint:      {output_dir}/checkpoint-final/")
        
        # Save training config for reference
        config_path = output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(training_config.to_dict(), f, indent=2)
        print(f"  • Training config:       {config_path}")
        
        # Save model config
        model_config_path = output_dir / "model_config.json"
        with open(model_config_path, 'w') as f:
            json.dump(model_config.to_dict(), f, indent=2)
        print(f"  • Model config:          {model_config_path}")
        
        # Save comprehensive metrics report
        metrics_report = {
            "training_metrics": metrics,
            "model_statistics": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": frozen_params,
                "param_memory_mb": param_memory_mb,
                "vocab_size": vocab_size,
                "max_seq_length": args.max_seq_len,
                "num_layers": model_config.num_layers,
                "dim_model": model_config.dim_model,
                "num_heads": model_config.num_heads,
                "dim_ffn": model_config.dim_ffn,
                "architecture_features": features,
            },
            "computational_statistics": {
                "dataset_size": len(tokenized_dataset),
                "total_tokens_processed": len(tokenized_dataset) * avg_seq_len,
                "estimated_flops_tflops": total_flops / 1e12,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation,
                "effective_batch_size": args.batch_size * args.gradient_accumulation,
            },
            "tokenizer_info": {
                "type": "NTFTokenizer (EthioBBPE)" if tokenizer else "Character-level (fallback)",
                "vocab_size": vocab_size,
                "pretrained_path": args.tokenizer_path or "Default EthioBBPE",
            },
            "hyperparameters": {
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "mixed_precision": args.mixed_precision,
                "dropout": args.dropout,
            }
        }
        
        metrics_path = output_dir / "training_metrics_report.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_report, f, indent=2)
        print(f"  • Metrics report:        {metrics_path}")
        
        print("\n" + "=" * 60)
        print("✅ BASE KNOWLEDGE TRAINING COMPLETE!")
        print("=" * 60)
        print("\n📌 NEXT STEPS:")
        print("  1. The model now contains Ethiopian Orthodox religious knowledge")
        print("  2. This checkpoint serves as your FROZEN BASE KNOWLEDGE")
        print("  3. For future tasks, load this checkpoint and freeze the backbone")
        print("  4. Fine-tune only adapter layers or task-specific heads")
        print("\n⚠️  IMPORTANT: This model should now be treated as frozen base knowledge.")
        print("   Do not continue training on unrelated data without proper safeguards.")
        print("=" * 60 + "\n")
        
        return metrics
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user.")
        print("Saving current checkpoint...")
        trainer._save_checkpoint()
        print("Checkpoint saved. You can resume training later.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train blank slate transformer on Ethiopian religious texts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing dataset parquet files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save checkpoints and logs")
    
    # Tokenizer options
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to pretrained tokenizer or HuggingFace model ID")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Path to vocabulary file (vocab.json)")
    parser.add_argument("--merges_file", type=str, default=None,
                        help="Path to BPE merges file (merges.txt)")
    parser.add_argument("--max_vocab_size", type=int, default=16000,
                        help="Maximum vocabulary size for fallback tokenizer")
    
    # Model architecture
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size preset")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length (alias: --max_seq_length)")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Maximum sequence length (alias for --max_seq_len)")
    parser.add_argument("--no_rope", action="store_true",
                        help="Disable Rotary Positional Embeddings")
    parser.add_argument("--no_swiglu", action="store_true",
                        help="Disable SwiGLU activation (use GELU)")
    parser.add_argument("--no_rmsnorm", action="store_true",
                        help="Disable RMSNorm (use LayerNorm)")
    parser.add_argument("--no_tie_embeddings", action="store_true",
                        help="Don't tie input/output embeddings")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Peak learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of warmup steps (overrides warmup_ratio if specified)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # Checkpointing and logging
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log metrics every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Performance options
    parser.add_argument("--mixed_precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Mixed precision mode")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for memory efficiency")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="Don't shuffle training data")
    
    # Dataset options
    parser.add_argument("--include_english", action="store_true",
                        help="Include English verses for bilingual training")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Run name
    parser.add_argument("--run_name", type=str, default="blank_slate_ethiopian_religious",
                        help="Name of the training run")
    
    # Data loading optimization
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of subprocesses for data loading (default: 4, increase for GPU)")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches to prefetch per worker (default: 2)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
