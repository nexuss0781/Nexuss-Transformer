#!/usr/bin/env python3
"""
Blank Slate Training Script for Synaxarium and Canon Biblical Datasets

This script trains a decoder-only transformer from scratch on Ethiopian Orthodox
religious texts (Synaxarium) and Amharic-English parallel Bible corpus.

The trained model will have these datasets as frozen base knowledge.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import torch
from datasets import Dataset

# Add project root and Nexuss-Transformer to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Nexuss-Transformer"))

from models.config import NTFConfig
from models.transformer import NexussTransformer
from training.config import TrainingConfig, Precision, OptimizerType, SchedulerType
from training.trainer import Trainer
from training.data import DataCollatorForLanguageModeling


def load_synaxarium_dataset(parquet_path: str) -> List[str]:
    """
    Load Synaxarium dataset and format as text.
    
    Args:
        parquet_path: Path to synaxarium_dataset.parquet
    
    Returns:
        List of formatted text strings
    """
    df = pd.read_parquet(parquet_path)
    texts = []
    
    for _, row in df.iterrows():
        # Format: Month Day\nText
        month = row['ወር']
        day = row['ቀን']
        content = row['መጽሃፍ']
        
        text = f"ስንክሳር - ወር: {month}, ቀን: {day}\n{content}"
        texts.append(text)
    
    print(f"Loaded {len(texts)} Synaxarium entries")
    return texts


def load_canon_biblical_dataset(parquet_path: str, include_english: bool = False) -> List[str]:
    """
    Load Canon Biblical dataset and format as text.
    
    Args:
        parquet_path: Path to train-00000-of-00001.parquet
        include_english: Whether to include English verses (for bilingual training)
    
    Returns:
        List of formatted text strings
    """
    df = pd.read_parquet(parquet_path)
    texts = []
    
    for _, row in df.iterrows():
        book = row['መጽሐፍ']
        chapter = row['ምዕራፍ']
        verse_num = row['ቁጥር']
        amharic = row['ጥቅስ']
        english = row.get('verse', '')
        
        if include_english and english:
            # Bilingual format
            text = f"መጽሐፍ ቅዱስ - {book} {chapter}:{verse_num}\nአማርኛ: {amharic}\nEnglish: {english}"
        else:
            # Amharic only
            text = f"መጽሐፍ ቅዱስ - {book} {chapter}:{verse_num}\n{amharic}"
        
        texts.append(text)
    
    print(f"Loaded {len(texts)} Canon Biblical verses")
    return texts


def create_training_dataset(
    synaxarium_texts: List[str],
    biblical_texts: List[str],
    shuffle: bool = True
) -> Dataset:
    """
    Combine datasets and create HuggingFace Dataset.
    
    Args:
        synaxarium_texts: Texts from Synaxarium dataset
        biblical_texts: Texts from Canon Biblical dataset
        shuffle: Whether to shuffle the combined dataset
    
    Returns:
        HuggingFace Dataset object
    """
    # Combine all texts
    all_texts = synaxarium_texts + biblical_texts
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Synaxarium entries: {len(synaxarium_texts)}")
    print(f"Biblical verses: {len(biblical_texts)}")
    print(f"Total samples: {len(all_texts)}")
    
    # Calculate character statistics
    total_chars = sum(len(t) for t in all_texts)
    avg_chars = total_chars / len(all_texts) if all_texts else 0
    print(f"Total characters: {total_chars:,}")
    print(f"Average length: {avg_chars:.1f} chars")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({"text": all_texts})
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    return dataset


def create_tokenizer(vocab_size: int = 16000):
    """
    Create or load tokenizer.
    
    For Ethiopian languages, we use EthioBBPE or a simple character-level tokenizer.
    This function creates a basic tokenizer for demonstration.
    
    Args:
        vocab_size: Target vocabulary size
    
    Returns:
        Tokenizer object
    """
    try:
        # Try to use EthioBBPE if available
        from ethiobbpe import EthioBBPE
        tokenizer = EthioBBPE()
        print("Using EthioBBPE tokenizer")
        return tokenizer
    except ImportError:
        print("EthioBBPE not available, using character-level tokenizer")
        return None


class SimpleCharTokenizer:
    """Simple character-level tokenizer for blank slate training."""
    
    def __init__(self, texts: List[str], max_vocab_size: int = 16000):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_vocab(texts, max_vocab_size)
    
    def _build_vocab(self, texts: List[str], max_vocab_size: int):
        """Build character-level vocabulary from texts."""
        # Special tokens
        self.char_to_idx['<pad>'] = 0
        self.char_to_idx['<unk>'] = 1
        self.char_to_idx['<bos>'] = 2
        self.char_to_idx['<eos>'] = 3
        
        # Collect all characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Add characters to vocab
        idx = len(self.char_to_idx)
        for char in sorted(all_chars):
            if idx >= max_vocab_size:
                break
            self.char_to_idx[char] = idx
            idx += 1
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        print(f"Vocabulary size: {len(self.char_to_idx)}")
    
    def encode(self, text: str, max_length: int = None) -> List[int]:
        """Encode text to token IDs."""
        ids = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in text]
        if max_length:
            ids = ids[:max_length]
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([self.idx_to_char.get(idx, '<unk>') for idx in ids])
    
    @property
    def vocab_size(self) -> int:
        return len(self.char_to_idx)
    
    @property
    def pad_token_id(self) -> int:
        return self.char_to_idx['<pad>']


class TokenizedDataset(torch.utils.data.Dataset):
    """Dataset with on-the-fly tokenization."""
    
    def __init__(self, hf_dataset: Dataset, tokenizer, max_length: int = 512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataset[idx]['text']
        
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        else:
            input_ids = self.tokenizer.encode(text)[:self.max_length]
        
        # Ensure minimum length
        if len(input_ids) == 0:
            input_ids = [self.tokenizer.pad_token_id]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def main():
    """Main training function."""
    
    # =====================
    # Configuration
    # =====================
    
    # Paths
    data_dir = Path(__file__).parent / "data"
    synaxarium_path = data_dir / "synaxarium_dataset.parquet"
    biblical_path = data_dir / "train-00000-of-00001.parquet"
    output_dir = Path(__file__).parent / "outputs" / "blank_slate_ethiopian_religious"
    
    # Verify data files exist
    assert synaxarium_path.exists(), f"Synaxarium dataset not found: {synaxarium_path}"
    assert biblical_path.exists(), f"Biblical dataset not found: {biblical_path}"
    
    print("=" * 60)
    print("BLANK SLATE TRAINING - Ethiopian Religious Texts")
    print("=" * 60)
    
    # =====================
    # Load Datasets
    # =====================
    
    print("\n[1/6] Loading datasets...")
    synaxarium_texts = load_synaxarium_dataset(str(synaxarium_path))
    biblical_texts = load_canon_biblical_dataset(str(biblical_path), include_english=False)
    
    # =====================
    # Create Combined Dataset
    # =====================
    
    print("\n[2/6] Creating combined dataset...")
    dataset = create_training_dataset(synaxarium_texts, biblical_texts)
    
    # =====================
    # Create Tokenizer
    # =====================
    
    print("\n[3/6] Creating tokenizer...")
    tokenizer = SimpleCharTokenizer(
        synaxarium_texts + biblical_texts,
        max_vocab_size=8000  # Character-level + common subwords
    )
    vocab_size = tokenizer.vocab_size
    
    # =====================
    # Create Model
    # =====================
    
    print("\n[4/6] Creating blank slate model...")
    
    # Use small model config for efficient training
    model_config = NTFConfig.small()
    model_config.vocab_size = vocab_size
    model_config.max_seq_len = 512
    model_config.use_rope = True
    model_config.activation = "swiglu"
    model_config.dropout = 0.1
    
    model = NexussTransformer(model_config)
    
    param_info = model.count_parameters()
    total_params = param_info['total'] / 1e6
    print(f"Model architecture: Small ({total_params:.2f}M parameters)")
    print(f"  - Vocabulary: {vocab_size}")
    print(f"  - Embedding dim: {model_config.d_model}")
    print(f"  - Layers: {model_config.n_layers}")
    print(f"  - Heads: {model_config.n_heads}")
    print(f"  - Max sequence: {model_config.max_seq_len}")
    
    # =====================
    # Prepare Training Dataset
    # =====================
    
    print("\n[5/6] Preparing training dataset...")
    max_length = model_config.max_seq_len
    train_dataset = TokenizedDataset(dataset, tokenizer, max_length=max_length)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
    )
    
    # =====================
    # Configure Training
    # =====================
    
    print("\n[6/6] Configuring training...")
    
    training_config = TrainingConfig(
        # Output
        output_dir=str(output_dir),
        run_name="blank_slate_ethiopian_religious",
        
        # Training duration - adjust based on your needs
        num_train_epochs=10,
        max_steps=-1,
        
        # Batch sizes
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        
        # Learning rate
        learning_rate=1e-3,
        weight_decay=0.01,
        
        # Optimizer & scheduler
        optimizer=OptimizerType.ADAMW,
        scheduler=SchedulerType.LINEAR,
        warmup_ratio=0.05,
        
        # Precision
        mixed_precision=Precision.FP32,
        
        # Checkpointing
        save_steps=100,
        save_total_limit=3,
        
        # Logging
        logging_steps=10,
        eval_steps=100,
        
        # Performance
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        
        # Regularization
        max_grad_norm=1.0,
        
        # Reproducibility
        seed=42,
        
        # Logging backend
        report_to="none",
    )
    
    effective_batch_size = training_config.effective_batch_size
    print(f"Training configuration:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Epochs: {training_config.num_train_epochs}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Learning rate: {training_config.learning_rate}")
    print(f"  - Warmup ratio: {training_config.warmup_ratio}")
    print(f"  - Save every {training_config.save_steps} steps")
    
    # =====================
    # Initialize Trainer
    # =====================
    
    print("\n" + "=" * 60)
    print("INITIALIZING TRAINER...")
    print("=" * 60)
    
    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=None,  # No separate eval set for now
        data_collator=data_collator,
    )
    
    # =====================
    # Start Training
    # =====================
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    print("This will train a blank slate model from random initialization.")
    print("The model will learn Ethiopian Orthodox religious texts as base knowledge.")
    print("Training logs will be saved to:", output_dir)
    print("=" * 60)
    
    # Train the model
    metrics = trainer.train()
    
    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Final training loss: {metrics['train_loss']:.4f}")
    print(f"Global steps: {metrics['global_step']}")
    print(f"Epochs trained: {metrics['epochs_trained']}")
    print(f"Training time: {metrics['training_time_seconds']:.1f} seconds")
    print(f"Samples per second: {metrics['samples_per_second']:.2f}")
    print(f"\nModel saved to: {output_dir}")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
