"""
Data collation for language modeling.

Provides efficient batch collation with padding and label handling
for autoregressive language model training.
Integrated with NTFTokenizer (EthioBBPE-based) for Ethiopian languages.
"""

from typing import Dict, List, Optional, Any, Union
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset

try:
    from tokenizer import NTFTokenizer
    NTF_TOKENIZER_AVAILABLE = True
except ImportError:
    NTF_TOKENIZER_AVAILABLE = False
    NTFTokenizer = None


class TextDataset(Dataset):
    """
    Simple text dataset that loads texts from a file.
    
    Args:
        file_path: Path to text file (one sample per line)
        max_length: Maximum sequence length
        tokenizer_name: Tokenizer to use ('char' for character-level)
    """
    
    def __init__(
        self,
        file_path: str,
        max_length: int = 512,
        tokenizer_name: str = "char",
    ):
        self.max_length = max_length
        
        # Load texts from file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]
        
        # Create simple character-level tokenizer
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character-level vocabulary."""
        all_chars = set()
        for text in self.texts:
            all_chars.update(text)
        
        # Special tokens
        self.char_to_idx['<pad>'] = 0
        self.char_to_idx['<unk>'] = 1
        
        idx = len(self.char_to_idx)
        for char in sorted(all_chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                idx += 1
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text to character IDs."""
        return [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in text]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        input_ids = self._tokenize(text)[:self.max_length]
        
        # Ensure we have at least some tokens
        if len(input_ids) == 0:
            input_ids = [self.char_to_idx['<pad>']]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "labels": torch.stack([item["labels"] for item in x]),
        },
    )


class DataCollatorForLanguageModeling:
    """
    Data collator for language modeling tasks.
    
    Features:
    - Dynamic padding to max length in batch
    - Label creation for next-token prediction
    - Optional masking for MLM (not used in decoder-only)
    - Efficient tensor conversion
    - Compatible with EthioBBPE and HuggingFace tokenizers
    
    Args:
        pad_token_id: Token ID for padding
        max_length: Maximum sequence length (None for dynamic)
        return_tensors: Type of tensors to return ('pt', 'np')
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            examples: List of dictionaries with 'input_ids' and optional 'labels'
        
        Returns:
            Batch dictionary with padded tensors
        """
        # Extract input_ids from examples
        input_ids_list = [example["input_ids"] for example in examples]
        
        # Handle labels
        has_labels = "labels" in examples[0]
        if has_labels:
            labels_list = [example["labels"] for example in examples]
        else:
            # Use input_ids as labels (standard for LM)
            labels_list = input_ids_list
        
        # Truncate if max_length specified
        if self.max_length is not None:
            input_ids_list = [ids[:self.max_length] for ids in input_ids_list]
            labels_list = [lbl[:self.max_length] for lbl in labels_list]
        
        # Convert to tensors
        input_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
        labels_tensors = [torch.tensor(lbl, dtype=torch.long) for lbl in labels_list]
        
        # Pad sequences
        input_ids = pad_sequence(
            input_ids_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            labels_tensors,
            batch_first=True,
            padding_value=-100,  # Ignore padding in loss calculation
        )
        
        # Create attention mask
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_training_dataset(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
    stride: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """
    Create a dataset from raw texts.
    
    Supports NTFTokenizer, EthioBBPE, and HuggingFace tokenizers.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance (NTFTokenizer, EthioBBPE, or HF tokenizer)
        max_length: Maximum sequence length
        stride: Stride for chunking long texts (None for truncation)
    
    Returns:
        Dataset with tokenized examples
    """
    from datasets import Dataset
    
    # Detect tokenizer type
    is_ntf_tokenizer = NTF_TOKENIZER_AVAILABLE and isinstance(tokenizer, NTFTokenizer)
    is_ethiobbpe = hasattr(tokenizer, 'encode_batch') or type(tokenizer).__name__ in ['EthioBBPE', 'NTFTokenizer']
    
    # Tokenize all texts
    def tokenize_function(examples):
        if is_ntf_tokenizer:
            # NTFTokenizer - use callable interface
            encoded = tokenizer(examples["text"], add_special_tokens=True, padding=False, truncation=True, max_length=max_length)
            return {"input_ids": encoded["input_ids"]}
        elif is_ethiobbpe:
            # EthioBBPE-style tokenizer with encode_batch
            encoded = tokenizer.encode_batch(examples["text"])
            # Handle both TokenizerOutput objects and dict returns
            if hasattr(encoded[0], 'ids'):
                input_ids = [item.ids for item in encoded]
            else:
                input_ids = [item["ids"] for item in encoded]
            return {"input_ids": input_ids}
        else:
            # HuggingFace tokenizer
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_special_tokens_mask=False,
            )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # Group texts if needed for longer context
    if stride is not None:
        tokenized_dataset = _group_texts(tokenized_dataset, max_length, stride)
    
    return tokenized_dataset


def _group_texts(
    dataset,
    max_length: int,
    stride: int,
) -> torch.utils.data.Dataset:
    """Group shorter sequences into longer ones."""
    
    def group_function(examples):
        # Concatenate all input_ids
        concatenated_ids = sum(examples["input_ids"], [])
        
        # Calculate number of chunks
        total_length = len(concatenated_ids)
        if total_length <= max_length:
            return {"input_ids": [concatenated_ids]}
        
        # Create chunks with stride
        chunks = []
        for i in range(0, total_length - max_length + 1, stride):
            chunks.append(concatenated_ids[i : i + max_length])
        
        # Add final chunk if there's remaining data
        if total_length % stride != 0 or total_length < max_length:
            remaining = total_length - ((total_length // stride) * stride)
            if remaining > 0:
                chunks.append(concatenated_ids[-max_length:])
        
        return {"input_ids": chunks}
    
    return dataset.map(
        group_function,
        batched=True,
        desc="Grouping texts",
    )
