"""
Data collation for language modeling.

Provides efficient batch collation with padding and label handling
for autoregressive language model training.
"""

from typing import Dict, List, Optional, Any
import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForLanguageModeling:
    """
    Data collator for language modeling tasks.
    
    Features:
    - Dynamic padding to max length in batch
    - Label creation for next-token prediction
    - Optional masking for MLM (not used in decoder-only)
    - Efficient tensor conversion
    
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
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        stride: Stride for chunking long texts (None for truncation)
    
    Returns:
        Dataset with tokenized examples
    """
    from datasets import Dataset
    
    # Tokenize all texts
    def tokenize_function(examples):
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
