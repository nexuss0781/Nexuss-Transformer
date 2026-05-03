"""
NTF Tokenizer - Official Nexuss Transformer Framework Tokenizer

This module provides the official tokenizer for the NTF framework,
integrating the EthioBBPE (Ethiopic Byte-level BPE) tokenizer with
full support for Amharic, Ge'ez script, and Ethiopian languages.

Features:
- Byte-level BPE tokenization optimized for Ge'ez script
- Perfect reconstruction of Ethiopic characters and punctuation
- Support for special tokens (<pad>, <unk>, <s>, </s>, <mask>)
- Compressed vocabulary loading (gzip)
- Batch encoding/decoding
- Compatible with HuggingFace transformers ecosystem

Based on: https://huggingface.co/Nexuss0781/Ethio-BBPE
"""

import json
import gzip
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class TokenizerOutput:
    """Output container for tokenizer encoding operations."""
    ids: List[int]
    tokens: List[str]
    offsets: Optional[List[Tuple[int, int]]] = None
    attention_mask: Optional[List[int]] = None
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> int:
        return self.ids[idx]


class NTFTokenizer:
    """
    Official NTF Tokenizer based on EthioBBPE.
    
    A production-ready Byte-level BPE tokenizer specifically designed for
    Amharic biblical and religious texts, achieving perfect reconstruction
    of complex Ge'ez script, ancient punctuation, and liturgical content.
    
    Args:
        vocab_file: Path to vocabulary file (vocab.json or vocab.json.gz)
        merges_file: Path to merge rules file (merges.txt)
        special_tokens: List of special tokens to add
        unk_token: Unknown token string
        pad_token: Padding token string
        bos_token: Beginning of sequence token string
        eos_token: End of sequence token string
        mask_token: Mask token string
        
    Example:
        >>> tokenizer = NTFTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")
        >>> text = "ሰላም ለኢዮብ ዘኢነበበ ከንቶ ።"
        >>> encoded = tokenizer.encode(text)
        >>> print(f"Tokens: {encoded.tokens}")
        >>> print(f"IDs: {encoded.ids}")
        >>> decoded = tokenizer.decode(encoded.ids)
        >>> assert text == decoded  # Perfect reconstruction!
    """
    
    # Default special tokens
    DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
    
    # Default configuration
    DEFAULT_CONFIG = {
        "vocab_size": 16000,
        "min_frequency": 2,
        "lowercase": False,
    }
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        mask_token: str = "<mask>",
        use_compressed: bool = True,
    ):
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.use_compressed = use_compressed
        
        # Special tokens
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS.copy()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        
        # Vocabulary and merges
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # Byte-level mapping
        self.byte_encoder: Dict[int, str] = {}
        self.byte_decoder: Dict[str, int] = {}
        self._build_byte_mapping()
        
        # Load vocabulary and merges if files provided
        if vocab_file:
            self._load_vocab(vocab_file)
        if merges_file:
            self._load_merges(merges_file)
        
        # Build BPE ranks
        self._build_bpe_ranks()
        
        # Cache for tokenization
        self._cache: Dict[str, List[str]] = {}
    
    def _build_byte_mapping(self):
        """Build byte-level character mapping for byte-level BPE."""
        # Map bytes 0-255 to unicode characters
        chars = [chr(n) for n in range(256)]
        
        # Shift non-printable characters to printable range
        # This is the standard byte-level BPE approach
        byte_encoder = {}
        byte_decoder = {}
        
        n = 0
        for b in range(256):
            # Skip control characters and map them to printable unicode
            if b < 33 or (b > 126 and b < 161) or b > 254:
                # Map to unicode private use area or other printable chars
                byte_encoder[b] = chr(256 + n)
                byte_decoder[chr(256 + n)] = b
                n += 1
            else:
                byte_encoder[b] = chr(b)
                byte_decoder[chr(b)] = b
        
        self.byte_encoder = byte_encoder
        self.byte_decoder = byte_decoder
    
    def _load_vocab(self, vocab_file: str):
        """Load vocabulary from file (supports .json and .json.gz)."""
        path = Path(vocab_file)
        
        if path.suffix == '.gz' or str(path).endswith('.json.gz'):
            # Load compressed vocabulary
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                self.vocab = json.load(f)
        else:
            # Load uncompressed vocabulary
            with open(path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        
        # Build reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def _load_merges(self, merges_file: str):
        """Load BPE merge rules from file."""
        self.merges = []
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        self.merges.append((parts[0], parts[1]))
    
    def _build_bpe_ranks(self):
        """Build BPE merge rank dictionary."""
        self.bpe_ranks = {merge: i for i, merge in enumerate(self.merges)}
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "NTFTokenizer":
        """
        Load a pretrained tokenizer from HuggingFace Hub or local path.
        
        Args:
            model_name_or_path: HuggingFace model ID or local directory path
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            NTFTokenizer instance
            
        Example:
            >>> tokenizer = NTFTokenizer.from_pretrained("Nexuss0781/Ethio-BBPE")
        """
        # Check if it's a local path
        if os.path.isdir(model_name_or_path):
            local_path = model_name_or_path
        else:
            # Try to download from HuggingFace Hub
            try:
                from huggingface_hub import snapshot_download
                local_path = snapshot_download(repo_id=model_name_or_path)
            except ImportError:
                raise ImportError(
                    "Please install huggingface-hub: pip install huggingface-hub"
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not load tokenizer from '{model_name_or_path}'. "
                    f"Make sure it's a valid HuggingFace model ID or local path."
                ) from e
        
        # Priority 1: Try tokenizer.json (standard HF format)
        tokenizer_json_candidates = [
            os.path.join(local_path, "tokenizer.json"),
            os.path.join(local_path, "models", "tokenizer.json"),
        ]
        for candidate in tokenizer_json_candidates:
            if os.path.exists(candidate):
                return cls.from_file(candidate, **kwargs)
        
        # Priority 2: Try vocab files
        vocab_file = None
        merges_file = None
        
        # Check for compressed vocab first (real gzip file, not LFS pointer)
        compressed_vocab = os.path.join(local_path, "vocab.json.gz")
        if os.path.exists(compressed_vocab) and cls._is_valid_gzip(compressed_vocab):
            vocab_file = compressed_vocab
        else:
            regular_vocab = os.path.join(local_path, "vocab.json")
            if os.path.exists(regular_vocab):
                vocab_file = regular_vocab
        
        # Check models subdirectory
        if not vocab_file:
            models_vocab_gz = os.path.join(local_path, "models", "vocab.json.gz")
            if os.path.exists(models_vocab_gz) and cls._is_valid_gzip(models_vocab_gz):
                vocab_file = models_vocab_gz
            else:
                models_vocab = os.path.join(local_path, "models", "vocab.json")
                if os.path.exists(models_vocab):
                    vocab_file = models_vocab
        
        # Find merges file
        merges_candidates = [
            os.path.join(local_path, "merges.txt"),
            os.path.join(local_path, "models", "merges.txt"),
        ]
        for candidate in merges_candidates:
            if os.path.exists(candidate):
                merges_file = candidate
                break
        
        if not vocab_file:
            raise FileNotFoundError(
                f"Could not find valid tokenizer files in {local_path}. "
                f"Looking for tokenizer.json, vocab.json, or vocab.json.gz"
            )
        
        # Use compressed by default if available
        use_compressed = kwargs.pop('use_compressed', vocab_file.endswith('.gz'))
        
        return cls(
            vocab_file=vocab_file,
            merges_file=merges_file,
            use_compressed=use_compressed,
            **kwargs
        )
    
    @staticmethod
    def _is_valid_gzip(file_path: str) -> bool:
        """Check if a file is a valid gzip file (not an LFS pointer)."""
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(2)
                return magic == b'\x1f\x8b'
        except Exception:
            return False
    
    @classmethod
    def from_file(cls, tokenizer_json_path: str, **kwargs) -> "NTFTokenizer":
        """
        Load tokenizer from a standard tokenizer.json file.
        
        Args:
            tokenizer_json_path: Path to tokenizer.json file
            **kwargs: Additional arguments
            
        Returns:
            NTFTokenizer instance
        """
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # Extract vocab from model
        vocab = tokenizer_data.get("model", {}).get("vocab", {})
        
        # Create temporary vocab file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
            temp_vocab = f.name
        
        try:
            # Get special tokens
            added_tokens = tokenizer_data.get("added_tokens", [])
            special_tokens = [t["content"] for t in added_tokens if t.get("special", False)]
            
            tokenizer = cls(
                vocab_file=temp_vocab,
                special_tokens=special_tokens if special_tokens else None,
                **kwargs
            )
            
            # Store additional metadata
            tokenizer.normalizer_config = tokenizer_data.get("normalizer")
            tokenizer.pre_tokenizer_config = tokenizer_data.get("pre_tokenizer")
            tokenizer.decoder_config = tokenizer_data.get("decoder")
            
            return tokenizer
        finally:
            os.unlink(temp_vocab)
    
    def _tokenize_to_bytes(self, text: str) -> List[str]:
        """Convert text to byte-level characters using HuggingFace tokenizers ByteLevel."""
        # Use the tokenizers library's ByteLevel encoding for proper handling
        try:
            from tokenizers import Tokenizer, pre_tokenizers
            # Create a minimal tokenizer with just ByteLevel pretokenization
            byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
            # The pretokenize method returns list of (token, offsets) tuples
            result = []
            # Process character by character through byte level
            for char in text:
                # Encode each character to bytes, then map to byte-level chars
                char_bytes = char.encode('utf-8')
                for b in char_bytes:
                    result.append(self.byte_encoder[b])
            return result
        except Exception:
            # Fallback to manual byte encoding
            return [self.byte_encoder[b] for b in text.encode('utf-8')]
    
    def _convert_tokens_to_text(self, tokens: List[str]) -> str:
        """Convert byte-level tokens back to UTF-8 text."""
        # Convert byte-level chars back to bytes
        byte_values = []
        for char in tokens:
            if char in self.byte_decoder:
                byte_values.append(self.byte_decoder[char])
            else:
                byte_values.append(ord('?'))
        
        # Decode bytes to UTF-8 string
        try:
            return bytes(byte_values).decode('utf-8')
        except UnicodeDecodeError:
            return ''.join(chr(b) if b < 256 else '?' for b in byte_values)
    
    def _get_pairs(self, tokens: List[str]) -> set:
        """Get all adjacent pairs of tokens."""
        return set(zip(tokens[:-1], tokens[1:]))
    
    def _bpe(self, tokens: List[str]) -> List[str]:
        """Apply BPE merges to a list of tokens."""
        # Check cache
        token_str = ' '.join(tokens)
        if token_str in self._cache:
            return self._cache[token_str].copy()
        
        word = tokens.copy()
        
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # Find the pair with the lowest rank (earliest merge)
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            if best_pair is None or best_pair not in self.bpe_ranks:
                break
            
            first, second = best_pair
            
            # Merge all occurrences of the best pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            
            if len(word) == 1:
                break
        
        # Cache result
        self._cache[token_str] = word.copy()
        return word
    
    def encode(self, text: str, add_special_tokens: bool = True) -> TokenizerOutput:
        """
        Encode a single text string to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            TokenizerOutput with ids, tokens, and optional metadata
        """
        # Convert to byte-level tokens
        byte_tokens = self._tokenize_to_bytes(text)
        
        # Apply BPE
        bpe_tokens = self._bpe(byte_tokens)
        
        # Convert to IDs
        token_ids = []
        output_tokens = []
        
        for token in bpe_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
                output_tokens.append(token)
            else:
                # Use UNK token
                token_ids.append(self.vocab.get(self.unk_token, 1))
                output_tokens.append(self.unk_token)
        
        # Add special tokens
        if add_special_tokens:
            bos_id = self.vocab.get(self.bos_token, 2)
            eos_id = self.vocab.get(self.eos_token, 3)
            token_ids = [bos_id] + token_ids + [eos_id]
            output_tokens = [self.bos_token] + output_tokens + [self.eos_token]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        return TokenizerOutput(
            ids=token_ids,
            tokens=output_tokens,
            attention_mask=attention_mask,
        )
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[TokenizerOutput]:
        """
        Encode a batch of text strings.
        
        Args:
            texts: List of input text strings
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of TokenizerOutput objects
        """
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        # Filter special tokens if requested
        if skip_special_tokens:
            special_ids = {
                self.vocab.get(self.pad_token, 0),
                self.vocab.get(self.unk_token, 1),
                self.vocab.get(self.bos_token, 2),
                self.vocab.get(self.eos_token, 3),
                self.vocab.get(self.mask_token, 4),
            }
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        # Convert IDs to tokens
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append(self.unk_token)
        
        # Convert byte-level tokens back to text
        return self._convert_tokens_to_text(tokens)
    
    def decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        """
        Decode a batch of token ID sequences.
        
        Args:
            batch_token_ids: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_token_ids]
    
    def _convert_tokens_to_text(self, tokens: List[str]) -> str:
        """Convert byte-level tokens back to UTF-8 text."""
        # Concatenate tokens
        concatenated = ''.join(tokens)
        
        # Convert byte-level chars back to bytes
        byte_values = []
        for char in concatenated:
            if char in self.byte_decoder:
                byte_values.append(self.byte_decoder[char])
            else:
                # Handle unknown characters (shouldn't happen with proper vocab)
                byte_values.append(ord('?'))
        
        # Decode bytes to UTF-8 string
        try:
            return bytes(byte_values).decode('utf-8')
        except UnicodeDecodeError:
            # Fallback: return what we can
            return ''.join(chr(b) if b < 256 else '?' for b in byte_values)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the full vocabulary dictionary."""
        return self.vocab.copy()
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to ID(s)."""
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.vocab.get(self.unk_token, 1))
        return [self.vocab.get(token, self.vocab.get(self.unk_token, 1)) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert ID(s) to token(s)."""
        if isinstance(ids, int):
            return self.id_to_token.get(ids, self.unk_token)
        return [self.id_to_token.get(id_, self.unk_token) for id_ in ids]
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size property."""
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.vocab.get(self.pad_token, 0)
    
    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.vocab.get(self.unk_token, 1)
    
    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.vocab.get(self.bos_token, 2)
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.vocab.get(self.eos_token, 3)
    
    @property
    def mask_token_id(self) -> int:
        """Mask token ID."""
        return self.vocab.get(self.mask_token, 4)
    
    def save_pretrained(self, save_directory: str, filename_prefix: Optional[str] = None):
        """
        Save tokenizer files to a directory.
        
        Args:
            save_directory: Directory to save tokenizer files
            filename_prefix: Optional prefix for filenames
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = filename_prefix + "_" if filename_prefix else ""
        
        # Save vocab (compressed)
        vocab_path = save_dir / f"{prefix}vocab.json.gz"
        with gzip.open(vocab_path, 'wt', encoding='utf-8', compresslevel=9) as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        if self.merges:
            merges_path = save_dir / f"{prefix}merges.txt"
            with open(merges_path, 'w', encoding='utf-8') as f:
                f.write("#version: 0.2\n")
                for merge in self.merges:
                    f.write(f"{merge[0]} {merge[1]}\n")
        
        # Save special tokens map
        special_tokens_map = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "mask_token": self.mask_token,
        }
        special_tokens_path = save_dir / f"{prefix}special_tokens_map.json"
        with open(special_tokens_path, 'w', encoding='utf-8') as f:
            json.dump(special_tokens_map, f, indent=2)
        
        # Save tokenizer config
        tokenizer_config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "model_max_length": 512,
        }
        config_path = save_dir / f"{prefix}tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tokenize text(s) - callable interface compatible with HF tokenizers.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Add special tokens
            padding: Pad sequences to max length
            truncation: Truncate sequences to max length
            max_length: Maximum sequence length
            return_tensors: Return type ('pt' for PyTorch, 'np' for numpy, None for lists)
            
        Returns:
            Dictionary with input_ids, attention_mask, and optionally other fields
        """
        is_batched = isinstance(text, list)
        
        if not is_batched:
            text = [text]
        
        # Encode all texts
        outputs = self.encode_batch(text, add_special_tokens=add_special_tokens)
        
        # Extract ids and masks
        input_ids = [out.ids for out in outputs]
        attention_mask = [out.attention_mask for out in outputs]
        
        # Apply truncation
        if truncation and max_length is not None:
            input_ids = [ids[:max_length] for ids in input_ids]
            attention_mask = [mask[:max_length] for mask in attention_mask]
        
        # Apply padding
        if padding or (truncation and max_length is not None):
            max_len = max(len(ids) for ids in input_ids)
            if max_length is not None:
                max_len = min(max_len, max_length)
            
            pad_id = self.pad_token_id
            for i in range(len(input_ids)):
                pad_length = max_len - len(input_ids[i])
                if pad_length > 0:
                    input_ids[i].extend([pad_id] * pad_length)
                    attention_mask[i].extend([0] * pad_length)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        elif return_tensors == "np":
            import numpy as np
            input_ids = np.array(input_ids, dtype=np.int64)
            attention_mask = np.array(attention_mask, dtype=np.int64)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if not is_batched and return_tensors is None:
            result = {k: v[0] if hasattr(v, '__getitem__') else v for k, v in result.items()}
        
        return result
    
    def __repr__(self) -> str:
        return (
            f"NTFTokenizer(vocab_size={self.vocab_size}, "
            f"unk_token='{self.unk_token}', "
            f"pad_token='{self.pad_token}', "
            f"bos_token='{self.bos_token}', "
            f"eos_token='{self.eos_token}')"
        )


# Convenience function for quick access
def get_ntf_tokenizer(
    model_name: str = "Nexuss0781/Ethio-BBPE",
    **kwargs
) -> NTFTokenizer:
    """
    Get the official NTF tokenizer.
    
    Args:
        model_name: Model name or path for the tokenizer
        **kwargs: Additional arguments passed to NTFTokenizer.from_pretrained
        
    Returns:
        NTFTokenizer instance
        
    Example:
        >>> tokenizer = get_ntf_tokenizer()
        >>> encoded = tokenizer("ሰላም ዓለም")
    """
    return NTFTokenizer.from_pretrained(model_name, **kwargs)


__all__ = [
    "NTFTokenizer",
    "TokenizerOutput",
    "get_ntf_tokenizer",
]
