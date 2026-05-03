"""
NTF Tokenizer Package - Official Tokenizer for Nexuss Transformer Framework

This package provides the official NTF tokenizer based on EthioBBPE,
optimized for Ethiopian languages and Ge'ez script.
"""

from tokenizer.ntf_tokenizer import (
    NTFTokenizer,
    TokenizerOutput,
    get_ntf_tokenizer,
)

__version__ = "1.0.0"
__all__ = [
    "NTFTokenizer",
    "TokenizerOutput",
    "get_ntf_tokenizer",
]
