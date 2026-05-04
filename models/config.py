"""
Model configuration for decoder-only transformer.

Provides a comprehensive configuration class with all hyperparameters
needed for building a blank slate transformer model.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.
    
    Supports multiple quantization methods:
    - bitsandbytes: 4-bit and 8-bit quantization
    - gptq: GPU-aware INT4 quantization
    - awq: Activation-aware weight quantization
    
    Args:
        method: Quantization method ('bitsandbytes', 'gptq', 'awq')
        load_in_4bit: Load model in 4-bit precision
        load_in_8bit: Load model in 8-bit precision
        bnb_4bit_quant_type: 4-bit quantization type ('nf4', 'fp4')
        bnb_4bit_compute_dtype: Compute dtype for 4-bit operations
        bnb_4bit_use_double_quant: Use nested quantization
        gptq_bits: Number of bits for GPTQ
        gptq_group_size: Group size for GPTQ
        awq_w_bit: Bit width for AWQ
        awq_group_size: Group size for AWQ
    """
    
    # Method selection
    method: str = "bitsandbytes"  # 'bitsandbytes', 'gptq', 'awq'
    
    # BitsAndBytes configuration
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # GPTQ configuration
    gptq_bits: int = 4
    gptq_group_size: int = 128
    
    # AWQ configuration
    awq_w_bit: int = 4
    awq_group_size: int = 128
    
    def to_dict(self) -> dict:
        """Convert to dictionary for transformers integration"""
        result = {
            "method": self.method,
        }
        
        if self.method == "bitsandbytes":
            if self.load_in_4bit:
                result["load_in_4bit"] = True
                result["bnb_4bit_quant_type"] = self.bnb_4bit_quant_type
                result["bnb_4bit_compute_dtype"] = self.bnb_4bit_compute_dtype
                result["bnb_4bit_use_double_quant"] = self.bnb_4bit_use_double_quant
            elif self.load_in_8bit:
                result["load_in_8bit"] = True
        
        elif self.method == "gptq":
            result["bits"] = self.gptq_bits
            result["group_size"] = self.gptq_group_size
            
        elif self.method == "awq":
            result["w_bit"] = self.awq_w_bit
            result["group_size"] = self.awq_group_size
        
        return result


@dataclass
class NTFConfig:
    """
    Configuration class for Nexuss Transformer Framework decoder-only model.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model embeddings
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        d_ff: Dimension of the feed-forward network (default: 4 * d_model)
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
        tie_word_embeddings: Whether to tie input/output embeddings
        activation: Activation function ('gelu', 'relu', 'silu', 'swiglu')
        bias: Whether to use bias in linear layers
        rope_theta: Base for RoPE positional encoding
        use_rope: Whether to use Rotary Positional Embeddings
        sliding_window: Sliding window attention size (None for global attention)
        attention_dropout: Dropout for attention weights
        hidden_dropout: Dropout for hidden states
        initializer_range: Standard deviation for weight initialization
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        use_cache: Whether to use KV cache during inference
        gradient_checkpointing: Whether to use gradient checkpointing
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm
        use_swiglu: Whether to use SwiGLU activation
        quantization: Optional QuantizationConfig for model loading
    """
    
    # Core architecture
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    max_seq_len: int = 2048
    
    # FFN configuration
    d_ff: Optional[int] = None  # Defaults to 4 * d_model
    activation: str = "gelu"
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    
    # Embedding configuration
    tie_word_embeddings: bool = True
    
    # Positional encoding
    use_rope: bool = True
    rope_theta: float = 10000.0
    
    # Attention mechanism
    sliding_window: Optional[int] = None
    
    # Linear layers
    bias: bool = False
    
    # Initialization
    initializer_range: float = 0.02
    
    # Special tokens
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Runtime options
    use_cache: bool = True
    gradient_checkpointing: bool = False
    use_rmsnorm: bool = True
    
    # Quantization configuration
    quantization: Optional["QuantizationConfig"] = None
    
    def __post_init__(self):
        """Validate and set default values."""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        
        # Validate activation
        valid_activations = ["gelu", "relu", "silu", "swiglu", "gelu_new"]
        if self.activation not in valid_activations:
            raise ValueError(
                f"Activation must be one of {valid_activations}, got '{self.activation}'"
            )
        
        # Validate dimensions
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        
        # Validate dropout
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0 <= self.hidden_dropout < 1:
            raise ValueError("hidden_dropout must be in [0, 1)")
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    @classmethod
    def small(cls) -> "NTFConfig":
        """Small model configuration (~60M params)."""
        return cls(
            vocab_size=50257,
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_seq_len=1024,
            d_ff=2048,
        )
    
    @classmethod
    def medium(cls) -> "NTFConfig":
        """Medium model configuration (~350M params)."""
        return cls(
            vocab_size=50257,
            d_model=1024,
            n_heads=16,
            n_layers=24,
            max_seq_len=2048,
            d_ff=4096,
        )
    
    @classmethod
    def large(cls) -> "NTFConfig":
        """Large model configuration (~1.5B params)."""
        return cls(
            vocab_size=50257,
            d_model=2048,
            n_heads=32,
            n_layers=24,
            max_seq_len=4096,
            d_ff=8192,
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "NTFConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
