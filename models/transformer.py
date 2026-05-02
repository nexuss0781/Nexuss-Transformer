"""
Decoder-only Transformer Model

Implements a blank-slate decoder-only transformer architecture optimized for
autoregressive language modeling. Built from scratch with PyTorch, integrating
seamlessly with Hugging Face's ecosystem.
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import NTFConfig


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    
    Implements rotary embeddings as described in "RoFormer: Enhanced Transformer
    with Rotary Position Embedding" (Su et al., 2021).
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build sin/cos cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cached sin/cos values for efficiency."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for the given sequence length.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, heads, seq_len, dim)
            seq_len: Sequence length (uses cached if None)
        
        Returns:
            cos, sin tensors for rotation
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine values
        sin: Sine values
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
    
    Returns:
        Rotated query and key tensors
    """
    # cos/sin are (seq_len, head_dim), need to broadcast to (batch, heads, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than standard LayerNorm, used in many modern architectures
    like LLaMA and PaLM.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(torch.float32)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation.
    
    Combines SwiSH activation with gating mechanism for improved performance.
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FeedForward(nn.Module):
    """
    Feed-forward network with configurable activation.
    """
    
    def __init__(self, config: NTFConfig):
        super().__init__()
        self.config = config
        
        if config.activation == "swiglu":
            self.ffn = SwiGLU(config.d_model, config.d_ff)
        else:
            # Standard FFN with specified activation
            activations = {
                "gelu": nn.GELU(),
                "gelu_new": nn.GELU(approximate="tanh"),
                "relu": nn.ReLU(),
                "silu": nn.SiLU(),
            }
            activation_fn = activations.get(config.activation, nn.GELU())
            
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff, bias=config.bias),
                activation_fn,
                nn.Dropout(config.hidden_dropout),
                nn.Linear(config.d_ff, config.d_model, bias=config.bias),
                nn.Dropout(config.hidden_dropout),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network."""
        return self.ffn(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional sliding window.
    """
    
    def __init__(self, config: NTFConfig, is_causal: bool = True):
        super().__init__()
        self.config = config
        self.is_causal = is_causal
        self.num_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        
        # QKV projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Output projection
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        ) if config.use_rope else None
        
        # Sliding window attention
        self.sliding_window = config.sliding_window
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional attention mask
            past_key_value: Cached key/value for incremental decoding
            use_cache: Whether to return key/value cache
        
        Returns:
            Output tensor and optional key/value cache
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Concatenate with past key/value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # Cache current key/value
        present_key_value = None
        if use_cache:
            present_key_value = (k, v)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1,
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))
        
        # Apply sliding window if configured
        if self.sliding_window is not None:
            window_mask = torch.ones_like(attn_weights, dtype=torch.bool)
            window_mask = torch.triu(window_mask, diagonal=self.sliding_window)
            attn_weights.masked_fill_(window_mask, float("-inf"))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.config.d_model)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, present_key_value


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block with pre-normalization.
    """
    
    def __init__(self, config: NTFConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-normalization
        self.input_layernorm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Attention and FFN
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block.
        
        Uses pre-normalization architecture for better training stability.
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # FFN with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class NexussTransformer(nn.Module):
    """
    Complete decoder-only transformer model.
    
    A blank-slate transformer designed for autoregressive language modeling,
    featuring:
    - Rotary positional embeddings (RoPE)
    - RMS normalization
    - SwiGLU activation option
    - Sliding window attention
    - Gradient checkpointing support
    - KV caching for efficient inference
    
    Args:
        config: NTFConfig instance with model hyperparameters
    """
    
    def __init__(self, config: NTFConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Output projection (tied with embeddings if configured)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get token embedding layer."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set token embedding layer."""
        self.embed_tokens = value
    
    def get_output_embeddings(self) -> nn.Linear:
        """Get output projection layer."""
        if self.lm_head is None:
            return self.embed_tokens
        return self.lm_head
    
    def tie_weights(self):
        """Tie input and output embeddings."""
        self.config.tie_word_embeddings = True
        self.lm_head = None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            position_ids: Position IDs (not used with RoPE)
            past_key_values: Cached key/values for each layer
            inputs_embeds: Alternative to input_ids
            labels: Labels for computing loss
            use_cache: Return KV cache
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            return_dict: Return dict vs tuple
        
        Returns:
            Model output with logits, loss (if labels provided), and optional caches/states
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        output_attentions = output_attentions or False
        output_hidden_states = output_hidden_states or False
        
        # Get embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
        else:
            raise ValueError("Must provide either input_ids or inputs_embeds")
        
        hidden_states = self.embed_dropout(inputs_embeds)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to additive mask
            attention_mask = attention_mask.to(hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Process through layers
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                hidden_states, present = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_value,
                    use_cache,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            
            if use_cache:
                presents += (present,)
            
            if output_attentions:
                # Would need to modify attention layer to return weights
                pass
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Compute logits
        if self.lm_head is None:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        
        # Return results
        if not return_dict:
            result = (logits,) + (presents, all_hidden_states, all_attentions)
            return (loss,) + result if loss is not None else result
        
        from types import SimpleNamespace
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling (top-p)
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
        
        Returns:
            Generated token IDs
        """
        self.eval()
        
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        past_key_values = None
        
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id
        
        for _ in range(max_length):
            # Forward pass
            outputs = self(
                input_ids=generated,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits /= temperature
            
            # Apply top-k
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update cache
            past_key_values = outputs.past_key_values
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def count_parameters(self) -> dict:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "total_millions": f"{total / 1e6:.2f}M",
        }
