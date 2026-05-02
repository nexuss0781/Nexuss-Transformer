#!/usr/bin/env python3
"""
Nexuss Transformer Framework (NTF) - Professional LLM Training System

A complete, production-ready framework for training, fine-tuning, and aligning
transformer language models with Ethiopian language support.

Framework Components (22+):
- Model Architecture: NexussTransformer with RoPE, SwiGLU, RMSNorm
- Training: Full distributed training with Accelerate
- Fine-tuning: LoRA, full fine-tuning, layer freezing
- RLHF: DPO and PPO alignment
- Continual Learning: EWC, replay buffers, GEM
- Evaluation: Perplexity, accuracy, BLEU, ROUGE, throughput
- Versioning: Semantic versioning, model registry, releases

Usage:
    # CLI
    ntf train --config pretrain_small
    ntf finetune --config finetune_lora.yaml --override lora.r=32
    ntf align --config dpo_alignment.yaml --method dpo
    ntf evaluate --model-path ./outputs/final --metrics perplexity
    
    # Python API
    from framework import QuickPipeline
    pipeline = QuickPipeline('small')
    pipeline.train(data='corpus.jsonl').finetune_lora(data='instructions.jsonl')
"""

from framework.components import (
    NTF_COMPONENT_REGISTRY,
    ComponentCategory,
    ComponentMetadata,
    get_component,
    list_components,
    get_components_by_category,
    generate_cli_help,
)

__version__ = "1.0.0"
__author__ = "Nexuss AI Research"
__all__ = [
    "__version__",
    "NTF_COMPONENT_REGISTRY",
    "ComponentCategory", 
    "ComponentMetadata",
    "get_component",
    "list_components",
    "get_components_by_category",
    "generate_cli_help",
]

# Print framework info on import (optional)
def print_framework_info():
    """Print summary of available components."""
    print(generate_cli_help())
