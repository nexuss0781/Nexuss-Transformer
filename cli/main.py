#!/usr/bin/env python3
"""
Nexuss Transformer Framework - Command Line Interface

Professional CLI for training, fine-tuning, and aligning transformer models.
Supports YAML/JSON configurations with command-line overrides.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from cli.config_loader import (
    load_config,
    resolve_config_path,
    parse_cli_overrides,
    apply_cli_overrides,
    list_available_configs,
)
from cli.commands import (
    train_command,
    finetune_command,
    align_command,
    evaluate_command,
    convert_command,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    
    parser = argparse.ArgumentParser(
        prog="ntf",
        description="Nexuss Transformer Framework - Professional LLM Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  ntf train --config pretrain_small
  
  # Train with custom config and overrides
  ntf train --config my_config.yaml --override model.hidden_size=1024 training.learning_rate=1e-4
  
  # Fine-tune with LoRA
  ntf finetune --config finetune_lora.yaml
  
  # Run DPO alignment
  ntf align --config dpo_alignment.yaml --method dpo
  
  # Evaluate a model
  ntf evaluate --model ./outputs/checkpoint-5000 --eval-data test.jsonl
  
  # Convert model to ONNX
  ntf convert --model ./outputs/final --format onnx

Available configs: {configs}
        """.strip(),
    )
    
    # Add available configs to epilog
    available = ", ".join(list_available_configs()) or "none"
    parser.epilog = parser.epilog.format(configs=available)
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ========== TRAIN COMMAND ==========
    train_parser = subparsers.add_parser(
        "train",
        help="Pre-train or continue training a model",
        description="Execute pre-training or continued training with configurable architecture."
    )
    train_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML/JSON) or config name"
    )
    train_parser.add_argument(
        "-o", "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values (dot notation: key=value)"
    )
    train_parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    train_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # ========== FINETUNE COMMAND ==========
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune a model (full or LoRA)",
        description="Fine-tune a pretrained model using full fine-tuning or LoRA."
    )
    finetune_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    finetune_parser.add_argument(
        "-o", "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values"
    )
    finetune_parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and use full fine-tuning"
    )
    finetune_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # ========== ALIGN COMMAND ==========
    align_parser = subparsers.add_parser(
        "align",
        help="Align model using RLHF (DPO/PPO)",
        description="Perform RLHF alignment using DPO or PPO."
    )
    align_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    align_parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["dpo", "ppo"],
        default="dpo",
        help="Alignment method (default: dpo)"
    )
    align_parser.add_argument(
        "-o", "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values"
    )
    align_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # ========== EVALUATE COMMAND ==========
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model",
        description="Evaluate model performance on various metrics."
    )
    eval_parser.add_argument(
        "-m", "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "-d", "--eval-data",
        type=str,
        help="Path to evaluation dataset"
    )
    eval_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Optional configuration file"
    )
    eval_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["perplexity"],
        help="Metrics to compute (default: perplexity)"
    )
    eval_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # ========== CONVERT COMMAND ==========
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert model to different formats",
        description="Convert model to ONNX, GGUF, SafeTensors, etc."
    )
    convert_parser.add_argument(
        "-m", "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    convert_parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["onnx", "gguf", "safetensors"],
        default="onnx",
        help="Output format (default: onnx)"
    )
    convert_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./outputs/converted",
        help="Output directory"
    )
    convert_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Optional configuration file"
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for NTF CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        # Load configuration
        if hasattr(args, 'config') and args.config:
            config_path = resolve_config_path(args.config)
            config = load_config(config_path)
            
            # Apply CLI overrides
            if hasattr(args, 'override') and args.override:
                overrides = parse_cli_overrides(args.override)
                config = apply_cli_overrides(config, overrides)
        else:
            config = {}
        
        # Execute command
        if args.command == "train":
            return train_command(config, args)
        elif args.command == "finetune":
            # Handle --no-lora flag
            if hasattr(args, 'no_lora') and args.no_lora:
                if 'lora' not in config:
                    config['lora'] = {}
                config['lora']['enable'] = False
            return finetune_command(config, args)
        elif args.command == "align":
            return align_command(config, args)
        elif args.command == "evaluate":
            return evaluate_command(config, args)
        elif args.command == "convert":
            return convert_command(config, args)
        else:
            parser.print_help()
            return 1
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
