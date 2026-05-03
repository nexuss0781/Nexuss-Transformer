"""
Logging utilities for Nexuss Transformer Framework
Provides setup_logging and debug logging capabilities
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for NTF.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs to
        format_string: Custom format string (default: includes timestamp, level, module)
        
    Returns:
        Configured logger instance
    """
    # Default format with detailed info for debugging
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"
        )
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("ntf")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ntf") -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def set_log_level(level: str):
    """Set the logging level for the NTF logger."""
    logger = logging.getLogger("ntf")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


class DebugLogger:
    """
    Enhanced debug logger for training and model debugging.
    
    Provides methods for:
    - Logging tensor statistics
    - Tracking gradient norms
    - Monitoring memory usage
    - Debugging NaN/Inf values
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("ntf.debug")
        
    def log_tensor_stats(self, name: str, tensor, step: int = 0):
        """Log statistics for a tensor"""
        if tensor is None:
            return
            
        stats = {
            "shape": tuple(tensor.shape),
            "mean": tensor.mean().item() if tensor.numel() > 0 else 0,
            "std": tensor.std().item() if tensor.numel() > 1 else 0,
            "min": tensor.min().item() if tensor.numel() > 0 else 0,
            "max": tensor.max().item() if tensor.numel() > 0 else 0,
            "has_nan": bool(torch.isnan(tensor).any()) if hasattr(torch, 'isnan') else False,
            "has_inf": bool(torch.isinf(tensor).any()) if hasattr(torch, 'isinf') else False,
        }
        
        self.logger.debug(f"[Step {step}] {name}: {stats}")
        
    def log_gradient_norms(self, model, step: int = 0):
        """Log gradient norms for all parameters"""
        total_norm = 0
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                layer_norms[name] = param_norm.item()
        
        total_norm = total_norm ** 0.5
        
        self.logger.debug(f"[Step {step}] Total gradient norm: {total_norm:.6f}")
        
        # Log top 5 largest gradients
        sorted_norms = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, norm in sorted_norms:
            self.logger.debug(f"  {name}: {norm:.6f}")
    
    def check_nan_inf(self, value, name: str = "value", raise_error: bool = False) -> bool:
        """Check if a value contains NaN or Inf"""
        import torch
        
        has_issues = False
        
        if isinstance(value, torch.Tensor):
            has_nan = bool(torch.isnan(value).any())
            has_inf = bool(torch.isinf(value).any())
            
            if has_nan or has_inf:
                has_issues = True
                msg = f"{name} contains "
                if has_nan:
                    msg += "NaN"
                if has_inf:
                    msg += " and " if has_nan else ""
                if has_inf:
                    msg += "Inf"
                
                if raise_error:
                    raise ValueError(msg)
                else:
                    self.logger.warning(msg)
        
        return has_issues


# Convenience function for validating configs
def validate_config(config) -> list:
    """
    Validate a configuration object and return list of errors.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    try:
        # Try to trigger __post_init__ validation if it exists
        if hasattr(config, '__post_init__'):
            config.__post_init__()
    except Exception as e:
        errors.append(str(e))
    
    # Check for required attributes based on config type
    if hasattr(config, 'learning_rate'):
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
    
    if hasattr(config, 'per_device_train_batch_size'):
        if config.per_device_train_batch_size <= 0:
            errors.append("per_device_train_batch_size must be positive")
    
    if hasattr(config, 'max_seq_len'):
        if config.max_seq_len <= 0:
            errors.append("max_seq_len must be positive")
    
    return errors
