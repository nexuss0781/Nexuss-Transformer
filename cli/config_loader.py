"""
Configuration loader for NTF CLI.

Loads and merges YAML/JSON configurations with command-line overrides.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = path.suffix.lower()
    
    if suffix in ['.yaml', '.yml']:
        return load_yaml(config_path)
    elif suffix == '.json':
        return load_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result 
            and isinstance(result[key], dict) 
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_cli_overrides(overrides: list) -> Dict[str, Any]:
    """
    Parse command-line overrides in dot notation.
    
    Example: ["model.hidden_size=1024", "training.learning_rate=1e-4"]
    
    Args:
        overrides: List of "key=value" strings
        
    Returns:
        Nested dictionary of overrides
    """
    result = {}
    
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Use 'key=value'")
        
        key, value = override.split('=', 1)
        
        # Try to parse value as JSON (handles numbers, booleans, null)
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # Keep as string if not valid JSON
            parsed_value = value
        
        # Build nested dictionary
        keys = key.split('.')
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = parsed_value
    
    return result


def apply_cli_overrides(
    config: Dict[str, Any], 
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply CLI overrides to configuration.
    
    Args:
        config: Base configuration
        overrides: CLI overrides dictionary
        
    Returns:
        Updated configuration
    """
    return merge_configs(config, overrides)


def get_default_config_path(config_name: str) -> Optional[str]:
    """
    Get default configuration path from configs directory.
    
    Args:
        config_name: Name of configuration (without extension)
        
    Returns:
        Path to configuration file or None
    """
    configs_dir = Path(__file__).parent.parent / "configs"
    
    # Try different extensions
    for ext in ['.yaml', '.yml', '.json']:
        config_path = configs_dir / f"{config_name}{ext}"
        if config_path.exists():
            return str(config_path)
    
    return None


def resolve_config_path(config_path: str) -> str:
    """
    Resolve configuration path, checking default configs directory.
    
    Args:
        config_path: Configuration name or path
        
    Returns:
        Resolved configuration path
    """
    # If it's already a full path, use it
    if Path(config_path).is_absolute():
        return config_path
    
    # Check if file exists relative to current directory
    if Path(config_path).exists():
        return config_path
    
    # Try default configs directory
    default_path = get_default_config_path(config_path)
    if default_path:
        return default_path
    
    raise FileNotFoundError(
        f"Configuration '{config_path}' not found. "
        f"Available configs: {list_available_configs()}"
    )


def list_available_configs() -> list:
    """List available configuration files in configs directory."""
    configs_dir = Path(__file__).parent.parent / "configs"
    
    if not configs_dir.exists():
        return []
    
    configs = []
    for ext in ['*.yaml', '*.yml', '*.json']:
        configs.extend([f.stem for f in configs_dir.glob(ext)])
    
    return sorted(set(configs))
