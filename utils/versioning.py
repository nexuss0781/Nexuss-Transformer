"""
Model Versioning and Release Management for Nexuss Transformer Framework
Semantic versioning, model registry, and release workflow
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib


class ModelStage(Enum):
    """Model development stages"""
    EXPERIMENTAL = "experimental"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Semantic version for model releases"""
    
    major: int = 0  # Breaking changes
    minor: int = 0  # New features, backward compatible
    patch: int = 0  # Bug fixes
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def increment_major(self):
        self.major += 1
        self.minor = 0
        self.patch = 0
    
    def increment_minor(self):
        self.minor += 1
        self.patch = 0
    
    def increment_patch(self):
        self.patch += 1
    
    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))


@dataclass
class ModelMetadata:
    """Metadata for model version tracking"""
    
    # Version info
    version: str
    name: str
    stage: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Model architecture
    model_type: str = "nexuss-transformer"
    num_parameters: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    
    # Training info
    training_dataset: str = ""
    training_steps: int = 0
    training_loss: float = 0.0
    validation_loss: float = 0.0
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    
    # Files and checksums
    model_path: str = ""
    config_path: str = ""
    tokenizer_path: str = ""
    model_hash: str = ""
    config_hash: str = ""
    
    # Lineage
    parent_version: Optional[str] = None
    fine_tuned_from: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Notes
    description: str = ""
    changelog: List[str] = field(default_factory=list)


class ModelRegistry:
    """Registry for tracking model versions and releases"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.registry_path / "models"
        self.metadata_dir = self.registry_path / "metadata"
        self.releases_dir = self.registry_path / "releases"
        
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.releases_dir.mkdir(exist_ok=True)
        
        self.index_file = self.registry_path / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load model index from disk"""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {"models": {}, "latest": {}}
            self._save_index()
    
    def _save_index(self):
        """Save model index to disk"""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def compute_file_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def count_parameters(self, model) -> int:
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def register_model(
        self,
        model,
        tokenizer,
        config,
        metadata: ModelMetadata,
        save_path: Optional[str] = None,
    ) -> str:
        """Register a new model version"""
        
        version = metadata.version
        model_name = metadata.name
        
        # Create model directory
        if save_path is None:
            model_dir = self.models_dir / model_name / version
        else:
            model_dir = Path(save_path)
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        config.save_pretrained(model_dir) if hasattr(config, 'save_pretrained') else None
        
        # Compute hashes
        model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
        if model_files:
            metadata.model_hash = self.compute_file_hash(str(model_files[0]))
        
        config_file = model_dir / "config.json"
        if config_file.exists():
            metadata.config_hash = self.compute_file_hash(str(config_file))
        
        # Count parameters
        metadata.num_parameters = self.count_parameters(model)
        
        # Save metadata
        metadata.model_path = str(model_dir)
        metadata.config_path = str(config_file)
        metadata.tokenizer_path = str(model_dir)
        
        metadata_file = self.metadata_dir / model_name / f"{version}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Update index
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = []
        
        self.index["models"][model_name].append(version)
        self.index["latest"][model_name] = version
        
        self._save_index()
        
        return str(model_dir)
    
    def get_model(self, name: str, version: Optional[str] = None) -> tuple:
        """Get model path and metadata by name and version"""
        
        if version is None:
            version = self.index["latest"].get(name)
            if version is None:
                raise ValueError(f"No model found with name: {name}")
        
        if name not in self.index["models"]:
            raise ValueError(f"Model '{name}' not found in registry")
        
        if version not in self.index["models"][name]:
            raise ValueError(f"Version '{version}' not found for model '{name}'")
        
        # Load metadata
        metadata_file = self.metadata_dir / name / f"{version}.json"
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        model_path = Path(metadata["model_path"])
        
        return model_path, metadata
    
    def create_release(
        self,
        name: str,
        version: str,
        release_notes: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """Create an official release package"""
        
        # Get model
        model_path, metadata = self.get_model(name, version)
        
        # Create release directory
        release_dir = self.releases_dir / name / version
        release_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        for item in model_path.iterdir():
            if item.is_file():
                shutil.copy2(item, release_dir)
        
        # Create release manifest
        manifest = {
            "name": name,
            "version": version,
            "stage": ModelStage.PRODUCTION.value,
            "release_date": datetime.now().isoformat(),
            "release_notes": release_notes,
            "tags": tags or [],
            "metadata": metadata,
        }
        
        manifest_file = release_dir / "release_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Create README for release
        readme_content = f"""# {name} v{version}

## Release Information
- **Release Date**: {manifest['release_date']}
- **Stage**: Production
- **Parameters**: {metadata['num_parameters']:,}

## Description
{metadata.get('description', 'No description provided.')}

## Changelog
"""
        for change in metadata.get('changelog', ['Initial release']):
            readme_content += f"- {change}\n"
        
        readme_content += f"\n## Release Notes\n{release_notes}\n"
        
        readme_file = release_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)
        
        return str(release_dir)
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models and their versions"""
        return self.index["models"]
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version of a model"""
        return self.index["latest"].get(name)
    
    def get_version_history(self, name: str) -> List[Dict[str, Any]]:
        """Get full version history with metadata"""
        
        if name not in self.index["models"]:
            return []
        
        history = []
        for version in self.index["models"][name]:
            metadata_file = self.metadata_dir / name / f"{version}.json"
            
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    history.append(json.load(f))
        
        return history
    
    def promote_model(
        self,
        name: str,
        version: str,
        new_stage: ModelStage,
        changelog_entry: str = "",
    ):
        """Promote model to new stage (e.g., development -> production)"""
        
        metadata_file = self.metadata_dir / name / f"{version}.json"
        
        if not metadata_file.exists():
            raise ValueError(f"Model version not found: {name} v{version}")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        old_stage = metadata["stage"]
        metadata["stage"] = new_stage.value
        
        if changelog_entry:
            metadata["changelog"].append(changelog_entry)
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Promoted {name} v{version} from {old_stage} to {new_stage.value}")
    
    def archive_model(self, name: str, version: str, reason: str = ""):
        """Archive a model version"""
        
        self.promote_model(
            name, 
            version, 
            ModelStage.ARCHIVED,
            changelog_entry=f"Archived: {reason}" if reason else "Archived"
        )


def create_model_metadata(
    name: str,
    version: str,
    stage: ModelStage = ModelStage.DEVELOPMENT,
    description: str = "",
    parent_version: Optional[str] = None,
    fine_tuned_from: Optional[str] = None,
    tags: Optional[List[str]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    evaluation_metrics: Optional[Dict[str, float]] = None,
) -> ModelMetadata:
    """Helper function to create model metadata"""
    
    return ModelMetadata(
        version=version,
        name=name,
        stage=stage.value,
        description=description,
        parent_version=parent_version,
        fine_tuned_from=fine_tuned_from,
        tags=tags or [],
        training_config=training_config or {},
        evaluation_metrics=evaluation_metrics or {},
    )
