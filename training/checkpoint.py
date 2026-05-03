"""
Checkpoint management for training save/restore.

Handles saving and loading model checkpoints, optimizer states,
and training metadata with versioning support.
"""

import os
import json
import shutil
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import torch


class CheckpointManager:
    """
    Manages checkpoint saving and loading with versioning.
    
    Features:
    - Automatic checkpoint naming with timestamps
    - Limit total saved checkpoints
    - Best model tracking
    - Metadata preservation
    """
    
    def __init__(
        self,
        output_dir: str,
        save_total_limit: int = 3,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Base directory for checkpoints
            save_total_limit: Maximum number of checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.best_model_dir = self.output_dir / "best_model"
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved checkpoints
        self.saved_checkpoints = []
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        global_step: int,
        epoch: int,
        config: Any,
    ) -> str:
        """
        Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            global_step: Current training step
            epoch: Current epoch
            config: Training configuration
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint-{global_step:06d}_{timestamp}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, checkpoint_path / "model.pt")
        
        # Save optimizer and scheduler
        torch.save({
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, checkpoint_path / "training_state.pt")
        
        # Save metadata
        metadata = {
            "global_step": global_step,
            "epoch": epoch,
            "timestamp": timestamp,
            "config": config.to_dict() if hasattr(config, 'to_dict') else vars(config),
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Track checkpoint
        self.saved_checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints if limit exceeded
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def save_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        global_step: int,
        epoch: int,
        metrics: Dict[str, float],
    ) -> str:
        """
        Save best model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            global_step: Current training step
            epoch: Current epoch
            metrics: Evaluation metrics
        
        Returns:
            Path to best model directory
        """
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, self.best_model_dir / "model.pt")
        
        # Save training state
        torch.save({
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, self.best_model_dir / "training_state.pt")
        
        # Save metrics
        metadata = {
            "global_step": global_step,
            "epoch": epoch,
            "metrics": metrics,
            "saved_at": datetime.now().isoformat(),
        }
        with open(self.best_model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(self.best_model_dir)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint for resuming training.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        
        Returns:
            Dictionary with model_state, optimizer_state, scheduler_state, and metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model state
        model_state = torch.load(checkpoint_path / "model.pt", map_location="cpu")
        
        # Load training state
        training_state = torch.load(
            checkpoint_path / "training_state.pt",
            map_location="cpu",
        )
        
        # Load metadata
        with open(checkpoint_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        return {
            "model_state": model_state,
            "optimizer_state": training_state["optimizer_state"],
            "scheduler_state": training_state["scheduler_state"],
            **metadata,
        }
    
    def load_best_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load best model.
        
        Args:
            model_path: Optional path to best model (uses default if None)
        
        Returns:
            Model state dictionary
        """
        if model_path is None:
            model_path = self.best_model_dir
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Best model not found: {model_path}")
        
        model_state = torch.load(model_path / "model.pt", map_location="cpu")
        
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        return {
            "model_state": model_state,
            "metadata": metadata,
        }
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding save_total_limit."""
        if len(self.saved_checkpoints) <= self.save_total_limit:
            return
        
        # Sort by modification time
        sorted_checkpoints = sorted(
            self.saved_checkpoints,
            key=lambda p: p.stat().st_mtime,
        )
        
        # Remove oldest checkpoints
        for checkpoint in sorted_checkpoints[:-self.save_total_limit]:
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            self.saved_checkpoints.remove(checkpoint)
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return [str(p) for p in self.checkpoints_dir.glob("checkpoint-*")]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Sort by name (includes timestamp)
        return max(checkpoints)
