"""
Main training loop for LLM pre-training and fine-tuning.

Integrates with Hugging Face Accelerate for distributed training support,
mixed precision, and gradient accumulation.
"""

import os
import math
import time
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from tqdm import tqdm

from training.config import TrainingConfig, OptimizerType, SchedulerType, Precision
from training.checkpoint import CheckpointManager


class Trainer:
    """
    Main trainer class for LLM training.
    
    Features:
    - Distributed training via Accelerate
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing and resume
    - Logging and metrics tracking
    - Validation during training
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        optimizers: tuple = (None, None),
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Function to collate batches
            optimizers: Tuple of (optimizer, scheduler) or (None, None) for defaults
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        
        # Initialize accelerator
        self.accelerator = self._create_accelerator()
        
        # Setup model and optimizer
        optimizer, scheduler = optimizers
        self.optimizer = self._create_optimizer() if optimizer is None else optimizer
        self.scheduler = self._create_scheduler() if scheduler is None else scheduler
        
        # Prepare everything with accelerator
        self._prepare_model_and_optimizers()
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            save_total_limit=config.save_total_limit,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.logs_history = []
        
        # Loss tracker
        self.loss_fct = nn.CrossEntropyLoss()
    
    def _create_accelerator(self) -> Accelerator:
        """Create Accelerator instance with proper configuration."""
        # Determine mixed precision
        mixed_precision = "no"
        if self.config.mixed_precision == Precision.FP16:
            mixed_precision = "fp16"
        elif self.config.mixed_precision == Precision.BF16:
            mixed_precision = "bf16"
        
        # Use gradient_accumulation_steps directly in Accelerator
        # Note: ddp_find_unused_parameters is now configured via ddp_config in newer versions
        accelerator_kwargs = {
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "mixed_precision": mixed_precision,
            "log_with": self.config.report_to if self.config.report_to != "none" else None,
            "project_dir": self.config.output_dir,
        }
        
        # Add DDP config for newer accelerate versions (0.28.0+)
        try:
            from accelerate.utils import DistributedDataParallelKwargs
            kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=self.config.ddp_find_unused_parameters
            )
            accelerator_kwargs["kwargs_handlers"] = [kwargs]
        except (ImportError, TypeError):
            # Fallback for older versions
            pass
        
        return Accelerator(**accelerator_kwargs)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        # Get parameters with weight decay handling
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        if self.config.optimizer == OptimizerType.ADAMW:
            return AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        elif self.config.optimizer == OptimizerType.ADAM:
            return torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            return AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
            )
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler."""
        # Calculate total steps
        if self.config.max_steps > 0:
            num_training_steps = self.config.max_steps
        else:
            num_training_steps = (
                len(self.train_dataset)
                // (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)
                * self.config.num_train_epochs
            )
        
        num_warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps > 0
            else int(num_training_steps * self.config.warmup_ratio)
        )
        
        def lr_lambda(current_step: int) -> float:
            """Learning rate multiplier."""
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            
            if self.config.scheduler == SchedulerType.LINEAR:
                return max(0.0, 1.0 - progress)
            elif self.config.scheduler == SchedulerType.COSINE:
                return (1.0 + math.cos(math.pi * progress)) / 2.0
            elif self.config.scheduler == SchedulerType.CONSTANT:
                return 1.0
            elif self.config.scheduler == SchedulerType.CONSTANT_WITH_WARMUP:
                return 1.0
            else:
                return max(0.0, 1.0 - progress)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _prepare_model_and_optimizers(self):
        """Prepare model, optimizer, and dataloaders with accelerator."""
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                num_workers=self.config.dataloader_num_workers,
                prefetch_factor=self.config.dataloader_prefetch_factor if self.config.dataloader_num_workers > 0 else None,
                pin_memory=self.config.dataloader_pin_memory,
            )
        else:
            self.train_dataloader = None
        
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=self.config.dataloader_num_workers,
                prefetch_factor=self.config.dataloader_prefetch_factor if self.config.dataloader_num_workers > 0 else None,
                pin_memory=self.config.dataloader_pin_memory,
            )
        else:
            self.eval_dataloader = None
        
        # Prepare with accelerator
        if self.train_dataloader:
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Training metrics dictionary
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Calculate training steps
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
        else:
            max_steps = (
                len(self.train_dataloader)
                * self.config.num_train_epochs
                // self.config.gradient_accumulation_steps
            )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(
            range(max_steps),
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            
            for step, batch in enumerate(self.train_dataloader):
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else self._compute_loss(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = loss.item() * self.config.gradient_accumulation_steps
                        self._log_metrics(step, avg_loss, progress_bar)
                    
                    # Evaluation
                    if (
                        self.eval_dataloader
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        self._save_best_checkpoint(eval_metrics)
                    
                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()
                    
                    # Update progress bar
                    progress_bar.update(1)
                    total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Check if max steps reached
                if self.global_step >= max_steps:
                    break
            
            if self.global_step >= max_steps:
                break
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Calculate final metrics
        training_time = time.time() - start_time
        metrics = {
            "train_loss": total_loss / max_steps if max_steps > 0 else 0,
            "global_step": self.global_step,
            "epochs_trained": self.epoch + 1,
            "training_time_seconds": training_time,
            "samples_per_second": (
                len(self.train_dataset) * self.config.num_train_epochs / training_time
                if self.train_dataset else 0
            ),
        }
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on eval dataset."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else self._compute_loss(batch)
                
                # Gather losses across devices
                loss = self.accelerator.gather_for_metrics(loss)
                total_loss += loss.sum().item()
                total_samples += loss.shape[0]
        
        self.model.train()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_samples": total_samples,
        }
        
        return metrics
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss manually if model doesn't return it."""
        input_ids = batch.get("input_ids")
        labels = batch.get("labels", input_ids)
        
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        return loss
    
    def _log_metrics(
        self,
        step: int,
        loss: float,
        progress_bar: tqdm,
    ):
        """Log training metrics."""
        lr = self.scheduler.get_last_lr()[0]
        
        metrics = {
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/global_step": self.global_step,
            "train/epoch": self.epoch,
        }
        
        self.logs_history.append(metrics)
        
        # Update progress bar description
        progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "lr": f"{lr:.2e}",
        })
        
        # Log to accelerator
        self.accelerator.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.accelerator.unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.epoch,
            config=self.config,
        )
        
        if self.accelerator.is_main_process:
            print(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_best_checkpoint(self, metrics: Dict[str, float]):
        """Save best checkpoint based on evaluation metric."""
        current_metric = metrics.get("eval_loss", float('inf'))
        
        if self.best_metric is None or current_metric < self.best_metric:
            self.best_metric = current_metric
            best_path = self.checkpoint_manager.save_best_checkpoint(
                model=self.accelerator.unwrap_model(self.model),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                global_step=self.global_step,
                epoch=self.epoch,
                metrics=metrics,
            )
            
            if self.accelerator.is_main_process:
                print(f"New best model saved to {best_path} (eval_loss: {current_metric:.4f})")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training."""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        
        print(f"Resumed from checkpoint: {checkpoint_path} (step {self.global_step})")
