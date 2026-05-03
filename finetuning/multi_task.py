"""
Multi-Task Learning Implementation for NTF
Supports task-specific heads for different fine-tuning objectives
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    """Supported task types for multi-task learning."""
    CLASSIFICATION = "classification"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    GENERATION = "generation"


@dataclass
class TaskHeadConfig:
    """Configuration for a task-specific head."""
    
    task_name: str
    head_type: TaskType
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.head_type, str):
            self.head_type = TaskType(self.head_type)


class ClassificationHead(nn.Module):
    """Classification head for sequence classification tasks."""
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Use pooled output (last hidden state of [CLS] or mean pooling)
        if attention_mask is not None:
            # Mean pooling with mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states[:, -1, :]  # Last token
        
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class SequenceToSequenceHead(nn.Module):
    """Sequence-to-sequence head for generation tasks."""
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__()
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output_projection(hidden_states)


class TokenClassificationHead(nn.Module):
    """Token-level classification head (NER, POS tagging, etc.)."""
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        return self.classifier(hidden_states)


class QuestionAnsweringHead(nn.Module):
    """Head for extractive question answering."""
    
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start and end logits
        
    def forward(self, hidden_states: torch.Tensor) -> tuple:
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


class TaskHead(nn.Module):
    """Wrapper for task-specific heads."""
    
    HEAD_CLASSES = {
        TaskType.CLASSIFICATION: ClassificationHead,
        TaskType.SEQUENCE_TO_SEQUENCE: SequenceToSequenceHead,
        TaskType.TOKEN_CLASSIFICATION: TokenClassificationHead,
        TaskType.QUESTION_ANSWERING: QuestionAnsweringHead,
    }
    
    def __init__(self, config: TaskHeadConfig, hidden_size: int, vocab_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.task_name = config.task_name
        self.head_type = config.head_type
        
        head_config = dict(config.config)
        head_config["hidden_size"] = hidden_size
        
        if vocab_size is not None:
            head_config["vocab_size"] = vocab_size
        
        head_class = self.HEAD_CLASSES.get(head_type)
        if head_class is None:
            raise ValueError(f"Unsupported task type: {head_type}")
        
        self.head = head_class(**head_config)
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.head(hidden_states, **kwargs)


class MultiTaskModel(nn.Module):
    """
    Multi-task model with task-specific heads sharing a common base.
    
    Args:
        base_model: Base transformer model
        base_model_name: Name or path of base model
    """
    
    def __init__(self, base_model=None, base_model_name: Optional[str] = None):
        super().__init__()
        
        if base_model is None and base_model_name is None:
            raise ValueError("Must provide either base_model or base_model_name")
        
        if base_model is None:
            from transformers import AutoModelForCausalLM
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        else:
            self.base_model = base_model
        
        # Get hidden size from base model
        self.hidden_size = getattr(self.base_model.config, 'hidden_size', 768)
        self.vocab_size = getattr(self.base_model.config, 'vocab_size', None)
        
        # Task heads registry
        self.task_heads: Dict[str, TaskHead] = nn.ModuleDict()
        self.active_task: Optional[str] = None
        
        # Task weights for balanced training
        self.task_weights: Dict[str, float] = {}
    
    def add_task_head(
        self,
        task_name: str,
        head_type: Union[str, TaskType],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Add a task-specific head to the model.
        
        Args:
            task_name: Unique name for this task
            head_type: Type of task (classification, seq2seq, etc.)
            config: Task-specific configuration
        """
        if config is None:
            config = {}
        
        task_config = TaskHeadConfig(
            task_name=task_name,
            head_type=head_type,
            config=config
        )
        
        task_head = TaskHead(task_config, self.hidden_size, self.vocab_size)
        self.task_heads[task_name] = task_head
        self.task_weights[task_name] = 1.0  # Default equal weight
    
    def set_task_weights(self, weights: Dict[str, float]):
        """Set weights for each task in multi-task training."""
        for task_name, weight in weights.items():
            if task_name in self.task_heads:
                self.task_weights[task_name] = weight
    
    def set_active_task(self, task_name: str):
        """Set the currently active task for single-task inference."""
        if task_name not in self.task_heads:
            raise ValueError(f"Task '{task_name}' not found. Available: {list(self.task_heads.keys())}")
        self.active_task = task_name
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through base model and task head.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for loss computation
            task_name: Task to use (overrides active_task)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        # Determine which task to use
        task = task_name or self.active_task
        
        if task is None and len(self.task_heads) == 1:
            task = list(self.task_heads.keys())[0]
        elif task is None:
            raise ValueError("No task specified and multiple heads available")
        
        if task not in self.task_heads:
            raise ValueError(f"Task '{task}' not found")
        
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last hidden state
        hidden_states = base_outputs.hidden_states[-1]
        
        # Apply task head
        head = self.task_heads[task]
        head_output = head(hidden_states, attention_mask=attention_mask)
        
        result = {"logits": head_output}
        
        # Compute loss if labels provided
        if labels is not None:
            if head.head_type == TaskType.CLASSIFICATION:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(head_output.view(-1, head.num_labels), labels.view(-1))
            elif head.head_type == TaskType.SEQUENCE_TO_SEQUENCE:
                shift_logits = head_output[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(head_output.view(-1, head_output.size(-1)), labels.view(-1))
            
            result["loss"] = loss
        
        return result
    
    def get_num_tasks(self) -> int:
        """Return number of task heads."""
        return len(self.task_heads)
    
    def list_tasks(self) -> List[str]:
        """Return list of task names."""
        return list(self.task_heads.keys())


class MultiTaskTrainer:
    """
    Trainer for multi-task learning with task-balanced loss.
    
    Args:
        model: MultiTaskModel instance
        task_datasets: Dictionary mapping task names to datasets
        task_weights: Optional dictionary of task weights
    """
    
    def __init__(
        self,
        model: MultiTaskModel,
        task_datasets: Dict[str, Any],
        task_weights: Optional[Dict[str, float]] = None,
        tokenizer=None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.task_datasets = task_datasets
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set task weights
        if task_weights:
            self.model.set_task_weights(task_weights)
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        batch_sizes: Dict[str, int] = None,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Train one epoch across all tasks.
        
        Args:
            optimizer: Optimizer for training
            batch_sizes: Batch size per task
            gradient_accumulation_steps: Steps before optimizer update
            
        Returns:
            Dictionary of losses per task
        """
        self.model.train()
        task_losses = {task: 0.0 for task in self.task_datasets.keys()}
        task_counts = {task: 0 for task in self.task_datasets.keys()}
        
        # Simple round-robin training across tasks
        for task_name, dataset in self.task_datasets.items():
            weight = self.model.task_weights.get(task_name, 1.0)
            
            for batch in dataset:
                # Move batch to device
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                         for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                    task_name=task_name
                )
                
                loss = outputs["loss"] * weight
                loss.backward()
                
                optimizer.step()
                
                task_losses[task_name] += loss.item() / weight
                task_counts[task_name] += 1
        
        # Average losses
        avg_losses = {
            task: task_losses[task] / max(task_counts[task], 1)
            for task in task_losses
        }
        
        return avg_losses
    
    def evaluate(
        self,
        eval_datasets: Dict[str, Any],
        metrics_fn: Optional[Dict[str, callable]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all tasks.
        
        Args:
            eval_datasets: Evaluation datasets per task
            metrics_fn: Optional metric functions per task
            
        Returns:
            Dictionary of metrics per task
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for task_name, dataset in eval_datasets.items():
                task_results = {"loss": 0.0, "count": 0}
                
                for batch in dataset:
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items()}
                    
                    outputs = self.model(
                        input_ids=inputs.get("input_ids"),
                        attention_mask=inputs.get("attention_mask"),
                        labels=inputs.get("labels"),
                        task_name=task_name
                    )
                    
                    task_results["loss"] += outputs["loss"].item()
                    task_results["count"] += 1
                
                if task_results["count"] > 0:
                    task_results["loss"] /= task_results["count"]
                
                results[task_name] = task_results
        
        return results
