"""Finetuning package - PEFT, LoRA, and layer freezing utilities."""

from finetuning.peft_finetune import PEFTTrainer, LoRAConfig, setup_lora
from finetuning.freeze import LayerFreezer, freeze_layers
from finetuning.full_finetune import FullFinetuneTrainer, full_finetune
from finetuning.multi_task import (
    MultiTaskModel,
    MultiTaskTrainer,
    TaskHead,
    TaskType,
    TaskHeadConfig,
    ClassificationHead,
    SequenceToSequenceHead,
    TokenClassificationHead,
    QuestionAnsweringHead,
)
from finetuning.p_tuning import (
    PTuningModel,
    PTuningConfig,
    PTuningMethod,
    setup_p_tuning,
)

__all__ = [
    "PEFTTrainer",
    "LoRAConfig",
    "setup_lora",
    "LayerFreezer",
    "freeze_layers",
    "FullFinetuneTrainer",
    "full_finetune",
    # Multi-task learning
    "MultiTaskModel",
    "MultiTaskTrainer",
    "TaskHead",
    "TaskType",
    "TaskHeadConfig",
    "ClassificationHead",
    "SequenceToSequenceHead",
    "TokenClassificationHead",
    "QuestionAnsweringHead",
    # P-Tuning / Prefix Tuning
    "PTuningModel",
    "PTuningConfig",
    "PTuningMethod",
    "setup_p_tuning",
]
