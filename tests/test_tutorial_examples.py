"""
Test Suite for Tutorial Code Examples
Ensures all code examples in tutorials remain functional
"""

import pytest
import os
import sys
import torch
from datasets import Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTutorial03:
    """Test Tutorial 03: Full Fine-Tuning examples"""
    
    def test_full_finetuning_basic(self):
        """Test basic full fine-tuning workflow"""
        from ntf.config import NTFConfig, ModelConfig, TrainingConfig
        from ntf.models import ModelRegistry
        from ntf.finetuning import FullFinetuneTrainer
        
        config = NTFConfig(
            model=ModelConfig(name="facebook/opt-125m"),
            training=TrainingConfig(
                output_dir="./test_output",
                num_train_epochs=1,
                per_device_train_batch_size=2,
            )
        )
        
        registry = ModelRegistry(config.model)
        model, tokenizer = registry.load_model_and_tokenizer()
        
        train_data = Dataset.from_dict({
            "text": ["Hello world", "Test sentence"] * 10
        })
        
        trainer = FullFinetuneTrainer(
            model=model,
            config=config.training,
            train_dataset=train_data,
            tokenizer=tokenizer
        )
        
        trainer.train()
        
        assert os.path.exists("./test_output")


class TestTutorial05:
    """Test Tutorial 05: PEFT/LoRA examples"""
    
    def test_lora_setup(self):
        """Test LoRA adapter setup"""
        from ntf.finetuning import LoRAConfig, PEFTTrainer
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        model, tokenizer = registry.load_model_and_tokenizer()
        
        lora_config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        
        trainer = PEFTTrainer(model, lora_config, tokenizer)
        
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in trainer.model.parameters())
        
        assert trainable_params < all_params
        assert trainable_params > 0
    
    def test_p_tuning_setup(self):
        """Test P-Tuning setup"""
        from ntf.finetuning import PTuningConfig, PTuningMethod, setup_p_tuning
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        model, tokenizer = registry.load_model_and_tokenizer()
        
        config = PTuningConfig(
            method=PTuningMethod.P_TUNING_V2,
            num_virtual_tokens=20,
        )
        
        peft_model = setup_p_tuning(model, method="p_tuning_v2", num_virtual_tokens=20)
        
        assert peft_model is not None


class TestTutorial04:
    """Test Tutorial 04: Continual Learning examples"""
    
    def test_ewc_regularization(self):
        """Test EWC regularization setup"""
        from ntf.utils import EWCConfig, EWCRegularizer, ContinualLearningWrapper
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        model, tokenizer = registry.load_model_and_tokenizer()
        
        ewc_config = EWCConfig(ewc_lambda=1000.0)
        ewc = EWCRegularizer(model, ewc_config)
        
        assert ewc is not None
        assert ewc.config.ewc_lambda == 1000.0
    
    def test_si_regularization(self):
        """Test Synaptic Intelligence regularization"""
        from ntf.utils import SIRegularizer, ContinualLearningWrapper
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        model, _ = registry.load_model_and_tokenizer()
        
        wrapper = ContinualLearningWrapper(model, method="si")
        wrapper.apply_si_regularization(c=0.1)
        
        assert wrapper.si is not None
        assert wrapper.si.c == 0.1
    
    def test_lwf_regularization(self):
        """Test Learning without Forgetting"""
        from ntf.utils import LwFRegularizer, ContinualLearningWrapper
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        model, _ = registry.load_model_and_tokenizer()
        
        wrapper = ContinualLearningWrapper(model, method="lwf")
        wrapper.apply_lwf_regularization(alpha=0.5)
        
        assert wrapper.lwf is not None
        assert wrapper.lwf.alpha == 0.1


class TestMultiTask:
    """Test Multi-Task Learning (Spec 4.1.1)"""
    
    def test_multi_task_model_creation(self):
        """Test creating multi-task model with multiple heads"""
        from ntf.finetuning import MultiTaskModel, TaskType, MultiTaskTrainer
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        base_model, tokenizer = registry.load_model_and_tokenizer()
        
        model = MultiTaskModel(base_model=base_model)
        
        model.add_task_head(
            task_name="classification",
            head_type=TaskType.CLASSIFICATION,
            config={"num_labels": 5}
        )
        
        model.add_task_head(
            task_name="summarization",
            head_type=TaskType.SEQUENCE_TO_SEQUENCE,
            config={"max_length": 512}
        )
        
        assert model.get_num_tasks() == 2
        assert "classification" in model.list_tasks()
        assert "summarization" in model.list_tasks()
    
    def test_multi_task_forward(self):
        """Test forward pass through multi-task model"""
        from ntf.finetuning import MultiTaskModel, TaskType
        from ntf.models import ModelRegistry
        import torch
        
        registry = ModelRegistry("facebook/opt-125m")
        base_model, tokenizer = registry.load_model_and_tokenizer()
        
        model = MultiTaskModel(base_model=base_model)
        model.add_task_head(
            task_name="classification",
            head_type=TaskType.CLASSIFICATION,
            config={"num_labels": 3}
        )
        
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones((2, 10))
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_name="classification"
        )
        
        assert "logits" in output
        assert output["logits"].shape[0] == 2
        assert output["logits"].shape[1] == 3


class TestContinualLearningWrapper:
    """Test ContinualLearningWrapper API (Spec 4.1.2)"""
    
    def test_wrapper_api(self):
        """Test the unified ContinualLearningWrapper API"""
        from ntf.utils import ContinualLearningWrapper
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        model, _ = registry.load_model_and_tokenizer()
        
        wrapper = ContinualLearningWrapper(model, method="ewc")
        
        # Test EWC
        wrapper.apply_ewc_regularization(lambda_ewc=0.5)
        assert wrapper.ewc is not None
        
        # Test SI
        wrapper2 = ContinualLearningWrapper(model, method="si")
        wrapper2.apply_si_regularization(c=0.1)
        assert wrapper2.si is not None
        
        # Test LwF
        wrapper3 = ContinualLearningWrapper(model, method="lwf")
        wrapper3.apply_lwf_regularization(alpha=0.5)
        assert wrapper3.lwf is not None
    
    def test_progressive_unfreeze(self):
        """Test progressive unfreezing strategy"""
        from ntf.utils import ContinualLearningWrapper
        from ntf.models import ModelRegistry
        
        registry = ModelRegistry("facebook/opt-125m")
        model, _ = registry.load_model_and_tokenizer()
        
        wrapper = ContinualLearningWrapper(model)
        wrapper.progressive_unfreeze(
            start_layers=4,
            unfreeze_every_n_epochs=2,
            max_layers=12
        )
        
        assert hasattr(wrapper, 'start_layers')
        assert wrapper.start_layers == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
