"""
RLHF Pipeline for Nexuss Transformer Framework
Integrates reward model with PPO training for reinforcement learning from human feedback
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from reward.ppo_trainer import PPOTrainerConfig, create_ppo_trainer, train_ppo


class RLHFPipeline:
    """
    End-to-end RLHF pipeline integrating reward model and PPO training.
    
    This class provides a unified interface for running RLHF with:
    - Policy model (the model being optimized)
    - Reward model (learned from human preferences)
    - Reference model (frozen SFT model for KL penalty)
    """
    
    def __init__(
        self,
        policy_model: PreTrainedModel,
        reward_model: PreTrainedModel,
        reference_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[PPOTrainerConfig] = None,
    ):
        """
        Initialize RLHF Pipeline.
        
        Args:
            policy_model: The policy model to optimize (can be same as reference initially)
            reward_model: Trained reward model for scoring responses
            reference_model: Frozen reference model for KL divergence penalty
            tokenizer: Tokenizer for text processing
            config: PPO configuration options
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config or PPOTrainerConfig()
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Freeze reward model  
        for param in self.reward_model.parameters():
            param.requires_grad = False
            
        self.ppo_trainer = None
    
    def _compute_reward(self, prompt: str, response: str) -> float:
        """Compute reward for a prompt-response pair using the reward model"""
        # Tokenize input
        text = prompt + response
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True
        ).to(self.reward_model.device if hasattr(self.reward_model, 'device') else 'cuda')
        
        # Get reward score
        with torch.no_grad():
            if hasattr(self.reward_model, 'compute_reward'):
                reward = self.reward_model.compute_reward(
                    inputs['input_ids'],
                    inputs.get('attention_mask', None)
                )
            else:
                outputs = self.reward_model(**inputs)
                reward = outputs.logits.squeeze(-1)
            
            return reward.item() if isinstance(reward, torch.Tensor) else reward
    
    def run_ppo(
        self,
        prompts: List[str],
        num_iterations: int = 100,
        kl_coeff: float = 0.2,
        batch_size: Optional[int] = None,
    ):
        """
        Run PPO optimization loop.
        
        Args:
            prompts: List of prompt strings for generation
            num_iterations: Number of PPO iterations
            kl_coeff: KL penalty coefficient
            batch_size: Batch size for PPO (overrides config if provided)
            
        Returns:
            Trained policy model
        """
        if batch_size:
            self.config.batch_size = batch_size
        
        # Update KL coefficient
        self.config.kl_coeff = kl_coeff
        
        # Create dataset from prompts
        dataset = [{"prompt": p} for p in prompts]
        
        # Create PPO trainer
        self.ppo_trainer = create_ppo_trainer(
            policy_model=self.policy_model,
            ref_model=self.reference_model,
            tokenizer=self.tokenizer,
            config=self.config,
        )
        
        # Override reward computation to use our reward model
        original_compute_reward = getattr(self, '_compute_reward', None)
        
        # Training loop
        print(f"Starting PPO training for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Sample batch of prompts
            batch_prompts = prompts[iteration % len(prompts):(iteration % len(prompts)) + self.config.batch_size]
            if len(batch_prompts) < self.config.batch_size:
                batch_prompts = prompts[:self.config.batch_size]
            
            # Generate responses and compute rewards
            query_tensors = []
            response_tensors = []
            rewards = []
            
            for prompt in batch_prompts:
                # Tokenize query
                query_tensor = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
                query_tensors.append(query_tensor)
                
                # Generate response
                with torch.no_grad():
                    output = self.policy_model.generate(
                        query_tensor.unsqueeze(0).to(self.policy_model.device),
                        max_new_tokens=self.config.max_length // 2,
                        do_sample=True,
                        top_p=self.config.top_p,
                        temperature=self.config.temperature,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response_tensor = output[0, len(query_tensor):]
                response_tensors.append(response_tensor)
                
                # Decode and compute reward
                response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                reward = self._compute_reward(prompt, response_text)
                rewards.append(torch.tensor(reward))
            
            # Run PPO step
            if self.ppo_trainer:
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # Log progress
                if iteration % max(1, num_iterations // 10) == 0:
                    avg_reward = sum(r.item() if isinstance(r, torch.Tensor) else r for r in rewards) / len(rewards)
                    print(f"Iteration {iteration}: Avg Reward = {avg_reward:.4f}")
        
        print("PPO training complete!")
        return self.policy_model
    
    def save_policy(self, output_path: str):
        """Save the trained policy model"""
        self.policy_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Policy model saved to {output_path}")
    
    def evaluate_policy(self, test_prompts: List[str]) -> List[Dict[str, Any]]:
        """Evaluate the policy on test prompts"""
        results = []
        
        for prompt in test_prompts:
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.policy_model.device)
            
            with torch.no_grad():
                output = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            reward = self._compute_reward(prompt, response)
            
            results.append({
                "prompt": prompt,
                "response": response,
                "reward": reward
            })
        
        return results
